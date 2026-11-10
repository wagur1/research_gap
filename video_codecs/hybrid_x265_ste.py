import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_dct_matrix(n: int = 8, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Orthonormal 1D DCT-II matrix of size NxN for block transform simulation."""
    mat = torch.zeros((n, n), device=device, dtype=dtype)
    for k in range(n):
        for i in range(n):
            alpha = math.sqrt(1.0 / n) if k == 0 else math.sqrt(2.0 / n)
            mat[k, i] = alpha * math.cos(math.pi * (2 * i + 1) * k / (2.0 * n))
    return mat


class DifferentiableVideoProxy(nn.Module):
    """
    Differentiable QP- and CTU-Delta-QP-Aware Video Codec Proxy for x265 / VVenC.
    Fixes the naive Virtual Codec in Zhao et al. (arXiv:2512.15331) by:
      1. Simulating 8x8 block DCT quantization modulated by both base QP severity `s`
         AND spatial CTU Delta-QP offsets `ctu_qp_offsets`.
      2. Explicitly modeling Intra (t=0) vs. Inter-frame residual coding (t>0) so
         temporal background flickering is penalized in the rate & distortion gradients.
    """

    def __init__(self, block_size: int = 8) -> None:
        super().__init__()
        self.block_size = block_size
        dct_1d = create_dct_matrix(block_size)
        self.register_buffer("dct_mat", dct_1d)

        # Frequency sensitivity weights (higher frequencies quantized more aggressively like HEVC scaling lists)
        u = torch.arange(block_size, dtype=torch.float32).unsqueeze(1)
        v = torch.arange(block_size, dtype=torch.float32).unsqueeze(0)
        freq_weight = 1.0 + 0.25 * (u + v)
        self.register_buffer("freq_weight", freq_weight.view(1, 1, 1, block_size, block_size))

    def block_dct2d(self, x: torch.Tensor) -> torch.Tensor:
        """Applies 2D DCT to non-overlapping 8x8 blocks of x [B, C, H, W]."""
        b, c, h, w = x.shape
        bs = self.block_size
        x_blocks = x.view(b, c, h // bs, bs, w // bs, bs).permute(0, 1, 2, 4, 3, 5)
        x_blocks = x_blocks.reshape(b, c, -1, bs, bs)
        dct_m = self.dct_mat.to(dtype=x.dtype)
        coeff = torch.matmul(dct_m, torch.matmul(x_blocks, dct_m.t()))
        return coeff

    def block_idct2d(self, coeff: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Applies 2D IDCT to 8x8 block coefficients [B, C, N_blocks, 8, 8]."""
        b, c, _, bs, _ = coeff.shape
        dct_m = self.dct_mat.to(dtype=coeff.dtype)
        blocks = torch.matmul(dct_m.t(), torch.matmul(coeff, dct_m))
        blocks = blocks.view(b, c, h // bs, w // bs, bs, bs).permute(0, 1, 2, 4, 3, 5)
        return blocks.reshape(b, c, h, w)

    def quantize_blocks(
        self, x: torch.Tensor, q_step_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable DCT quantization + proxy entropy rate estimation.
        Args:
            x: Signal (Intra frame or Inter residual) [B, C, H, W].
            q_step_map: Spatial quantization step per 8x8 block [B, 1, N_blocks, 1, 1].
        """
        b, c, h, w = x.shape
        coeff = self.block_dct2d(x)  # [B, C, N_blocks, 8, 8]
        effective_q = q_step_map * self.freq_weight.to(dtype=x.dtype)  # [B, 1, N_blocks, 8, 8]

        scaled = coeff / (effective_q + 1e-5)
        # Straight-Through Estimator (STE) for rounding + cubic relaxation (Reich et al., WACV 2024)
        rounded = torch.round(scaled)
        quantized = scaled + (rounded - scaled).detach()

        recon_coeff = quantized * effective_q
        x_recon = self.block_idct2d(recon_coeff, h, w)

        # Differentiable proxy rate (smooth L1 log-Laplacian entropy surrogate on DCT coefficients)
        rate_bpp = torch.log2(1.0 + scaled.abs()).mean(dim=(1, 2, 3, 4))
        return x_recon, rate_bpp

    def forward(
        self,
        video_edit: torch.Tensor,
        severity: torch.Tensor,
        ctu_qp_offsets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            video_edit: [B, T, 3, H, W] in [0, 1].
            severity: [B] normalized severity in [0, 1] (maps to QP in [22, 47]).
            ctu_qp_offsets: Optional [B, T, H_qg, W_qg] Delta-QP offsets in [-6, +6].
        """
        b, t, c, h, w = video_edit.shape
        bs = self.block_size
        num_blocks_h, num_blocks_w = h // bs, w // bs

        # Base QP from severity s in [0, 1] -> QP in [22, 47]
        base_qp = 22.0 + 25.0 * severity.view(b, 1, 1, 1)  # [B, 1, 1, 1]

        recon_frames = []
        frame_rates = []
        prev_recon: Optional[torch.Tensor] = None

        for step in range(t):
            if ctu_qp_offsets is not None:
                d_qp = ctu_qp_offsets[:, step].unsqueeze(1)  # [B, 1, H_qg, W_qg]
                d_qp_8x8 = F.interpolate(
                    d_qp, size=(num_blocks_h, num_blocks_w), mode="nearest"
                )
            else:
                d_qp_8x8 = torch.zeros(
                    (b, 1, num_blocks_h, num_blocks_w),
                    device=video_edit.device,
                    dtype=video_edit.dtype,
                )

            # HEVC quantization step relationship: QStep proportional to 2^((QP - 4)/6)
            block_qp = base_qp + d_qp_8x8  # [B, 1, H/8, W/8]
            q_step = 0.008 * torch.pow(2.0, (block_qp - 22.0) / 6.0)
            q_step_flat = q_step.view(b, 1, -1, 1, 1)

            curr = video_edit[:, step]
            if step == 0 or prev_recon is None:
                # Intra frame (I-frame)
                recon_t, rate_t = self.quantize_blocks(curr, q_step_flat)
                recon_t = torch.clamp(recon_t, 0.0, 1.0)
            else:
                # Inter frame (P-frame): encode temporal residual against previous reconstructed frame
                res_t = curr - prev_recon
                recon_res_t, rate_t = self.quantize_blocks(res_t, q_step_flat * 1.15)
                recon_t = torch.clamp(prev_recon + recon_res_t, 0.0, 1.0)

            recon_frames.append(recon_t)
            frame_rates.append(rate_t)
            prev_recon = recon_t

        video_recon = torch.stack(recon_frames, dim=1)  # [B, T, 3, H, W]
        rate_per_frame = torch.stack(frame_rates, dim=1)  # [B, T]
        return {
            "video_recon": video_recon,
            "rate_proxy_bpp": rate_per_frame.mean(dim=1),
            "rate_per_frame": rate_per_frame,
        }


class HybridX265STE(nn.Module):
    """
    Hybrid Forward-Real-Codec / Backward-Proxy Straight-Through Estimator (STE).
    Extends Lu et al. (IEEE TCSVT 2024, arXiv:2206.05650) from single-image BPG to video x265:
      - Forward pass: If `use_real_x265_in_forward=True` and ffmpeg/x265 is available, encodes
        the clip with real libx265 to obtain exact decoded frames `I_hat_x265` and real bpp.
      - Backward pass: Propagates gradients through `DifferentiableVideoProxy` via:
            I_hat = I_hat_proxy + (I_hat_x265 - I_hat_proxy).detach()
    """

    def __init__(self, use_real_x265_in_forward: bool = False, preset: str = "medium") -> None:
        super().__init__()
        self.proxy = DifferentiableVideoProxy(block_size=8)
        self.use_real_x265 = use_real_x265_in_forward and (shutil.which("ffmpeg") is not None)
        self.preset = preset

    @torch.no_grad()
    def encode_decode_x265_clip(
        self, clip: torch.Tensor, qp: int
    ) -> Tuple[torch.Tensor, float]:
        """
        Encodes a single clip [T, 3, H, W] in [0, 1] using real x265 via ffmpeg pipe/tempfiles
        and returns reconstructed clip [T, 3, H, W] and exact bits-per-pixel (bpp).
        """
        t, _, h, w = clip.shape
        frames_np = (clip.permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_in = Path(tmpdir) / "in.rgb"
            bitstream = Path(tmpdir) / "out.hevc"
            raw_out = Path(tmpdir) / "out.rgb"
            frames_np.tofile(str(raw_in))

            cmd_enc = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "30",
                "-i", str(raw_in),
                "-c:v", "libx265", "-preset", self.preset,
                "-x265-params", f"qp={qp}:aq-mode=2:log-level=none",
                str(bitstream),
            ]
            subprocess.run(cmd_enc, check=True)

            cmd_dec = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(bitstream),
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                str(raw_out),
            ]
            subprocess.run(cmd_dec, check=True)

            total_bits = bitstream.stat().st_size * 8.0
            bpp = total_bits / float(t * h * w)
            dec_np = np.fromfile(str(raw_out), dtype=np.uint8).reshape(t, h, w, 3)
            recon_clip = torch.from_numpy(dec_np).to(device=clip.device, dtype=clip.dtype) / 255.0
            recon_clip = recon_clip.permute(0, 3, 1, 2).contiguous()
            return recon_clip, bpp

    def forward(
        self,
        video_edit: torch.Tensor,
        severity: torch.Tensor,
        ctu_qp_offsets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        proxy_out = self.proxy(video_edit, severity, ctu_qp_offsets)
        video_proxy = proxy_out["video_recon"]

        if not self.use_real_x265:
            return proxy_out

        # Hybrid STE mode: run real x265 in forward pass, proxy in backward pass
        b = video_edit.shape[0]
        real_recons = []
        real_bpps = []
        for idx in range(b):
            qp_val = int(round(22.0 + 25.0 * float(severity[idx].item())))
            recon_i, bpp_i = self.encode_decode_x265_clip(video_edit[idx].detach(), qp_val)
            real_recons.append(recon_i)
            real_bpps.append(bpp_i)

        video_real = torch.stack(real_recons, dim=0)
        # Straight-Through Estimator (STE)
        video_ste = video_proxy + (video_real - video_proxy).detach()
        rate_real = torch.tensor(real_bpps, device=video_edit.device, dtype=video_edit.dtype)
        rate_ste = proxy_out["rate_proxy_bpp"] + (rate_real - proxy_out["rate_proxy_bpp"]).detach()

        return {
            "video_recon": video_ste,
            "rate_proxy_bpp": rate_ste,
            "rate_per_frame": proxy_out["rate_per_frame"],
        }
