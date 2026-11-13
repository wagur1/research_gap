import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        ac_intra_weight = torch.ones((1, 1, 1, block_size, block_size), dtype=torch.float32)
        ac_intra_weight[..., 0, 0] = 0.12
        self.register_buffer("ac_intra_weight", ac_intra_weight)

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
        self, x: torch.Tensor, q_step_map: torch.Tensor, is_inter: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable DCT quantization + calibrated C-infinity deadzone entropy rate estimation.
        Args:
            x: Signal (Intra frame or Inter residual) [B, C, H, W].
            q_step_map: Spatial quantization step per 8x8 block [B, 1, N_blocks, 1, 1].
            is_inter: True for P-frame residual coding, False for I-frame coding.
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

        # Calibrated C-infinity deadzone surrogate matching real libx265 I/P bpp across CRF 22..37
        mag = torch.sqrt(scaled * scaled + 1e-6)
        gate = torch.sigmoid((mag - 0.65) / 0.22)
        rate_scale = 0.36 if is_inter else 1.45
        rate_weight = 1.0 if is_inter else self.ac_intra_weight.to(dtype=x.dtype)
        rate_bpp = rate_scale * (rate_weight * gate * torch.log2(1.0 + mag)).mean(
            dim=(1, 2, 3, 4)
        )
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
                # Match the 16x16 macro-CTU resolution of the real ffmpeg addroi filtergraph
                gh, gw = min(16, h // 16), min(16, w // 16)
                d_qp_macro = F.interpolate(
                    d_qp, size=(gh, gw), mode="bilinear", align_corners=False
                )
                d_qp_8x8 = F.interpolate(
                    d_qp_macro, size=(num_blocks_h, num_blocks_w), mode="nearest"
                )
            else:
                d_qp_8x8 = torch.zeros(
                    (b, 1, num_blocks_h, num_blocks_w),
                    device=video_edit.device,
                    dtype=video_edit.dtype,
                )

            d_qp_8x8 = torch.clamp(d_qp_8x8, min=-0.18 * 51.0, max=0.14 * 51.0)
            # HEVC quantization step relationship: QStep proportional to 2^((QP - 4)/6)
            block_qp = torch.clamp(base_qp + d_qp_8x8, min=0.0, max=51.0)  # [B, 1, H/8, W/8]
            q_step = 0.008 * torch.pow(2.0, (block_qp - 22.0) / 6.0)
            q_step_flat = q_step.view(b, 1, -1, 1, 1)

            curr = video_edit[:, step]
            if step == 0 or prev_recon is None:
                # Intra frame (I-frame)
                recon_t, rate_t = self.quantize_blocks(curr, q_step_flat, is_inter=False)
                recon_t = torch.clamp(recon_t, 0.0, 1.0)
            else:
                # Inter frame (P-frame): encode temporal residual against previous reconstructed frame
                res_t = curr - prev_recon
                recon_res_t, rate_t = self.quantize_blocks(
                    res_t, q_step_flat * 1.15, is_inter=True
                )
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

    @staticmethod
    def _build_addroi_filter_from_offsets(
        ctu_qp_offsets: Optional[torch.Tensor], h: int, w: int
    ) -> List[str]:
        """
        Converts dynamic per-frame spatial CTU Delta-QP tensor [T, H_qg, W_qg] (or [H_qg, W_qg])
        in [-6, +6] into an exact per-frame ffmpeg `-vf split=T...;select='eq(n,t)',addroi=...;interleave=n=T`
        filtergraph (R05 fix: never averages `ctu_qp_offsets` across time `t`, so each frame `t`
        receives its own dynamic `AV_FRAME_DATA_REGIONS_OF_INTEREST` -> `x265_picture.quantOffsets`).
        """
        if ctu_qp_offsets is None:
            return []
        if ctu_qp_offsets.dim() == 2:
            grid_t = ctu_qp_offsets.detach().unsqueeze(0).unsqueeze(1)  # [1, 1, H_qg, W_qg]
        elif ctu_qp_offsets.dim() == 3:
            grid_t = ctu_qp_offsets.detach().unsqueeze(1)  # [T, 1, H_qg, W_qg]
        else:
            return []

        if float(grid_t.abs().max().item()) < 1e-4:
            return []

        t_frames = grid_t.shape[0]
        gh, gw = min(16, h // 16), min(16, w // 16)
        grid_16x16 = F.interpolate(grid_t, size=(gh, gw), mode="bilinear", align_corners=False)[:, 0]  # [T, gh, gw]
        cell_h, cell_w = h // gh, w // gw

        def build_frame_rois(frame_grid: torch.Tensor) -> List[str]:
            rois: List[str] = []
            for r in range(gh):
                y0 = r * cell_h
                bh = (h - y0) if r == gh - 1 else cell_h
                c = 0
                while c < gw:
                    d_qp = float(frame_grid[r, c].item())
                    if abs(d_qp) < 0.15:
                        c += 1
                        continue
                    # In ffmpeg addroi, qoffset in [-1, 1] scales quantizer range (51 QP steps in HEVC).
                    # Allow up to +0.14 (+7.14 QP) for background coarsening and up to -0.18 (-9.18 QP)
                    # for high-importance foreground ROI blocks.
                    qoff = max(-0.18, min(0.14, d_qp / 51.0))
                    # Merge horizontally adjacent cells with virtually identical qoffset (<= 0.005)
                    # to keep the addroi filter chain compact at 16x16 resolution.
                    c_end = c + 1
                    qoff_sum = qoff
                    while c_end < gw:
                        d_qp_next = float(frame_grid[r, c_end].item())
                        if abs(d_qp_next) < 0.15:
                            break
                        qoff_next = max(-0.18, min(0.14, d_qp_next / 51.0))
                        if abs(qoff_next - qoff) > 0.005:
                            break
                        qoff_sum += qoff_next
                        c_end += 1
                    qoff_avg = qoff_sum / float(c_end - c)
                    x0 = c * cell_w
                    bw = (w - x0) if c_end == gw else (c_end - c) * cell_w
                    clear_flag = 1 if len(rois) == 0 else 0
                    rois.append(
                        f"addroi=x={x0}:y={y0}:w={bw}:h={bh}:qoffset={qoff_avg:.5f}:clear={clear_flag}"
                    )
                    c = c_end
            return rois

        # Check if all T frames share an identical map (e.g., t=1 or iwa_gop_qp_reuse)
        all_same = t_frames == 1 or bool((grid_16x16 - grid_16x16[0:1]).abs().max().item() < 1e-5)
        if all_same:
            rois_0 = build_frame_rois(grid_16x16[0])
            return ["-vf", ",".join(rois_0)] if rois_0 else []

        # Build exact per-frame dynamic ROI filtergraph via split + select + addroi + interleave
        split_pads = "".join(f"[s{i}]" for i in range(t_frames))
        branches: List[str] = [f"split={t_frames}{split_pads}"]
        out_pads = "".join(f"[v{i}]" for i in range(t_frames))
        any_roi = False
        for i in range(t_frames):
            f_rois = build_frame_rois(grid_16x16[i])
            if f_rois:
                any_roi = True
                chain = ",".join([f"select='eq(n,{i})'"] + f_rois)
            else:
                chain = f"select='eq(n,{i})'"
            branches.append(f"[s{i}]{chain}[v{i}]")

        if not any_roi:
            return []
        branches.append(f"{out_pads}interleave=n={t_frames}")
        return ["-vf", ";".join(branches)]

    @torch.no_grad()
    def encode_decode_x265_clip(
        self,
        clip: torch.Tensor,
        qp: int,
        ctu_qp_offsets: Optional[torch.Tensor] = None,
        return_packet_sizes: bool = False,
    ) -> Tuple[torch.Tensor, float, torch.Tensor, List[str]]:
        """
        Encodes a single clip [T, 3, H, W] in [0, 1] using REAL `libx265` via ffmpeg,
        applies CTU/ROI quantOffsets via `addroi`, and extracts exact per-frame entropy-coded
        stream bytes (`pkt_size`) and frame types (`I`/`P`) via `ffprobe`.
        Fails closed if `ffmpeg` or `ffprobe` is missing or returns an error.
        """
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            raise RuntimeError("Real x265 evaluation requires both `ffmpeg` and `ffprobe` in PATH.")

        t, _, h, w = clip.shape
        frames_np = (
            torch.clamp(clip, 0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy() * 255.0
        ).round().astype(np.uint8)

        vf_args = self._build_addroi_filter_from_mask_or_offsets(ctu_qp_offsets, h, w)

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_in = Path(tmpdir) / "in.rgb"
            bitstream = Path(tmpdir) / "out.mp4"
            raw_out = Path(tmpdir) / "out.rgb"
            frames_np.tofile(str(raw_in))

            # If the filtergraph string exceeds 8,000 chars on Windows
            # (where CreateProcess caps command lines at 32,767 chars), pass it via temporary file (`-/vf <path>`).
            # On Linux (POSIX), sysconf(_SC_ARG_MAX) allows up to 2MB, so pass directly to prevent option errors on older ffmpeg.
            vf_cli_args = vf_args
            if sys.platform == "win32" and len(vf_args) == 2 and vf_args[0] == "-vf" and len(vf_args[1]) > 8000:
                vf_script = Path(tmpdir) / "vf_script.txt"
                vf_script.write_text(vf_args[1], encoding="utf-8")
                vf_cli_args = ["-/vf", str(vf_script)]

            # Use IPPP Low-Delay structure (bframes=0, keyint=t) and crf=qp with aq-mode=2
            # so libx265 honors AV_FRAME_DATA_REGIONS_OF_INTEREST -> x265_picture.quantOffsets
            x265_params = (
                f"crf={qp}:bframes=0:keyint={t}:min-keyint={t}:scenecut=0:aq-mode=2:log-level=none"
            )
            cmd_enc = (
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{w}x{h}",
                    "-r",
                    "25",
                    "-i",
                    str(raw_in),
                ]
                + vf_cli_args
                + [
                    "-c:v",
                    "libx265",
                    "-preset",
                    self.preset,
                    "-x265-params",
                    x265_params,
                    str(bitstream),
                ]
            )
            subprocess.run(cmd_enc, check=True)

            # Extract exact per-frame encoded byte sizes and frame types (I/P) via ffprobe
            cmd_probe = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_packets",
                "-show_frames",
                "-show_entries",
                "packet=pos,size:frame=pict_type,pkt_size,pkt_pos",
                "-of",
                "json",
                str(bitstream),
            ]
            probe_res = subprocess.run(cmd_probe, capture_output=True, text=True, check=True)
            probe_data = json.loads(probe_res.stdout)
            combined = probe_data.get("packets_and_frames", [])
            probe_frames = probe_data.get("frames", [row for row in combined if row.get("type") == "frame"])
            packets = probe_data.get("packets", [row for row in combined if row.get("type") == "packet"])
            packet_by_pos = {}
            for packet in packets:
                position = str(packet.get("pos", ""))
                size = int(packet.get("size", 0))
                if not position or position in packet_by_pos or size <= 0:
                    raise RuntimeError("ffprobe returned invalid or duplicate video packets")
                packet_by_pos[position] = size
            if len(packet_by_pos) != t:
                raise RuntimeError(f"ffprobe returned {len(packet_by_pos)} video packets, expected {t}")
            packet_sizes = []
            matched_positions = set()
            if len(probe_frames) != t:
                raise RuntimeError(
                    f"ffprobe returned {len(probe_frames)} frames, expected {t}"
                )

            frame_bpps_list: List[float] = []
            frame_types: List[str] = []
            for pf_idx, pf in enumerate(probe_frames):
                if "pkt_size" not in pf or "pict_type" not in pf:
                    raise RuntimeError(f"ffprobe frame {pf_idx} missing pkt_size or pict_type: {pf}")
                position = str(pf.get("pkt_pos", ""))
                if position in matched_positions or position not in packet_by_pos:
                    raise RuntimeError("ffprobe frame has no unique corresponding video packet")
                matched_positions.add(position)
                pkt_bytes = packet_by_pos[position]
                if pkt_bytes != int(pf["pkt_size"]):
                    raise RuntimeError("ffprobe packet size differs from matched frame pkt_size")
                packet_sizes.append(pkt_bytes)
                pict_t = str(pf["pict_type"]).strip()
                if pkt_bytes <= 0:
                    raise RuntimeError(f"ffprobe frame {pf_idx} returned non-positive pkt_size={pkt_bytes}")
                if pict_t not in ("I", "P", "B"):
                    raise RuntimeError(f"ffprobe frame {pf_idx} returned invalid pict_type={pict_t}")
                frame_bpps_list.append((pkt_bytes * 8.0) / float(h * w))
                frame_types.append(pict_t)

            cmd_dec = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(bitstream),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                str(raw_out),
            ]
            subprocess.run(cmd_dec, check=True)

            frame_bpps = torch.tensor(
                frame_bpps_list, device=clip.device, dtype=clip.dtype
            )
            total_bpp = float(frame_bpps.mean().item())
            dec_np = np.fromfile(str(raw_out), dtype=np.uint8).reshape(t, h, w, 3)
            recon_clip = (
                torch.from_numpy(dec_np).to(device=clip.device, dtype=clip.dtype) / 255.0
            )
            recon_clip = recon_clip.permute(0, 3, 1, 2).contiguous()
            if return_packet_sizes:
                return recon_clip, total_bpp, frame_bpps, frame_types, packet_sizes
            return recon_clip, total_bpp, frame_bpps, frame_types

    def _build_addroi_filter_from_mask_or_offsets(
        self, ctu_qp_offsets: Optional[torch.Tensor], h: int, w: int
    ) -> List[str]:
        return self._build_addroi_filter_from_offsets(ctu_qp_offsets, h, w)

    @torch.no_grad()
    def evaluate_real_x265(
        self,
        video_edit: torch.Tensor,
        qp: int,
        ctu_qp_offsets: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Strict real-codec evaluation pathway (Gate 0.3 & Gate 0.4):
        Never uses the proxy; encodes every clip with real `libx265`, passes `ctu_qp_offsets`
        via `addroi` -> `x265_picture.quantOffsets`, and measures actual stream bytes via `ffprobe`.
        """
        b = video_edit.shape[0]
        recons = []
        frame_rate_rows = []
        all_types: List[List[str]] = []
        all_packet_sizes: List[List[int]] = []

        for idx in range(b):
            off_i = ctu_qp_offsets[idx] if ctu_qp_offsets is not None else None
            recon_i, _, f_bpps_i, f_types_i, sizes_i = self.encode_decode_x265_clip(
                video_edit[idx].detach(), qp, off_i, return_packet_sizes=True
            )
            recons.append(recon_i)
            frame_rate_rows.append(f_bpps_i)
            all_types.append(f_types_i)
            all_packet_sizes.append(sizes_i)

        video_recon = torch.stack(recons, dim=0)
        rate_per_frame = torch.stack(frame_rate_rows, dim=0)  # [B, T]
        i_rates: List[float] = []
        p_rates: List[float] = []
        for b_i, f_types in enumerate(all_types):
            for t_i, ft in enumerate(f_types):
                val = float(rate_per_frame[b_i, t_i].item())
                if ft == "I":
                    i_rates.append(val)
                elif ft in ("P", "B"):
                    p_rates.append(val)
        return {
            "video_recon": video_recon,
            "rate_per_frame": rate_per_frame,
            "total_bpp": float(rate_per_frame.mean().item()),
            "i_frame_bpp": float(np.mean(i_rates)) if i_rates else float("nan"),
            "p_frame_bpp": float(np.mean(p_rates)) if p_rates else float("nan"),
            "frame_types": all_types[0] if all_types else [],
            "packet_sizes": all_packet_sizes[0] if b == 1 else all_packet_sizes,
            "height": video_edit.shape[-2],
            "width": video_edit.shape[-1],
            "bitrate_accounting": "mp4_video_packet_payload_excluding_container_and_global_headers",
            "rate_control_mode": f"crf={qp}:aq-mode=2:bframes=0:keyint={video_edit.shape[1]}",
        }

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

        # Hybrid STE mode: run real x265 in forward pass (with ctu_qp_offsets!), proxy in backward pass
        b = video_edit.shape[0]
        def _encode_one(idx: int):
            qp_val = int(round(22.0 + 25.0 * float(severity[idx].item())))
            off_i = ctu_qp_offsets[idx].detach() if ctu_qp_offsets is not None else None
            recon_i, bpp_i, f_bpps_i, _ = self.encode_decode_x265_clip(
                video_edit[idx].detach(), qp_val, off_i
            )
            return recon_i, bpp_i, f_bpps_i

        if b > 1:
            workers = min(b, max(2, os.cpu_count() or 4), 8)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_encode_one, range(b)))
        else:
            results = [_encode_one(0)]

        real_recons = [r[0] for r in results]
        real_bpps = [r[1] for r in results]
        real_frame_bpps = [r[2] for r in results]

        video_real = torch.stack(real_recons, dim=0)
        frame_bpps_real = torch.stack(real_frame_bpps, dim=0)
        # Straight-Through Estimator (STE) for both reconstructed pixels and per-frame rates
        video_ste = video_proxy + (video_real - video_proxy).detach()
        rate_real = torch.tensor(real_bpps, device=video_edit.device, dtype=video_edit.dtype)
        rate_ste = proxy_out["rate_proxy_bpp"] + (rate_real - proxy_out["rate_proxy_bpp"]).detach()
        rate_per_frame_ste = (
            proxy_out["rate_per_frame"]
            + (frame_bpps_real - proxy_out["rate_per_frame"]).detach()
        )

        return {
            "video_recon": video_ste,
            "rate_proxy_bpp": rate_ste,
            "rate_per_frame": rate_per_frame_ste,
        }
