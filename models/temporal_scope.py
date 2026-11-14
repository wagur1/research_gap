import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Dict, Optional, Tuple


_GAUSSIAN_5X5_CACHE: Dict[Tuple[int, str, torch.dtype], torch.Tensor] = {}


class SeverityConditioner(nn.Module):
    """
    Encodes (codec_family_id, normalized_severity) into a compact conditioning vector.
    Extends SCOPE (arXiv:2610.39778) Eq. (1)-(3) to video codec families:
      0: x265 (HEVC)
      1: VVenC (VVC)
      2: x264 (AVC)
      3: Intra/BPG proxy
    """

    def __init__(
        self,
        num_families: int = 4,
        family_emb_dim: int = 8,
        cond_dim: int = 64,
    ) -> None:
        super().__init__()
        self.family_emb = nn.Embedding(num_families, family_emb_dim)
        # Basis: [s, s^2, s^3, h_0.50(s), h_0.75(s), h_0.90(s)] -> 6 dims
        in_dim = family_emb_dim + 6
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
        )

    @staticmethod
    def severity_basis(s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: Tensor of shape [B] or [B, 1] in [0, 1].
        Returns:
            Basis tensor of shape [B, 6].
        """
        if s.dim() == 1:
            s = s.unsqueeze(-1)
        s = torch.clamp(s, 0.0, 1.0)
        h50 = torch.clamp((s - 0.50) / 0.50, 0.0, 1.0)
        h75 = torch.clamp((s - 0.75) / 0.25, 0.0, 1.0)
        h90 = torch.clamp((s - 0.90) / 0.10, 0.0, 1.0)
        return torch.cat([s, s ** 2, s ** 3, h50, h75, h90], dim=-1)

    def forward(self, family_id: torch.Tensor, severity: torch.Tensor) -> torch.Tensor:
        ef = self.family_emb(family_id)  # [B, 8]
        phi_s = self.severity_basis(severity)  # [B, 6]
        return self.mlp(torch.cat([ef, phi_s], dim=-1))  # [B, cond_dim]


class FiLMConvBlock(nn.Module):
    """
    Two 3x3 Conv + optional GroupNorm + GELU layers modulated by Feature-wise Linear Modulation (FiLM):
        FiLM(x, c) = x * (1 + gamma(c)) + beta(c)
    """

    def __init__(
        self, in_ch: int, out_ch: int, cond_dim: int = 64, use_norm: bool = True
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        if use_norm:
            num_groups = min(8, out_ch)
            while num_groups > 1 and (out_ch % num_groups) != 0:
                num_groups -= 1
            self.norm1: nn.Module = nn.GroupNorm(num_groups, out_ch)
            self.norm2: nn.Module = nn.GroupNorm(num_groups, out_ch)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act2 = nn.GELU()
        self.film_proj = nn.Linear(cond_dim, out_ch * 2)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        gamma_beta = self.film_proj(c).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        h = h * (1.0 + gamma) + beta
        return self.act2(h)


class TemporalConvGRUCell(nn.Module):
    """
    Lightweight Bottleneck ConvGRU at H/4 x W/4 resolution.
    Ensures temporal consistency of object attention A_t and residuals across video frames
    so x265 Motion Estimation does not suffer from frame-to-frame pre-editing flicker.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.conv_gates = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1)
        self.conv_can = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)

    def forward(
        self, x_t: torch.Tensor, h_prev: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if h_prev is None:
            h_prev = torch.zeros_like(x_t)
        combined = torch.cat([x_t, h_prev], dim=1)
        gates = torch.sigmoid(self.conv_gates(combined))
        reset_gate, update_gate = torch.chunk(gates, 2, dim=1)
        candidate = torch.tanh(self.conv_can(torch.cat([x_t, reset_gate * h_prev], dim=1)))
        h_curr = (1.0 - update_gate) * h_prev + update_gate * candidate
        return h_curr


def gaussian_blur_5x5(x: torch.Tensor) -> torch.Tensor:
    """Fast separable 5x5 Gaussian blur with per-(channel, device, dtype) cached weights."""
    c = x.shape[1]
    key = (c, str(x.device), x.dtype)
    weight = _GAUSSIAN_5X5_CACHE.get(key)
    if weight is None:
        kernel_1d = torch.tensor(
            [1.0, 4.0, 6.0, 4.0, 1.0], device=x.device, dtype=x.dtype
        ) / 16.0
        kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, 5, 5)
        weight = kernel_2d.repeat(c, 1, 1, 1).contiguous()
        _GAUSSIAN_5X5_CACHE[key] = weight
    x_pad = F.pad(x, (2, 2, 2, 2), mode="reflect")
    return F.conv2d(x_pad, weight, groups=c)


class TemporalSCOPE(nn.Module):
    """
    T-SCOPE: Temporally-Consistent Semantic Pre-Editor & CTU Delta-QP Predictor.
    - Operates on a video clip I of shape [B, T, 3, H, W] (or single frame [B, 3, H, W]).
    - Outputs:
        1. I_edit: Pre-edited video [B, T, 3, H, W] ready for standard x265/VVenC encoding.
        2. attn_map: Object attention A_t [B, T, 1, H, W] in [0, 1].
        3. delta_obj, delta_bg: Bounded residual edits [B, T, 3, H, W].
        4. ctu_qp_offsets: Block-level Delta-QP map [B, T, H/qg_size, W/qg_size] for x265 quantOffsets.
    """

    def __init__(
        self,
        base_ch: int = 48,
        num_families: int = 4,
        cond_dim: int = 64,
        max_obj_res: float = 0.09,
        max_bg_res: float = 0.03,
        qg_size: int = 16,
        max_delta_qp: float = 4.0,
    ) -> None:
        super().__init__()
        self.max_obj_res = max_obj_res
        self.max_bg_res = max_bg_res
        self.qg_size = qg_size
        self.max_delta_qp = max_delta_qp

        self.conditioner = SeverityConditioner(
            num_families=num_families, family_emb_dim=8, cond_dim=cond_dim
        )

        # Encoder: HxW -> H/2xW/2 -> H/4xW/4 (+ H/8xW/8 global context branch)
        # Input channels: 3 (current frame I_t) + 3 (temporal frame diff I_t - I_{t-1}) = 6
        # Skip GroupNorm at full HxW resolution (enc1/dec1) to avoid FP32 bandwidth bottlenecks;
        # retain GroupNorm at H/2..H/8 where feature normalization matters most.
        self.enc1 = FiLMConvBlock(6, base_ch, cond_dim, use_norm=False)
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, kernel_size=4, stride=2, padding=1)
        self.enc2 = FiLMConvBlock(base_ch * 2, base_ch * 2, cond_dim, use_norm=True)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=4, stride=2, padding=1)
        self.bottleneck = FiLMConvBlock(base_ch * 4, base_ch * 4, cond_dim, use_norm=True)
        self.down3 = nn.Conv2d(base_ch * 4, base_ch * 4, kernel_size=4, stride=2, padding=1)
        self.enc3 = FiLMConvBlock(base_ch * 4, base_ch * 4, cond_dim, use_norm=True)
        self.ctx_fuse = FiLMConvBlock(base_ch * 8, base_ch * 4, cond_dim, use_norm=True)

        # Recurrent Temporal Alignment at H/4 x W/4
        self.temporal_gru = TemporalConvGRUCell(base_ch * 4)

        # Decoder: H/4xW/4 -> H/2xW/2 -> HxW
        self.up2 = nn.Conv2d(base_ch * 4, base_ch * 2, kernel_size=3, padding=1)
        self.dec2 = FiLMConvBlock(base_ch * 4, base_ch * 2, cond_dim, use_norm=True)
        self.up1 = nn.Conv2d(base_ch * 2, base_ch, kernel_size=3, padding=1)
        self.dec1 = FiLMConvBlock(base_ch * 2, base_ch, cond_dim, use_norm=False)

        # Output prediction heads
        self.attn_head = nn.Conv2d(base_ch, 1, kernel_size=3, padding=1)
        self.attn_cond_bias = nn.Linear(cond_dim, 1)
        self.obj_res_head = nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)
        self.bg_res_head = nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)
        self.bg_blur_strength = nn.Sequential(
            nn.Linear(cond_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        # Temporal EMA gate head for static background locking across P-frames
        self.bg_temporal_gate = nn.Conv2d(base_ch, 1, kernel_size=3, padding=1)
        nn.init.constant_(self.bg_temporal_gate.bias, 1.0)

    def compute_ctu_qp_offsets(
        self, attn_map: torch.Tensor, reuse_iframe_across_gop: bool = False
    ) -> torch.Tensor:
        """
        Converts pixel-level attention A_t [B, T, 1, H, W] into x265 QG/CTU-level
        Delta-QP offsets [B, T, H//qg_size, W//qg_size] using an absolute threshold pivot
        so sparse foreground preserves negative/neutral QP while background harvests +QP savings.
        """
        b, t, _, h, w = attn_map.shape
        attn_flat = attn_map.view(b * t, 1, h, w)
        avg_qg = F.avg_pool2d(
            attn_flat, kernel_size=self.qg_size, stride=self.qg_size
        )
        max_qg = F.max_pool2d(
            attn_flat, kernel_size=self.qg_size, stride=self.qg_size
        )
        # Foreground-sensitive pool mix prevents small pedestrians in 32x32 CTU cells from being coarsened.
        qg_attn = 0.75 * avg_qg + 0.25 * max_qg  # [B*T, 1, H_qg, W_qg]

        tau = 0.30
        bg_part = F.relu(tau - qg_attn) / tau
        delta_qp = self.max_delta_qp * (1.75 * bg_part)
        delta_qp = torch.clamp(delta_qp, min=0.0, max=0.14 * 51.0)
        delta_qp = delta_qp.view(b, t, delta_qp.shape[-2], delta_qp.shape[-1])

        if reuse_iframe_across_gop and t > 1:
            delta_qp = delta_qp[:, 0:1].expand_as(delta_qp)
        return delta_qp

    def forward_single_step(
        self,
        i_t: torch.Tensor,
        i_prev: torch.Tensor,
        c: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
        bg_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single frame step t with recurrent bottleneck state h_prev and previous background bg_prev."""
        diff_t = i_t - i_prev
        x_in = torch.cat([i_t, diff_t], dim=1)  # [B, 6, H, W]

        f1 = self.enc1(x_in, c)  # [B, C, H, W]
        f2 = self.enc2(self.down1(f1), c)  # [B, 2C, H/2, W/2]
        f3 = self.bottleneck(self.down2(f2), c)  # [B, 4C, H/4, W/4]
        f4 = self.enc3(self.down3(f3), c)  # [B, 4C, H/8, W/8]
        f4_up = F.interpolate(f4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        f3_ctx = self.ctx_fuse(torch.cat([f3, f4_up], dim=1), c)  # [B, 4C, H/4, W/4]

        h_curr = self.temporal_gru(f3_ctx, h_prev)  # [B, 4C, H/4, W/4]

        u2 = F.interpolate(h_curr, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.dec2(torch.cat([self.up2(u2), f2], dim=1), c)

        u1 = F.interpolate(u2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.dec1(torch.cat([self.up1(u1), f1], dim=1), c)

        attn_bias = self.attn_cond_bias(c).view(-1, 1, 1, 1)
        attn_t = torch.sigmoid(self.attn_head(u1) + attn_bias)  # [B, 1, H, W]
        delta_obj_t = self.max_obj_res * torch.tanh(self.obj_res_head(u1))
        delta_bg_t = self.max_bg_res * torch.tanh(self.bg_res_head(u1))

        # Object-preserving branch
        i_obj_t = torch.clamp(i_t + delta_obj_t, 0.0, 1.0)

        # Severity-conditioned low-pass background branch (5x5 -> multi-scale ~11x11 Gaussian blend)
        blur_mild = gaussian_blur_5x5(i_t)
        blur_strong = F.interpolate(
            gaussian_blur_5x5(F.avg_pool2d(blur_mild, kernel_size=2, stride=2)),
            size=i_t.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        blur_mix = self.bg_blur_strength(c).view(-1, 1, 1, 1)
        blur_i_t = (1.0 - blur_mix) * blur_mild + blur_mix * blur_strong
        i_bg_spatial = torch.clamp(blur_i_t + delta_bg_t, 0.0, 1.0)
        if bg_prev is not None:
            # Strong motion gating (exp(-30*mm)) avoids double-exposure ghosting on panning cameras
            diff_mag = diff_t.abs().mean(dim=1, keepdim=True)
            motion_mag = 0.7 * diff_mag + 0.3 * F.avg_pool2d(
                diff_mag, kernel_size=5, stride=1, padding=2
            )
            static_mask = torch.exp(-30.0 * motion_mag)
            temporal_weight = 0.92 * torch.sigmoid(self.bg_temporal_gate(u1)) * static_mask
            i_bg_t = temporal_weight * bg_prev + (1.0 - temporal_weight) * i_bg_spatial
        else:
            i_bg_t = i_bg_spatial

        # Direct foreground/background attention blend without deadzone saturation
        eff_attn_t = attn_t
        i_edit_t = eff_attn_t * i_obj_t + (1.0 - eff_attn_t) * i_bg_t
        return i_edit_t, attn_t, delta_obj_t, delta_bg_t, h_curr, i_bg_t

    def forward(
        self,
        video: torch.Tensor,
        family_id: torch.Tensor,
        severity: torch.Tensor,
        reuse_iframe_qp: bool = False,
        disable_temporal: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            video: [B, T, 3, H, W] or [B, 3, H, W] in [0, 1].
            family_id: [B] integer codec family index (0=x265, 1=VVenC, 2=x264, 3=Intra).
            severity: [B] normalized compression severity in [0, 1].
            reuse_iframe_qp: If True, reuses I-frame CTU Delta-QP across the GOP (IWA-QP mode).
            disable_temporal: If True, processes every frame independently (h_prev=None, bg_prev=None,
                i_prev=i_t), providing a strict, noise-free framewise Spatial-SCOPE baseline.
        """
        single_frame_input = video.dim() == 4
        if single_frame_input:
            video = video.unsqueeze(1)  # [B, 1, 3, H, W]

        b, t, _, h, w = video.shape
        c = self.conditioner(family_id, severity)  # [B, cond_dim]

        edits, attns, d_objs, d_bgs = [], [], [], []
        h_prev: Optional[torch.Tensor] = None
        bg_prev: Optional[torch.Tensor] = None

        use_ckpt = self.training and torch.is_grad_enabled() and t > 1
        for step in range(t):
            i_t = video[:, step]
            if disable_temporal:
                i_prev = i_t
                h_prev = None
                bg_prev = None
            else:
                i_prev = video[:, step - 1] if step > 0 else i_t
            if use_ckpt:
                i_edit_t, attn_t, d_obj_t, d_bg_t, h_prev, bg_prev = grad_checkpoint(
                    self.forward_single_step,
                    i_t,
                    i_prev,
                    c,
                    h_prev,
                    bg_prev,
                    use_reentrant=False,
                )
            else:
                i_edit_t, attn_t, d_obj_t, d_bg_t, h_prev, bg_prev = self.forward_single_step(
                    i_t, i_prev, c, h_prev, bg_prev
                )
            edits.append(i_edit_t)
            attns.append(attn_t)
            d_objs.append(d_obj_t)
            d_bgs.append(d_bg_t)

        i_edit = torch.stack(edits, dim=1)  # [B, T, 3, H, W]
        attn_map = torch.stack(attns, dim=1)  # [B, T, 1, H, W]
        delta_obj = torch.stack(d_objs, dim=1)
        delta_bg = torch.stack(d_bgs, dim=1)
        ctu_qp_offsets = self.compute_ctu_qp_offsets(
            attn_map, reuse_iframe_across_gop=reuse_iframe_qp
        )

        if single_frame_input:
            return {
                "i_edit": i_edit.squeeze(1),
                "attn_map": attn_map.squeeze(1),
                "delta_obj": delta_obj.squeeze(1),
                "delta_bg": delta_bg.squeeze(1),
                "ctu_qp_offsets": ctu_qp_offsets.squeeze(1),
            }

        return {
            "i_edit": i_edit,
            "attn_map": attn_map,
            "delta_obj": delta_obj,
            "delta_bg": delta_bg,
            "ctu_qp_offsets": ctu_qp_offsets,
        }
