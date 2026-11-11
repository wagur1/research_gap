from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.temporal_scope import gaussian_blur_5x5


class VCMTemporalPreEditLoss(nn.Module):
    """
    Complete Training Objective for T-SCOPE on Video Coding for Machines:
      L_total = L_task + L_attn + L_bg_spatial + L_temp + L_reg
    Combines:
      - SCOPE (arXiv:2610.39778) Eq. (14)-(19): Masked FPN teacher-student consistency,
        BCE+Dice attention supervision, spatial background simplification, residual TV.
      - New Temporal Inter-Frame Suppression Loss (L_temp): Explicitly suppresses
        inter-frame residual energy in background regions (M_bg) and smooths object
        residuals across consecutive frames so x265 P/B-frames achieve minimal bitrate.
    """

    def __init__(
        self,
        lambda_det: float = 2.0,
        lambda_attn: float = 4.0,
        lambda_attn_sparse: float = 1.5,
        lambda_bg_flat: float = 0.8,
        lambda_bg_smooth: float = 0.8,
        lambda_rate: float = 0.5,
        lambda_temp_bg: float = 1.5,
        lambda_temp_obj: float = 0.5,
        lambda_res_l1: float = 0.1,
        lambda_res_tv: float = 0.1,
    ) -> None:
        super().__init__()
        self.lambda_det = lambda_det
        self.lambda_attn = lambda_attn
        self.lambda_attn_sparse = lambda_attn_sparse
        self.lambda_bg_flat = lambda_bg_flat
        self.lambda_bg_smooth = lambda_bg_smooth
        self.lambda_rate = lambda_rate
        self.lambda_temp_bg = lambda_temp_bg
        self.lambda_temp_obj = lambda_temp_obj
        self.lambda_res_l1 = lambda_res_l1
        self.lambda_res_tv = lambda_res_tv

    @staticmethod
    def total_variation(x: torch.Tensor) -> torch.Tensor:
        tv_h = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
        tv_w = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
        return tv_h + tv_w

    @staticmethod
    def spatial_gradient_l1(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        dx = (x[..., :, 1:] - x[..., :, :-1]).abs()
        dy = (x[..., 1:, :] - x[..., :-1, :]).abs()
        mx = mask[..., :, 1:]
        my = mask[..., 1:, :]
        grad_x = (mx * dx).sum() / (mx.sum() * x.shape[2] + 1e-6)
        grad_y = (my * dy).sum() / (my.sum() * x.shape[2] + 1e-6)
        return grad_x + grad_y

    @staticmethod
    def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        inter = (pred * target).sum(dim=(-3, -2, -1))
        union = pred.sum(dim=(-3, -2, -1)) + target.sum(dim=(-3, -2, -1))
        dice = (2.0 * inter + eps) / (union + eps)
        return (1.0 - dice).mean()

    def forward(
        self,
        i_edit: torch.Tensor,
        i_recon: torch.Tensor,
        attn_map: torch.Tensor,
        delta_obj: torch.Tensor,
        delta_bg: torch.Tensor,
        rate_proxy_bpp: torch.Tensor,
        m_attn: torch.Tensor,
        m_det: torch.Tensor,
        teacher_fpn_feats: Dict[str, torch.Tensor],
        student_fpn_feats: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            i_edit: Pre-edited video [B, T, 3, H, W].
            i_recon: Reconstructed video after Codec/Proxy [B, T, 3, H, W].
            attn_map: Predicted attention A_t [B, T, 1, H, W].
            delta_obj, delta_bg: Predicted residuals [B, T, 3, H, W].
            rate_proxy_bpp: Estimated bitrate [B].
            m_attn: Tight GT object mask [B, T, 1, H, W].
            m_det: Expanded detector context mask [B, T, 1, H, W].
            teacher_fpn_feats: Dict of clean FPN feature tensors {level: [B*T, C_l, H_l, W_l]}.
            student_fpn_feats: Dict of compressed FPN feature tensors {level: [B*T, C_l, H_l, W_l]}.
        """
        b, t, c, h, w = i_edit.shape
        m_bg = 1.0 - m_det  # [B, T, 1, H, W]

        # 1. Detector Feature Consistency Loss (Eq. 18 in SCOPE)
        loss_det = torch.tensor(0.0, device=i_edit.device, dtype=i_edit.dtype)
        m_det_flat = m_det.view(b * t, 1, h, w)
        for level_name, f_t in teacher_fpn_feats.items():
            f_s = student_fpn_feats[level_name]
            m_l = F.interpolate(m_det_flat, size=f_t.shape[-2:], mode="nearest")
            diff_l = F.smooth_l1_loss(f_s, f_t.detach(), reduction="none")
            masked_diff = (m_l * diff_l).sum() / (m_l.sum() * f_t.shape[1] + 1e-6)
            loss_det = loss_det + masked_diff
        loss_task = self.lambda_det * loss_det

        # 2. Attention Supervision Loss (Eq. 17 in SCOPE) — disable AMP autocast for BCE safety on CUDA
        with torch.amp.autocast("cuda", enabled=False):
            attn_fp32 = torch.clamp(attn_map.float(), 1e-6, 1.0 - 1e-6)
            target_fp32 = m_attn.float()
            loss_bce = F.binary_cross_entropy(attn_fp32, target_fp32)
        loss_dice = self.dice_loss(attn_map, m_attn)
        loss_sparse = (m_bg * attn_map).mean()
        loss_att_group = (
            self.lambda_attn * (loss_bce + loss_dice)
            + self.lambda_attn_sparse * loss_sparse
        )

        # 3. Spatial Background Simplification + Rate Loss (Eq. 15-16 in SCOPE)
        i_recon_flat = i_recon.view(b * t, c, h, w)
        blur_recon = gaussian_blur_5x5(i_recon_flat).view(b, t, c, h, w)
        loss_bg_flat = (m_bg * (i_recon - blur_recon).abs()).sum() / (
            m_bg.sum() * c + 1e-6
        )
        loss_bg_smooth = self.spatial_gradient_l1(i_recon, m_bg)
        loss_bg_group = (
            self.lambda_bg_flat * loss_bg_flat
            + self.lambda_bg_smooth * loss_bg_smooth
            + self.lambda_rate * rate_proxy_bpp.mean()
        )

        # 4. Temporal Inter-Frame Residual Suppression Loss (Core Video Contribution)
        if t > 1:
            joint_bg_mask = m_bg[:, 1:] * m_bg[:, :-1]  # [B, T-1, 1, H, W]
            inter_bg_diff = (i_edit[:, 1:] - i_edit[:, :-1]).abs()
            loss_temp_bg = (joint_bg_mask * inter_bg_diff).sum() / (
                joint_bg_mask.sum() * c + 1e-6
            )
            loss_temp_obj = (delta_obj[:, 1:] - delta_obj[:, :-1]).abs().mean()
            loss_temp_group = (
                self.lambda_temp_bg * loss_temp_bg + self.lambda_temp_obj * loss_temp_obj
            )
        else:
            loss_temp_group = torch.tensor(0.0, device=i_edit.device, dtype=i_edit.dtype)

        # 5. Residual Regularization (Eq. 19 in SCOPE)
        loss_res_l1 = delta_obj.abs().mean() + delta_bg.abs().mean()
        loss_res_tv = self.total_variation(delta_obj) + self.total_variation(delta_bg)
        loss_reg_group = (
            self.lambda_res_l1 * loss_res_l1 + self.lambda_res_tv * loss_res_tv
        )

        total_loss = (
            loss_task
            + loss_att_group
            + loss_bg_group
            + loss_temp_group
            + loss_reg_group
        )

        return {
            "loss_total": total_loss,
            "loss_task": loss_task,
            "loss_att": loss_att_group,
            "loss_bg": loss_bg_group,
            "loss_temp": loss_temp_group,
            "loss_reg": loss_reg_group,
        }
