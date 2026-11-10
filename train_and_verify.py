import time
import torch
from models.temporal_scope import TemporalSCOPE
from video_codecs.hybrid_x265_ste import HybridX265STE
from losses.vcm_temporal_loss import VCMTemporalPreEditLoss


def verify_pipeline() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running T-SCOPE verification on device: {device}")

    model = TemporalSCOPE(
        base_ch=48, num_families=4, cond_dim=64, qg_size=16, max_delta_qp=4.0
    ).to(device)
    codec_ste = HybridX265STE(use_real_x265_in_forward=False).to(device)
    criterion = VCMTemporalPreEditLoss().to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"T-SCOPE trainable parameters: {num_params:,} ({num_params / 1e6:.3f} M)")

    # Simulate batch of video clips: B=2, T=4 frames, 256x256
    b, t, h, w = 2, 4, 256, 256
    video = torch.rand((b, t, 3, h, w), device=device)
    family_id = torch.tensor([0, 1], device=device, dtype=torch.long)  # 0=x265, 1=VVenC
    severity = torch.tensor([0.35, 0.75], device=device, dtype=torch.float32)

    # Synthetic GT object masks and FPN teacher/student features
    m_attn = torch.zeros((b, t, 1, h, w), device=device)
    m_attn[:, :, :, 64:160, 64:160] = 1.0
    m_det = torch.zeros((b, t, 1, h, w), device=device)
    m_det[:, :, :, 48:176, 48:176] = 1.0

    t0 = time.perf_counter()
    out = model(video, family_id, severity, reuse_iframe_qp=False)
    codec_out = codec_ste(out["i_edit"], severity, out["ctu_qp_offsets"])

    # Simulate frozen FPN feature extraction (P2: H/4, P3: H/8)
    i_clean_flat = video.view(b * t, 3, h, w)
    i_recon_flat = codec_out["video_recon"].view(b * t, 3, h, w)
    teacher_fpn = {
        "p2": torch.nn.functional.avg_pool2d(i_clean_flat, 4).repeat(1, 16, 1, 1).detach(),
        "p3": torch.nn.functional.avg_pool2d(i_clean_flat, 8).repeat(1, 16, 1, 1).detach(),
    }
    student_fpn = {
        "p2": torch.nn.functional.avg_pool2d(i_recon_flat, 4).repeat(1, 16, 1, 1),
        "p3": torch.nn.functional.avg_pool2d(i_recon_flat, 8).repeat(1, 16, 1, 1),
    }

    losses = criterion(
        i_edit=out["i_edit"],
        i_recon=codec_out["video_recon"],
        attn_map=out["attn_map"],
        delta_obj=out["delta_obj"],
        delta_bg=out["delta_bg"],
        rate_proxy_bpp=codec_out["rate_proxy_bpp"],
        m_attn=m_attn,
        m_det=m_det,
        teacher_fpn_feats=teacher_fpn,
        student_fpn_feats=student_fpn,
    )

    losses["loss_total"].backward()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    grad_norm = sum(
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    )
    print(f"Output shapes:")
    print(f"  i_edit         : {tuple(out['i_edit'].shape)}")
    print(f"  attn_map       : {tuple(out['attn_map'].shape)}")
    print(f"  ctu_qp_offsets : {tuple(out['ctu_qp_offsets'].shape)} (min={out['ctu_qp_offsets'].min():.2f}, max={out['ctu_qp_offsets'].max():.2f})")
    print(f"  video_recon    : {tuple(codec_out['video_recon'].shape)}")
    print(f"  rate_per_frame : {codec_out['rate_per_frame'].detach().cpu().numpy().round(4).tolist()}")
    print(
        f"Losses: total={losses['loss_total'].item():.4f} | "
        f"task={losses['loss_task'].item():.4f} | "
        f"att={losses['loss_att'].item():.4f} | "
        f"bg={losses['loss_bg'].item():.4f} | "
        f"temp={losses['loss_temp'].item():.4f} | "
        f"reg={losses['loss_reg'].item():.4f}"
    )
    print(f"Backward pass verified! Total grad norm: {grad_norm:.4f} | Step time: {elapsed_ms:.1f} ms")


if __name__ == "__main__":
    verify_pipeline()
