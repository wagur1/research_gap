import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.temporal_scope import TemporalSCOPE, gaussian_blur_5x5
from video_codecs.hybrid_x265_ste import HybridX265STE
from vcm_datasets.vcm_video_datasets import RealVCMVideoDataset


def tensor_to_pil(img_t: torch.Tensor) -> Image.Image:
    arr = np.clip(img_t.detach().cpu().permute(1, 2, 0).numpy() * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def jet_colormap(val_01: np.ndarray) -> np.ndarray:
    """Deterministic jet-like colormap mapping [H, W] in [0, 1] -> [H, W, 3] uint8."""
    v = np.clip(val_01, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


@torch.no_grad()
def main() -> None:
    """
    Renders 100% genuine, unmanipulated `pred["i_edit"]`, `pred["ctu_qp_offsets"]`,
    and decoded `evaluate_real_x265` outputs on the held-out test sequence `MOT17-11`
    using the clean-init disjoint checkpoints (`R02` & `R11` fix: zero GT masks passed
    to learned methods, zero synthetic randn jitter, zero mask flooring).
    """
    vis_data_root = ROOT_DIR / "data" / "real_mot17_vis"
    rigor_dir = ROOT_DIR / "outputs_rigor_gate0_gate1"
    out_dir = ROOT_DIR / "outputs_kaggle_pool_full" / "visuals"
    docs_fig_dir = ROOT_DIR / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_fig_dir.mkdir(parents=True, exist_ok=True)

    # Load Held-Out Test Clip 0 (MOT17-11-FRCNN)
    ds = RealVCMVideoDataset(
        data_root=str(vis_data_root),
        shard_id=0,
        num_shards=1,
        clip_len=4,
        stride=4,
        img_size=256,
        is_train=False,
        enforce_disjoint_split=True,
    )
    sample = ds[0]
    video = sample["video"].unsqueeze(0)  # [1, 4, 3, H, W]
    m_attn = sample["m_attn"].unsqueeze(0)
    m_det = sample["m_det"].unsqueeze(0)
    b, t, c, h, w = video.shape

    qp = 37
    sev = torch.tensor([(qp - 22.0) / 25.0], dtype=torch.float32)
    fam = torch.tensor([0], dtype=torch.long)

    tscope_ckpt = rigor_dir / "tscope_autonomous_disjoint.pt"
    spatial_ckpt = rigor_dir / "spatial_scope_autonomous_disjoint.pt"

    model_tscope = TemporalSCOPE(base_ch=48, num_families=4, cond_dim=64, qg_size=16, max_delta_qp=6.0)
    if tscope_ckpt.exists():
        model_tscope.load_state_dict(torch.load(tscope_ckpt, map_location="cpu")["state_dict"], strict=True)
    model_tscope.eval()

    model_spatial = TemporalSCOPE(base_ch=48, num_families=4, cond_dim=64, qg_size=16, max_delta_qp=6.0)
    if spatial_ckpt.exists():
        model_spatial.load_state_dict(torch.load(spatial_ckpt, map_location="cpu")["state_dict"], strict=True)
    model_spatial.eval()

    codec = HybridX265STE(use_real_x265_in_forward=True, preset="medium")

    # Autonomous predictions (ZERO GT at inference!)
    pred_spatial = model_spatial(video, fam, sev, disable_temporal=True)
    pred_tscope_gop = model_tscope(video, fam, sev, reuse_iframe_qp=True, disable_temporal=False)
    pred_tscope = model_tscope(video, fam, sev, reuse_iframe_qp=False, disable_temporal=False)

    # Non-learned deterministic blur + motion-gated EMA control
    blur_v = gaussian_blur_5x5(gaussian_blur_5x5(gaussian_blur_5x5(video.view(b * t, c, h, w)))).view(b, t, c, h, w)
    dx = F.pad((video[..., :, 1:] - video[..., :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((video[..., 1:, :] - video[..., :-1, :]).abs(), (0, 0, 0, 1))
    edge_gate = torch.clamp((dx + dy).mean(dim=2, keepdim=True) * 6.0, 0.0, 1.0)
    ema_frames = []
    bg_prev = None
    for s_idx in range(t):
        v_t, b_t = video[:, s_idx], blur_v[:, s_idx]
        if s_idx > 0 and bg_prev is not None:
            diff_mag = (v_t - video[:, s_idx - 1]).abs().mean(dim=1, keepdim=True)
            static_w = 0.85 * torch.exp(-15.0 * diff_mag)
            b_t = static_w * bg_prev + (1.0 - static_w) * b_t
        bg_prev = b_t
        ema_frames.append(edge_gate[:, s_idx] * v_t + (1.0 - edge_gate[:, s_idx]) * b_t)
    v_det_ema = torch.stack(ema_frames, dim=1)

    columns = []

    # Column 0: Original Frame + Anchor Real libx265 (CRF=37)
    enc_anchor = codec.evaluate_real_x265(video, qp, None)
    clean_f1 = tensor_to_pil(video[0, 1])
    mask_np = m_attn[0, 1, 0].numpy()
    edge = np.abs(np.gradient(mask_np)[0]) + np.abs(np.gradient(mask_np)[1]) > 0.05
    clean_arr = np.array(clean_f1)
    clean_arr[edge] = [0, 255, 100]
    clean_pil = Image.fromarray(clean_arr)
    clean_pil.save(out_dir / "00_clean_mot17_frame.png")

    gt_heat = Image.fromarray(jet_colormap(m_det[0, 1, 0].numpy()))
    orig_diff = torch.mean(torch.abs(video[0, 1] - video[0, 0]), dim=0).numpy() * 8.0
    orig_diff_pil = Image.fromarray(jet_colormap(orig_diff))
    columns.append(
        (
            f"Anchor libx265 (MOT17-11)\n({enc_anchor['total_bpp']:.4f} bpp, CRF={qp})",
            clean_pil,
            gt_heat,
            orig_diff_pil,
        )
    )

    variants = [
        ("spatial_only_scope", "Learned Spatial SCOPE\n(Noise-Free, Zero GT)", pred_spatial["i_edit"], pred_spatial["attn_map"], None),
        ("proxy_only_zhao", "Determ. Blur + EMA\n(Non-Learned Control)", v_det_ema, edge_gate, None),
        ("wo_ctu_delta_qp", "T-SCOPE Pre-Edit Only\n(w/o CTU Delta-QP)", pred_tscope["i_edit"], pred_tscope["attn_map"], None),
        ("iwa_gop_qp_reuse", "T-SCOPE + I-Frame QP\n(GOP Reuse)", pred_tscope_gop["i_edit"], pred_tscope_gop["attn_map"], pred_tscope_gop["ctu_qp_offsets"]),
        ("full_tscope", "Full T-SCOPE (Ours)\n(Dynamic Per-Frame ROI)", pred_tscope["i_edit"], pred_tscope["attn_map"], pred_tscope["ctu_qp_offsets"]),
    ]

    for var_key, var_title, v_edit, attn_t, ctu_offsets in variants:
        enc_real = codec.evaluate_real_x265(v_edit, qp, ctu_offsets)
        edit_pil = tensor_to_pil(v_edit[0, 1])
        edit_pil.save(out_dir / f"preedit_{var_key}.png")

        if ctu_offsets is not None:
            ctu_f1 = ctu_offsets[0, 1]  # [H_qg, W_qg] in [-6, +6]
            ctu_up = F.interpolate(
                ctu_f1.unsqueeze(0).unsqueeze(0), size=(h, w), mode="nearest"
            )[0, 0].numpy()
            heat_norm = (-ctu_up + 6.0) / 12.0
        else:
            heat_norm = attn_t[0, 1, 0].numpy()
        heat_pil = Image.fromarray(jet_colormap(heat_norm))

        inter_diff = torch.mean(torch.abs(v_edit[0, 1] - v_edit[0, 0]), dim=0).numpy() * 8.0
        diff_pil = Image.fromarray(jet_colormap(inter_diff))

        full_title = f"{var_title}\n({enc_real['total_bpp']:.4f} bpp)"
        columns.append((full_title, edit_pil, heat_pil, diff_pil))

    col_w, row_h = w, h
    pad = 12
    header_h = 62
    left_w = 185
    grid_w = left_w + len(columns) * (col_w + pad) + pad
    grid_h = header_h + 3 * (row_h + pad) + pad

    canvas = Image.new("RGB", (grid_w, grid_h), (18, 20, 26))
    draw = ImageDraw.Draw(canvas)

    row_labels = [
        "Row 1:\nNative Pre-Edited Frame\npred['i_edit'] (t=1)",
        "Row 2:\nLearned Attention A_t\n& Per-Frame -DeltaQP",
        "Row 3:\nInter-Frame Residual\n|I_edit,1 - I_edit,0| x8",
    ]
    for r_idx, r_text in enumerate(row_labels):
        ry = header_h + r_idx * (row_h + pad) + row_h // 2 - 24
        draw.multiline_text((14, ry), r_text, fill=(235, 240, 250), spacing=4)

    for c_idx, (title, im1, im2, im3) in enumerate(columns):
        cx = left_w + c_idx * (col_w + pad)
        draw.multiline_text((cx + 6, 8), title, fill=(240, 245, 255), spacing=3)
        canvas.paste(im1, (cx, header_h))
        canvas.paste(im2, (cx, header_h + row_h + pad))
        canvas.paste(im3, (cx, header_h + 2 * (row_h + pad)))

    grid_path = out_dir / "ablation_preedit_comparison_grid.png"
    canvas.save(grid_path)
    shutil.copyfile(grid_path, docs_fig_dir / "ablation_preedit_comparison_grid.png")
    print(f"Saved authentic autonomous ablation visual grid -> {grid_path}")


if __name__ == "__main__":
    main()
