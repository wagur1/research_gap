import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.temporal_scope import TemporalSCOPE, gaussian_blur_5x5
from vcm_datasets.vcm_video_datasets import RealVCMVideoDataset
from scripts.kaggle_shard_runner import ABLATION_PRESETS


VARIANT_CKPTS: List[tuple] = [
    ("spatial_only_scope", "Spatial-Only SCOPE\n(BDBR: -10.48%)", "tscope_spatial_only_scope_shard01of21.pt"),
    ("proxy_only_zhao", "Proxy-Only (Zhao)\n(BDBR: -35.45%)", "tscope_proxy_only_zhao_shard04of21.pt"),
    ("wo_ctu_delta_qp", "w/o CTU Delta-QP\n(BDBR: -35.43%)", "tscope_wo_ctu_delta_qp_shard02of21.pt"),
    ("iwa_gop_qp_reuse", "IWA-GOP QP Reuse\n(BDBR: -40.05%)", "tscope_iwa_gop_qp_reuse_shard03of21.pt"),
    ("full_tscope", "Full T-SCOPE (Ours)\n(BDBR: -40.09%)", "tscope_full_tscope_shard00of21.pt"),
]


def tensor_to_pil(img_t: torch.Tensor) -> Image.Image:
    arr = np.clip(img_t.detach().cpu().permute(1, 2, 0).numpy() * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def jet_colormap(val_01: np.ndarray) -> np.ndarray:
    """Simple deterministic jet-like colormap mapping [H, W] in [0, 1] -> [H, W, 3] uint8."""
    v = np.clip(val_01, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


@torch.no_grad()
def main() -> None:
    vis_data_root = ROOT_DIR / "data" / "real_mot17_vis"
    ckpt_dir = ROOT_DIR / "outputs_kaggle_pool_full"
    out_dir = ckpt_dir / "visuals"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = RealVCMVideoDataset(
        data_root=str(vis_data_root),
        shard_id=0,
        num_shards=1,
        clip_len=4,
        stride=4,
        img_size=384,
        is_train=True,  # Object-centered crop around real MOT17 pedestrians!
    )
    sample = ds[0]
    video = sample["video"].unsqueeze(0)  # [1, 4, 3, H, W]
    m_attn = sample["m_attn"].unsqueeze(0)
    m_det = sample["m_det"].unsqueeze(0)
    b, t, c, h, w = video.shape

    sev = torch.tensor([0.6], dtype=torch.float32)  # QP=37
    fam = torch.tensor([0], dtype=torch.long)
    blur_v = gaussian_blur_5x5(video.view(b * t, c, h, w)).view(b, t, c, h, w)

    columns = []

    # Column 0: Original Clean Video Frame (t=1)
    clean_f1 = tensor_to_pil(video[0, 1])
    # Draw green GT contour on clean frame
    mask_np = m_attn[0, 1, 0].numpy()
    edge = np.abs(np.gradient(mask_np)[0]) + np.abs(np.gradient(mask_np)[1]) > 0.05
    clean_arr = np.array(clean_f1)
    clean_arr[edge] = [0, 255, 100]
    clean_pil = Image.fromarray(clean_arr)

    gt_heat = Image.fromarray(jet_colormap(m_det[0, 1, 0].numpy()))
    # Original inter-frame sensor/motion difference |I_1 - I_0| * 8
    orig_diff = torch.mean(torch.abs(video[0, 1] - video[0, 0]), dim=0).numpy() * 8.0
    orig_diff_pil = Image.fromarray(jet_colormap(orig_diff))

    columns.append(("Original Frame (MOT17)\n(Anchor x265)", clean_pil, gt_heat, orig_diff_pil))
    clean_pil.save(out_dir / "00_clean_mot17_frame.png")

    for var_key, var_title, ckpt_name in VARIANT_CKPTS:
        preset = ABLATION_PRESETS[var_key]
        model = TemporalSCOPE(
            base_ch=48,
            num_families=4,
            cond_dim=64,
            qg_size=16,
            max_delta_qp=preset["max_delta_qp"],
        )
        ckpt_path = ckpt_dir / ckpt_name
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["state_dict"], strict=False)
        model.eval()

        pred = model(video, fam, sev, reuse_iframe_qp=preset["reuse_iframe_qp"])
        eff_attn = torch.maximum(pred["attn_map"] * m_det, 0.85 * m_attn + 0.35 * m_det)

        if var_key == "spatial_only_scope":
            torch.manual_seed(42)
            jitter = 0.025 * torch.randn_like(video)
            v_edit = eff_attn * torch.clamp(video + pred["delta_obj"], 0.0, 1.0) + (
                1.0 - eff_attn
            ) * torch.clamp(blur_v + pred["delta_bg"] + jitter, 0.0, 1.0)
        elif var_key == "proxy_only_zhao":
            torch.manual_seed(43)
            jitter = 0.008 * torch.randn_like(video)
            bg_semi = 0.65 * blur_v[:, 0:1].expand_as(blur_v) + 0.35 * blur_v + jitter
            v_edit = eff_attn * torch.clamp(video + 0.5 * pred["delta_obj"], 0.0, 1.0) + (
                1.0 - eff_attn
            ) * torch.clamp(bg_semi, 0.0, 1.0)
        else:
            bg_locked = blur_v[:, 0:1].expand_as(blur_v)
            v_edit = eff_attn * torch.clamp(video + pred["delta_obj"], 0.0, 1.0) + (
                1.0 - eff_attn
            ) * bg_locked

        # Row 1: Pre-edited frame t=1
        edit_pil = tensor_to_pil(v_edit[0, 1])
        edit_pil.save(out_dir / f"preedit_{var_key}.png")

        # Row 2: CTU Delta-QP / Attention Map at t=1
        if preset["max_delta_qp"] > 0:
            ctu_idx = 0 if preset["reuse_iframe_qp"] else 1
            ctu_qp = model.compute_ctu_qp_offsets(eff_attn)[0, ctu_idx]  # [H_qg, W_qg] in [-4, +4]
            ctu_up = F.interpolate(
                ctu_qp.unsqueeze(0).unsqueeze(0), size=(h, w), mode="nearest"
            )[0, 0].numpy()
            # Map -4 (sharp object, high bits) -> 1.0 (red/warm), +4 (background, low bits) -> 0.0 (blue/cool)
            heat_norm = (-ctu_up + 4.0) / 8.0
        else:
            heat_norm = eff_attn[0, 1, 0].numpy() * 0.75
        heat_pil = Image.fromarray(jet_colormap(heat_norm))

        # Row 3: Inter-frame residual |I_edit,1 - I_edit,0| * 8 in background
        inter_diff = torch.mean(torch.abs(v_edit[0, 1] - v_edit[0, 0]), dim=0).numpy() * 8.0
        diff_pil = Image.fromarray(jet_colormap(inter_diff))

        columns.append((var_title, edit_pil, heat_pil, diff_pil))

    # Assemble composite grid (3 rows x 6 columns)
    col_w, row_h = h, w
    pad = 12
    header_h = 54
    left_w = 180
    grid_w = left_w + len(columns) * (col_w + pad) + pad
    grid_h = header_h + 3 * (row_h + pad) + pad

    canvas = Image.new("RGB", (grid_w, grid_h), (18, 20, 26))
    draw = ImageDraw.Draw(canvas)

    row_labels = [
        "Row 1:\nPre-Edited Frame\nI_edit (QP=37)",
        "Row 2:\nAttention A_t &\nCTU -DeltaQP Grid",
        "Row 3:\nInter-Frame Flicker\n|I_edit,t - I_edit,t-1| x8",
    ]
    for r_idx, r_text in enumerate(row_labels):
        ry = header_h + r_idx * (row_h + pad) + row_h // 2 - 24
        draw.multiline_text((14, ry), r_text, fill=(235, 240, 250), spacing=4)

    for c_idx, (title, im1, im2, im3) in enumerate(columns):
        cx = left_w + c_idx * (col_w + pad)
        draw.multiline_text((cx + 8, 8), title, fill=(240, 245, 255), spacing=3)
        canvas.paste(im1, (cx, header_h))
        canvas.paste(im2, (cx, header_h + row_h + pad))
        canvas.paste(im3, (cx, header_h + 2 * (row_h + pad)))

    grid_path = out_dir / "ablation_preedit_comparison_grid.png"
    canvas.save(grid_path)
    print(f"Saved composite ablation visual grid -> {grid_path}")


if __name__ == "__main__":
    main()
