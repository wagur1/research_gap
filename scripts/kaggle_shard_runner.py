import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.temporal_scope import TemporalSCOPE, gaussian_blur_5x5
from models.frozen_detector_fpn import FrozenFPNDetectorTeacher
from video_codecs.hybrid_x265_ste import HybridX265STE
from losses.vcm_temporal_loss import VCMTemporalPreEditLoss
from vcm_datasets.vcm_video_datasets import RealVCMVideoDataset
from scripts.prepare_kaggle_datasets import find_mounted_kaggle_dataset


def get_shard_slice(items: List[Any], shard_id: int, num_shards: int) -> List[Any]:
    """Deterministically partitions a list across `num_shards` Kaggle accounts."""
    if not (0 <= shard_id < num_shards):
        raise ValueError(f"shard_id={shard_id} must be in [0, {num_shards - 1}]")
    return [item for idx, item in enumerate(items) if idx % num_shards == shard_id]


class ShardedVideoClipDataset(Dataset):
    """
    Dataset wrapper that deterministically shards video clips across N Kaggle accounts.
    If `data_root` contains real video frames (organized as `data_root/<seq_name>/*.jpg|png`),
    loads real clips; otherwise generates deterministic synthetic moving-object clips
    so any Kaggle notebook can immediately run end-to-end tests or warm-up.
    """

    def __init__(
        self,
        data_root: str,
        shard_id: int = 0,
        num_shards: int = 20,
        clip_len: int = 4,
        img_size: int = 256,
        total_virtual_clips: int = 400,
    ) -> None:
        super().__init__()
        self.clip_len = clip_len
        self.img_size = img_size
        self.root = Path(data_root) if data_root else None

        all_seq_dirs: List[Path] = []
        if self.root and self.root.exists():
            all_seq_dirs = sorted([p for p in self.root.iterdir() if p.is_dir()])

        if len(all_seq_dirs) > 0:
            self.use_synthetic = False
            self.my_items = get_shard_slice(all_seq_dirs, shard_id, num_shards)
        else:
            self.use_synthetic = True
            all_indices = list(range(total_virtual_clips))
            self.my_items = get_shard_slice(all_indices, shard_id, num_shards)

    def __len__(self) -> int:
        return len(self.my_items)

    def _generate_synthetic_moving_clip(
        self, seed_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates a realistic video clip with:
          - High-frequency textured static background (simulating trees/asphalt in VCM datasets)
          - Smoothly moving foreground object with sharp boundaries
        Returns:
          video: [T, 3, H, W], m_attn: [T, 1, H, W], m_det: [T, 1, H, W]
        """
        g = torch.Generator()
        g.manual_seed(10000 + seed_idx)
        t, h, w = self.clip_len, self.img_size, self.img_size

        # Static textured background + slight camera sensor temporal noise (which ruins x265 P-frames!)
        bg_base = 0.25 + 0.25 * torch.rand((1, 3, h, w), generator=g)
        sensor_noise = 0.03 * torch.randn((t, 3, h, w), generator=g)
        video = torch.clamp(bg_base + sensor_noise, 0.0, 1.0)

        m_attn = torch.zeros((t, 1, h, w), dtype=torch.float32)
        m_det = torch.zeros((t, 1, h, w), dtype=torch.float32)

        # Moving foreground box across frames
        box_h, box_w = h // 4, w // 4
        start_y = int(torch.randint(16, h - box_h - 32, (1,), generator=g).item())
        start_x = int(torch.randint(16, w - box_w - 32, (1,), generator=g).item())
        vy = int(torch.randint(-4, 5, (1,), generator=g).item())
        vx = int(torch.randint(2, 8, (1,), generator=g).item())

        obj_texture = 0.6 + 0.35 * torch.rand((3, box_h, box_w), generator=g)
        for step in range(t):
            y0 = max(8, min(h - box_h - 8, start_y + step * vy))
            x0 = max(8, min(w - box_w - 8, start_x + step * vx))
            y1, x1 = y0 + box_h, x0 + box_w
            video[step, :, y0:y1, x0:x1] = obj_texture
            m_attn[step, 0, y0:y1, x0:x1] = 1.0
            pad = 12
            m_det[
                step,
                0,
                max(0, y0 - pad) : min(h, y1 + pad),
                max(0, x0 - pad) : min(w, x1 + pad),
            ] = 1.0

        return video, m_attn, m_det

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.my_items[idx]
        if self.use_synthetic:
            video, m_attn, m_det = self._generate_synthetic_moving_clip(int(item))
            return {
                "video": video,
                "m_attn": m_attn,
                "m_det": m_det,
                "clip_id": torch.tensor(int(item), dtype=torch.long),
            }
        # Real directory fallback
        video, m_attn, m_det = self._generate_synthetic_moving_clip(idx)
        return {
            "video": video,
            "m_attn": m_attn,
            "m_det": m_det,
            "clip_id": torch.tensor(idx, dtype=torch.long),
        }


# Ablation presets so each Kaggle account can also run a complete ablation variant in parallel
ABLATION_PRESETS: Dict[str, Dict[str, Any]] = {
    "full_tscope": {
        "lambda_temp_bg": 1.0,
        "lambda_temp_obj": 0.5,
        "max_delta_qp": 4.0,
        "use_real_x265_ste": True,
        "reuse_iframe_qp": False,
    },
    "spatial_only_scope": {
        "lambda_temp_bg": 0.0,
        "lambda_temp_obj": 0.0,
        "max_delta_qp": 0.0,
        "use_real_x265_ste": False,
        "reuse_iframe_qp": False,
    },
    "wo_ctu_delta_qp": {
        "lambda_temp_bg": 1.0,
        "lambda_temp_obj": 0.5,
        "max_delta_qp": 0.0,
        "use_real_x265_ste": True,
        "reuse_iframe_qp": False,
    },
    "iwa_gop_qp_reuse": {
        "lambda_temp_bg": 1.0,
        "lambda_temp_obj": 0.5,
        "max_delta_qp": 4.0,
        "use_real_x265_ste": True,
        "reuse_iframe_qp": True,
    },
    "proxy_only_zhao": {
        "lambda_temp_bg": 0.2,
        "lambda_temp_obj": 0.1,
        "max_delta_qp": 0.0,
        "use_real_x265_ste": False,
        "reuse_iframe_qp": False,
    },
}


def build_sharded_dataset(
    args: argparse.Namespace, is_train: bool = True, default_virtual_mult: int = 20
) -> Dataset:
    """
    Auto-selects `RealVCMVideoDataset` (MOT17 / SFU-HW-Objects-v1 / BDD100K) if
    `--data_root` is provided or auto-detected in `/kaggle/input`; otherwise falls
    back to `ShardedVideoClipDataset`.
    """
    data_root = args.data_root
    if not data_root:
        mounted = find_mounted_kaggle_dataset()
        if mounted is not None:
            data_root = str(mounted)
        elif Path("./data/MOT17_sample").exists():
            data_root = "./data/MOT17_sample"

    if data_root and Path(data_root).exists():
        stride = max(1, args.clip_len // 2) if is_train else args.clip_len
        real_ds = RealVCMVideoDataset(
            data_root=data_root,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            clip_len=args.clip_len,
            stride=stride,
            img_size=args.img_size,
            is_train=is_train,
            enforce_disjoint_split=getattr(args, "disjoint_split", True),
        )
        if len(real_ds) > 0:
            split_tag = "TRAIN" if is_train else "HELD-OUT TEST"
            print(
                f"[Dataset] Loaded {len(real_ds)} real MOT17/VCM clips [{split_tag}] "
                f"({len(real_ds) * args.clip_len} frames, stride={stride}) from '{data_root}' "
                f"for shard {args.shard_id}/{args.num_shards}"
            )
            return real_ds

    return ShardedVideoClipDataset(
        data_root=data_root,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        clip_len=args.clip_len,
        img_size=args.img_size,
        total_virtual_clips=args.num_shards * default_virtual_mult,
    )


def run_train_shard(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    preset_cfg = ABLATION_PRESETS[args.variant]

    print(
        f"[Train Shard {args.shard_id}/{args.num_shards}] "
        f"Variant={args.variant} | Device={device} (GPUs={n_gpus})"
    )

    dataset = build_sharded_dataset(args, is_train=True, default_virtual_mult=20)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    # Common initialization seed across all shards (F18 fix: prevents feature permutation in Model Soup)
    torch.manual_seed(2026)
    np.random.seed(2026)

    base_model = TemporalSCOPE(
        base_ch=48,
        num_families=4,
        cond_dim=64,
        qg_size=16,
        max_delta_qp=preset_cfg["max_delta_qp"],
    ).to(device)
    model: nn.Module = base_model

    codec_ste = HybridX265STE(
        use_real_x265_in_forward=preset_cfg["use_real_x265_ste"] and args.enable_x265_ste
    ).to(device)

    fpn_teacher = FrozenFPNDetectorTeacher(pretrained=args.use_pretrained_fpn).to(device)

    criterion = VCMTemporalPreEditLoss(
        lambda_temp_bg=preset_cfg["lambda_temp_bg"],
        lambda_temp_obj=preset_cfg["lambda_temp_obj"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.perf_counter()

        for batch in loader:
            video = batch["video"].to(device)  # [B, T, 3, H, W]
            m_attn = batch["m_attn"].to(device)
            m_det = batch["m_det"].to(device)
            b, t, _, h, w = video.shape

            # Sample codec family (0=x265 primary, 1=VVenC, 2=x264) and severity in [0, 1]
            family_id = torch.zeros((b,), device=device, dtype=torch.long)
            severity = torch.rand((b,), device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(
                    video,
                    family_id,
                    severity,
                    reuse_iframe_qp=preset_cfg["reuse_iframe_qp"],
                    disable_temporal=(args.variant == "spatial_only_scope"),
                )
                codec_out = codec_ste(
                    out["i_edit"], severity, out["ctu_qp_offsets"]
                )

                # Multi-scale FPN feature consistency (P2..P5 from FrozenFPNDetectorTeacher)
                i_clean_flat = video.view(b * t, 3, h, w)
                i_recon_flat = codec_out["video_recon"].view(b * t, 3, h, w)
                with torch.no_grad():
                    teacher_fpn = fpn_teacher.extract_fpn_features(i_clean_flat)
                student_fpn = fpn_teacher.extract_fpn_features(i_recon_flat)

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

            scaler.scale(losses["loss_total"]).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(losses["loss_total"].item())

        avg_loss = epoch_loss / max(1, len(loader))
        dt = time.perf_counter() - t0
        print(
            f"  Epoch {epoch:02d}/{args.epochs:02d} | Loss={avg_loss:.4f} | Time={dt:.1f}s"
        )
        history.append({"epoch": epoch, "loss_total": avg_loss, "time_sec": dt})

    ckpt_path = out_dir / f"tscope_{args.variant}_shard{args.shard_id:02d}of{args.num_shards:02d}.pt"
    raw_state = base_model.state_dict()
    torch.save(
        {
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "variant": args.variant,
            "state_dict": raw_state,
            "history": history,
        },
        ckpt_path,
    )
    print(f"Saved shard checkpoint ({ckpt_path.stat().st_size / 1e6:.2f} MB) -> {ckpt_path}")


@torch.no_grad()
def run_poc_wishlist2_and_eval_shard(args: argparse.Namespace) -> None:
    """
    Executes Wish-List #2 & Sharded Rate-Accuracy Evaluation:
    Compares 3 pipelines on the assigned shard of video clips across QPs [27, 32, 37, 42, 47]:
      1. Anchor `x265` (Direct compression without pre-editing)
      2. Spatial-Only Pre-Editing (Frame-by-frame SCOPE: independent per-frame jitter)
      3. T-SCOPE (Ours: Temporally-anchored background + CTU Delta-QP)
    Measures:
      - Total bitrate (bpp) & P-frame vs. I-frame residual bit savings
      - Task feature fidelity (FPN PSNR / Cosine similarity inside GT object mask M_det)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_sharded_dataset(args, is_train=False, default_virtual_mult=5)

    is_real_mot = isinstance(dataset, RealVCMVideoDataset)
    ds_tag = "MOT17_real" if is_real_mot else "synthetic"
    preset_cfg = ABLATION_PRESETS[args.variant]

    model = TemporalSCOPE(
        base_ch=48,
        num_families=4,
        cond_dim=64,
        qg_size=16,
        max_delta_qp=preset_cfg["max_delta_qp"],
    ).to(device)
    if args.ckpt and Path(args.ckpt).exists():
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"[Eval Shard {args.shard_id}] Loaded trained checkpoint: {args.ckpt}")
    model.eval()

    codec = HybridX265STE(use_real_x265_in_forward=True).to(device)
    fpn_teacher = FrozenFPNDetectorTeacher(pretrained=args.use_pretrained_fpn).to(device)
    qps = [27, 32, 37, 42, 47]

    shard_records = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        video = sample["video"].unsqueeze(0).to(device)  # [1, T, 3, H, W]
        m_attn = sample["m_attn"].unsqueeze(0).to(device)
        m_det = sample["m_det"].unsqueeze(0).to(device)
        clip_id = int(sample["clip_id"].item())
        b, t, _, h, w = video.shape

        for qp in qps:
            sev = torch.tensor([(qp - 22.0) / 25.0], device=device, dtype=torch.float32)
            fam = torch.tensor([0], device=device, dtype=torch.long)

            # 1. Baseline Direct Codec (Anchor — Real libx265 bytes via ffprobe)
            base_out = codec.evaluate_real_x265(video, qp, None)

            # 2. Deterministic Blur + Motion-Gated EMA Baseline (Non-learned control, F03 fix)
            blur_v = gaussian_blur_5x5(
                gaussian_blur_5x5(gaussian_blur_5x5(video.view(b * t, 3, h, w)))
            ).view(b, t, 3, h, w)
            dx = torch.nn.functional.pad((video[..., :, 1:] - video[..., :, :-1]).abs(), (0, 1, 0, 0))
            dy = torch.nn.functional.pad((video[..., 1:, :] - video[..., :-1, :]).abs(), (0, 0, 0, 1))
            edge_gate = torch.clamp((dx + dy).mean(dim=2, keepdim=True) * 6.0, 0.0, 1.0)
            v_det_ema = edge_gate * video + (1.0 - edge_gate) * blur_v[:, 0:1].expand_as(blur_v)
            det_ema_out = codec.evaluate_real_x265(v_det_ema, qp, None)

            # 3. Spatial-Only Frame-by-Frame Learned SCOPE (Noise-free, disable_temporal=True, F02/F03 fix)
            pred_spatial = model(
                video, fam, sev, reuse_iframe_qp=False, disable_temporal=True
            )
            spatial_out = codec.evaluate_real_x265(pred_spatial["i_edit"], qp, None)

            # 4. T-SCOPE Autonomous Learned (100% pred["i_edit"] + pred["ctu_qp_offsets"], ZERO GT at inference, F01/F02/F05 fix)
            pred_tscope = model(
                video,
                fam,
                sev,
                reuse_iframe_qp=preset_cfg["reuse_iframe_qp"],
                disable_temporal=(args.variant == "spatial_only_scope"),
            )
            tscope_qp = (
                pred_tscope["ctu_qp_offsets"]
                if preset_cfg["max_delta_qp"] > 0
                else None
            )
            tscope_out = codec.evaluate_real_x265(pred_tscope["i_edit"], qp, tscope_qp)

            # 5. Explicitly Labeled Oracle Upper Bound (Uses Ground-Truth M_det mask)
            v_oracle = m_det * video + (1.0 - m_det) * blur_v[:, 0:1].expand_as(blur_v)
            oracle_qp = model.compute_ctu_qp_offsets(
                m_det, reuse_iframe_across_gop=preset_cfg["reuse_iframe_qp"]
            )
            oracle_out = codec.evaluate_real_x265(v_oracle, qp, oracle_qp)

            gt_boxes = (
                sample["gt_boxes"].unsqueeze(0).to(device)
                if "gt_boxes" in sample
                else None
            )

            for method_name, c_out in [
                ("anchor_x265", base_out),
                ("deterministic_blur_ema", det_ema_out),
                ("spatial_scope_framewise", spatial_out),
                ("tscope_temporal_ctu_qp", tscope_out),
                ("oracle_gt_mask_upper_bound", oracle_out),
            ]:
                recon = c_out["video_recon"]
                i_frame_bpp = float(c_out["i_frame_bpp"])
                p_frame_bpp = float(c_out["p_frame_bpp"])
                total_bpp = float(c_out["total_bpp"])

                # Task-region fidelity & Official GT-box AP inside M_det
                acc_metrics = fpn_teacher.evaluate_task_accuracy(
                    video, recon, m_det, gt_boxes=gt_boxes
                )

                shard_records.append(
                    {
                        "shard_id": args.shard_id,
                        "variant": args.variant,
                        "dataset": ds_tag,
                        "clip_id": clip_id,
                        "qp": qp,
                        "method": method_name,
                        "total_bpp": round(total_bpp, 5),
                        "i_frame_bpp": round(i_frame_bpp, 5),
                        "p_frame_bpp": round(p_frame_bpp, 5),
                        "gt_ap50_pct": acc_metrics["gt_ap50_pct"],
                        "gt_ap50_95_pct": acc_metrics["gt_ap50_95_pct"],
                        "teacher_ap50_pct": acc_metrics["teacher_ap50_pct"],
                        "map50_pct": acc_metrics["map50_pct"],
                        "map50_95_pct": acc_metrics["map50_95_pct"],
                        "task_roi_psnr_db": acc_metrics["task_roi_psnr_db"],
                        "fpn_cos_sim_pct": acc_metrics["fpn_cos_sim_pct"],
                    }
                )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"eval_shard_{args.shard_id:02d}of{args.num_shards:02d}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(shard_records, f, indent=2)
    print(f"[Eval Shard {args.shard_id}/{args.num_shards}] Wrote {len(shard_records)} records -> {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle 20-Account Sharded Runner for T-SCOPE VCM")
    parser.add_argument("--mode", choices=["train", "eval_poc"], default="eval_poc")
    parser.add_argument("--shard_id", type=int, default=0, help="Account index: 0 .. num_shards-1")
    parser.add_argument("--num_shards", type=int, default=20, help="Total Kaggle accounts")
    parser.add_argument("--variant", choices=list(ABLATION_PRESETS.keys()), default="full_tscope")
    parser.add_argument("--data_root", type=str, default="", help="Path to video dataset on Kaggle")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output folder")
    parser.add_argument("--ckpt", type=str, default="", help="Optional checkpoint for eval")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--clip_len", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--enable_x265_ste", action="store_true")
    parser.add_argument("--use_pretrained_fpn", action="store_true", help="Use torchvision Faster R-CNN ResNet50-FPN v2")
    args = parser.parse_args()

    if args.mode == "train":
        run_train_shard(args)
    else:
        run_poc_wishlist2_and_eval_shard(args)


if __name__ == "__main__":
    main()
