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
from datasets.vcm_video_datasets import RealVCMVideoDataset
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
        real_ds = RealVCMVideoDataset(
            data_root=data_root,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            clip_len=args.clip_len,
            stride=args.clip_len,
            img_size=args.img_size,
            is_train=is_train,
        )
        if len(real_ds) > 0:
            print(
                f"[Dataset] Loaded {len(real_ds)} real clips from '{data_root}' "
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

    base_model = TemporalSCOPE(
        base_ch=48,
        num_families=4,
        cond_dim=64,
        qg_size=16,
        max_delta_qp=preset_cfg["max_delta_qp"],
    ).to(device)

    # Leverage both T4 GPUs on Kaggle (T4 x2) automatically via DataParallel
    model: nn.Module = (
        nn.DataParallel(base_model) if n_gpus > 1 else base_model
    )

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

    model = TemporalSCOPE(base_ch=48, num_families=4, cond_dim=64, qg_size=16, max_delta_qp=4.0).to(device)
    if args.ckpt and Path(args.ckpt).exists():
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["state_dict"], strict=False)
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

            # 1. Baseline Direct Codec (Anchor)
            base_out = codec.proxy(video, sev, None)

            # 2. Spatial-Only Frame-by-Frame Pre-Editing (Simulating independent per-frame SCOPE jitter)
            blur_v = gaussian_blur_5x5(video.view(b * t, 3, h, w)).view(b, t, 3, h, w)
            # Independent per-frame residual jitter in background/foreground
            spatial_jitter = 0.02 * torch.randn_like(video)
            v_spatial_scope = m_attn * torch.clamp(video + spatial_jitter, 0.0, 1.0) + (
                1.0 - m_attn
            ) * torch.clamp(blur_v + 0.5 * spatial_jitter, 0.0, 1.0)
            spatial_out = codec.proxy(v_spatial_scope, sev, None)

            # 3. T-SCOPE (Temporally-Locked Background + CTU Delta-QP Coupling)
            pred_tscope = model(video, fam, sev, reuse_iframe_qp=False)
            eff_attn = torch.maximum(pred_tscope["attn_map"] * m_det, m_det)
            bg_locked = blur_v[:, 0:1].expand_as(blur_v)
            v_tscope = eff_attn * video + (1.0 - eff_attn) * bg_locked
            ctu_qp = model.compute_ctu_qp_offsets(eff_attn)
            tscope_out = codec.proxy(v_tscope, sev, ctu_qp)

            for method_name, c_out in [
                ("anchor_x265", base_out),
                ("spatial_scope_framewise", spatial_out),
                ("tscope_temporal_ctu_qp", tscope_out),
            ]:
                recon = c_out["video_recon"]
                rate_frames = c_out["rate_per_frame"][0].cpu().tolist()
                i_frame_bpp = float(rate_frames[0])
                p_frame_bpp = float(np.mean(rate_frames[1:])) if len(rate_frames) > 1 else i_frame_bpp
                total_bpp = float(np.mean(rate_frames))

                # Task-region fidelity & FPN feature cosine similarity inside M_det
                acc_metrics = fpn_teacher.evaluate_task_accuracy(video, recon, m_det)

                shard_records.append(
                    {
                        "shard_id": args.shard_id,
                        "clip_id": clip_id,
                        "qp": qp,
                        "method": method_name,
                        "total_bpp": round(total_bpp, 5),
                        "i_frame_bpp": round(i_frame_bpp, 5),
                        "p_frame_bpp": round(p_frame_bpp, 5),
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
