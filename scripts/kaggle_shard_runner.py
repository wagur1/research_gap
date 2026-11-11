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
from scripts.experiment_protocol import CODEC_CONFIGURATION, CRFS, METRIC_POLICY, EXPECTED_TEST_FRAME_IDS, EXCLUDED_TEST_TAIL_FRAME_IDS, sha256_file, validate_soup_provenance


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
    args: argparse.Namespace,
    is_train: bool = True,
    default_virtual_mult: int = 20,
    split: str = None,
    shard_id: int = None,
    num_shards: int = None,
) -> Dataset:
    """
    Auto-selects `RealVCMVideoDataset` (MOT17 / SFU-HW-Objects-v1 / BDD100K) if
    `--data_root` is provided or auto-detected in `/kaggle/input`; otherwise falls
    back to `ShardedVideoClipDataset`. `split` in {train, val, test} picks MOT17 sequences.
    """
    data_root = args.data_root
    if not data_root:
        mounted = find_mounted_kaggle_dataset()
        if mounted is not None:
            data_root = str(mounted)
        elif Path("./data/MOT17_sample").exists():
            data_root = "./data/MOT17_sample"

    sid = args.shard_id if shard_id is None else shard_id
    nsh = args.num_shards if num_shards is None else num_shards
    if data_root and Path(data_root).exists():
        stride = max(1, args.clip_len // 2) if is_train else args.clip_len
        real_ds = RealVCMVideoDataset(
            data_root=data_root,
            shard_id=sid,
            num_shards=nsh,
            clip_len=args.clip_len,
            stride=stride,
            img_size=args.img_size,
            is_train=is_train,
            enforce_disjoint_split=getattr(args, "disjoint_split", True),
            split=split,
        )
        if len(real_ds) > 0:
            split_tag = (split or ("train" if is_train else "test")).upper()
            print(
                f"[Dataset] Loaded {len(real_ds)} real MOT17/VCM clips [{split_tag}] "
                f"({len(real_ds) * args.clip_len} frames, stride={stride}) from '{data_root}' "
                f"for shard {sid}/{nsh}"
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

    dataset = build_sharded_dataset(
        args,
        is_train=True,
        default_virtual_mult=20,
        split="train",
        shard_id=args.data_shard_id if args.data_shard_id >= 0 else None,
        num_shards=args.data_num_shards if args.data_num_shards > 0 else None,
    )
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, args.patience // 3)
    )

    # Validation on a sequence-disjoint split (MOT17-09) for convergence detection; test is never used here.
    val_ds = None
    if args.patience > 0:
        val_candidate = build_sharded_dataset(
            args, is_train=False, split="val", shard_id=0, num_shards=1
        )
        if isinstance(val_candidate, RealVCMVideoDataset) and len(val_candidate) > 0:
            val_ds = torch.utils.data.Subset(
                val_candidate, list(range(0, len(val_candidate), max(1, args.val_subsample)))
            )
    VAL_SEVERITIES = (0.3, 0.7)

    @torch.no_grad()
    def validate() -> float:
        model.eval()
        total, count = 0.0, 0
        for vb in DataLoader(val_ds, batch_size=args.batch_size, shuffle=False):
            v_video = vb["video"].to(device)
            v_attn = vb["m_attn"].to(device)
            v_det = vb["m_det"].to(device)
            vb_n, vt, _, vh, vw = v_video.shape
            for sev_val in VAL_SEVERITIES:
                v_sev = torch.full((vb_n,), sev_val, device=device, dtype=torch.float32)
                v_fam = torch.zeros((vb_n,), device=device, dtype=torch.long)
                v_out = model(
                    v_video,
                    v_fam,
                    v_sev,
                    reuse_iframe_qp=preset_cfg["reuse_iframe_qp"],
                    disable_temporal=(args.variant == "spatial_only_scope"),
                )
                v_codec = codec_ste(v_out["i_edit"], v_sev, v_out["ctu_qp_offsets"])
                v_teacher = fpn_teacher.extract_fpn_features(v_video.view(vb_n * vt, 3, vh, vw))
                v_student = fpn_teacher.extract_fpn_features(
                    v_codec["video_recon"].view(vb_n * vt, 3, vh, vw)
                )
                v_losses = criterion(
                    i_edit=v_out["i_edit"],
                    i_recon=v_codec["video_recon"],
                    attn_map=v_out["attn_map"],
                    delta_obj=v_out["delta_obj"],
                    delta_bg=v_out["delta_bg"],
                    rate_proxy_bpp=v_codec["rate_proxy_bpp"],
                    m_attn=v_attn,
                    m_det=v_det,
                    teacher_fpn_feats=v_teacher,
                    student_fpn_feats=v_student,
                )
                total += float(v_losses["loss_total"].item())
                count += 1
        model.train()
        return total / max(1, count)

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    epochs_since_best = 0
    stopped_early = False

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    training_configuration = {
        "initialization_seed": 2026, "preset": preset_cfg, "codec": CODEC_CONFIGURATION,
        "epochs": args.epochs, "patience": args.patience, "min_epochs": args.min_epochs,
        "lr": args.lr, "batch_size": args.batch_size, "img_size": args.img_size,
        "clip_len": args.clip_len, "val_every": args.val_every, "val_subsample": args.val_subsample,
        "data_shard_id": args.data_shard_id, "data_num_shards": args.data_num_shards,
        "pretrained_detector_requested": args.use_pretrained_fpn,
        "real_detector_loaded": fpn_teacher.use_torchvision,
        "validation_severities": list(VAL_SEVERITIES),
    }
    try:
        training_configuration["git_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        training_configuration["git_revision"] = None

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

        if val_ds is not None and (epoch % args.val_every == 0 or epoch == args.epochs):
            val_loss = validate()
            scheduler.step(val_loss)
            history[-1]["val_loss"] = val_loss
            history[-1]["lr"] = optimizer.param_groups[0]["lr"]
            improved = val_loss < best_val - args.min_delta
            print(
                f"    [Val] epoch {epoch:03d} val_loss={val_loss:.4f} best={best_val:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} {'*' if improved else ''}"
            )
            if improved:
                best_val, best_epoch, epochs_since_best = val_loss, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
                # Incremental best-checkpoint save so a timed-out kernel (e.g. CPU fallback) still returns usable weights
                torch.save(
                    {
                        "shard_id": args.shard_id,
                        "num_shards": args.num_shards,
                        "variant": args.variant,
                        "state_dict": best_state,
                        "history": history,
                        "best_epoch": best_epoch,
                        "best_val_loss": best_val,
                        "stopped_early": False,
                        "epochs_run": len(history),
                        "partial": True,
                        "training_configuration": training_configuration,
                        "train_sequences": ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-10"],
                        "val_sequences": ["MOT17-09"],
                    },
                    out_dir / f"tscope_{args.variant}_shard{args.shard_id:02d}of{args.num_shards:02d}.pt",
                )
            else:
                epochs_since_best += args.val_every
            if epoch >= args.min_epochs and epochs_since_best >= args.patience:
                stopped_early = True
                print(f"  Converged: no val improvement for {epochs_since_best} epochs. Stop at epoch {epoch}.")
                break

    ckpt_path = out_dir / f"tscope_{args.variant}_shard{args.shard_id:02d}of{args.num_shards:02d}.pt"
    raw_state = best_state if best_state is not None else base_model.state_dict()
    torch.save(
        {
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "variant": args.variant,
            "state_dict": raw_state,
            "history": history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val if best_state is not None else None,
            "stopped_early": stopped_early,
            "epochs_run": len(history),
            "training_configuration": training_configuration,
            "train_sequences": ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-10"],
            "val_sequences": ["MOT17-09"],
        },
        ckpt_path,
    )
    print(f"Saved shard checkpoint ({ckpt_path.stat().st_size / 1e6:.2f} MB) -> {ckpt_path}")


@torch.no_grad()
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
    if not isinstance(dataset, RealVCMVideoDataset) or not args.use_pretrained_fpn or not getattr(args, "disjoint_split", True):
        raise RuntimeError("Final evaluation requires real held-out MOT17 data, disjoint splits, and pretrained Faster R-CNN")
    expected_frame_ids = dataset.physical_frame_manifest()
    if expected_frame_ids != EXPECTED_TEST_FRAME_IDS or args.clip_len != 4 or args.img_size != 256:
        raise RuntimeError("Final evaluation requires all 1,648 complete-clip MOT17 test frames at the preserved 256px/4-frame protocol")

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
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
        validate_soup_provenance(ckpt, args.variant, require_validation=True)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print(f"[Eval Shard {args.shard_id}] Loaded trained checkpoint: {args.ckpt}")
    else:
        raise FileNotFoundError(
            f"[R15 Gate] Validated soup checkpoint required for eval_poc but not found: '{args.ckpt}'."
        )
    model.eval()

    # Load dedicated spatial_only_scope checkpoint if available in same folder (R08 fix)
    spatial_model = TemporalSCOPE(
        base_ch=48, num_families=4, cond_dim=64, qg_size=16, max_delta_qp=0.0
    ).to(device)
    spatial_ckpt = Path(args.ckpt).parent / "tscope_soup_spatial_only_scope.pt"
    if not spatial_ckpt.is_file():
        raise FileNotFoundError("Final evaluation requires the dedicated spatial variant soup")
    s_state = torch.load(spatial_ckpt, map_location=device, weights_only=True)
    validate_soup_provenance(s_state, "spatial_only_scope", require_validation=True)
    spatial_model.load_state_dict(s_state["state_dict"], strict=True)
    spatial_model.eval()

    codec = HybridX265STE(use_real_x265_in_forward=True).to(device)
    fpn_teacher = FrozenFPNDetectorTeacher(pretrained=args.use_pretrained_fpn).to(device)
    if not fpn_teacher.use_torchvision or fpn_teacher.detector is None:
        raise RuntimeError("Final AP requires actual pretrained Faster R-CNN weights; detector fallback is forbidden")
    qps = CRFS
    metadata = {
        "shard_id": args.shard_id, "num_shards": args.num_shards, "variant": args.variant,
        "checkpoint_sha256": sha256_file(args.ckpt), "spatial_checkpoint_sha256": sha256_file(spatial_ckpt),
        "crfs": CRFS, "dataset": "MOT17", "split": "test", "expected_frame_ids": expected_frame_ids,
        "excluded_tail_frame_ids": EXCLUDED_TEST_TAIL_FRAME_IDS,
        "detector": {"name": "fasterrcnn_resnet50_fpn_v2", "pretrained": True, "weights": "FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT"},
        "codec": CODEC_CONFIGURATION, "metric_policy": METRIC_POLICY,
        "preprocessing": {"resize": "bilinear_square", "img_size": args.img_size, "clip_len": args.clip_len, "stride": args.clip_len, "tail_policy": "complete_nonoverlapping_clips_only"},
        "soup_validation": ckpt["validation"],
    }

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

            # 2. Deterministic Blur + Motion-Gated EMA Baseline (Non-learned control, F03/R14 fix)
            blur_v = gaussian_blur_5x5(
                gaussian_blur_5x5(gaussian_blur_5x5(video.view(b * t, 3, h, w)))
            ).view(b, t, 3, h, w)
            dx = torch.nn.functional.pad((video[..., :, 1:] - video[..., :, :-1]).abs(), (0, 1, 0, 0))
            dy = torch.nn.functional.pad((video[..., 1:, :] - video[..., :-1, :]).abs(), (0, 0, 0, 1))
            edge_gate = torch.clamp((dx + dy).mean(dim=2, keepdim=True) * 6.0, 0.0, 1.0)
            ema_frames = []
            bg_prev = None
            for step_t in range(t):
                v_t = video[:, step_t]
                b_t = blur_v[:, step_t]
                if step_t > 0 and bg_prev is not None:
                    diff_mag = (v_t - video[:, step_t - 1]).abs().mean(dim=1, keepdim=True)
                    static_w = 0.85 * torch.exp(-15.0 * diff_mag)
                    b_t = static_w * bg_prev + (1.0 - static_w) * b_t
                bg_prev = b_t
                e_t = edge_gate[:, step_t]
                ema_frames.append(e_t * v_t + (1.0 - e_t) * b_t)
            v_det_ema = torch.stack(ema_frames, dim=1)
            det_ema_out = codec.evaluate_real_x265(v_det_ema, qp, None)

            # 3. Spatial-Only Frame-by-Frame Learned SCOPE (Noise-free, disable_temporal=True, F02/F03/R08 fix)
            pred_spatial = spatial_model(
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
            ignore_boxes = (
                sample["ignore_boxes"].unsqueeze(0).to(device)
                if "ignore_boxes" in sample
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
                    video, recon, m_det, gt_boxes=gt_boxes, ignore_boxes=ignore_boxes,
                    require_real_detector=True, return_raw_detections=True,
                )

                shard_records.append(
                    {
                        "shard_id": args.shard_id,
                        "variant": args.variant,
                        "dataset": ds_tag,
                        "clip_id": clip_id,
                        "qp": qp,
                        "crf": qp,
                        "frame_ids": sample["frame_ids"],
                        "packet_sizes": c_out["packet_sizes"],
                        "frame_types": c_out["frame_types"],
                        "height": h,
                        "width": w,
                        "raw_detections": {key: [value.tolist() for value in acc_metrics["raw_detections"][key]] for key in ("gt_boxes", "ignore_boxes", "pred_boxes", "pred_scores")},
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
    json_path = out_dir / f"eval_shard_{args.shard_id:02d}of{args.num_shards:02d}_{args.variant}.json"
    for record in shard_records:
        for key, value in record.items():
            if isinstance(value, float) and not np.isfinite(value):
                record[key] = None
    with open(json_path, "x", encoding="utf-8") as f:
        json.dump({"schema_version": 2, "metadata": metadata, "records": shard_records}, f, indent=2, allow_nan=False)
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
    parser.add_argument("--allow_untrained", action="store_true", help="Allow eval_poc without a checkpoint for smoke tests")
    parser.add_argument("--data_shard_id", type=int, default=-1, help="Train data slice index within this variant (default: shard_id)")
    parser.add_argument("--data_num_shards", type=int, default=-1, help="Number of train data slices for this variant (default: num_shards)")
    parser.add_argument("--patience", type=int, default=0, help="Early-stop patience in epochs on val loss (0 disables validation)")
    parser.add_argument("--val_every", type=int, default=2)
    parser.add_argument("--val_subsample", type=int, default=2, help="Use every k-th val clip")
    parser.add_argument("--min_epochs", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=0.005, help="Minimum val-loss decrease counted as improvement")
    args = parser.parse_args()

    if args.mode == "train":
        run_train_shard(args)
    else:
        run_poc_wishlist2_and_eval_shard(args)


if __name__ == "__main__":
    main()
