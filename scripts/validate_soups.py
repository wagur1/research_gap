"""Measure prepared soups on the training validation protocol, never held-out test AP."""
import argparse
import copy
import datetime
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from losses.vcm_temporal_loss import VCMTemporalPreEditLoss
from models.frozen_detector_fpn import FrozenFPNDetectorTeacher
from models.temporal_scope import TemporalSCOPE
from scripts.experiment_protocol import VARIANTS, VALIDATION_PROTOCOL, sha256_file, validate_soup_provenance, validate_release_validation
from scripts.kaggle_shard_runner import ABLATION_PRESETS
from vcm_datasets.vcm_video_datasets import RealVCMVideoDataset
from video_codecs.hybrid_x265_ste import HybridX265STE


@torch.no_grad()
def extract_validation_features(teacher, images, chunk_size=1):
    if type(chunk_size) is not int or chunk_size < 1 or len(images) == 0:
        raise ValueError("Validation feature chunks require a positive size and nonempty frames")
    chunks = [teacher.extract_fpn_features(chunk) for chunk in images.split(chunk_size)]
    keys = tuple(chunks[0])
    if any(tuple(chunk) != keys for chunk in chunks):
        raise ValueError("Validation feature levels differ between chunks")
    return {key: torch.cat([chunk[key] for chunk in chunks], dim=0) for key in keys}


@torch.no_grad()
def validate(args):
    dataset = RealVCMVideoDataset(args.data_root, shard_id=0, num_shards=1, clip_len=4, stride=4, img_size=256, is_train=False, enforce_disjoint_split=True, split="val")
    if len(dataset) != VALIDATION_PROTOCOL["full_clips"] or any(meta["type"] != "mot_frames" or meta["seq_name"].split("-FRCNN")[0] != "MOT17-09" for meta in dataset.clips):
        raise ValueError("Release validation requires all 131 complete MOT17-09 clips (524 frames), not the four-frame local visualization subset")
    frame_paths = [path for meta in dataset.clips for path in meta["frame_paths"]]
    frame_numbers = [int(path.stem) for path in frame_paths]
    if frame_numbers != list(range(1, 525)):
        raise ValueError("Validation frames must match the complete MOT17-09 evaluation manifest")
    gt_path = dataset.clips[0]["frame_paths"][0].parent.parent / "gt" / "gt.txt"
    fingerprints = {"gt_sha256": sha256_file(gt_path), "frames": [{"frame": int(path.stem), "sha256": sha256_file(path)} for path in frame_paths]}
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(4)
    teacher = FrozenFPNDetectorTeacher(pretrained=True).to(device)
    if not teacher.use_torchvision or teacher.detector is None:
        raise RuntimeError("Validation requires original pretrained Faster R-CNN v2 features")
    teacher.eval()
    selected_indices = list(range(0, len(dataset), VALIDATION_PROTOCOL["val_subsample"]))
    subset = Subset(dataset, selected_indices)
    loader = DataLoader(subset, batch_size=VALIDATION_PROTOCOL["batch_size"], shuffle=False)
    chunk_size = getattr(args, "feature_chunk_size", 1)
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("Validation feature chunk size must be positive")
    import torchvision
    provenance = {
        "measured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": str(device), "dtype": "float32", "feature_chunk_size": chunk_size,
        "feature_preprocessing": "normalized_256px_images_direct_backbone_no_detector_transform",
        "dependencies": {"python": platform.python_version(), "torch": str(torch.__version__), "torchvision": str(torchvision.__version__)},
        "source_sha256": {relative: sha256_file(ROOT_DIR / relative) for relative in ("scripts/validate_soups.py", "scripts/experiment_protocol.py", "models/frozen_detector_fpn.py", "models/temporal_scope.py", "losses/vcm_temporal_loss.py", "vcm_datasets/vcm_video_datasets.py", "video_codecs/hybrid_x265_ste.py")},
        "training_protocol_evidence": "Original shard-01 worker has batch_size=8 and no val_subsample override; pre-repair HEAD b30baa4144e4f4b797062e9b35542a7046996519 defaults to val_subsample=2. Original remote training revision was unpinned and is not asserted.",
    }
    weight_path = Path(torch.hub.get_dir()) / "checkpoints" / "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
    provenance["detector"] = {"name": "fasterrcnn_resnet50_fpn_v2", "weights": "FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT", "cached_weights_sha256": sha256_file(weight_path) if weight_path.is_file() else None}
    prepared = {}
    for variant in args.variants:
        source = Path(args.soup_dir) / f"tscope_soup_{variant}.pt"
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
        validate_soup_provenance(checkpoint, variant)
        preset = ABLATION_PRESETS[variant]
        model = TemporalSCOPE(base_ch=48, num_families=4, cond_dim=64, qg_size=16, max_delta_qp=preset["max_delta_qp"]).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        codec = HybridX265STE(use_real_x265_in_forward=preset["use_real_x265_ste"]).to(device)
        if preset["use_real_x265_ste"] and not codec.use_real_x265:
            raise RuntimeError("Real-codec training variants cannot validate through a proxy fallback")
        criterion = VCMTemporalPreEditLoss(lambda_temp_bg=preset["lambda_temp_bg"], lambda_temp_obj=preset["lambda_temp_obj"]).to(device)
        execution = copy.deepcopy(provenance)
        execution["codec_preset"] = copy.deepcopy(preset)
        if preset["use_real_x265_ste"]:
            execution["ffmpeg_version"] = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
        total = count = 0
        batch_sizes = []
        for batch_index, batch in enumerate(loader, start=1):
            video = batch["video"].to(device)
            attn, mask = batch["m_attn"].to(device), batch["m_det"].to(device)
            b, t, _, h, w = video.shape
            batch_sizes.append(b)
            teacher_features = extract_validation_features(teacher, video.reshape(b * t, 3, h, w), chunk_size)
            for severity_value in VALIDATION_PROTOCOL["severities"]:
                severity = torch.full((b,), severity_value, device=device)
                output = model(video, torch.zeros(b, dtype=torch.long, device=device), severity, reuse_iframe_qp=preset["reuse_iframe_qp"], disable_temporal=variant == "spatial_only_scope")
                decoded = codec(output["i_edit"], severity, output["ctu_qp_offsets"])
                features = extract_validation_features(teacher, decoded["video_recon"].reshape(b * t, 3, h, w), chunk_size)
                loss = criterion(i_edit=output["i_edit"], i_recon=decoded["video_recon"], attn_map=output["attn_map"], delta_obj=output["delta_obj"], delta_bg=output["delta_bg"], rate_proxy_bpp=decoded["rate_proxy_bpp"], m_attn=attn, m_det=mask, teacher_fpn_feats=teacher_features, student_fpn_feats=features)
                value = float(loss["loss_total"].item())
                if not math.isfinite(value):
                    raise ValueError("Soup validation loss is nonfinite")
                total += value
                count += 1
                del features, loss, decoded, output
            del teacher_features
            print(f"{variant}: validation batch {batch_index}/{len(loader)} complete", flush=True)
        value = total / count
        if not math.isfinite(value):
            raise ValueError("Soup validation loss is nonfinite")
        selected_frames = [int(path.stem) for index in selected_indices for path in dataset.clips[index]["frame_paths"]]
        execution.update(selected_clip_indices=selected_indices, measured_frame_ids=[f"MOT17-09/{number:06d}" for number in selected_frames], logical_batch_sizes=batch_sizes, batch_severity_loss_count=count)
        checkpoint["validation"] = {"status": "validated", "split": "MOT17-09", "release_ready": True, "loss_total": value, "data_fingerprints": fingerprints, "source_soup_sha256": sha256_file(source), "protocol": copy.deepcopy(VALIDATION_PROTOCOL), "execution_provenance": execution, "constituent_best_losses": [member["best_val_loss"] for member in checkpoint["constituents"]], "selection": "uniform_soup_no_test_selection"}
        validate_release_validation(checkpoint["validation"])
        prepared[variant] = checkpoint
        print(f"{variant}: validation loss={value:.6f}; no test AP used", flush=True)
        del model, codec, criterion
    destination = Path(args.output_dir)
    paths = {variant: destination / f"tscope_soup_{variant}.pt" for variant in prepared}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Validated soup output exists; use a new directory")
    destination.mkdir(parents=True, exist_ok=True)
    for variant, path in paths.items():
        with path.open("xb") as handle:
            torch.save(prepared[variant], handle)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--soup_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feature_chunk_size", type=int, default=1)
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
