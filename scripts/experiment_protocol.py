"""Versioned contracts for the preserved T-SCOPE experiment."""
import hashlib
import json
import math
from pathlib import Path

import torch

SCHEMA_VERSION = 2
CRFS = [22, 27, 32, 37]
VARIANTS = ["full_tscope", "spatial_only_scope", "wo_ctu_delta_qp", "iwa_gop_qp_reuse", "proxy_only_zhao"]
METHODS = ["anchor_x265", "deterministic_blur_ema", "spatial_scope_framewise", "tscope_temporal_ctu_qp", "oracle_gt_mask_upper_bound"]
TRAIN_SEQUENCES = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-10"]
VAL_SEQUENCES = ["MOT17-09"]
EXPECTED_TEST_FRAME_IDS = [f"MOT17-11/{number:06d}" for number in range(1, 901)] + [f"MOT17-13/{number:06d}" for number in range(1, 749)]
EXCLUDED_TEST_TAIL_FRAME_IDS = ["MOT17-13/000749", "MOT17-13/000750"]
PREPROCESSING = {"resize": "bilinear_square", "img_size": 512, "clip_len": 4, "stride": 4, "tail_policy": "complete_nonoverlapping_clips_only"}
VALIDATION_PROTOCOL = {"full_clips": 131, "val_subsample": 2, "batch_size": 8, "severities": [0.3, 0.7], "img_size": 512, "aggregation": "mean_batch_severity_loss_matching_training", "held_out_test_used": False}
CODEC_CONFIGURATION = {
    "encoder": "libx265", "preset": "medium", "rate_control": "crf", "fps": 25,
    "aq_mode": 2, "bframes": 0, "qg_size": 16, "actual_roi_grid_max": 16,
    "positive_qoffset_cap": 0.14, "negative_qoffset_cap": -0.18,
    "bitrate_accounting": "mp4_video_packet_payload_excluding_container_and_global_headers",
}
METRIC_POLICY = {
    "name": "dataset_level_101_point_pedestrian_ap", "class_id": 1,
    "score_threshold": 0.01, "iou_thresholds": [round(0.50 + 0.05 * i, 2) for i in range(10)],
    "ignore_policy": "unmatched_prediction_suppressed_if_ignore_box_iou_at_evaluation_threshold",
    "full_official_coco_equivalence": False,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest_files(manifest_path, suffix):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("expected_shards") != 21 or not isinstance(manifest.get("shards"), list):
        raise ValueError("Expected a 21-shard collection manifest")
    files = []
    seen_shards = set()
    for entry in manifest["shards"]:
        sid = entry.get("shard_id")
        if type(sid) is not int or not 0 <= sid < 21 or sid in seen_shards:
            raise ValueError("Invalid or duplicate manifested shard")
        seen_shards.add(sid)
        if not entry.get("downloaded"):
            continue
        if str(entry.get("status", "")).upper().split(".")[-1] != "COMPLETE":
            raise ValueError(f"Shard {sid} is not complete")
        directory = Path(entry["directory"]).resolve()
        if not directory.is_relative_to(path.parent):
            raise ValueError("Manifest directory escapes collection")
        seen_files = set()
        for item in entry.get("files", []):
            relative = Path(item["path"])
            candidate = directory / relative
            if relative.is_absolute() or ":" in str(relative) or ".." in relative.parts or candidate.is_symlink() or not candidate.resolve().is_relative_to(directory):
                raise ValueError("Unsafe manifested artifact path")
            if str(relative) in seen_files:
                raise ValueError("Duplicate manifested file")
            seen_files.add(str(relative))
            if candidate.suffix != suffix:
                continue
            if not candidate.is_file() or candidate.stat().st_size != item.get("bytes") or sha256_file(candidate) != item.get("sha256"):
                raise ValueError(f"Artifact hash or size mismatch: {candidate.name}")
            files.append((entry, candidate, item["sha256"]))
    return manifest, files


def verify_convergence_log(log, sid, variant):
    expected_finish = f"=== Shard {sid}/21 ({variant}) finished successfully! ==="
    expected_saved = "Saved shard checkpoint" in log and f"/kaggle/working/tscope_{variant}_shard{sid:02d}of21.pt" in log
    if "Converged: no val improvement for 10 epochs." not in log or (expected_finish not in log and not expected_saved):
        raise ValueError("Log does not corroborate exact shard convergence and completion")
    return {"startup_visible": f"Variant={variant}" in log, "completion_identity_verified": True}


def validate_checkpoint(checkpoint, sid, variant):
    from models.temporal_scope import TemporalSCOPE

    if checkpoint.get("shard_id") != sid or checkpoint.get("num_shards") != 21 or checkpoint.get("variant") != variant:
        raise ValueError("Checkpoint identity mismatch")
    if checkpoint.get("partial", False) or checkpoint.get("stopped_early") is not True:
        raise ValueError("Final soup requires a completed early-stopped checkpoint")
    if checkpoint.get("train_sequences") != TRAIN_SEQUENCES or checkpoint.get("val_sequences") != VAL_SEQUENCES:
        raise ValueError("Checkpoint split provenance mismatch")
    history = checkpoint.get("history", [])
    epochs, best_epoch = checkpoint.get("epochs_run", 0), checkpoint.get("best_epoch", 0)
    best_val = checkpoint.get("best_val_loss")
    if type(epochs) is not int or type(best_epoch) is not int or len(history) != epochs or epochs < 12 or not 1 <= best_epoch <= epochs or epochs - best_epoch < 10:
        raise ValueError("Invalid convergence history or patience")
    if not isinstance(best_val, (float, int)) or not math.isfinite(best_val):
        raise ValueError("Missing finite best validation loss")
    if [row.get("epoch") for row in history] != list(range(1, epochs + 1)):
        raise ValueError("Non-contiguous checkpoint history")
    if not math.isclose(history[best_epoch - 1].get("val_loss", float("nan")), best_val, abs_tol=1e-6):
        raise ValueError("Best validation loss differs from history")
    if any(not math.isfinite(row.get("loss_total", float("nan"))) or ("val_loss" in row and not math.isfinite(row["val_loss"])) for row in history):
        raise ValueError("Nonfinite checkpoint history")
    scale = 4.0 if variant in ("full_tscope", "iwa_gop_qp_reuse") else 0.0
    model = TemporalSCOPE(base_ch=48, num_families=4, cond_dim=64, qg_size=16, max_delta_qp=scale)
    state = checkpoint.get("state_dict", {})
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("Checkpoint parameter keys mismatch")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape or value.dtype != expected[key].dtype or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Invalid checkpoint tensor: {key}")
    model.load_state_dict(state, strict=True)
    return model


def validate_soup_provenance(soup, variant, require_validation=False):
    if soup.get("schema_version") != SCHEMA_VERSION or soup.get("variant") != variant or variant not in VARIANTS:
        raise ValueError("Invalid same-variant soup schema")
    members = soup.get("constituents")
    expected_full = {sid for sid in range(21) if VARIANTS[sid % 5] == variant}
    if not isinstance(members, list) or len(members) < 1 or soup.get("merged_shards") != len(members):
        raise ValueError("Soup constituent membership is incomplete")
    ids = [member.get("shard_id") for member in members if isinstance(member, dict)]
    if len(ids) != len(members) or len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < 21 for i in ids):
        raise ValueError("Soup constituent shard identities differ from expected variant")
    if soup.get("merged_shards") == len(expected_full) and set(ids) != expected_full:
        raise ValueError("Soup constituent shard identities differ from expected variant")
    for member in members:
        for key in ("checkpoint_sha256", "log_sha256"):
            value = member.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Missing constituent hash provenance")
    if soup.get("strict_load_passed") is not True or soup.get("inference_smoke_passed") is not True:
        raise ValueError("Soup lacks strict load and inference smoke evidence")
    configuration = soup.get("configuration", {})
    if configuration.get("codec") != CODEC_CONFIGURATION or configuration.get("initialization_seed") != 2026:
        raise ValueError("Soup configuration differs from preserved training")
    expected_scale = 4.0 if variant in ("full_tscope", "iwa_gop_qp_reuse") else 0.0
    if configuration.get("max_delta_qp") != expected_scale:
        raise ValueError("Soup QG scale differs from preserved variant")
    if require_validation:
        validate_release_validation(soup.get("validation", {}))
    return soup


def validate_release_validation(validation):
    if validation.get("status") != "validated" or validation.get("release_ready") is not True or validation.get("split") != "MOT17-09":
        raise ValueError("Soup must pass MOT17-09 validation before final evaluation")
    loss = validation.get("loss_total")
    if type(loss) not in (int, float) or not math.isfinite(loss):
        raise ValueError("Missing finite measured soup validation loss")
    if validation.get("protocol") != VALIDATION_PROTOCOL:
        raise ValueError("Soup validation protocol differs from preserved training")
    fingerprints = validation.get("data_fingerprints", {})
    frames = fingerprints.get("frames", [])
    if not isinstance(frames, list) or [row.get("frame") for row in frames if isinstance(row, dict)] != list(range(1, 525)):
        raise ValueError("Soup validation requires the complete MOT17-09 frame fingerprints")
    hashes = [fingerprints.get("gt_sha256"), validation.get("source_soup_sha256")] + [row.get("sha256") for row in frames]
    if any(not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value) for value in hashes):
        raise ValueError("Invalid validation data or source soup hashes")
    if validation.get("selection") != "uniform_soup_no_test_selection":
        raise ValueError("Validation must not use held-out test selection")
    return validation


def restore_raw_detections(raw, count):
    result = {}
    required_keys = ("gt_boxes", "ignore_boxes", "pred_boxes", "pred_scores")
    optional_keys = ("yolo_pred_boxes", "yolo_pred_scores") if "yolo_pred_boxes" in raw or "yolo_pred_scores" in raw else ()
    for key in required_keys + optional_keys:
        values = raw.get(key)
        if not isinstance(values, list) or len(values) != count:
            raise ValueError(f"Invalid raw detection frame count: {key}")
        tensors = []
        for value in values:
            tensor = torch.tensor(value, dtype=torch.float32)
            if key in ("pred_scores", "yolo_pred_scores"):
                if tensor.ndim != 1 or not bool(torch.isfinite(tensor).all()) or bool(((tensor < 0) | (tensor > 1)).any()):
                    raise ValueError("Invalid prediction scores")
            else:
                if tensor.numel() == 0 and isinstance(value, list) and not value:
                    tensor = tensor.reshape(0, 4)
                if tensor.ndim != 2 or tensor.shape[1] != 4 or not bool(torch.isfinite(tensor).all()):
                    raise ValueError("Invalid box coordinates")
                if bool(((tensor[:, 2:] - tensor[:, :2]) <= 0).any()) or bool((tensor < 0).any()):
                    raise ValueError("Degenerate or negative box coordinates")
            tensors.append(tensor)
        result[key] = tensors
    if any(boxes.shape[0] != scores.shape[0] for boxes, scores in zip(result["pred_boxes"], result["pred_scores"])):
        raise ValueError("Prediction boxes and scores differ in count")
    if "yolo_pred_boxes" in result and any(boxes.shape[0] != scores.shape[0] for boxes, scores in zip(result["yolo_pred_boxes"], result["yolo_pred_scores"])):
        raise ValueError("YOLO prediction boxes and scores differ in count")
    return result
