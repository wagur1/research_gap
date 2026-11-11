import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch

from scripts.experiment_protocol import (
    CODEC_CONFIGURATION, CRFS, METHODS, VARIANTS, EXPECTED_TEST_FRAME_IDS,
    EXCLUDED_TEST_TAIL_FRAME_IDS, METRIC_POLICY, PREPROCESSING, read_manifest_files,
    restore_raw_detections, sha256_file, validate_checkpoint, validate_release_validation, verify_convergence_log,
)


def bjontegaard_delta_rate(
    rate_anchor: List[float],
    acc_anchor: List[float],
    rate_test: List[float],
    acc_test: List[float],
) -> float:
    """
    Standard VCEG-M33 / MPEG-VCM Bjontegaard Delta Bit-Rate (BDBR %) calculation
    using cubic polynomial fitting in log-rate domain over the common accuracy interval.
    """
    arrays = [np.asarray(value, dtype=np.float64) for value in (rate_anchor, acc_anchor, rate_test, acc_test)]
    if any(value.ndim != 1 or len(value) < 4 or not np.isfinite(value).all() for value in arrays):
        raise ValueError("BDBR requires at least four finite operating points")
    if len(arrays[0]) != len(arrays[1]) or len(arrays[2]) != len(arrays[3]):
        raise ValueError("BDBR rate and accuracy lengths differ")
    if any(np.any(value <= 0) for value in (arrays[0], arrays[2])):
        raise ValueError("BDBR requires strictly positive rates")
    if len(np.unique(arrays[1])) < 4 or len(np.unique(arrays[3])) < 4:
        raise ValueError("BDBR requires four distinct accuracy values per curve")
    r1 = np.log10(arrays[0])
    a1 = arrays[1]
    r2 = np.log10(arrays[2])
    a2 = arrays[3]

    order1 = np.argsort(a1)
    a1, r1 = a1[order1], r1[order1]
    order2 = np.argsort(a2)
    a2, r2 = a2[order2], r2[order2]

    min_int = max(np.min(a1), np.min(a2))
    max_int = min(np.max(a1), np.max(a2))
    if max_int <= min_int or np.isnan(min_int) or np.isnan(max_int):
        return float("nan")

    p1 = np.polyfit(a1, r1, deg=min(3, len(a1) - 1))
    p2 = np.polyfit(a2, r2, deg=min(3, len(a2) - 1))

    int1 = np.polyint(p1)
    int2 = np.polyint(p2)

    avg_diff = (np.polyval(int2, max_int) - np.polyval(int2, min_int)) / (
        max_int - min_int
    ) - (np.polyval(int1, max_int) - np.polyval(int1, min_int)) / (max_int - min_int)
    return float((np.power(10.0, avg_diff) - 1.0) * 100.0)


def average_compatible_states(states):
    if not states:
        raise ValueError("Cannot average empty checkpoint collection")
    reference = states[0]
    for state in states:
        if set(state) != set(reference):
            raise ValueError("Soup state keys differ")
        for key, value in state.items():
            first = reference[key]
            if not isinstance(value, torch.Tensor) or value.shape != first.shape or value.dtype != first.dtype or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Incompatible soup tensor: {key}")
            if not value.is_floating_point() and not torch.equal(value, first):
                raise ValueError(f"Non-floating soup buffer differs: {key}")
    result = {}
    for key, first in reference.items():
        if first.is_floating_point():
            result[key] = torch.stack([state[key].to(torch.float64) for state in states]).mean(0).to(first.dtype)
        else:
            result[key] = first.clone()
    return result


def merge_checkpoints_soup(ckpt_dir: str, output_ckpt: str, manifest_path=None, variants=None):
    """
    Aggregates trained shard checkpoints from multiple Kaggle accounts via
    Model Soup / Federated Averaging, strictly grouped by `variant` (F18 fix:
    never averages across different ablation variants).
    Final artifacts require a checked manifest; directory globbing is not provenance.
    """
    if manifest_path is None:
        raise ValueError("Soup preparation requires --manifest; implicit directory merging is disabled")
    manifest, files = read_manifest_files(manifest_path, ".pt")
    selected = list(VARIANTS if variants is None else variants)
    if not selected or len(set(selected)) != len(selected) or any(v not in VARIANTS for v in selected):
        raise ValueError("Invalid soup variant selection")
    by_variant = {variant: {} for variant in selected}
    for entry, path, digest in files:
        sid = entry["shard_id"]
        variant = VARIANTS[sid % len(VARIANTS)]
        if variant not in by_variant:
            continue
        expected_filename = f"tscope_{variant}_shard{sid:02d}of21.pt"
        if path.name != expected_filename or entry.get("expected_variant") != variant or sid in by_variant[variant]:
            raise ValueError("Unexpected or duplicate shard checkpoint in manifest")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        validate_checkpoint(checkpoint, sid, variant)
        log_path = path.parent / f"t-scope-vcm-shard-{sid:02d}.log"
        log_entry = next((item for item in entry.get("files", []) if item["path"] == log_path.name), None)
        if not log_entry or sha256_file(log_path) != log_entry.get("sha256"):
            raise ValueError("Missing checked convergence log")
        log = log_path.read_text(encoding="utf-8")
        verify_convergence_log(log, sid, variant)
        by_variant[variant][sid] = (checkpoint, path, digest, log_entry["sha256"])
    outputs = {}
    for variant, collection in by_variant.items():
        expected = {sid for sid in range(21) if VARIANTS[sid % 5] == variant}
        if set(collection) != expected:
            raise ValueError(f"Incomplete {variant}: missing shards {sorted(expected - set(collection))}")
        rows = [collection[sid] for sid in sorted(collection)]
        state = average_compatible_states([row[0]["state_dict"] for row in rows])
        model = validate_checkpoint({**rows[0][0], "state_dict": state}, rows[0][0]["shard_id"], variant)
        model.eval()
        with torch.no_grad():
            prediction = model(torch.zeros(1, 2, 3, 32, 32), torch.zeros(1, dtype=torch.long), torch.tensor([0.3]), reuse_iframe_qp=variant == "iwa_gop_qp_reuse", disable_temporal=variant == "spatial_only_scope")
        if not all(bool(torch.isfinite(value).all()) for value in prediction.values() if isinstance(value, torch.Tensor)):
            raise ValueError("Soup inference returned nonfinite outputs")
        constituents = [{"shard_id": row[0]["shard_id"], "checkpoint_sha256": row[2], "log_sha256": row[3], "best_epoch": row[0]["best_epoch"], "best_val_loss": row[0]["best_val_loss"], "epochs_run": row[0]["epochs_run"], "source_path": str(row[1])} for row in rows]
        outputs[variant] = {
            "schema_version": 2, "variant": variant, "merged_shards": len(rows), "state_dict": state,
            "constituents": constituents, "strict_load_passed": True, "inference_smoke_passed": True,
            "configuration": {"codec": CODEC_CONFIGURATION, "initialization_seed": 2026, "max_delta_qp": 4.0 if variant in ("full_tscope", "iwa_gop_qp_reuse") else 0.0},
            "validation": {"status": "not_evaluated", "split": "MOT17-09", "release_ready": False},
            "source_manifest_sha256": sha256_file(manifest_path),
            "provenance_limitations": ["Original convergence checkpoints lack a pinned training revision and full configuration; retain original logs and source snapshot."],
        }
    destination = Path(ckpt_dir)
    paths = {variant: destination / f"tscope_soup_{variant}.pt" for variant in selected}
    alias = Path(output_ckpt) if output_ckpt and "full_tscope" in selected else None
    if any(path.exists() for path in paths.values()) or (alias is not None and alias.exists()):
        raise FileExistsError("Soup output already exists; use a new output directory")
    destination.mkdir(parents=True, exist_ok=True)
    for variant, path in paths.items():
        with path.open("xb") as handle:
            torch.save(outputs[variant], handle)
        print(f"Prepared {variant}: {outputs[variant]['merged_shards']} checked shards -> {path}; validation pending")
    if alias is not None and alias not in paths.values():
        alias.parent.mkdir(parents=True, exist_ok=True)
        with alias.open("xb") as handle:
            torch.save(outputs["full_tscope"], handle)
    return {variant: str(path) for variant, path in paths.items()}


def summarize_records(records):
    from models.frozen_detector_fpn import FrozenFPNDetectorTeacher

    raw = {key: [] for key in ("gt_boxes", "ignore_boxes", "pred_boxes", "pred_scores")}
    pixels = i_pixels = p_pixels = 0
    bits = i_bits = p_bits = 0
    roi_sum = roi_frames = 0
    for record in records:
        count = len(record["frame_ids"])
        restored = restore_raw_detections(record["raw_detections"], count)
        for key in raw:
            raw[key].extend(restored[key])
        area = record["height"] * record["width"]
        for size, frame_type in zip(record["packet_sizes"], record["frame_types"]):
            bits += size * 8
            pixels += area
            if frame_type == "I":
                i_bits += size * 8
                i_pixels += area
            else:
                p_bits += size * 8
                p_pixels += area
        roi = record.get("task_roi_psnr_db")
        if roi is not None and np.isfinite(roi):
            roi_sum += roi * count
            roi_frames += count
    metrics = FrozenFPNDetectorTeacher.compute_dataset_level_coco_metrics(raw["gt_boxes"], raw["pred_boxes"], raw["pred_scores"], raw["ignore_boxes"])
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("Global AP unavailable: no active held-out GT")
    return {"total_bpp": bits / pixels, "i_frame_bpp": i_bits / i_pixels if i_pixels else None,
            "p_frame_bpp": p_bits / p_pixels if p_pixels else None,
            "roi_psnr_db": roi_sum / roi_frames if roi_frames else None,
            "roi_psnr_aggregation": "frame_weighted_mean_clip_roi_psnr_diagnostic_only",
            "packet_bytes": bits // 8, "pixel_count": pixels, "frame_count": len(raw["gt_boxes"]), **metrics}


def build_bdbr_summary(per_qp, anchor):
    result = {"per_qp": per_qp, "bdbr_unavailable_reasons": {}}
    for rate_key, prefix in (("total_bpp", "tot"), ("p_frame_bpp", "p")):
        for accuracy_key, suffix in (("gt_ap50_pct", "gt_ap50_pct"), ("gt_ap50_95_pct", "gt_ap50_95_pct")):
            field = f"bdbr_{prefix}_{suffix}"
            try:
                value = bjontegaard_delta_rate(
                    [anchor[str(crf)][rate_key] for crf in CRFS],
                    [anchor[str(crf)][accuracy_key] for crf in CRFS],
                    [per_qp[str(crf)][rate_key] for crf in CRFS],
                    [per_qp[str(crf)][accuracy_key] for crf in CRFS],
                )
                if not np.isfinite(value):
                    raise ValueError("No positive common accuracy interval")
                result[field] = value
            except (ValueError, TypeError, np.linalg.LinAlgError) as error:
                result[field] = None
                result["bdbr_unavailable_reasons"][field] = str(error)
    return result


def merge_eval_jsons_and_print_table(eval_dir: str, manifest_path=None, output_path=None):
    """
    Aggregates `eval_shard_*.json` across all Kaggle accounts, computes global
    I-frame bpp, P-frame bpp, total bpp, and BDBR (%) relative to `anchor_x265`.
    Global AP uses raw decoded-frame detections, never averaged clip scores.
    """
    if manifest_path is None:
        raise ValueError("Final evaluation requires --manifest; legacy clip-AP summaries are not final benchmarks")
    _, files = read_manifest_files(manifest_path, ".json")
    bundles = {}
    common = None
    hashes = {}
    expected_ids = None
    grouped = {}
    spatial_hash = None
    ground_truth = {}
    for entry, path, digest in files:
        if not path.name.startswith("eval_shard_"):
            continue
        bundle = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(bundle, dict) or bundle.get("schema_version") != 2:
            raise ValueError("Legacy or malformed evaluation schema")
        metadata = bundle.get("metadata", {})
        sid, variant = metadata.get("shard_id"), metadata.get("variant")
        if sid != entry["shard_id"] or metadata.get("num_shards") != 21 or variant not in VARIANTS or (sid, variant) in bundles:
            raise ValueError("Invalid or duplicate evaluation shard identity")
        if path.name != f"eval_shard_{sid:02d}of21_{variant}.json":
            raise ValueError("Evaluation filename differs from identity")
        if metadata.get("crfs") != CRFS or metadata.get("dataset") != "MOT17" or metadata.get("split") != "test":
            raise ValueError("Evaluation operating points or held-out protocol mismatch")
        cohort = {key: metadata.get(key) for key in ("detector", "codec", "metric_policy", "preprocessing")}
        if any(value is None for value in cohort.values()):
            raise ValueError("Evaluation provenance is incomplete")
        detector = cohort["detector"]
        if detector.get("name") != "fasterrcnn_resnet50_fpn_v2" or detector.get("pretrained") is not True:
            raise ValueError("Final AP requires identified pretrained Faster R-CNN v2")
        if common is None:
            common = cohort
        elif common != cohort:
            raise ValueError("Evaluation cohorts have different detector/codec/metric/preprocessing settings")
        if any(cohort["codec"].get(key) != value for key, value in CODEC_CONFIGURATION.items()):
            raise ValueError("Evaluation changed preserved codec configuration")
        if cohort["metric_policy"] != METRIC_POLICY or cohort["preprocessing"] != PREPROCESSING:
            raise ValueError("Evaluation metric/ignore/preprocessing policy differs from declared protocol")
        validation = metadata.get("soup_validation", {})
        validate_release_validation(validation)
        ids = metadata.get("expected_frame_ids")
        if ids != EXPECTED_TEST_FRAME_IDS or metadata.get("excluded_tail_frame_ids") != EXCLUDED_TEST_TAIL_FRAME_IDS:
            raise ValueError("Final benchmark requires the complete 1,648-frame held-out protocol and explicit tail exclusions")
        if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)) or any(not isinstance(value, str) or not (value.startswith("MOT17-11/") or value.startswith("MOT17-13/")) for value in ids):
            raise ValueError("Invalid held-out physical frame manifest")
        if expected_ids is None:
            expected_ids = ids
        elif expected_ids != ids:
            raise ValueError("Expected physical frames differ across evaluation cohorts")
        model_hash = metadata.get("checkpoint_sha256")
        spatial = metadata.get("spatial_checkpoint_sha256")
        if any(not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value) for value in (model_hash, spatial)):
            raise ValueError("Missing checkpoint hashes")
        if hashes.setdefault(variant, model_hash) != model_hash:
            raise ValueError("A variant evaluated different checkpoint weights across shards")
        if spatial_hash is None:
            spatial_hash = spatial
        elif spatial_hash != spatial:
            raise ValueError("Spatial baseline checkpoint differs across cohorts")
        bundles[sid, variant] = {"sha256": digest, "path": str(path)}
        records = bundle.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError("Empty evaluation shard")
        for record in records:
            method, crf = record.get("method"), record.get("crf")
            if record.get("shard_id") != sid or record.get("variant") != variant or method not in METHODS or crf not in CRFS or record.get("qp", crf) != crf:
                raise ValueError("Record identity or operating point mismatch")
            frames = record.get("frame_ids")
            if not isinstance(frames, list) or not frames or len(frames) != len(set(frames)) or not set(frames).issubset(ids):
                raise ValueError("Record physical frames invalid")
            sizes, types = record.get("packet_sizes"), record.get("frame_types")
            h, w = record.get("height"), record.get("width")
            if type(h) is not int or type(w) is not int or h <= 0 or w <= 0 or not isinstance(sizes, list) or not isinstance(types, list) or len(sizes) != len(frames) or len(types) != len(frames):
                raise ValueError("Packet sizes or pixel denominators invalid")
            if h != PREPROCESSING["img_size"] or w != PREPROCESSING["img_size"] or len(frames) != PREPROCESSING["clip_len"] or types != ["I"] + ["P"] * (len(frames) - 1):
                raise ValueError("Record pixels, clip length, or frame structure differs from preserved evaluation")
            if any(type(value) is not int or value <= 0 for value in sizes) or any(value not in ("I", "P", "B") for value in types):
                raise ValueError("Invalid packet byte size or frame type")
            restored = restore_raw_detections(record.get("raw_detections", {}), len(frames))
            group = grouped.setdefault((variant, method, crf), {"frames": set(), "records": [], "by_shard": {}})
            if group["frames"].intersection(frames):
                raise ValueError("Duplicate physical frame in one evaluation curve")
            group["frames"].update(frames)
            group["records"].append(record)
            group["by_shard"].setdefault(sid, set()).update(frames)
            for index, frame in enumerate(frames):
                gt = (restored["gt_boxes"][index].tolist(), restored["ignore_boxes"][index].tolist(), h, w)
                if ground_truth.setdefault(frame, gt) != gt:
                    raise ValueError("GT or spatial preprocessing differs for a paired physical frame")
    expected_pairs = {(sid, variant) for sid in range(21) for variant in VARIANTS}
    if set(bundles) != expected_pairs:
        raise ValueError(f"Incomplete evaluation: {len(bundles)}/105 shard-variant outputs")
    if hashes.get("spatial_only_scope") != spatial_hash:
        raise ValueError("Spatial reference hash differs from spatial variant soup")
    all_ids = set(expected_ids)
    summaries = {}
    reference_partition = None
    for variant in VARIANTS:
        methods = {}
        for method in METHODS:
            per_qp = {}
            for crf in CRFS:
                group = grouped.get((variant, method, crf))
                if group is None or group["frames"] != all_ids or set(group["by_shard"]) != set(range(21)):
                    raise ValueError(f"Incomplete paired frame coverage: {variant}/{method}/CRF{crf}")
                if reference_partition is None:
                    reference_partition = group["by_shard"]
                elif reference_partition != group["by_shard"]:
                    raise ValueError("Physical frame partition differs across methods, CRFs, or variants")
                records = sorted(group["records"], key=lambda row: tuple(row["frame_ids"]))
                per_qp[str(crf)] = summarize_records(records)
            methods[method] = per_qp
        summaries[variant] = {method: build_bdbr_summary(per_qp, methods["anchor_x265"]) for method, per_qp in methods.items()}
    result = {"schema_version": 2, "crfs": CRFS, "eligibility": {"final_ready": True, "complete_shard_variants": 105, "frame_count": len(expected_ids)},
              "configuration": common, "checkpoint_hashes": hashes, "expected_frame_ids": expected_ids,
              "overall_methods_summary": summaries["full_tscope"],
              "ablation_variants_summary": {variant: summaries[variant]["tscope_temporal_ctu_qp"] for variant in VARIANTS},
              "variant_methods_summary": summaries,
              "source_files": list(bundles.values()), "source_manifest_sha256": sha256_file(manifest_path)}
    destination = Path(output_path) if output_path else Path(eval_dir) / "merged_benchmarks.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print("\n=== PAIRED DATASET-LEVEL PEDESTRIAN AP / VIDEO-PACKET PAYLOAD BDBR ===")
    for variant, summary in result["ablation_variants_summary"].items():
        value = summary["bdbr_tot_gt_ap50_pct"]
        print(f"{variant:<24} | AP@50 BDBR: {value:+.2f}%" if value is not None else f"{variant:<24} | AP@50 BDBR unavailable: {summary['bdbr_unavailable_reasons']['bdbr_tot_gt_ap50_pct']}")
    print(f"Verified final benchmark -> {destination}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Kaggle Shard Checkpoints & Eval JSONs")
    parser.add_argument("--eval_dir", type=str, default="./outputs")
    parser.add_argument("--ckpt_dir", type=str, default="./outputs")
    parser.add_argument("--merged_ckpt", type=str, default="./outputs/tscope_merged_soup.pt")
    parser.add_argument("--manifest", required=True, help="Checked isolated collection manifest")
    parser.add_argument("--action", choices=["soup", "eval"], required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=None)
    args = parser.parse_args()

    if args.action == "eval":
        merge_eval_jsons_and_print_table(args.eval_dir, args.manifest)
    else:
        merge_checkpoints_soup(args.ckpt_dir, args.merged_ckpt, args.manifest, args.variants)


if __name__ == "__main__":
    main()
