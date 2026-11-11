import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from models.frozen_detector_fpn import FrozenFPNDetectorTeacher as Teacher
from scripts.experiment_protocol import CODEC_CONFIGURATION, CRFS, METRIC_POLICY, METHODS, VARIANTS, restore_raw_detections, sha256_file, validate_release_validation, verify_convergence_log
from scripts.merge_shards import average_compatible_states, bjontegaard_delta_rate, build_bdbr_summary, merge_checkpoints_soup, merge_eval_jsons_and_print_table, summarize_records


def boxes(values=()):
    return torch.tensor(values, dtype=torch.float32).reshape(-1, 4)


class APTests(unittest.TestCase):
    def test_global_ap_not_mean_clip_ap(self):
        gt = [boxes([[0, 0, 10, 10]]), boxes([[0, 0, 10, 10]])]
        predicted = [gt[0], boxes([[20, 20, 30, 30]])]
        scores = [torch.tensor([0.5]), torch.tensor([0.9])]
        global_ap = Teacher.compute_dataset_level_ap(gt, predicted, scores)
        mean = sum(Teacher.compute_dataset_level_ap([g], [p], [s]) for g, p, s in zip(gt, predicted, scores)) / 2
        self.assertAlmostEqual(mean, 0.5)
        self.assertAlmostEqual(global_ap, 25.5 / 101)
        self.assertNotAlmostEqual(global_ap, mean)

    def test_empty_gt_false_positive_penalizes(self):
        gt = [boxes(), boxes([[0, 0, 10, 10]])]
        predicted = [boxes([[0, 0, 10, 10]]), gt[1]]
        self.assertAlmostEqual(Teacher.compute_dataset_level_ap(gt, predicted, [torch.tensor([0.9]), torch.tensor([0.5])]), 0.5)

    def test_ignore_policy(self):
        gt = [boxes(), boxes([[0, 0, 10, 10]])]
        predicted = [boxes([[0, 0, 10, 10]]), gt[1]]
        self.assertEqual(Teacher.compute_dataset_level_ap(gt, predicted, [torch.tensor([0.9]), torch.tensor([0.5])], [predicted[0], boxes()]), 1.0)

    def test_duplicate_detection_is_false_positive(self):
        gt = [boxes([[0, 0, 10, 10]]), boxes([[0, 0, 10, 10]])]
        predicted = [boxes([[0, 0, 10, 10], [0, 0, 10, 10]]), gt[1]]
        self.assertLess(Teacher.compute_dataset_level_ap(gt, predicted, [torch.tensor([0.9, 0.8]), torch.tensor([0.7])]), 1.0)

    def test_misaligned_frames_rejected(self):
        with self.assertRaises(ValueError):
            Teacher.compute_dataset_level_ap([boxes()], [], [])

    def test_score_box_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            Teacher.compute_dataset_level_ap([boxes()], [boxes([[0, 0, 10, 10]])], [torch.tensor([])])

    def test_ignore_count_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            Teacher.compute_dataset_level_ap([boxes()], [boxes()], [torch.tensor([])], [])

    def test_empty_serialization_shape(self):
        raw = restore_raw_detections({key: [[]] for key in ("gt_boxes", "ignore_boxes", "pred_boxes", "pred_scores")}, 1)
        self.assertEqual(tuple(raw["gt_boxes"][0].shape), (0, 4))
        self.assertEqual(tuple(raw["pred_scores"][0].shape), (0,))

    def test_bad_serialization_rejected(self):
        with self.assertRaises(ValueError):
            restore_raw_detections({"gt_boxes": [[]]}, 1)


class SoupAndRateTests(unittest.TestCase):
    def test_validation_status_without_evidence_rejected(self):
        with self.assertRaises(ValueError):
            validate_release_validation({"status": "validated", "release_ready": True})

    def test_truncated_log_exact_identity(self):
        log = "Converged: no val improvement for 10 epochs. Stop at epoch 12.\n=== Shard 5/21 (full_tscope) finished successfully! ==="
        self.assertFalse(verify_convergence_log(log, 5, "full_tscope")["startup_visible"])
        with self.assertRaises(ValueError):
            verify_convergence_log(log, 10, "full_tscope")

    def test_float_average_and_integer_preservation(self):
        result = average_compatible_states([{"weight": torch.tensor([1.0]), "count": torch.tensor(4)}, {"weight": torch.tensor([3.0]), "count": torch.tensor(4)}])
        self.assertEqual(result["weight"].item(), 2)
        self.assertEqual(result["count"].dtype, torch.int64)

    def test_nonfloat_difference_rejected(self):
        with self.assertRaises(ValueError):
            average_compatible_states([{"count": torch.tensor(1)}, {"count": torch.tensor(2)}])

    def test_dtype_and_key_difference_rejected(self):
        with self.assertRaises(ValueError):
            average_compatible_states([{"a": torch.tensor([1.0])}, {"b": torch.tensor([1.0])}])
        with self.assertRaises(ValueError):
            average_compatible_states([{"a": torch.tensor([1.0])}, {"a": torch.tensor([1.0], dtype=torch.float64)}])

    def test_no_manifest_no_merge(self):
        with self.assertRaises(ValueError):
            merge_checkpoints_soup("unused", "unused.pt")
        with self.assertRaises(ValueError):
            merge_eval_jsons_and_print_table("unused")

    def test_bdbr_constant_savings(self):
        self.assertAlmostEqual(bjontegaard_delta_rate([1, 2, 4, 8], [1, 2, 3, 4], [.8, 1.6, 3.2, 6.4], [1, 2, 3, 4]), -20.0)

    def test_invalid_bdbr_rejected(self):
        for rate, accuracy in (([0, 2, 3, 4], [1, 2, 3, 4]), ([1, 2, 3, 4], [1, 1, 2, 3]), ([1, 2], [1, 2])):
            with self.assertRaises(ValueError):
                bjontegaard_delta_rate(rate, accuracy, rate, accuracy)

    def test_nonoverlap_is_unavailable(self):
        anchor = {str(crf): {"total_bpp": i + 1, "p_frame_bpp": i + 1, "gt_ap50_pct": i, "gt_ap50_95_pct": i} for i, crf in enumerate(CRFS)}
        test = copy.deepcopy(anchor)
        for row in test.values():
            row["gt_ap50_pct"] += 10
            row["gt_ap50_95_pct"] += 10
        result = build_bdbr_summary(test, anchor)
        self.assertIsNone(result["bdbr_tot_gt_ap50_pct"])
        self.assertIn("bdbr_tot_gt_ap50_pct", result["bdbr_unavailable_reasons"])

    def test_weighted_integer_packet_rates(self):
        def row(size, side):
            return {"frame_ids": ["x"], "packet_sizes": [size], "frame_types": ["I"], "height": side, "width": side, "raw_detections": {"gt_boxes": [[[0, 0, 1, 1]]], "ignore_boxes": [[]], "pred_boxes": [[[0, 0, 1, 1]]], "pred_scores": [[1.0]]}, "task_roi_psnr_db": 20.0}
        summary = summarize_records([row(10, 10), row(10, 20)])
        self.assertAlmostEqual(summary["total_bpp"], 160 / 500)
        self.assertEqual(summary["gt_ap50_pct"], 100.0)


class PairedMergeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ids = [f"MOT17-{11 if sid < 10 else 13:02d}/{sid+1:06d}" for sid in range(21)]
        patcher = patch("scripts.merge_shards.EXPECTED_TEST_FRAME_IDS", self.ids)
        patcher.start()
        self.addCleanup(patcher.stop)
        protocol_patcher = patch("scripts.merge_shards.PREPROCESSING", {"img_size": 32, "clip_len": 1})
        protocol_patcher.start()
        self.addCleanup(protocol_patcher.stop)
        self.validation = {"status": "validated", "release_ready": True, "split": "MOT17-09", "loss_total": 1.0,
                           "protocol": {"full_clips": 131, "val_subsample": 2, "batch_size": 8, "severities": [0.3, 0.7], "img_size": 256, "aggregation": "mean_batch_severity_loss_matching_training", "held_out_test_used": False},
                           "data_fingerprints": {"gt_sha256": "a" * 64, "frames": [{"frame": number, "sha256": "b" * 64} for number in range(1, 525)]}, "source_soup_sha256": "c" * 64, "selection": "uniform_soup_no_test_selection"}

    def tearDown(self):
        self.temp.cleanup()

    def write_collection(self, mutate=None):
        manifest = {"expected_shards": 21, "shards": []}
        for sid in range(21):
            directory = self.root / f"shard_{sid:02d}"
            directory.mkdir()
            entry = {"shard_id": sid, "status": "COMPLETE", "downloaded": True, "directory": str(directory), "files": []}
            for index, variant in enumerate(VARIANTS):
                metadata = {"shard_id": sid, "num_shards": 21, "variant": variant, "crfs": CRFS, "dataset": "MOT17", "split": "test", "expected_frame_ids": self.ids, "excluded_tail_frame_ids": ["MOT17-13/000749", "MOT17-13/000750"], "checkpoint_sha256": str(index + 1) * 64, "spatial_checkpoint_sha256": "2" * 64, "detector": {"name": "fasterrcnn_resnet50_fpn_v2", "pretrained": True}, "codec": CODEC_CONFIGURATION, "metric_policy": METRIC_POLICY, "preprocessing": {"img_size": 32, "clip_len": 1}, "soup_validation": copy.deepcopy(self.validation)}
                records = []
                for crf in CRFS:
                    for method in METHODS:
                        records.append({"shard_id": sid, "variant": variant, "method": method, "crf": crf, "qp": crf, "frame_ids": [self.ids[sid]], "packet_sizes": [10 + crf], "frame_types": ["I"], "height": 32, "width": 32, "task_roi_psnr_db": 20.0, "raw_detections": {"gt_boxes": [[[0, 0, 2, 2]]], "ignore_boxes": [[]], "pred_boxes": [[[0, 0, 2, 2]]], "pred_scores": [[1.0]]}})
                bundle = {"schema_version": 2, "metadata": metadata, "records": records}
                if mutate:
                    mutate(sid, variant, bundle)
                path = directory / f"eval_shard_{sid:02d}of21_{variant}.json"
                path.write_text(json.dumps(bundle), encoding="utf-8")
                entry["files"].append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
            manifest["shards"].append(entry)
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_complete_paired_contract(self):
        manifest = self.write_collection()
        with patch.object(Teacher, "compute_dataset_level_coco_metrics", return_value={"gt_ap50_pct": 50.0, "gt_ap50_95_pct": 30.0}):
            result = merge_eval_jsons_and_print_table(str(self.root), manifest)
        self.assertTrue(result["eligibility"]["final_ready"])
        self.assertEqual(len(result["ablation_variants_summary"]), 5)
        self.assertEqual(result["overall_methods_summary"]["anchor_x265"]["per_qp"]["22"]["frame_count"], 21)
        self.assertIsNone(result["ablation_variants_summary"]["full_tscope"]["bdbr_tot_gt_ap50_pct"])

    def test_missing_record_rejected(self):
        manifest = self.write_collection(lambda sid, variant, bundle: bundle["records"].pop() if sid == 0 and variant == "full_tscope" else None)
        with self.assertRaises(ValueError):
            merge_eval_jsons_and_print_table(str(self.root), manifest)
        self.assertFalse((self.root / "merged_benchmarks.json").exists())

    def test_duplicate_physical_frame_rejected(self):
        manifest = self.write_collection(lambda sid, variant, bundle: bundle["records"].append(copy.deepcopy(bundle["records"][0])) if sid == 0 and variant == "full_tscope" else None)
        with self.assertRaises(ValueError):
            merge_eval_jsons_and_print_table(str(self.root), manifest)

    def test_variant_checkpoint_mismatch_rejected(self):
        manifest = self.write_collection(lambda sid, variant, bundle: bundle["metadata"].update(checkpoint_sha256="a" * 64) if sid == 1 and variant == "full_tscope" else None)
        with self.assertRaises(ValueError):
            merge_eval_jsons_and_print_table(str(self.root), manifest)

    def test_malformed_raw_detection_rejected(self):
        manifest = self.write_collection(lambda sid, variant, bundle: bundle["records"][0]["raw_detections"].update(pred_scores=[[]]) if sid == 0 and variant == "full_tscope" else None)
        with self.assertRaises(ValueError):
            merge_eval_jsons_and_print_table(str(self.root), manifest)


if __name__ == "__main__":
    unittest.main()
