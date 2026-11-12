"""Input-only regression tests. Synthetic fixtures are never plotted or encoded."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import gen_fig_bdbr_curves as curves
from scripts import generate_ablation_visuals as visuals


def method_fixture():
    method = {
        "per_qp": {str(q): {"total_bpp": 0.05 + (37 - q) * 0.005,
                            "p_frame_bpp": 0.03 + (37 - q) * 0.003,
                            "roi_psnr_db": 25.0 + (37 - q) * 0.2,
                            "gt_ap50_pct": 40.0 + (37 - q) * 0.5,
                            "gt_ap50_95_pct": 20.0 + (37 - q) * 0.2}
                   for q in curves.EXPECTED_CRFS},
        "bdbr_unavailable_reasons": {},
    }
    method.update({key: -1.0 for key in curves.BDBR_FIELDS.values()})
    return method


def bundle_fixture():
    return {
        "schema_version": 2,
        "eligibility": {"final_ready": True},
        "crfs": list(curves.EXPECTED_CRFS),
        "configuration": {"codec": dict(curves.PRESERVED_CODEC_SETTINGS)},
        "overall_methods_summary": {key: method_fixture() for key in curves.METHOD_LABELS},
        "ablation_variants_summary": {key: method_fixture() for key in curves.VARIANT_LABELS},
    }


def soup_fixture(variant="full_tscope"):
    return {"schema_version": 2, "variant": variant, "merged_shards": 2,
            "constituents": ["shard0.pt", "shard1.pt"],
            "validation": {"status": "not_evaluated"}, "state_dict": {"tensor": object()}}


class BenchmarkInputTests(unittest.TestCase):
    def test_schema_v2_complete_protocol_is_valid(self):
        bundle = bundle_fixture()
        self.assertIs(curves.validate_bundle(bundle), bundle)
        extracted = curves.extract_curve(bundle["overall_methods_summary"], "anchor_x265", bundle["crfs"])
        self.assertEqual(extracted["total_bpp"], [0.125, 0.1, 0.07500000000000001, 0.05])
        self.assertEqual(len(extracted["gt_ap95"]), 4)

    def test_old_or_unspecified_schema_fails(self):
        for version in (None, 1, "2", True, 2.0):
            with self.subTest(version=version):
                bundle = bundle_fixture()
                bundle["schema_version"] = version
                with self.assertRaisesRegex(curves.PlotInputError, "schema_version"):
                    curves.validate_bundle(bundle)

    def test_no_root_final_ready_fallback(self):
        bundle = bundle_fixture()
        del bundle["eligibility"]
        bundle["final_ready"] = True
        with self.assertRaisesRegex(curves.PlotInputError, "eligibility"):
            curves.validate_bundle(bundle)

    def test_partial_or_nonboolean_gate_fails(self):
        for ready in (False, None, 1, "true"):
            with self.subTest(ready=ready):
                bundle = bundle_fixture()
                bundle["eligibility"]["final_ready"] = ready
                with self.assertRaisesRegex(curves.PlotInputError, "final_ready"):
                    curves.validate_bundle(bundle)

    def test_actual_crfs_only_no_old_summary_fallback(self):
        for crfs in (None, [27, 32, 37, 42, 47], [22, 27, 32], [22, 27, 32, 32],
                     [37, 32, 27, 22], ["22", "27", "32", "37"], [22.0, 27, 32, 37]):
            with self.subTest(crfs=crfs):
                bundle = bundle_fixture()
                bundle["crfs"] = crfs
                with self.assertRaisesRegex(curves.PlotInputError, "crfs"):
                    curves.validate_bundle(bundle)
        with self.assertRaises(curves.PlotInputError):
            curves.validate_bundle({"summary": {"old": method_fixture()}})

    def test_full_cohort_methods_cannot_come_from_ablation(self):
        bundle = bundle_fixture()
        del bundle["overall_methods_summary"]["tscope_temporal_ctu_qp"]
        with self.assertRaisesRegex(curves.PlotInputError, "missing method.*tscope_temporal_ctu_qp"):
            curves.validate_bundle(bundle)

    def test_missing_variant_or_section_fails(self):
        bundle = bundle_fixture()
        del bundle["ablation_variants_summary"]["proxy_only_zhao"]
        with self.assertRaisesRegex(curves.PlotInputError, "missing method.*proxy_only_zhao"):
            curves.validate_bundle(bundle)
        del bundle["ablation_variants_summary"]
        with self.assertRaisesRegex(curves.PlotInputError, "ablation_variants_summary"):
            curves.validate_bundle(bundle)

    def test_missing_extra_or_nonstring_per_qp_keys_fails(self):
        for mutation in ("missing", "extra", "integer"):
            with self.subTest(mutation=mutation):
                bundle = bundle_fixture()
                pqp = bundle["overall_methods_summary"]["anchor_x265"]["per_qp"]
                if mutation == "missing":
                    del pqp["22"]
                elif mutation == "extra":
                    pqp["42"] = copy.deepcopy(pqp["37"])
                else:
                    pqp[22] = pqp.pop("22")
                with self.assertRaisesRegex(curves.PlotInputError, "string CRF keys"):
                    curves.validate_bundle(bundle)

    def test_each_curve_field_requires_finite_measurements(self):
        for field in curves.CURVE_FIELDS.values():
            for value in (None, float("nan"), float("inf"), -float("inf"), "2.1", True):
                with self.subTest(field=field, value=value):
                    bundle = bundle_fixture()
                    bundle["overall_methods_summary"]["anchor_x265"]["per_qp"]["22"][field] = value
                    with self.assertRaisesRegex(curves.PlotInputError, "finite numeric"):
                        curves.validate_bundle(bundle)
            bundle = bundle_fixture()
            del bundle["ablation_variants_summary"]["full_tscope"]["per_qp"]["22"][field]
            with self.assertRaisesRegex(curves.PlotInputError, field):
                curves.validate_bundle(bundle)

    def test_positive_rates_and_percent_ap_are_enforced(self):
        for field, value in (("total_bpp", 0), ("p_frame_bpp", -0.01),
                             ("gt_ap50_pct", -1), ("gt_ap50_95_pct", 101)):
            with self.subTest(field=field):
                bundle = bundle_fixture()
                bundle["overall_methods_summary"]["anchor_x265"]["per_qp"]["22"][field] = value
                with self.assertRaises(curves.PlotInputError):
                    curves.validate_bundle(bundle)

    def test_null_bdbr_is_annotated_not_fabricated(self):
        bundle = bundle_fixture()
        method = bundle["ablation_variants_summary"]["full_tscope"]
        for field in curves.BDBR_FIELDS.values():
            method[field] = None
            method["bdbr_unavailable_reasons"][field] = "no common AP interval"
        curves.validate_bundle(bundle)
        extracted = curves.extract_curve(bundle["ablation_variants_summary"], "full_tscope", bundle["crfs"])
        for field in curves.BDBR_FIELDS:
            self.assertIsNone(extracted[field])
            self.assertEqual(curves.bdbr_text(extracted, field), "BDBR unavailable: no common AP interval")

    def test_all_bdbr_fields_required_with_finite_or_reasoned_null(self):
        for field in curves.BDBR_FIELDS.values():
            for value in (None, float("nan"), float("inf"), "-3", True):
                with self.subTest(field=field, value=value):
                    bundle = bundle_fixture()
                    bundle["overall_methods_summary"]["anchor_x265"][field] = value
                    with self.assertRaises(curves.PlotInputError):
                        curves.validate_bundle(bundle)
            bundle = bundle_fixture()
            del bundle["overall_methods_summary"]["anchor_x265"][field]
            with self.assertRaisesRegex(curves.PlotInputError, "missing"):
                curves.validate_bundle(bundle)

    def test_preserved_codec_grid_is_labelled_distinctly(self):
        caption = curves.codec_caption(bundle_fixture())
        self.assertIn("QG=16 px", caption)
        self.assertIn("16x16 regions", caption)
        self.assertIn("[-0.18, +0.14]", caption)
        for key in curves.PRESERVED_CODEC_SETTINGS:
            bundle = bundle_fixture()
            bundle["configuration"]["codec"][key] = 123
            with self.assertRaisesRegex(curves.PlotInputError, key):
                curves.validate_bundle(bundle)

    def test_validate_only_loads_json_without_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            output = Path(directory) / "never-render"
            path.write_text(json.dumps(bundle_fixture(), allow_nan=False), encoding="utf-8")
            with patch.object(curves, "render_figures") as render, patch("builtins.print"):
                curves.main(["--benchmark", str(path), "--output-dir", str(output), "--validate-only"])
                render.assert_not_called()
            self.assertFalse(output.exists())

    def test_invalid_bundle_fails_before_render_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.json"
            bundle = bundle_fixture()
            bundle["eligibility"]["final_ready"] = False
            path.write_text(json.dumps(bundle), encoding="utf-8")
            with patch.object(curves, "render_figures") as render, patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    curves.main(["--benchmark", str(path), "--output-dir", str(Path(directory) / "output")])
                render.assert_not_called()

    def test_missing_or_bad_json_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.json"
            with self.assertRaisesRegex(curves.PlotInputError, "Cannot read merged benchmark"):
                curves.load_benchmark(path)
            path.write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(curves.PlotInputError, "Cannot read merged benchmark"):
                curves.load_benchmark(path)


class QualitativeInputTests(unittest.TestCase):
    def test_schema2_soup_not_evaluated_or_validated_allowed_for_preedit(self):
        for status in ("not_evaluated", "validated"):
            soup = soup_fixture()
            soup["validation"]["status"] = status
            self.assertIs(visuals.validate_soup_metadata(soup, "full_tscope"), soup)

    def test_raw_shard_old_wrong_variant_or_empty_soup_is_rejected(self):
        for key, value in (("schema_version", 1), ("variant", "spatial_only_scope"),
                           ("merged_shards", 0), ("merged_shards", True), ("constituents", []),
                           ("constituents", ["shard0.pt", "shard0.pt"]), ("validation", {}),
                           ("validation", {"status": "failed"}), ("state_dict", {})):
            with self.subTest(key=key):
                soup = soup_fixture()
                soup[key] = value
                with self.assertRaises(curves.PlotInputError):
                    visuals.validate_soup_metadata(soup, "full_tscope")
        for key in ("variant", "merged_shards", "constituents", "validation", "state_dict"):
            soup = soup_fixture()
            del soup[key]
            with self.assertRaises(curves.PlotInputError):
                visuals.validate_soup_metadata(soup, "full_tscope")

    def test_contradictory_config_rejected(self):
        soup = soup_fixture()
        soup["configuration"] = {"model": {"qg_size": 32}}
        with self.assertRaisesRegex(curves.PlotInputError, "contradicts preserved"):
            visuals.validate_soup_metadata(soup, "full_tscope")
        soup["configuration"] = {"model": {"max_delta_qp": 6.0}}
        with self.assertRaisesRegex(curves.PlotInputError, "contradicts preserved"):
            visuals.validate_soup_metadata(soup, "full_tscope")
        soup["configuration"] = {"ablation": {"max_delta_qp": 6.0}}
        with self.assertRaisesRegex(curves.PlotInputError, "contradicts preserved"):
            visuals.validate_soup_metadata(soup, "full_tscope")

    def test_loader_uses_weights_only_and_strict_state_dict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "explicit.pt"
            path.touch()
            soup = soup_fixture()
            torch_mock = Mock()
            torch_mock.load.return_value = soup
            torch_mock.is_tensor.return_value = True
            torch_mock.isfinite.return_value.all.return_value.item.return_value = True
            factory = Mock()
            model, loaded = visuals.load_soup_model(path, "full_tscope", torch_module=torch_mock, model_factory=factory)
            torch_mock.load.assert_called_once_with(path, map_location="cpu", weights_only=True)
            model.load_state_dict.assert_called_once_with(soup["state_dict"], strict=True)
            model.eval.assert_called_once()
            self.assertIs(loaded, soup)
            self.assertEqual(factory.call_args.kwargs["max_delta_qp"], 4.0)
            self.assertEqual(factory.call_args.kwargs["qg_size"], 16)

    def test_missing_checkpoint_never_constructs_random_model(self):
        with tempfile.TemporaryDirectory() as directory:
            factory = Mock()
            torch_mock = Mock()
            with self.assertRaisesRegex(curves.PlotInputError, "Missing explicit"):
                visuals.load_soup_model(Path(directory) / "absent.pt", "full_tscope",
                                        torch_module=torch_mock, model_factory=factory)
            factory.assert_not_called()
            torch_mock.load.assert_not_called()

    def test_wrong_variant_or_nonfinite_weights_fail_before_model_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "soup.pt"
            path.touch()
            for wrong_variant in (True, False):
                factory = Mock()
                torch_mock = Mock()
                torch_mock.load.return_value = soup_fixture("spatial_only_scope" if wrong_variant else "full_tscope")
                torch_mock.is_tensor.return_value = True
                torch_mock.isfinite.return_value.all.return_value.item.return_value = False
                with self.assertRaises(curves.PlotInputError):
                    visuals.load_soup_model(path, "full_tscope", torch_module=torch_mock, model_factory=factory)
                factory.assert_not_called()

    def test_strict_load_failure_does_not_return_random_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "soup.pt"
            path.touch()
            torch_mock = Mock()
            torch_mock.load.return_value = soup_fixture()
            torch_mock.is_tensor.return_value = True
            torch_mock.isfinite.return_value.all.return_value.item.return_value = True
            factory = Mock()
            factory.return_value.load_state_dict.side_effect = RuntimeError("missing key")
            with self.assertRaisesRegex(curves.PlotInputError, "Cannot safely load.*missing key"):
                visuals.load_soup_model(path, "full_tscope", torch_module=torch_mock, model_factory=factory)
            factory.return_value.eval.assert_not_called()

    def test_explicit_checkpoint_selection_and_no_shared_full_checkpoint(self):
        self.assertEqual(visuals.parse_checkpoints(["proxy_only_zhao=C:/data/proxy.pt"])["proxy_only_zhao"], Path("C:/data/proxy.pt"))
        for values in ([], ["bad=x.pt"], ["full_tscope="], ["full_tscope"],
                       ["full_tscope=x.pt", "full_tscope=y.pt"],
                       ["full_tscope=x.pt", "wo_ctu_delta_qp=x.pt"]):
            with self.subTest(values=values), self.assertRaises(curves.PlotInputError):
                visuals.parse_checkpoints(values)

    def test_heldout_identity_selection_is_stable_under_reordering(self):
        selected = {"type": "mot_frames", "seq_name": "MOT17-11-FRCNN",
                    "frame_paths": [Path(f"{n:06d}.jpg") for n in range(5, 9)]}
        other = {"type": "mot_frames", "seq_name": "MOT17-13-FRCNN",
                 "frame_paths": [Path(f"{n:06d}.jpg") for n in range(1, 5)]}
        for clips in ([selected, other], [other, selected]):
            index, time, paths = visuals.select_heldout_clip(clips, "MOT17-11-FRCNN", 5, 6, 4)
            self.assertIs(clips[index], selected)
            self.assertEqual(time, 1)
            self.assertEqual(paths[time].name, "000006.jpg")

    def test_missing_duplicate_train_or_nonconsecutive_clip_fails(self):
        clip = {"type": "mot_frames", "seq_name": "MOT17-11-FRCNN",
                "frame_paths": [Path(f"{n:06d}.jpg") for n in range(1, 5)]}
        for clips, seq, start, frame in (([], "MOT17-11-FRCNN", 1, 2),
                                       ([clip, clip], "MOT17-11-FRCNN", 1, 2),
                                       ([clip], "MOT17-02-FRCNN", 1, 2),
                                       ([clip], "MOT17-11-FRCNN", 1, 8)):
            with self.subTest(seq=seq, frame=frame), self.assertRaises(curves.PlotInputError):
                visuals.select_heldout_clip(clips, seq, start, frame, 4)
        clip["frame_paths"] = [Path(f"{n:06d}.jpg") for n in (1, 2, 4, 5)]
        with self.assertRaisesRegex(curves.PlotInputError, "nonconsecutive"):
            visuals.select_heldout_clip([clip], "MOT17-11-FRCNN", 1, 2, 4)

    def test_truthful_internal_and_multifactor_variant_labels(self):
        self.assertIn("Internal proxy-trained baseline", curves.VARIANT_LABELS["proxy_only_zhao"])
        self.assertIn("not prior-art reproduction", curves.VARIANT_LABELS["proxy_only_zhao"])
        spatial = visuals.VARIANT_SETTINGS["spatial_only_scope"]
        self.assertFalse(spatial["use_real_x265_ste"])
        self.assertEqual(spatial["max_delta_qp"], 0.0)
        self.assertEqual(spatial["lambda_temp_bg"], 0.0)
        self.assertEqual(spatial["lambda_temp_obj"], 0.0)
        self.assertIn("proxy-trained", curves.VARIANT_LABELS["spatial_only_scope"])


if __name__ == "__main__":
    unittest.main()
