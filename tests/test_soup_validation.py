"""Focused tests for faithful, memory-bounded training-protocol soup validation."""
import copy
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from losses.vcm_temporal_loss import VCMTemporalPreEditLoss
from models.frozen_detector_fpn import FrozenFPNDetectorTeacher
from scripts.experiment_protocol import VALIDATION_PROTOCOL, validate_release_validation
from scripts.validate_soups import extract_validation_features, validate


class FakeFeatureTeacher:
    """Frame-independent feature extraction that exposes chunk sizes and ordering."""

    def __init__(self):
        self.call_sizes = []

    def extract_fpn_features(self, images):
        self.call_sizes.append(images.shape[0])
        return {
            f"p{level}": F.avg_pool2d(images, kernel_size=stride, stride=stride)
            for level, stride in zip(range(2, 6), (4, 8, 16, 32))
        }


def validation_evidence():
    return {
        "status": "validated",
        "release_ready": True,
        "split": "MOT17-09",
        "loss_total": 1.0,
        "protocol": copy.deepcopy(VALIDATION_PROTOCOL),
        "data_fingerprints": {
            "gt_sha256": "a" * 64,
            "frames": [{"frame": number, "sha256": "b" * 64} for number in range(1, 525)],
        },
        "source_soup_sha256": "c" * 64,
        "selection": "uniform_soup_no_test_selection",
    }


def criterion_inputs(img_size=64):
    """Two differently masked frames in one intact temporal clip."""
    generator = torch.Generator().manual_seed(2026)
    clean = torch.rand((1, 2, 3, img_size, img_size), generator=generator)
    edited = clean * 0.93 + 0.02
    reconstructed = edited * 0.98 + 0.01
    scale = img_size // 64
    mask = torch.zeros((1, 2, 1, img_size, img_size))
    mask[:, 0, :, 4 * scale:16 * scale, 4 * scale:16 * scale] = 1.0
    mask[:, 1, :, 8 * scale:56 * scale, 8 * scale:56 * scale] = 1.0
    tight_mask = torch.zeros_like(mask)
    tight_mask[:, 0, :, 6 * scale:14 * scale, 6 * scale:14 * scale] = 1.0
    tight_mask[:, 1, :, 12 * scale:52 * scale, 12 * scale:52 * scale] = 1.0
    inputs = {
        "i_edit": edited,
        "i_recon": reconstructed,
        "attn_map": 0.2 + 0.6 * clean[:, :, :1],
        "delta_obj": 0.01 * clean,
        "delta_bg": -0.005 * clean,
        "rate_proxy_bpp": torch.tensor([0.25]),
        "m_attn": tight_mask,
        "m_det": mask,
    }
    return clean.reshape(2, 3, img_size, img_size), reconstructed.reshape(2, 3, img_size, img_size), inputs


class PreservedValidationProtocolTests(unittest.TestCase):
    def test_exact_training_protocol(self):
        self.assertEqual(VALIDATION_PROTOCOL, {
            "full_clips": 131,
            "val_subsample": 2,
            "batch_size": 8,
            "severities": [0.3, 0.7],
            "img_size": 512,
            "aggregation": "mean_batch_severity_loss_matching_training",
            "held_out_test_used": False,
        })

    def test_release_gate_accepts_corrected_subsample(self):
        evidence = validation_evidence()
        self.assertIs(validate_release_validation(evidence), evidence)

    def test_release_gate_rejects_old_subsample_four(self):
        evidence = validation_evidence()
        evidence["protocol"]["val_subsample"] = 4
        with self.assertRaisesRegex(ValueError, "protocol"):
            validate_release_validation(evidence)

    def test_training_clip_and_batch_schedule_includes_two_clip_tail(self):
        dataset = list(range(VALIDATION_PROTOCOL["full_clips"]))
        indices = list(range(0, len(dataset), VALIDATION_PROTOCOL["val_subsample"]))
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=VALIDATION_PROTOCOL["batch_size"], shuffle=False)
        batches = [batch.tolist() for batch in loader]
        self.assertEqual(indices, list(range(0, 131, 2)))
        self.assertEqual(len(subset), 66)
        self.assertEqual([len(batch) for batch in batches], [8] * 8 + [2])
        self.assertEqual([index for batch in batches for index in batch], indices)
        self.assertEqual(batches[-1], [128, 130])
        self.assertEqual(len(batches) * len(VALIDATION_PROTOCOL["severities"]), 18)
        measured_frames = [frame for index in indices for frame in range(index * 4 + 1, index * 4 + 5)]
        self.assertEqual(len(measured_frames), 264)
        self.assertEqual(measured_frames[:4], [1, 2, 3, 4])
        self.assertEqual(measured_frames[-4:], [521, 522, 523, 524])


class ChunkedValidationFeatureTests(unittest.TestCase):
    @torch.no_grad()
    def test_default_chunk_size_is_one_and_preserves_levels_and_frames(self):
        images = torch.arange(5, dtype=torch.float32).view(5, 1, 1, 1).expand(5, 3, 64, 64)
        whole_teacher = FakeFeatureTeacher()
        expected = whole_teacher.extract_fpn_features(images)
        teacher = FakeFeatureTeacher()
        actual = extract_validation_features(teacher, images)
        self.assertEqual(teacher.call_sizes, [1] * 5)
        self.assertEqual(list(actual), ["p2", "p3", "p4", "p5"])
        for level in expected:
            torch.testing.assert_close(actual[level], expected[level], rtol=0, atol=0)
            self.assertEqual(actual[level][:, 0, 0, 0].tolist(), list(range(5)))

    @torch.no_grad()
    def test_nondivisible_chunks_preserve_short_last_chunk(self):
        images = torch.arange(5, dtype=torch.float32).view(5, 1, 1, 1).expand(5, 3, 64, 64)
        expected = FakeFeatureTeacher().extract_fpn_features(images)
        for chunk_size, call_sizes in ((2, [2, 2, 1]), (3, [3, 2]), (8, [5])):
            with self.subTest(chunk_size=chunk_size):
                teacher = FakeFeatureTeacher()
                actual = extract_validation_features(teacher, images, chunk_size=chunk_size)
                self.assertEqual(teacher.call_sizes, call_sizes)
                self.assertEqual(list(actual), list(expected))
                for level in expected:
                    torch.testing.assert_close(actual[level], expected[level], rtol=0, atol=0)

    @torch.no_grad()
    def test_chunking_preserves_whole_clip_heterogeneous_mask_criterion(self):
        clean, reconstructed, inputs = criterion_inputs()
        criterion = VCMTemporalPreEditLoss(lambda_temp_bg=1.0, lambda_temp_obj=0.5)
        teacher = FakeFeatureTeacher()
        whole_losses = criterion(
            **inputs,
            teacher_fpn_feats=teacher.extract_fpn_features(clean),
            student_fpn_feats=teacher.extract_fpn_features(reconstructed),
        )
        chunked_losses = criterion(
            **inputs,
            teacher_fpn_feats=extract_validation_features(teacher, clean),
            student_fpn_feats=extract_validation_features(teacher, reconstructed),
        )
        self.assertNotEqual(inputs["m_det"][:, 0].sum().item(), inputs["m_det"][:, 1].sum().item())
        self.assertGreater(whole_losses["loss_task"].item(), 0)
        self.assertGreater(whole_losses["loss_temp"].item(), 0)
        for key in whole_losses:
            self.assertTrue(torch.isfinite(chunked_losses[key]).item())
            torch.testing.assert_close(chunked_losses[key], whole_losses[key], rtol=0, atol=0)


class FakeCompleteValidationDataset:
    """Complete manifest metadata, with tiny in-memory tensors instead of real files."""

    def __init__(self, root):
        sequence = Path(root) / "MOT17-09-FRCNN" / "img1"
        self.clips = [
            {
                "type": "mot_frames",
                "seq_name": "MOT17-09-FRCNN",
                "frame_paths": [sequence / f"{number:06d}.jpg" for number in range(index * 4 + 1, index * 4 + 5)],
            }
            for index in range(131)
        ]
        self.accessed_indices = []

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        self.accessed_indices.append(index)
        return {
            "video": torch.full((4, 3, 32, 32), float(index)),
            "m_attn": torch.ones((4, 1, 32, 32)),
            "m_det": torch.ones((4, 1, 32, 32)),
        }


class FakeValidationTeacher(FakeFeatureTeacher):
    use_torchvision = True
    detector = object()

    def __init__(self):
        super().__init__()
        self.eval_called = False

    def to(self, device):
        return self

    def eval(self):
        self.eval_called = True
        return self


class FakeValidationModel:
    def __init__(self):
        self.calls = []
        self.strict_load_called = False
        self.eval_called = False

    def to(self, device):
        return self

    def load_state_dict(self, state, strict):
        self.strict_load_called = strict

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, video, family, severity, reuse_iframe_qp, disable_temporal):
        self.calls.append({
            "indices": video[:, 0, 0, 0, 0].tolist(),
            "family": family.tolist(),
            "severity": severity.tolist(),
            "reuse_iframe_qp": reuse_iframe_qp,
            "disable_temporal": disable_temporal,
        })
        b, t, _, _, _ = video.shape
        return {
            "i_edit": video,
            "attn_map": torch.ones_like(video[:, :, :1]),
            "delta_obj": torch.zeros_like(video),
            "delta_bg": torch.zeros_like(video),
            "ctu_qp_offsets": torch.zeros((b, t, 2, 2)),
        }


class FakeValidationCodec:
    use_real_x265 = False

    def to(self, device):
        return self

    def __call__(self, video, severity, offsets):
        return {"video_recon": video, "rate_proxy_bpp": torch.zeros(video.shape[0])}


class FakeValidationCriterion:
    def __init__(self, values):
        self.values = list(values)
        self.batch_sizes = []

    def to(self, device):
        return self

    def __call__(self, **inputs):
        index = len(self.batch_sizes)
        self.batch_sizes.append(inputs["i_edit"].shape[0])
        return {"loss_total": torch.tensor(self.values[index])}


@contextmanager
def mocked_validation(root, values):
    dataset = FakeCompleteValidationDataset(root)
    teacher = FakeValidationTeacher()
    model = FakeValidationModel()
    criterion = FakeValidationCriterion(values)
    checkpoint = {"state_dict": {}, "constituents": [{"best_val_loss": 1.0}]}
    with ExitStack() as stack:
        dataset_factory = stack.enter_context(patch("scripts.validate_soups.RealVCMVideoDataset", return_value=dataset))
        stack.enter_context(patch("scripts.validate_soups.FrozenFPNDetectorTeacher", return_value=teacher))
        stack.enter_context(patch("scripts.validate_soups.TemporalSCOPE", return_value=model))
        stack.enter_context(patch("scripts.validate_soups.HybridX265STE", return_value=FakeValidationCodec()))
        stack.enter_context(patch("scripts.validate_soups.VCMTemporalPreEditLoss", return_value=criterion))
        stack.enter_context(patch("scripts.validate_soups.sha256_file", return_value="a" * 64))
        stack.enter_context(patch("scripts.validate_soups.torch.load", return_value=checkpoint))
        stack.enter_context(patch("scripts.validate_soups.validate_soup_provenance"))
        # No process-global thread changes or optional torchvision imports are needed by these mocks.
        stack.enter_context(patch("scripts.validate_soups.torch.set_num_threads"))
        stack.enter_context(patch.dict("sys.modules", {"torchvision": SimpleNamespace(__version__="test") }))
        stack.enter_context(patch("builtins.print"))
        saved = stack.enter_context(patch("scripts.validate_soups.torch.save"))
        yield SimpleNamespace(dataset=dataset, dataset_factory=dataset_factory, teacher=teacher,
                              model=model, criterion=criterion, checkpoint=checkpoint, saved=saved)


class FullValidatorRegressionTests(unittest.TestCase):
    def test_validate_uses_actual_schedule_and_equal_batch_severity_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "out"
            args = SimpleNamespace(data_root=directory, soup_dir=directory, output_dir=str(destination),
                                   variants=["spatial_only_scope"], device="cpu")
            # A deliberately large tail loss distinguishes equal batch weighting from sample weighting.
            values = list(range(1, 17)) + [100.0, 200.0]
            with mocked_validation(directory, values) as mocks:
                paths = validate(args)
                evidence = mocks.checkpoint["validation"]
                execution = evidence["execution_provenance"]
                self.assertEqual(mocks.dataset.accessed_indices, list(range(0, 131, 2)))
                self.assertEqual(mocks.criterion.batch_sizes, [8] * 16 + [2] * 2)
                self.assertEqual(execution["logical_batch_sizes"], [8] * 8 + [2])
                self.assertEqual(execution["selected_clip_indices"], list(range(0, 131, 2)))
                self.assertEqual(execution["batch_severity_loss_count"], 18)
                expected_frames = [f"MOT17-09/{number:06d}" for index in range(0, 131, 2)
                                   for number in range(index * 4 + 1, index * 4 + 5)]
                self.assertEqual(execution["measured_frame_ids"], expected_frames)
                self.assertEqual(evidence["protocol"], VALIDATION_PROTOCOL)
                self.assertAlmostEqual(evidence["loss_total"], sum(values) / 18)
                sample_weighted = sum(loss * size for loss, size in zip(values, mocks.criterion.batch_sizes)) / sum(mocks.criterion.batch_sizes)
                self.assertNotAlmostEqual(evidence["loss_total"], sample_weighted)
                self.assertEqual(len(mocks.model.calls), 18)
                for index, call in enumerate(mocks.model.calls):
                    expected_batch = list(range(0, 131, 2))[(index // 2) * 8:(index // 2 + 1) * 8]
                    self.assertEqual(call["indices"], expected_batch)
                    self.assertEqual(call["family"], [0] * len(expected_batch))
                    for severity in call["severity"]:
                        self.assertAlmostEqual(severity, (0.3, 0.7)[index % 2], places=6)
                    self.assertFalse(call["reuse_iframe_qp"])
                    self.assertTrue(call["disable_temporal"])
                self.assertTrue(mocks.teacher.eval_called)
                self.assertTrue(mocks.model.eval_called)
                self.assertTrue(mocks.model.strict_load_called)
                self.assertEqual(mocks.teacher.call_sizes, [1] * (264 * 3))
                self.assertEqual(execution["feature_chunk_size"], 1)
                mocks.dataset_factory.assert_called_once_with(directory, shard_id=0, num_shards=1,
                    clip_len=4, stride=4, img_size=VALIDATION_PROTOCOL["img_size"], is_train=False, enforce_disjoint_split=True, split="val")
                self.assertEqual(paths, {"spatial_only_scope": destination / "tscope_soup_spatial_only_scope.pt"})
                mocks.saved.assert_called_once()
                self.assertTrue(paths["spatial_only_scope"].is_file())

    def test_nonfinite_loss_aborts_immediately_without_output(self):
        for nonfinite in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(loss=nonfinite), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "out"
                args = SimpleNamespace(data_root=directory, soup_dir=directory, output_dir=str(destination),
                                       variants=["spatial_only_scope"], device="cpu", feature_chunk_size=1)
                with mocked_validation(directory, [nonfinite]) as mocks:
                    with self.assertRaisesRegex(ValueError, "nonfinite"):
                        validate(args)
                    mocks.saved.assert_not_called()
                    self.assertFalse(destination.exists())
                    self.assertEqual(len(mocks.criterion.batch_sizes), 1)
                    self.assertNotIn("validation", mocks.checkpoint)


class OfflineRealDetectorEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from torchvision.models.detection import (
                FasterRCNN_ResNet50_FPN_V2_Weights,
                fasterrcnn_resnet50_fpn_v2,
            )
        except (ImportError, RuntimeError) as error:
            raise unittest.SkipTest(f"Torchvision detection unavailable: {error}")
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        cached_weights = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
        if not cached_weights.is_file():
            raise unittest.SkipTest("Pretrained Faster R-CNN v2 weights are not cached; downloads forbidden")
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(4)
        cls.addClassCleanup(torch.set_num_threads, previous_threads)
        # Build without any pretrained-weight request, then load local weights explicitly.
        # A download attempt is a failure, never a fallback to random detector weights.
        with patch("torch.hub.download_url_to_file", side_effect=AssertionError("Network downloads forbidden")):
            detector = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
            detector.load_state_dict(torch.load(cached_weights, map_location="cpu", weights_only=True), strict=True)
        detector.eval()
        for parameter in detector.parameters():
            parameter.requires_grad_(False)
        cls.teacher = FrozenFPNDetectorTeacher(pretrained=False)
        cls.teacher.detector = detector
        cls.teacher.use_torchvision = True
        cls.teacher.eval()
        cls.addClassCleanup(delattr, cls, "teacher")

    @torch.no_grad()
    def test_cached_pretrained_features_and_heterogeneous_loss_match(self):
        clean, reconstructed, inputs = criterion_inputs(img_size=VALIDATION_PROTOCOL["img_size"])
        whole_clean = self.teacher.extract_fpn_features(clean)
        whole_reconstructed = self.teacher.extract_fpn_features(reconstructed)
        chunked_clean = extract_validation_features(self.teacher, clean, chunk_size=1)
        chunked_reconstructed = extract_validation_features(self.teacher, reconstructed, chunk_size=1)
        self.assertEqual(list(whole_clean), ["p2", "p3", "p4", "p5"])
        p2_side = VALIDATION_PROTOCOL["img_size"] // 4
        self.assertEqual(tuple(whole_clean["p2"].shape), (2, 256, p2_side, p2_side))
        for level in whole_clean:
            with self.subTest(level=level):
                torch.testing.assert_close(chunked_clean[level], whole_clean[level], rtol=1e-4, atol=1e-5)
                torch.testing.assert_close(chunked_reconstructed[level], whole_reconstructed[level], rtol=1e-4, atol=1e-5)
        criterion = VCMTemporalPreEditLoss(lambda_temp_bg=1.0, lambda_temp_obj=0.5)
        whole_losses = criterion(**inputs, teacher_fpn_feats=whole_clean, student_fpn_feats=whole_reconstructed)
        chunked_losses = criterion(**inputs, teacher_fpn_feats=chunked_clean, student_fpn_feats=chunked_reconstructed)
        for key in whole_losses:
            self.assertTrue(torch.isfinite(chunked_losses[key]).item())
            torch.testing.assert_close(chunked_losses[key], whole_losses[key], rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
