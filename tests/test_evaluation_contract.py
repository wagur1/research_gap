import argparse
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.experiment_protocol import EXPECTED_TEST_FRAME_IDS
from scripts.kaggle_shard_runner import run_poc_wishlist2_and_eval_shard
from scripts.validate_soups import validate
from video_codecs.hybrid_x265_ste import HybridX265STE


class EvaluationContractTests(unittest.TestCase):
    def test_expected_complete_frame_count(self):
        self.assertEqual(len(EXPECTED_TEST_FRAME_IDS), 1648)
        self.assertEqual(len(set(EXPECTED_TEST_FRAME_IDS)), 1648)
        self.assertNotIn("MOT17-13/000749", EXPECTED_TEST_FRAME_IDS)

    def test_final_evaluation_rejects_synthetic(self):
        args = argparse.Namespace(use_pretrained_fpn=True, disjoint_split=True)
        with patch("scripts.kaggle_shard_runner.build_sharded_dataset", return_value=object()):
            with self.assertRaisesRegex(RuntimeError, "real held-out"):
                run_poc_wishlist2_and_eval_shard(args)

    def test_release_validation_rejects_missing_data_before_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(data_root=directory, soup_dir=directory, output_dir=str(Path(directory) / "out"), variants=["spatial_only_scope"], device="cpu")
            with self.assertRaisesRegex(ValueError, "131"):
                validate(args)
            self.assertFalse((Path(directory) / "out").exists())

    def test_dynamic_roi_graph_retains_each_frame(self):
        offsets = torch.zeros(3, 2, 2)
        offsets[0, 0, 0] = -4
        offsets[1, 1, 1] = 8
        offsets[2, 0, 1] = -2
        args = HybridX265STE._build_addroi_filter_from_offsets(offsets, 32, 32)
        graph = args[1]
        self.assertIn("split=3", graph)
        for index in range(3):
            self.assertIn(f"select='eq(n,{index})'", graph)
        self.assertIn("concat=n=3:v=1:a=0,settb=1/25,setpts=N", graph)
        self.assertIn("qoffset=0.14000", graph)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Real codec tools unavailable")
    def test_real_codec_integer_packets_and_decode(self):
        generator = torch.Generator().manual_seed(11)
        video = torch.rand(1, 3, 3, 32, 32, generator=generator)
        offsets = torch.zeros(1, 3, 2, 2)
        offsets[0, 0, 0, 0] = -4
        offsets[0, 1, 1, 1] = 4
        offsets[0, 2, 0, 1] = -2
        output = HybridX265STE().evaluate_real_x265(video, 32, offsets)
        self.assertEqual(tuple(output["video_recon"].shape), tuple(video.shape))
        self.assertEqual(output["frame_types"], ["I", "P", "P"])
        self.assertEqual(len(output["packet_sizes"]), 3)
        self.assertTrue(all(type(size) is int and size > 0 for size in output["packet_sizes"]))
        self.assertAlmostEqual(output["total_bpp"], sum(output["packet_sizes"]) * 8 / (3 * 32 * 32), places=5)
        self.assertTrue(torch.isfinite(output["video_recon"]).all())


if __name__ == "__main__":
    unittest.main()
