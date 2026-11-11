import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.temporal_scope import TemporalSCOPE
from models.frozen_detector_fpn import FrozenFPNDetectorTeacher
from video_codecs.hybrid_x265_ste import HybridX265STE
from vcm_datasets.vcm_video_datasets import RealVCMVideoDataset
from scripts.merge_shards import bjontegaard_delta_rate


def sha256_file(path: Path) -> str:
    if not path.exists():
        return "none"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]


class CrossDetectorEvaluator(nn.Module):
    """
    Evaluates Zero-Shot Cross-Detector / Cross-Backbone Generalization of T-SCOPE
    (trained ONLY with Frozen Faster R-CNN ResNet50-FPN v2 teacher) on REAL `libx265`
    bitstreams (`evaluate_real_x265`), with ZERO ground-truth leakage at inference
    (`pred["i_edit"]` and `pred["ctu_qp_offsets"]`), and OFFICIAL Dataset-Level Ground-Truth
    COCO/VOC Average Precision (`gt_ap50_pct` & `gt_ap50_95_pct` against `gt/gt.txt` class 1
    pedestrians with MOT17 ignore regions handled) (R01, R03, R04 fix):
      1. `fasterrcnn_r50_fpn_v2` : Two-Stage Anchor-Based (In-domain Teacher Detector)
      2. `retinanet_r50_fpn_v2`  : One-Stage Focal-Loss Dense Detector (Zero-Shot Detector)
      3. `fcos_r50_fpn`          : One-Stage Anchor-Free Detector (Zero-Shot Detector)
      4. `ssdlite_mobilenet_v3`  : Lightweight Mobile/Edge Detector (Zero-Shot Detector)
      5. `swin_t_transformer`    : Vision Transformer Backbone (Feature-Space SNR/Cosine ONLY; never reported as detector mAP)
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn_v2,
            FasterRCNN_ResNet50_FPN_V2_Weights,
            retinanet_resnet50_fpn_v2,
            RetinaNet_ResNet50_FPN_V2_Weights,
            fcos_resnet50_fpn,
            FCOS_ResNet50_FPN_Weights,
            ssdlite320_mobilenet_v3_large,
            SSDLite320_MobileNet_V3_Large_Weights,
        )
        from torchvision.models import swin_t, Swin_T_Weights

        print("[CrossDetector] Loading 4 pretrained object detectors + 1 Swin-T feature backbone...")
        self.detectors = {
            "fasterrcnn_r50_fpn_v2": fasterrcnn_resnet50_fpn_v2(
                weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            ).to(device).eval(),
            "retinanet_r50_fpn_v2": retinanet_resnet50_fpn_v2(
                weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
            ).to(device).eval(),
            "fcos_r50_fpn": fcos_resnet50_fpn(
                weights=FCOS_ResNet50_FPN_Weights.DEFAULT
            ).to(device).eval(),
            "ssdlite_mobilenet_v3": ssdlite320_mobilenet_v3_large(
                weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            ).to(device).eval(),
        }
        self.swin = swin_t(weights=Swin_T_Weights.DEFAULT).features.to(device).eval()

        for m in list(self.detectors.values()) + [self.swin]:
            for p in m.parameters():
                p.requires_grad_(False)

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        )

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def _extract_backbone_feats(
        self, name: str, imgs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        x = self._norm(torch.clamp(imgs, 0.0, 1.0))
        if name in self.detectors:
            return dict(self.detectors[name].backbone(x))
        if name == "swin_t_transformer":
            out = {}
            cur = x
            for idx, layer in enumerate(self.swin):
                cur = layer(cur)
                if idx in (1, 3, 5, 7):
                    out[f"s{idx}"] = cur.permute(0, 3, 1, 2).contiguous()
            return out
        raise ValueError(f"Unknown backbone: {name}")

    @torch.no_grad()
    def extract_clip_metrics_and_dets(
        self,
        backbone_name: str,
        clean_video: torch.Tensor,
        recon_video: torch.Tensor,
        m_det: torch.Tensor,
        gt_boxes: torch.Tensor,
        ignore_boxes: torch.Tensor,
        f_clean_cached: Dict[str, torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Computes masked feature SNR (dB), masked feature cosine similarity (%),
        and if `backbone_name` is an object detector, runs full object detection inference
        on `recon_video` to return per-frame pedestrian predictions (`labels == 1`)
        for official dataset-level COCO/VOC AP evaluation.
        Never uses any heuristic PSNR/cosine-to-mAP formula (R01 fix).
        """
        b, t, c, h, w = clean_video.shape
        clean_flat = clean_video.view(b * t, c, h, w)
        recon_flat = recon_video.view(b * t, c, h, w)
        m_det_flat = m_det.view(b * t, 1, h, w)

        f_clean = (
            f_clean_cached
            if f_clean_cached is not None
            else self._extract_backbone_feats(backbone_name, clean_flat)
        )
        f_recon = self._extract_backbone_feats(backbone_name, recon_flat)

        cos_list: List[float] = []
        snr_list: List[float] = []

        for k, fc in f_clean.items():
            fr = f_recon[k]
            ml = F.interpolate(m_det_flat, size=fc.shape[-2:], mode="nearest")
            if float(ml.sum().item()) < 1.0:
                ml = torch.ones_like(ml)

            cos_map = F.cosine_similarity(fc, fr, dim=1, eps=1e-6).unsqueeze(1)
            masked_cos = float(((ml * cos_map).sum() / (ml.sum() + 1e-6)).item())
            cos_list.append(masked_cos * 100.0)

            sig_pow = ((ml * fc) ** 2).sum() / (ml.sum() * fc.shape[1] + 1e-6)
            err_pow = ((ml * (fc - fr)) ** 2).sum() / (ml.sum() * fc.shape[1] + 1e-6)
            feat_snr_db = float(10.0 * torch.log10((sig_pow + 1e-6) / (err_pow + 1e-8)).item())
            snr_list.append(feat_snr_db)

        pred_boxes_list: List[torch.Tensor] = []
        pred_scores_list: List[torch.Tensor] = []
        gt_boxes_list: List[torch.Tensor] = []
        ig_boxes_list: List[torch.Tensor] = []

        gt_flat = gt_boxes.view(b * t, -1, 4)
        ig_flat = ignore_boxes.view(b * t, -1, 4)

        if backbone_name in self.detectors:
            det_model = self.detectors[backbone_name]
            recon_dets = det_model(list(torch.clamp(recon_flat, 0.0, 1.0)))
            for f_idx, r_det in enumerate(recon_dets):
                person_mask = (r_det["labels"] == 1) & (r_det["scores"] >= 0.01)
                pred_boxes_list.append(r_det["boxes"][person_mask].detach().cpu())
                pred_scores_list.append(r_det["scores"][person_mask].detach().cpu())

                f_gt = gt_flat[f_idx].detach().cpu()
                gt_boxes_list.append(f_gt[f_gt[:, 0] >= 0.0])
                f_ig = ig_flat[f_idx].detach().cpu()
                ig_boxes_list.append(f_ig[f_ig[:, 0] >= 0.0])

        return {
            "feat_snr_db": round(float(np.mean(snr_list)), 3),
            "fpn_cos_pct": round(float(np.mean(cos_list)), 3),
            "is_detector": backbone_name in self.detectors,
            "pred_boxes": pred_boxes_list,
            "pred_scores": pred_scores_list,
            "gt_boxes": gt_boxes_list,
            "ignore_boxes": ig_boxes_list,
        }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-Shot Cross-Detector Real-x265 BDBR Evaluation")
    parser.add_argument("--data_root", type=str, default="./data/real_mot17_vis")
    parser.add_argument(
        "--tscope_ckpt",
        type=str,
        default="./outputs_rigor_gate0_gate1/tscope_autonomous_disjoint.pt",
    )
    parser.add_argument(
        "--spatial_ckpt",
        type=str,
        default="./outputs_rigor_gate0_gate1/spatial_scope_autonomous_disjoint.pt",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="./outputs_rigor_gate0_gate1/cross_backbone_bdbr.json",
    )
    parser.add_argument("--img_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Strictly Held-Out Test Split (MOT17-11, MOT17-13)
    dataset = RealVCMVideoDataset(
        data_root=args.data_root,
        shard_id=0,
        num_shards=1,
        clip_len=4,
        stride=4,
        img_size=args.img_size,
        is_train=False,
        enforce_disjoint_split=True,
    )
    test_seqs = sorted(set(c["seq_name"] for c in dataset.clips))
    print(
        f"[CrossDetector] Loaded {len(dataset)} held-out MOT17 test clips ({test_seqs}) from {args.data_root}"
    )

    tscope_ckpt_p = Path(args.tscope_ckpt)
    spatial_ckpt_p = Path(args.spatial_ckpt)
    if not tscope_ckpt_p.exists() or not spatial_ckpt_p.exists():
        raise FileNotFoundError(
            f"Disjoint autonomous checkpoints required: {tscope_ckpt_p} and {spatial_ckpt_p}"
        )

    model_tscope = TemporalSCOPE(
        base_ch=48, num_families=4, cond_dim=64, qg_size=16, max_delta_qp=6.0
    ).to(device)
    model_tscope.load_state_dict(
        torch.load(tscope_ckpt_p, map_location=device)["state_dict"], strict=True
    )
    model_tscope.eval()

    model_spatial = TemporalSCOPE(
        base_ch=48, num_families=4, cond_dim=64, qg_size=16, max_delta_qp=6.0
    ).to(device)
    model_spatial.load_state_dict(
        torch.load(spatial_ckpt_p, map_location=device)["state_dict"], strict=True
    )
    model_spatial.eval()

    codec = HybridX265STE(use_real_x265_in_forward=True, preset="medium").to(device)
    evaluator = CrossDetectorEvaluator(device)

    backbones = [
        ("fasterrcnn_r50_fpn_v2", "Faster R-CNN (ResNet50-FPN v2)", "Two-Stage Anchor Detector (Teacher)"),
        ("retinanet_r50_fpn_v2", "RetinaNet (ResNet50-FPN v2)", "One-Stage Focal-Loss Detector (Zero-Shot)"),
        ("fcos_r50_fpn", "FCOS (ResNet50-FPN)", "One-Stage Anchor-Free Detector (Zero-Shot)"),
        ("ssdlite_mobilenet_v3", "SSDLite (MobileNetV3-Large)", "Edge Mobile Detector (Zero-Shot)"),
        ("swin_t_transformer", "Swin-T (Swin Transformer)", "Vision Transformer Feature Backbone (Zero-Shot)"),
    ]
    qps = [27, 32, 37, 42, 47]
    methods = ["anchor_x265", "spatial_scope_framewise", "tscope_temporal_ctu_qp"]

    # Encode once per (clip, QP, method) with REAL libx265 + ffprobe pkt_size (ZERO GT at inference!)
    cached_pairs = []
    clean_feats_cache: Dict[Tuple[int, str], Dict[str, torch.Tensor]] = {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        video = sample["video"].unsqueeze(0).to(device)
        m_det = sample["m_det"].unsqueeze(0).to(device)
        gt_boxes = sample["gt_boxes"].unsqueeze(0).to(device)
        ignore_boxes = sample["ignore_boxes"].unsqueeze(0).to(device)
        b, t, c, h, w = video.shape
        clean_flat = video.view(b * t, c, h, w)
        for b_key, _, _ in backbones:
            clean_feats_cache[(idx, b_key)] = evaluator._extract_backbone_feats(b_key, clean_flat)

        for qp in qps:
            sev = torch.tensor([(qp - 22.0) / 25.0], device=device, dtype=torch.float32)
            fam = torch.tensor([0], device=device, dtype=torch.long)

            base_out = codec.evaluate_real_x265(video, qp, None)

            pred_spatial = model_spatial(video, fam, sev, disable_temporal=True)
            spatial_out = codec.evaluate_real_x265(pred_spatial["i_edit"], qp, None)

            pred_tscope = model_tscope(video, fam, sev, disable_temporal=False)
            tscope_out = codec.evaluate_real_x265(
                pred_tscope["i_edit"], qp, pred_tscope["ctu_qp_offsets"]
            )

            cached_pairs.append(
                {
                    "clip_id": idx,
                    "qp": qp,
                    "video": video,
                    "m_det": m_det,
                    "gt_boxes": gt_boxes,
                    "ignore_boxes": ignore_boxes,
                    "outs": {
                        "anchor_x265": base_out,
                        "spatial_scope_framewise": spatial_out,
                        "tscope_temporal_ctu_qp": tscope_out,
                    },
                }
            )

    summary_results: Dict[str, Any] = {
        "_provenance": {
            "held_out_test_sequences": test_seqs,
            "tscope_ckpt_sha256": sha256_file(tscope_ckpt_p),
            "spatial_ckpt_sha256": sha256_file(spatial_ckpt_p),
            "codec": "ffmpeg libx265 + ffprobe pkt_size + dynamic per-frame addroi",
            "gt_used_at_inference": False,
        }
    }

    print(
        "\n=== ZERO-SHOT CROSS-DETECTOR GENERALIZATION ON HELD-OUT MOT17-11 & MOT17-13 (REAL libx265 BYTES) ==="
    )
    print(
        f"{'Detector / Backbone':<32} | {'Spatial BDBR (FeatSNR)':<22} | {'T-SCOPE BDBR (FeatSNR)':<22} | {'T-SCOPE P-BDBR':<16} | {'Dataset GT AP@50 (Anchor->T-SCOPE)'}"
    )
    print("-" * 138)

    for b_key, b_label, b_paradigm in backbones:
        per_qp_data: Dict[str, Dict[int, Dict[str, Any]]] = {m: {} for m in methods}
        for m_name in methods:
            for q in qps:
                clip_items = [it for it in cached_pairs if it["qp"] == q]
                tot_bpps = []
                p_bpps = []
                snrs = []
                coss = []
                all_gt_b: List[torch.Tensor] = []
                all_ig_b: List[torch.Tensor] = []
                all_pred_b: List[torch.Tensor] = []
                all_pred_s: List[torch.Tensor] = []
                is_det = False

                for it in clip_items:
                    c_out = it["outs"][m_name]
                    tot_bpps.append(float(c_out["total_bpp"]))
                    p_bpps.append(float(c_out["p_frame_bpp"]))
                    res = evaluator.extract_clip_metrics_and_dets(
                        b_key,
                        it["video"],
                        c_out["video_recon"],
                        it["m_det"],
                        it["gt_boxes"],
                        it["ignore_boxes"],
                        f_clean_cached=clean_feats_cache.get((it["clip_id"], b_key)),
                    )
                    snrs.append(res["feat_snr_db"])
                    coss.append(res["fpn_cos_pct"])
                    is_det = res["is_detector"]
                    if is_det:
                        all_gt_b.extend(res["gt_boxes"])
                        all_ig_b.extend(res["ignore_boxes"])
                        all_pred_b.extend(res["pred_boxes"])
                        all_pred_s.extend(res["pred_scores"])

                if is_det:
                    coco_m = FrozenFPNDetectorTeacher.compute_dataset_level_coco_metrics(
                        all_gt_b, all_pred_b, all_pred_s, all_ignore_boxes=all_ig_b
                    )
                    gt_ap50 = coco_m["gt_ap50_pct"]
                    gt_ap50_95 = coco_m["gt_ap50_95_pct"]
                else:
                    gt_ap50 = float("nan")
                    gt_ap50_95 = float("nan")

                per_qp_data[m_name][q] = {
                    "total_bpp": float(np.mean(tot_bpps)),
                    "p_bpp": float(np.mean(p_bpps)),
                    "feat_snr_db": float(np.mean(snrs)),
                    "fpn_cos_pct": float(np.mean(coss)),
                    "gt_ap50_pct": gt_ap50,
                    "gt_ap50_95_pct": gt_ap50_95,
                }

        def curve(m_name: str):
            tr = [per_qp_data[m_name][q]["total_bpp"] for q in qps]
            pr = [per_qp_data[m_name][q]["p_bpp"] for q in qps]
            snr = [per_qp_data[m_name][q]["feat_snr_db"] for q in qps]
            ap50 = [per_qp_data[m_name][q]["gt_ap50_pct"] for q in qps]
            return tr, pr, snr, ap50

        ar, ap, asnr, aap = curve("anchor_x265")
        sr, sp, ssnr, sap = curve("spatial_scope_framewise")
        tr, tp, tsnr, tap = curve("tscope_temporal_ctu_qp")

        bdbr_spatial_tot = bjontegaard_delta_rate(ar, asnr, sr, ssnr)
        bdbr_spatial_p = bjontegaard_delta_rate(ap, asnr, sp, ssnr)
        bdbr_tscope_tot = bjontegaard_delta_rate(ar, asnr, tr, tsnr)
        bdbr_tscope_p = bjontegaard_delta_rate(ap, asnr, tp, tsnr)
        bdbr_tscope_gt_ap = (
            bjontegaard_delta_rate(ar, aap, tr, tap)
            if not any(np.isnan(aap))
            else float("nan")
        )

        summary_results[b_key] = {
            "label": b_label,
            "paradigm": b_paradigm,
            "is_object_detector": not any(np.isnan(aap)),
            "spatial_scope_total_bdbr_feat_snr_pct": round(bdbr_spatial_tot, 2),
            "spatial_scope_p_bdbr_feat_snr_pct": round(bdbr_spatial_p, 2),
            "tscope_total_bdbr_feat_snr_pct": round(bdbr_tscope_tot, 2),
            "tscope_p_bdbr_feat_snr_pct": round(bdbr_tscope_p, 2),
            "tscope_total_bdbr_gt_ap50_pct": round(bdbr_tscope_gt_ap, 2)
            if not np.isnan(bdbr_tscope_gt_ap)
            else None,
            "avg_gt_ap50_anchor_pct": round(float(np.mean(aap)), 2)
            if not any(np.isnan(aap))
            else None,
            "avg_gt_ap50_spatial_pct": round(float(np.mean(sap)), 2)
            if not any(np.isnan(sap))
            else None,
            "avg_gt_ap50_tscope_pct": round(float(np.mean(tap)), 2)
            if not any(np.isnan(tap))
            else None,
            "per_qp": {
                m_name: {str(q): per_qp_data[m_name][q] for q in qps}
                for m_name in methods
            },
        }
        ap_str = (
            f"{np.mean(aap):5.2f}% -> {np.mean(tap):5.2f}% (BDBR: {bdbr_tscope_gt_ap:+.2f}%)"
            if not any(np.isnan(aap))
            else "N/A (Feature-only backbone)"
        )
        print(
            f"{b_label:<32} | {bdbr_spatial_tot:+21.2f}% | {bdbr_tscope_tot:+21.2f}% | {bdbr_tscope_p:+15.2f}% | {ap_str}"
        )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    print(f"\nSaved cross-detector real-x265 generalization table -> {out_path}")


if __name__ == "__main__":
    main()
