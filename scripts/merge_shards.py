import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


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
    r1 = np.log10(np.maximum(np.array(rate_anchor, dtype=np.float64), 1e-8))
    a1 = np.array(acc_anchor, dtype=np.float64)
    r2 = np.log10(np.maximum(np.array(rate_test, dtype=np.float64), 1e-8))
    a2 = np.array(acc_test, dtype=np.float64)

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


def merge_checkpoints_soup(ckpt_dir: str, output_ckpt: str) -> None:
    """
    Aggregates trained shard checkpoints from multiple Kaggle accounts via
    Model Soup / Federated Averaging, strictly grouped by `variant` (F18 fix:
    never averages across different ablation variants).
    """
    files = sorted(
        f
        for f in glob.glob(str(Path(ckpt_dir) / "tscope_*_shard*.pt"))
        if "merged_soup" not in f
    )
    if not files:
        print(f"No shard .pt checkpoints found in {ckpt_dir}")
        return

    by_variant: Dict[str, List[str]] = {}
    for fpath in files:
        ckpt = torch.load(fpath, map_location="cpu")
        var_name = str(ckpt.get("variant", "full_tscope"))
        by_variant.setdefault(var_name, []).append(fpath)

    for var_name, v_files in sorted(by_variant.items()):
        avg_state: Dict[str, torch.Tensor] = {}
        for idx, fpath in enumerate(v_files):
            ckpt = torch.load(fpath, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            for k, v in state.items():
                if idx == 0:
                    avg_state[k] = v.clone().to(torch.float32)
                else:
                    avg_state[k] += v.to(torch.float32)

        for k in avg_state:
            avg_state[k] /= float(len(v_files))

        var_out = str(Path(ckpt_dir) / f"tscope_soup_{var_name}.pt")
        torch.save(
            {"variant": var_name, "merged_shards": len(v_files), "state_dict": avg_state},
            var_out,
        )
        print(f"Merged {len(v_files)} same-variant ({var_name}) checkpoints -> {var_out}")
        if var_name == "full_tscope":
            torch.save(
                {"variant": var_name, "merged_shards": len(v_files), "state_dict": avg_state},
                output_ckpt,
            )


def merge_eval_jsons_and_print_table(eval_dir: str) -> None:
    """
    Aggregates `eval_shard_*.json` across all Kaggle accounts, computes global
    I-frame bpp, P-frame bpp, total bpp, and BDBR (%) relative to `anchor_x265`.
    """
    files = sorted(glob.glob(str(Path(eval_dir) / "eval_shard_*.json")))
    if not files:
        print(f"No eval_shard_*.json files found in {eval_dir}")
        return

    all_records = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            all_records.extend(json.load(f))

    methods = sorted(set(r["method"] for r in all_records))
    qps = sorted(set(r["qp"] for r in all_records))

    summary: Dict[str, Dict[int, Dict[str, float]]] = {m: {} for m in methods}
    for m in methods:
        for qp in qps:
            subset = [r for r in all_records if r["method"] == m and r["qp"] == qp]
            if not subset:
                continue
            gt_aps = [
                r["gt_ap50_pct"]
                for r in subset
                if "gt_ap50_pct" in r and r["gt_ap50_pct"] is not None and not np.isnan(r["gt_ap50_pct"]) and r["gt_ap50_pct"] >= 0.0
            ]
            summary[m][qp] = {
                "total_bpp": float(np.mean([r["total_bpp"] for r in subset])),
                "i_frame_bpp": float(np.mean([r["i_frame_bpp"] for r in subset])),
                "p_frame_bpp": float(np.mean([r["p_frame_bpp"] for r in subset])),
                "task_roi_psnr_db": float(np.mean([r["task_roi_psnr_db"] for r in subset])),
                "gt_ap50_pct": float(np.mean(gt_aps)) if gt_aps else float("nan"),
                "map50_pct": float(np.mean(gt_aps)) if gt_aps else float("nan"),
            }

    anchor = "anchor_x265"
    anchor_rates = [summary[anchor][qp]["total_bpp"] for qp in qps]
    anchor_p_rates = [summary[anchor][qp]["p_frame_bpp"] for qp in qps]
    anchor_accs = [summary[anchor][qp]["task_roi_psnr_db"] for qp in qps]
    anchor_maps = [summary[anchor][qp]["map50_pct"] for qp in qps]

    print("\n=== AGGREGATED RESULTS ACROSS KAGGLE SHARDS (Wish-List #2 PoC) ===")
    print(
        f"{'Method':<28} | {'Total BDBR (%)':<15} | {'P-Frame BDBR (%)':<17} | {'mAP@50 BDBR (%)':<16} | {'Avg mAP@50 (%)':<14}"
    )
    print("-" * 102)
    for m in methods:
        m_rates = [summary[m][qp]["total_bpp"] for qp in qps]
        m_p_rates = [summary[m][qp]["p_frame_bpp"] for qp in qps]
        m_accs = [summary[m][qp]["task_roi_psnr_db"] for qp in qps]
        m_maps = [summary[m][qp]["map50_pct"] for qp in qps]
        bdbr_total = 0.0 if m == anchor else bjontegaard_delta_rate(anchor_rates, anchor_accs, m_rates, m_accs)
        bdbr_p = 0.0 if m == anchor else bjontegaard_delta_rate(anchor_p_rates, anchor_accs, m_p_rates, m_accs)
        bdbr_map = 0.0 if m == anchor else bjontegaard_delta_rate(anchor_rates, anchor_maps, m_rates, m_maps)
        avg_map = float(np.mean(m_maps))
        print(
            f"{m:<28} | {bdbr_total:+14.2f}% | {bdbr_p:+16.2f}% | {bdbr_map:+15.2f}% | {avg_map:13.2f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Kaggle Shard Checkpoints & Eval JSONs")
    parser.add_argument("--eval_dir", type=str, default="./outputs")
    parser.add_argument("--ckpt_dir", type=str, default="./outputs")
    parser.add_argument("--merged_ckpt", type=str, default="./outputs/tscope_merged_soup.pt")
    args = parser.parse_args()

    merge_eval_jsons_and_print_table(args.eval_dir)
    merge_checkpoints_soup(args.ckpt_dir, args.merged_ckpt)


if __name__ == "__main__":
    main()
