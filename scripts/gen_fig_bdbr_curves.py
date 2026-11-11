#!/usr/bin/env python3
"""
Generates IEEE TCSVT Rate-Distortion / Rate-Accuracy (BD-Rate) curves and
Zero-Shot Cross-Detector bar chart directly from the verified Gate 0 + Gate 1
real-`libx265` evidence bundle (`outputs_rigor_gate0_gate1/rigor_gate0_gate1_report.json`)
and real-`libx265` cross-detector bundle (`outputs_rigor_gate0_gate1/cross_backbone_bdbr.json`)
(`R02` & `R11` fix: zero proxy bpp, zero oracle mask leakage, zero synthetic mAP formulas).
"""
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
RIGOR_DIR = ROOT_DIR / "outputs_rigor_gate0_gate1"
VIS_DIR = ROOT_DIR / "outputs_kaggle_pool_full" / "visuals"
DOCS_FIG_DIR = ROOT_DIR / "docs" / "figures"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.8,
        "legend.fontsize": 7.8,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#E0E0E0",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    }
)

COLOR_ANCHOR = "#5C6B73"   # Slate Gray (Anchor libx265)
COLOR_DET_EMA = "#8E9AAF"  # Cool Gray (Deterministic Blur+EMA)
COLOR_SPATIAL = "#0072B2"  # Blue (Learned Spatial-Only SCOPE)
COLOR_PREONLY = "#2A9D8F"  # Teal (Learned T-SCOPE Pre-Edit Only)
COLOR_OURS = "#E76F51"     # Burnt Coral (Learned Full T-SCOPE)
COLOR_ORACLE = "#6D597A"   # Muted Purple (Oracle GT-Mask Bound)


def extract_curve(summary_dict, method_key, qps):
    m = summary_dict[method_key]
    pqp = m["per_qp"]
    return {
        "total_bpp": np.array([pqp[str(q)]["total_bpp"] for q in qps], dtype=np.float64),
        "p_bpp": np.array([pqp[str(q)]["p_frame_bpp"] for q in qps], dtype=np.float64),
        "roi_psnr": np.array([pqp[str(q)]["roi_psnr_db"] for q in qps], dtype=np.float64),
        "fpn_cos": np.array([pqp[str(q)]["fpn_cos_pct"] for q in qps], dtype=np.float64),
        "gt_ap50": np.array([pqp[str(q)]["dataset_gt_ap50_pct"] for q in qps], dtype=np.float64),
        "gt_ap95": np.array([pqp[str(q)]["dataset_gt_ap50_95_pct"] for q in qps], dtype=np.float64),
        "bdbr_tot_psnr": m["real_total_bdbr_roi_psnr_pct"],
        "bdbr_p_psnr": m["real_p_frame_bdbr_roi_psnr_pct"],
        "bdbr_gt_ap50": m["real_total_bdbr_gt_ap50_pct"],
        "bdbr_gt_ap95": m["real_total_bdbr_gt_ap50_95_pct"],
    }


def main() -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_FIG_DIR.mkdir(parents=True, exist_ok=True)

    report_path = RIGOR_DIR / "rigor_gate0_gate1_report.json"
    cross_path = RIGOR_DIR / "cross_backbone_bdbr.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing verified Gate 0/1 report: {report_path}")

    report = json.load(open(report_path, "r", encoding="utf-8"))
    g1 = report["gate1_summary"]
    qps = report["provenance"]["qps"]

    a = extract_curve(g1, "1_anchor_x265", qps)
    d = extract_curve(g1, "3_deterministic_blur_ema", qps)
    s = extract_curve(g1, "5_learned_spatial_scope", qps)
    w = extract_curve(g1, "6_learned_tscope_preedit_only", qps)
    f = extract_curve(g1, "7_learned_tscope_full", qps)
    o = extract_curve(g1, "8_oracle_gt_mask_bound", qps)

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.8))
    plt.subplots_adjust(wspace=0.24, hspace=0.33)

    # -------------------------------------------------------------------------
    # Panel (a): Real libx265 Total Bitrate (BPP) vs. Task ROI-PSNR (dB)
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    ax.plot(a["total_bpp"], a["roi_psnr"], "--o", color=COLOR_ANCHOR, label="1. Anchor libx265 (0.00%)")
    ax.plot(
        d["total_bpp"],
        d["roi_psnr"],
        ":v",
        color=COLOR_DET_EMA,
        label=f"3. Determ. Blur+EMA ({d['bdbr_tot_psnr']:+.2f}% BDBR)",
    )
    ax.plot(
        s["total_bpp"],
        s["roi_psnr"],
        "-.s",
        color=COLOR_SPATIAL,
        label=f"5. Learned Spatial SCOPE ({s['bdbr_tot_psnr']:+.2f}% BDBR)",
    )
    ax.plot(
        w["total_bpp"],
        w["roi_psnr"],
        ":^",
        color=COLOR_PREONLY,
        label=f"6. T-SCOPE Pre-Edit Only ({w['bdbr_tot_psnr']:+.2f}% BDBR)",
    )
    ax.plot(
        f["total_bpp"],
        f["roi_psnr"],
        "-D",
        color=COLOR_OURS,
        linewidth=2.4,
        label=f"7. Learned Full T-SCOPE ({f['bdbr_tot_psnr']:+.2f}% BDBR)",
    )
    ax.plot(
        o["total_bpp"],
        o["roi_psnr"],
        "--*",
        color=COLOR_ORACLE,
        alpha=0.8,
        label=f"8. Oracle GT-Mask Bound ({o['bdbr_tot_psnr']:+.2f}% BDBR)",
    )

    for qp, x_pt, y_pt in zip(qps, f["total_bpp"], f["roi_psnr"]):
        ax.annotate(
            f"CRF{qp}",
            (x_pt, y_pt),
            textcoords="offset points",
            xytext=(-14, 5),
            fontsize=7.2,
            color="#B23A22",
            weight="bold",
        )

    ax.set_title("(a) Real libx265 Total BPP vs. Task ROI-PSNR (Held-Out MOT17-11 & 13)")
    ax.set_xlabel("Real libx265 Bitrate via ffprobe (bpp) $\\leftarrow$ Lower is Better")
    ax.set_ylabel("Task Region PSNR inside $M_{\\mathrm{det}}$ (dB) $\\rightarrow$")
    ax.legend(loc="lower right")

    # -------------------------------------------------------------------------
    # Panel (b): Real libx265 P-Frame Bitrate vs. Task ROI-PSNR (dB)
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    ax.plot(a["p_bpp"], a["roi_psnr"], "--o", color=COLOR_ANCHOR, label="1. Anchor libx265 (0.00% P-BDBR)")
    ax.plot(
        d["p_bpp"],
        d["roi_psnr"],
        ":v",
        color=COLOR_DET_EMA,
        label=f"3. Determ. Blur+EMA ({d['bdbr_p_psnr']:+.2f}% P-BDBR)",
    )
    ax.plot(
        s["p_bpp"],
        s["roi_psnr"],
        "-.s",
        color=COLOR_SPATIAL,
        label=f"5. Learned Spatial SCOPE ({s['bdbr_p_psnr']:+.2f}% P-BDBR)",
    )
    ax.plot(
        w["p_bpp"],
        w["roi_psnr"],
        ":^",
        color=COLOR_PREONLY,
        label=f"6. T-SCOPE Pre-Edit Only ({w['bdbr_p_psnr']:+.2f}% P-BDBR)",
    )
    ax.plot(
        f["p_bpp"],
        f["roi_psnr"],
        "-D",
        color=COLOR_OURS,
        linewidth=2.4,
        label=f"7. Learned Full T-SCOPE ({f['bdbr_p_psnr']:+.2f}% P-BDBR)",
    )
    ax.plot(
        o["p_bpp"],
        o["roi_psnr"],
        "--*",
        color=COLOR_ORACLE,
        alpha=0.8,
        label=f"8. Oracle GT-Mask Bound ({o['bdbr_p_psnr']:+.2f}% P-BDBR)",
    )

    ax.set_title("(b) Real libx265 Inter-Frame (P-Frame) BPP vs. Task ROI-PSNR")
    ax.set_xlabel("Real P-Frame Bitrate via ffprobe (bpp) $\\leftarrow$ Lower is Better")
    ax.set_ylabel("Task Region PSNR inside $M_{\\mathrm{det}}$ (dB) $\\rightarrow$")
    ax.legend(loc="lower right")

    # -------------------------------------------------------------------------
    # Panel (c): Official Dataset-Level Ground-Truth AP@50 & AP@[.5:.95] vs. Real BPP
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    ap50_tag = f"{f['bdbr_gt_ap50']:+.2f}%" if f["bdbr_gt_ap50"] is not None else "N/A"
    s_ap50_tag = f"{s['bdbr_gt_ap50']:+.2f}%" if s["bdbr_gt_ap50"] is not None else "N/A"
    ax.plot(a["total_bpp"], a["gt_ap50"], "--o", color=COLOR_ANCHOR, label="1. Anchor libx265 (Dataset GT AP@50)")
    ax.plot(
        s["total_bpp"],
        s["gt_ap50"],
        "-.s",
        color=COLOR_SPATIAL,
        label=f"5. Spatial SCOPE (GT AP@50 BDBR: {s_ap50_tag})",
    )
    ax.plot(
        f["total_bpp"],
        f["gt_ap50"],
        "-D",
        color=COLOR_OURS,
        linewidth=2.4,
        label=f"7. Full T-SCOPE (GT AP@50 BDBR: {ap50_tag})",
    )
    ax.plot(
        o["total_bpp"],
        o["gt_ap50"],
        "--*",
        color=COLOR_ORACLE,
        alpha=0.8,
        label="8. Oracle GT-Mask (Dataset GT AP@50)",
    )

    ax.set_title("(c) Official Dataset-Level Pedestrian GT AP@50 vs. Real libx265 BPP")
    ax.set_xlabel("Real libx265 Bitrate via ffprobe (bpp) $\\leftarrow$ Lower is Better")
    ax.set_ylabel("Dataset-Level Ground-Truth AP@50 (%) $\\rightarrow$")
    ax.legend(loc="lower right")

    # -------------------------------------------------------------------------
    # Panel (d): Zero-Shot Cross-Detector Real-x265 Generalization BDBR (%)
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    if cross_path.exists():
        cross_bb = json.load(open(cross_path, "r", encoding="utf-8"))
        bb_keys = [
            "fasterrcnn_r50_fpn_v2",
            "retinanet_r50_fpn_v2",
            "fcos_r50_fpn",
            "ssdlite_mobilenet_v3",
            "swin_t_transformer",
        ]
        bb_short = [
            "Faster R-CNN\n(Teacher Det)",
            "RetinaNet\n(Zero-Shot Det)",
            "FCOS\n(Zero-Shot Det)",
            "SSDLite\n(Zero-Shot Det)",
            "Swin-T\n(Feat Backbone)",
        ]
        x = np.arange(len(bb_keys))
        width = 0.26

        s_vals = [cross_bb[k]["spatial_scope_total_bdbr_feat_snr_pct"] for k in bb_keys]
        t_tot_vals = [cross_bb[k]["tscope_total_bdbr_feat_snr_pct"] for k in bb_keys]
        t_p_vals = [cross_bb[k]["tscope_p_bdbr_feat_snr_pct"] for k in bb_keys]

        b1 = ax.bar(x - width, s_vals, width, label="Spatial SCOPE (Total BDBR)", color=COLOR_SPATIAL, alpha=0.85)
        b2 = ax.bar(x, t_tot_vals, width, label="Full T-SCOPE (Total BDBR)", color=COLOR_OURS)
        b3 = ax.bar(x + width, t_p_vals, width, label="Full T-SCOPE (P-Frame BDBR)", color=COLOR_PREONLY)

        ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle="-")
        for bars in (b1, b2, b3):
            for bar in bars:
                h_val = bar.get_height()
                va = "bottom" if h_val >= 0 else "top"
                y_off = 0.6 if h_val >= 0 else -1.0
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h_val + y_off,
                    f"{h_val:+.1f}%",
                    ha="center",
                    va=va,
                    fontsize=6.6,
                    weight="bold",
                    color="#222222",
                )
        ax.set_xticks(x)
        ax.set_xticklabels(bb_short, fontsize=8.2)
        min_v = min(min(s_vals), min(t_tot_vals), min(t_p_vals), -25.0)
        max_v = max(max(s_vals), max(t_tot_vals), max(t_p_vals), 10.0)
        ax.set_ylim(min_v - 8.0, max_v + 8.0)
        ax.set_title("(d) Zero-Shot Cross-Detector Real-x265 BDBR (%) (Held-Out Split)")
        ax.set_ylabel("Real libx265 BDBR (%) $\\leftarrow$ Lower is Better")
        ax.legend(loc="lower left", ncol=1, fontsize=7.4)

    png_path = VIS_DIR / "fig_bdbr_curves.png"
    pdf_path = VIS_DIR / "fig_bdbr_curves.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    shutil.copyfile(png_path, DOCS_FIG_DIR / "fig_bdbr_curves.png")
    shutil.copyfile(pdf_path, DOCS_FIG_DIR / "fig_bdbr_curves.pdf")
    print(f"Saved verified real-x265 BD-Rate curves -> {png_path} and {pdf_path}")


if __name__ == "__main__":
    main()
