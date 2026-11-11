#!/usr/bin/env python3
"""
Generates IEEE TCSVT Rate-Distortion / Rate-Accuracy (BD-Rate) curves and
Ablation / Cross-Detector bar chart directly from the verified full-scale
Kaggle disjoint-split real-`libx265` evidence bundle
(`outputs_rigor_gate0_gate1/kaggle_21shards_disjoint_summary.json`)
(`R02` & `R11` fix: 100% real libx265 + ffprobe pkt_size + gt/gt.txt AP, zero GT at inference).
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
COLOR_PREONLY = "#2A9D8F"  # Teal (T-SCOPE w/o CTU Delta-QP)
COLOR_IWA = "#E9C46A"      # Gold (T-SCOPE IWA-GOP QP Reuse)
COLOR_OURS = "#E76F51"     # Burnt Coral (Learned Full T-SCOPE)
COLOR_ORACLE = "#6D597A"   # Muted Purple (Oracle GT-Mask Bound)


def extract_curve(summary_dict, method_key, qps):
    m = summary_dict[method_key]
    pqp = m["per_qp"]
    return {
        "total_bpp": np.array([pqp[str(q)]["total_bpp"] for q in qps], dtype=np.float64),
        "p_bpp": np.array([pqp[str(q)]["p_frame_bpp"] for q in qps], dtype=np.float64),
        "roi_psnr": np.array([pqp[str(q)]["roi_psnr_db"] for q in qps], dtype=np.float64),
        "gt_ap50": np.array([pqp[str(q)]["gt_ap50_pct"] for q in qps], dtype=np.float64),
        "gt_ap95": np.array([pqp[str(q)]["gt_ap50_95_pct"] for q in qps], dtype=np.float64),
        "bdbr_tot_gt50": m["bdbr_tot_gt_ap50_pct"],
        "bdbr_p_gt50": m["bdbr_p_gt_ap50_pct"],
        "bdbr_tot_gt95": m["bdbr_tot_gt_ap50_95_pct"],
        "bdbr_p_gt95": m["bdbr_p_gt_ap50_95_pct"],
    }


def main() -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_FIG_DIR.mkdir(parents=True, exist_ok=True)

    k_path = RIGOR_DIR / "kaggle_21shards_disjoint_summary.json"
    if not k_path.exists():
        raise FileNotFoundError(f"Missing Kaggle disjoint summary: {k_path}")

    bundle = json.load(open(k_path, "r", encoding="utf-8"))
    overall = bundle["overall_methods_summary"]
    ablation = bundle["ablation_variants_summary"]
    qps = [27, 32, 37, 42, 47]

    a = extract_curve(overall, "anchor_x265", qps)
    d = extract_curve(overall, "deterministic_blur_ema", qps)
    s = extract_curve(overall, "spatial_scope_framewise", qps)
    f = extract_curve(ablation, "full_tscope", qps)
    w = extract_curve(ablation, "wo_ctu_delta_qp", qps)
    o = extract_curve(overall, "oracle_gt_mask_upper_bound", qps)

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.8))
    plt.subplots_adjust(wspace=0.24, hspace=0.33)

    # -------------------------------------------------------------------------
    # Panel (a): Real libx265 Total Bitrate (BPP) vs. Ground-Truth AP@50 (%)
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    ax.plot(a["total_bpp"], a["gt_ap50"], "--o", color=COLOR_ANCHOR, label="1. Anchor libx265 (0.00% BDBR)")
    ax.plot(
        d["total_bpp"],
        d["gt_ap50"],
        ":v",
        color=COLOR_DET_EMA,
        label=f"2. Determ. Blur+EMA ({d['bdbr_tot_gt50']:+.2f}% BDBR)",
    )
    ax.plot(
        w["total_bpp"],
        w["gt_ap50"],
        ":^",
        color=COLOR_PREONLY,
        label=f"3. T-SCOPE w/o CTU ΔQP ({w['bdbr_tot_gt50']:+.2f}% BDBR)",
    )
    ax.plot(
        s["total_bpp"],
        s["gt_ap50"],
        "-.s",
        color=COLOR_SPATIAL,
        label=f"4. Learned Spatial SCOPE ({s['bdbr_tot_gt50']:+.2f}% BDBR)",
    )
    ax.plot(
        f["total_bpp"],
        f["gt_ap50"],
        "-D",
        color=COLOR_OURS,
        linewidth=2.4,
        label=f"5. Full T-SCOPE Ours ({f['bdbr_tot_gt50']:+.2f}% BDBR)",
    )
    ax.plot(
        o["total_bpp"],
        o["gt_ap50"],
        "--*",
        color=COLOR_ORACLE,
        alpha=0.82,
        label=f"6. Oracle GT-Mask Bound ({o['bdbr_tot_gt50']:+.2f}% BDBR)",
    )

    for qp, x_pt, y_pt in zip(qps, f["total_bpp"], f["gt_ap50"]):
        ax.annotate(
            f"CRF{qp}",
            (x_pt, y_pt),
            textcoords="offset points",
            xytext=(-16, 5),
            fontsize=7.2,
            color="#B23A22",
            weight="bold",
        )

    ax.set_title("(a) Real libx265 Total BPP vs. Pedestrian GT AP@50 (Held-Out MOT17-11 & 13)")
    ax.set_xlabel("Real libx265 Bitrate via ffprobe (bpp) $\\leftarrow$ Lower is Better")
    ax.set_ylabel("Ground-Truth Pedestrian AP@50 (%) $\\rightarrow$")
    ax.legend(loc="lower right")

    # -------------------------------------------------------------------------
    # Panel (b): Real libx265 P-Frame Bitrate vs. Ground-Truth AP@50 (%)
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    ax.plot(a["p_bpp"], a["gt_ap50"], "--o", color=COLOR_ANCHOR, label="1. Anchor libx265 (0.00% P-BDBR)")
    ax.plot(
        d["p_bpp"],
        d["gt_ap50"],
        ":v",
        color=COLOR_DET_EMA,
        label=f"2. Determ. Blur+EMA ({d['bdbr_p_gt50']:+.2f}% P-BDBR)",
    )
    ax.plot(
        w["p_bpp"],
        w["gt_ap50"],
        ":^",
        color=COLOR_PREONLY,
        label=f"3. T-SCOPE w/o CTU ΔQP ({w['bdbr_p_gt50']:+.2f}% P-BDBR)",
    )
    ax.plot(
        s["p_bpp"],
        s["gt_ap50"],
        "-.s",
        color=COLOR_SPATIAL,
        label=f"4. Learned Spatial SCOPE ({s['bdbr_p_gt50']:+.2f}% P-BDBR)",
    )
    ax.plot(
        f["p_bpp"],
        f["gt_ap50"],
        "-D",
        color=COLOR_OURS,
        linewidth=2.4,
        label=f"5. Full T-SCOPE Ours ({f['bdbr_p_gt50']:+.2f}% P-BDBR)",
    )
    ax.plot(
        o["p_bpp"],
        o["gt_ap50"],
        "--*",
        color=COLOR_ORACLE,
        alpha=0.82,
        label=f"6. Oracle GT-Mask Bound ({o['bdbr_p_gt50']:+.2f}% P-BDBR)",
    )

    ax.set_title("(b) Real libx265 Inter-Frame (P-Frame) BPP vs. GT AP@50")
    ax.set_xlabel("Real P-Frame Bitrate via ffprobe (bpp) $\\leftarrow$ Lower is Better")
    ax.set_ylabel("Ground-Truth Pedestrian AP@50 (%) $\\rightarrow$")
    ax.legend(loc="lower right")

    # -------------------------------------------------------------------------
    # Panel (c): Real libx265 Total Bitrate vs. Strict COCO GT AP@[.50:.95] (%)
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    ax.plot(a["total_bpp"], a["gt_ap95"], "--o", color=COLOR_ANCHOR, label="1. Anchor libx265 (0.00% BDBR)")
    ax.plot(
        w["total_bpp"],
        w["gt_ap95"],
        ":^",
        color=COLOR_PREONLY,
        label=f"3. T-SCOPE w/o CTU ΔQP ({w['bdbr_tot_gt95']:+.2f}% BDBR)",
    )
    ax.plot(
        s["total_bpp"],
        s["gt_ap95"],
        "-.s",
        color=COLOR_SPATIAL,
        label=f"4. Learned Spatial SCOPE ({s['bdbr_tot_gt95']:+.2f}% BDBR)",
    )
    ax.plot(
        f["total_bpp"],
        f["gt_ap95"],
        "-D",
        color=COLOR_OURS,
        linewidth=2.4,
        label=f"5. Full T-SCOPE Ours ({f['bdbr_tot_gt95']:+.2f}% BDBR)",
    )
    ax.plot(
        o["total_bpp"],
        o["gt_ap95"],
        "--*",
        color=COLOR_ORACLE,
        alpha=0.82,
        label=f"6. Oracle GT-Mask Bound ({o['bdbr_tot_gt95']:+.2f}% BDBR)",
    )

    ax.set_title("(c) Real libx265 Total BPP vs. Strict COCO GT AP@[.50:.95]")
    ax.set_xlabel("Real libx265 Bitrate via ffprobe (bpp) $\\leftarrow$ Lower is Better")
    ax.set_ylabel("Ground-Truth COCO AP@[.50:.95] (%) $\\rightarrow$")
    ax.legend(loc="lower right")

    # -------------------------------------------------------------------------
    # Panel (d): 5-Variant Ablation Study on Disjoint Held-Out MOT17-11 & 13
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    var_keys = [
        "wo_ctu_delta_qp",
        "spatial_only_scope",
        "proxy_only_zhao",
        "iwa_gop_qp_reuse",
        "full_tscope",
    ]
    var_labels = [
        "w/o CTU\nΔQP",
        "Spatial-Only\nSCOPE",
        "Proxy-Only\n(No STE)",
        "IWA-GOP\nQP Reuse",
        "Full T-SCOPE\n(Ours)",
    ]
    x = np.arange(len(var_keys))
    width = 0.26

    tot_ap50 = [ablation[k]["bdbr_tot_gt_ap50_pct"] for k in var_keys]
    p_ap50 = [ablation[k]["bdbr_p_gt_ap50_pct"] for k in var_keys]
    tot_ap95 = [ablation[k]["bdbr_tot_gt_ap50_95_pct"] for k in var_keys]

    b1 = ax.bar(x - width, tot_ap50, width, label="Total BDBR (GT AP@50)", color=COLOR_OURS)
    b2 = ax.bar(x, p_ap50, width, label="P-Frame BDBR (GT AP@50)", color=COLOR_SPATIAL, alpha=0.88)
    b3 = ax.bar(x + width, tot_ap95, width, label="Total BDBR (GT AP@[.5:.95])", color=COLOR_PREONLY)

    ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle="-")
    for bars in (b1, b2, b3):
        for bar in bars:
            h_val = bar.get_height()
            va = "bottom" if h_val >= 0 else "top"
            y_off = 0.5 if h_val >= 0 else -0.8
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h_val + y_off,
                f"{h_val:+.1f}%",
                ha="center",
                va=va,
                fontsize=6.5,
                weight="bold",
                color="#222222",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(var_labels, fontsize=8.3)
    ax.set_ylim(-32.0, 5.0)
    ax.set_title("(d) 5-Variant Ablation BDBR (%) on Held-Out MOT17-11 & 13 (Real libx265)")
    ax.set_ylabel("Real libx265 BDBR (%) $\\leftarrow$ Lower is Better")
    ax.legend(loc="lower left", ncol=1, fontsize=7.4)

    png_path = VIS_DIR / "fig_bdbr_curves.png"
    pdf_path = VIS_DIR / "fig_bdbr_curves.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    shutil.copyfile(png_path, DOCS_FIG_DIR / "fig_bdbr_curves.png")
    shutil.copyfile(pdf_path, DOCS_FIG_DIR / "fig_bdbr_curves.pdf")
    print(f"Saved full-scale disjoint real-x265 BD-Rate curves -> {png_path} and {pdf_path}")


if __name__ == "__main__":
    main()
