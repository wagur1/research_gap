#!/usr/bin/env python3
"""Plot only a schema-v2, final-ready merged benchmark; never legacy evidence.

No input/output defaults or copying into the manuscript are provided. Validation
is dependency-light so --validate-only and input tests do not render figures.
"""
import argparse
import json
import math
from pathlib import Path

EXPECTED_CRFS = [22, 27, 32, 37]
PRESERVED_CODEC_SETTINGS = {
    "qg_size": 16,
    "actual_roi_grid_max": 16,
    "positive_qoffset_cap": 0.14,
    "negative_qoffset_cap": -0.18,
}
CURVE_FIELDS = {
    "total_bpp": "total_bpp",
    "p_bpp": "p_frame_bpp",
    "roi_psnr": "roi_psnr_db",
    "gt_ap50": "gt_ap50_pct",
    "gt_ap95": "gt_ap50_95_pct",
}
BDBR_FIELDS = {
    "bdbr_tot_gt50": "bdbr_tot_gt_ap50_pct",
    "bdbr_p_gt50": "bdbr_p_gt_ap50_pct",
    "bdbr_tot_gt95": "bdbr_tot_gt_ap50_95_pct",
    "bdbr_p_gt95": "bdbr_p_gt_ap50_95_pct",
}
METHOD_LABELS = {
    "anchor_x265": "Anchor libx265",
    "deterministic_blur_ema": "Deterministic blur + EMA (non-learned)",
    "spatial_scope_framewise": "Framewise spatial (proxy-trained, no QG offsets)",
    "tscope_temporal_ctu_qp": "Full T-SCOPE (per-frame QG offsets)",
    "oracle_gt_mask_upper_bound": "Oracle GT mask (uses GT; not deployable)",
}
VARIANT_LABELS = {
    "wo_ctu_delta_qp": "T-SCOPE without QG offsets",
    "spatial_only_scope": "Spatial: framewise, proxy-trained, no QG offsets",
    "proxy_only_zhao": "Internal proxy-trained baseline (not prior-art reproduction)",
    "iwa_gop_qp_reuse": "T-SCOPE with I-frame QG map reuse",
    "full_tscope": "Full T-SCOPE (per-frame QG offsets)",
}
COLORS = ["#666666", "#E69F00", "#0072B2", "#D55E00", "#CC79A7"]
MARKERS = ["o", "v", "s", "D", "*"]


class PlotInputError(ValueError):
    """Evidence is not eligible for final figures."""


def _object(value, context):
    if not isinstance(value, dict):
        raise PlotInputError(f"{context}: expected an object")
    return value


def _finite(value, context):
    # JSON numeric strings, booleans, NaN and infinities are not measurements.
    if type(value) not in (int, float) or not math.isfinite(value):
        raise PlotInputError(f"{context}: expected a finite numeric measurement, got {value!r}")
    return float(value)


def validate_crfs(crfs):
    if (not isinstance(crfs, list) or any(type(q) is not int for q in crfs)
            or crfs != EXPECTED_CRFS):
        raise PlotInputError(f"crfs: expected the actual protocol {EXPECTED_CRFS}, got {crfs!r}")
    return list(crfs)


def extract_curve(summary_dict, method_key, qps):
    """Validate complete real curves and nullable BDBR; do not fit/fill data."""
    qps = validate_crfs(qps)
    summary_dict = _object(summary_dict, "summary")
    if method_key not in summary_dict:
        raise PlotInputError(f"summary: missing method {method_key!r}")
    method = _object(summary_dict[method_key], method_key)
    pqp = _object(method.get("per_qp"), f"{method_key}.per_qp")
    if set(pqp) != {str(q) for q in qps}:
        raise PlotInputError(f"{method_key}.per_qp: expected exactly string CRF keys {qps}")
    curve = {key: [] for key in CURVE_FIELDS}
    for q in qps:
        row = _object(pqp[str(q)], f"{method_key}.per_qp[{q}]")
        for target, field in CURVE_FIELDS.items():
            value = _finite(row.get(field), f"{method_key}.CRF{q}.{field}")
            if field in ("total_bpp", "p_frame_bpp") and value <= 0:
                raise PlotInputError(f"{method_key}.CRF{q}.{field}: bitrate must be positive")
            if field in ("gt_ap50_pct", "gt_ap50_95_pct") and not 0 <= value <= 100:
                raise PlotInputError(f"{method_key}.CRF{q}.{field}: AP must be in [0, 100]")
            curve[target].append(value)
    reasons = _object(method.get("bdbr_unavailable_reasons", {}),
                      f"{method_key}.bdbr_unavailable_reasons")
    curve["bdbr_unavailable_reasons"] = {}
    for target, field in BDBR_FIELDS.items():
        if field not in method:
            raise PlotInputError(f"{method_key}: missing {field}")
        value = method[field]
        if value is None:
            reason = reasons.get(field)
            if not isinstance(reason, str) or not reason.strip():
                raise PlotInputError(f"{method_key}.{field}: null BDBR requires an unavailable reason")
            curve[target] = None
            curve["bdbr_unavailable_reasons"][target] = reason.strip()
        else:
            curve[target] = _finite(value, f"{method_key}.{field}")
    return curve


def validate_codec_settings(bundle):
    configuration = _object(bundle.get("configuration"), "configuration")
    codec = _object(configuration.get("codec"), "configuration.codec")
    for key, expected in PRESERVED_CODEC_SETTINGS.items():
        actual = _finite(codec.get(key), f"configuration.codec.{key}")
        if actual != expected:
            raise PlotInputError(f"configuration.codec.{key}: expected preserved setting {expected}, got {actual}")
    return codec


def codec_caption(bundle):
    codec = validate_codec_settings(bundle)
    return (f"Model QG={codec['qg_size']} px; applied ROI grid <= "
            f"{codec['actual_roi_grid_max']}x{codec['actual_roi_grid_max']} regions; "
            f"qoffset caps [{codec['negative_qoffset_cap']:+.2f}, {codec['positive_qoffset_cap']:+.2f}]")


def validate_bundle(bundle):
    bundle = _object(bundle, "benchmark")
    if type(bundle.get("schema_version")) is not int or bundle["schema_version"] != 2:
        raise PlotInputError("benchmark: schema_version=2 is required; legacy summaries are not accepted")
    eligibility = _object(bundle.get("eligibility"), "eligibility")
    if eligibility.get("final_ready") is not True:
        raise PlotInputError("benchmark: eligibility.final_ready must be true; no partial/smoke-test figures")
    crfs = validate_crfs(bundle.get("crfs"))
    validate_codec_settings(bundle)
    # The merge contract defines overall_methods_summary as the full cohort only.
    # Never substitute an ablation shard summary for the full method curve.
    for section, required in (("overall_methods_summary", METHOD_LABELS),
                              ("ablation_variants_summary", VARIANT_LABELS)):
        summary = _object(bundle.get(section), section)
        for key in required:
            extract_curve(summary, key, crfs)
        for key in summary:
            extract_curve(summary, key, crfs)
    return bundle


def load_benchmark(path):
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            bundle = json.load(handle)
    except (OSError, ValueError) as exc:
        raise PlotInputError(f"Cannot read merged benchmark {path}: {exc}") from exc
    return validate_bundle(bundle)


def bdbr_text(curve, field):
    value = curve[field]
    if value is None:
        return "BDBR unavailable: " + curve["bdbr_unavailable_reasons"][field]
    return f"{value:+.2f}% BDBR"


def _style(plt):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "legend.fontsize": 7, "legend.frameon": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.15, "lines.linewidth": 1.8,
    })


def _save(fig, output_dir, stem, plt):
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png", dpi=300)
    plt.close(fig)


def render_figures(bundle, output_dir):
    # Validate everything before importing rendering libraries or creating output.
    validate_bundle(bundle)
    import matplotlib.pyplot as plt
    import numpy as np

    _style(plt)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    crfs = bundle["crfs"]
    curves = {k: extract_curve(bundle["overall_methods_summary"], k, crfs) for k in METHOD_LABELS}
    panels = [
        ("total_bpp", "gt_ap50", "bdbr_tot_gt50", "Total bitrate (bpp)", "GT AP@50 (%)"),
        ("p_bpp", "gt_ap50", "bdbr_p_gt50", "P-frame bitrate (bpp)", "GT AP@50 (%)"),
        ("total_bpp", "gt_ap95", "bdbr_tot_gt95", "Total bitrate (bpp)", "GT AP@[.50:.95] (%)"),
        ("p_bpp", "gt_ap95", "bdbr_p_gt95", "P-frame bitrate (bpp)", "GT AP@[.50:.95] (%)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, (xf, yf, bf, xl, yl) in zip(axes.flat, panels):
        for idx, (key, label) in enumerate(METHOD_LABELS.items()):
            curve = curves[key]
            ax.plot(curve[xf], curve[yf], marker=MARKERS[idx], color=COLORS[idx],
                    linestyle="--" if key == "anchor_x265" else "-",
                    label=f"{label}\n{bdbr_text(curve, bf)}")
            if key == "tscope_temporal_ctu_qp":
                for q, x, y in zip(crfs, curve[xf], curve[yf]):
                    ax.annotate(f"CRF {q}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.set_xlabel(xl + " (lower is better)")
        ax.set_ylabel(yl)
        ax.legend(loc="best")
    fig.suptitle("Full-cohort held-out rate-accuracy curves (merged final-ready evidence)\n" + codec_caption(bundle), fontsize=10)
    _save(fig, output_dir, "fig_bdbr_curves", plt)

    variants = {k: extract_curve(bundle["ablation_variants_summary"], k, crfs) for k in VARIANT_LABELS}
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for ax, (_, _, bf, xl, yl) in zip(axes.flat, panels):
        unavailable = []
        for idx, (key, label) in enumerate(VARIANT_LABELS.items()):
            curve = variants[key]
            value = curve[bf]
            if value is None:
                # No placeholder zero-height bar: unavailable is explicitly text.
                ax.text(idx, 0.02, "N/A", transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
                unavailable.append(f"{key}: {curve['bdbr_unavailable_reasons'][bf]}")
            else:
                ax.bar(idx, value, color=COLORS[idx], width=0.65)
                ax.annotate(f"{value:+.1f}%", (idx, value), xytext=(0, 3 if value >= 0 else -3),
                            textcoords="offset points", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xticks(np.arange(len(VARIANT_LABELS)))
        ax.set_xticklabels(list(VARIANT_LABELS.values()), rotation=20, ha="right", fontsize=7)
        ax.set_ylabel(f"{xl.split(' (')[0]} BDBR at {yl} (%)")
        ax.margins(y=0.2)
        if unavailable:
            ax.set_title("Unavailable BDBR\n" + "\n".join(unavailable), fontsize=7)
    fig.suptitle("Multi-factor ablations; separate variant cohorts, not recurrence-only comparisons\n" + codec_caption(bundle), fontsize=10)
    _save(fig, output_dir, "fig_ablation_bdbr", plt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True, help="Schema-v2 final-ready merged JSON")
    parser.add_argument("--output-dir", type=Path, help="Explicit local destination; no manuscript copying")
    parser.add_argument("--validate-only", action="store_true", help="Check input without rendering or writing")
    args = parser.parse_args(argv)
    try:
        bundle = load_benchmark(args.benchmark)
        if args.validate_only:
            print(f"Validated final-ready benchmark: {args.benchmark}; CRFs={bundle['crfs']}")
            return
        if args.output_dir is None:
            parser.error("--output-dir is required unless --validate-only")
        render_figures(bundle, args.output_dir)
        print(f"Saved local curves and ablation plots to {args.output_dir}")
    except PlotInputError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
