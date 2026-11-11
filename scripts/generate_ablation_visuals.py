#!/usr/bin/env python3
"""Explicit, held-out pre-edit inspection with variant-specific schema-v2 soups.

This is not a decoded-codec/performance figure. No GT overlays, codec execution,
random/fallback checkpoints, manuscript copies or automatic example selection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.gen_fig_bdbr_curves import (
    PlotInputError, VARIANT_LABELS, codec_caption, load_benchmark,
)

# Preserved experimental settings, not a proposed retraining configuration.
# Spatial changes temporal inputs/EMA, temporal losses, STE and QG together.
VARIANT_SETTINGS = {
    "full_tscope": {"lambda_temp_bg": 1.0, "lambda_temp_obj": 0.5, "max_delta_qp": 4.0,
                    "use_real_x265_ste": True, "reuse_iframe_qp": False},
    "spatial_only_scope": {"lambda_temp_bg": 0.0, "lambda_temp_obj": 0.0, "max_delta_qp": 0.0,
                           "use_real_x265_ste": False, "reuse_iframe_qp": False},
    "wo_ctu_delta_qp": {"lambda_temp_bg": 1.0, "lambda_temp_obj": 0.5, "max_delta_qp": 0.0,
                        "use_real_x265_ste": True, "reuse_iframe_qp": False},
    "iwa_gop_qp_reuse": {"lambda_temp_bg": 1.0, "lambda_temp_obj": 0.5, "max_delta_qp": 4.0,
                         "use_real_x265_ste": True, "reuse_iframe_qp": True},
    "proxy_only_zhao": {"lambda_temp_bg": 0.2, "lambda_temp_obj": 0.1, "max_delta_qp": 0.0,
                        "use_real_x265_ste": False, "reuse_iframe_qp": False},
}
MODEL_SETTINGS = {"base_ch": 48, "num_families": 4, "cond_dim": 64, "qg_size": 16,
                  "max_obj_res": 0.09, "max_bg_res": 0.03}


def validate_soup_metadata(checkpoint, expected_variant):
    """Reject raw/shard checkpoints, wrong variants and absent soup provenance."""
    if expected_variant not in VARIANT_SETTINGS:
        raise PlotInputError(f"Unknown variant {expected_variant!r}")
    if not isinstance(checkpoint, dict) or type(checkpoint.get("schema_version")) is not int or checkpoint["schema_version"] != 2:
        raise PlotInputError("Qualitative checkpoints must be schema-version-2 soups")
    if checkpoint.get("variant") != expected_variant:
        raise PlotInputError(f"Soup variant mismatch: expected {expected_variant}, got {checkpoint.get('variant')!r}")
    count = checkpoint.get("merged_shards")
    members = checkpoint.get("constituents")
    if type(count) is not int or count < 1 or not isinstance(members, list) or len(members) != count:
        raise PlotInputError(f"{expected_variant}: merged_shards/constituents soup provenance is missing or inconsistent")
    if any(not isinstance(member, (str, dict)) or not member for member in members):
        raise PlotInputError(f"{expected_variant}: invalid soup constituent provenance")
    if len({json.dumps(member, sort_keys=True) for member in members}) != count:
        raise PlotInputError(f"{expected_variant}: duplicate soup constituents")
    validation = checkpoint.get("validation")
    if not isinstance(validation, dict) or validation.get("status") not in ("not_evaluated", "validated"):
        raise PlotInputError(f"{expected_variant}: soup validation.status must be not_evaluated or validated")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise PlotInputError(f"{expected_variant}: nonempty soup state_dict is required")
    # Soup variant metadata selects the approved preserved configuration above.
    # If configuration is included, it must not contradict those settings.
    configuration = checkpoint.get("configuration")
    if configuration is not None:
        if not isinstance(configuration, dict):
            raise PlotInputError(f"{expected_variant}: configuration must be an object")
        expected_model = dict(MODEL_SETTINGS, max_delta_qp=VARIANT_SETTINGS[expected_variant]["max_delta_qp"])
        for section, expected in (("model", expected_model), ("ablation", VARIANT_SETTINGS[expected_variant])):
            recorded = configuration.get(section, {})
            if not isinstance(recorded, dict):
                raise PlotInputError(f"{expected_variant}: configuration.{section} must be an object")
            for key, value in expected.items():
                if key in recorded and (type(recorded[key]) is not type(value) or recorded[key] != value):
                    raise PlotInputError(f"{expected_variant}: configuration.{section}.{key} contradicts preserved settings")
    return checkpoint


def load_soup_model(path, variant, *, torch_module=None, model_factory=None):
    path = Path(path)
    if not path.is_file():
        raise PlotInputError(f"Missing explicit soup checkpoint for {variant}: {path}")
    if torch_module is None:
        import torch as torch_module
    if model_factory is None:
        from models.temporal_scope import TemporalSCOPE
        model_factory = TemporalSCOPE
    try:
        checkpoint = torch_module.load(path, map_location="cpu", weights_only=True)
        validate_soup_metadata(checkpoint, variant)
        state = checkpoint["state_dict"]
        for key, value in state.items():
            if not isinstance(key, str) or not torch_module.is_tensor(value):
                raise PlotInputError(f"{variant}: state_dict must contain named tensors")
            if not torch_module.isfinite(value).all().item():
                raise PlotInputError(f"{variant}: nonfinite checkpoint tensor {key}")
        model = model_factory(**MODEL_SETTINGS, max_delta_qp=VARIANT_SETTINGS[variant]["max_delta_qp"])
        model.load_state_dict(state, strict=True)
        model.eval()
    except PlotInputError:
        raise
    except Exception as exc:
        raise PlotInputError(f"Cannot safely load {variant} soup {path}: {exc}") from exc
    return model, checkpoint


def parse_checkpoints(values):
    result = {}
    for value in values:
        variant, sep, path = value.partition("=")
        if not sep or variant not in VARIANT_SETTINGS or not path.strip():
            raise PlotInputError("--checkpoint requires VARIANT=PATH with a known ablation variant")
        if variant in result:
            raise PlotInputError(f"Duplicate checkpoint argument for {variant}")
        result[variant] = Path(path)
    if not result:
        raise PlotInputError("At least one explicit --checkpoint VARIANT=PATH is required")
    if len({path.resolve() for path in result.values()}) != len(result):
        raise PlotInputError("Each displayed variant needs its own soup checkpoint, not a shared full model")
    return result


def _frame_number(path):
    stem = Path(path).stem
    if not stem.isdecimal():
        raise PlotInputError(f"Ambiguous MOT frame identity: {path}")
    return int(stem)


def select_heldout_clip(clips, sequence, clip_start_frame, frame_number, clip_len):
    """Address identity, never ds[0] or dataset ordering-dependent clip IDs."""
    if not re.fullmatch(r"MOT17-(11|13)(-FRCNN)?", sequence):
        raise PlotInputError(f"Sequence must explicitly be a held-out MOT17-11/13 FRCNN video, got {sequence!r}")
    if any(type(v) is not int or v < 1 for v in (clip_start_frame, frame_number, clip_len)):
        raise PlotInputError("Clip start, selected frame and clip length must be positive integers")
    matches = []
    for index, clip in enumerate(clips):
        if clip.get("seq_name") != sequence or clip.get("type") != "mot_frames":
            continue
        paths = clip.get("frame_paths", [])
        numbers = [_frame_number(path) for path in paths]
        if numbers and numbers[0] == clip_start_frame:
            if numbers != list(range(clip_start_frame, clip_start_frame + clip_len)):
                raise PlotInputError("Selected held-out clip has incomplete/nonconsecutive frame identities")
            if frame_number not in numbers:
                raise PlotInputError("Selected frame is not in the explicitly selected clip")
            matches.append((index, numbers.index(frame_number), paths))
    if len(matches) != 1:
        raise PlotInputError(f"Expected one clip for {sequence} starting at frame {clip_start_frame}; found {len(matches)}")
    return matches[0]


def tensor_to_pil(img_t):
    import numpy as np
    from PIL import Image
    # Only display conversion quantization; no added noise, GT outlines or edits.
    arr = (img_t.detach().cpu().permute(1, 2, 0).numpy() * 255.0).round()
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True, help="Final-ready merged schema-v2 bundle")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Explicit local destination (no manuscript copying)")
    parser.add_argument("--checkpoint", action="append", required=True, metavar="VARIANT=PATH")
    parser.add_argument("--sequence", required=True, help="Exact held-out sequence directory name")
    parser.add_argument("--clip-start-frame", type=int, required=True, help="1-based MOT source frame identity")
    parser.add_argument("--frame-number", type=int, required=True, help="1-based MOT source frame to display")
    parser.add_argument("--crf", type=int, required=True, help="Must be in the benchmark crfs")
    parser.add_argument("--clip-len", type=int, default=4, help="Preserved evaluation clip length")
    parser.add_argument("--img-size", type=int, default=256, help="Preserved square evaluation resize, not native resolution")
    parser.add_argument("--validate-only", action="store_true", help="Check inputs/soup weights/identity; no inference or figures")
    args = parser.parse_args(argv)
    try:
        bundle = load_benchmark(args.benchmark)
        if args.crf not in bundle["crfs"]:
            raise PlotInputError(f"CRF {args.crf} is absent from the benchmark protocol {bundle['crfs']}")
        if args.clip_len != 4 or args.img_size != 256:
            raise PlotInputError("Approved preserved evaluation settings require clip_len=4 and img_size=256; no silent resize change")
        if not args.data_root.is_dir():
            raise PlotInputError(f"Missing real held-out data directory: {args.data_root}")
        paths = parse_checkpoints(args.checkpoint)
        if not args.validate_only and args.output_dir is None:
            raise PlotInputError("--output-dir is required unless --validate-only")
        import torch
        from vcm_datasets.vcm_video_datasets import RealVCMVideoDataset
        from PIL import Image, ImageDraw, ImageFont
        ds = RealVCMVideoDataset(data_root=str(args.data_root), shard_id=0, num_shards=1,
                                 clip_len=args.clip_len, stride=args.clip_len, img_size=args.img_size,
                                 is_train=False, enforce_disjoint_split=True, split="test")
        index, time_index, source_paths = select_heldout_clip(
            ds.clips, args.sequence, args.clip_start_frame, args.frame_number, args.clip_len)
        for path in source_paths:
            if not Path(path).is_file():
                raise PlotInputError(f"Missing selected source frame: {path}")
        sample = ds[index]
        if sample["seq_name"] != args.sequence:
            raise PlotInputError("Dataset returned a different sequence from the selected identity")
        video = sample["video"].unsqueeze(0)
        if tuple(video.shape) != (1, args.clip_len, 3, args.img_size, args.img_size) or not torch.isfinite(video).all().item():
            raise PlotInputError("Held-out video shape/finite-data validation failed")
        if video.min().item() < 0 or video.max().item() > 1:
            raise PlotInputError("Held-out video must be in [0, 1]")
        loaded = {variant: load_soup_model(path, variant) for variant, path in paths.items()}
        benchmark_hashes = bundle.get("checkpoint_hashes", {})
        for variant, path in paths.items():
            if _sha256(path) != benchmark_hashes.get(variant):
                raise PlotInputError(f"{variant}: qualitative checkpoint differs from evaluated benchmark weights")
        if any(frame not in bundle.get("expected_frame_ids", []) for frame in sample["frame_ids"]):
            raise PlotInputError("Selected qualitative clip was not part of the evaluated frame manifest")
        if args.validate_only:
            print(f"Validated soups and held-out {args.sequence} frame {args.frame_number}; no inference/rendering")
            return
        columns = [("Original (evaluation resize)\nNo GT overlay; no codec", tensor_to_pil(video[0, time_index]), None)]
        provenance = {
            "benchmark": str(args.benchmark.resolve()), "sequence": args.sequence,
            "clip_start_frame": args.clip_start_frame, "frame_number": args.frame_number,
            "source_frames": [{"path": str(Path(path).resolve()), "sha256": _sha256(path)} for path in source_paths],
            "crf": args.crf, "clip_len": args.clip_len, "img_size": args.img_size,
            "display": "pre-edit only; not decoded; no bitrate/AP claim; no GT at model inference",
            "codec_settings": bundle["configuration"]["codec"], "checkpoints": {},
        }
        with torch.no_grad():
            fam = torch.tensor([0], dtype=torch.long)
            severity = torch.tensor([(args.crf - 22.0) / 25.0], dtype=torch.float32)
            for variant, (model, checkpoint) in loaded.items():
                settings = VARIANT_SETTINGS[variant]
                pred = model(video, fam, severity, reuse_iframe_qp=settings["reuse_iframe_qp"],
                             disable_temporal=(variant == "spatial_only_scope"))
                edit = pred["i_edit"]
                if edit.shape != video.shape or not torch.isfinite(edit).all().item() or edit.min().item() < 0 or edit.max().item() > 1:
                    raise PlotInputError(f"{variant}: invalid pre-edit output; refusing a misleading figure")
                # Shared scale and source reference, not temporal residuals or decoded distortion.
                difference = (edit[0, time_index] - video[0, time_index]).abs().mean(dim=0) * 8.0
                diff_rgb = difference.unsqueeze(0).expand(3, -1, -1)
                title = VARIANT_LABELS[variant] + f"\nSoup status: {checkpoint['validation']['status']}"
                columns.append((title, tensor_to_pil(edit[0, time_index]), tensor_to_pil(diff_rgb)))
                provenance["checkpoints"][variant] = {
                    "path": str(paths[variant].resolve()), "sha256": _sha256(paths[variant]),
                    "validation": checkpoint["validation"], "merged_shards": checkpoint["merged_shards"],
                    "constituents": checkpoint["constituents"], "preserved_settings": settings,
                }
        # Outputs are created only after all input and inference checks pass.
        import textwrap
        size, pad, left = args.img_size, 12, 200
        font = ImageFont.load_default(size=13)
        canvas_width = left + len(columns) * (size + pad) + pad
        caption = (f"{args.sequence}, source frame {args.frame_number}; clip starts {args.clip_start_frame}; "
                   f"CRF condition {args.crf}; resized {size}x{size}\nPre-edit inspection only; not decoded; no performance claim\n"
                   + codec_caption(bundle))
        caption = "\n".join(textwrap.fill(line, width=max(40, canvas_width // 8)) for line in caption.splitlines())
        titles = ["\n".join(textwrap.fill(line, width=29) for line in title.splitlines())
                  for title, _, _ in columns]
        title_y = 16 + len(caption.splitlines()) * 18
        header = title_y + max(len(title.splitlines()) for title in titles) * 18 + pad
        canvas = Image.new("RGB", (canvas_width, header + 2 * (size + pad) + pad), "white")
        draw = ImageDraw.Draw(canvas)
        draw.multiline_text((10, 8), caption, fill="black", font=font)
        draw.multiline_text((10, header + size // 2), "Model pre-edit\n(or original)\nEvaluation resize", fill="black", font=font)
        draw.multiline_text((10, header + size + pad + size // 2), "Mean abs edit-source x8\nShared grayscale [0,1]\nDisplay saturates at 1\nOriginal: N/A", fill="black", font=font)
        for idx, (_, preedit, difference) in enumerate(columns):
            x = left + idx * (size + pad)
            draw.multiline_text((x, title_y), titles[idx], fill="black", font=font)
            canvas.paste(preedit, (x, header))
            if difference is not None:
                canvas.paste(difference, (x, header + size + pad))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        canvas.save(args.output_dir / "ablation_preedit_comparison_grid.png")
        with (args.output_dir / "ablation_preedit_provenance.json").open("w", encoding="utf-8") as handle:
            json.dump(provenance, handle, indent=2, allow_nan=False)
        print(f"Saved pre-edit-only inspection and source provenance to {args.output_dir}")
    except PlotInputError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
