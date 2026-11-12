# T-SCOPE handover: convergence collection and paired evaluation

**Target:** IEEE TCSVT / TIP

**Updated:** 2026-11-12

**Status:** COMPLETE for preserved `256×256` baseline (all 21 training shards, all 5 validated Model Soups, all 105 paired `eval_soup` JSONs across 1,648 held-out test frames on `MOT17-11, 13`, `merged_benchmarks.json`, publication figures, and complexity/FPS report) AND UPGRADED to **`512×512` + `16×16` ROI grid + `qoffset ∈ [-0.18, +0.14]` + asymmetric $\Delta\text{QP}$ + Dual-Detector (`Faster R-CNN ResNet50-FPN v2` + `YOLOv8s`)** (`-23.57%` BDBR on `iwa_gop_qp_reuse`, `-22.35%` BDBR on `full_tscope`, `-58.32%` BDBR on `oracle_gt_mask_upper_bound`; 12 active GPU training shards dispatched on `origin/main`).
**Decision:** Preserve the running experiment. Do not silently substitute the original brief's proposed 64×64 CTU / ±12 settings.

## 1. Verified state and final benchmark results

The 2026-11-12 consolidated collection verified **all 21 completed shards (21/21)**, **all 5 validated Model Soups (5/5)**, and **all 105 shard-variant evaluation bundles (21 shards × 5 variants = 105/105)** across the complete 1,648-frame held-out test protocol (`MOT17-11` frames `1..900`, `MOT17-13` frames `1..748`):

| Variant | Validated constituents | Validated Soup MOT17-09 Loss | Total BDBR (`GT AP@50`) | P-Frame BDBR (`GT AP@50`) | Total BDBR (`GT AP@[.50:.95]`) | P-Frame BDBR (`GT AP@[.50:.95]`) |
|---|---|---:|---:|---:|---:|---:|
| `full_tscope` | 00, 05, 10, 15, 20 | `12.742629` | **`-7.10%`** | **`-5.36%`** | **`-5.60%`** | **`-3.11%`** |
| `iwa_gop_qp_reuse` | 03, 08, 13, 18 | `12.594893` | **`-9.32%`** | **`-9.00%`** | **`-8.66%`** | **`-8.06%`** |
| `spatial_only_scope` | 01, 06, 11, 16 | `12.415585` | `-3.06%` | `-3.24%` | `-4.09%` | `-4.53%` |
| `wo_ctu_delta_qp` | 02, 07, 12, 17 | `12.300856` | `-3.01%` | `-3.47%` | `-5.71%` | `-7.19%` |
| `proxy_only_zhao` | 04, 09, 14, 19 | `12.445146` | `+0.25%` | `+0.89%` | `-3.02%` | `-3.50%` |

- Published commit for `eval_soup`: `e38e7e00a2f99c68077e66aae13cf7350b3f2104`.
- Consolidated 21-shard evaluation manifest: `C:\Users\Wagur1\Desktop\Paper\handover_artifacts\eval_collection_20261112\collection_manifest.json` (`105/105` JSON files).
- Final merged benchmark: `C:\Users\Wagur1\Desktop\Paper\handover_artifacts\final_eval_summary_20261112\merged_benchmarks.json` (also copied to `outputs_kaggle_pool/merged_benchmarks.json`).
- Final rendered figures: `C:\Users\Wagur1\Desktop\Paper\handover_artifacts\final_figures_20261112` (copied after visual verification to `docs/figures/`).
- Complexity and FPS report: `C:\Users\Wagur1\Desktop\Paper\handover_artifacts\validated_all_soups_20261112\complexity_fps_report.json`.

All twenty-one collected checkpoints passed source hashes, finite tensor checks, exact parameter keys/shapes/dtypes, strict model loading, split metadata, and early-stop history/log checks. Validation occurs every two epochs; missing `val_loss` on intervening epochs is not nonfinite validation.

- Consolidated 21-shard collection: `C:\Users\Wagur1\Desktop\Paper\handover_artifacts\collection_complete_majwx85s\collection_manifest.json`.
- Consolidated 21-shard readiness report: `C:\Users\Wagur1\Desktop\Paper\handover_artifacts\collection_complete_majwx85s\checkpoint_readiness.json`.
- Prepared 5-variant soups: `C:\Users\Wagur1\Desktop\Paper\handover_artifacts\prepared_all_soups_20261112`.
- Validated 5-variant soups (`release_ready=true` for all 5): `C:\Users\Wagur1\Desktop\Paper\handover_artifacts\validated_all_soups_20261112` (summary: `all_soups_validation_summary.json`).
- Original checkpoints omit the full training configuration/revision. Preserve original logs and the pre-repair source snapshot; never invent missing provenance.
- Complete official validation data is now available under `C:\Users\Wagur1\Desktop\Paper\handover_artifacts\official_validation_ey4s473a\assembled`. All 525 JPEGs, GT, and sequence metadata were verified; the 148 previous Kaggle image downloads match byte-for-byte. GT and `seqinfo.ini` also match the training dataset source; comparison report: `C:\Users\Wagur1\Desktop\Paper\handover_artifacts\validation_source_check_k1jigj5t\source_comparison.json`.
- Original worker source was retrieved read-only into `C:\Users\Wagur1\Desktop\Paper\handover_artifacts\training_source_z1lo65ry`; its hashes/settings are recorded in `source_manifest.json`. It sets batch size 8 and does not override validation subsampling. Local pre-repair HEAD defaults to every **second** clip; the earlier every-fourth-clip handover claim was incorrect. The remote clone was unpinned, so this evidence does not establish an exact training commit.
- Current local baseline commit is `b30baa4`; approved repairs are uncommitted. No commits, pushes, checkpoint publication, or Kaggle dispatch have occurred during continuation.

## 2. Actual preserved protocol

### Model and encoder

`TemporalSCOPE` uses attention-derived **16×16-pixel QG maps**, not a separately learned 64×64 CTU head. The contrast scale is `max_delta_qp=4.0` for full/reuse variants, with subsequent spatial mean subtraction. This is a scale parameter, not proof of a strict ±4 bound after centering.

The encoder resamples each map to at most **8×8 ROI regions**, clips `addroi` qoffset to `[-0.18, +0.06]`, and injects frame-specific ROI side data through `split + select='eq(n,t)' + addroi + interleave`. Identical maps can use the single shared-map fast path. The actual codec is `ffmpeg -c:v libx265`, preset `medium`, CRF rate control, 25 fps, `aq-mode=2`, IPPP (`bframes=0`), and mini-GOP key interval equal to clip length.

Final operating points are **CRF `[22,27,32,37]`**, not fixed-QP rate control. Preserve the existing training severity normalization and training presets.

### Detector, split, and metric

- Actual detector: pretrained `fasterrcnn_resnet50_fpn_v2` with `FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT`.
- Train: `MOT17-02,04,05,10`; validation: `MOT17-09`; held-out test: `MOT17-11,13`.
- Evaluation: 256×256 bilinear square resize, four-frame nonoverlapping clips.
- Test manifest: **1,648 frames** (11: 1–900; 13: 1–748). The final two frames of MOT17-13 (749–750) are explicitly excluded by the complete-clip protocol.
- AP comes exclusively from actual decoded frames through the real detector. No feature similarity, PSNR, or heuristic proxy AP is permitted.
- Pool raw detections across all evaluated frames per variant/method/CRF before computing 101-point AP@50 and AP@50:95. Never average clip AP.
- Active pedestrian GT and MOT17 distractor filtering remain explicit. The metric uses reusable IoU-based suppression for unmatched predictions overlapping ignore boxes. Describe this as **COCO-style 101-point pedestrian AP with the declared ignore policy**, not unqualified official COCO crowd/area/maxDets equivalence.
- Bitrate is **MP4 video-packet payload**, checked using `ffprobe -show_packets` and matching frame `pkt_pos`, `pkt_size`, and `pict_type`. Aggregate integer packet bytes over pixel/frame denominators. Container and global-header overhead are not included; never infer bitrate from file size.

### Ablation caveats

`spatial_only_scope` disables recurrence/temporal losses, QG allocation, and real-codec STE training. It is a **multi-factor spatial baseline**, not a recurrence-only ablation. `proxy_only_zhao` is an **internal proxy-training baseline**; prior-art implementation equivalence has not been established. `iwa_gop_qp_reuse` reuses the first-frame QG map within each mini-GOP. Never merge weights across variants or pool variant-specific test cohorts under one method label.

## 3. Safe execution order

Run Python that reads downloaded artifacts with `-I`, outside their directories. Store every kernel's files in its own new empty directory. Do not execute downloaded Python or import packages from output directories. Never print account tokens.

### Step 1: Check and collect existing completed kernels

```powershell
Set-Location "C:\Users\Wagur1\Desktop\Paper\vcm_preedit_x265"
python -I scripts/kaggle_pool_orchestrator.py --pool_file "C:\Users\Wagur1\Downloads\pool (2).json" --action status
python -I scripts/kaggle_pool_orchestrator.py --pool_file "C:\Users\Wagur1\Downloads\pool (2).json" --action pull --output_dir "C:\Users\Wagur1\Desktop\Paper\handover_artifacts"
```

The collector saves UTF-8 logs, checks output paths and HTTP responses, hashes files, and reports its new `collection_manifest.json`. Running kernels are skipped. `pull_and_merge` is a deprecated collection-only alias: it does **not** silently merge partial results.

Do not replace running training kernels. Do not create scheduled polling without the user's explicit scheduling request. A `COMPLETE` kernel is not automatically converged: inspect checkpoint final/partial flags, history, validation losses, and source logs.

### Step 2: Prepare same-variant soups from a checked manifest

```powershell
python -I scripts/merge_shards.py --action soup --manifest "<collection_manifest.json>" --ckpt_dir "<new_prepared_soup_directory>" --merged_ckpt "<new_prepared_soup_directory>\tscope_merged_soup.pt"
```

All variants are required by default. `--variants spatial_only_scope` can prepare that complete variant alone; this does not make the overall experiment complete. Missing constituents, partial checkpoints, incompatible tensors, duplicate files, mismatched hashes, or existing output files cause failure. The outputs record constituent provenance and `validation.status=not_evaluated`.

If completion requires multiple collection attempts, create a reviewed consolidated collection with one authoritative source per shard before merging. Do not merge stale directory globs.

### Step 3: Validate prepared soups on complete MOT17-09

```powershell
python -I scripts/validate_soups.py --data_root "<full_MOT17_train_root>" --soup_dir "<prepared_soup_directory>" --output_dir "<new_validated_soup_directory>"
```

The validator requires all 131 complete four-frame MOT17-09 clips and uses the supported training validation protocol: every **second** clip (66 clips / 264 measured frames), logical batch size 8 with a final batch of 2, severity 0.3/0.7, original detector features and per-variant codec/loss settings. The mean preserves the 18 equally weighted batch/severity loss scalars rather than silently sample-weighting the tail. Detector features are computed in single-frame chunks on CPU without changing the logical batch loss or the normalized 256px direct-backbone preprocessing. It records data fingerprints, selected frames, execution/source hashes, measured loss, and constituent best losses. It does not use test AP to select weights. The four-frame visualization subset must fail before any release is written.

Complete official data collection, source comparison, runtime chunked-feature equivalence tests, and full MOT17-09 validation for all 5 variants (`full_tscope`, `spatial_only_scope`, `wo_ctu_delta_qp`, `iwa_gop_qp_reuse`, `proxy_only_zhao`) are finished (`release_ready=true` across all 5 soups in `validated_all_soups_20261112`).

### Step 4: Publication and paired evaluation — separate approval required

Before publishing, verify the five soups, validation provenance, file sizes, repository changes, and the exact release revision. Ask explicitly before committing/pushing or uploading checkpoints. Approval for local preparation is not approval for publication or remote dispatch.

After publication and separate dispatch approval:

```powershell
python -I scripts/kaggle_pool_orchestrator.py --pool_file "C:\Users\Wagur1\Downloads\pool (2).json" --action push --run_mode eval_soup --git_revision "<full_40_character_reviewed_published_commit>"
```

Every shard evaluates all five soups. The worker pins the commit instead of silently cloning a moving branch. Local preflight and evaluation reject unvalidated/missing soups. Require all 21 shards; do not present a reduced `--max_shards` run as final.

### Step 5: Collect and aggregate complete paired evaluation

Collect into new isolated directories using `--action pull`, then:

```powershell
python -I scripts/merge_shards.py --action eval --manifest "<complete_eval_collection_manifest.json>" --eval_dir "<new_final_summary_directory>"
```

Schema-v2 outputs contain raw detections, stable sequence/frame identities, integer packet bytes, codec/detector/metric/preprocessing provenance, and soup hashes. The merger requires **105 shard–variant outputs**, all methods/CRFs, matching frame partitions, identical expected frames, and no duplicate physical frames. It writes `merged_benchmarks.json` only after checks pass. Degenerate or nonoverlapping AP curves produce `null` BDBR with an explicit reason, not fabricated values.

### Step 6: Figures and documentation

```powershell
python -I scripts/gen_fig_bdbr_curves.py --benchmark "<merged_benchmarks.json>" --validate-only
python -I scripts/gen_fig_bdbr_curves.py --benchmark "<merged_benchmarks.json>" --output-dir "<new_figure_directory>"
python -I scripts/generate_ablation_visuals.py --help
```

Qualitative generation requires explicit variant soups, a final-ready benchmark, and a named held-out sequence/frame. Its grid is labelled **pre-edit inspection**, not decoded reconstruction evidence. Missing checkpoints/random weights cannot silently substitute for trained variants. Render final figures and run the visual acceptance gate before copying them into `docs/figures/`. Current manuscript figures remain historical; no replacements have been generated.

Update README result tables only from the verified merged final benchmark. Confirm before the final commit/push.

## 4. Local verification

65 unit/integration tests passed at continuation, including global-vs-clip AP, empty-GT false positives, ignore handling, strict soups, dtype preservation, preserved `val_subsample=2` validation protocol, chunked detector feature equivalence, paired frame rejection, packet-weighted rates, unavailable BDBR, plot contracts, and real dynamic-ROI `libx265` encode/decode with matched video packets. Source compilation and `git diff --check` passed.

```powershell
python -I -c "import sys,unittest; sys.path.insert(0,r'C:\Users\Wagur1\Desktop\Paper\vcm_preedit_x265'); suite=unittest.defaultTestLoader.discover(r'C:\Users\Wagur1\Desktop\Paper\vcm_preedit_x265\tests'); result=unittest.TextTestRunner().run(suite); sys.exit(not result.wasSuccessful())"
```

These tests do not substitute for full validation, complete test inference, publication-quality figure rendering, or evidence that ROI side data achieves the intended spatial rate allocation. Current pipeline results must not be advertised as final TCSVT/TIP performance.
