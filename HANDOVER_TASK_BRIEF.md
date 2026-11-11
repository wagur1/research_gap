# T-SCOPE HANDOVER TASK BRIEF & EXECUTION PLAYBOOK
**Target Publication:** IEEE Transactions on Circuits and Systems for Video Technology (TCSVT) / IEEE Transactions on Image Processing (TIP)  
**Project Title:** T-SCOPE: Temporally-Consistent Semantic Pre-Editing and CTU-Level Rate Allocation for Video Coding for Machines  
**Document Type:** AI Agent Handover / Next-Shift Operational Blueprint  
**Last Updated:** November 2026  
**Status:** In-Progress (Converged Training Phase across 21 Kaggle Shards)

---

## 1. Executive Summary & Core Contribution

### 1.1 The Scientific Objective
Video Coding for Machines (VCM) focuses on compressing video for downstream machine analysis (e.g., object detection, tracking) rather than human visual pleasure. Traditional video codecs (e.g., standard HEVC/libx265) allocate bits to maintain pixel-level fidelity everywhere, wasting bitrate on machine-irrelevant background.

**T-SCOPE** resolves this via a dual-mechanism framework:
1. **Temporally-Consistent Semantic Pre-Editing (IWA-Net):** An Instance-Weighted Attention network estimates a dynamic attention mask $A_t \in [0, 1]^{H \times W}$ and a residual pre-edit $\Delta_{\text{obj}}$. High-relevance foreground regions are enhanced for feature extraction; background is smoothed to eliminate high-frequency bits.
2. **CTU-Level Rate Allocation ($\Delta\text{QP}_t$ Injection):** Computes a per-CTU (64x64) quantization parameter delta map $\Delta\text{QP}_t \in [-12, +12]$. Negative $\Delta\text{QP}$ (higher quality/lower quantization) is allocated to machine-salient CTUs, while positive $\Delta\text{QP}$ compresses background heavily. This is injected into actual `libx265` via per-frame dynamic filtergraphs using `quantOffsets`.

### 1.2 Downstream Machine Task
- **Detector:** Pretrained Faster R-CNN with ResNet-50-FPN (`torchvision.models.detection.fasterrcnn_resnet50_fpn`).
- **Benchmark:** MOT17 pedestrian dataset (`mark == 1 and class_id == 1`). Ignore/distractor classes (`2, 7, 8, 12` or `mark == 0`) are filtered correctly according to MOT17 protocol.
- **Metric:** **Dataset-Level 101-Point COCO AP** (`AP@50`, `AP@50:95`) evaluated across all decoded frames, paired with Bjontegaard Delta Rate (**BD-Rate** / BDBR) against unedited HEVC baseline.

---

## 2. Infrastructure & Workspace Map

| Resource | Path / Identifier | Description |
| :--- | :--- | :--- |
| **Local Project Dir** | `C:\Users\Wagur1\Desktop\Paper\vcm_preedit_x265` | Main git repository root |
| **Remote Git Repo** | `https://github.com/wagur1/research_gap.git` | Branch: `main` |
| **Kaggle Account Pool** | `C:\Users\Wagur1\Downloads\pool (2).json` | 21 authenticated Kaggle accounts (`username: token`) |
| **Latest Git Commit** | `f6b288f` | Incremental best-val checkpointing & `typing.Any` fix |
| **Python Environment** | Python 3.11+ / PyTorch 2.x + CUDA | Standard scientific stack + `ffmpeg`/`ffprobe` |

---

## 3. Strict Theoretical & Methodological Ground Rules (NON-NEGOTIABLES)

If you modify or evaluate the system, **you must strictly adhere to these rules**:

1. **NO Proxy AP Formulas:**
   - Never use cosine feature similarity, MSE, SNR, or heuristic scaling to approximate mAP.
   - All detection AP numbers must be derived by feeding the actual decoded H.265 bitstream into Faster R-CNN and computing true 101-point COCO AP against ground-truth annotations.
2. **Real `libx265` Encoding & Bitrate Measurement:**
   - Videos must be encoded using real `ffmpeg -c:v libx265`.
   - Bitrates must be calculated strictly from packet sizes extracted via `ffprobe -show_packets` (`pkt_size > 0` for `pict_type in ("I", "P", "B")`). Do not calculate bitrate from file size estimates.
3. **Dynamic CTU Injection Filtergraph:**
   - Per-frame $\Delta\text{QP}_t$ maps must be piped into `libx265` using the dynamic filtergraph construct:
     `split + select='eq(n,t)' + addroi + interleave`.
4. **Strict Dataset Splitting:**
   - **Train:** `MOT17-02`, `MOT17-04`, `MOT17-05`, `MOT17-10`
   - **Validation:** `MOT17-09` (used for `val_loss` early stopping)
   - **Held-Out Test:** `MOT17-11`, `MOT17-13`
   - Zero test ground truth leakage during inference.
5. **Understanding VCM Metric Behavior:**
   - In VCM, a negative BDBR on `GT AP@50` (e.g. `-24.84%`) means that for the same downstream accuracy, the model requires **24.84% fewer bits**. This is the primary victory.
   - BDBR on `ROI-PSNR` will be positive (`+21.32%`), because pixel fidelity was sacrificed to save bitrate for the machine. Do not mistake positive PSNR BDBR for an error; this is standard VCM behavior.

---

## 4. Current State Snapshot (At Handover)

### 4.1 Kaggle 21-Shard Convergence Status
21 shards were dispatched with convergence flags:
`--epochs 60 --patience 10 --min_epochs 12 --lr 2e-4 --batch_size 8 --ablation_sweep`.

- **Completed Shards (Early Stopped / Converged):**
  - `Shard 04` (`dngbolm`, `proxy_only_zhao`): Converged at Epoch 12 (`val_loss=13.1437`), checkpoint saved.
  - `Shard 06` (`tranthihongdieu`, `spatial_only_scope`): Completed.
- **Running Shards:**
  - 13 Shards running on GPU T4: `01, 02, 03, 05, 07, 08, 09, 10, 11, 12, 13, 14, 16`.
  - 6 Shards running on CPU: `00, 15, 17, 18, 19, 20`.

### 4.2 Ablation Variant Mapping Across 21 Shards
The 21 accounts rotate through 5 ablation variants:
1. `full_tscope` (IWA-Net + Temporal Mask + CTU $\Delta\text{QP}$)
2. `spatial_only_scope` (No temporal GRU/consistency)
3. `wo_ctu_delta_qp` (Pre-editing only, no CTU rate allocation)
4. `iwa_gop_qp_reuse` (GOP-level QP reuse baseline)
5. `proxy_only_zhao` (Prior-art proxy baseline Zhao et al.)

---

## 5. Next-Agent Step-by-Step Playbook

Follow these sequential steps to complete the experimental pipeline and paper artifacts:

### Step 1: Monitor & Wait for All 21 Shards to Complete
Check the status of all kernels periodically:
```powershell
Set-Location -Path "C:\Users\Wagur1\Desktop\Paper\vcm_preedit_x265"
python scripts/kaggle_pool_orchestrator.py --pool_file "C:\Users\Wagur1\Downloads\pool (2).json" --action status
```

To inspect log output of a specific shard (e.g., Shard 02):
```powershell
python -c "
import json, os
from kaggle.api.kaggle_api_extended import KaggleApi
with open(r'C:\Users\Wagur1\Downloads\pool (2).json') as f: pool = list(json.load(f).items())
user, key = pool[2]
os.environ['KAGGLE_USERNAME'], os.environ['KAGGLE_KEY'] = user, key
api = KaggleApi(); api.authenticate()
api.kernels_output(f'{user}/t-scope-vcm-shard-02', path='./tmp_log')
print(open('./tmp_log/t-scope-vcm-shard-02.log').readlines()[-20:])
"
```
*Wait until all or the vast majority of shards show `KernelWorkerStatus.COMPLETE`.*

---

### Step 2: Pull Converged Checkpoints and Build Model Soups
When shards finish training, pull all outputs and automatically compute uniform weight averaging (Model Soup) for each variant:
```powershell
Set-Location -Path "C:\Users\Wagur1\Desktop\Paper\vcm_preedit_x265"
python scripts/kaggle_pool_orchestrator.py --pool_file "C:\Users\Wagur1\Downloads\pool (2).json" --action pull_and_merge --output_dir ./outputs_kaggle_converged
```

**Verification:**
Check that the 5 soup checkpoints are generated:
- `outputs_kaggle_converged/tscope_soup_full_tscope.pt`
- `outputs_kaggle_converged/tscope_soup_spatial_only_scope.pt`
- `outputs_kaggle_converged/tscope_soup_wo_ctu_delta_qp.pt`
- `outputs_kaggle_converged/tscope_soup_iwa_gop_qp_reuse.pt`
- `outputs_kaggle_converged/tscope_soup_proxy_only_zhao.pt`

Copy soup checkpoints to `release_ckpts/` and commit:
```powershell
New-Item -ItemType Directory -Force -Path "./release_ckpts"
Copy-Item ./outputs_kaggle_converged/tscope_soup_*.pt ./release_ckpts/
git add release_ckpts/
git commit -m "feat(ckpt): add converged model soup checkpoints across 21 shards"
git push origin main
```

---

### Step 3: Dispatch Paired Evaluation Across 21 Shards (`eval_soup`)
Once model soup weights are committed to GitHub, dispatch the evaluation run across all 21 shards. Each shard will evaluate all 5 variant soups on its disjoint test slice (`MOT17-11` and `MOT17-13`):
```powershell
python scripts/kaggle_pool_orchestrator.py --pool_file "C:\Users\Wagur1\Downloads\pool (2).json" --action push --run_mode eval_soup
```
This triggers real `libx265` encoding across QP points `[22, 27, 32, 37]`, `ffprobe` frame packet byte extraction, and 101-point COCO AP evaluation.

---

### Step 4: Pull & Aggregate Final Test Evaluation
Monitor status using `--action status`. When all shards reach `COMPLETE`:
```powershell
python scripts/kaggle_pool_orchestrator.py --pool_file "C:\Users\Wagur1\Downloads\pool (2).json" --action pull_and_merge --output_dir ./outputs_kaggle_converged_eval
```
This merges all disjoint test slices into:
`./outputs_kaggle_converged_eval/merged_benchmarks.json` and prints the final BDBR summary table.

---

### Step 5: Update Visuals & Paper Figures
Run the plotting scripts to regenerate publication figures using the new converged evaluation results:
```powershell
# 1. Regenerate RD curves and BDBR bar charts
python scripts/gen_fig_bdbr_curves.py

# 2. Regenerate qualitative ablation visual grid (attention maps, pre-edits, CTU delta QP heatmaps)
python scripts/generate_ablation_visuals.py
```
Check that the figures in `docs/figures/` are refreshed:
- `docs/figures/fig_bdbr_curves.png`
- `docs/figures/ablation_preedit_comparison_grid.png`

---

### Step 6: Update Documentation & Final Git Push
1. Open `README.md`, update Section 5 (**Empirical Validation & Benchmark Results**) with the new final converged BDBR table.
2. Commit and push everything to GitHub:
```powershell
git add docs/figures/ README.md
git commit -m "docs: finalize converged 21-shard BDBR results and publication figures"
git push origin main
```

---

## 6. Key Codebase Index

| File | Purpose |
| :--- | :--- |
| [`models/temporal_scope.py`](file:///C:/Users/Wagur1/Desktop/Paper/vcm_preedit_x265/models/temporal_scope.py) | Full T-SCOPE architecture (IWA-Net, Temporal GRU, Mask Head $A_t$, CTU $\Delta\text{QP}$ Head) |
| [`models/frozen_detector_fpn.py`](file:///C:/Users/Wagur1/Desktop/Paper/vcm_preedit_x265/models/frozen_detector_fpn.py) | Frozen Faster R-CNN with ResNet-50-FPN downstream detector |
| [`scripts/kaggle_shard_runner.py`](file:///C:/Users/Wagur1/Desktop/Paper/vcm_preedit_x265/scripts/kaggle_shard_runner.py) | Standalone script executed on Kaggle. Implements training with `ReduceLROnPlateau`, EarlyStopping, and test evaluation |
| [`scripts/kaggle_pool_orchestrator.py`](file:///C:/Users/Wagur1/Desktop/Paper/vcm_preedit_x265/scripts/kaggle_pool_orchestrator.py) | Multi-account orchestrator: manages authentication, pushing kernels, status tracking, downloading, and Model Soup merging |
| [`vcm_datasets/vcm_video_datasets.py`](file:///C:/Users/Wagur1/Desktop/Paper/vcm_preedit_x265/vcm_datasets/vcm_video_datasets.py) | MOT17 dataset loader with train/val/test split handling, pedestrian filtering, and ignore masks |
| [`scripts/gen_fig_bdbr_curves.py`](file:///C:/Users/Wagur1/Desktop/Paper/vcm_preedit_x265/scripts/gen_fig_bdbr_curves.py) | Generates IEEE-ready Rate-Distortion curves and BDBR comparative charts |
| [`scripts/generate_ablation_visuals.py`](file:///C:/Users/Wagur1/Desktop/Paper/vcm_preedit_x265/scripts/generate_ablation_visuals.py) | Generates side-by-side qualitative visual grids comparing ablation variants |

---

## 7. Edge Cases & Troubleshooting

- **Python 3.13 on Kaggle:** Kaggle environments now run Python 3.13. Avoid missing type imports (e.g., `Any` must be imported explicitly from `typing`).
- **Kaggle 12-Hour Kernel Limit:** If a CPU shard reaches 12 hours, incremental checkpointing has already saved the best checkpoint up to that point. The orchestrator's `pull_and_merge` will pick up the saved `.pt` file.
- **Kaggle API Rate Limits (HTTP 429):** If Kaggle throws a 429 error during status checks, increase the sleep interval between accounts in `kaggle_pool_orchestrator.py` by 1.5 seconds.
- **ffmpeg `libx265` dynamic filtergraph:** If ffmpeg fails on a machine, ensure `ffmpeg` with `libx265` is in system PATH and filtergraph syntax quotes match Windows PowerShell vs Linux bash standards.

---
*End of Handover Brief.*
