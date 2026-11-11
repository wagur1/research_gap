# IEEE TCSVT / TIP Blueprint: Temporally-Consistent Semantic Pre-Editing & Wrapper-Aware CTU Rate Control for Video Coding for Machines

## 1. Title Options & Target Venues
- **Primary Title:** *T-SCOPE: Temporally-Consistent Semantic Pre-Editing and CTU-Level Rate Allocation for Video Coding for Machines*
- **Alternative Title:** *Bridging Pixel Pre-Editing and In-Codec Rate Control for Standard-Compliant Video Coding for Machines*
- **Scope Note (`R11` Calibration):** Empirical evaluation in this repository validates standard-compliant **`libx265` (HEVC)** with dynamic per-frame `addroi` $\to$ `x265_picture.quantOffsets` under `crf={27,32,37,42,47}:aq-mode=2` (`IPPP` low-delay). Theoretical references to the Data Processing Inequality (DPI) / Information Bottleneck (Bajić 2025) serve as design motivation rather than formal mutual-information proofs, and cross-codec transfer to `VVenC` (H.266/VVC) is designated as planned journal extension work.
- **Target Venues:**
  1. **IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)** (Q1, IF ~8.4) — Best fit; direct successor to Lu et al. (TCSVT 2024) and Yang et al. (TCSVT 2024).
  2. **IEEE Transactions on Image Processing (TIP)** (Q1, IF ~10.8) — Strong fit if theoretical Rate-Accuracy / Information Bottleneck analysis (building on Bajić 2025) is emphasized.
  3. **ACM Multimedia / IEEE ICME / IEEE ICASSP / DCC** — Fast conference track if needed before journal extension.

---

## 2. Core Storyline & How It Defeats Existing SOTA

### 2.1 The Three Unsolved Bottlenecks in Existing Literature
1. **Spatial-Only Pre-Editing Breaks Inter-Frame Prediction in `x265` (Critique of SCOPE [Dani et al., 2026] & TSW-Pre [Yang et al., 2024]):**
   - Image pre-editors modify each frame independently: $I_{\text{edit}, t} = A_t \odot (I_t + \Delta_{\text{obj}, t}) + (1 - A_t) \odot (\text{Blur}(I_t) + \Delta_{\text{bg}, t})$.
   - In `x265` (IPPP / IBBP), independent frame-wise residuals $\Delta_{\text{obj}, t}$ and fluctuating masks $A_t$ introduce high-frequency temporal jitter between $I_{\text{edit}, t-1}$ and $I_{\text{edit}, t}$.
   - Consequently, `x265` Motion Estimation (ME) fails to find low-cost matches, inflating Motion Vector Differences (MVD), inter-prediction residuals, and forcing P/B-frame Coding Tree Units (CTUs) into high-bit Intra modes.
2. **Full-Frame MSE & Proxy Mismatch in Video Pre-Processing (Critique of Zhao et al. [DCC 2024 / arXiv:2512.15331]):**
   - Zhao et al. penalize full-frame pixel distortion $\mathcal{L}_D = \|X - \hat{X}\|_2^2$, forcing the pre-processor to preserve task-irrelevant background textures. By the Data Processing Inequality (Bajić [MIPR 2025]), preserving background mutual information prevents approaching the theoretical Rate-Accuracy bound.
   - Furthermore, Zhao et al. train solely against a hand-crafted Virtual Codec without `x265` in the forward loop, suffering severe proxy-to-real codec mismatch at low bitrates.
3. **Decoupling Between Pixel Pre-Editing and Codec-Internal RDO (Critique of SCOPE & Synthesis with WA-RDO [Fernández-Menduiña et al., ICASSP 2026]):**
   - Pre-editors already compute a spatial task-importance map $A_t \in [0, 1]^{H\times W}$ to blend foreground and background pixels, but discard $A_t$ before encoding.
   - Meanwhile, `x265` natively supports CTU/QG-level quantization offsets (`x265_picture.quantOffsets`) that adjust internal Rate-Distortion Optimization (RDO) weights $\lambda_{\text{RDO}}$ and $\text{QP}$ with zero bitstream overhead.

---

## 3. Mathematical Formulation (Section III of the Paper)

### 3.1 Problem Formulation via Rate-Accuracy & Information Bottleneck
Given an input video sequence $\mathcal{I} = \{I_1, \dots, I_T\}$ with $I_t \in [0, 1]^{3\times H\times W}$, a target standard video encoder $\mathcal{E}_f(\cdot; q, \mathbf{\Delta Q}_t)$ belonging to codec family $f \in \{\text{HEVC/x265}, \text{VVC/VVenC}, \text{AVC/x264}\}$ operating at base quantization parameter $q \in [0, 51]$ (mapped to normalized compression severity $s \in [0, 1]$), our goal is to learn a single conditional spatio-temporal pre-editor $\mathcal{P}_\theta$ that outputs both an edited video sequence $\{I_{\text{edit}, t}\}_{t=1}^T$ and CTU-level quantization offsets $\{\mathbf{\Delta Q}_t\}_{t=1}^T$:

$$\min_{\theta} \sum_{t=1}^{T} \Big[ R\big(\mathcal{E}_f(I_{\text{edit}, t}; q, \mathbf{\Delta Q}_t)\big) + \lambda \cdot \mathcal{D}_{\text{task}}\big(F_T(I_t), F_S(\hat{I}_t)\big) \Big]$$

where $\hat{I}_t = \mathcal{D}_f(\mathcal{E}_f(I_{\text{edit}, t}; q, \mathbf{\Delta Q}_t))$ is the standard-decoded frame and $F_T, F_S$ are multi-scale teacher and student feature maps from a frozen downstream vision backbone (e.g., FPN levels $\ell \in \{P_2, P_3, P_4, P_5\}$).

### 3.2 Codec-Family and Non-Linear Severity Conditioning
Following and extending SCOPE, we map codec family $f$ via learnable embedding $e_f \in \mathbb{R}^8$ and expand normalized QP severity $s = \frac{q - q_{\min}}{q_{\max} - q_{\min}} \in [0, 1]$ into a 6-D polynomial-hinge basis:

$$\phi(s) = \big[s, \, s^2, \, s^3, \, h_{0.50}(s), \, h_{0.75}(s), \, h_{0.90}(s)\big], \quad h_\tau(s) = \text{clip}\left(\frac{s - \tau}{1 - \tau}, 0, 1\right)$$

The conditioning vector $c = \text{MLP}([e_f, \phi(s)]) \in \mathbb{R}^{64}$ modulates all encoder-decoder stages via Feature-wise Linear Modulation ($\text{FiLM}(x, c) = x \odot (1 + \gamma(c)) + \beta(c)$).

### 3.3 Recurrent Spatio-Temporal Attention & Asymmetric Residual Blending
To eliminate inter-frame flickering without heavy 3D convolutions or optical-flow networks at inference, we insert a lightweight **Bottleneck ConvGRU** (`TemporalAlignmentBlock`) at the $H/4 \times W/4$ bottleneck feature level $Z_t$:

$$H_t = \text{ConvGRU}(Z_t, H_{t-1})$$

From the decoded feature map at frame $t$, three lightweight $3\times 3$ heads predict:
1. **Temporal-Consistent Object Attention Map:** $A_t = \sigma(\text{Head}_{\text{attn}}(U_t)) \in [0, 1]^{1\times H\times W}$.
2. **Bounded Object Preservation Residual:** $\Delta_{\text{obj}, t} = \epsilon_{\text{obj}} \tanh(\text{Head}_{\text{obj}}(U_t)) \in [-\epsilon_{\text{obj}}, \epsilon_{\text{obj}}]^{3\times H\times W}$ ($\epsilon_{\text{obj}} = 0.09$).
3. **Temporally-Anchored Background Branch:**
   $$I_{\text{bg}, t} = \text{clip}\Big(\alpha_{\text{ema}} I_{\text{bg}, t-1} + (1 - \alpha_{\text{ema}})\text{Blur}(I_t) + \Delta_{\text{bg}, t}, \, 0, \, 1\Big)$$
   where $\Delta_{\text{bg}, t} \in [-\epsilon_{\text{bg}}, \epsilon_{\text{bg}}]^{3\times H\times W}$ ($\epsilon_{\text{bg}} = 0.03$) and $\alpha_{\text{ema}} \in [0, 1)$ is gated by inter-frame change to prevent ghosting on moving camera backgrounds while freezing static background noise across P/B-frames.
4. **Blended Pre-Edited Frame:**
   $$I_{\text{edit}, t} = A_t \odot \text{clip}(I_t + \Delta_{\text{obj}, t}, 0, 1) + (1 - A_t) \odot I_{\text{bg}, t}$$

### 3.4 Attention-Driven CTU $\Delta\text{QP}$ Coupling (`quantOffsets` in `x265`)
Instead of discarding $A_t$, we partition $A_t$ into Quantization Groups (QG / CTU blocks of size $B \times B$, e.g., $B = 16$ or $32$ in `x265`) via average pooling:

$$A_{t, (u,v)}^{\text{QG}} = \frac{1}{B^2} \sum_{(x,y) \in \Omega_{u,v}} A_t(x, y)$$

We convert $A_{t, (u,v)}^{\text{QG}}$ into zero-sum centered quantization offsets passed directly to `x265_picture.quantOffsets`:

$$\mathbf{\Delta Q}_{t, (u,v)} = \kappa_{\max} \cdot \left(\bar{A}_t^{\text{QG}} - A_{t, (u,v)}^{\text{QG}}\right) \in [-\kappa_{\max}, +\kappa_{\max}]$$

where $\bar{A}_t^{\text{QG}}$ is the frame-mean attention and $\kappa_{\max} \in [3.0, 6.0]$ is the maximum QP swing.
- **Foreground blocks ($A_{t,(u,v)}^{\text{QG}} > \bar{A}_t^{\text{QG}}$):** Receive $\mathbf{\Delta Q} < 0$ (finer quantization, lower RDO $\lambda_{\text{SSE}} = 0.85 \cdot 2^{(\text{QP} + \Delta Q - 12)/3}$, preserving fine object boundaries).
- **Background blocks ($A_{t,(u,v)}^{\text{QG}} < \bar{A}_t^{\text{QG}}$):** Receive $\mathbf{\Delta Q} > 0$ (coarser quantization + higher RDO $\lambda$, forcing `x265` to select `SKIP`/`MERGE` modes on the already blurred background).
- **GOP-Level Reuse Variant (IWA-QP, inspired by IWA-RDO [Fernández-Menduiña et al., 2026]):** Compute $A_{t}$ and $\mathbf{\Delta Q}_t$ only on Keyframes / Mini-GOP anchors and propagate across $K$ inter-frames to cut encoder pre-processing overhead by $K\times$.

### 3.5 Hybrid Forward-`x265` / Backward-Proxy Straight-Through Estimator (STE)
To eliminate the virtual-to-real codec mismatch in Zhao et al. [2024]:
- **Forward Pass:** Encode $\{I_{\text{edit}, t}\}_{t=1}^T$ with real `x265` (using `--aq-mode` / `quantOffsets`) to obtain exact decoded frames $\hat{I}_{\text{x265}, t}$ and exact bitrate $R_{\text{x265}, t}$.
- **Backward Pass:** Pass $\{I_{\text{edit}, t}\}_{t=1}^T$ through a differentiable QP-conditioned Intra/Inter DCT Proxy $\mathcal{C}_{\text{proxy}}(I_{\text{edit}, t}, I_{\text{edit}, t-1}; s, \mathbf{\Delta Q}_t)$ to obtain $\hat{I}_{\text{proxy}, t}$ and differentiable rate surrogate $\mathcal{L}_{\text{rate}, t}$.
- **STE Coupling:**
  $$\hat{I}_t = \hat{I}_{\text{proxy}, t} + \text{stop\_grad}\big(\hat{I}_{\text{x265}, t} - \hat{I}_{\text{proxy}, t}\big)$$

### 3.6 Total Training Objective
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \mathcal{L}_{\text{attn}} + \mathcal{L}_{\text{bg\_spatial}} + \mathcal{L}_{\text{temp}} + \mathcal{L}_{\text{reg}}$$

1. **Task Consistency Loss ($\mathcal{L}_{\text{task}}$):** Masked Smooth-$L_1$ across FPN levels $\ell \in \{P_2 \dots P_5\}$:
   $$\mathcal{L}_{\text{task}} = \lambda_{\text{det}} \sum_{t=1}^T \sum_{\ell} \frac{\big\| M_{\text{det}, t}^\ell \odot \rho\big(F_S^\ell(\hat{I}_t) - F_T^\ell(I_t)\big) \big\|_1}{\sum M_{\text{det}, t}^\ell + \epsilon}$$
2. **Attention Supervision ($\mathcal{L}_{\text{attn}}$):** $\text{BCE}(A_t, M_{\text{attn}, t}) + \text{Dice}(A_t, M_{\text{attn}, t}) + \lambda_{\text{sparse}}\|(1 - M_{\text{det}, t})\odot A_t\|_1$.
3. **Spatial Background Simplification ($\mathcal{L}_{\text{bg\_spatial}}$):** Penalizes high-frequency gradients and unblurred residuals in $M_{\text{bg}, t} = 1 - M_{\text{det}, t}$.
4. **Temporal Inter-Frame Residual Suppression ($\mathcal{L}_{\text{temp}}$ — New Video Contribution):**
   $$\mathcal{L}_{\text{temp}} = \lambda_{\text{temp\_bg}} \sum_{t=2}^T \frac{\big\| M_{\text{bg}, t} \odot M_{\text{bg}, t-1} \odot (I_{\text{edit}, t} - I_{\text{edit}, t-1}) \big\|_1}{\sum M_{\text{bg}, t} M_{\text{bg}, t-1} + \epsilon} + \lambda_{\text{temp\_obj}} \sum_{t=2}^T \|\Delta_{\text{obj}, t} - \Delta_{\text{obj}, t-1}\|_1$$
   This directly minimizes the P/B-frame prediction residual energy fed into `x265` Motion Estimation.

---

## 4. Paper Structure & Planned Experimental Tables (IEEE TCSVT 12–14 Pages)

- **Section I: Introduction** (1.5 pages) — Figure 1: Rate-Accuracy + P-frame Bit Allocation comparison showing why frame-by-frame SCOPE fails on `x265` inter-frames while T-SCOPE suppresses P-frame residuals and allocates CTU bits to objects.
- **Section II: Related Work** (1.5 pages) — (A) Pixel-Domain VCM & Standard-Compliant Pre-Processing, (B) Feature Coding & In-Codec RDO (WA-RDO), (C) Rate-Accuracy Bounds in VCM.
- **Section III: Proposed Method** (3.5 pages) — Architecture diagram (Fig. 2), Temporal ConvGRU + Asymmetric Blending, Attention-to-CTU $\Delta\text{QP}$ Mapping, Hybrid `x265`-STE Training Pipeline.
- **Section IV: Experimental Results** (5 pages):
  - **Table I (Main Result on `x265` Low-Delay P & Random Access):** BDBR (%) and BD-mAP / BD-MOTA across **SFU-HW-Objects-v1** (MPEG-VCM CTC), **TVD**, and **BDD100K-MOT** on 3 detectors/trackers (Faster R-CNN, YOLO11x, RT-DETR-l + ByteTrack). Compare against:
    1. Anchor `x265` (`medium`, QPs $\{27, 32, 37, 42, 47\}$)
    2. Lu et al. (TCSVT 2024)
    3. Zhao et al. (DCC 2024 Video Preprocessor)
    4. SCOPE (2026, applied frame-by-frame)
    5. **T-SCOPE (Ours)**
  - **Table II (Zero-Shot Cross-Detector Transfer on Real `libx265` & Planned Cross-Codec Extension):** Evaluate the Faster R-CNN-trained T-SCOPE zero-shot across 4 distinct object detectors (`Faster R-CNN`, `RetinaNet`, `FCOS`, `SSDLite`) using official dataset-level COCO/VOC AP on held-out `MOT17` sequences (`scripts/eval_cross_backbone_bdbr.py`), and report feature-space SNR on `Swin-T` separately. Cross-codec evaluation on `VVenC` (H.266/VVC) is reserved as future extension.
  - **Table III (Ablation Study on `x265`):**
    - Row 1: Direct `x265` (0.0% BDBR)
    - Row 2: Spatial Pre-Edit only (SCOPE baseline on video) $\to$ saves Intra bits, hurts P-frame ME
    - Row 3: + Temporal ConvGRU & $\mathcal{L}_{\text{temp}}$ $\to$ massive drop in P/B-frame bpp
    - Row 4: + Attention-Driven CTU `quantOffsets` ($\Delta\text{QP}$) $\to$ extra $-5\%$ to $-8\%$ BDBR at zero neural FLOP cost
    - Row 5: Proxy-only training (Zhao et al.) vs. Hybrid Forward-`x265` STE training
  - **Table IV (Complexity & GOP-Subsampled `IWA-QP` Analysis):** FPS, Params (M), MACs, and encoding time reduction on T4 GPU + CPU `x265` (showing `x265` actually encodes *faster* after T-SCOPE because simplified backgrounds trigger early CU Skip termination).

---

## 5. GPU Budget Allocation (600h T4 / week)
- **Step 0 (Offline Teacher Caching — ~15 GPU hours):** Pre-compute and save compressed FP16 FPN features ($P_2 \dots P_5$) and GT masks for training clips ($512\times 512$, $T=4$ frames/clip) so the frozen ResNeXt-101 detector does not occupy VRAM or FLOPs on every epoch.
- **Step 1 (Warm-up Training with Differentiable Video Proxy — ~60 GPU hours):** Train `TemporalSCOPE` for 30 epochs with batch size 4 ($T=4$ frames) on 1–2 T4 GPUs (<7 GB VRAM).
- **Step 2 (Fine-tuning with Hybrid `x265`-STE — ~80 GPU hours):** Enable `HybridX265STE` for 10 epochs so the pre-editor adapts to exact `x265` block quantization and `quantOffsets`.
- **Step 3 (Evaluation & Ablations across `x265`, `VVenC`, 3 Detectors — ~60 GPU hours):** Total <220 GPU hours (well inside your 600h/week T4 budget, leaving >380h for hyperparameter sweeps and Wish-List Item #2).
