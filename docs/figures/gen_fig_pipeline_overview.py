#!/usr/bin/env python3
"""Generate publication-ready overview pipeline figure for T-SCOPE.

Follows scientific-figure-making standards (figures4papers style):
- Clean vector layout for IEEE TCSVT / IEEE TIP.
- Exact T-SCOPE workflow: Multi-modal inputs, Severity Conditioner,
  TemporalSCOPE U-Net with ConvGRU bottleneck, Dual-branch pre-editing with
  Motion-gated EMA, Asymmetric CTU-level rate allocation, libx265 encoding with
  differentiable proxy STE, and Dual-Detector machine vision evaluation.
- Outputs high-res PDF and 300 DPI PNG.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def create_pipeline_figure(output_png: Path, output_pdf: Path):
    # Standard dimensions for full-width IEEE 2-column landscape figure
    fig, ax = plt.subplots(figsize=(17.5, 8.8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Typography setup
    font_main = "DejaVu Sans"
    plt.rcParams["font.sans-serif"] = [font_main, "Arial", "Helvetica", "sans-serif"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "dejavusans"

    # Color Palette (ColorBrewer & figures4papers style)
    c_input_bg = "#EFF6FF"       # Soft Blue
    c_input_border = "#2563EB"
    c_input_text = "#1E3A8A"

    c_cond_bg = "#F5F3FF"        # Soft Purple
    c_cond_border = "#7C3AED"
    c_cond_text = "#4C1D95"

    c_net_bg = "#F0FDF4"         # Soft Green (Temporal)
    c_net_border = "#16A34A"
    c_net_text = "#14532D"

    c_edit_bg = "#FEF3C7"        # Soft Amber/Yellow
    c_edit_border = "#D97706"
    c_edit_text = "#78350F"

    c_qp_bg = "#FFEDD5"          # Soft Orange
    c_qp_border = "#EA580C"
    c_qp_text = "#7C2D12"

    c_codec_bg = "#F1F5F9"       # Slate Gray (HEVC Codec)
    c_codec_border = "#475569"
    c_codec_text = "#0F172A"

    c_task_bg = "#EEF2FF"        # Soft Indigo (Dual-Detector)
    c_task_border = "#4F46E5"
    c_task_text = "#1E1B4B"

    c_proxy_bg = "#FDF2F8"       # Soft Pink/Rose (STE Proxy)
    c_proxy_border = "#DB2777"
    c_proxy_text = "#831843"

    # Helper function for card containers
    def draw_container(x, y, w, h, title, subtitle="", bg_col="#FFFFFF", border_col="#CBD5E1", header_col=None):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                             facecolor=bg_col, edgecolor=border_col, linewidth=1.4, zorder=1)
        ax.add_patch(box)
        if header_col:
            hbar = FancyBboxPatch((x + 0.5, y + h - 4.2), w - 1.0, 3.6, boxstyle="round,pad=0.2,rounding_size=0.7",
                                  facecolor=header_col, edgecolor="none", zorder=2)
            ax.add_patch(hbar)
            ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="center", fontsize=10.2,
                    fontweight="bold", color="#FFFFFF", zorder=3)
        else:
            ax.text(x + 1.2, y + h - 2.2, title, ha="left", va="center", fontsize=10.2,
                    fontweight="bold", color="#1E293B", zorder=3)
        if subtitle:
            ax.text(x + w / 2, y + h - 5.5, subtitle, ha="center", va="center", fontsize=8.0,
                    fontstyle="italic", color="#475569", zorder=3)

    # Helper function for sub-modules
    def draw_module(x, y, w, h, text_main, text_sub="", bg_col="#FFFFFF", border_col="#94A3B8",
                    text_col="#0F172A", fontsize=8.8, bold=True, zorder=4):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
                             facecolor=bg_col, edgecolor=border_col, linewidth=1.2, zorder=zorder)
        ax.add_patch(box)
        if text_sub:
            ax.text(x + w / 2, y + h * 0.65, text_main, ha="center", va="center", fontsize=fontsize,
                    fontweight="bold" if bold else "normal", color=text_col, zorder=zorder + 1)
            ax.text(x + w / 2, y + h * 0.30, text_sub, ha="center", va="center", fontsize=fontsize * 0.82,
                    fontweight="normal", color=text_col, zorder=zorder + 1)
        else:
            ax.text(x + w / 2, y + h / 2, text_main, ha="center", va="center", fontsize=fontsize,
                    fontweight="bold" if bold else "normal", color=text_col, zorder=zorder + 1)

    # Helper function for drawing arrows
    def draw_arrow(p1, p2, col="#1E293B", style="-|>", lw=1.5, ls="-", rad=0.0, label="", label_pos=0.5,
                   label_offset=(0, 1.2), label_col=None, fontsize=7.6):
        conn = f"arc3,rad={rad}" if rad != 0 else "arc3"
        arrow = FancyArrowPatch(p1, p2, arrowstyle=style, color=col, linewidth=lw,
                                linestyle=ls, connectionstyle=conn, mutation_scale=11, zorder=5)
        ax.add_patch(arrow)
        if label:
            lx = p1[0] + (p2[0] - p1[0]) * label_pos + label_offset[0]
            ly = p1[1] + (p2[1] - p1[1]) * label_pos + label_offset[1]
            tcol = label_col if label_col else col
            ax.text(lx, ly, label, ha="center", va="center", fontsize=fontsize,
                    fontweight="bold", color=tcol, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#FFFFFF", edgecolor="none", alpha=0.92))

    # ==========================================
    # 0. MAIN HEADER TITLE
    # ==========================================
    ax.text(50.0, 97.6, "T-SCOPE: Temporally-Consistent Semantic Pre-Editing & Asymmetric CTU Rate Allocation for VCM",
            ha="center", va="center", fontsize=14.0, fontweight="bold", color="#0F172A")
    ax.text(50.0, 95.1, "Standard-Compliant HEVC libx265 Video Compression for Machine Vision Tasks (Dual-Detector: Faster R-CNN v2 & YOLOv8s)",
            ha="center", va="center", fontsize=9.0, color="#475569")

    # ==========================================
    # 1. PANEL 1: MULTI-MODAL TEMPORAL INPUTS & CONDITIONING
    # ==========================================
    draw_container(1.0, 5.0, 14.8, 88.0, "1. Inputs & Rate Target", "Temporal Video & Severity",
                   bg_col="#F8FAFC", border_col="#94A3B8", header_col="#1E40AF")

    # Frame I_t
    draw_module(2.2, 74.0, 12.4, 11.5, r"Frame $I_t$", "Current RGB Frame\n" + r"$3 \times H \times W$",
                bg_col=c_input_bg, border_col=c_input_border, text_col=c_input_text)

    # Frame diff ΔI_t
    draw_module(2.2, 59.5, 12.4, 11.5, r"Motion Residual $\Delta I_t$", r"$|I_t - I_{t-1}|$" + "\nTemporal Frame Diff",
                bg_col=c_input_bg, border_col=c_input_border, text_col=c_input_text)

    # Concat block
    draw_module(3.2, 48.0, 10.4, 7.5, r"Concat $[I_t, \Delta I_t]$", r"$6 \times H \times W$ Tensor",
                bg_col="#DBEAFE", border_col="#3B82F6", text_col="#1E40AF", fontsize=8.2)

    draw_arrow((8.4, 74.0), (8.4, 55.5), col="#2563EB", lw=1.3)
    draw_arrow((8.4, 59.5), (8.4, 55.5), col="#2563EB", lw=1.3)

    # Target Rate Severity & Phase
    draw_module(2.2, 29.5, 12.4, 13.5, r"Rate Severity $s \in [0, 1]$",
                r"Target CRF $\in [22, 37]$" + "\n" + r"+ Phase $e_f \in \{0, 1\}$",
                bg_col=c_cond_bg, border_col=c_cond_border, text_col=c_cond_text, fontsize=8.2)

    # Severity Conditioner (MLP)
    draw_module(2.2, 7.5, 12.4, 17.5, "Severity Conditioner",
                r"Sinusoidal $\phi(s)$" + "\n+ 3-Layer MLP\n" + r"$\Downarrow$" + "\nFiLM Vector " + r"$c_t \in \mathbb{R}^{d}$",
                bg_col="#EDE9FE", border_col="#6D28D9", text_col="#4C1D95", fontsize=8.2)

    draw_arrow((8.4, 29.5), (8.4, 25.0), col="#7C3AED", lw=1.3)

    # ==========================================
    # 2. PANEL 2: TEMPORALSCOPE PRE-EDITING ARCHITECTURE
    # ==========================================
    draw_container(16.8, 5.0, 33.2, 88.0, "2. TemporalSCOPE Pre-Editing Architecture", "Recurrent Spatio-Temporal U-Net with FiLM",
                   bg_col="#F8FAFC", border_col="#94A3B8", header_col="#15803D")

    # Upper sub-container: Recurrent U-Net Backbone
    draw_container(18.0, 48.0, 30.8, 38.5, "Recurrent Multi-Scale Backbone", "Latent Feature Modulation & ConvGRU",
                   bg_col="#FFFFFF", border_col="#CBD5E1")

    # FiLM Encoder
    draw_module(19.2, 51.0, 8.6, 27.5, "FiLM Encoder",
                "Conv + ResBlocks\nDownsample $4\\times$\nOutput: $H/4 \\times W/4$\nFiLM Modulation:\n" + r"$\gamma(c) \cdot x + \beta(c)$",
                bg_col=c_net_bg, border_col=c_net_border, text_col=c_net_text, fontsize=8.0)

    # ConvGRU Bottleneck
    draw_module(29.1, 51.0, 9.6, 27.5, "ConvGRU Bottleneck",
                r"$H/4 \times W/4$" + " Latents\n" +
                r"$h_t = \mathrm{GRU}(z_t, h_{t-1})$" + "\n\n" +
                "Propagates Temporal\nRecurrent State\nEliminates Inter-frame\nFlicker & Noise",
                bg_col="#DCFCE7", border_col="#16A34A", text_col="#14532D", fontsize=7.8)

    # Recurrent loop on ConvGRU
    draw_arrow((37.0, 51.0), (37.0, 49.2), col="#16A34A", lw=1.3, style="-")
    draw_arrow((37.0, 49.2), (30.8, 49.2), col="#16A34A", lw=1.3, style="-")
    draw_arrow((30.8, 49.2), (30.8, 51.0), col="#16A34A", lw=1.3, style="-|>",
               label=r"$h_{t-1}$", label_pos=0.5, label_offset=(0, -1.3), label_col="#15803D")

    # FiLM Decoder
    draw_module(40.0, 51.0, 7.6, 27.5, "FiLM Decoder",
                "Upsample $4\\times$\nSkip Connections\nFrom Encoder\nMulti-Head Maps:\n" + r"$\{A_t, \Delta_{\mathrm{obj}}, \sigma\}$",
                bg_col=c_net_bg, border_col=c_net_border, text_col=c_net_text, fontsize=7.8)

    # Connections inside U-Net
    draw_arrow((27.8, 64.7), (29.1, 64.7), col="#16A34A", lw=1.4, label=r"$z_t$", label_pos=0.5, label_offset=(0, 1.2))
    draw_arrow((38.7, 64.7), (40.0, 64.7), col="#16A34A", lw=1.4, label=r"$h_t$", label_pos=0.5, label_offset=(0, 1.2))
    # Skip connection over ConvGRU
    draw_arrow((23.5, 78.5), (43.8, 78.5), col="#15803D", lw=1.3, ls=":", rad=0.25, label="Skip Connection", label_pos=0.5, label_offset=(0, 1.4))

    # Routing from Panel 1 to Panel 2
    draw_arrow((13.6, 51.7), (19.2, 64.7), col="#2563EB", lw=1.4, label=r"$[I_t, \Delta I_t]$", label_pos=0.45)
    # Routing FiLM conditioning vector
    draw_arrow((14.6, 16.2), (19.2, 53.5), col="#7C3AED", lw=1.3, ls="--", rad=0.2, label=r"FiLM $c_t$", label_pos=0.45)

    # Lower sub-container: Dual-Branch Pre-Editing & Synthesis
    draw_container(18.0, 7.5, 30.8, 38.0, "Dual-Branch Pre-Editing & Adaptive Blending", "Foreground Task Residual + Motion-Gated Background",
                   bg_col="#FFFFFF", border_col="#CBD5E1")

    # Foreground Sub-branch
    draw_module(19.2, 23.5, 13.5, 15.0, "Foreground Branch (Task)",
                r"Object Residual $\Delta_{\mathrm{obj}, t}$" + "\n" +
                r"$I_{\mathrm{obj}, t} = I_t + \Delta_{\mathrm{obj}, t}$" + "\n" +
                "Preserves Salient Edges\n& Human Machine Features",
                bg_col="#FEF3C7", border_col="#F59E0B", text_col="#92400E", fontsize=7.6)

    # Background Sub-branch
    draw_module(34.1, 23.5, 13.5, 15.0, "Background Branch (Bitrate)",
                "Adaptive Gaussian Blur $I_{\\mathrm{blur}}$\n" +
                r"+ Motion-Gated EMA Locking:" + "\n" +
                r"$I_{\mathrm{bg}, t} = (1 - M_{\mathrm{st}})I_{\mathrm{blur}} + M_{\mathrm{st}}\hat{I}_{\mathrm{bg}, t-1}$" + "\n" +
                "Eliminates Background High-Freq",
                bg_col="#FEF3C7", border_col="#F59E0B", text_col="#92400E", fontsize=7.4)

    # Blending Box
    draw_module(21.0, 9.5, 24.8, 11.5, r"Dynamic Alpha Blending:  $\mathbf{I}_{\mathbf{edit}, t} = A_t \odot I_{\mathrm{obj}, t} + (1 - A_t) \odot I_{\mathrm{bg}, t}$",
                r"Guided by Spatio-Temporal Attention Map $A_t \in [0, 1]^{1 \times H \times W}$",
                bg_col="#FDE68A", border_col="#D97706", text_col="#78350F", fontsize=8.0)

    # Arrows from Decoder to Dual-Branch and Blending
    draw_arrow((43.8, 51.0), (26.0, 38.5), col="#D97706", lw=1.3, rad=-0.15, label=r"$\Delta_{\mathrm{obj}}$", label_pos=0.45)
    draw_arrow((43.8, 51.0), (40.8, 38.5), col="#D97706", lw=1.3, rad=0.1, label=r"$\sigma_{\mathrm{blur}}$", label_pos=0.45)
    draw_arrow((26.0, 23.5), (28.0, 21.0), col="#D97706", lw=1.2)
    draw_arrow((40.8, 23.5), (38.8, 21.0), col="#D97706", lw=1.2)

    # ==========================================
    # 3. PANEL 3: ASYMMETRIC CTU RATE ALLOCATION (QG-16)
    # ==========================================
    draw_container(51.0, 5.0, 16.5, 88.0, "3. Asymmetric CTU Rate", "16x16 ROI Grid & addroi",
                   bg_col="#F8FAFC", border_col="#94A3B8", header_col="#C2410C")

    # Attention Map Downsampling
    draw_module(52.2, 70.0, 14.1, 15.0, r"Attention Map $A_t$",
                "Soft Spatial-Temporal Mask\n" + r"$A_t \in [0, 1]^{1 \times H \times W}$" + "\n" +
                r"Average Pooling $\to 16 \times 16$" + "\nAligned with HEVC CTU Grid",
                bg_col=c_qp_bg, border_col=c_qp_border, text_col=c_qp_text, fontsize=8.0)

    # Asymmetric QP Mapping
    draw_module(52.2, 39.5, 14.1, 26.5, "Asymmetric QP Mapping",
                r"Piecewise ROI Allocation:" + "\n\n" +
                r"$\bullet$ Foreground ($A_t > \tau$):" + "\n" +
                r"$\Delta\mathrm{QP}_{\mathrm{fg}} \in [-4.0, 0]$" + "\n" +
                "(High Machine Fidelity)\n\n" +
                r"$\bullet$ Background ($A_t \leq \tau$):" + "\n" +
                r"$\Delta\mathrm{QP}_{\mathrm{bg}} \in [0, +7.2]$" + "\n" +
                "(Aggressive Bitrate Saving)",
                bg_col="#FFF7ED", border_col="#EA580C", text_col="#9A3412", fontsize=7.8)

    draw_arrow((59.25, 70.0), (59.25, 66.0), col="#EA580C", lw=1.3)

    # FFmpeg addroi Generation & RLE
    draw_module(52.2, 7.5, 14.1, 28.0, "FFmpeg addroi & RLE",
                "Quantization Group: QG=16\n\n" +
                "Normalized Quant Offset:\n" +
                r"$q_{\mathrm{offset}} \in [-0.18, +0.14]$" + "\n\n" +
                "Horizontal Run-Length\nEncoding (RLE) to prevent\nOS Command Overflow\n" +
                r"$\Downarrow$" + "\n" +
                r"x265 CTU $\Delta\mathrm{QP}$ Map",
                bg_col=c_qp_bg, border_col=c_qp_border, text_col=c_qp_text, fontsize=7.6)

    draw_arrow((59.25, 39.5), (59.25, 35.5), col="#EA580C", lw=1.3)

    # Arrow routing Attention from Decoder to Panel 3
    draw_arrow((47.6, 64.7), (52.2, 77.5), col="#D97706", lw=1.4, rad=-0.15, label=r"$A_t$ Map", label_pos=0.45)

    # ==========================================
    # 4. PANEL 4: CODEC & DIFFERENTIABLE PROXY (STE)
    # ==========================================
    draw_container(68.5, 5.0, 15.0, 88.0, "4. Codec & STE Proxy", "Forward x265 / Backward Proxy",
                   bg_col="#F8FAFC", border_col="#94A3B8", header_col="#334155")

    # Real HEVC libx265 Encoder Block
    draw_module(69.8, 52.0, 12.4, 33.0, "Standard-Compliant\nHEVC libx265",
                "FFmpeg libx265 Command:\n" +
                "-vf addroi=...\n" +
                "-c:v libx265\n" +
                "-crf {base_crf}\n" +
                "--qg-size 16\n\n" +
                "Generates Standard HEVC\nBitstream & Real Rate:\n" +
                r"Total BPP ($R_t$)" + "\n" +
                r"$\Downarrow$" + "\n" +
                r"Decoded Frame $\hat{I}_t$",
                bg_col=c_codec_bg, border_col=c_codec_border, text_col=c_codec_text, fontsize=7.8)

    # Differentiable Video Proxy (STE)
    draw_module(69.8, 7.5, 12.4, 38.0, "Differentiable Proxy\n(STE Backward)",
                "Non-differentiable x265\nsurrogate during training:\n\n" +
                r"$\hat{I}_t^{\mathrm{fwd}} = \mathrm{x265}(I_{\mathrm{edit}})$" + "\n" +
                r"$\hat{I}_t^{\mathrm{bwd}} = \mathrm{Proxy}(I_{\mathrm{edit}})$" + "\n\n" +
                "Straight-Through Estimator:\n" +
                r"$\frac{\partial \mathcal{L}}{\partial I_{\mathrm{edit}}} \approx \frac{\partial \mathcal{L}}{\partial \hat{I}_t^{\mathrm{proxy}}}$" + "\n\n" +
                "Enables End-to-End\nGradient Propagation\nWithout x265 Simulation",
                bg_col=c_proxy_bg, border_col=c_proxy_border, text_col=c_proxy_text, fontsize=7.6)

    # Forward routing from Pre-edited frame to Codec
    draw_arrow((45.8, 15.2), (69.8, 68.5), col="#2563EB", lw=1.5, rad=-0.25, label=r"$I_{\mathrm{edit}, t}$", label_pos=0.45)
    # Rate offset routing to Codec
    draw_arrow((66.3, 21.5), (69.8, 60.0), col="#EA580C", lw=1.4, rad=-0.15, label="addroi", label_pos=0.45)

    # Forward and Backward coupling
    draw_arrow((76.0, 52.0), (76.0, 45.5), col="#475569", lw=1.3, label=r"$\hat{I}_t$", label_pos=0.5)
    draw_arrow((73.5, 45.5), (73.5, 52.0), col="#DB2777", lw=1.3, ls="--", label="STE Hook", label_pos=0.5)

    # Backward gradient arrow returning to TemporalSCOPE
    draw_arrow((69.8, 26.5), (45.8, 12.0), col="#DB2777", lw=1.5, ls="--", rad=0.18,
               label=r"$\nabla_{I_{\mathrm{edit}}} \mathcal{L}_{\mathrm{task}}$", label_pos=0.55, label_col="#BE185D")

    # ==========================================
    # 5. PANEL 5: DUAL-DETECTOR TASK EVALUATION & LOSS
    # ==========================================
    draw_container(84.5, 5.0, 14.5, 88.0, "5. Downstream Evaluation", "Dual-Detector VCM Objectives",
                   bg_col="#F8FAFC", border_col="#94A3B8", header_col="#3730A3")

    # Detector A: Faster R-CNN v2
    draw_module(85.8, 69.5, 11.9, 15.5, "Faster R-CNN v2",
                "Two-stage Detector\n" +
                "ResNet-50 + FPN\n" +
                r"Features $P_2 - P_5$" + "\n" +
                r"Task AP: $AP_{50}, AP_{75}$",
                bg_col=c_task_bg, border_col=c_task_border, text_col=c_task_text, fontsize=7.8)

    # Detector B: YOLOv8s
    draw_module(85.8, 50.0, 11.9, 15.5, "YOLOv8s",
                "One-stage Detector\n" +
                "Anchor-free C2f Spine\n" +
                "Decoupled Head\n" +
                r"Task AP: $mAP_{50-95}$",
                bg_col=c_task_bg, border_col=c_task_border, text_col=c_task_text, fontsize=7.8)

    # Multi-Objective Loss Block
    draw_module(85.8, 7.5, 11.9, 38.0, "Multi-Task Loss\nFormulation",
                "Total Loss:\n\n" +
                r"$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{task}}$" + "\n" +
                r"$+ \lambda_R \mathcal{L}_{\mathrm{rate}}$" + "\n" +
                r"$+ \lambda_T \mathcal{L}_{\mathrm{temporal}}$" + "\n" +
                r"$+ \lambda_{\mathrm{bg}} \mathcal{L}_{\mathrm{bg}}$" + "\n\n" +
                "Verified Performance:\n" +
                r"$\bullet\ \mathrm{BDBR}_{50} \leq -23.5\%$" + "\n" +
                r"$\bullet\ \mathrm{BDBR}_{95} \leq -19.8\%$" + "\n" +
                r"$\bullet\ \Delta\mathrm{BPP} < 0$ (Savings)\n" +
                r"$\bullet$ Dual-Detector Zero-Drop",
                bg_col="#F5F3FF", border_col="#6366F1", text_col="#312E81", fontsize=7.5)

    # Routing from Codec to Detectors
    draw_arrow((82.2, 70.0), (85.8, 77.0), col="#1E293B", lw=1.3, label=r"$\hat{I}_t$", label_pos=0.5)
    draw_arrow((82.2, 63.0), (85.8, 57.5), col="#1E293B", lw=1.3, label=r"$\hat{I}_t$", label_pos=0.5)

    # Routing from Detectors to Loss
    draw_arrow((91.75, 69.5), (91.75, 65.5), col="#4F46E5", lw=1.2)
    draw_arrow((91.75, 50.0), (91.75, 45.5), col="#4F46E5", lw=1.2)

    # Bitrate feed to Loss
    draw_arrow((82.2, 55.0), (85.8, 28.0), col="#475569", lw=1.3, rad=-0.18, label=r"$R_t$ (bpp)", label_pos=0.45)

    # Loss Gradients feedback to STE proxy
    draw_arrow((85.8, 20.0), (82.2, 22.0), col="#DB2777", lw=1.3, ls="--", label="Gradients", label_pos=0.5)

    # ==========================================
    # 6. BOTTOM LEGEND BAR
    # ==========================================
    legend_box = FancyBboxPatch((16.0, 0.8), 68.0, 3.2, boxstyle="round,pad=0.15,rounding_size=0.6",
                                facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.1, zorder=7)
    ax.add_patch(legend_box)

    items = [
        (17.5, 2.4, "#2563EB", "-", "Image & Tensor Flow"),
        (33.0, 2.4, "#15803D", "-", r"Recurrent State ($h_t$)"),
        (48.5, 2.4, "#EA580C", "-", r"CTU $\Delta\mathrm{QP}$ Allocation"),
        (66.5, 2.4, "#DB2777", "--", "STE Proxy Gradients (Backward)"),
    ]
    for lx, ly, lcol, lstyle, ltext in items:
        arr = FancyArrowPatch((lx, ly), (lx + 2.6, ly), arrowstyle="-|>", color=lcol,
                                linewidth=1.6, linestyle=lstyle, mutation_scale=8, zorder=8)
        ax.add_patch(arr)
        ax.text(lx + 3.2, ly, ltext, va="center", fontsize=8.0, fontweight="bold", color="#334155", zorder=8)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {output_png} and {output_pdf}")


if __name__ == "__main__":
    out_dir = Path("docs/figures")
    png_path = out_dir / "fig_pipeline_overview.png"
    pdf_path = out_dir / "fig_pipeline_overview.pdf"
    create_pipeline_figure(png_path, pdf_path)
