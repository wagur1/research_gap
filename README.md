# T-SCOPE: Temporally-Consistent Semantic Pre-Editing and CTU-Level Rate Allocation for Video Coding for Machines (`x265` / `VVenC`)

Official PyTorch implementation, mathematical formulation, and 21-account Kaggle distributed sharding framework for **T-SCOPE** (targeted at **IEEE TCSVT / IEEE TIP**).

---

## 1. Phân tích các Paper SOTA và Lỗ hổng Nghiên cứu (Research Gaps)

Các nghiên cứu hiện tại về tiền xử lý (Pre-processing / Pre-editing) và tối ưu hóa Rate-Distortion trong **Video Coding for Machines (VCM)** đang gặp 3 nút thắt lớn khi ghép nối với bộ mã hóa video chuẩn như **HEVC (`x265`)** và **VVC (`VVenC`)**:

| # | Paper & Venue | Hướng tiếp cận & Cơ chế cốt lõi | Nhược điểm / Lỗ hổng kỹ thuật (Research Gap) |
|---|---|---|---|
| **1** | **A Preprocessing Framework for Video Machine Vision under Compression**<br>*(Fei Zhao et al., ByteDance & PKU — IEEE DCC 2024 / `arXiv:2512.15331`)* | - Đặt mạng tiền xử lý 2 nhánh (Spatial 2D Conv + Temporal 3D Conv) trước `x264`/`x265`.<br>- Dùng *Differentiable Virtual Video Codec* (tìm kiếm khung $t-1$, lượng tử hóa $f_q \in [30, 50]$, entropy Ballé) khi huấn luyện.<br>- Hàm mục tiêu: $\mathcal{L} = \alpha(\mathcal{L}_D + \lambda \mathcal{L}_R) + \mathcal{L}_{\text{Acc}}$ với $\mathcal{L}_D = \text{MSE}(X, \hat{X})$. Giảm $\sim 16.2\%$ BD-Rate trên `x265`. | 1. **Mâu thuẫn hàm mục tiêu:** Phạt $\text{MSE}(X, \hat{X})$ trên toàn bộ khung hình buộc bộ tiền xử lý phải giữ lại cả chi tiết vùng nền vô ích, đi ngược lại lý thuyết Information Bottleneck.<br>2. **Codec Mismatch:** Chỉ huấn luyện với *Virtual Codec* mà không có `x265` thật trong vòng lặp forward, gây lệch nghiêm trọng ở bitrate thấp.<br>3. **Thiếu điều kiện hóa (Unconditioned):** Phải huấn luyện 10 mô hình riêng biệt cho 10 mạng backbone và chỉ kiểm thử trên tác vụ thô (Action Recognition, Single-Object Tracking), chưa giải quyết bài toán Dense Video Detection / Multi-Object Tracking (MOT). |
| **2** | **SCOPE: Semantic COdec Pre-Editing for Object-Preserving Compression**<br>*(Silvia Dani, Leonardo Galteri, Marco Bertini — `arXiv:2610.39778`, 2026)* | - Tiền biên tập ảnh có điều kiện theo họ codec và mức độ nén $(f, s)$ qua lớp FiLM.<br>- Tách 2 nhánh bất đối xứng: giữ chi tiết vật thể $I_{\text{obj}} = \text{clip}(I + \Delta_{\text{obj}})$ và làm phẳng nền $I_{\text{bg}} = \text{clip}(\text{Blur}(I) + \Delta_{\text{bg}})$, trộn qua bản đồ chú ý $A$: $I_{\text{edit}} = A \odot I_{\text{obj}} + (1 - A) \odot I_{\text{bg}}$.<br>- Chỉ cần giám sát từ mask GT và FPN của Faster R-CNN đóng băng lúc train; lúc suy luận chạy độc lập đạt 84.7 FPS, giảm $21.74\%$ BDBR trên BPG. | 1. **Mù thời gian (Spatial-Only):** Xử lý từng khung hình độc lập. Khi áp dụng lên chuỗi video trước `x265` (cấu trúc IPPP/IBBP), bản đồ chú ý $A_t$ và phần dư $\Delta_{\text{obj}, t}, \Delta_{\text{bg}, t}$ biến thiên hỗn loạn giữa frame $t-1$ và $t$.<br>2. **Phá vỡ dự đoán liên khung (Inter-prediction):** Nhiễu nhấp nháy theo thời gian làm bộ ước lượng chuyển động (Motion Estimation) của `x265` không tìm được khối khớp, đẩy bit của Motion Vector Difference (MVD) và phần dư P/B-frame tăng vọt, thậm chí ép `x265` phải chuyển sang chế độ Intra-CU tốn bit.<br>3. **Lãng phí bản đồ chú ý $A$:** Dự đoán được vùng vật thể quan trọng $A \in [0, 1]^{H\times W}$ nhưng vứt bỏ sau bước trộn pixel thay vì truyền vào bộ điều khiển lượng tử của encoder. |
| **3** | **Task-Switchable Pre-Processor for Image Compression for Multiple Machine Vision Tasks (TSW-Pre)**<br>*(Mingyi Yang et al. — IEEE TCSVT, Vol. 34, No. 7, 2024)* | - Dùng một mạng U-Net dùng chung kết hợp với $K$ bộ điều biến ngữ nghĩa chuyên biệt theo tác vụ (*Task-Attentive Semantic Modulation - TSM*) dùng lớp Spatial Feature Transform (SFT).<br>- Tối ưu kết hợp Rate-Task Loss và FPN Feature Distillation ($\mathcal{L}_{\text{dist}}$). Giảm $32.63\%$ BD-BR khi chuyển giao sang BPG trên COCO Detection. | 1. **Chi phí tính toán cực nặng ở Encoder:** Để tạo đặc trưng ngữ nghĩa $\psi_i$ đưa vào lớp SFT, bộ tiền xử lý bắt buộc phải chạy một mạng **ResNeXt-101** ngay ở đầu vào trong **mọi lần suy luận (inference)** — không khả thi cho mã hóa video thời gian thực ở thiết bị biên.<br>2. **Không thích ứng theo QP và thời gian:** Chỉ thiết kế cho ảnh tĩnh và không điều biến theo mức lượng tử hóa (QP) của bộ mã hóa. |
| **4** | **Preprocessing Enhanced Image Compression for Machine Vision**<br>*(Guo Lu et al. — IEEE TCSVT, Vol. 34, No. 12, 2024 / `arXiv:2206.05650`)* | - Kiến trúc 2 nhánh song song ($1\times 1$ Conv + U-Net) tích hợp *Quantization Adaptation Layer (QAL)* nhận trực tiếp `QP` của codec chuẩn.<br>- **Cơ chế Hybrid Forward-Codec / Backward-Proxy:** Chiều forward chạy BPG thật để lấy ảnh giải mã $\hat{X}$ và bitrate thật $R_t$; chiều backward truyền đạo hàm qua mạng proxy Minnen ($\hat{Y}, R_p$). Giảm $20.3\%$ BDBR trên BPG. | 1. Chỉ hỗ trợ nén Intra trên ảnh tĩnh (BPG / JPEG), không mô hình hóa được dự đoán liên khung (Inter-frame residual) của video.<br>2. Ràng buộc $\mathcal{D}_{\text{pre}} = d(X, \bar{X})$ với hệ số $\beta = 0.5$ trên toàn khung hình ngăn cản mạng xóa triệt để thông tin thừa ở vùng nền như cơ chế tách nhánh của SCOPE. |
| **5** | **Wrapper-Aware Rate-Distortion Optimization in Feature Coding for Machines (WA-RDO)**<br>*(S. Fernández-Menduiña et al., USC & InterDigital — IEEE ICASSP 2026)* | - Trong kiến trúc Feature Coding (MPEG FCTM), thay thế SSE-RDO của codec chuẩn (AVC/HEVC) bằng hàm lỗi bình phương có trọng số dựa trên phác thảo ma trận Jacobian $h(z) = \text{diag}(J_s(z)^\top J_s(z))$.<br>- **IWA-RDO:** Chỉ tính bản đồ quan trọng ở I-frame và tái sử dụng cho các P/B-frame trong cùng GOP, giúp HEVC đạt hiệu năng ngang VVC và giảm $28.13\%$ thời gian mã hóa. | 1. Chỉ áp dụng cho **Feature Coding (FCM)** (nén các bản đồ đặc trưng đã đóng gói thành khung 2D), chưa khai thác cho **Pixel-domain Video Pre-processing**.<br>2. Việc giữ cố định bản đồ trọng số của I-frame cho toàn bộ GOP (IWA-RDO) sẽ bị lệch không gian khi vật thể di chuyển nhanh ở các frame cuối GOP. |
| **6** | **Rate-Accuracy Bounds in Visual Coding for Machines**<br>*(Ivan V. Bajić — IEEE MIPR 2025 / `arXiv:2505.14980`)* | - Thiết lập cận lý thuyết Rate-Accuracy (RA) bằng cách mô hình hóa bài toán thị giác máy như mã hóa nguồn rời rạc không nhớ (DMS) $K$ ký hiệu dưới độ méo Hamming: $R(D_H) \approx A \log_2(K)$.<br>- Chứng minh các hệ thống VCM SOTA hiện tại trên ImageNet ($K=1000$) và COCO Detection còn cách cận lý thuyết từ **$100\times$ đến $1000\times$ về bitrate**. | - Bài báo thuần lý thuyết.<br>- Chỉ ra nguyên lý cốt lõi thông qua **Data Processing Inequality (DPI)**: để tiến gần cận lý thuyết, bộ tiền xử lý bắt buộc phải loại bỏ tối đa thông tin tương hỗ (mutual information) không liên quan đến tác vụ ở cả chiều không gian và chiều thời gian, đồng thời chưng cất đặc trưng ở các tầng sâu (deep FPN layers). |

---

## 2. Phương pháp Đề xuất: T-SCOPE (Đóng góp Cốt lõi của Chúng ta)

Từ bảng phân tích trên, chúng ta đề xuất **T-SCOPE (*Temporally-Consistent Semantic Pre-Editing and CTU-Level Rate Allocation*)** — một khung tiền xử lý video tương thích hoàn toàn với bộ mã hóa chuẩn **`libx265` (HEVC)** (với cấu trúc mở rộng sang `VVenC` / VVC trong công việc tương lai), giải quyết đồng thời cả 3 nút thắt mà không cần gắn mạng backbone nặng ở bộ mã hóa. *(Lưu ý phạm vi `R11`: Các dẫn chiếu tới Data Processing Inequality / Information Bottleneck đóng vai trò nguyên lý định hướng thiết kế hàm mất mát, và thực nghiệm kiểm chứng trong kho mã nguồn này tập trung trên bộ mã hóa thực `libx265` qua `ffmpeg` + `ffprobe` + `addroi`)*:

![T-SCOPE System Overview Pipeline](docs/figures/fig_pipeline_overview.png)
*Hình 1: Sơ đồ kiến trúc tổng quan của T-SCOPE theo tiêu chuẩn bài báo IEEE TCSVT / TIP (Vector PDF: [`docs/figures/fig_pipeline_overview.pdf`](docs/figures/fig_pipeline_overview.pdf)). Pipeline gồm 5 khối chức năng chính: (1) Nhập chuỗi video và vector điều kiện hóa nén $c_t$; (2) Mạng U-Net TemporalSCOPE điều biến FiLM với ConvGRU Bottleneck và tiền biên tập 2 nhánh; (3) Phân phối $\Delta\text{QP}$ CTU bất đối xứng qua FFmpeg `addroi` 16×16; (4) Vòng lặp nén HEVC `libx265` chuẩn với đạo hàm ngược qua STE Proxy; (5) Giám sát tác vụ đa bộ dò hạ nguồn (Faster R-CNN v2 + YOLOv8s).*

```
Input Video {I_t} ──► [TemporalSCOPE (3.80M params, ConvGRU + FiLM)]
                             │
                             ├──► 1. Pre-edited Video {I_edit,t} (Bảo toàn vật thể + Khóa tĩnh nền liên khung)
                             │
                             └──► 2. Bản đồ chú ý A_t ──► Hạ mẫu QG/CTU ──► ΔQP_t (x265 quantOffsets)
                                                                                  │
                                                                                  ▼
Downstream Task (Faster R-CNN / YOLO / ByteTrack) ◄── Giải mã chuẩn ◄── Bộ mã hóa x265 / VVenC
```

### 2.1. Điều kiện hóa Đa Họ Codec và Phi tuyến Mức độ Nén (`SeverityConditioner`)
Thay vì huấn luyện một mô hình riêng cho từng mức QP hay từng bộ mã hóa như Zhao et al., T-SCOPE nhận cặp điều kiện $(f, s)$ với $f \in \{\text{x265}, \text{VVenC}, \text{x264}, \text{Intra}\}$ (nhúng qua $e_f \in \mathbb{R}^8$) và mức độ nén chuẩn hóa $s = \frac{\text{QP} - 22}{25} \in [0, 1]$ được khai triển qua cơ sở đa thức - bản lề 6 chiều:

$$\phi(s) = \big[s, \, s^2, \, s^3, \, h_{0.50}(s), \, h_{0.75}(s), \, h_{0.90}(s)\big], \quad h_\tau(s) = \text{clip}\left(\frac{s - \tau}{1 - \tau}, 0, 1\right)$$

Vector điều kiện $c = \text{MLP}([e_f, \phi(s)]) \in \mathbb{R}^{64}$ điều biến toàn bộ các tầng tích chập của mạng U-Net thông qua cơ chế Feature-wise Linear Modulation:

$$\text{FiLM}(x, c) = x \odot (1 + \gamma(c)) + \beta(c)$$

### 2.2. Tiền biên tập Bất đối xứng Nhất quán Thời gian (`TemporalConvGRU` & Motion-Gated Background Locking)
Để ngăn chặn hiện tượng nhấp nháy liên khung (inter-frame flickering) làm hỏng Motion Estimation của `x265` mà không tốn chi phí tính Optical Flow:
1. **Đầu vào nhận thức biến thiên thời gian:** Ghép kênh khung hình hiện tại $I_t$ và hiệu liên khung $I_t - I_{t-1}$ (6 kênh đầu vào).
2. **Trạng thái ẩn Bottleneck ConvGRU (`TemporalConvGRUCell`):** Đặt tại độ phân giải cổ chai $H/4 \times W/4$:
   $$H_t = \text{ConvGRU}(Z_t, H_{t-1})$$
   giúp bản đồ chú ý vật thể $A_t \in [0, 1]^{1\times H\times W}$ và phần dư bảo vệ vật thể $\Delta_{\text{obj}, t} \in [-0.09, +0.09]^{3\times H\times W}$ dịch chuyển mượt dọc theo quỹ đạo chuyển động.
3. **Nhánh nền khóa tĩnh có cổng chuyển động (Motion-Gated EMA Background Branch):**
   - Nhánh vật thể giữ nguyên độ sắc nét biên:
     $$I_{\text{obj}, t} = \text{clip}(I_t + \Delta_{\text{obj}, t}, 0, 1)$$
   - Nhánh nền kết hợp làm mờ Gaussian $5\times 5$ trong không gian với trung bình trượt hàm mũ (EMA) theo thời gian được điều khiển bởi mặt nạ đứng yên $M_{\text{static}, t} = \exp(-15 \cdot \|I_t - I_{t-1}\|_1)$:
     $$I_{\text{bg\_spatial}, t} = \text{clip}(\text{Blur}_{5\times 5}(I_t) + \Delta_{\text{bg}, t}, 0, 1), \quad \|\Delta_{\text{bg}, t}\|_\infty \le 0.03$$
     $$I_{\text{bg}, t} = w_{\text{temp}, t} \odot I_{\text{bg}, t-1} + (1 - w_{\text{temp}, t}) \odot I_{\text{bg\_spatial}, t}$$
   - Khi vùng nền đứng yên ($M_{\text{static}, t} \to 1$), $I_{\text{bg}, t}$ khóa chặt theo $I_{\text{bg}, t-1}$, triệt tiêu hoàn toàn nhiễu cảm biến giữa các frame và ép `x265` chọn chế độ **Skip/Merge CU** tốn gần 0 bit trên P/B-frames. Khi camera di chuyển ($M_{\text{static}, t} \to 0$), cổng tự động nhả về $I_{\text{bg\_spatial}, t}$ để không gây hiện tượng bóng ma (ghosting).
   - Khung hình xuất ra:
     $$I_{\text{edit}, t} = A_t \odot I_{\text{obj}, t} + (1 - A_t) \odot I_{\text{bg}, t}$$

### 2.3. Điều phối Lượng tử hóa Khối CTU $\Delta\text{QP}$ Bất đối xứng từ Bản đồ Chú ý (`compute_ctu_qp_offsets`)
Kế thừa tư tưởng của **WA-RDO** nhưng thực thi trực tiếp qua giao diện chuẩn `x265_picture.quantOffsets` (`ffmpeg addroi`) trên lưới **`16×16` vùng** (`32×32 px`/ô ở `512×512`) mà không cần sửa đổi bitstream:
- Thay vì vứt bỏ bản đồ chú ý $A_t$ sau khi trộn pixel như SCOPE, T-SCOPE hạ mẫu $A_t$ bằng Average Pooling xuống lưới nhóm lượng tử (Quantization Group kích thước $16\times 16$):
  $$A_{t, (u,v)}^{\text{QG}} = \text{AvgPool}_{16\times 16}(A_t)_{(u,v)}$$
- **Phân bổ $\Delta\text{QP}$ bất đối xứng tiền cảnh / hậu cảnh:** Vì người đi bộ trong MOT17 chỉ chiếm $10\%\text{–}20\%$ diện tích khung hình ($\bar{A}_t^{\text{QG}} \approx 0.15$), chuẩn hóa đối xứng zero-mean truyền thống làm vùng nền chỉ tăng tối đa $+2.5\text{ QP}$ (`qoffset` $\le +0.05$). T-SCOPE chuẩn hóa tách biệt hai nhánh tiền cảnh và hậu cảnh quanh $\bar{A}_t^{\text{QG}}$:
  $$f_{t,(u,v)} = \text{clip}\left(\frac{A_{t,(u,v)}^{\text{QG}} - \bar{A}_t^{\text{QG}}}{\max(A_t^{\text{QG}}) - \bar{A}_t^{\text{QG}}}, 0, 1\right), \quad b_{t,(u,v)} = \text{clip}\left(\frac{\bar{A}_t^{\text{QG}} - A_{t,(u,v)}^{\text{QG}}}{\bar{A}_t^{\text{QG}} - \min(A_t^{\text{QG}})}, 0, 1\right)$$
  $$\mathbf{\Delta Q}_{t, (u,v)} = \kappa_{\max} \cdot \big(1.8 \, b_{t,(u,v)} - 1.0 \, f_{t,(u,v)}\big) \in [-4.0, \, +7.2]\text{ QP}, \quad \text{qoffset} \in [-0.18, \, +0.14]$$
- **Cơ chế tác động kép:**
  - Khối chứa vật thể ($A_{t,(u,v)}^{\text{QG}} > \bar{A}_t^{\text{QG}}$) nhận $\mathbf{\Delta Q} \in [-4.0, 0]\text{ QP}$ $\to$ giảm bước lượng tử và giảm hệ số nhân Lagrange $\lambda_{\text{RDO}} = 0.85 \cdot 2^{(\text{QP} + \Delta Q - 12)/3}$ trong `x265`, bảo toàn cấu trúc tần số cao cho cả 2 họ Detector (**Faster R-CNN ResNet50-FPN v2** và **YOLOv8s**).
  - Khối thuộc vùng nền ($A_{t,(u,v)}^{\text{QG}} < \bar{A}_t^{\text{QG}}$) vừa được làm phẳng ở mức pixel, vừa nhận $\mathbf{\Delta Q} \in [0, +7.2]\text{ QP}$ (`qoffset` tới $+0.14$) trên lưới `16×16` có gộp run-length ngang, cắt giảm tới $\sim 50\%$ bitrate vùng nền.
  - Hỗ trợ chế độ **IWA-QP (`reuse_iframe_across_gop=True`)**: chỉ tính $\mathbf{\Delta Q}$ tại I-frame và dùng lại cho toàn bộ mini-GOP.

### 2.4. Huấn luyện Hybrid Forward-`x265` / Backward-Proxy (Straight-Through Estimator)
Để xóa bỏ hiện tượng lệch (codec mismatch) giữa bộ mô phỏng và `x265` thật:
- **Forward Pass:** Nén chuỗi $I_{\text{edit}, 1:T}$ qua `libx265` thật để nhận video giải mã chính xác $\hat{I}_{\text{x265}, t}$ và số bit thực tế $R_{\text{x265}}$.
- **Backward Pass:** Truyền đạo hàm qua [`DifferentiableVideoProxy`](./video_codecs/hybrid_x265_ste.py) — bộ mô phỏng biến đổi DCT $8\times 8$ có phân biệt mã hóa Intra ($t=0$) và mã hóa phần dư liên khung Inter ($t > 0$, $I_{\text{edit}, t} - \hat{I}_{t-1}$) với bước lượng tử hóa phụ thuộc cả `base_qp` lẫn $\mathbf{\Delta Q}_t$:
  $$\hat{I}_t = \hat{I}_{\text{proxy}, t} + \text{stop\_grad}\big(\hat{I}_{\text{x265}, t} - \hat{I}_{\text{proxy}, t}\big)$$

### 2.5. Hàm Mục tiêu 5 Thành phần Đa Detector (`VCMTemporalPreEditLoss` — Faster R-CNN FPN + YOLOv8s CSPDarknet)
Chỉ huấn luyện duy nhất mạng `TemporalSCOPE` (**3.80M tham số**), giữ đóng băng hoàn toàn cả hai mạng Detector (**Faster R-CNN ResNet50-FPN v2** двух-stage và **YOLOv8s** one-stage anchor-free):

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \mathcal{L}_{\text{att}} + \mathcal{L}_{\text{bg}} + \mathcal{L}_{\text{temp}} + \mathcal{L}_{\text{reg}}$$

1. **Dual-Detector Task Feature Consistency ($\mathcal{L}_{\text{task}}$):** Khoảng cách Smooth-$L_1$ có mặt nạ $M_{\text{det}, t}^\ell$ giữa đặc trưng đa tỷ lệ của ảnh sạch $F_T^\ell(I_t)$ và ảnh sau giải mã $F_S^\ell(\hat{I}_t)$ trên cả 4 tầng FPN `{p2, p3, p4, p5}` của **Faster R-CNN ResNet50-FPN v2** lẫn 3 tầng CSPDarknet `{yolo_p3, yolo_p4, yolo_p5}` của **YOLOv8s**.
2. **Attention Supervision ($\mathcal{L}_{\text{att}}$):** Tổng của Binary Cross-Entropy, Dice Loss giữa $A_t$ với mặt nạ hộp giới hạn chặt $M_{\text{attn}, t}$, và phạt độ thưa trên vùng nền $\|(1 - M_{\text{det}, t}) \odot A_t\|_1$.
3. **Spatial Background Simplification & Rate ($\mathcal{L}_{\text{bg}}$):** Làm phẳng độ lệch so với ảnh làm mờ và giảm đạo hàm không gian trong vùng $M_{\text{bg}, t} = 1 - M_{\text{det}, t}$ cộng với ước lượng bitrate $\mathcal{L}_{\text{rate}}$.
4. **Temporal Inter-Frame Residual Suppression ($\mathcal{L}_{\text{temp}}$ — Đóng góp mới cho Video):**
   $$\mathcal{L}_{\text{temp}} = \lambda_{\text{temp\_bg}} \frac{\big\| M_{\text{bg}, t} M_{\text{bg}, t-1} \odot (I_{\text{edit}, t} - I_{\text{edit}, t-1}) \big\|_1}{\sum M_{\text{bg}, t} M_{\text{bg}, t-1} + \epsilon} + \lambda_{\text{temp\_obj}} \|\Delta_{\text{obj}, t} - \Delta_{\text{obj}, t-1}\|_1$$
   Trực tiếp ép phần dư dự đoán liên khung ở vùng nền về 0 và làm mượt phần dư vật thể qua các frame liên tiếp.
5. **Residual Regularization ($\mathcal{L}_{\text{reg}}$):** Phạt chuẩn $L_1$ và Total Variation (TV) lên $\Delta_{\text{obj}, t}, \Delta_{\text{bg}, t}$.

---

## 3. Cấu trúc Thư mục Codebase

```text
├── TCSVT_Paper_Blueprint.md          # Đề cương chi tiết bài báo IEEE TCSVT & bảng biểu thực nghiệm
├── models/
│   └── temporal_scope.py             # Mạng TemporalSCOPE (3.80M params), FiLM, ConvGRU, CTU Delta-QP
├── video_codecs/
│   └── hybrid_x265_ste.py            # DifferentiableVideoProxy (Intra/Inter DCT) & HybridX265STE
├── losses/
│   └── vcm_temporal_loss.py          # Hàm loss 5 thành phần VCMTemporalPreEditLoss
├── vcm_datasets/
│   └── vcm_video_datasets.py         # Bộ nạp dữ liệu thật: MOT17 (gt.txt), SFU-HW-Objects-v1 (.yuv), BDD100K
├── scripts/
│   ├── prepare_kaggle_datasets.py    # Tự động dò /kaggle/input/mot17 & tải nhãn chuẩn MPEG-VCM SFU-HW
│   ├── kaggle_shard_runner.py        # Worker chạy đánh giá BDBR (Wish-List #2) và huấn luyện theo Shard
│   ├── merge_shards.py               # Tính BDBR (%) chuẩn VCEG-M33 & gộp trọng số Model Soup (FedAvg)
│   └── kaggle_pool_orchestrator.py   # Script điều phối tự động 21 tài khoản Kaggle từ máy cá nhân
└── train_and_verify.py               # Script kiểm chứng nhanh toàn bộ pipeline Forward/Backward
```

---

## 4. Vận hành Cụm 21 Tài khoản Kaggle (>600h GPU `T4 x2` / tuần)

### Cách 1: Điều khiển tự động toàn bộ 21 tài khoản từ máy local (`kaggle_pool_orchestrator.py`)
Không cần mở trình duyệt, chạy trực tiếp trên máy tính cá nhân với file danh sách token `pool.json`:

```powershell
# 1. Kiểm tra xác thực toàn bộ 21 tài khoản Kaggle
python scripts/kaggle_pool_orchestrator.py --pool_file "C:\Users\Wagur1\Downloads\pool (2).json" --action verify_pool

# 2. Huấn luyện hội tụ ở 512x512 với Dual-Detector (Faster R-CNN ResNet50-FPN v2 + YOLOv8s), tự động bỏ qua acc hết quota GPU
python -I scripts/kaggle_pool_orchestrator.py --pool_file "C:\Users\Wagur1\Downloads\pool (2).json" --action push --run_mode train --ablation_sweep --img_size 512 --batch_size 4 --epochs 40 --patience 10 --min_epochs 10 --min_gpu_hours 1.0 --skip_exhausted --use_yolov8s

# 3. Xem trạng thái chạy của cả 21 tài khoản
python -I scripts/kaggle_pool_orchestrator.py --pool_file "C:\Users\Wagur1\Downloads\pool (2).json" --action status

# 4. Tải các output hoàn tất vào thư mục riêng từng kernel; không tự gộp kết quả thiếu
python -I scripts/kaggle_pool_orchestrator.py --pool_file "C:\Users\Wagur1\Downloads\pool (2).json" --action pull --output_dir ./outputs_kaggle_pool

# 5. Gộp checkpoint từ manifest đã kiểm tra (xem thêm docs/V1_PRELIMINARY_BENCHMARK_REPORT.md)
python -I scripts/merge_shards.py --action soup --manifest "<collection_manifest.json>" --ckpt_dir "<new_soup_dir>" --merged_ckpt "<new_soup_dir>/tscope_merged_soup.pt"
```

### Cách 2: Chạy thủ công 1-Cell trên Kaggle Notebook (`GPU T4 x2`)
Nếu chạy trực tiếp trên giao diện web Kaggle (đã gắn Input Dataset `ahmedsamir1598/mot17challenge`):

```python
SHARD_ID = 0        # Đổi từ 0 -> 20 tương ứng 21 tài khoản Kaggle
NUM_SHARDS = 21
VARIANT = "full_tscope"  # Hoặc: spatial_only_scope | wo_ctu_delta_qp | iwa_gop_qp_reuse | proxy_only_zhao

!git clone https://github.com/wagur1/research_gap.git /kaggle/working/research_gap || (cd /kaggle/working/research_gap && git pull)
%cd /kaggle/working/research_gap

!python scripts/prepare_kaggle_datasets.py --target_dir /kaggle/working/data --download_sfu_labels

!python scripts/kaggle_shard_runner.py \
    --mode eval_poc \
    --shard_id {SHARD_ID} \
    --num_shards {NUM_SHARDS} \
    --output_dir /kaggle/working/outputs

!python scripts/kaggle_shard_runner.py \
    --mode train \
    --shard_id {SHARD_ID} \
    --num_shards {NUM_SHARDS} \
    --variant {VARIANT} \
    --epochs 10 \
    --batch_size 8 \
    --enable_x265_ste \
    --output_dir /kaggle/working/outputs
```

---


## 5. Kết quả Thực nghiệm Đóng gói v1 (`512×512`, `libx265` + `ffprobe` + 1.648 Frame Held-Out)

> **Báo cáo đầy đủ:** [`docs/V1_PRELIMINARY_BENCHMARK_REPORT.md`](./docs/V1_PRELIMINARY_BENCHMARK_REPORT.md) | **Dữ liệu gốc hợp lệ (`final_ready=true`, `105/105` JSON):** [`outputs_kaggle_pool/merged_benchmarks.json`](./outputs_kaggle_pool/merged_benchmarks.json)
> - **Tập kiểm thử:** `1,648` khung hình không chồng lấn (`MOT17-11`: `900` frame `000001..000900`; `MOT17-13`: `748` frame `000001..000748`, loại 2 frame đuôi `749..750`), resize `bilinear_square` `512×512`, `clip_len=4`.
> - **Giao thức mã hóa & dò:** `ffmpeg -c:v libx265 -preset medium -x265-params aq-mode=2:bframes=0` (`IPPP`, `CRF ∈ {22, 27, 32, 37}`), lưới `addroi` tối đa `16×16` vùng (`qoffset ∈ [-0.18, +0.14]`), bitrate đo từ byte payload video-packet MP4 (`ffprobe -show_packets`), detector `fasterrcnn_resnet50_fpn_v2` và `YOLOv8s` với dataset-level 101-point pedestrian AP.
> - **5/5 Model Soup kiểm chứng trên `MOT17-09` (`release_ckpts/`):** `full_tscope` (`val_loss=3.141676`), `iwa_gop_qp_reuse` (`3.117673`), `wo_ctu_delta_qp` (`3.136038`), `proxy_only_zhao` (`1.959265`), `spatial_only_scope` (`2.114932`).

### 5.1. So sánh Tổng thể trên 1.648 Frame Held-Out (`overall_methods_summary`)

*(Giá trị BDBR âm = tiết kiệm bitrate ở cùng AP; giá trị dương = tốn thêm bitrate)*

| Phương pháp | Total BDBR (`FRCNN AP50`) | Total BDBR (`FRCNN AP50:95`) | P-BPP BDBR (`FRCNN AP50`) | P-BPP BDBR (`FRCNN AP50:95`) | Total BDBR (`YOLO AP50`) | Total BDBR (`YOLO AP50:95`) | P-BPP BDBR (`YOLO AP50`) | P-BPP BDBR (`YOLO AP50:95`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `anchor_x265` | `0.00%` | `0.00%` | `0.00%` | `0.00%` | `0.00%` | `0.00%` | `0.00%` | `0.00%` |
| `deterministic_blur_ema` | `+33.87%` | `+36.18%` | `+71.86%` | `+76.36%` | `+27.52%` | `+34.82%` | `+60.45%` | `+73.80%` |
| `spatial_scope_framewise` | **`-16.20%`** | **`-14.44%`** | **`-16.59%`** | **`-14.22%`** | **`-14.69%`** | **`-12.22%`** | **`-14.57%`** | **`-11.11%`** |
| `tscope_temporal_ctu_qp` (`full_tscope`) | `-15.52%` | `-11.52%` | `+18.53%` | `+25.52%` | `-13.86%` | `-9.92%` | `+21.33%` | `+28.31%` |
| `oracle_gt_mask_upper_bound` | **`-43.57%`** | **`-44.25%`** | **`-33.50%`** | **`-34.59%`** | **`-33.74%`** | **`-41.68%`** | **`-17.37%`** | **`-30.30%`** |

### 5.2. Đường cong từng mức CRF (`1,648` frames) & Ablation 5 Biến thể (`ablation_variants_summary`)

| Phương pháp | CRF | Total bpp | I-frame bpp | P-frame bpp | ROI PSNR (dB) | FRCNN AP50 | FRCNN AP50:95 | YOLO AP50 | YOLO AP50:95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `anchor_x265` | 22 / 27 / 32 / 37 | `0.6606 / 0.3352 / 0.1662 / 0.0795` | `1.4567 / 0.8652 / 0.4849 / 0.2538` | `0.3952 / 0.1586 / 0.0600 / 0.0214` | `32.95 / 29.77 / 26.79 / 23.98` | `56.39% / 53.93% / 49.33% / 39.45%` | `33.08% / 31.70% / 28.68% / 22.42%` | `50.46% / 48.55% / 44.76% / 36.33%` | `29.37% / 28.19% / 25.64% / 20.26%` |
| `spatial_scope_framewise` | 22 / 27 / 32 / 37 | `0.5000 / 0.2626 / 0.1351 / 0.0678` | `1.1429 / 0.6914 / 0.3976 / 0.2162` | `0.2857 / 0.1196 / 0.0477 / 0.0184` | `30.76 / 28.50 / 26.09 / 23.66` | `56.66% / 53.93% / 48.57% / 39.48%` | `32.91% / 31.31% / 28.11% / 22.10%` | `50.55% / 48.64% / 44.22% / 36.05%` | `29.26% / 27.94% / 25.02% / 19.74%` |
| `tscope_temporal_ctu_qp` (`full_tscope`) | 22 / 27 / 32 / 37 | `0.3905 / 0.2076 / 0.1079 / 0.0548` | `0.8019 / 0.4871 / 0.2822 / 0.1575` | `0.2534 / 0.1144 / 0.0497 / 0.0206` | `28.33 / 26.70 / 24.80 / 22.81` | `54.02% / 51.25% / 46.66% / 37.68%` | `31.66% / 29.79% / 26.30% / 20.42%` | `48.90% / 46.69% / 42.14% / 33.68%` | `28.32% / 26.61% / 23.42% / 17.72%` |
| `oracle_gt_mask_upper_bound` | 22 / 27 / 32 / 37 | `0.2994 / 0.1626 / 0.0850 / 0.0434` | `0.6676 / 0.4065 / 0.2350 / 0.1306` | `0.1767 / 0.0813 / 0.0350 / 0.0143` | `32.42 / 29.21 / 26.18 / 23.41` | `56.85% / 54.06% / 47.73% / 37.15%` | `32.97% / 31.21% / 27.66% / 21.62%` | `48.71% / 47.15% / 43.22% / 34.46%` | `28.65% / 27.42% / 24.88% / 19.64%` |

| Variant (`tscope_soup_<variant>.pt`) | MOT17-09 Val Loss | Total FRCNN AP50 | Total FRCNN AP50:95 | P-BPP FRCNN AP50 | P-BPP FRCNN AP50:95 | Total YOLO AP50 | Total YOLO AP50:95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `full_tscope` | `3.141676` | `-15.52%` | `-11.52%` | `+18.53%` | `+25.52%` | `-13.86%` | `-9.92%` |
| `spatial_only_scope` | `2.114932` | **`-16.20%`** | **`-14.44%`** | **`-16.59%`** | **`-14.22%`** | `-14.69%` | `-12.22%` |
| `wo_ctu_delta_qp` | `3.136038` | `-8.03%` | `-11.55%` | `+5.88%` | `+0.51%` | **`-15.96%`** | **`-13.62%`** |
| `iwa_gop_qp_reuse` | `3.117673` | `-12.54%` | `-13.85%` | `+19.71%` | `+16.94%` | `-15.09%` | `-11.47%` |
| `proxy_only_zhao` | `1.959265` | `-8.34%` | `-8.79%` | `+7.61%` | `+7.07%` | `-6.97%` | `-7.19%` |

Đồ thị đường cong & ablation: [`docs/figures/fig_bdbr_curves.png`](./docs/figures/fig_bdbr_curves.png), [`docs/figures/fig_ablation_bdbr.png`](./docs/figures/fig_ablation_bdbr.png), [`docs/figures/ablation_preedit_comparison_grid.png`](./docs/figures/ablation_preedit_comparison_grid.png).

### 5.3. Phân rã Theo Chuỗi (`MOT17-11` Camera Cầm tay vs `MOT17-13` Camera Xe buýt Di chuyển Nhanh)

| Sequence (Frames) | Biến thể / Phương pháp | Total FRCNN AP@50 | Total FRCNN AP@50:95 | P-BPP FRCNN AP@50 | P-BPP FRCNN AP@50:95 | Total YOLO AP@50 | Total YOLO AP@50:95 |
|---|---|---:|---:|---:|---:|---:|---:|
| `MOT17-11` (`900`) | `full_tscope` | `-34.43%` | **`-17.31%`** | `-22.50%` | `+5.55%` | `-25.33%` | **`-24.35%`** |
| `MOT17-11` (`900`) | `spatial_only_scope` | `-34.70%` | `-15.11%` | `-40.17%` | `-14.93%` | `+30.47%` | `-16.11%` |
| `MOT17-11` (`900`) | `wo_ctu_delta_qp` | `-23.66%` | `-22.97%` | `-20.57%` | `-19.15%` | `-28.42%` | `-27.34%` |
| `MOT17-11` (`900`) | `iwa_gop_qp_reuse` | `-28.57%` | `-21.49%` | `-15.35%` | `-3.39%` | `-31.76%` | `-27.80%` |
| `MOT17-11` (`900`) | `proxy_only_zhao` | `-26.96%` | `-9.29%` | `-23.31%` | `+2.39%` | `-32.12%` | `-22.86%` |
| `MOT17-11` (`900`) | `oracle_gt_mask_upper_bound` | `-47.01%` | `-41.59%` | `-44.07%` | `-36.38%` | `-13.13%` | `-43.51%` |
| `MOT17-13` (`748`) | `full_tscope` | `-4.07%` | `-0.22%` | `+64.41%` | `+72.02%` | `-1.93%` | `-0.33%` |
| `MOT17-13` (`748`) | `spatial_only_scope` | **`-8.17%`** | **`-7.17%`** | **`-4.50%`** | **`-3.05%`** | **`-10.87%`** | **`-8.15%`** |
| `MOT17-13` (`748`) | `wo_ctu_delta_qp` | `+5.73%` | `+4.86%` | `+38.80%` | `+36.88%` | `-4.50%` | `-3.85%` |
| `MOT17-13` (`748`) | `iwa_gop_qp_reuse` | `-3.08%` | `-2.11%` | `+57.61%` | `+58.90%` | `+0.03%` | `-2.08%` |
| `MOT17-13` (`748`) | `proxy_only_zhao` | `-0.82%` | `-0.33%` | `+30.09%` | `+30.72%` | `-0.17%` | `+2.10%` |
| `MOT17-13` (`748`) | `oracle_gt_mask_upper_bound` | `-52.64%` | `-52.08%` | `-47.46%` | `-46.80%` | `-46.33%` | `-45.37%` |

- **Điểm nghẽn v1 trên `MOT17-13`:** Khi camera lia nhanh, hiệu liên khung `I_t - I_{t-1}` chưa bù chuyển động gây ghosting/làm phình `bg_attn`, kéo giảm `ROI PSNR` (`28.33 dB` vs `32.95 dB` ở CRF 22) và làm phình P-frame BDBR (`+72.02%` ở `FRCNN AP50:95`). Hướng nâng cấp v2 được phát triển riêng tại `wagur1/tscope-v2`.

### 5.4. Độ phức tạp Tính toán & Tốc độ Mã hóa Thực tế (`outputs_kaggle_pool/complexity_fps_report.json`)

| Mô hình / Cấu hình | Tổng Tham số (`M`) | Active (`M`) | GMACs (`256×256`) | GMACs (`512×512`) | Thời gian `libx265` (`ms/frame`) | Tốc độ `libx265` (`FPS`) |
|---|---:|---:|---:|---:|---:|---:|
| Detector (`fasterrcnn_resnet50_fpn_v2`) | `43.7123` | `43.7123` | — | — | — | — |
| `anchor_x265` | `0.0000` | `0.0000` | `0.00` | `0.00` | `23.41 ms` | `42.71 FPS` |
| `spatial_only_scope` | `3.8034` | `1.8122` | `31.3221` | `125.2881` | `23.90 ms` (`+2.06%`) | `41.85 FPS` |
| `wo_ctu_delta_qp` / `proxy_only_zhao` | `3.8034` | `3.8034` | `31.3433` | `125.3731` | `23.69 ms` (`+1.17%`) | `42.22 FPS` |
| `iwa_gop_qp_reuse` | `3.8034` | `3.8034` | `31.3433` | `125.3731` | `24.33 ms` (`+3.91%`) | `41.10 FPS` |
| `full_tscope` | `3.8034` | `3.8034` | `31.3433` | `125.3731` | `24.48 ms` (`+4.56%`) | `40.85 FPS` |

---

## 6. Ghi chú Hiệu năng GPU & Tùy chọn Tăng FPS (`512×512`)

Đo thực tế trên **Kaggle Tesla T4** (`commit b824416`, `TemporalSCOPE` `6.11M` tham số, clip `T=4` ở `512×512`):
- **Inference (`AMP FP16`, `B=1`):** **`19.10 FPS`** (`52.36 ms/frame`, peak VRAM `0.485 GB`).
- **Inference (`FP32`, `B=1`):** **`5.52 FPS`** (`181.21 ms/frame`, peak VRAM `0.610 GB`).
- **Training (`B=2, T=4`, FRCNN + YOLOv8s + `libx265` STE):** **`3.219 GB`** peak VRAM (nhờ gradient checkpointing theo chunk FPN và từng bước `T`), tốc độ `3.562 s/step`.

### Tùy chọn tối ưu FPS nếu cần vượt `30–40+ FPS` ngay trên Tesla T4 (không ép ở đợt train hiện tại)
1. **Thu gọn kênh / lớp ở tầng full-res `512×512`:** Hạ `TemporalSCOPE(base_ch=32)` (thay vì `48`) hoặc chuyển `enc1`/`dec1` sang depthwise-separable conv — giữ nguyên độ phân giải biên `512×512` cho người đi bộ nhỏ & halo `4px` nhưng giảm đáng kể GMACs.
2. **Dual-resolution `PixelUnshuffle(2)` stem:** Đưa khối UNet chính xuống `H/2` (`256×256`) và giữ 1 nhánh residual nhẹ ở `H×W` để vừa tăng mạnh FPS vừa hạn chế thô ô `2×2` ở viền mask.
3. **Tối ưu runtime triển khai:** Dùng `torch.compile` hoặc TensorRT FP16 ở pha suy luận (không làm thay đổi kiến trúc hay BD-rate).
