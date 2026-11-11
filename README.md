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

### 2.3. Điều phối Lượng tử hóa Khối CTU $\Delta\text{QP}$ từ Bản đồ Chú ý (`compute_ctu_qp_offsets`)
Kế thừa tư tưởng của **WA-RDO** nhưng thực thi trực tiếp qua giao diện chuẩn `x265_picture.quantOffsets` mà không cần sửa đổi bitstream:
- Thay vì vứt bỏ bản đồ chú ý $A_t$ sau khi trộn pixel như SCOPE, T-SCOPE hạ mẫu $A_t$ bằng Average Pooling xuống lưới nhóm lượng tử (Quantization Group / CTU kích thước $16\times 16$ hoặc $64\times 64$):
  $$A_{t, (u,v)}^{\text{QG}} = \text{AvgPool}_{16\times 16}(A_t)_{(u,v)}$$
- Tính độ lệch lượng tử hóa cân bằng tổng bằng 0 (Zero-sum $\Delta\text{QP}$):
  $$\mathbf{\Delta Q}_{t, (u,v)} = \text{clip}\Big(\kappa_{\max} \cdot \big(\bar{A}_t^{\text{QG}} - A_{t, (u,v)}^{\text{QG}}\big), \, -\kappa_{\max}, \, +\kappa_{\max}\Big), \quad \kappa_{\max} = 4.0$$
- **Cơ chế tác động kép:**
  - Khối chứa vật thể ($A_{t,(u,v)}^{\text{QG}} > \bar{A}_t^{\text{QG}}$) nhận $\mathbf{\Delta Q} < 0$ $\to$ giảm bước lượng tử và giảm hệ số nhân Lagrange $\lambda_{\text{RDO}} = 0.85 \cdot 2^{(\text{QP} + \Delta Q - 12)/3}$ trong `x265`, bảo toàn cấu trúc tần số cao cho mạng Detector.
  - Khối thuộc vùng nền ($A_{t,(u,v)}^{\text{QG}} < \bar{A}_t^{\text{QG}}$) vừa đã được làm phẳng ở mức pixel, vừa nhận $\mathbf{\Delta Q} > 0$ trong `x265`.
  - Hỗ trợ thêm chế độ **IWA-QP (`reuse_iframe_across_gop=True`)**: chỉ tính $\mathbf{\Delta Q}$ tại I-frame và dùng lại cho toàn bộ mini-GOP.

### 2.4. Huấn luyện Hybrid Forward-`x265` / Backward-Proxy (Straight-Through Estimator)
Để xóa bỏ hiện tượng lệch (codec mismatch) giữa bộ mô phỏng và `x265` thật:
- **Forward Pass:** Nén chuỗi $I_{\text{edit}, 1:T}$ qua `libx265` thật để nhận video giải mã chính xác $\hat{I}_{\text{x265}, t}$ và số bit thực tế $R_{\text{x265}}$.
- **Backward Pass:** Truyền đạo hàm qua [`DifferentiableVideoProxy`](./video_codecs/hybrid_x265_ste.py) — bộ mô phỏng biến đổi DCT $8\times 8$ có phân biệt mã hóa Intra ($t=0$) và mã hóa phần dư liên khung Inter ($t > 0$, $I_{\text{edit}, t} - \hat{I}_{t-1}$) với bước lượng tử hóa phụ thuộc cả `base_qp` lẫn $\mathbf{\Delta Q}_t$:
  $$\hat{I}_t = \hat{I}_{\text{proxy}, t} + \text{stop\_grad}\big(\hat{I}_{\text{x265}, t} - \hat{I}_{\text{proxy}, t}\big)$$

### 2.5. Hàm Mục tiêu 5 Thành phần (`VCMTemporalPreEditLoss`)
Chỉ huấn luyện duy nhất mạng `TemporalSCOPE` (**3.80M tham số**), giữ đóng băng hoàn toàn mạng Detector:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \mathcal{L}_{\text{att}} + \mathcal{L}_{\text{bg}} + \mathcal{L}_{\text{temp}} + \mathcal{L}_{\text{reg}}$$

1. **Task Feature Consistency ($\mathcal{L}_{\text{task}}$):** Khoảng cách Smooth-$L_1$ có mặt nạ $M_{\text{det}, t}^\ell$ giữa đặc trưng FPN của ảnh sạch $F_T^\ell(I_t)$ và ảnh sau giải mã $F_S^\ell(\hat{I}_t)$.
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

# 2. Huấn luyện mới (chỉ sau xác nhận; không thay thế các kernel đang chạy)
python -I scripts/kaggle_pool_orchestrator.py --pool_file "C:\Users\Wagur1\Downloads\pool (2).json" --action push --run_mode train --epochs 60 --patience 10 --min_epochs 12 --lr 2e-4 --batch_size 8 --ablation_sweep

# 3. Xem trạng thái chạy của cả 21 tài khoản
python -I scripts/kaggle_pool_orchestrator.py --pool_file "C:\Users\Wagur1\Downloads\pool (2).json" --action status

# 4. Tải các output hoàn tất vào thư mục riêng từng kernel; không tự gộp kết quả thiếu
python -I scripts/kaggle_pool_orchestrator.py --pool_file "C:\Users\Wagur1\Downloads\pool (2).json" --action pull --output_dir ./outputs_kaggle_pool

# 5. Gộp từ manifest đã kiểm tra; xem HANDOVER_TASK_BRIEF.md cho validation và eval_soup
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


## 5. Kết quả Thực nghiệm (`libx265` thật + `ffprobe pkt_size` + GT AP trên tập Held-Out `MOT17-11` & `MOT17-13`)

> **Cập nhật handover 2026-11-12:** Đã thu thập và kiểm tra đủ **21/21 checkpoint hội tụ** (`collection_complete_majwx85s`); gộp và hoàn tất validation đầy đủ trên 525 frame `MOT17-09` cho **cả 5/5 Model Soup** (`validated_all_soups_20261112`, `release_ready=true`: `wo_ctu_delta_qp` `12.3009`, `spatial_only_scope` `12.4156`, `proxy_only_zhao` `12.4451`, `iwa_gop_qp_reuse` `12.5949`, `full_tscope` `12.7426`). Pipeline cục bộ đã sửa AP toàn bộ dataset, tách variant, kiểm tra frame ghép cặp, packet bytes và validation protocol; 65 test pass. Sẵn sàng publish 5 soup và dispatch `eval_soup` trên 21 shard để lấy bảng BDBR cuối cùng ngay khi được duyệt. Xem [playbook cập nhật](./HANDOVER_TASK_BRIEF.md).
>
> **Protocol được giữ nguyên:** QG 16×16, scale ΔQP=4 cho full/reuse (có mean-centering), ROI tối đa 8×8 và qoffset clamp `[-0.18,+0.06]`; không phải CTU 64×64/±12. Detector thực là `fasterrcnn_resnet50_fpn_v2`. Đánh giá cuối dùng CRF `[22,27,32,37]`, AP pedestrian 101-point với ignore-IoU policy được công bố, và bitrate payload video-packet MP4 (không gồm container/global headers). Spatial là baseline nhiều thay đổi, không phải ablation chỉ recurrence; `proxy_only_zhao` là baseline proxy-training nội bộ, chưa xác minh tương đương prior art.
>
> **Các bảng bên dưới là số liệu lịch sử:** clip AP từng được tính rồi gộp trung bình, không phải AP toàn bộ dataset. Chỉ dùng để mô tả pilot; không trích làm benchmark cuối cùng. Final merger schema-v2 không chấp nhận các JSON cũ để tạo kết quả cuối.

> **Phạm vi và giới hạn (đọc trước khi trích số):**
> - Nguồn: [`outputs_rigor_gate0_gate1/kaggle_21shards_disjoint_summary.json`](./outputs_rigor_gate0_gate1/kaggle_21shards_disjoint_summary.json), gộp **19/21 shard** hoàn tất lúc thu thập (shard 15 và 18 còn `RUNNING`, chưa có eval). Tổng 374 lượt clip x QP, 9350 bản ghi mã hóa.
> - Train: `MOT17-02, 04, 05, 09, 10`. Test: `MOT17-11, 13` (rời nhau). Mô hình học không nhận GT lúc suy luận (`pred["i_edit"]`, `pred["ctu_qp_offsets"]`). Dòng Oracle dùng mặt nạ GT, chỉ là cận trên.
> - Mỗi shard train **5 epoch** trên ~87 clip, loss giảm rất ít (≈5.75 → 5.68). Mô hình **chưa hội tụ**. Số liệu dưới đây thuộc mô hình huấn luyện nhẹ, không phải trần hiệu năng.
> - **`ROI-PSNR` BDBR của phương pháp học dương** (`tscope_temporal_ctu_qp`: `+21.32%`, `spatial_scope_framewise`: `+19.99%`). Lợi ích nằm ở AP phát hiện, không phải độ trung thực pixel trong vùng ROI.
> - Mỗi shard chỉ đánh giá ~20 clip held-out. Chưa có khoảng tin cậy. Chênh lệch vài điểm giữa các biến thể có thể nằm trong nhiễu.
> - Các bảng 21-shard cũ (proxy codec, mặt nạ Oracle, `-40.09%` BDBR) đã bị thu hồi. Chỉ còn trong lịch sử commit.

### 5.1. Gate 0 (kiểm định can thiệp, pilot cục bộ, 5/5 PASS)

| Gate | Phép thử | Kết quả | Số liệu (`MOT17-11`, `CRF=32`) |
| :--- | :--- | :---: | :--- |
| `G0.1` | Trọng số đã train vs ngẫu nhiên | PASS | `0.09729 bpp` (trained) vs `0.18289 bpp` (random) |
| `G0.2` | Không rò GT lúc suy luận | PASS | đầu vào chỉ `video, family_id, severity` |
| `G0.3` | `libx265` thật + `ffprobe` | PASS | `I,P,P,P`; `per_frame_bpp=[0.31763, 0.01831, 0.03503, 0.01819]` |
| `G0.4` | Can thiệp ΔQP (off/on/inverted) | PASS | `0.19998` / `0.09729` / `0.09558 bpp` |
| `G0.5` | Faster R-CNN thật trên `gt/gt.txt` | PASS | `GT AP@50=13.106%`, `AP@[.5:.95]=5.560%` |

Chi tiết: [`rigor_gate0_gate1_report.json`](./outputs_rigor_gate0_gate1/rigor_gate0_gate1_report.json). Lưu ý `G0.4`: ROI-PSNR giảm khi bật ΔQP (`25.893 → 23.754 dB`), chỉ bitrate giảm rõ.

### 5.2. So sánh tổng thể trên Held-Out Test (BDBR so với `anchor_x265`, tính trên GT AP)

| Phương pháp | Total BDBR (`GT AP@50`) | P-Frame BDBR (`GT AP@50`) | Total BDBR (`GT AP@[.5:.95]`) | Avg Total / P-BPP | Avg `GT AP@50` / `AP@[.5:.95]` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `anchor_x265` | `+0.00%` | `+0.00%` | `+0.00%` | `0.2358` / `0.1107` | `29.40%` / `17.51%` |
| `deterministic_blur_ema` (không học) | `+33.16%` | `+86.77%` | `+44.14%` | `0.2398` / `0.1403` | `23.74%` / `13.17%` |
| `spatial_scope_framewise` | `-17.60%` | `-17.79%` | `-14.86%` | `0.1500` / `0.0605` | `26.52%` / `15.34%` |
| **`tscope_temporal_ctu_qp`** (gộp mọi variant) | `-18.77%` | `-18.48%` | `-15.15%` | `0.1526` / `0.0631` | `26.87%` / `15.46%` |
| `oracle_gt_mask_upper_bound` (cận trên, dùng GT) | `-27.53%` | `-27.74%` | `-31.54%` | `0.1418` / `0.0605` | `28.75%` / `17.59%` |

Đọc kết quả: T-SCOPE hơn Spatial SCOPE `1.17` điểm BDBR trên AP@50 (`-18.77%` vs `-17.60%`). Khoảng cách nhỏ. Oracle (`-27.53%`) còn xa, nên chất lượng bản đồ chú ý tự dự đoán là nút thắt.

### 5.3. Ablation theo variant (`tscope_temporal_ctu_qp`, mỗi variant đo với anchor cùng tập clip)

| Variant | Số clip | Total BDBR (`GT AP@50`) | P-Frame BDBR | Total BDBR (`GT AP@[.5:.95]`) | Avg `GT AP@50` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `wo_ctu_delta_qp` | 79 | `-14.85%` | `-14.20%` | `-12.28%` | `27.23%` |
| `spatial_only_scope` | 79 | `-16.54%` | `-15.44%` | `-14.78%` | `25.89%` |
| `proxy_only_zhao` | 78 | `-17.67%` | `-17.45%` | `-12.72%` | `27.01%` |
| `iwa_gop_qp_reuse` | 59 | `-19.99%` | `-19.01%` | `-15.72%` | `27.22%` |
| `full_tscope` | 79 | `-24.84%` | `-26.69%` | `-20.20%` | `27.07%` |

`full_tscope` đạt `-24.84%` (tốt nhất). Các variant còn lại gần nhau, chưa đủ bằng chứng để kết luận đóng góp từng thành phần.

### 5.4. Tổng quát hóa đa detector (pilot cục bộ, 2 clip held-out, mô hình train 12 epoch trên 5 clip)

Pilot **không ủng hộ** tuyên bố zero-shot: Full T-SCOPE có BDBR (`FeatSNR`) dương trên cả 4 detector. BDBR `GT AP@50` chỉ âm trên RetinaNet (`-0.92%`) và SSDLite (`-8.85%`), dương trên Faster R-CNN (`+23.21%`) và FCOS (`+7.16%`). Swin-T chỉ đo đặc trưng, không tính. Cần chạy lại với checkpoint Kaggle và nhiều clip hơn. Số liệu: [`cross_backbone_bdbr.json`](./outputs_rigor_gate0_gate1/cross_backbone_bdbr.json).

