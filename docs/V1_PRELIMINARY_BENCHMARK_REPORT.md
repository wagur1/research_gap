# Báo cáo Kết quả Sơ bộ v1 — T-SCOPE (`512×512`, 21-Shard Paired Evaluation)

**Kho lưu trữ baseline v1:** `wagur1/research_gap`  
**Artifact đóng gói cuối v1:** [`outputs_kaggle_pool/merged_benchmarks.json`](../outputs_kaggle_pool/merged_benchmarks.json) (`source_manifest_sha256: 7bdb7d7178fdefa2ad22096774109072af699af1f44f36e9f9d1b0cc440e6d40`)  
**Cập nhật:** 2026-11-15  

---

## 1. Phạm vi giao thức và Tính toàn vẹn (Verification Scope)

- **Lược đồ & Hợp lệ:** `schema_version = 2`, `eligibility.final_ready = true`, `complete_shard_variants = 105` (21 shards × 5 ablation variants).
- **Tập kiểm thử độc lập (Held-out Test):** `1,648` khung hình đầy đủ (không chồng lấn, `clip_len = 4`, `stride = 4`, resize `bilinear_square` `512×512`):
  - `MOT17-11` (`900` frames, `000001..000900`): camera cầm tay hướng trước trong trung tâm thương mại, người đi bộ lớn ở tiền cảnh.
  - `MOT17-13` (`748` frames, `000001..000748`; loại 2 frame đuôi `000749..000750`): camera gắn trên xe buýt di chuyển nhanh, chuyển động nền toàn cục lớn.
- **Bộ mã hóa (`libx265`):** `preset = medium`, `rate_control = crf` (`CRF ∈ {22, 27, 32, 37}`), `fps = 25`, `aq_mode = 2`, `bframes = 0` (`IPPP`), `qg_size = 16`, lưới `addroi` tối đa `16×16` vùng, `qoffset ∈ [-0.18, +0.14]`, đo bit từ payload gói video MP4.
- **Bộ dò & Thước đo:** `Faster R-CNN ResNet50-FPN v2` (`FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT`) và `YOLOv8s`, tính dataset-level 101-point pedestrian AP (`score_threshold = 0.01`, áp dụng suppress hộp `ignore`).

### Trọng số Soup đã kiểm chứng trên `MOT17-09` (`release_ckpts/`)

| Variant | Số shard | Danh sách shard | Val Loss (`MOT17-09`) | SHA-256 (prefix) |
|---|---:|---|---:|---|
| `full_tscope` | 5 | `0, 5, 10, 15, 20` | `3.141676` | `b73380e9fd0e...` |
| `iwa_gop_qp_reuse` | 4 | `3, 8, 13, 18` | `3.117673` | `8c6b3ee18dbc...` |
| `wo_ctu_delta_qp` | 4 | `2, 7, 12, 17` | `3.136038` | `a3680ac3d8b8...` |
| `proxy_only_zhao` | 4 | `4, 9, 14, 19` | `1.959265` | `54504ada6ffa...` |
| `spatial_only_scope` | 4 | `1, 6, 11, 16` | `2.114932` | `25b65864375d...` |

---

## 2. Kết quả Tổng hợp Toàn tập (`1,648` frames)

### 2.1. So sánh với Anchor `libx265` và Upper Bound (`overall_methods_summary`)

*(Giá trị âm = tiết kiệm bitrate ở cùng AP; giá trị dương = tốn thêm bitrate)*

| Phương pháp | Total BDBR FRCNN AP50 | Total BDBR FRCNN AP50:95 | Total BDBR YOLO AP50 | Total BDBR YOLO AP50:95 |
| --- | ---: | ---: | ---: | ---: |
| `anchor_x265` | `0.00%` | `0.00%` | `0.00%` | `0.00%` |
| `deterministic_blur_ema` | `+33.87%` | `+36.18%` | `+27.52%` | `+34.82%` |
| `spatial_scope_framewise` | **`-16.20%`** | **`-14.44%`** | **`-14.69%`** | **`-12.22%`** |
| `tscope_temporal_ctu_qp` (`full_tscope`) | `-15.52%` | `-11.52%` | `-13.86%` | `-9.92%` |
| `oracle_gt_mask_upper_bound` | **`-43.57%`** | **`-44.25%`** | **`-33.74%`** | **`-41.68%`** |

### 2.2. Bảng Ablation 5 Biến thể (`ablation_variants_summary`)

| Variant | Total FRCNN AP50 | Total FRCNN AP50:95 | Total YOLO AP50 | Total YOLO AP50:95 |
| --- | ---: | ---: | ---: | ---: |
| `full_tscope` | `-15.52%` | `-11.52%` | `-13.86%` | `-9.92%` |
| `spatial_only_scope` | **`-16.20%`** | **`-14.44%`** | `-14.69%` | `-12.22%` |
| `wo_ctu_delta_qp` | `-8.03%` | `-11.55%` | **`-15.96%`** | **`-13.62%`** |
| `iwa_gop_qp_reuse` | `-12.54%` | `-13.85%` | `-15.09%` | `-11.47%` |
| `proxy_only_zhao` | `-8.34%` | `-8.79%` | `-6.97%` | `-7.19%` |

### 2.3. Chi tiết từng điểm CRF (Rate–Accuracy Curves, `1,648` frames)

| Phương pháp | CRF | Total bpp | I-frame bpp | ROI PSNR (dB) | FRCNN AP50 | FRCNN AP50:95 | YOLO AP50 | YOLO AP50:95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `anchor_x265` | 22 | `0.6606` | `1.4567` | `32.95` | `56.39%` | `33.08%` | `50.46%` | `29.37%` |
| `anchor_x265` | 27 | `0.3352` | `0.8652` | `29.77` | `53.93%` | `31.70%` | `48.55%` | `28.19%` |
| `anchor_x265` | 32 | `0.1662` | `0.4849` | `26.79` | `49.33%` | `28.68%` | `44.76%` | `25.64%` |
| `anchor_x265` | 37 | `0.0795` | `0.2538` | `23.98` | `39.45%` | `22.42%` | `36.33%` | `20.26%` |
| `spatial_scope_framewise` | 22 | `0.5000` | `1.1429` | `30.76` | `56.66%` | `32.91%` | `50.55%` | `29.26%` |
| `spatial_scope_framewise` | 27 | `0.2626` | `0.6914` | `28.50` | `53.93%` | `31.31%` | `48.64%` | `27.94%` |
| `spatial_scope_framewise` | 32 | `0.1351` | `0.3976` | `26.09` | `48.57%` | `28.11%` | `44.22%` | `25.02%` |
| `spatial_scope_framewise` | 37 | `0.0678` | `0.2162` | `23.66` | `39.48%` | `22.10%` | `36.05%` | `19.74%` |
| `tscope_temporal_ctu_qp` (`full_tscope`) | 22 | `0.3905` | `0.8019` | `28.33` | `54.02%` | `31.66%` | `48.90%` | `28.32%` |
| `tscope_temporal_ctu_qp` (`full_tscope`) | 27 | `0.2076` | `0.4871` | `26.70` | `51.25%` | `29.79%` | `46.69%` | `26.61%` |
| `tscope_temporal_ctu_qp` (`full_tscope`) | 32 | `0.1079` | `0.2822` | `24.80` | `46.66%` | `26.30%` | `42.14%` | `23.42%` |
| `tscope_temporal_ctu_qp` (`full_tscope`) | 37 | `0.0548` | `0.1575` | `22.81` | `37.68%` | `20.42%` | `33.68%` | `17.72%` |
| `oracle_gt_mask_upper_bound` | 22 | `0.2994` | `0.6676` | `32.42` | `56.85%` | `32.97%` | `48.71%` | `28.65%` |
| `oracle_gt_mask_upper_bound` | 27 | `0.1626` | `0.4065` | `29.21` | `54.06%` | `31.21%` | `47.15%` | `27.42%` |
| `oracle_gt_mask_upper_bound` | 32 | `0.0850` | `0.2350` | `26.18` | `47.73%` | `27.66%` | `43.22%` | `24.88%` |
| `oracle_gt_mask_upper_bound` | 37 | `0.0434` | `0.1306` | `23.41` | `37.15%` | `21.62%` | `34.46%` | `19.64%` |

*(Hình đồ thị tổng hợp: [`docs/figures/fig_bdbr_curves.png`](./figures/fig_bdbr_curves.png), [`docs/figures/fig_ablation_bdbr.png`](./figures/fig_ablation_bdbr.png), [`docs/figures/ablation_preedit_comparison_grid.png`](./figures/ablation_preedit_comparison_grid.png).)*

---

## 3. Phân rã Theo Chuỗi (`MOT17-11` vs `MOT17-13`) & Chẩn đoán Nguyên nhân Gốc

### 3.1. BDBR theo từng chuỗi kiểm thử

| Chuỗi (Số frame) | Biến thể / Phương pháp | Total FRCNN AP50 | Total FRCNN AP50:95 | Total YOLO AP50 | Total YOLO AP50:95 |
| --- | --- | ---: | ---: | ---: | ---: |
| **`MOT17-11`** (`900`) | `full_tscope` | `-34.43%` | **`-17.31%`** | `-25.33%` | **`-24.35%`** |
| **`MOT17-11`** (`900`) | `spatial_only_scope` | `-34.70%` | `-15.11%` | `+30.47%` * | `-16.11%` |
| **`MOT17-11`** (`900`) | `wo_ctu_delta_qp` | `-23.66%` | `-22.97%` | `-28.42%` | `-27.34%` |
| **`MOT17-11`** (`900`) | `iwa_gop_qp_reuse` | `-28.57%` | `-21.49%` | `-31.76%` | `-27.80%` |
| **`MOT17-11`** (`900`) | `proxy_only_zhao` | `-26.96%` | `-9.29%` | `-32.12%` | `-22.86%` |
| **`MOT17-11`** (`900`) | `oracle_gt_mask_upper_bound` | `-47.01%` | `-41.59%` | `-13.13%` | `-43.51%` |
| **`MOT17-13`** (`748`) | `full_tscope` | `-4.07%` | **`-0.22%`** | `-1.93%` | **`-0.33%`** |
| **`MOT17-13`** (`748`) | `spatial_only_scope` | **`-8.17%`** | **`-7.17%`** | **`-10.87%`** | **`-8.15%`** |
| **`MOT17-13`** (`748`) | `wo_ctu_delta_qp` | `+5.73%` | `+4.86%` | `-4.50%` | `-3.85%` |
| **`MOT17-13`** (`748`) | `iwa_gop_qp_reuse` | `-3.08%` | `-2.11%` | `+0.03%` | `-2.08%` |
| **`MOT17-13`** (`748`) | `proxy_only_zhao` | `-0.82%` | `-0.33%` | `-0.17%` | `+2.10%` |
| **`MOT17-13`** (`748`) | `oracle_gt_mask_upper_bound` | `-52.64%` | `-52.08%` | `-46.33%` | `-45.37%` |

*\*Lưu ý ngoại suy PCHIP: trên `MOT17-11`, đường `spatial_only_scope` YOLO AP@50 nằm sát `anchor_x265` ở vùng bão hòa CRF 22–27 (`62.50%` vs `62.49%`), làm tích phân BDBR nhạy dao động spline tại ngưỡng 50; ở thước đo chặt hơn `YOLO AP@50:95`, `spatial_only_scope` đạt `-16.11%` còn `full_tscope` đạt `-24.35%`.*

### 3.2. Điểm nghẽn cốt lõi phát hiện ở v1 (Cơ sở thiết kế v2)

1. **Sụp đổ ở camera chuyển động nhanh (`MOT17-13`):** Trên `MOT17-11`, `full_tscope` vượt `spatial_only_scope` về `FRCNN AP50:95` (`-17.31%` vs `-15.11%`) và `YOLO AP50:95` (`-24.35%` vs `-16.11%`). Nhưng trên `MOT17-13` (xe buýt chạy, chiếm gần 1/2 số clip test), `full_tscope` gần như mất hết lợi thế (`-0.22%` FRCNN AP50:95, `+72.02%` P-BPP50:95), làm kéo tụt kết quả gộp toàn tập xuống sau `spatial_only_scope`.
2. **Rò rỉ làm mờ lên tiền cảnh (`ROI PSNR` & `AP` giảm):** Ở CRF 22 trên toàn tập, `full_tscope` đạt `FRCNN AP50 = 54.02%` (`ROI PSNR = 28.33 dB`) so với `anchor_x265` `56.39%` (`32.95 dB`) và `spatial_scope_framewise` `56.66%` (`30.76 dB`). Trên trực quan [`ablation_preedit_comparison_grid.png`](./figures/ablation_preedit_comparison_grid.png), sai khác `|I_edit - I|` ở `full_tscope` vẫn sáng rõ trên người đi bộ tiền cảnh trong khi `spatial_only_scope` giữ nền đen bảo toàn người đi bộ. Nguyên nhân do `diff_t = I_t - I_{t-1}` toàn cục chưa căn chỉnh làm `bg_attn` phình lên khi camera lia (`0.222 -> 0.396`), `fg_attn` không bão hòa về `1.0`, và `bg_prev` EMA chưa bù chuyển động kéo bóng mờ chồng lên vật thể.
3. **Bù bit I-frame đẩy sang P-frame & tụt `dqp` trên P-frame:** Khi nền nhấp nháy ở `MOT17-13`, `P-frame BPP` ở CRF 37 của `full_tscope` tăng lên `0.0184` so với `0.0165` của `anchor_x265` (tức dùng nhiều bit P-frame hơn mà AP lại thấp hơn: `18.01%` vs `22.11%`), dẫn đến `P-BPP BDBR = +72.02%` trên `MOT17-13` và `+25.52%` toàn tập. Ràng buộc phân bổ $\Delta\text{QP}$ và `P-frame BPP` cần được đưa trực tiếp vào hàm mất mát và cơ chế gating thời gian ở bản nâng cấp v2.
