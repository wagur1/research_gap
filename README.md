# T-SCOPE: Temporally-Consistent Semantic Pre-Editing & CTU Rate Allocation for VCM (`x265` / `VVenC`)

Official codebase and 20-Account Kaggle Sharding Pipeline for **T-SCOPE** (targeting **IEEE TCSVT / TIP**).

---

## 1. Chiến lược Chia Shard cho 20+ Tài khoản Kaggle (600+ giờ GPU T4/tuần)

Mỗi tài khoản Kaggle có **30h/tuần với 2x GPU T4 (32GB tổng VRAM)** nhưng không kết nối mạng nội bộ (NCCL) trực tiếp giữa các tài khoản. Hệ thống [`scripts/kaggle_shard_runner.py`](./scripts/kaggle_shard_runner.py) và [`scripts/merge_shards.py`](./scripts/merge_shards.py) hỗ trợ **3 chế độ chia tải độc lập (Embarrassingly Parallel + Model Soup)**:

### Chế độ A: Chia 20 Shard Đánh giá BD-Rate & Chạy Wish-List #2 (`--mode eval_poc`)
- Toàn bộ tập video kiểm thử (SFU-HW / TVD / BDD100K) $\times$ 5 mức `QP = [27, 32, 37, 42, 47]` được băm tự động theo `--shard_id 0..19 --num_shards 20`.
- Mỗi acc Kaggle chỉ xuất 1 file nhẹ `outputs/eval_shard_XXof20.json` (~20 KB).
- Gom 20 file JSON về máy và chạy `python scripts/merge_shards.py` để tính **BDBR (%)** toàn cục.

### Chế độ B: Chia 5 Nhánh Ablation Song song (`--mode train --variant ...`)
Thay vì chạy tuần tự mất 5 ngày, gán mỗi nhóm 4 acc Kaggle chạy 1 biến thể trong Bảng Ablation (tận dụng tự động cả **2 card T4** trong mỗi notebook qua `nn.DataParallel`):
- **Acc 0–3:** `--variant full_tscope` (Mô hình đầy đủ: Temporal ConvGRU + $\mathcal{L}_{\text{temp}}$ + CTU $\Delta\text{QP}$ + Hybrid `x265` STE)
- **Acc 4–7:** `--variant spatial_only_scope` (Baseline SCOPE áp từng frame — chứng minh mất nhất quán thời gian)
- **Acc 8–11:** `--variant wo_ctu_delta_qp` (Tắt điều phối `quantOffsets` $\Delta\text{QP}$ để đo đóng góp của RDO)
- **Acc 12–15:** `--variant iwa_gop_qp_reuse` (Chỉ tính $\Delta\text{QP}$ ở I-frame và dùng lại cho cả GOP theo kiểu IWA-RDO)
- **Acc 16–19:** `--variant proxy_only_zhao` (Baseline Zhao et al. DCC 2024 — chỉ train với Virtual Codec, không có STE)

### Chế độ C: Gộp Checkpoint Đa Tài khoản bằng Federated Model Soup (DiLoCo / FedAvg)
- Mỗi tài khoản train trên `shard_id` của tập dữ liệu và xuất checkpoint `tscope_<variant>_shardXXof20.pt` (chỉ nặng **~14.5 MB**).
- Chạy `python scripts/merge_shards.py --ckpt_dir ./outputs` để trung bình hóa trọng số (**Model Soup**), giúp tăng khả năng tổng quát hóa (zero-shot sang `VVenC` và các detector mới).

---

## 2. Copy-Paste 1-Cell Để Chạy Ngay Trên Kaggle Notebook (Bật Accelerator: `GPU T4 x2`)

Dán đoạn dưới vào **1 cell duy nhất** trên Kaggle, chỉ cần đổi `SHARD_ID = 0` (từ `0` đến `19` ứng với 20 acc):

```python
# === CELL CHẠY TRÊN KAGGLE (BẬT GPU T4 x2) ===
SHARD_ID = 0        # Đổi từ 0 -> 19 cho 20 tài khoản Kaggle
NUM_SHARDS = 20
VARIANT = "full_tscope"  # Hoặc: spatial_only_scope | wo_ctu_delta_qp | iwa_gop_qp_reuse | proxy_only_zhao

!git clone https://github.com/wagur1/research_gap.git /kaggle/working/research_gap || (cd /kaggle/working/research_gap && git pull)
%cd /kaggle/working/research_gap

# 0. Tự động nhận diện /kaggle/input/mot17 (hoặc bdd100k, sfu-hw-objects-v1) và tải nhãn chuẩn MPEG-VCM SFU-HW
!python scripts/prepare_kaggle_datasets.py --target_dir /kaggle/working/data --download_sfu_labels

# 1. Chạy thử nghiệm Wish-List #2 (So sánh P-frame bitrate giữa Spatial SCOPE vs. T-SCOPE) trên Shard hiện tại
!python scripts/kaggle_shard_runner.py \
    --mode eval_poc \
    --shard_id {SHARD_ID} \
    --num_shards {NUM_SHARDS} \
    --output_dir /kaggle/working/outputs

# 2. Huấn luyện T-SCOPE trên Shard hiện tại (Tự động dùng cả 2 GPU T4 x2)
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

Sau khi các acc chạy xong, tải các file trong `/kaggle/working/outputs` về cùng 1 thư mục và chạy:
```bash
python scripts/merge_shards.py --eval_dir ./outputs --ckpt_dir ./outputs
```
