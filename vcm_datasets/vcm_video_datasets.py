import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image


# Official MPEG-VCM / HEVC CTC 18 sequences in SFU-HW-Objects-v1 (Choi & Bajic)
SFU_HW_SEQUENCES: Dict[str, Tuple[int, int, int, str]] = {
    # SeqName: (Width, Height, FPS, Class)
    "Traffic": (2560, 1600, 30, "A"),
    "PeopleOnStreet": (2560, 1600, 30, "A"),
    "Kimono": (1920, 1080, 24, "B"),
    "ParkScene": (1920, 1080, 24, "B"),
    "Cactus": (1920, 1080, 50, "B"),
    "BasketballDrive": (1920, 1080, 50, "B"),
    "BQTerrace": (1920, 1080, 60, "B"),
    "BasketballDrill": (832, 480, 50, "C"),
    "BQMall": (832, 480, 60, "C"),
    "PartyScene": (832, 480, 50, "C"),
    "RaceHorsesC": (832, 480, 30, "C"),
    "BasketballPass": (416, 240, 50, "D"),
    "BQSquare": (416, 240, 60, "D"),
    "BlowingBubbles": (416, 240, 50, "D"),
    "RaceHorses": (416, 240, 30, "D"),
    "FourPeople": (1280, 720, 60, "E"),
    "Johnny": (1280, 720, 60, "E"),
    "KristenAndSara": (1280, 720, 60, "E"),
}


def get_shard_slice(items: List[Any], shard_id: int, num_shards: int) -> List[Any]:
    """Deterministically partitions items across `num_shards` Kaggle accounts."""
    if not (0 <= shard_id < num_shards):
        raise ValueError(f"shard_id={shard_id} must be in [0, {num_shards - 1}]")
    return [item for idx, item in enumerate(items) if idx % num_shards == shard_id]


def parse_mot_gt_file(
    gt_path: Path,
) -> Tuple[Dict[int, List[Tuple[float, float, float, float]]], Dict[int, List[Tuple[float, float, float, float]]]]:
    """
    Parses MOT17 / MOT20 ground-truth file `gt/gt.txt` following official MOTChallenge protocol (R04 fix).
    Format per line: frame_id, track_id, x, y, w, h, mark, class_id, visibility
      - Active pedestrian targets: `mark == 1 and class_id == 1` (Pedestrian).
      - Ignore / distractor regions: `mark == 0 or class_id in (2, 7, 8, 12)` (Person on vehicle, static person, distractor, reflection).
      - Non-person vehicle/object classes (`class_id in (3, 4, 5, 6, 9, 10, 11)`) are excluded.
    Returns:
        (active_boxes_by_frame, ignore_boxes_by_frame) mapping 1-based frame_id -> list of (x1, y1, x2, y2).
    """
    active_boxes: Dict[int, List[Tuple[float, float, float, float]]] = {}
    ignore_boxes: Dict[int, List[Tuple[float, float, float, float]]] = {}
    if not gt_path.exists():
        return active_boxes, ignore_boxes

    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame_id = int(float(parts[0]))
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            if w <= 1.0 or h <= 1.0:
                continue
            box = (x, y, x + w, y + h)
            if len(parts) >= 8:
                mark = int(float(parts[6]))
                cls_id = int(float(parts[7]))
                if mark == 1 and cls_id == 1:
                    active_boxes.setdefault(frame_id, []).append(box)
                elif mark == 0 or cls_id in (2, 7, 8, 12):
                    ignore_boxes.setdefault(frame_id, []).append(box)
            else:
                active_boxes.setdefault(frame_id, []).append(box)
    return active_boxes, ignore_boxes


def parse_yolo_label_file(
    txt_path: Path, width: int, height: int
) -> List[Tuple[float, float, float, float]]:
    """
    Parses SFU-HW-Objects-v1 YOLO format label file (`class_id cx cy w h` normalized in [0,1]).
    """
    boxes: List[Tuple[float, float, float, float]] = []
    if not txt_path.exists():
        return boxes
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cx, cy, bw, bh = (
                float(parts[1]) * width,
                float(parts[2]) * height,
                float(parts[3]) * width,
                float(parts[4]) * height,
            )
            x1, y1 = max(0.0, cx - 0.5 * bw), max(0.0, cy - 0.5 * bh)
            x2, y2 = min(float(width), cx + 0.5 * bw), min(float(height), cy + 0.5 * bh)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes


def boxes_to_masks(
    boxes: List[Tuple[float, float, float, float]],
    height: int,
    width: int,
    context_pad: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs spatial masks M_attn and M_det from bounding boxes (SCOPE Eq. 11-12):
      - M_attn: Tight object mask [1, H, W]
      - M_det : Expanded detector context mask [1, H, W] (M_bg = 1 - M_det)
    """
    m_attn = torch.zeros((1, height, width), dtype=torch.float32)
    m_det = torch.zeros((1, height, width), dtype=torch.float32)

    for x1, y1, x2, y2 in boxes:
        ix1 = max(0, min(width - 1, int(round(x1))))
        iy1 = max(0, min(height - 1, int(round(y1))))
        ix2 = max(ix1 + 1, min(width, int(round(x2))))
        iy2 = max(iy1 + 1, min(height, int(round(y2))))
        m_attn[0, iy1:iy2, ix1:ix2] = 1.0

        dx1 = max(0, ix1 - context_pad)
        dy1 = max(0, iy1 - context_pad)
        dx2 = min(width, ix2 + context_pad)
        dy2 = min(height, iy2 + context_pad)
        m_det[0, dy1:dy2, dx1:dx2] = 1.0

    return m_attn, m_det


def read_raw_yuv420_frame(
    yuv_path: Path, width: int, height: int, frame_idx: int
) -> torch.Tensor:
    """
    Reads a single frame from an uncompressed 8-bit YUV 4:2:0 planar (.yuv) file
    (standard MPEG-VCM / SFU-HW-Objects-v1 format) and converts to RGB tensor [3, H, W] in [0, 1].
    """
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    frame_bytes = y_size + 2 * uv_size

    with open(yuv_path, "rb") as f:
        f.seek(frame_idx * frame_bytes)
        raw = f.read(frame_bytes)
        if len(raw) < frame_bytes:
            raise EOFError(f"Frame {frame_idx} out of range in {yuv_path}")

    buf = np.frombuffer(raw, dtype=np.uint8)
    y = buf[:y_size].reshape((height, width)).astype(np.float32)
    u = (
        buf[y_size : y_size + uv_size]
        .reshape((height // 2, width // 2))
        .repeat(2, axis=0)
        .repeat(2, axis=1)
        .astype(np.float32)
    )
    v = (
        buf[y_size + uv_size :]
        .reshape((height // 2, width // 2))
        .repeat(2, axis=0)
        .repeat(2, axis=1)
        .astype(np.float32)
    )

    # BT.601 YUV to RGB conversion
    y_val = y - 16.0
    u_val = u - 128.0
    v_val = v - 128.0

    r = 1.164 * y_val + 1.596 * v_val
    g = 1.164 * y_val - 0.392 * u_val - 0.813 * v_val
    b = 1.164 * y_val + 2.017 * u_val

    rgb = np.stack([r, g, b], axis=0) / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    return torch.from_numpy(rgb.astype(np.float32))


# Disjoint Train / Validation / Held-Out Test Split on MOT17 (Gate 1 / F08 fix).
# Validation (MOT17-09) is used only for early stopping and LR scheduling; test (11, 13) is never touched
# until final evaluation, so convergence is decided without test-set leakage.
MOT17_TRAIN_PREFIXES = ("MOT17-02", "MOT17-04", "MOT17-05", "MOT17-10")
MOT17_VAL_PREFIXES = ("MOT17-09",)
MOT17_TEST_PREFIXES = ("MOT17-11", "MOT17-13")


class RealVCMVideoDataset(Dataset):
    """
    Unified Sharded Dataset for MOT17, SFU-HW-Objects-v1, and BDD100K-MOT.
    Automatically discovers sequences under `data_root`:
      - MOT17 structure: `data_root/**/img1/*.jpg` + `gt/gt.txt`
      - SFU-HW structure: `data_root/**/*.yuv` + `labels/<seq>_<frame>.txt`
      - Generic image sequence structure: `data_root/<seq>/*.jpg|png`
    Slices all discovered multi-frame clips across `num_shards` Kaggle accounts.
    When `enforce_disjoint_split=True`, strictly separates MOT17 sequences into:
      - Train sequences: MOT17-02, MOT17-04, MOT17-05, MOT17-09, MOT17-10
      - Held-Out Test sequences: MOT17-11 (moving street camera), MOT17-13 (moving bus camera)
    """

    def __init__(
        self,
        data_root: str,
        shard_id: int = 0,
        num_shards: int = 20,
        clip_len: int = 4,
        stride: int = 4,
        img_size: int = 512,
        is_train: bool = True,
        enforce_disjoint_split: bool = False,
        split: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root) if data_root else None
        self.clip_len = clip_len
        self.stride = stride
        self.img_size = img_size
        self.is_train = is_train
        self.enforce_disjoint_split = enforce_disjoint_split
        # `split` in {"train", "val", "test"} selects MOT17 sequences; default follows `is_train`.
        self.split = split if split is not None else ("train" if is_train else "test")
        if self.split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {self.split}")

        all_clips = self._discover_clips()
        self.clips = get_shard_slice(all_clips, shard_id, num_shards)

    def _discover_clips(self) -> List[Dict[str, Any]]:
        clips: List[Dict[str, Any]] = []
        if not self.data_root or not self.data_root.exists():
            return clips

        # 1. Discover MOT17 / MOT20 sequences (contain `img1` directory)
        for img1_dir in sorted(self.data_root.rglob("img1")):
            if not img1_dir.is_dir():
                continue
            seq_dir = img1_dir.parent
            # MOT17 has duplicate folders for SDP/FRCNN/DPM; keep only one copy per video sequence
            if any(tag in seq_dir.name for tag in ("-SDP", "-DPM")):
                continue
            if self.enforce_disjoint_split and seq_dir.name.startswith("MOT17-"):
                allowed = {
                    "train": MOT17_TRAIN_PREFIXES,
                    "val": MOT17_VAL_PREFIXES,
                    "test": MOT17_TEST_PREFIXES,
                }[self.split]
                if not any(seq_dir.name.startswith(p) for p in allowed):
                    continue
            gt_path = seq_dir / "gt" / "gt.txt"
            if not gt_path.exists():
                continue
            gt_map, ignore_map = parse_mot_gt_file(gt_path)
            frames = sorted(img1_dir.glob("*.jpg")) + sorted(img1_dir.glob("*.png"))
            for start in range(0, len(frames) - self.clip_len + 1, self.stride):
                clips.append(
                    {
                        "type": "mot_frames",
                        "seq_name": seq_dir.name,
                        "frame_paths": frames[start : start + self.clip_len],
                        "gt_map": gt_map,
                        "ignore_map": ignore_map,
                    }
                )

        # 2. Discover SFU-HW-Objects-v1 raw YUV sequences (`*.yuv`)
        for yuv_path in sorted(self.data_root.rglob("*.yuv")):
            # Parse WxH from filename e.g. BasketballDrive_1920x1080_50.yuv or SFU table
            m = re.search(r"_(\d+)x(\d+)_", yuv_path.name)
            if m:
                w, h = int(m.group(1)), int(m.group(2))
            else:
                prefix = yuv_path.stem.split("_")[0]
                w, h = SFU_HW_SEQUENCES.get(prefix, (416, 240, 30, "D"))[:2]

            frame_bytes = int(w * h * 1.5)
            num_frames = yuv_path.stat().st_size // max(1, frame_bytes)
            label_dir = yuv_path.parent / "labels" / yuv_path.stem
            for start in range(0, num_frames - self.clip_len + 1, self.stride):
                clips.append(
                    {
                        "type": "sfu_yuv",
                        "seq_name": yuv_path.stem,
                        "yuv_path": yuv_path,
                        "width": w,
                        "height": h,
                        "start_frame": start,
                        "label_dir": label_dir,
                    }
                )

        return clips

    def __len__(self) -> int:
        return len(self.clips)

    @staticmethod
    def _pack_boxes_tensor(boxes_list: List[torch.Tensor], max_boxes: int = 256) -> torch.Tensor:
        """Pads variable-length per-frame boxes into a fixed [T, max_boxes, 4] tensor (-1 padded, never silently truncates)."""
        t = len(boxes_list)
        out = torch.full((t, max_boxes, 4), -1.0, dtype=torch.float32)
        for f_idx, b_t in enumerate(boxes_list):
            if b_t.numel() == 0:
                continue
            if b_t.shape[0] > max_boxes:
                raise RuntimeError(f"Frame has {b_t.shape[0]} boxes exceeding max_boxes={max_boxes}")
            n = b_t.shape[0]
            out[f_idx, :n] = b_t[:n]
        return out

    @staticmethod
    def _transform_boxes_list(
        raw_boxes_per_frame: List[List[Tuple[float, float, float, float]]],
        target: int,
        sx: float = 1.0,
        sy: float = 1.0,
        x0: float = 0.0,
        y0: float = 0.0,
    ) -> torch.Tensor:
        out_boxes: List[torch.Tensor] = []
        for f_boxes in raw_boxes_per_frame:
            valid = []
            for x1, y1, x2, y2 in f_boxes:
                nx1 = max(0.0, min(float(target), (x1 - x0) * sx))
                ny1 = max(0.0, min(float(target), (y1 - y0) * sy))
                nx2 = max(0.0, min(float(target), (x2 - x0) * sx))
                ny2 = max(0.0, min(float(target), (y2 - y0) * sy))
                if (nx2 - nx1) >= 2.0 and (ny2 - ny1) >= 2.0:
                    valid.append([nx1, ny1, nx2, ny2])
            out_boxes.append(
                torch.tensor(valid, dtype=torch.float32)
                if valid
                else torch.zeros((0, 4), dtype=torch.float32)
            )
        return RealVCMVideoDataset._pack_boxes_tensor(out_boxes)

    def _crop_or_resize_clip(
        self,
        video: torch.Tensor,
        m_attn: torch.Tensor,
        m_det: torch.Tensor,
        raw_boxes_per_frame: List[List[Tuple[float, float, float, float]]],
        raw_ignore_per_frame: List[List[Tuple[float, float, float, float]]],
        clip_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Object-centered crop of size `img_size x img_size` (divisible by 16 for x265 QG/CTU)
        during training (matching SCOPE Sec. 3.2), or resize during eval, and transforms both
        active pedestrian ground-truth boxes and MOT17 ignore regions into output coordinates.
        """
        t, c, h, w = video.shape
        target = (self.img_size // 16) * 16

        if h < target or w < target or not self.is_train:
            v_out = F.interpolate(video, size=(target, target), mode="bilinear", align_corners=False)
            ma_out = F.interpolate(m_attn, size=(target, target), mode="nearest")
            md_out = F.interpolate(m_det, size=(target, target), mode="nearest")
            sx, sy = float(target) / float(w), float(target) / float(h)
            return (
                v_out,
                ma_out,
                md_out,
                self._transform_boxes_list(raw_boxes_per_frame, target, sx=sx, sy=sy),
                self._transform_boxes_list(raw_ignore_per_frame, target, sx=sx, sy=sy),
            )

        # Find object center from m_attn on first frame if available
        ys, xs = torch.where(m_attn[0, 0] > 0.5)
        if len(ys) > 0:
            cy = int(ys.float().mean().item())
            cx = int(xs.float().mean().item())
            y0 = max(0, min(h - target, cy - target // 2))
            x0 = max(0, min(w - target, cx - target // 2))
        else:
            g = torch.Generator().manual_seed(clip_idx)
            y0 = int(torch.randint(0, h - target + 1, (1,), generator=g).item())
            x0 = int(torch.randint(0, w - target + 1, (1,), generator=g).item())

        return (
            video[:, :, y0 : y0 + target, x0 : x0 + target],
            m_attn[:, :, y0 : y0 + target, x0 : x0 + target],
            m_det[:, :, y0 : y0 + target, x0 : x0 + target],
            self._transform_boxes_list(raw_boxes_per_frame, target, x0=float(x0), y0=float(y0)),
            self._transform_boxes_list(raw_ignore_per_frame, target, x0=float(x0), y0=float(y0)),
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self.clips[idx]
        frames_t, mattn_t, mdet_t = [], [], []
        raw_boxes_per_frame: List[List[Tuple[float, float, float, float]]] = []
        raw_ignore_per_frame: List[List[Tuple[float, float, float, float]]] = []
        has_annotation_source = False

        if meta["type"] == "mot_frames":
            has_annotation_source = True
            gt_map = meta["gt_map"]
            ignore_map = meta.get("ignore_map", {})
            for fpath in meta["frame_paths"]:
                frame_num = int(re.sub(r"\D", "", fpath.stem) or "1")
                img = Image.open(fpath).convert("RGB")
                w, h = img.size
                tensor_img = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
                boxes = gt_map.get(frame_num, [])
                ig_boxes = ignore_map.get(frame_num, [])
                ma, md = boxes_to_masks(boxes, h, w, context_pad=16)
                frames_t.append(tensor_img)
                mattn_t.append(ma)
                mdet_t.append(md)
                raw_boxes_per_frame.append(boxes)
                raw_ignore_per_frame.append(ig_boxes)

        elif meta["type"] == "sfu_yuv":
            w, h = meta["width"], meta["height"]
            start_f = meta["start_frame"]
            has_annotation_source = meta["label_dir"].exists()
            for offset in range(self.clip_len):
                f_idx = start_f + offset
                tensor_img = read_raw_yuv420_frame(meta["yuv_path"], w, h, f_idx)
                lbl_file = meta["label_dir"] / f"{meta['seq_name']}_{f_idx:03d}.txt"
                boxes = parse_yolo_label_file(lbl_file, w, h)
                ma, md = boxes_to_masks(boxes, h, w, context_pad=16)
                frames_t.append(tensor_img)
                mattn_t.append(ma)
                mdet_t.append(md)
                raw_boxes_per_frame.append(boxes)
                raw_ignore_per_frame.append([])

        video = torch.stack(frames_t, dim=0)  # [T, 3, H, W]
        m_attn = torch.stack(mattn_t, dim=0)  # [T, 1, H, W]
        m_det = torch.stack(mdet_t, dim=0)  # [T, 1, H, W]

        # Only synthesize center ROI if the sequence has NO annotation source at all (R04 fix: never invent fake ROI on empty labeled frames)
        if not has_annotation_source and m_det.sum() == 0:
            _, _, h_v, w_v = video.shape
            m_attn[:, :, h_v // 4 : 3 * h_v // 4, w_v // 4 : 3 * w_v // 4] = 1.0
            m_det[:, :, h_v // 5 : 4 * h_v // 5, w_v // 5 : 4 * w_v // 5] = 1.0

        video, m_attn, m_det, gt_boxes, ignore_boxes = self._crop_or_resize_clip(
            video, m_attn, m_det, raw_boxes_per_frame, raw_ignore_per_frame, idx
        )
        return {
            "video": video,
            "m_attn": m_attn,
            "m_det": m_det,
            "gt_boxes": gt_boxes,
            "ignore_boxes": ignore_boxes,
            "clip_id": torch.tensor(idx, dtype=torch.long),
            "seq_name": meta["seq_name"],
        }
