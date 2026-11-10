import argparse
import os
import subprocess
from pathlib import Path
from typing import Optional


KAGGLE_SEARCH_PATHS = [
    "/kaggle/input/mot17challenge/MOT17/train",
    "/kaggle/input/mot17challenge",
    "/kaggle/input/mot-17/MOT17/train",
    "/kaggle/input/mot-17",
    "/kaggle/input/mot17",
    "/kaggle/input/motchallenge-mot17",
    "/kaggle/input/sfu-hw-objects-v1",
    "/kaggle/input/bdd100k",
]


def find_mounted_kaggle_dataset() -> Optional[Path]:
    """Recursively detects MOT17, SFU-HW-Objects-v1, or BDD100K inside /kaggle/input."""
    kaggle_in = Path("/kaggle/input")
    if kaggle_in.exists():
        # 1. Look for MOT17/MOT20 train sequences containing both `img1` and `gt/gt.txt`
        for gt_file in kaggle_in.rglob("gt.txt"):
            seq_dir = gt_file.parent.parent
            if (seq_dir / "img1").is_dir():
                train_root = seq_dir.parent
                print(f"[Dataset Auto-Detect] Found real MOT train root at: {train_root}")
                return train_root
        # 2. Look for SFU-HW raw YUV files
        for yuv_file in kaggle_in.rglob("*.yuv"):
            print(f"[Dataset Auto-Detect] Found raw YUV root at: {yuv_file.parent}")
            return yuv_file.parent

    for p_str in KAGGLE_SEARCH_PATHS:
        p = Path(p_str)
        if p.exists():
            print(f"[Dataset Auto-Detect] Found mounted Kaggle dataset at: {p}")
            return p
    return None


def download_sfu_hw_annotations(target_dir: Path) -> Path:
    """
    Clones the official SFU-HW-Objects-v1 dataset annotations repository
    (by Hyomin Choi & Ivan Bajic, MPEG-VCM CTC standard).
    """
    sfu_dir = target_dir / "SFU-HW-Objects-v1"
    if not sfu_dir.exists():
        print(f"Cloning official SFU-HW-Objects-v1 labels -> {sfu_dir} ...")
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/hyominchoi/SFU-HW-Objects-v1.git",
                str(sfu_dir),
            ],
            check=True,
        )
    else:
        print(f"SFU-HW-Objects-v1 already present at {sfu_dir}")
    return sfu_dir


def create_mot17_sample_structure(target_dir: Path, num_sequences: int = 4, frames_per_seq: int = 8) -> Path:
    """
    Creates a valid MOT17-formatted benchmark directory (`MOT17/train/<seq>/img1/*.jpg` + `gt/gt.txt`)
    when running on a fresh Kaggle account without `/kaggle/input/mot17` attached, or for instant CI tests.
    """
    from PIL import Image
    import numpy as np

    mot_root = target_dir / "MOT17_sample" / "train"
    for seq_idx in range(1, num_sequences + 1):
        seq_dir = mot_root / f"MOT17-{seq_idx:02d}-FRCNN"
        img1_dir = seq_dir / "img1"
        gt_dir = seq_dir / "gt"
        img1_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(100 + seq_idx)
        h, w = 256, 256
        bg = rng.integers(50, 140, size=(h, w, 3), dtype=np.uint8)

        gt_lines = []
        for f_idx in range(1, frames_per_seq + 1):
            frame = bg.copy()
            # Add slight temporal sensor noise in background
            noise = rng.integers(-6, 7, size=(h, w, 3), dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Moving pedestrian/vehicle target box
            x0 = 24 + f_idx * 6
            y0 = 60 + (seq_idx * 10)
            bw, bh = 56, 80
            frame[y0 : y0 + bh, x0 : x0 + bw] = [210, 80, 60]

            img_path = img1_dir / f"{f_idx:06d}.jpg"
            Image.fromarray(frame).save(img_path, quality=95)
            # MOT17 gt.txt line: frame, id, x, y, w, h, mark, class_id, vis
            gt_lines.append(f"{f_idx},1,{x0},{y0},{bw},{bh},1,1,1.0\n")

        with open(gt_dir / "gt.txt", "w", encoding="utf-8") as f:
            f.writelines(gt_lines)

    print(f"Prepared MOT17-formatted structure at: {mot_root}")
    return mot_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MOT17 & SFU-HW-Objects-v1 on Kaggle")
    parser.add_argument("--target_dir", type=str, default="./data")
    parser.add_argument("--download_sfu_labels", action="store_true")
    parser.add_argument("--init_mot_sample", action="store_true")
    args = parser.parse_args()

    target = Path(args.target_dir)
    target.mkdir(parents=True, exist_ok=True)

    mounted = find_mounted_kaggle_dataset()
    if mounted is not None:
        print(f"Use `--data_root {mounted}` in scripts/kaggle_shard_runner.py")

    if args.download_sfu_labels:
        download_sfu_hw_annotations(target)

    if args.init_mot_sample or mounted is None:
        create_mot17_sample_structure(target)


if __name__ == "__main__":
    main()
