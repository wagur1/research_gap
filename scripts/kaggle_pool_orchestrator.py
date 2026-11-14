import argparse
import datetime
import time
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.merge_shards import merge_checkpoints_soup, merge_eval_jsons_and_print_table
from scripts.experiment_protocol import sha256_file, VARIANTS, validate_soup_provenance
from scripts.kaggle_output_download import download_kernel_outputs


# 5 Ablation variants mapped cyclically or explicitly across the 21 accounts
ABLATION_ROTATION = [
    "full_tscope",
    "spatial_only_scope",
    "wo_ctu_delta_qp",
    "iwa_gop_qp_reuse",
    "proxy_only_zhao",
]


def load_account_pool(pool_file: str) -> List[Tuple[str, str]]:
    path = Path(pool_file)
    if not path.exists():
        raise FileNotFoundError(f"Pool file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, str] = json.load(f)
    return [(user.strip(), token.strip()) for user, token in data.items()]


def get_authenticated_api(username: str, token: str):
    """Authenticates a single Kaggle account in-process without leaking credentials to disk."""
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = token
    os.environ["KAGGLE_API_TOKEN"] = token
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def build_worker_script(
    shard_id: int,
    num_shards: int,
    variant: str,
    epochs: int,
    batch_size: int,
    img_size: int,
    run_mode: str,
    data_shard_id: int = -1,
    data_num_shards: int = -1,
    patience: int = 0,
    min_epochs: int = 10,
    lr: float = 1e-4,
    git_revision: str = "",
    use_yolov8s: bool = True,
    grad_accum_steps: int = 2,
) -> str:
    """
    Generates the standalone Python script pushed to each Kaggle account's kernel.
    Clones the latest code from https://github.com/wagur1/research_gap.git,
    installs ffmpeg if missing, and writes outputs directly to `/kaggle/working`.
    Run modes: `train`, `eval_poc` (per-shard ckpt), `both`, and `eval_soup`
    (evaluates every variant's merged soup from `release_ckpts/` on this shard's test slice).
    """
    return f'''import os
import subprocess
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

SHARD_ID = {shard_id}
NUM_SHARDS = {num_shards}
VARIANT = "{variant}"
EPOCHS = {epochs}
BATCH_SIZE = {batch_size}
GRAD_ACCUM_STEPS = {grad_accum_steps}
IMG_SIZE = {img_size}
RUN_MODE = "{run_mode}"
DATA_SHARD_ID = {data_shard_id}
DATA_NUM_SHARDS = {data_num_shards}
PATIENCE = {patience}
MIN_EPOCHS = {min_epochs}
LR = {lr}
GIT_REVISION = {git_revision!r}
USE_YOLOV8S = {use_yolov8s!r}

if USE_YOLOV8S:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "ultralytics"], check=False)

repo_dir = Path("/tmp/research_gap")
if repo_dir.exists():
    subprocess.run(["rm", "-rf", str(repo_dir)], check=False)
subprocess.run(
    ["git", "clone", "https://github.com/wagur1/research_gap.git", str(repo_dir)],
    check=True,
)
if GIT_REVISION:
    subprocess.run(["git", "checkout", "--detach", GIT_REVISION], cwd=str(repo_dir), check=True)
    actual_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), text=True).strip()
    if actual_revision != GIT_REVISION:
        raise RuntimeError("Pinned evaluation revision mismatch")

env = os.environ.copy()
env["PYTHONPATH"] = str(repo_dir) + os.pathsep + env.get("PYTHONPATH", "")
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ckpt_path = f"/kaggle/working/tscope_{{VARIANT}}_shard{{SHARD_ID:02d}}of{{NUM_SHARDS:02d}}.pt"

if RUN_MODE in ("train", "both"):
    cmd_train = [
        sys.executable,
        str(repo_dir / "scripts" / "kaggle_shard_runner.py"),
        "--mode", "train",
        "--shard_id", str(SHARD_ID),
        "--num_shards", str(NUM_SHARDS),
        "--data_shard_id", str(DATA_SHARD_ID),
        "--data_num_shards", str(DATA_NUM_SHARDS),
        "--variant", VARIANT,
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--grad_accum_steps", str(GRAD_ACCUM_STEPS),
        "--img_size", str(IMG_SIZE),
        "--lr", str(LR),
        "--patience", str(PATIENCE),
        "--min_epochs", str(MIN_EPOCHS),
        "--enable_x265_ste",
        "--use_pretrained_fpn",
        "--output_dir", "/kaggle/working",
    ]
    if USE_YOLOV8S:
        cmd_train.append("--use_yolov8s")
    subprocess.run(cmd_train, cwd=str(repo_dir), env=env, check=True)

if RUN_MODE in ("eval_poc", "both"):
    cmd_eval = [
        sys.executable,
        str(repo_dir / "scripts" / "kaggle_shard_runner.py"),
        "--mode", "eval_poc",
        "--shard_id", str(SHARD_ID),
        "--num_shards", str(NUM_SHARDS),
        "--variant", VARIANT,
        "--img_size", str(IMG_SIZE),
        "--ckpt", ckpt_path,
        "--use_pretrained_fpn",
        "--output_dir", "/kaggle/working",
    ]
    if USE_YOLOV8S:
        cmd_eval.append("--use_yolov8s")
    subprocess.run(cmd_eval, cwd=str(repo_dir), env=env, check=True)

if RUN_MODE == "eval_soup":
    backup_map = {{4: 0, 5: 1, 6: 2, 7: 3, 10: 8, 11: 9, 13: 12, 15: 14, 16: 20}}
    assigned_shards = [SHARD_ID]
    if SHARD_ID in backup_map:
        assigned_shards.append(backup_map[SHARD_ID])
    cmd_eval = [
        sys.executable,
        str(repo_dir / "scripts" / "kaggle_shard_runner.py"),
        "--mode", "eval_all_variants",
        "--shard_ids", *[str(s) for s in assigned_shards],
        "--num_shards", str(NUM_SHARDS),
        "--img_size", str(IMG_SIZE),
        "--soup_dir", str(repo_dir / "release_ckpts"),
        "--use_pretrained_fpn",
        "--output_dir", "/kaggle/working",
    ]
    if USE_YOLOV8S:
        cmd_eval.append("--use_yolov8s")
    subprocess.run(cmd_eval, cwd=str(repo_dir), env=env, check=True)

print(f"=== Shard {{SHARD_ID}}/{{NUM_SHARDS}} ({{VARIANT}}) finished successfully! ===")
'''


def verify_pool(accounts: List[Tuple[str, str]]) -> None:
    print(f"Verifying {len(accounts)} Kaggle accounts in pool...")
    ok_count = 0
    for idx, (user, token) in enumerate(accounts):
        try:
            api = get_authenticated_api(user, token)
            # Lightweight API call to verify token validity
            _ = api.competitions_list(page=1)
            print(f"  [{idx:02d}/{len(accounts):02d}] {user:<22} : OK (Authenticated)")
            ok_count += 1
        except Exception as e:
            print(f"  [{idx:02d}/{len(accounts):02d}] {user:<22} : FAILED ({e})")
    print(f"\nPool Summary: {ok_count}/{len(accounts)} accounts verified ready.")


def push_all_shards(args: argparse.Namespace, accounts: List[Tuple[str, str]]) -> None:
    if args.run_mode == "eval_soup":
        import torch
        revision = args.git_revision
        if len(revision) != 40 or any(value not in "0123456789abcdef" for value in revision):
            raise ValueError("eval_soup requires --git_revision with a full published commit hash")
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True).strip()
        if revision != head:
            raise ValueError("Pinned evaluation revision must match local reviewed HEAD")
        if subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=ROOT_DIR, text=True).strip():
            raise ValueError("Commit and review local changes before evaluation dispatch")
        for variant in VARIANTS:
            path = ROOT_DIR / "release_ckpts" / f"tscope_soup_{variant}.pt"
            soup = torch.load(path, map_location="cpu", weights_only=True)
            validate_soup_provenance(soup, variant, require_validation=True)
            published = subprocess.check_output(["git", "show", f"{revision}:release_ckpts/{path.name}"], cwd=ROOT_DIR)
            import hashlib
            if hashlib.sha256(published).hexdigest() != sha256_file(path):
                raise ValueError("Local checkpoint differs from pinned published revision")
    num_shards = len(accounts)
    max_Idx = min(num_shards, args.max_shards) if args.max_shards > 0 else num_shards
    print(
        f"Pushing up to {max_Idx}/{num_shards} Kaggle GPU kernels "
        f"(mode={args.run_mode}, dataset={args.dataset_slug}, min_gpu_hours={args.min_gpu_hours:.2f}h)..."
    )

    # Filter out accounts whose weekly GPU quota is exhausted (< args.min_gpu_hours)
    active_shards: List[Tuple[int, str, str, float]] = []
    for shard_id in range(max_Idx):
        user, token = accounts[shard_id]
        try:
            api = get_authenticated_api(user, token)
            q = api.quota_view()
            used_h = q.gpu_quota.time_used.total_seconds() / 3600.0
            tot_h = q.gpu_quota.total_time_allowed.total_seconds() / 3600.0
            rem_h = max(0.0, tot_h - used_h)
            if args.skip_exhausted and rem_h < args.min_gpu_hours:
                print(
                    f"  [Shard {shard_id:02d}/{num_shards:02d}] SKIP {user:<22} "
                    f"(GPU quota exhausted: {rem_h:.2f}h / {tot_h:.1f}h remaining)"
                )
                continue
            active_shards.append((shard_id, user, token, rem_h))
        except Exception as e:
            print(f"  [Shard {shard_id:02d}/{num_shards:02d}] Quota check error on {user}: {e}")

    if not active_shards:
        print("No active GPU accounts with sufficient quota found!")
        return

    # Assign variants evenly across the ACTIVE GPU accounts so every ablation variant
    # gets balanced GPU shards and full training data coverage
    active_variants = [
        ABLATION_ROTATION[idx % len(ABLATION_ROTATION)] if args.ablation_sweep else args.variant
        for idx in range(len(active_shards))
    ]

    for active_idx, (shard_id, user, token, rem_h) in enumerate(active_shards):
        variant = active_variants[active_idx]
        same_variant_indices = [
            i for i in range(len(active_shards)) if active_variants[i] == variant
        ]
        data_shard_id = same_variant_indices.index(active_idx)
        data_num_shards = len(same_variant_indices)
        slug = f"t-scope-vcm-shard-{shard_id:02d}"
        kernel_id = f"{user}/{slug}"

        try:
            api = get_authenticated_api(user, token)
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                worker_code = build_worker_script(
                    shard_id=shard_id,
                    num_shards=num_shards,
                    variant=variant,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    img_size=args.img_size,
                    run_mode=args.run_mode,
                    data_shard_id=data_shard_id,
                    data_num_shards=data_num_shards,
                    patience=args.patience,
                    min_epochs=args.min_epochs,
                    lr=args.lr,
                    git_revision=args.git_revision,
                    use_yolov8s=args.use_yolov8s,
                    grad_accum_steps=getattr(args, "grad_accum_steps", 2),
                )
                (tmp_path / "worker.py").write_text(worker_code, encoding="utf-8")

                metadata = {
                    "id": kernel_id,
                    "title": f"T-SCOPE VCM Shard {shard_id:02d}",
                    "code_file": "worker.py",
                    "language": "python",
                    "kernel_type": "script",
                    "is_private": "true",
                    "enable_gpu": "true",
                    "enable_tpu": "false",
                    "enable_internet": "true",
                    "dataset_sources": [args.dataset_slug] if args.dataset_slug else [],
                    "competition_sources": [],
                    "kernel_sources": [],
                    "model_sources": [],
                }
                (tmp_path / "kernel-metadata.json").write_text(
                    json.dumps(metadata, indent=2), encoding="utf-8"
                )

                res = api.kernels_push(str(tmp_path))
                ref_url = getattr(res, "url", "") or ""
                err_msg = getattr(res, "error", "") or ""
                print(
                    f"  [Shard {shard_id:02d}/{num_shards:02d}] Pushed ({rem_h:5.2f}h rem) -> {kernel_id:<36} "
                    f"| variant={variant:<18} (slice {data_shard_id+1}/{data_num_shards}) | {ref_url} {err_msg}"
                )
        except Exception as e:
            print(f"  [Shard {shard_id:02d}/{num_shards:02d}] ERROR on {user}: {e}")


def check_status_all(args: argparse.Namespace, accounts: List[Tuple[str, str]]) -> None:
    num_shards = len(accounts)
    max_idx = min(num_shards, args.max_shards) if args.max_shards > 0 else num_shards
    print(f"Checking kernel status across {max_idx} accounts...")
    for shard_id in range(max_idx):
        user, token = accounts[shard_id]
        kernel_id = f"{user}/t-scope-vcm-shard-{shard_id:02d}"
        try:
            api = get_authenticated_api(user, token)
            st = api.kernels_status(kernel_id)
            status_str = getattr(st, "status", str(st))
            msg = getattr(st, "failureMessage", "") or ""
            print(f"  [Shard {shard_id:02d}] {kernel_id:<40} : {status_str} {msg}")
        except Exception as e:
            print(f"  [Shard {shard_id:02d}] {kernel_id:<40} : {e}")


def _collect_dir_files(directory: Path) -> List[Dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(directory)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]


def _route_backup_eval_outputs(manifest: Dict[str, Any], collection: Path, num_shards: int) -> None:
    """Moves recovered backup eval_shard_<sid>of<N>_*.json outputs to each target shard entry."""
    pattern = re.compile(rf"^eval_shard_(\d+)of{num_shards:02d}_.+\.json$")
    entry_by_id = {entry["shard_id"]: entry for entry in manifest["shards"]}
    touched_shards = set()

    for host_entry in manifest["shards"]:
        host_dir_str = host_entry.get("directory")
        if not host_dir_str:
            continue
        host_dir = Path(host_dir_str)
        host_sid = int(host_entry["shard_id"])
        for path in sorted(host_dir.rglob("*.json")):
            if not path.is_file() or path.is_symlink():
                continue
            match = pattern.match(path.name)
            if not match:
                continue
            target_sid = int(match.group(1))
            if target_sid == host_sid or target_sid not in entry_by_id:
                continue
            target_entry = entry_by_id[target_sid]
            target_dir_str = target_entry.get("directory")
            if not target_dir_str:
                target_dir = Path(tempfile.mkdtemp(prefix=f"shard_{target_sid:02d}_", dir=collection))
                target_entry["directory"] = str(target_dir)
                target_entry["downloaded"] = True
            else:
                target_dir = Path(target_dir_str)
            dest = target_dir / path.name
            if dest.exists():
                path.unlink()
            else:
                shutil.move(str(path), str(dest))
                recovered = target_entry.setdefault("recovered_from_shards", [])
                if host_sid not in recovered:
                    recovered.append(host_sid)
            touched_shards.add(host_sid)
            touched_shards.add(target_sid)

    for sid in touched_shards:
        entry = entry_by_id[sid]
        entry["files"] = _collect_dir_files(Path(entry["directory"]))
        if entry.get("recovered_from_shards"):
            print(f"  [Shard {sid:02d}] Routed {len(entry['files'])} files from backup shard(s) {entry['recovered_from_shards']}")


def pull_and_merge_all(args: argparse.Namespace, accounts: List[Tuple[str, str]]) -> None:
    num_shards = len(accounts)
    if num_shards != 21:
        raise ValueError("Handover collection requires the existing 21-account pool")
    max_idx = min(num_shards, args.max_shards) if args.max_shards > 0 else num_shards
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    collection = Path(tempfile.mkdtemp(prefix="collection_", dir=out_dir))
    manifest = {"schema_version": 2, "expected_shards": num_shards, "collected_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "shards": []}
    print(f"Collecting completed outputs into isolated directories under {collection}...")
    for shard_id in range(max_idx):
        user, token = accounts[shard_id]
        kernel_id = f"{user}/t-scope-vcm-shard-{shard_id:02d}"
        entry = {"shard_id": shard_id, "kernel_id": kernel_id, "expected_variant": ABLATION_ROTATION[shard_id % 5], "downloaded": False}
        try:
            api = get_authenticated_api(user, token)
            status = api.kernels_status(kernel_id)
            entry["status"] = str(getattr(status, "status", status))
            if entry["status"].upper().split(".")[-1] != "COMPLETE":
                print(f"  [Shard {shard_id:02d}] {entry['status']}; skipped")
            else:
                destination = Path(tempfile.mkdtemp(prefix=f"shard_{shard_id:02d}_", dir=collection))
                entry["directory"] = str(destination)
                download_kernel_outputs(api, kernel_id, destination)
                entry["files"] = _collect_dir_files(destination)
                entry["downloaded"] = True
                print(f"  [Shard {shard_id:02d}] Downloaded {len(entry['files'])} files")
        except Exception as error:
            entry["error_type"] = type(error).__name__
            print(f"  [Shard {shard_id:02d}] Collection failed: {type(error).__name__}; credentials not logged")
        manifest["shards"].append(entry)
        (collection / "collection_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        time.sleep(1.5)
    _route_backup_eval_outputs(manifest, collection, num_shards)
    (collection / "collection_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Collection manifest -> {collection / 'collection_manifest.json'}")
    print("No automatic merge performed. Use scripts/merge_shards.py with the checked manifest.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Master Orchestrator for 21 Kaggle Accounts (Push, Status, Pull & Merge)"
    )
    parser.add_argument(
        "--pool_file",
        type=str,
        default=r"C:\Users\Wagur1\Downloads\pool (2).json",
        help="Path to JSON mapping {username: KGAT_token}",
    )
    parser.add_argument(
        "--action",
        choices=["verify_pool", "push", "status", "pull", "pull_and_merge"],
        default="verify_pool",
    )
    parser.add_argument(
        "--run_mode",
        choices=["eval_poc", "train", "both", "eval_soup"],
        default="both",
        help="What each Kaggle kernel executes",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="full_tscope",
        help="Ablation variant if --ablation_sweep is False",
    )
    parser.add_argument(
        "--ablation_sweep",
        action="store_true",
        help="Assigns the 5 ablation variants across the 21 accounts automatically",
    )
    parser.add_argument(
        "--dataset_slug",
        type=str,
        default="ahmedsamir1598/mot17challenge",
        help="Kaggle dataset slug automatically mounted in each kernel",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Max epochs (upper bound when --patience > 0)")
    parser.add_argument("--patience", type=int, default=0, help="Early-stop patience on MOT17-09 val loss; 0 = fixed epochs")
    parser.add_argument("--min_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2, help="Per-GPU batch size (2 for 512x512 with dual detectors)")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps (effective batch size = 4)")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--max_shards", type=int, default=0, help="0 = use all accounts")
    parser.add_argument("--skip_exhausted", action="store_true", default=True, help="Skip accounts with exhausted weekly GPU quota")
    parser.add_argument("--min_gpu_hours", type=float, default=1.0, help="Minimum remaining GPU hours required to dispatch a kernel")
    parser.add_argument("--use_yolov8s", action="store_true", default=True, help="Enable dual-detector Faster R-CNN + YOLOv8s teacher/evaluator")
    parser.add_argument("--git_revision", default="", help="Full reviewed and published commit hash required for eval_soup")
    parser.add_argument("--output_dir", type=str, default="./outputs_kaggle_pool")
    args = parser.parse_args()

    accounts = load_account_pool(args.pool_file)
    if args.action == "verify_pool":
        verify_pool(accounts)
    elif args.action == "push":
        push_all_shards(args, accounts)
    elif args.action == "status":
        check_status_all(args, accounts)
    elif args.action in ("pull", "pull_and_merge"):
        pull_and_merge_all(args, accounts)


if __name__ == "__main__":
    main()
