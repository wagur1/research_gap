import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.merge_shards import merge_checkpoints_soup, merge_eval_jsons_and_print_table


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
) -> str:
    """
    Generates the standalone Python script pushed to each Kaggle account's kernel.
    Clones the latest code from https://github.com/wagur1/research_gap.git,
    installs ffmpeg if missing, and writes outputs directly to `/kaggle/working`.
    """
    return f'''import os
import subprocess
import sys
from pathlib import Path

SHARD_ID = {shard_id}
NUM_SHARDS = {num_shards}
VARIANT = "{variant}"
EPOCHS = {epochs}
BATCH_SIZE = {batch_size}
IMG_SIZE = {img_size}
RUN_MODE = "{run_mode}"

repo_dir = Path("/tmp/research_gap")
if repo_dir.exists():
    subprocess.run(["rm", "-rf", str(repo_dir)], check=False)
subprocess.run(
    ["git", "clone", "https://github.com/wagur1/research_gap.git", str(repo_dir)],
    check=True,
)

env = os.environ.copy()
env["PYTHONPATH"] = str(repo_dir) + os.pathsep + env.get("PYTHONPATH", "")

if RUN_MODE in ("eval_poc", "both"):
    cmd_eval = [
        sys.executable,
        str(repo_dir / "scripts" / "kaggle_shard_runner.py"),
        "--mode", "eval_poc",
        "--shard_id", str(SHARD_ID),
        "--num_shards", str(NUM_SHARDS),
        "--img_size", str(IMG_SIZE),
        "--use_pretrained_fpn",
        "--output_dir", "/kaggle/working",
    ]
    subprocess.run(cmd_eval, cwd=str(repo_dir), env=env, check=True)

if RUN_MODE in ("train", "both"):
    cmd_train = [
        sys.executable,
        str(repo_dir / "scripts" / "kaggle_shard_runner.py"),
        "--mode", "train",
        "--shard_id", str(SHARD_ID),
        "--num_shards", str(NUM_SHARDS),
        "--variant", VARIANT,
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--img_size", str(IMG_SIZE),
        "--enable_x265_ste",
        "--use_pretrained_fpn",
        "--output_dir", "/kaggle/working",
    ]
    subprocess.run(cmd_train, cwd=str(repo_dir), env=env, check=True)

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
    num_shards = len(accounts)
    max_Idx = min(num_shards, args.max_shards) if args.max_shards > 0 else num_shards
    print(
        f"Pushing {max_Idx}/{num_shards} Kaggle GPU kernels "
        f"(mode={args.run_mode}, dataset={args.dataset_slug})..."
    )

    for shard_id in range(max_Idx):
        user, token = accounts[shard_id]
        variant = (
            ABLATION_ROTATION[shard_id % len(ABLATION_ROTATION)]
            if args.ablation_sweep
            else args.variant
        )
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
                    f"  [Shard {shard_id:02d}/{num_shards:02d}] Pushed -> {kernel_id:<38} "
                    f"| variant={variant:<18} | {ref_url} {err_msg}"
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


def pull_and_merge_all(args: argparse.Namespace, accounts: List[Tuple[str, str]]) -> None:
    num_shards = len(accounts)
    max_idx = min(num_shards, args.max_shards) if args.max_shards > 0 else num_shards
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pulling outputs from {max_idx} Kaggle accounts into {out_dir}...")
    pulled = 0
    for shard_id in range(max_idx):
        user, token = accounts[shard_id]
        kernel_id = f"{user}/t-scope-vcm-shard-{shard_id:02d}"
        try:
            api = get_authenticated_api(user, token)
            api.kernels_output(kernel_id, path=str(out_dir), quiet=True)
            print(f"  [Shard {shard_id:02d}] Downloaded outputs from {kernel_id}")
            pulled += 1
        except Exception as e:
            print(f"  [Shard {shard_id:02d}] Skip {kernel_id} ({e})")

    print(f"\nPulled {pulled}/{max_idx} shards. Running aggregation & Model Soup merge...")
    merge_eval_jsons_and_print_table(str(out_dir))
    merge_checkpoints_soup(str(out_dir), str(out_dir / "tscope_merged_soup.pt"))


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
        choices=["verify_pool", "push", "status", "pull_and_merge"],
        default="verify_pool",
    )
    parser.add_argument(
        "--run_mode",
        choices=["eval_poc", "train", "both"],
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--max_shards", type=int, default=0, help="0 = use all accounts")
    parser.add_argument("--output_dir", type=str, default="./outputs_kaggle_pool")
    args = parser.parse_args()

    accounts = load_account_pool(args.pool_file)
    if args.action == "verify_pool":
        verify_pool(accounts)
    elif args.action == "push":
        push_all_shards(args, accounts)
    elif args.action == "status":
        check_status_all(args, accounts)
    elif args.action == "pull_and_merge":
        pull_and_merge_all(args, accounts)


if __name__ == "__main__":
    main()
