"""
Download the AuxFormer pretrained checkpoints.

The checkpoints live in the official AuxFormer repo (MediaBrain-SJTU/AuxFormer)
under the ckpt/ folder, which is committed to git and therefore reachable via
raw.githubusercontent.com. Total download size is roughly 90 MB.

By default we only fetch the H3.6M checkpoints (short + long). Pass
--all to also fetch CMU and 3DPW.

Usage:
    python download_auxformer_ckpts.py            # H3.6M only
    python download_auxformer_ckpts.py --all      # everything
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path


BASE_URL = "https://github.com/MediaBrain-SJTU/AuxFormer/raw/main/ckpt"

GROUPS = {
    "h36m": [
        "pretrain_h36m_ckpt.pth.tar",
        "pretrain_h36m_ckpt_long.pth.tar",
    ],
    "cmu": [
        "pretrain_cmu_ckpt.pth.tar",
        "pretrain_cmu_ckpt_long.pth.tar",
    ],
    "3dpw": [
        "pretrain_3dpw_ckpt.pth.tar",
        "pretrain_3dpw_ckpt_long.pth.tar",
    ],
    # Reference model from the official repo, kept for completeness.
    "extra": [
        "r_h36m_ckpt.pth.tar",
    ],
}


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _progress(name: str):
    def _hook(block_num, block_size, total_size):
        if total_size <= 0:
            return
        done = min(block_num * block_size, total_size)
        bar = "#" * int(40 * done / total_size)
        sys.stdout.write(
            f"\r  {name:<35} [{bar:<40}] {_human_size(done)} / {_human_size(total_size)}"
        )
        sys.stdout.flush()
    return _hook


def fetch(name: str, dest_dir: Path, force: bool = False):
    dest = dest_dir / name
    if dest.exists() and not force:
        print(f"  {name} already present, skipping.")
        return
    url = f"{BASE_URL}/{name}"
    print(f"  fetching {url}")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress(name))
        sys.stdout.write("\n")
    except Exception as e:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"failed to download {url}: {e}")


def main():
    p = argparse.ArgumentParser(description="Download AuxFormer pretrained checkpoints.")
    p.add_argument("--all", action="store_true", help="Fetch CMU and 3DPW too.")
    p.add_argument("--force", action="store_true", help="Re-download even if file exists.")
    p.add_argument("--dest", type=str, default=None,
                   help="Destination directory (default: auxformer/ckpt/).")
    args = p.parse_args()

    if args.dest is None:
        dest_dir = Path(__file__).resolve().parent / "auxformer" / "ckpt"
    else:
        dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    groups = ["h36m"]
    if args.all:
        groups += ["cmu", "3dpw", "extra"]

    print(f"Downloading checkpoints into {dest_dir}")
    for g in groups:
        for fname in GROUPS[g]:
            fetch(fname, dest_dir, force=args.force)

    print()
    print("Done. Checkpoints in:", dest_dir)
    for f in sorted(dest_dir.glob("*.pth.tar")):
        print(f"  {f.name}  ({_human_size(f.stat().st_size)})")


if __name__ == "__main__":
    main()
