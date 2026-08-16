"""Download the full APEBench dataset release with resumable transfers."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/working/datasets_ape2d_full"),
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    snapshot_download(
        repo_id="thuerey-group/pde-transformer-ape2d-full",
        repo_type="dataset",
        local_dir=output_dir,
        allow_patterns=["*.hdf5", "*.json", "README.md"],
    )
    files = sorted(output_dir.glob("*.hdf5"))
    size_gib = sum(path.stat().st_size for path in files) / 1024**3
    elapsed_minutes = (time.time() - started_at) / 60
    print(
        f"download_complete files={len(files)} size={size_gib:.2f}GiB "
        f"elapsed={elapsed_minutes:.1f}min"
    )


if __name__ == "__main__":
    main()
