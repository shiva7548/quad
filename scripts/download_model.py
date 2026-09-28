#!/usr/bin/env python3
"""Download the quad model weights from the Hugging Face Hub.

The weights (5-8 GB) are too large for GitHub, so they are hosted on
Hugging Face and fetched on demand by this script.

Usage:
    python scripts/download_model.py                        # uses MODEL_REPO env var
    python scripts/download_model.py --repo your-name/quad-model
    MODEL_REPO=your-name/quad-model python scripts/download_model.py

For a private model repo, log in first (needs a READ token):
    huggingface-cli login
"""

import argparse
import os
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("Missing dependency. Run:  pip install -r requirements.txt")

DEFAULT_REPO = os.environ.get("MODEL_REPO", "your-hf-username/quad-model")
DEFAULT_DIR = os.environ.get("MODEL_DIR", os.path.join("models", "quad-model"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download quad model weights from the Hugging Face Hub.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"HF repo id (default: {DEFAULT_REPO} or $MODEL_REPO)")
    parser.add_argument("--dir", default=DEFAULT_DIR, help=f"Local folder to download into (default: {DEFAULT_DIR})")
    parser.add_argument("--revision", default="main", help="Branch, tag, or commit to download (default: main)")
    args = parser.parse_args()

    print(f"Downloading '{args.repo}' (revision {args.revision}) -> {args.dir}")
    path = snapshot_download(repo_id=args.repo, revision=args.revision, local_dir=args.dir)
    print(f"Done. Model saved to: {path}")


if __name__ == "__main__":
    main()
