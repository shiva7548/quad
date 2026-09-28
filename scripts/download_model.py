#!/usr/bin/env python3
"""Download project assets (model + vector DB) from the Hugging Face Hub.

The big files live on Hugging Face because GitHub rejects files over 100 MB:

  * Qwen3.5-9B.Q4_K_M.gguf   ~5.8 GB  -> Layer 2 LLM judge model
  * qdrant_storage/                   -> prebuilt Layer 1 vector index (~14k vectors)
  * datasets/                         -> malicious-prompt CSVs used to build the index

The 'venv/' folder in the HF repo is NOT downloaded (it is a re-creatable
Python virtual environment - use pip install -r requirements.txt instead).

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --repo linto777/my-qdrant-project --dir .
"""

import argparse
import os
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("Missing dependency. Run:  pip install -r requirements.txt")

DEFAULT_REPO = os.environ.get("MODEL_REPO", "linto777/my-qdrant-project")
DEFAULT_DIR = os.environ.get("MODEL_DIR", ".")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download quad project assets from the Hugging Face Hub.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"HF repo id (default: {DEFAULT_REPO} or $MODEL_REPO)")
    parser.add_argument("--dir", default=DEFAULT_DIR, help=f"Local folder to download into (default: {DEFAULT_DIR})")
    parser.add_argument("--revision", default="main", help="Branch, tag, or commit to download (default: main)")
    parser.add_argument("--model-only", action="store_true", help="Download only the GGUF judge model (skip qdrant_storage/datasets)")
    args = parser.parse_args()

    if args.model_only:
        allow_patterns = ["*.gguf"]
        ignore_patterns = ["venv/*", "**/venv/*"]
    else:
        allow_patterns = None
        ignore_patterns = ["venv/*", "**/venv/*", ".gitattributes"]

    print(f"Downloading '{args.repo}' (revision {args.revision}) -> {args.dir}")
    print("  (venv/ is skipped - install packages with: pip install -r requirements.txt)")
    path = snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        local_dir=args.dir,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
    print(f"Done. Assets saved to: {path}")


if __name__ == "__main__":
    main()
