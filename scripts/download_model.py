#!/usr/bin/env python3
"""Download project assets (model + vector DB) from the Hugging Face Hub.

The big files live on Hugging Face because GitHub rejects files over 100 MB:

  * Qwen3.5-9B.Q4_K_M.gguf   ~5.8 GB  -> Layer 2 LLM judge model
  * qdrant_storage/                   -> prebuilt Layer 1 vector index (~14k vectors)
  * datasets/                         -> malicious-prompt CSVs used to build the index

The 'venv/' folder in the HF repo is NOT downloaded (it is a re-creatable
Python virtual environment - use pip install -r requirements.txt instead).

Usage:
    python scripts/download_model.py                 # everything (~6.4 GB)
    python scripts/download_model.py --no-datasets   # model + vector DB only (~5.9 GB)
    python scripts/download_model.py --model-only    # just the GGUF judge (~5.8 GB)
    python scripts/download_model.py --repo linto777/my-qdrant-project --dir .

The prebuilt qdrant_storage/ is what makes the first run instant: it already
holds the ~14k malicious-prompt vectors, so nothing is re-embedded and no
HF_TOKEN is needed. The datasets/ folder (~511 MB, 17 CSVs) is only used when
the index is rebuilt from scratch with FORCE_REBUILD=true.
"""

import argparse
import os
import sys

DEFAULT_REPO = os.environ.get("MODEL_REPO", "linto777/my-qdrant-project")
DEFAULT_DIR = os.environ.get("MODEL_DIR", ".")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download quad project assets from the Hugging Face Hub.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"HF repo id (default: {DEFAULT_REPO} or $MODEL_REPO)")
    parser.add_argument("--dir", default=DEFAULT_DIR, help=f"Local folder to download into (default: {DEFAULT_DIR})")
    parser.add_argument("--revision", default="main", help="Branch, tag, or commit to download (default: main)")
    parser.add_argument("--model-only", action="store_true", help="Download only the GGUF judge model (skip qdrant_storage/datasets)")
    parser.add_argument("--no-datasets", action="store_true",
                        help="Skip datasets/ (~511 MB): the prebuilt qdrant_storage/ already "
                             "contains the vectors, so datasets are only needed for FORCE_REBUILD")
    args = parser.parse_args()

    if args.model_only and args.no_datasets:
        parser.error("--model-only and --no-datasets are mutually exclusive")

    # Imported after argument validation so --help and mistakes work before pip install.
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("Missing dependency 'huggingface_hub'.\n"
                 "Run:  pip install -r requirements.txt")

    if args.model_only:
        allow_patterns = ["*.gguf"]
        ignore_patterns = ["venv/*", "**/venv/*"]
    elif args.no_datasets:
        allow_patterns = None
        ignore_patterns = ["venv/*", "**/venv/*", ".gitattributes", "datasets/*", "**/datasets/*"]
    else:
        allow_patterns = None
        ignore_patterns = ["venv/*", "**/venv/*", ".gitattributes"]

    print(f"Downloading '{args.repo}' (revision {args.revision}) -> {args.dir}")
    print("  (venv/ is skipped - install packages with: pip install -r requirements.txt)")
    if args.model_only:
        print("  (--model-only: only the .gguf - Layer 1's qdrant_storage will NOT be here)")
    elif args.no_datasets:
        print("  (--no-datasets: .gguf + qdrant_storage, skipping the ~511 MB of CSVs)")
    print("  expected: Qwen3.5-9B.Q4_K_M.gguf (5.78 GB) + qdrant_storage/ with the "
          "prebuilt ~14k-vector index")
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
