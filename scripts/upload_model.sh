#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-time upload of the trained model (5-8 GB) to the Hugging Face Hub.
#
# Why Hugging Face and not GitHub?
#   - GitHub rejects any file over 100 MB in normal git, and Git LFS free
#     tier caps each file at 2 GB (paid plans 4-5 GB). Your 5-8 GB model
#     does not fit.
#   - The Hugging Face Hub is free for public models and is built for
#     multi-GB model files (it uses LFS / Xet under the hood).
#
# Run this from the machine where the trained weights are.
# ---------------------------------------------------------------------------
set -euo pipefail

# 1) Install the Hugging Face CLI (once)
pip install -U "huggingface_hub[cli]"

# 2) Log in with an ACCESS TOKEN that has WRITE permission
#    Create one at: https://huggingface.co/settings/tokens  (role: write)
huggingface-cli login

# 3) Upload.
#    - Put the weight files in ./models/quad-model first
#      (if it is one huge file, split it: `safetensors.torch.save_model(..., max_shard_size="4GB")`
#       or `model.save_pretrained(dir, max_shard_size="4GB")` - shards under 5 GB upload reliably)
#    - Replace 'your-hf-username' with your Hugging Face username/org.
HF_REPO="your-hf-username/quad-model"

hf upload "${HF_REPO}" ./models/quad-model \
  --repo-type model \
  --commit-message "Upload quad model weights"

echo "Uploaded. Anyone can now fetch it with:"
echo "  python scripts/download_model.py --repo ${HF_REPO}"
