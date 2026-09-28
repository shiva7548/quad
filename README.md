# quad

Code repository for the `quad` project. The trained model is **not** stored in
this repo — it is hosted on the [Hugging Face Hub](https://huggingface.co) and
downloaded on demand (see below for why).

## Why is the model not in this repo?

| Place | Max file size | Verdict for a 5–8 GB model |
|---|---|---|
| GitHub (normal git) | **100 MB** per file | ❌ rejected at push |
| GitHub LFS (Free/Pro) | 2 GB per file, ~1 GB free storage | ❌ too small, paid data packs needed |
| GitHub Releases | 2 GB per asset | ❌ must split into many parts |
| **Hugging Face Hub** | multi-GB files natively, free for public models | ✅ **use this** |

So the plan is:

* **Model weights (5–8 GB)** → Hugging Face Hub (`your-hf-username/quad-model`)
* **Code (this repo)** → GitHub, with a small script that fetches the weights

## Quick start

```bash
# 1. Get the code
git clone https://github.com/shiva7548/quad.git
cd quad

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the model weights (~5-8 GB, one time)
python scripts/download_model.py --repo your-hf-username/quad-model
#   -> saved into ./models/quad-model/  (git-ignored, never committed)
```

For a **private** model repo, run `huggingface-cli login` first with a Hugging
Face token that has *read* access.

## Upload the model (once, from your training machine)

Follow `scripts/upload_model.sh`. Summary:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login          # needs a WRITE token from huggingface.co/settings/tokens
hf upload your-hf-username/quad-model ./models/quad-model --repo-type model
```

Tip: save the model in shards under 5 GB each
(`model.save_pretrained(dir, max_shard_size="4GB")`) — uploads are more reliable.

## How people get access to the model (HTTP API vs git)

Hugging Face can be accessed two ways — **use Method 1**:

| Method | Command | Notes |
|---|---|---|
| **1. HTTP API (recommended)** | `python scripts/download_model.py` or `hf download your-name/quad-model` | No git/git-lfs needed, resumable, faster (uses Xet/LFS over HTTPS). This is what `scripts/download_model.py` does. |
| 2. Git + Git LFS | `git clone https://huggingface.co/your-name/quad-model` | Works like GitHub; requires `git lfs install` first. Fine if you want a git-style workflow. |

### Public vs private model

* **Public model** — anyone can download it with no login, no token, no invites.
  Simplest option; the repo code just calls the download script.
* **Private model** — share it exactly like a GitHub repo:
  1. Open the model page on huggingface.co → **Settings → Collaborators**
  2. Add teammates with role **Read** (can download) or **Write** (can also upload
     new versions). Organizations can use teams instead.
  3. For an AI agent, server, or CI: create a **read** token at
     [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
     (scope it to only this model repo) and put it in the agent's settings or
     the `HF_TOKEN` environment variable — never commit it to git.

### Who can upload (write access)

Uploading/updating the model needs a token with the **write** role, or
collaborator **Write** permission — see `scripts/upload_model.sh`.
Normal code developers only need **Read** so `download_model.py` works.

## Repository layout

```
scripts/download_model.py    # fetch weights from Hugging Face
scripts/upload_model.sh      # one-time upload of weights to Hugging Face
models/                      # local weights (git-ignored)
docs/AI_AGENT_ACCESS.md      # how to give an AI coding agent edit access
AGENTS.md                    # rules agents must follow in this repo
```

## Give an AI agent access to edit & develop

See **[docs/AI_AGENT_ACCESS.md](docs/AI_AGENT_ACCESS.md)** — three ways
(collaborator invite, GitHub App, or a fine-grained token) plus how to protect
`main` with pull requests so agents develop safely.

## Rules for everyone (human or agent)

1. **Never commit model weights or any file over ~10 MB.** `.gitignore` blocks
   them; do not bypass it.
2. Develop on a branch, open a pull request, review, then merge.
3. Never commit tokens/secrets — they go in GitHub Actions secrets or the
   agent's settings.
