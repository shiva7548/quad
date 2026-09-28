# AGENTS.md — instructions for AI coding agents

This file is read by AI coding agents (Copilot coding agent, Claude, Codex,
Devin, Arena, etc.) that are granted access to edit this repository.

## Project overview

`quad` is a code repository. The trained model weights (5–8 GB) live on the
**Hugging Face Hub**, NOT in this repo. Never commit model weights or any file
over ~10 MB — GitHub rejects files over 100 MB and `.gitignore` already blocks
weight extensions.

## Getting the model

```bash
pip install -r requirements.txt
python scripts/download_model.py --repo your-hf-username/quad-model
```

## Working rules

1. **Never commit binaries** (weights, datasets, checkpoints). Download them
   with `scripts/download_model.py` instead.
2. Keep changes small and focused; open a pull request for review rather than
   pushing directly to `main` when branch protection is enabled.
3. Run `python -m compileall .` (or the project's tests once they exist) before
   committing Python changes.
4. Do not modify `.gitignore` to unblock binary files.
5. Do not commit secrets, tokens, or `.env` files.

## Layout

```
scripts/download_model.py   # fetch weights from Hugging Face
scripts/upload_model.sh     # one-time upload of weights to Hugging Face
models/                     # local weights (git-ignored)
docs/AI_AGENT_ACCESS.md     # how to grant an AI agent edit access
```
