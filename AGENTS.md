# AGENTS.md — instructions for AI coding agents

This file is read by AI coding agents (Copilot coding agent, Claude, Codex,
Devin, Arena, etc.) that are granted access to edit this repository.

## Project overview

`quad` is a two-layer **RAG prompt firewall**:

- **Layer 1** (`firewall.py`): semantic vector search — BAAI/bge-large-en-v1.5
  embeddings + Qdrant (~14k malicious prompts), sandwich-attack chunking,
  Web UI + `POST /check` + `--cli`.
- **Layer 2** (`layer2_judge.py`): local LLM judge using
  `Qwen3.5-9B.Q4_K_M.gguf` via `llama-cpp-python` for borderline/novel prompts.

Big files live on the Hugging Face repo `linto777/my-qdrant-project`
(GGUF model, `qdrant_storage/`, `datasets/`) — NEVER in this repo. GitHub
rejects files over 100 MB and `.gitignore` blocks them.

## Getting the model & vector DB

```bash
pip install -r requirements.txt
python scripts/download_model.py        # GGUF + qdrant_storage (skips venv/)
```

## Working rules

1. **Never commit binaries** (GGUF/weights, `qdrant_storage/`, `datasets/`,
   `venv/`). Download them with `scripts/download_model.py` instead.
2. Keep changes small and focused; open a pull request for review rather than
   pushing directly to `main` when branch protection is enabled.
3. Run `python -m compileall firewall.py layer2_judge.py scripts/` (or the
   project's tests once they exist) before committing Python changes.
4. Do not modify `.gitignore` to unblock binary files.
5. Do not commit secrets, tokens, or `.env` files.
6. Layer 2 must stay optional: if the GGUF or `llama-cpp-python` is missing,
   `firewall.py` falls back to Layer 1-only behavior — keep it that way.

## Layout

```
firewall.py                 # Layer 1 + Layer 2 wiring, HTTP server, CLI
layer2_judge.py             # LLM judge module (GGUF)
scratch*.py                 # threshold experiments
scripts/download_model.py   # fetch big files from Hugging Face
scripts/upload_model.sh     # one-time upload of big files to Hugging Face
models/                     # local weights if downloaded there (git-ignored)
docs/AI_AGENT_ACCESS.md     # how to grant an AI agent edit access
```
