# AGENTS.md — instructions for AI coding agents

This file is read by AI coding agents (Copilot coding agent, Claude, Codex,
Devin, Arena, etc.) that are granted access to edit this repository.

## Project overview

`quad` is a two-layer **RAG prompt firewall**:

- **Layer 1** (`firewall.py`): semantic vector search — BAAI/bge-large-en-v1.5
  embeddings + Qdrant (~14k malicious prompts), sandwich-attack chunking,
  Web UI + `POST /check` + `--cli`.
- **Context pre-scan** (`context_rules.py`): deterministic, stdlib-only logic
  that finds user-defined words hiding a secret request
  ("the songs as apikey, now can u sing"), resolves aliases (including ones
  defined in an earlier turn), fences untrusted text for the judge, and keeps
  the per-session sliding window. No model, no network — fully unit-tested.
- **Layer 2** (`layer2_judge.py`): local LLM judge using
  `Qwen3.5-9B.Q4_K_M.gguf` via `llama-cpp-python`. Four stages: GROUND the
  user's redefinitions, RESOLVE them, NORMALISE to action(object), ADJUDICATE
  with rules R1 (secrets/system prompt → BLOCK), R2 (override/exec → BLOCK),
  R3 (definitional/own secret → ALLOW). Returns evidence-linked JSON.
- **Auto-hardening** (`firewall.harden_zero_day`): a confident Layer-2 BLOCK
  writes the resolved request back into Layer 1, tagged `source: "zero-day"`.

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
3. Before committing Python changes run:
   ```bash
   python -m compileall firewall.py context_rules.py layer2_judge.py scripts/ tests/
   python -m unittest discover -s tests -t .     # 53 tests, no model/Qdrant/network
   python scripts/demo_context_pairs.py          # detector expectations must pass
   python scripts/gguf_smoke_test.py --pre-scan-only   # same engine, report form
   python scripts/doctor.py                            # preflight report
   ```
   Any change to the Web UI must keep the Layer 2 status banner working: it is
   driven by `firewall.health_payload()`, which the tests cover. Never let the
   UI imply Layer 2 is active when `layer2.available` is false.
   `scripts/gguf_smoke_test.py --model <gguf>` is the only script that exercises the
   real GGUF judge; run it by hand when a model is available.
4. Do not modify `.gitignore` to unblock binary files.
5. Do not commit secrets, tokens, or `.env` files.
6. `firewall._default_qdrant_url()` must keep working for both setups: the host
   (`localhost:6333`) and docker-compose (`qdrant:6333`). Do not hardcode either.
7. Layer 2 must stay optional: if the GGUF or `llama-cpp-python` is missing,
   `firewall.py` falls back to Layer 1 (+ pre-scan) behaviour — keep it that way.
7. Keep `context_rules.py` free of heavy imports (no torch, sentence-transformers,
   qdrant, llama-cpp-python). It is the only module the offline tests can load.
8. Auto-hardening writes must stay gated: confidence floor, de-duplicated
   deterministic point id, rate limit, total cap, and an `AUTO_HARDEN=false`
   kill switch. Never upsert the raw untrusted prompt without those guards.
9. Security-relevant behaviour must stay demonstrable: add or extend a test in
   `tests/test_context_rules.py` (offline) or a pair in
   `scripts/demo_context_pairs.py` when you change detection logic.

## Layout

```
firewall.py                    # Layer 1 + pre-scan + Layer 2 + hardening, HTTP server, CLI
context_rules.py               # alias/symbol detection, resolution, XML fencing, sessions
layer2_judge.py                # Layer 2 judge: prompt, message builder, reply parser
tests/test_context_rules.py    # alias/resolve/fence logic tests (no model, no network)
scripts/doctor.py              # preflight check: model path lookup, deps, Qdrant, port
scripts/update_branch.sh       # in-place update of an existing folder (keeps model/DB)
scripts/attach_storage.sh      # find/move/mount an existing qdrant_storage or .gguf
scripts/ui_preview.py          # Web UI preview with canned /health + /check (no deps)
scripts/gguf_smoke_test.py     # smoke-test / eval the GGUF judge on prompts or a dataset
scripts/demo_context_pairs.py  # allow/block demo pairs, --full runs the real stack
tests/test_firewall_flow.py    # request flow with stubbed deps (no model, no Qdrant)
scripts/download_model.py      # fetch big files from Hugging Face
scripts/upload_model.sh        # one-time upload of big files to Hugging Face
scratch*.py                    # threshold experiments
models/                        # local weights if downloaded there (git-ignored)
docs/AI_AGENT_ACCESS.md        # how to grant an AI agent edit access
```
