# quad — RAG Prompt Firewall (Layer 1 + Layer 2)

A two-layer AI **prompt firewall**: it detects malicious / adversarial prompts
(prompt injection, jailbreaks, sandwich attacks) before they reach your LLM.

| Layer | Name | Status | Tech |
|---|---|---|---|
| **Layer 1** | RAG & Semantic Vector Search | ✅ Complete | `SentenceTransformer("BAAI/bge-large-en-v1.5")` (1024-dim) + Qdrant with ~14,000+ malicious prompt vectors |
| **Layer 2** | LLM Guardrail / Judge | ✅ Integrated | Local `Qwen3.5-9B.Q4_K_M.gguf` (~5.8 GB) via `llama-cpp-python` |

**Layer 1** embeds the live prompt (split into clause chunks to defeat
*Sandwich attacks*) and compares it against known-malicious vectors.
**Layer 2** is a local LLM judge that classifies borderline or novel prompts
that vector search alone is unsure about.

* Web UI: `http://localhost:8000/`
* REST API: `POST /check` with `{"prompt": "..."}`
* Interactive CLI: `python firewall.py --cli`

## Why code on GitHub, model on Hugging Face?

GitHub rejects files over 100 MB — the 5.8 GB `.gguf` cannot live here.
So this repo holds the **code**, and the model + vector DB live in the
Hugging Face repo [`linto777/my-qdrant-project`](https://huggingface.co/linto777/my-qdrant-project)
and are fetched on demand by `scripts/download_model.py`.

## Quick start (local)

```bash
# 1. Get the code
git clone https://github.com/shiva7548/quad.git
cd quad

# 2. Install dependencies (do NOT copy a venv around)
pip install -r requirements.txt

# 3. Download the judge model + prebuilt vector DB (~6 GB, one time)
python scripts/download_model.py
#   -> Qwen3.5-9B.Q4_K_M.gguf  +  qdrant_storage/  +  datasets/

# 4. Start Qdrant (easiest with Docker)
docker run -d --name qdrant -p 6333:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant:latest
#   (or reuse the downloaded qdrant_storage so the 14k vectors are already there)

# 5. Run the firewall
export QDRANT_URL=http://localhost:6333
python firewall.py            # server + Web UI on http://localhost:8000/
python firewall.py --cli      # interactive CLI instead
```

If `qdrant_storage/` is empty, the first run automatically rebuilds the index
by downloading the malicious-prompt datasets and embedding ~14k prompts
(needs `HF_TOKEN` in `.env` for gated datasets).

## Run with Docker (local or server)

```bash
cp .env.example .env       # add your HF_TOKEN if you need dataset downloads
docker compose up --build
# Web UI on http://localhost:8000/
```

## Layer 2 configuration

| Env var | Default | Meaning |
|---|---|---|
| `LAYER2_MODE` | `gray` | `off` = Layer 1 only · `gray` = judge borderline scores only (fast) · `all` = judge every prompt Layer 1 doesn't block (strictest) |
| `LAYER2_MIN_SCORE` | `0.55` | Lower bound of the "gray zone" in `gray` mode |
| `LAYER2_MODEL_PATH` | `./Qwen3.5-9B.Q4_K_M.gguf` | Path to the GGUF judge model |
| `LAYER2_GPU_LAYERS` | `0` | `0` = CPU only; set `-1` (or e.g. `33`) with an NVIDIA GPU |
| `LAYER2_CONFIRM_BLOCKS` | `false` | `true` = Layer 2 re-verifies Layer 1 blocks (can rescue false positives) |

Example API response when Layer 2 judges a borderline prompt:

```json
{
  "status": "BLOCK",
  "reason": "Malicious prompt confirmed (Layer 2 LLM Judge)",
  "score": 0.61,
  "layer2": {"verdict": "BLOCK", "reason": "attempt to reveal system prompt", "confidence": 0.92}
}
```

If the GGUF file is missing, the firewall runs in **Layer 1 only** mode and
prints a warning — nothing breaks.

## Run online (deploy)

Any of these work — the app is a plain Python HTTP server on port 8000:

1. **Any cloud VM / VPS** (simplest): install Docker on the VM, clone this
   repo, `python scripts/download_model.py`, then `docker compose up -d`.
   Open port 8000 (put nginx/caddy + HTTPS in front for production).
   Needs ~8–10 GB RAM for CPU inference of the 9B Q4 model (less with GPU).
2. **Hugging Face Spaces (Docker SDK)**: create a Space, push this code,
   add a persistent volume for `qdrant_storage/` and the `.gguf`.
3. **Cloud container services** (Fly.io, Railway, Render, AWS ECS…):
   build the provided `Dockerfile`, mount a volume for the model + storage.

## Repository layout

```
firewall.py                 # Layer 1 vector search + HTTP/CLI + Layer 2 wiring
layer2_judge.py             # Layer 2 LLM judge (GGUF via llama-cpp-python)
scratch*.py                 # embedding-threshold experiments
scripts/download_model.py   # fetch GGUF + qdrant_storage from Hugging Face
scripts/upload_model.sh     # one-time upload of big files to Hugging Face
Dockerfile / docker-compose.yml
docs/AI_AGENT_ACCESS.md     # how to grant an AI coding agent edit access
AGENTS.md                   # rules agents must follow in this repo
```

## Rules for everyone (human or agent)

1. **Never commit model weights, `qdrant_storage/`, `datasets/`, or `venv/`.**
   `.gitignore` blocks them; do not bypass it.
2. Develop on a branch, open a pull request, review, then merge.
3. Never commit tokens/secrets — they go in `.env` (git-ignored).
