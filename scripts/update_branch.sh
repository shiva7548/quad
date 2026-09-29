#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Update an EXISTING quad folder to the newest code - without re-downloading
# the 5.8 GB model, the vector DB or the datasets.
#
#   cd ~/quad          # the folder you already have (the clone that failed to
#                      # be cloned again: "destination path 'quad' already exists")
#   bash scripts/update_branch.sh
#
# Or, if you do not have the script yet (it is new), use the 3-line manual
# version from the README:
#   git fetch origin arena/01a0eb9a-quad
#   git checkout -f -B arena/01a0eb9a-quad FETCH_HEAD
#
# What it does
#   * works whether the folder is a git clone (case A) or just unpacked files
#     with no .git (case B - someone downloaded a ZIP)
#   * backs up any local edits into ../quad-backup-<timestamp>/ first
#   * switches the code to the requested branch
#   * NEVER deletes or rewrites: *.gguf, models/, qdrant_storage/, datasets/,
#     venv/, .env  (they are git-ignored, so git leaves them alone)
#
# Options
#   --branch NAME   branch to switch to (default: arena/01a0eb9a-quad)
#   --repo URL      git remote to use (default: the existing origin, else GitHub)
#   --yes           do not ask for confirmation
# ---------------------------------------------------------------------------
set -euo pipefail

BRANCH="arena/01a0eb9a-quad"
REPO="${QUAD_REPO:-https://github.com/shiva7548/quad.git}"
ASSUME_YES=0

while [ $# -gt 0 ]; do
  case "$1" in
    --branch) BRANCH="$2"; shift 2 ;;
    --repo)   REPO="$2"; shift 2 ;;
    --yes|-y) ASSUME_YES=1; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

say()  { printf '%s\n' "$*"; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# --- 1. are we in a quad folder? -------------------------------------------
[ -f firewall.py ] || die "firewall.py not found. Run this from inside your quad folder (cd quad)."
[ -f layer2_judge.py ] || die "layer2_judge.py not found - is this really the quad folder?"

say "quad updater"
say "  folder : $(pwd)"
say "  target : $BRANCH"
say ""

# --- 2. what must survive (informational) ----------------------------------
KEEP=""
for item in Qwen3.5-9B.Q4_K_M.gguf models qdrant_storage datasets .env venv .venv; do
  if [ -e "$item" ]; then
    size=""
    if [ -f "$item" ]; then
      size=" ($(du -h "$item" 2>/dev/null | cut -f1))"
    elif [ -d "$item" ]; then
      size=" ($(du -sh "$item" 2>/dev/null | cut -f1))"
    fi
    KEEP="$KEEP    - $item$size
"
  fi
done
if [ -n "$KEEP" ]; then
  say "Preserved (git ignores these, they are never touched):"
  printf '%s' "$KEEP"
else
  say "No model / vector DB / datasets found in this folder yet -"
  say "after updating, download them with: python scripts/download_model.py"
fi
say ""

GGUF_FOUND=$(ls -1 ./*.gguf models/*.gguf 2>/dev/null | head -1 || true)
if [ -z "$GGUF_FOUND" ]; then
  say "Note: no .gguf file here yet. Download it once (5.8 GB) with:"
  say "      python scripts/download_model.py"
  say "      (it lands in this folder, which is where firewall.py looks for it)"
  say ""
fi

if [ "$ASSUME_YES" -eq 0 ]; then
  printf 'Proceed? [y/N] '
  read -r answer
  case "$answer" in
    y|Y|yes|YES) ;;
    *) say "Cancelled - nothing changed."; exit 0 ;;
  esac
fi

# --- 3. back up local edits ------------------------------------------------
BACKUP="../quad-backup-$(date +%Y%m%d-%H%M%S)"
backup_file() {  # $1 = relative path
  [ -e "$1" ] || return 0
  mkdir -p "$BACKUP/$(dirname "$1")"
  cp -a "$1" "$BACKUP/$1"
}

# --- 4. case A: it is already a git repository -----------------------------
if [ -d .git ]; then
  say "[1/4] Existing git repository found."
  if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    MODIFIED=$(git status --porcelain | awk '{print $2}')
    say "      Local changes detected - backing them up to $BACKUP"
    while IFS= read -r file; do
      [ -n "$file" ] && backup_file "$file"
    done <<< "$MODIFIED"
  fi

  say "[2/4] Fetching $BRANCH ..."
  git fetch origin "$BRANCH"

  say "[3/4] Switching code to $BRANCH (local edits discarded, backups kept) ..."
  git checkout -f -B "$BRANCH" FETCH_HEAD

# --- 5. case B: plain files, no .git --------------------------------------
else
  say "[1/4] No .git here (unpacked ZIP?) - initialising a repository."
  git init -q
  git remote remove origin 2>/dev/null || true
  git remote add origin "$REPO"

  say "[2/4] Fetching $BRANCH ..."
  git fetch -q origin "$BRANCH"

  say "[3/4] Moving aside files that the checkout would overwrite (backup: $BACKUP) ..."
  # Only files that the branch also contains need to move; the model, DB and
  # datasets are not in git, so they stay exactly where they are.
  MOVED=0
  while IFS= read -r file; do
    [ -n "$file" ] || continue
    if [ -e "$file" ]; then
      backup_file "$file"
      rm -f "$file"
      MOVED=$((MOVED + 1))
    fi
  done < <(git ls-tree -r --name-only FETCH_HEAD)

  git checkout -q -B "$BRANCH" FETCH_HEAD
  say "      replaced $MOVED code file(s); big files untouched."
fi

# --- 6. verify ------------------------------------------------------------
say "[4/4] Verifying ..."
for file in context_rules.py scripts/doctor.py scripts/ui_preview.py scripts/gguf_smoke_test.py; do
  if [ -f "$file" ]; then
    say "      ok   $file"
  else
    say "      MISSING $file  <- the checkout did not complete"
  fi
done

say ""
say "Now in: $(pwd)  (branch: $(git rev-parse --abbrev-ref HEAD), commit: $(git rev-parse --short HEAD))"
say ""
say "Next:"
say "  1) python scripts/doctor.py          # prints where it found the model + Layer 2 status"
say "  2) export QDRANT_URL=http://localhost:6333"
say "     docker start qdrant 2>/dev/null || docker run -d --name qdrant -p 6333:6333 \\"
say "        -v \"\$PWD/qdrant_storage:/qdrant/storage\" qdrant/qdrant:latest"
say "  3) python firewall.py                # Web UI: http://localhost:8000/"
say ""
say "If the backup is not needed:  rm -rf $BACKUP"
