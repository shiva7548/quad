#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Re-attach existing big files (qdrant_storage / *.gguf) to a fresh quad clone.
#
# Use this when the quad folder was deleted but the 5.8 GB model and/or the
# qdrant_storage folder (the ~14k vectors) still exist somewhere on the disk.
# Nothing here is destructive by default.
#
#   bash scripts/attach_storage.sh --find
#       Search the usual places and the Docker side, print what exists.
#
#   bash scripts/attach_storage.sh --use /old/path/qdrant_storage
#       Move that folder into the current quad folder (use --copy to copy).
#
#   bash scripts/attach_storage.sh --use-model /old/path/Qwen3.5-9B.Q4_K_M.gguf
#       Same for the model file.
#
#   bash scripts/attach_storage.sh --docker
#       Print the exact `docker run` that mounts the storage folder where it
#       actually lives (no moving at all). Add --run to execute it.
#
# Key fact: Qdrant reads whatever folder you mount, and the firewall only talks
# HTTP to Qdrant. So the storage does NOT have to be inside the code folder -
# only the .gguf does (or point LAYER2_MODEL_PATH at it).
# ---------------------------------------------------------------------------
set -euo pipefail

ACTION=""
TARGET=""
COPY=0
FORCE=0
RUN=0
SEARCH_ROOTS=("$PWD" "$PWD/.." "$HOME" "/data" "/opt" "/mnt" "/media" "/srv")
CONTAINER="qdrant"

while [ $# -gt 0 ]; do
  case "$1" in
    --find)        ACTION="find"; shift ;;
    --use)         ACTION="use"; TARGET="${2:-}"; shift 2 ;;
    --use-model)   ACTION="use-model"; TARGET="${2:-}"; shift 2 ;;
    --docker)      ACTION="docker"; shift ;;
    --copy)        COPY=1; shift ;;
    --force)       FORCE=1; shift ;;
    --run)         RUN=1; shift ;;
    --container)   CONTAINER="${2:-qdrant}"; shift 2 ;;
    --root)        SEARCH_ROOTS+=("${2:-}"); shift 2 ;;
    -h|--help)     sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

[ -n "$ACTION" ] || { sed -n '2,26p' "$0"; exit 2; }

say()  { printf '%s\n' "$*"; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
human() { du -sh "$1" 2>/dev/null | cut -f1; }

in_quad_folder() { [ -f firewall.py ] && [ -f layer2_judge.py ]; }

# ---------------------------------------------------------------------------
find_storage() {
  local base found
  for base in "${SEARCH_ROOTS[@]}"; do
    [ -d "$base" ] || continue
    while IFS= read -r found; do
      [ -n "$found" ] && printf '%s\n' "$found"
    done < <(find "$base" -maxdepth 4 -type d -name qdrant_storage 2>/dev/null)
  done | sort -u
}

find_models() {
  local base found
  for base in "${SEARCH_ROOTS[@]}"; do
    [ -d "$base" ] || continue
    while IFS= read -r found; do
      [ -n "$found" ] && printf '%s\n' "$found"
    done < <(find "$base" -maxdepth 4 -type f -name "*.gguf" -size +50M 2>/dev/null)
  done | sort -u
}

docker_report() {
  if ! command -v docker >/dev/null 2>&1; then
    say "docker: not installed or not on PATH (skip this part)"
    return
  fi
  if ! docker info >/dev/null 2>&1; then
    say "docker: daemon not reachable (is Docker Desktop running?)"
    return
  fi
  say "docker containers matching 'qdrant':"
  local ids
  ids=$(docker ps -a --filter "name=$CONTAINER" --format '{{.ID}}' 2>/dev/null || true)
  if [ -z "$ids" ]; then
    say "  (none)"
  else
    docker ps -a --filter "name=$CONTAINER" --format '  {{.ID}}  {{.Names}}  {{.Status}}' || true
    say "  mounts:"
    docker inspect "$CONTAINER" \
      --format '{{range .Mounts}}    {{.Source}} -> {{.Destination}}{{"\n"}}{{end}}' 2>/dev/null || true
    say "  -> reuse it with:  docker start $CONTAINER"
  fi
  say "docker named volumes (if you used one instead of a folder):"
  docker volume ls --format '  {{.Name}}' 2>/dev/null | grep -i qdrant || say "  (none)"
}

case "$ACTION" in
  find)
    say "Searching for existing assets (this can take a moment)..."
    say "roots: ${SEARCH_ROOTS[*]}"
    say ""
    say "qdrant_storage folders:"
    mapfile -t storages < <(find_storage)
    if [ "${#storages[@]}" -eq 0 ]; then
      say "  (none found - re-download it with: python scripts/download_model.py)"
    else
      for path in "${storages[@]}"; do
        marker=""
        [ -f "$path/collections/prompt_firewall/config.json" ] && marker="  <-- has the prompt_firewall collection"
        printf '  %-58s %s%s\n' "$path" "$(human "$path")" "$marker"
      done
    fi
    say ""
    say "GGUF model files (>50 MB):"
    mapfile -t models < <(find_models)
    if [ "${#models[@]}" -eq 0 ]; then
      say "  (none found - re-download it with: python scripts/download_model.py)"
    else
      for path in "${models[@]}"; do
        printf '  %-58s %s\n' "$path" "$(human "$path")"
      done
    fi
    say ""
    docker_report
    say ""
    say "Next:"
    say "  attach a folder :  bash scripts/attach_storage.sh --use <path>"
    say "  attach a model  :  bash scripts/attach_storage.sh --use-model <path>"
    say "  or mount as-is  :  bash scripts/attach_storage.sh --docker [--run]"
    ;;

  use|use-model)
    [ -n "$TARGET" ] || die "--use/--use-model needs a path"
    [ -e "$TARGET" ] || die "no such path: $TARGET"
    in_quad_folder || die "run this from inside the quad folder (cd into the fresh clone first)"

    if [ "$ACTION" = "use" ]; then
      NAME="qdrant_storage"
      [ -d "$TARGET" ] || die "$TARGET is not a directory"
      if [ -f "$TARGET/collections/prompt_firewall/config.json" ]; then
        say "Found the prompt_firewall collection in $TARGET - good."
      else
        say "Note: $TARGET has no collections/prompt_firewall/config.json."
        say "      It may be an empty or unrelated Qdrant storage folder."
      fi
    else
      NAME="$(basename "$TARGET")"
      case "$NAME" in
        *.gguf) ;;
        *) die "expected a .gguf file, got: $NAME" ;;
      esac
    fi

    if [ -e "$NAME" ] && [ "$FORCE" -eq 0 ]; then
      die "./$NAME already exists. Move it away, or pass --force to overwrite it."
    fi

    if [ "$COPY" -eq 1 ]; then
      say "Copying $TARGET -> ./$NAME  ($(human "$TARGET")) - needs free disk space"
      cp -a "$TARGET" "./$NAME"
    else
      say "Moving $TARGET -> ./$NAME"
      if ! mv "$TARGET" "./$NAME" 2>/dev/null; then
        die "could not move it (is the Qdrant container using it? try: docker stop $CONTAINER) - or use --copy, or mount it in place with --docker"
      fi
    fi
    say "Done. Now in place: ./$NAME ($(human "./$NAME"))"
    if [ "$ACTION" = "use" ]; then
      say ""
      say "Start/restart Qdrant so it reads this folder:"
      say "  docker rm -f $CONTAINER 2>/dev/null; docker run -d --name $CONTAINER -p 6333:6333 \\"
      say "    -v \"\$PWD/qdrant_storage:/qdrant/storage\" qdrant/qdrant:latest"
    fi
    ;;

  docker)
    in_quad_folder || say "Note: not inside a quad folder - paths below use \$PWD anyway."
    mapfile -t storages < <(find_storage)
    if [ "${#storages[@]}" -eq 0 ]; then
      die "no qdrant_storage found. Re-download it: python scripts/download_model.py"
    fi
    STORAGE="${storages[0]}"
    if [ "${#storages[@]}" -gt 1 ]; then
      say "Several storage folders found - using the first. Others:"
      printf '  %s\n' "${storages[@]:1}"
      say ""
    fi
    ABS=$(cd "$STORAGE" && pwd)
    say "Mounting the storage where it already lives (nothing is moved):"
    say "  $ABS"
    CMD=(docker run -d --name "$CONTAINER" -p 6333:6333 -v "$ABS:/qdrant/storage" qdrant/qdrant:latest)
    printf '  '; printf '%q ' "${CMD[@]}"; printf '\n'
    if [ "$RUN" -eq 1 ]; then
      say ""
      say "Running it (an existing container with the same name is removed first)..."
      docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
      "${CMD[@]}"
      say "Started. Check the vectors:"
      say "  curl -s localhost:6333/collections/prompt_firewall | python3 -m json.tool | grep points_count"
    else
      say ""
      say "Add --run to execute that, or copy/paste it."
    fi
    ;;
esac
