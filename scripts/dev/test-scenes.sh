#!/usr/bin/env bash
# Tier A test gate (plus optionally Tier B). Run from anywhere; paths are resolved
# relative to this script so the working directory cannot change the result.
#
#   ./scripts/dev/test-scenes.sh                     # Tier A: stdlib only, no network, no assets
#   ./scripts/dev/test-scenes.sh --assets            # also Tier B (skips loudly if roots unset)
#   ./scripts/dev/test-scenes.sh --assets --require-assets   # Tier B skips become failures
#   ./scripts/dev/test-scenes.sh --stdlib-only       # prove the core needs no third-party packages
#
# Isaac (Tier C) and VLM (Tier D) are separate: see test-isaac.sh / test-vlm.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# The orchestrator interpreter carries pytest, transformers/torch (for the real SBERT
# retrieval path), trimesh and pxr. The plain interpreter is used to prove the core is
# stdlib-only. Both must be invoked with Isaac's injected PYTHONPATH removed, otherwise
# imports resolve into /isaac-sim/... and the run is not representative.
ORCHESTRATOR_PYTHON="${PVSCENE_RUNTIME_PYTHON:-python}"
STDLIB_PYTHON="${PVSCENE_STDLIB_PYTHON:-python}"
ORCHESTRATOR_PYTHON="$(command -v "$ORCHESTRATOR_PYTHON" || true)"
STDLIB_PYTHON="$(command -v "$STDLIB_PYTHON" || true)"

INCLUDE_ASSETS=0
REQUIRE_ASSETS=0
STDLIB_ONLY=0
INCLUDE_DELIVERY=0
ENV_FILE=""
NEXT_IS_ENV_FILE=0
# Anything not recognised here is forwarded to pytest verbatim (e.g. -k, -x, a test path).
PASSTHROUGH=()
for arg in "$@"; do
  case "$arg" in
    --assets) INCLUDE_ASSETS=1 ;;
    --require-assets) INCLUDE_ASSETS=1; REQUIRE_ASSETS=1 ;;
    --stdlib-only) STDLIB_ONLY=1 ;;
    --delivery) INCLUDE_DELIVERY=1 ;;
    --env-file) NEXT_IS_ENV_FILE=1 ;;
    *) if [ "${NEXT_IS_ENV_FILE:-0}" = "1" ]; then ENV_FILE="$arg"; NEXT_IS_ENV_FILE=0
       else PASSTHROUGH+=("$arg"); fi ;;
  esac
done

TARGETS=("tests/scenes/unit" "tests/scenes/contract")
EXPR="not isaac and not vlm"
if [ "$INCLUDE_DELIVERY" -eq 0 ]; then
  EXPR="$EXPR and not delivery"
fi
if [ "$INCLUDE_ASSETS" -eq 1 ]; then
  TARGETS+=("tests/scenes/integration")
  # Tier B needs real roots. They live in a gitignored deployment file rather than in the
  # repository, so a missing file means the tier skips with a reason instead of silently
  # testing nothing.
  if [ -z "$ENV_FILE" ] && [ -f "$REPO_ROOT/.env.local" ]; then
    ENV_FILE="$REPO_ROOT/.env.local"
  fi
  if [ -n "$ENV_FILE" ]; then
    if [ ! -f "$ENV_FILE" ]; then
      echo "env file not found: $ENV_FILE" >&2
      exit 2
    fi
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
    echo "loaded deployment config from $ENV_FILE"
  fi
fi

if [ "$STDLIB_ONLY" -eq 1 ]; then
  # The plain interpreter has no pytest, and that is the point: it has no third-party
  # packages either, so it is the only honest way to prove the core is stdlib-only.
  if [ ! -x "$STDLIB_PYTHON" ]; then
    echo "stdlib interpreter not found: $STDLIB_PYTHON" >&2
    exit 2
  fi
  exec env -u PYTHONPATH -u PYTHONHOME "$STDLIB_PYTHON" -m rfw_w.scenes.validation.check_stdlib_only
fi

PY="$ORCHESTRATOR_PYTHON"

if [ ! -x "$PY" ]; then
  echo "interpreter not found or not executable: $PY" >&2
  echo "set PVSCENE_RUNTIME_PYTHON to a python with pytest (for example a Python environment with rfw-w[test] installed)" >&2
  exit 2
fi

# --require-assets promotes Tier B skips to failures. It is exported rather than passed
# as a pytest flag so the same conftest path works for both tiers.
export PVSCENE_REQUIRE_ASSETS="$REQUIRE_ASSETS"

exec env -u PYTHONPATH -u PYTHONHOME \
  "$PY" -m pytest "${TARGETS[@]}" -m "$EXPR" -p no:cacheprovider "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"
