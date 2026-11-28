#!/usr/bin/env bash
# Tier C: Isaac checks, driven as a script rather than collected by pytest.
#
# pytest availability inside /isaac-sim/python.sh is unverified, and Kit's shutdown path
# can overwrite the process exit code, so these checks emit a JSON report and are asserted
# on by a separate process. A run that dies without writing its report counts as a failure.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ISAAC_PYTHON="${PVSCENE_ISAAC_PYTHON:-/isaac-sim/python.sh}"
OUT="${1:-outputs/_probe/isaac_checks}"
TIMEOUT="${PVSCENE_ISAAC_TIMEOUT:-900}"
ACTIVE_GPU="${PVSCENE_ACTIVE_GPU:-0}"

if [ ! -x "$ISAAC_PYTHON" ]; then
  echo "Isaac python not found: $ISAAC_PYTHON (set PVSCENE_ISAAC_PYTHON)" >&2
  exit 2
fi

# A fresh output directory each run: stale images or a stale report must never be able to
# make a failed run look like a pass.
if [ -e "$OUT" ]; then
  echo "output directory already exists, choose a new one: $OUT" >&2
  exit 2
fi

set +e
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
  timeout "$TIMEOUT" "$ISAAC_PYTHON" tests/scenes/isaac/run_isaac_checks.py \
  --output-dir "$OUT" --active-gpu "$ACTIVE_GPU"
rc=$?
set -e

echo "isaac checks exit code: $rc"

if [ ! -f "$OUT/report.json" ]; then
  echo "FAIL: no report at $OUT/report.json -- a killed or crashed run is a failure" >&2
  exit 1
fi

exec env -u PYTHONPATH -u PYTHONHOME \
  "${PVSCENE_RUNTIME_PYTHON:-python}" \
  tests/scenes/isaac/assert_report.py --report "$OUT/report.json" --exit-code "$rc"