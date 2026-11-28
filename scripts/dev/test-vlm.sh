#!/usr/bin/env bash
# Tier D: a single real Visual Critic call. MANUAL OPT-IN ONLY.
#
# This sends a rendered image to an external service. It is never part of the default
# gate, and it will refuse to run unless PVSCENE_ALLOW_EXTERNAL_VLM=1 is set explicitly,
# so that no automated path can publish an image by accident.
#
# What may be sent is fixed by architecture §9: the rendered image plus opaque semantic
# IDs. File paths, logs, catalog records, checkpoints and credential values must never
# be in the payload; the test asserts this against a deny-list.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [ "${PVSCENE_ALLOW_EXTERNAL_VLM:-0}" != "1" ]; then
  echo "refusing to run: this sends a rendered image to an external service." >&2
  echo "re-run with PVSCENE_ALLOW_EXTERNAL_VLM=1 to acknowledge." >&2
  exit 2
fi

exec env -u PYTHONPATH -u PYTHONHOME \
  "${PVSCENE_RUNTIME_PYTHON:-python}" \
  -m pytest tests/scenes/vlm -m vlm -p no:cacheprovider -q "$@"