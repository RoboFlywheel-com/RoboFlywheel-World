#!/usr/bin/env bash
# No model calls or GPU simulator jobs are launched by the default gate.
set -euo pipefail
RFW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RFW_ROOT"
RFW_CHECK_PYTHON="${RFW_CHECK_PYTHON:-${PVSCENE_RUNTIME_PYTHON:-python}}"
env -u PYTHONPATH -u PYTHONHOME "$RFW_CHECK_PYTHON" - <<'PY'
import ast
from pathlib import Path
paths = [p for root in ('src', 'scripts', 'tests') for p in Path(root).rglob('*.py') if '__pycache__' not in p.parts]
for path in paths:
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
print(f'Python syntax: {len(paths)} files passed')
PY
env -u PYTHONPATH -u PYTHONHOME "$RFW_CHECK_PYTHON" scripts/scan_sensitive.py
if [ "${1:-}" = "--stdlib-only" ]; then
  exec bash scripts/dev/test-scenes.sh "$@"
fi
env -u PYTHONPATH -u PYTHONHOME "$RFW_CHECK_PYTHON" -m pytest tests/assets scripts/tests tests/test_*.py -q -p no:cacheprovider
bash scripts/dev/test-scenes.sh "$@"
