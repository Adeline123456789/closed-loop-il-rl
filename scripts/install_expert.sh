#!/usr/bin/env bash
# Pin publisher-player code and preserve Isaac's exact runtime requirements.
set -euo pipefail
cd "$(dirname "$0")/.."
archive=.cache/rl_games-6b3534f.zip
mkdir -p .cache
[[ -f "$archive" ]] || curl -L --fail --retry 3 --max-time 300 \
  https://codeload.github.com/isaac-sim/rl_games/zip/6b3534f29568158e9e29ec8bf83cc88fce5f0cae -o "$archive"
.venv/bin/python - <<'PY'
from pathlib import Path
import hashlib
p=Path('.cache/rl_games-6b3534f.zip')
if hashlib.sha256(p.read_bytes()).hexdigest()!='0551a6da42d49b95b01865309c9eab0ee057fa39d51b075d18ecc27c6f335941':
    raise SystemExit('Publisher player archive checksum mismatch')
PY
uv pip install --python .venv/bin/python --default-index https://pypi.org/simple \
  --constraint configs/isaac_constraints.txt --constraint configs/isaac_runtime_constraints.txt \
  --constraint configs/expert_constraints.txt \
  --build-constraints configs/isaac_build_constraints.txt "$archive" onnx googleapis-common-protos
uv pip check --python .venv/bin/python
