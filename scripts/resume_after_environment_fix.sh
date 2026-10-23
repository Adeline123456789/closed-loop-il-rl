#!/usr/bin/env bash
# Single recovery entry: isolate dependencies, then re-audit; do not skip P0.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
bash scripts/install_isaac.sh
FORCECRITIC_PYTHON=.venv/bin/python bash scripts/run_end_to_end.sh \
  --config configs/minimal_validation.yaml --resume auto --publish
