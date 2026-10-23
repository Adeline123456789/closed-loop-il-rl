#!/usr/bin/env bash
# Single recovery entry: isolate dependencies, then re-audit; do not skip P0.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
bash scripts/install_isaac.sh
license_accepted() {
  .venv/bin/python -c 'from forcecritic.doctor import license_state; raise SystemExit(0 if license_state()["accepted"] else 2)'
}
if ! license_accepted; then
  if [[ ! -t 0 ]]; then
    echo 'License is not accepted. Run this recovery command in an interactive terminal to review the vendor prompt.' >&2
    exit 2
  fi
  # The vendor displays its own terms and records only the user's direct answer.
  # Never set OMNI_KIT_ACCEPT_EULA or create its acceptance marker here.
  .venv/bin/python -c 'import isaacsim'
  license_accepted || { echo 'License remains unaccepted; no simulator launch.' >&2; exit 2; }
fi
FORCECRITIC_PYTHON=.venv/bin/python bash scripts/run_end_to_end.sh \
  --config configs/minimal_validation.yaml --resume auto --publish
