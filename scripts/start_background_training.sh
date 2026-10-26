#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .cache
if tmux has-session -t forcecritic-training 2>/dev/null; then
  echo 'Existing session: forcecritic-training; no duplicate launched.'
  exit 0
fi
if pgrep -f '[p]ython.*(scripts/train_forge.py|scripts/collect_demonstrations.py|scripts/background_training.py)' >/dev/null; then
  echo 'A training/collection process already exists; inspect it before launching another.' >&2
  exit 1
fi
tmux new-session -d -s forcecritic-training -c "$PWD" \
  'set -o pipefail; PYTHONPATH=src PYTHONUNBUFFERED=1 .venv/bin/python scripts/background_training.py 2>&1 | tee -a .cache/background_training.log'
echo 'Started tmux session forcecritic-training; log: .cache/background_training.log'
