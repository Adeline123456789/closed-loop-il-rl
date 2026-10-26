#!/usr/bin/env bash
# Explicit foreground continuation; every later phase requires real predecessor evidence.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
TASK_PYTHON=.venv/bin/python
TASK_RUN_ID=20261023T104759Z
TASK_MANIFEST=data/act_demonstrations/manifest.json
TASK_EXPERT=.cache/expert/447a6109cec7c228763d1bfc8cf0ab73f6558a1b/forge_peginsert_2026-06-30.pth
while pgrep -f '[p]ython.*scripts/collect_demonstrations.py' >/dev/null; do
  echo 'Existing demonstration collection is still running; waiting 10 seconds.'
  sleep 10
done
if [[ ! -f "$TASK_MANIFEST" ]]; then
  "$TASK_PYTHON" scripts/collect_demonstrations.py --run-id "$TASK_RUN_ID" --checkpoint "$TASK_EXPERT"
fi
"$TASK_PYTHON" -m forcecritic.cli doctor --run-id "$TASK_RUN_ID"
"$TASK_PYTHON" -m forcecritic.cli audit-data --run-id "$TASK_RUN_ID" --demonstrations "$TASK_MANIFEST"
"$TASK_PYTHON" -m forcecritic.cli smoke --num-envs 1 --run-id "$TASK_RUN_ID"
"$TASK_PYTHON" -m forcecritic.cli smoke --num-envs 8 --run-id "$TASK_RUN_ID"
for TASK_STAGE in baseline diagnostic short-ablation; do
  if "$TASK_PYTHON" -m forcecritic.cli run --stage "$TASK_STAGE" --run-id "$TASK_RUN_ID" --demonstrations "$TASK_MANIFEST"; then
    :
  else
    TASK_RC=$?
    "$TASK_PYTHON" -m forcecritic.cli report --run-id "$TASK_RUN_ID"
    echo "Stopped at $TASK_STAGE (exit $TASK_RC); see reports/$TASK_RUN_ID/."
    exit "$TASK_RC"
  fi
done
"$TASK_PYTHON" -m forcecritic.cli report --run-id "$TASK_RUN_ID"
echo "Short ablations finished. Inspect reports/$TASK_RUN_ID/decision.json before any pilot."
