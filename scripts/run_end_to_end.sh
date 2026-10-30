#!/usr/bin/env bash
# Reports and publication are attempted even after a prerequisite fails.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${FORCECRITIC_PYTHON:-python}"
export PYTHONPATH="$PWD/src"
# Exclude inherited ROS Python 3.10 plugins from the isolated Python 3.11/3.12 environments.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
config=configs/minimal_validation.yaml
resume=auto
publish=false
demonstrations=()
while (($#)); do
  case "$1" in
    --config) config="$2"; shift 2;;
    --resume) resume="$2"; shift 2;;
    --publish) publish=true; shift;;
    --demonstrations) demonstrations=(--demonstrations "$2"); shift 2;;
    *) echo "Unknown option: $1" >&2; exit 64;;
  esac
done
if [[ "$resume" != auto ]]; then export FORCECRITIC_RUN_ID="$resume"; fi
rc=0
"$PY" -m forcecritic.cli doctor --config "$config" || rc=$?
"$PY" -m forcecritic.cli audit-data --config "$config" "${demonstrations[@]}" || rc=$?
"$PY" -m forcecritic.cli smoke --num-envs 1 --config "$config" || rc=$?
if ((rc==0)); then "$PY" -m forcecritic.cli smoke --num-envs 8 --config "$config" || rc=$?; fi
for stage in baseline diagnostic short-ablation; do
  "$PY" -m forcecritic.cli run --stage "$stage" --config "$config" "${demonstrations[@]}" || { rc=$?; break; }
done
# Conditional stages consume a stage-specific decision from the preceding stage.
# Missing/negative evidence is recorded as a stop; this entry never manufactures GO.
if ((rc==0)); then
  run_id=$("$PY" -c 'import json; print(json.load(open("reports/PROJECT_STATUS.json"))["run_id"])') || rc=$?
  for stage in pilot confirm generalization; do
    ((rc==0)) || break
    decision="reports/$run_id/decision.json"
    if [[ "$stage" == generalization ]]; then
      "$PY" -m forcecritic.cli evaluate --suite generalization --no-training --require-go "$decision" --config "$config" || rc=$?
    else
      "$PY" -m forcecritic.cli run --stage "$stage" --require-go "$decision" --config "$config" || rc=$?
    fi
  done
fi
"$PY" -m forcecritic.cli report --config "$config" || rc=$?
if "$publish"; then "$PY" scripts/publish.py || rc=$?; fi
exit "$rc"
