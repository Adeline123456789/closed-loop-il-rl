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
while (($#)); do
  case "$1" in
    --config) config="$2"; shift 2;;
    --resume) resume="$2"; shift 2;;
    --publish) publish=true; shift;;
    *) echo "Unknown option: $1" >&2; exit 64;;
  esac
done
if [[ "$resume" != auto ]]; then export FORCECRITIC_RUN_ID="$resume"; fi
rc=0
"$PY" -m forcecritic.cli doctor --config "$config" || rc=$?
"$PY" -m forcecritic.cli audit-data --config "$config" || rc=$?
"$PY" -m forcecritic.cli smoke --num-envs 1 --config "$config" || rc=$?
if ((rc==0)); then "$PY" -m forcecritic.cli smoke --num-envs 8 --config "$config" || rc=$?; fi
for stage in baseline diagnostic short-ablation; do
  "$PY" -m forcecritic.cli run --stage "$stage" --config "$config" || { rc=$?; break; }
done
# GO stages can only be entered by a versioned decision; never auto-create one.
"$PY" -m forcecritic.cli report --config "$config" || rc=$?
if "$publish"; then "$PY" scripts/publish.py || rc=$?; fi
exit "$rc"
