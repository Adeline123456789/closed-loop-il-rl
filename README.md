# Force Critic → ACT: diagnostic-first implementation

Protocol: `force_critic_act_v2_diagnostic_first`. This is a P0 **license-blocked stop delivery**: learning/control components, measured data/runtime audits and actual failed-launch evidence. **P0 has not passed; B0/B1/B2/B3 robot results are NOT_RUN.** See [project status](reports/PROJECT_STATUS.json) for the actual run and report paths.

ACT is the actor: single RGB + proprioception + four past wrench/motion/command frames → ACT-CVAE → full 4×4 native action chunk. Twin critics output return/cost; their action gradients directly update ACT. Critic weights and observation conditions stay fixed during actor updates. Deployment and Q actions use z=0. There is no residual SAC, teacher distillation, world model, recoverability head, or learned filter.

Implemented components and boundaries are documented in each run's `implementation_audit.md`. In particular, a stage command returning a blocked prerequisite is **not** a completed robot training driver. Online collection, accepted simulator integration and complete later-stage orchestration still require P0 acceptance and implementation/validation. This repository does not claim the full research pipeline has run.

## Commands actually exercised on this machine

The installed candidate `.venv/bin/python` uses Python 3.11, Torch 2.7.0+cu128, Isaac Sim 5.0.0 and the pinned Isaac Lab v2.2.1 checkout. Its GPU tensor check and CPU tests passed. The actual simulator import stopped at the Omniverse EULA prompt, before scene creation. `.venv-cpu` is the separate historical Python 3.12 / Torch 2.10 verification environment; its logs are retained.

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m forcecritic.cli doctor --config configs/minimal_validation.yaml
PYTHONPATH=src .venv/bin/python -m forcecritic.cli audit-data --config configs/minimal_validation.yaml
```

The audit downloads metadata first and only one PegInsert parquet/video episode per success/failure set at pinned HF revisions. Raw downloads are ignored by Git. The observed schema does not settle camera extrinsics, sensor reference/sign, collection checkout or pre/post EMA action semantics. Therefore these episodes are not yet admitted for BC/TD training.

Additional software checks actually exercised (neither is a robot rollout):

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' .venv-cpu/bin/python scripts/inspect_baseline.py --output reports/20261023T104759Z/baseline_architecture.json
CUDA_VISIBLE_DEVICES='' .venv-cpu/bin/python scripts/audit_expert.py --output reports/20261023T104759Z/expert_download.json
```

The baseline factory uses a pretrained frozen ResNet18, an 8×8 image feature grid, width 512, eight attention heads, four encoder layers and seven decoder layers. The checkpoint audit verifies the pinned expert file's size/hash and safely reads tensor shapes; compatibility and insertion performance still require the official player. Download the expert first with `.venv-cpu/bin/python scripts/download_expert.py` if its ignored cache is absent.

The unified entry records blockers, generates reports and attempts publication even after a stage stops:

```bash
FORCECRITIC_PYTHON=.venv/bin/python bash scripts/run_end_to_end.sh \
  --config configs/minimal_validation.yaml --resume auto --publish
```

Exit 2 means a prerequisite block; exit 1 means an implementation/command error. Shell Git authentication failure is reported separately; Codex can publish through the authenticated GitHub connector. The CLI records exact commands, return codes, source hashes, Git SHA and configuration hash. Raw robot metric files are header-only while NOT_RUN; empty files do not represent 0% success.

## Candidate Isaac installation and smoke

[Version-locked official installation source](https://github.com/isaac-sim/IsaacLab/blob/0f00ca2b4b2d54d5f90006a92abb1b00a72b2f20/docs/source/setup/installation/pip_installation.rst) specifies Python 3.11 and Isaac Sim 5.0.0. The helper below does not accept a license on the user's behalf and does not modify drivers:

```bash
bash scripts/install_isaac.sh
PYTHONPATH=src .venv/bin/python -m forcecritic.cli smoke --num-envs 1
# Only after the real 1-env test passes:
PYTHONPATH=src .venv/bin/python -m forcecritic.cli smoke --num-envs 8
```

The smoke subprocess uses `--headless --enable_cameras`, stock PegInsert, and a diagnostic camera. It checks rendered RGB and six-axis sensor finiteness, not expert insertion performance or HF camera equivalence. Its logs, not the command examples, determine what actually ran. Full dependencies and asset loading must succeed before this can pass.

The next recovery command is:

```bash
bash scripts/resume_after_environment_fix.sh
```

Run it in an interactive terminal to review the [vendor agreement](https://docs.omniverse.nvidia.com/enterprise/latest/common/NVIDIA_Omniverse_License_Agreement.html). The script displays the vendor prompt and never supplies acceptance or writes its marker itself. If acceptance is declined or unavailable, it stops. Even after acceptance, P0 alignment and later robot orchestration still need resolution; no training capability is implied by accepting a license.

## Experimental discipline

The [task specification](CODEX_TASK.md) and [frozen configuration](configs/minimal_validation.yaml) retain the diagnostic-first budgets. Physical limits/tolerances are unresolved and remain null. No long training can start before accepted P0 evidence. O/R/P equivalence precedes pair probing. All prefix/restore/branch/evaluation steps count. B2/B3 use identical data and training-only masks; B3 adds finite pair contrast, not derivative supervision. Test labels cannot release Actor masks. Checkpoints save optimizers and RNG along with model state.

B1/B2/B3 robot efficacy, replay fidelity, calibration benefit, confirmation and OOD generalization remain unmeasured. The direct gradient mechanism passing software tests supports only implementation correctness, not improved insertion. No learned weights have been published.

Licenses and pinned code/data sources: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [upstream lock](upstream.lock.json). No raw HF datasets, simulator cache, environments or credentials are committed.
