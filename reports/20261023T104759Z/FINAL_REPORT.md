# Measured validation / stop report

Protocol: `force_critic_act_v2_diagnostic_first`. Run: `20261023T104759Z`.

## Outcome

P0 has not passed. No ACT pretraining or robot-method experiment is claimed. The software gradient path is tested separately; it is not evidence of improved insertion.

| Stage | Status | Reason |
|---|---|---|
| doctor | PASSED | Missing runtime: ;  |
| audit-data | BLOCKED | BLOCKED_DATA_ALIGNMENT: collection code/SHA and pre/post EMA action stage unavailable; BLOCKED_SENSOR_CONTRACT: data-card EE frame differs from pinned stock source; sign/reference/baseline unverified; BLOCKED_CAMERA_ALIGNMENT: collection camera extrinsics/intrinsics not supplied |
| smoke-1 | FAILED_IMPLEMENTATION | Real launcher attempted; see command log |
| smoke-8 | NOT_RUN | Predecessor gate not passed |
| baseline | BLOCKED | Preconditions not passed: audit-data, smoke-1, smoke-8 |
| diagnostic | BLOCKED | Preconditions not passed: audit-data, smoke-1, smoke-8, baseline |
| short-ablation | BLOCKED | Preconditions not passed: audit-data, smoke-1, smoke-8, baseline, diagnostic |
| pilot | BLOCKED | Preconditions not passed: audit-data, smoke-1, smoke-8, baseline, diagnostic, short-ablation |
| confirm | BLOCKED | Preconditions not passed: audit-data, smoke-1, smoke-8, baseline, diagnostic, short-ablation, pilot |
| generalization | BLOCKED | Preconditions not passed: audit-data, smoke-1, smoke-8, baseline, diagnostic, short-ablation, pilot, confirm |
| installation | PASSED | Candidate Python 3.11 / cu128 Torch / pinned Isaac Sim and Lab dependencies installed; camera/physics still require smoke |

## Runtime and inspected evidence

Hardware: NVIDIA GeForce RTX 3090, 24576 MiB, 575.51.03.
Interpreter: `/media/midea/F496AC8096AC4548/closed_loop_act/.venv/bin/python`; Python `3.11.15`; Torch `2.7.0+cu128`; Torch CUDA `12.8`; Isaac Sim `5.0.0.0`.
Isaac Lab checkout: `0f00ca2b4b2d54d5f90006a92abb1b00a72b2f20` (v2.2.1; distribution metadata version is a separate identifier).
Omniverse acceptance indicators: `{"status": "ACCEPTED", "accepted": true, "acceptance_marker_exists": true, "environment_acceptance_flag": false, "agreement_url": "https://docs.omniverse.nvidia.com/enterprise/latest/common/NVIDIA_Omniverse_License_Agreement.html"}`. No acceptance is inferred from installation.
Loaded assets, camera rendering, expert compatibility and robot efficacy require their own successful measured records. Package installation and the CUDA tensor check do not establish them.

- `cpu_tests`: return code `0`; 39 passed in 1.40s.
- `diff_check`: return code `0`; see command log.
- `angledusgar/forge-v1`: 984 PegInsert episodes in metadata; inspected 1 complete episode(s), 149 frames, at revision `0ebe4ff65ade3da8d2011d66cce0259870c95f78`.
- `angledusgar/forge-failure-v1`: 26 PegInsert episodes in metadata; inspected 1 complete episode(s), 149 frames, at revision `820392e1167df4d6f5d60f985ec4f3b1b4fd6237`.

Expert download integrity: `VERIFIED`; safe tensor-shape inspection: `PASSED`; official-player compatibility: `NOT_RUN`. No expert model-card performance is counted as a project result.

## Questions

1. Critic → ACT: implemented in `training/updates.py`; CPU tests change critic slope on fixed ACT/data, observe changed ACT output, and verify nonzero Q-only gradients with critic parameters fixed.
2. Ordinary direct update improves insertion: **insufficient evidence**, B0/B1 not run.
3. Replay fidelity rows: 0; diagnostic stage status: BLOCKED. Row count alone is not evidence of equivalence.
4. B2 vs B1 filtering: **insufficient evidence**, NOT_RUN.
5. B3 vs B2 calibration: **insufficient evidence**, NOT_RUN. Endpoint labels already contain contrast information.
6. Recorded continuation decision: **BLOCKED** — BLOCKED_LICENSE: Omniverse Kit requested EULA acceptance; no user acceptance was supplied and no scene was created.
7. Stages not passed/completed: audit-data, smoke-1, smoke-8, baseline, diagnostic, short-ablation, pilot, confirm, generalization. Reasons are listed above; an environment block is not an algorithm negative result.

## Raw result summary

| Arm / seed / split / policy | Status | Episodes | Success | Overload |
|---|---|---:|---|---|
| B0 / NA / NA / NA | NOT_RUN | 0 | NA | NA |
| B1 / NA / NA / NA | NOT_RUN | 0 | NA | NA |
| B2 / NA / NA / NA | NOT_RUN | 0 | NA | NA |
| B3 / NA / NA / NA | NOT_RUN | 0 | NA | NA |

## Cost and artifacts

Individual simulator transitions by category: `{"interaction": 0, "branch": 0, "prefix": 0, "restore": 0, "evaluation": 0}`.
Total accounted simulator transitions: 0. CPU unit fixtures are excluded from robot metrics.
Checkpoint index entries: 0. Publication is not implied by a local checkpoint; only verified Release entries may be used as download links.
Header-only fidelity/pair/actor/episode CSV files mean NOT_RUN, not zero success.

## Reproduce / resume

```bash
FORCECRITIC_PYTHON=.venv/bin/python bash scripts/run_end_to_end.sh --config configs/minimal_validation.yaml --resume auto --publish
```

This resumes audit artifacts, rechecks blocked prerequisites, and reports/publishes even when a gate stops training.
For the license boundary, run `bash scripts/resume_after_environment_fix.sh` in an interactive terminal. It displays the vendor prompt if needed and never supplies acceptance automatically. Acceptance alone does not resolve data/action/sensor/camera semantics.

## Scope still unverified

Consult the stage table and implementation_audit.md for unresolved interfaces and unimplemented orchestration; CPU software tests never establish robot efficacy.
Stage execution deliberately fails closed while P0 is incomplete. Higher-stage robot orchestration is not presented as experimentally validated.
See `data_audit.json`, `doctor.json`, `action_contract.md`, `sensor_contract.md`, `implementation_audit.md`, and command logs for exact evidence.
