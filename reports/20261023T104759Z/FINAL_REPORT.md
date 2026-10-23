# Measured validation / stop report

Protocol: `force_critic_act_v2_diagnostic_first`. Run: `20261023T104759Z`.

## Outcome

P0 has not passed. No ACT pretraining or robot-method experiment is claimed. The software gradient path is tested separately; it is not evidence of improved insertion.

| Stage | Status | Reason |
|---|---|---|
| doctor | BLOCKED | Missing runtime: isaacsim,isaaclab; Isaac candidate requires Python 3.11; this interpreter is CPU verification only |
| audit-data | BLOCKED | BLOCKED_DATA_ALIGNMENT: collection code/SHA and pre/post EMA action stage unavailable; BLOCKED_SENSOR_CONTRACT: data-card EE frame differs from pinned stock source; sign/reference/baseline unverified; BLOCKED_CAMERA_ALIGNMENT: collection camera extrinsics/intrinsics not supplied |
| smoke-1 | BLOCKED | Real launcher attempted; see command log |
| smoke-8 | NOT_RUN | Predecessor gate not passed |
| baseline | BLOCKED | Preconditions not passed: doctor, audit-data, smoke-1, smoke-8 |
| diagnostic | BLOCKED | Preconditions not passed: doctor, audit-data, smoke-1, smoke-8, baseline |
| short-ablation | BLOCKED | Preconditions not passed: doctor, audit-data, smoke-1, smoke-8, baseline, diagnostic |
| pilot | BLOCKED | Preconditions not passed: doctor, audit-data, smoke-1, smoke-8, baseline, diagnostic, short-ablation |
| confirm | BLOCKED | Preconditions not passed: doctor, audit-data, smoke-1, smoke-8, baseline, diagnostic, short-ablation, pilot |
| generalization | BLOCKED | Preconditions not passed: doctor, audit-data, smoke-1, smoke-8, baseline, diagnostic, short-ablation, pilot, confirm |
| installation | RUNNING | Alternate official PyTorch CUDA 12.8 source after NVIDIA response connection resets |

## Questions

1. Critic → ACT: implemented in `training/updates.py`; CPU tests change critic slope on fixed ACT/data, observe changed ACT output, and verify nonzero Q-only gradients with critic parameters fixed.
2. Ordinary direct update improves insertion: **insufficient evidence**, B0/B1 not run.
3. Replay/pair fidelity: **NOT_RUN**; no same-state claim and no HF snapshot reconstruction.
4. B2 vs B1 filtering: **insufficient evidence**, NOT_RUN.
5. B3 vs B2 calibration: **insufficient evidence**, NOT_RUN. Endpoint labels already contain contrast information.
6. Continue: repair P0 environment and measured sensor/action/camera contracts first. No GO_PILOT.
7. P1–P4 skipped by the P0 gate, not an algorithm negative result.

## Raw result summary

| Arm | Status | Episodes | Success | Overload |
|---|---|---:|---|---|
| B0 | NOT_RUN | 0 | NA | NA |
| B1 | NOT_RUN | 0 | NA | NA |
| B2 | NOT_RUN | 0 | NA | NA |
| B3 | NOT_RUN | 0 | NA | NA |

## Cost and artifacts

Individual simulator transitions by category: `{"interaction": 0, "branch": 0, "prefix": 0, "restore": 0, "evaluation": 0}`.
Total accounted simulator transitions: 0. CPU unit fixtures are excluded from robot metrics.
No trained checkpoint or Release asset exists; weights: **未发布**. No fake download URL.
Header-only fidelity/pair/actor/episode CSV files mean NOT_RUN, not zero success.

## Reproduce / resume

```bash
FORCECRITIC_PYTHON=.venv-cpu/bin/python bash scripts/run_end_to_end.sh --config configs/minimal_validation.yaml --resume auto --publish
```

This resumes audit artifacts, rechecks blocked prerequisites, and reports/publishes even when a gate stops training.
Use the audited Isaac Python instead once installed. Installation alone does not resolve collection semantics.

## Scope still unverified

Real simulator integration, expert compatibility, six-axis sign/reference/baseline, camera alignment, physical safety limits, replay fidelity, and all robot training/evaluation remain unverified.
Stage execution deliberately fails closed while P0 is incomplete. Higher-stage robot orchestration is not presented as experimentally validated.
See `data_audit.json`, `doctor.json`, `action_contract.md`, `sensor_contract.md`, `implementation_audit.md`, and command logs for exact evidence.
