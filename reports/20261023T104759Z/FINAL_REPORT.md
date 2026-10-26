# Measured validation / stop report

Protocol: `force_critic_act_v2_diagnostic_first`. Run: `20261023T104759Z`.

## Outcome

P0 has passed for same-version recollection; ACT robot outcomes have not been measured. Expert results and software gradients are not ACT insertion evidence.

| Stage | Status | Reason |
|---|---|---|
| doctor | PASSED | Missing runtime: ;  |
| audit-data | PASSED | Accepted hashed same-version recordings; HF semantics remain unresolved |
| smoke-1 | PASSED | Real launcher attempted; see command log |
| smoke-8 | PASSED | Real launcher attempted; see command log |
| baseline | NOT_RUN | No completed stage outcome yet. Live process/optimizer progress and queued successors are in training_progress.json. |
| diagnostic | NOT_RUN | No completed stage outcome yet. Live process/optimizer progress and queued successors are in training_progress.json. |
| short-ablation | NOT_RUN | No completed stage outcome yet. Live process/optimizer progress and queued successors are in training_progress.json. |
| pilot | NOT_RUN | No completed stage outcome yet. Live process/optimizer progress and queued successors are in training_progress.json. |
| confirm | NOT_RUN | No completed stage outcome yet. Live process/optimizer progress and queued successors are in training_progress.json. |
| generalization | NOT_RUN | No completed stage outcome yet. Live process/optimizer progress and queued successors are in training_progress.json. |
| installation | PASSED | Candidate Python 3.11 / cu128 Torch / pinned Isaac Sim and Lab dependencies installed; camera/physics still require smoke |

## Runtime and inspected evidence

Hardware: NVIDIA GeForce RTX 3090, 24576 MiB, 575.51.03.
Interpreter: `/media/midea/F496AC8096AC4548/closed_loop_act/.venv/bin/python`; Python `3.11.15`; Torch `2.7.0+cu128`; Torch CUDA `12.8`; Isaac Sim `5.0.0.0`.
Isaac Lab checkout: `0f00ca2b4b2d54d5f90006a92abb1b00a72b2f20` (v2.2.1; distribution metadata version is a separate identifier).
Omniverse acceptance indicators: `{"status": "ACCEPTED", "accepted": true, "acceptance_marker_exists": true, "environment_acceptance_flag": false, "agreement_url": "https://docs.omniverse.nvidia.com/enterprise/latest/common/NVIDIA_Omniverse_License_Agreement.html"}`. No acceptance is inferred from installation.
Loaded assets, camera rendering, expert compatibility and robot efficacy require their own successful measured records. Package installation and the CUDA tensor check do not establish them.

- `cpu_tests`: return code `0`; 49 passed in 1.47s.
- `diff_check`: return code `0`; see command log.
- `angledusgar/forge-v1`: 984 PegInsert episodes in metadata; inspected 1 complete episode(s), 149 frames, at revision `0ebe4ff65ade3da8d2011d66cce0259870c95f78`.
- `angledusgar/forge-failure-v1`: 26 PegInsert episodes in metadata; inspected 1 complete episode(s), 149 frames, at revision `820392e1167df4d6f5d60f985ec4f3b1b4fd6237`.

Expert download integrity: `VERIFIED`; safe tensor-shape inspection: `PASSED`. Measured expert development episodes: 8; stable successes: 8. No expert model-card performance is counted as a project result.

## Questions

1. Critic → ACT: implemented in `training/updates.py`; CPU tests change critic slope on fixed ACT/data, observe changed ACT output, and verify nonzero Q-only gradients with critic parameters fixed.
2. Ordinary direct update improves insertion: **insufficient evidence**, B0/B1 not run.
3. Replay fidelity rows: 0; diagnostic stage status: NOT_RUN. Row count alone is not evidence of equivalence.
4. B2 vs B1 filtering: **insufficient evidence**, NOT_RUN.
5. B3 vs B2 calibration: **insufficient evidence**, NOT_RUN. Endpoint labels already contain contrast information.
6. Recorded continuation decision: **NOT_RUN** — P0 is accepted; ACT training is active. Diagnostic/ablation go/no-go has not yet been measured. See training_progress.json.
7. Stages not passed/completed: baseline, diagnostic, short-ablation, pilot, confirm, generalization. Reasons are listed above; an environment block is not an algorithm negative result.

## Raw result summary

| Arm / seed / split / policy | Status | Episodes | Success | Overload |
|---|---|---:|---|---|
| B0 / NA / NA / NA | NOT_RUN | 0 | NA | NA |
| B1 / NA / NA / NA | NOT_RUN | 0 | NA | NA |
| B2 / NA / NA / NA | NOT_RUN | 0 | NA | NA |
| B3 / NA / NA / NA | NOT_RUN | 0 | NA | NA |

## Cost and artifacts

Individual simulator transitions by category: `{"interaction": 57224, "branch": 0, "prefix": 0, "restore": 4972, "evaluation": 172}`.
Total accounted simulator transitions: 62368. CPU unit fixtures are excluded from robot metrics.
Checkpoint index entries: 0. Publication is not implied by a local checkpoint; only verified Release entries may be used as download links.
Header-only fidelity/pair/actor/episode CSV files mean NOT_RUN, not zero success.

## Reproduce / resume

```bash
FORCECRITIC_PYTHON=.venv/bin/python bash scripts/run_end_to_end.sh --config configs/minimal_validation.yaml --resume auto --demonstrations data/act_demonstrations/manifest.json --publish
```

This resumes audit artifacts, rechecks blocked prerequisites, and reports/publishes even when a gate stops training.
Current doctor and actual smoke records determine runtime status. Historical license failures do not supersede later successful real launches. The new P0 contract does not retroactively validate old HF action/sensor/camera semantics.

## Scope still unverified

Consult the stage table and implementation_audit.md for unresolved interfaces and unimplemented orchestration; CPU software tests never establish robot efficacy.
Stage execution deliberately fails closed while P0 is incomplete. Higher-stage robot orchestration is not presented as experimentally validated.
See `data_audit.json`, `doctor.json`, `action_contract.md`, `sensor_contract.md`, `implementation_audit.md`, and command logs for exact evidence.
