# Measured validation / stop report

Protocol: `force_critic_act_v2_diagnostic_first`. Run: `20261023T104759Z`.

## Outcome

Raw completed episodes are summarized separately by seed, split and checkpoint; no pooling across policy versions. See paired_comparisons.csv for matched-parent differences.

| Stage | Status | Reason |
|---|---|---|
| doctor | PASSED | Missing runtime: ;  |
| audit-data | PASSED | Accepted hashed same-version recordings; HF semantics remain unresolved |
| smoke-1 | PASSED | Real launcher attempted; see command log |
| smoke-8 | PASSED | Real launcher attempted; see command log |
| baseline | PASSED | ACT0 has measured insertions; ordinary critic fitted on actual chunk transitions |
| diagnostic | FAILED_IMPLEMENTATION | TypeError('Object of type bool_ is not JSON serializable') |
| short-ablation | NOT_RUN | Not reached: same-version recollection is still running; all real predecessor gates remain required |
| pilot | NOT_RUN | Not reached: same-version recollection is still running; all real predecessor gates remain required |
| confirm | NOT_RUN | Not reached: same-version recollection is still running; all real predecessor gates remain required |
| generalization | NOT_RUN | Not reached: same-version recollection is still running; all real predecessor gates remain required |
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
2. Ordinary direct update: see B1-B0 matched-parent comparison; causal/continuation decisions require all diagnostic gates.
3. Replay fidelity rows: 36; diagnostic stage status: FAILED_IMPLEMENTATION. Row count alone is not evidence of equivalence.
4. B2 vs B1 filtering: see matched-parent comparison and independent diagnostic evidence.
5. B3 vs B2 calibration: see matched-parent comparison; endpoint labels already contain contrast information.
6. Recorded continuation decision: **NOT_RUN** — P0 accepted and current smoke-1/smoke-8 passed. ACT pretraining is running; no diagnostic continuation decision has been measured yet. See training_progress.json..
7. Stages not passed/completed: diagnostic, short-ablation, pilot, confirm, generalization. Reasons are listed above; an environment block is not an algorithm negative result.

## Raw result summary

| Arm / seed / split / policy | Status | Episodes | Success | Overload |
|---|---|---:|---|---|
| B0 / 0 / validation / 6343385 | COMPLETED | 32 | 0.1875 | 0.90625 |
| B1 / NA / NA / NA | NOT_RUN | 0 | NA | NA |
| B2 / NA / NA / NA | NOT_RUN | 0 | NA | NA |
| B3 / NA / NA / NA | NOT_RUN | 0 | NA | NA |

## Cost and artifacts

Individual simulator transitions by category: `{"interaction": 71539, "branch": 5909, "prefix": 337, "restore": 7111, "evaluation": 4412}`.
Total accounted simulator transitions: 89308. CPU unit fixtures are excluded from robot metrics.
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

## Accounting uncertainty

The accounted total above is a measured lower bound. A terminated lifecycle debugging probe lacked its final counter dump; its bounded unobserved advances are recorded separately, not invented as zero or mixed into measured counts.
```json
[
  {
    "source": "reports/20261023T104759Z/recreation_after_fix.json",
    "phase": "second environment reset in rejected cleanup-only probe",
    "category": "restore",
    "additional_individual_control_steps_min": 0,
    "additional_individual_control_steps_max": 10000,
    "basis": "Last saved ledger is before reset; PhysicsLedger enforces a 10000 individual-control-step cap. Process terminated without a final counter snapshot."
  }
]
```
