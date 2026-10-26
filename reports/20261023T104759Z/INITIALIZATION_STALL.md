# Second-environment initialization stall: measured diagnosis and recovery

The original baseline finished ACT pretraining and 32 real evaluations (6 stable successes). It then stopped producing execution logs at 2026-10-26 07:03:51 UTC while constructing the second, eight-environment simulator. This was not Critic training or an ablation.

The original process was interrupted after preserving the checkpoint and completed evaluation. Its attempt remains FAILED_IMPLEMENTATION. A bounded real reproduction (`recreation_before_fix.*`, timeout exit 124) captured the main thread in:

```text
gym.make -> DirectRLEnv.__init__ -> Isaac Lab SimulationContext.__init__
 -> Isaac Sim SimulationContext.__init__:152 -> stop():985
 -> Isaac Lab _app_control_on_stop_handle_fn:945 -> render():685
```

Pinned Lab's constructor registers its STOP callback before its parent constructor stops an already-playing timeline. With the previous context closed but the timeline still playing, STOP invokes a render-until-playing loop before initialization can finish. CPU/GPU activity was therefore not evidence of simulation interaction.

A cleanup-only attempt stopped the timeline and opened a new stage after unsubscription. It passed creation but did not finish the second environment's reset within 180 seconds. This attempt was rejected and its code was removed; it is not presented as a successful repair.

The final approach enforces one simulator lifecycle per process. `run_robot_stage.py` runs baseline evaluation and shared collection/Critic in separate Isaac processes, aggregates their measured accounting once, and stops on a failed child or failed gate. `train_forge.py` explicitly rejects constructing a second simulator in the same process. Short-ablation evaluation reuses one live simulator across arms with the same fixed reset/noise schedules, avoiding repeated context construction.

Independent real one-environment and eight-environment probes each completed creation, reset, an actual action and shutdown (exit 0). See `recreation_isolated_1.*` and `recreation_isolated_8.*`. This verifies the lifecycle approach, not learned-policy efficacy.

The exact ACT checkpoint and original 32-episode CSV are hash-verified by `baseline_validation_resume.json`. AST comparisons confirm the evaluated policy, evaluator and adapter are unchanged; the ACT model, pretraining code and frozen config are byte-identical to the original attempt. Only orchestration, collection telemetry and critic telemetry changed. No additional ACT updates or duplicate validation episodes are silently counted as new evidence.

The interrupted evaluation and lifecycle probes are reconciled in `stall_accounting_reconciliation.json`. The rejected cleanup-only probe lost its final reset counter dump: its measured lower bound and a separate 0–10000 additional control-step bound are explicitly retained. The upper bound is not called a measured count. These are debugging costs, not diagnostic success evidence.

Runtime collection progress is now written each chunk to `baseline_live_progress.json`; Critic optimizer counts/loss are written every 100 updates. A stalled child dumps its Python stack every 300 seconds. The ACT/diagnostic/ablation gates and budgets are unchanged.
