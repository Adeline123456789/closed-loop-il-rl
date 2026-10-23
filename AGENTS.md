# Agent instructions

Read `CODEX_TASK.md` before making changes. This project is frozen at `prd_mvp_v1_prior_gap`: ACT-CVAE, mode-conditioned SAC with twin reward critics, dual recoverability heads, and a deterministic idempotent directional projection. Do not add counterfactual/conditional-mutual-information scoring, stiffness regression, world models, or a new network.

Start with data, action, sensor, simulator-version and hardware audits. In FORGE v2.2.1, the seventh action is success prediction, not a gripper command; roll/pitch policy actions are disabled. Verify the selected checkout and dataset semantics before writing control code.

Run genuine Isaac Lab FORGE rollouts for all robot claims. Offline losses and mocks are not task success. Use explicit NOT_RUN/BLOCKED/FAILED/NEGATIVE statuses, retain negative results, and never fabricate metrics. Respect the pilot budget and use common teacher data and equal interaction budgets for the three ablation branches. No GPU rental, destructive system changes, secret commits, force pushes, or raw-dataset uploads.

Commit implementation, tests, configs, reproducibility metadata and actual results to the authorized repository. The task document is a plan, not an experiment report.
