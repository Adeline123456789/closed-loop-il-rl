# Agent instructions — force_critic_act_v2_diagnostic_first

Read README.md, CODEX_TASK.md and configs/minimal_validation.yaml before implementing. This protocol supersedes the former residual/distillation MVP. Git history is archival, not an alternative implementation instruction.

The required computation graph is ACT -> executable action chunk -> real FORGE data -> return/contact-cost critics -> action gradient -> ACT parameters. Use ACT-CVAE and twin return/cost critics only. No separate residual actor, dual recoverability network, prior-gap distillation, world model, stiffness identification, QP or extra learned gating model.

Implement a genuine differentiable actor update, not only successful-trajectory BC. Freeze critic parameters during actor steps but DO NOT detach actor actions or wrap Q(actor_action) in no_grad. Include an autograd test showing that changing the critic action gradient changes the ACT update on fixed data.

Start with hardware/data/action/force-frame/EMA/reset audits. Use real Isaac Lab FORGE for robot claims. The HF sets are simulated demonstrations, not simulator checkpoints. ft is six-dimensional and smoothed; ft_noisy is three-dimensional. Check native seven-dimensional action semantics before adapting the stock four controllable motion dimensions.

Diagnose replay fidelity, finite-action-effect identifiability and action-to-parameter update transfer BEFORE full training. Run the same-data Direct-value, Direct-filter, Calibrated-filter controls. The latter two differ only by the paired contrast loss. Do not treat a finite cross-contact-mode difference as an exact derivative or safety guarantee. Count ALL prefix, branch, training and evaluation simulator steps.

Respect split isolation, continuation-policy versions, budget limits and go/no-go rules. True negative findings are valid deliverables. Use NOT_RUN/BLOCKED/INCONCLUSIVE explicitly. Never invent metrics, use mocks as robot results, silently switch to a toy task, or tune on test data.

Work on codex/force-critic-act-v2 after inspecting existing changes. Commit and push actual code, tests, configurations, logs and truthful results to the authorized repository. Do not force-push, rent GPUs, alter system drivers, upload secrets/raw datasets/caches, or claim unfinished jobs will run after the session. No new architecture without a separately recorded protocol revision.
