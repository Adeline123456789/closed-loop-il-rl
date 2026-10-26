# Publication snapshot

Cutoff: 2026-10-26T03:30:49.939276+00:00.

- Real stock FORGE smoke-1 and smoke-8 passed after fixing Kit 107 registry selection and USD camera cloning. The EULA has been accepted. Historical license failures are archival.
- Real 24-step native action/EMA/six-axis force-filter checks passed. The camera image was visually reviewed. P0 acceptance is for newly collected same-version data; old HF semantics remain unresolved.
- Development expert: 8 stable successes / 8 completed episodes. Formal recollection cutoff: 12/20 completed runs, 212 stable successes / 216 completed episodes. All formal cutoff runs are train-partition runs; planned validation/test runs have not yet completed.
- ACT pretraining, online Critic fit, real Critic→ACT diagnostic and B1/B2/B3 outcomes: NOT_RUN. Later live drivers are implemented but await actual stage execution. Conditional pilot/confirmation/generalization remain incomplete.
- The active collection is an existing local process; this publication does not launch training. New local data after the cutoff is not included in these snapshot totals. Inspect data/act_demonstrations/progress.json for live counts.
- Some early-stage source verification hashes changed while later driver code was implemented. Their measured logs remain valid historical measurements; scripts/continue_training.sh refreshes doctor and repeats smoke prerequisites before training.
- Raw arrays, large weights/replay and Isaac caches are local. Published manifests identify raw files and SHA256 values. See TRAINING_GUIDE.md for exact paths, stage commands and progress monitoring.
