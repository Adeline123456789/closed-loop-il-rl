# Implementation and completion boundary

Implemented and CPU-tested: ACT-CVAE RGB/proprio/H=4 history model; twin R/C critics; Q-only and Q+BC+step updates; fixed condition features; TD/MC/pair losses; target updates; version checks; same-data arm cloning and bounded short-update engine; atomic model/optimizer/RNG checkpoint save/restore; block replay; action/wrench contracts; chunk termination and cost accounting; group splits; finite effects and equivalence checks; metadata-first real HF audit; real Isaac launch script; evidence CLI/report/publication.

Not implemented as a validated end-to-end robot system: accepted observation/reward/camera adapter, expert demonstration recollection, online replay collection, O/R/P state capture and replay, 648-branch experiment scheduler, full pilot/confirmation/generalization execution. The stage commands enforce predecessor gates; a parser or prerequisite check is not a completed stage runner. This is a P0 stop delivery, not a claim that all later orchestration is complete.

No pretrained ACT checkpoint exists. Default visual model must use pretrained ResNet18 weights before any baseline training; CPU gradient fixtures use an injected small backbone and are not baselines. No robot success claim follows from these fixtures.

All later experimentation is contingent on P0. The unresolved data/control contracts justify stopping before long training under the requested protocol. Work completed here remains useful and independently testable; robot-stage completion is not asserted.
