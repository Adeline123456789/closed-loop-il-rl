# Third-party provenance and licenses

All revisions are pinned in `upstream.lock.json`. `main`/HEAD was used only for discovery.

| Project | License | Use |
|---|---|---|
| tonyzhaozh/act | MIT, Copyright 2023 Tony Z. Zhao | ACT CVAE posterior, zero-latent deployment and action-query chunk architecture adapted using PyTorch modules |
| huggingface/lerobot | Apache-2.0, Copyright 2024 Hugging Face | ACT engineering reference, not installed as a second training environment |
| sfujim/TD3 | MIT | Twin critic, replay and Polyak update reference |
| sfujim/TD3_BC | MIT | Direct Q actor and BC reference; no legacy MuJoCo/D4RL installation |
| isaac-sim/IsaacLab v2.2.1 | BSD-3-Clause | Stock FORGE source audit and actual launcher; original copyright retained in upstream checkout |
| angledusgar/forge-v1, forge-failure-v1, forge-peginsert-v1 | Apache-2.0 as declared by fixed-revision model/data cards | Metadata and two PegInsert sample episodes audited; no original datasets/weights redistributed |

Full upstream source licenses are retained in `third_party/licenses/`. The local implementation does not copy a complete upstream training environment. Isaac Sim has a separate NVIDIA software license; package availability does not prove license or runtime readiness. The installation attempt and any license gate are recorded, not bypassed.

PyTorch/torchvision and other installed package licenses remain in their distributions. No paid GPU or external storage was used.
