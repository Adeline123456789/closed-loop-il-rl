# 本机数据与后续训练

项目目录：`/media/midea/F496AC8096AC4548/closed_loop_act`。分支：`codex/force-critic-act-v2`。运行 ID：`20261023T104759Z`。

本次发布快照见 `reports/20261023T104759Z/publication_snapshot/status.json`。真实 smoke-1、smoke-8 已通过；ACT 已训练 10,000 次并完成真实评估（6/32 稳定成功），共享交互 96 回合，普通 Critic 已更新 10,000 次。当前 D0 的真实回放门槛未通过，D1/D2 和 B1/B2/B3 未运行。

**当前不是训练中，而是按协议停止。** 修复后的 7 条保存轨迹给出 R/P 各 0/3 通过；另有原生段错误及未落盘步数未知的问题。新的守护器会阻止在预算缺口未核实前再次启动正式诊断，不会自动改容差、扩大预算或跳过门槛。详见 [修复与停止报告](reports/20261023T104759Z/DIAGNOSTIC_REPAIR.md)。下面的启动命令是操作接口，不表示当前已允许继续消融。

## 数据在哪里

以下路径均相对项目目录：

| 内容 | 本地位置 | 状态 |
|---|---|---|
| 新版 FORGE 示教 | `data/act_demonstrations/` | 20 个独立采集 run；前 14 train，中间 3 validation，最后 3 test |
| 合并训练清单 | `data/act_demonstrations/manifest.json` | 全部采集成功完成后才生成；当前没有此文件时不能启动 ACT |
| 采集进度 | `data/act_demonstrations/progress.json` | 已完成 run 和成功回合数 |
| 开发专家数据 | `data/development_1000_retry/` | 8 个成功开发回合，只用于审计/阈值，不进入训练或测试 |
| 首次失败采集 | `data/development_1000/` | 保留实际失败，不作成功示教 |
| HF 成功数据审计缓存 | `.cache/hf/angledusgar--forge-v1/0ebe4ff65ade3da8d2011d66cce0259870c95f78/` | metadata、1 条 PegInsert parquet 和对应视频；未下载完整数据集 |
| HF 失败数据审计缓存 | `.cache/hf/angledusgar--forge-failure-v1/820392e1167df4d6f5d60f985ec4f3b1b4fd6237/` | metadata、1 条 PegInsert parquet 和对应视频；未下载完整数据集 |
| 官方专家权重 | `.cache/expert/447a6109cec7c228763d1bfc8cf0ab73f6558a1b/forge_peginsert_2026-06-30.pth` | 213,431,615 字节，SHA256 已锁定 |
| ACT/Critic 后续权重 | `checkpoints/20261023T104759Z/` | 训练启动后生成；当前没有 ACT 权重 |

每个新采集 run 内有 `manifest.json`、`expert_smoke.csv` 和各回合目录。回合文件包含 `rgb.npy`、`state.npy`、`wrench.npy`、`motion.npy`、`previous_command.npy`、`action.npy`、原始子步力、EMA 后动作和实际目标。完整原始回合保留；BC 只读取第一次连续三步几何成功之前的有效前缀。失败回合不进入正向 BC。

GitHub 上传代码、配置、真实日志、CSV、清单和小图。原始数据、Isaac 缓存及大权重按 CODEX_TASK.md 保留在本机；仓库中的数据清单携带各文件 SHA256。旧 HF 的动作/相机契约仍未对齐，因此训练使用新采集数据。

## 推荐：受管理的后台顺序继续（用户已授权）

20 个采集批次已完成：376 回合，其中 367 个成功，合并 manifest 已存在。数组字段详解与 MP4 导出见 [DATA_FORMAT.md](DATA_FORMAT.md)。

```bash
cd /media/midea/F496AC8096AC4548/closed_loop_act
bash scripts/start_background_training.sh
tail -f .cache/background_training.log
```

启动器会拒绝重复训练进程。会话名 `forcecritic-training`，终端关闭后仍由本机 tmux 管理（机器必须保持开机且不休眠）。先发布当前采集快照，再运行 doctor → audit-data → smoke-1 → smoke-8 → baseline → diagnostic → short-ablation。初始快照、smoke 完成、后续各阶段结束及失败停止时均尝试自动 commit/push 并更新 PR #1。上传失败单独记录在 `.cache/background_publication.json`；不会被写成实验失败或成功。网络操作有超时和一次重试。

```bash
tmux list-sessions
tmux attach -t forcecritic-training
# 查看后按 Ctrl-b，再按 d 脱离，会继续运行。
cat reports/20261023T104759Z/background_job.json
cat .cache/background_publication.json
```

先检查已有会话和 `background_job.json`，不要同时运行下列前台命令。任何门槛失败即停，不自动扩预算，也不自动启动 pilot/confirm/generalization。

## 可选：前台顺序继续

```bash
cd /media/midea/F496AC8096AC4548/closed_loop_act
bash scripts/continue_training.sh 2>&1 | tee .cache/continue_training.log
```

脚本检测到现有 `collect_demonstrations.py` 进程时会等待，避免重复采集。全部示教完成后，按顺序执行 doctor、示教审计、smoke-1、smoke-8、baseline、diagnostic、short-ablation；每个阶段都检查真实前置证据。它不会自动运行未完成调度的 pilot/confirm/generalization。任一实验门槛或实现错误都会停止并生成报告，不会越过门槛。

采集结束后也可逐阶段执行：

```bash
export PYTHONPATH="$PWD/src"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
.venv/bin/python -m forcecritic.cli doctor --run-id 20261023T104759Z
.venv/bin/python -m forcecritic.cli audit-data --run-id 20261023T104759Z --demonstrations data/act_demonstrations/manifest.json
.venv/bin/python -m forcecritic.cli smoke --num-envs 1 --run-id 20261023T104759Z
.venv/bin/python -m forcecritic.cli smoke --num-envs 8 --run-id 20261023T104759Z
.venv/bin/python -m forcecritic.cli run --stage baseline --run-id 20261023T104759Z --demonstrations data/act_demonstrations/manifest.json
.venv/bin/python -m forcecritic.cli run --stage diagnostic --run-id 20261023T104759Z --demonstrations data/act_demonstrations/manifest.json
.venv/bin/python -m forcecritic.cli run --stage short-ablation --run-id 20261023T104759Z --demonstrations data/act_demonstrations/manifest.json
.venv/bin/python -m forcecritic.cli report --run-id 20261023T104759Z
```

逐条检查退出码。`baseline` 先用固定 200 条成功训练回合做 10,000 次 ACT 更新，再实际评估 32 个初始条件；只有行为与回报范围合格才训练普通 Critic。诊断先验证回放保真度，之后才采集配对与验证真实 Critic→ACT 更新。短消融按相同数据执行每组 2,000 次 Critic / 200 次 Actor 更新并实际评估。任何 NOT_RUN、BLOCKED、INCONCLUSIVE、NO_GO 都不是算法成功结果。

若以后需要重启采集，先确认原进程确已退出，再执行下列命令。已完成 run 可复用；遇到未完成/失败的不可变 run 会明确报错，不能覆盖它来隐藏尝试或额外预算。

```bash
PYTHONPATH=src .venv/bin/python scripts/collect_demonstrations.py \
  --run-id 20261023T104759Z \
  --checkpoint .cache/expert/447a6109cec7c228763d1bfc8cf0ab73f6558a1b/forge_peginsert_2026-06-30.pth
```

## 查看进度

各子阶段的中文解释、目标数量和 B0/B1/B2/B3 的确切区别见 [EXPERIMENT_PROGRESS.md](EXPERIMENT_PROGRESS.md)。面板已改为逐项清单；正在运行的 D0 只能核实到完整检查点的完成数，点内未提供的精确轨迹计数不会猜测。后续短消融逐组输出 Critic/ACT 更新计数及每回合评估进度。

推荐直接查看实时面板，显示 ACT 更新数/损失、真实评估回合数、后续诊断与消融队列、进程存活及上传状态：

```bash
cd /media/midea/F496AC8096AC4548/closed_loop_act
watch -n 5 '.venv/bin/python scripts/training_status.py'
```

后台监控每 30 秒向 `.cache/background_training.log` 追加 `LIVE` 心跳，完整状态写入 `reports/20261023T104759Z/training_progress.json`，从真实 checkpoint 中导出损失到 `results/20261023T104759Z/act_pretraining_loss.csv`。它不改训练参数、不重启优化器。`background_job.json` 的时间是进入阶段的时间，不是训练更新的时间；`run_manifest.json` 的阶段结果通常在阶段结束后写入，不能将其中旧的 `NOT_RUN` 当作当前没有进程。

面板另显示“最近实际执行记录距今”：只统计执行日志、checkpoint 与真实结果，排除监控心跳和派生损失 CSV。超过 5 分钟无执行记录会提示“疑似停滞或缺少遥测”，不会继续假装能估算完成时间。这个提示本身不是死锁的证明。ACT 100% 只表示预训练完成；只有 short-ablation 有实际完成结果、决策文件已写入，并且最近上传退出码为 0，才表示这一轮最小消融已完成并发布。STOPPED/FAILED 表示中途停止，必须看原因。

队列为 baseline（ACT 10000 更新 → 32 回合评估 → 共享真实交互 → 普通 Critic 最多 10000 更新）、diagnostic（保真度/配对/真实 Critic→ACT）、short-ablation（B1/B2/B3 各 2000 Critic + 200 Actor 更新，B0–B3 各 32 回合真实评估）。门槛未通过就停止并显示原因。共享采集每个动作块输出已完成回合、独立环境控制步和物理账本；Critic 每 100 次更新输出计数和 loss，见 `baseline_live_progress.json`。pilot/confirm/generalization 不在自动队列中。

2026-10-26 已定位同一 Isaac 进程重建第二个环境时的 STOP 回调等待循环；基线评估和共享采集现用独立进程，短消融各组复用同一个环境并按相同 seed 重置。原始 10000-update ACT 与 32 回合评估按 hash 恢复，不重新训练或重复计作新证据。完整诊断、失败尝试和计数不确定性见 `reports/20261023T104759Z/INITIALIZATION_STALL.md`。

采集总日志、当前子进程日志和 GPU：

```bash
tail -f .cache/collect_demonstrations_20261026.log
watch -n 5 cat data/act_demonstrations/progress.json
pgrep -af 'collect_demonstrations.py|collect_forge.py|train_forge.py'
watch -n 2 nvidia-smi
```

总日志在一个采集 run 结束时输出；run 内每个控制步的计数更新在对应 `data/act_demonstrations/NN_split_seed/manifest.json`，可 `watch -n 5` 查看。最新仿真日志在 `reports/20261023T104759Z/p0_collect_*.log`。

训练开始后：

```bash
tail -f .cache/background_training.log
watch -n 10 '.venv/bin/python -c "import json,pathlib; p=pathlib.Path(\"checkpoints/20261023T104759Z/act0.json\"); d=json.loads(p.read_text()) if p.exists() else {}; print({k:d.get(k) for k in (\"status\",\"completed_updates\",\"requested_updates\")})"'
```

`act0.json` 每 100 次更新保存进度；模型与优化器在滚动覆盖的 `act0.pt`。独立后台保留器每 1,000 次将完整 checkpoint 留在 `checkpoints/20261023T104759Z/act0_milestones/act0_update_XXXXXX.pt`；启用时另存当前版本。早于启用时间且已被覆盖的版本无法补回，详见 `reports/20261023T104759Z/checkpoint_retention.json`。保留器固定原子保存的文件版本，并读取 checkpoint 内部 update 核实命名；不改变当前训练代码或优化器。

```bash
ls -lh checkpoints/20261023T104759Z/act0_milestones/
cat reports/20261023T104759Z/checkpoint_retention.json
```

普通 Critic 的 10,000 次更新目前在阶段结束时保存 `critic0.pt` 和 `baseline_training.json`，没有逐步进度 JSON。仿真评估时 CSV 每回合刷新：

```bash
cat reports/PROJECT_STATUS.json
tail -f results/20261023T104759Z/act0_validation.csv
tail -f results/20261023T104759Z/paired_screening.csv
tail -f results/20261023T104759Z/paired_confirmation.csv
tail -f results/20261023T104759Z/short_B3_validation.csv
cat reports/20261023T104759Z/decision.json
```

文件只在相应阶段启动后出现。阶段实际命令、退出码和源文件 hash 在 `reports/20261023T104759Z/*command.json` 或 `*_command.json`，仿真输出在同目录 `.log`，各类别的真实物理步数在 `run_manifest.json` 与阶段 `.accounting.json`。查看这些文件时，无需改动任何协议或阈值。

## D0 停止后的普通直接 Q 对照

任务书允许保留普通直接 Q 路径。当前新增对照从原 ACT0、普通 Critic 和真实共享 replay 开始，完成 2,000 次 TD Critic、200 次 Q+BC+step ACT 更新；随后固定 ACT0 与更新后 ACT 各做 32 回合真实评估。配置在 `reports/20261023T104759Z/direct_q_plan.json`，完整结果只在两组评估完成后生成 `direct_q_result.json`。这不等于配对 D2 或 B1/B2/B3 通过；额外 BC 也使它不能单独归因 Q 的效果。

现有作业使用 tmux 会话 `forcecritic-direct-q`。查看进度无需重新启动：

```bash
watch -n 5 '.venv/bin/python scripts/training_status.py'
tail -f .cache/direct_q_background.log
tail -f results/20261023T104759Z/direct_Q_validation.csv
cat reports/20261023T104759Z/direct_q_job.json
```

`direct_Q_validation.csv` 在第二组初始化完成后出现；每完成一回合刷新。新评估使用 21000–21031，原 ACT0 20000–20031 的结果保留。每组独立仿真进程，逐物理步日志保证进度不是监控心跳。初始化最多 300 秒；连续 180 秒无物理进展抓栈并中断，最多 30 秒收尾，错误会写盘并发布，绝不自动改门槛或无限重跑。

后台入口为 `PYTHONPATH=src .venv/bin/python scripts/background_direct_q.py`；已有真实评估产物时会拒绝覆盖，不要重复启动。结束后后台作业自动更新最终报告、推送当前分支并更新 PR #1。原始图像、数据和权重仍在本地。
# 当前 v3 分阶段执行

```bash
cd /media/midea/F496AC8096AC4548/closed_loop_act
watch -n 5 .venv/bin/python scripts/training_status.py
tail -f .cache/stage_v3_background.log
tail -f reports/stage_v3_20261026/eval_A1_initial.log
```

第一阶段补 A1（BC＋同样步长约束），复用 A2（Q＋BC＋步长）以及 ACT0；不重新预训练。A1 初始 32 回合后按锁定规则判断，最多追加 A1/A2 各 32 回合，不增加训练。第一阶段未 GO 时，第二至第五阶段保持 NOT_RUN。阶段报告位于 `reports/stage_v3_20261026/FINAL_REPORT.md`。

管理进程启动命令（已有 `forcecritic-stage-v3` 会话时不要重复启动）：

```bash
tmux new-session -d -s forcecritic-stage-v3 'env PYTHONPATH=src .venv/bin/python -u scripts/background_stage_v3.py > .cache/stage_v3_background.log 2>&1'
```

管理程序记录每次仿真的命令、返回码和物理步；重复启动只接纳已完成且哈希一致的批次，失败批次须先审计，不覆盖。A1 训练的早期 detached 启动未保留操作系统退出码，执行清单如实标记为空，以已完成训练产物、日志和 checkpoint SHA 验证，不伪造退出码。

本文其余内容为历史训练与诊断命令；新执行顺序见 `PROTOCOL_STAGE_V3.md`。
