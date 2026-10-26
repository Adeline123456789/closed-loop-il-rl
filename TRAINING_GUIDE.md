# 本机数据与后续训练

项目目录：`/media/midea/F496AC8096AC4548/closed_loop_act`。分支：`codex/force-critic-act-v2`。运行 ID：`20261023T104759Z`。

本次发布快照见 `reports/20261023T104759Z/publication_snapshot/status.json`。真实 smoke-1、smoke-8 和 24 步动作/六维力探针已通过；新契约的开发专家实测为 8/8 稳定成功。ACT、在线 Critic、Critic→ACT 和 B1/B2/B3 真实结果尚未产生。代码已接入真实环境，但这些后续阶段仍待实际执行验证。软件测试不代表插接成功率。

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

推荐直接查看实时面板，显示 ACT 更新数/损失、真实评估回合数、后续诊断与消融队列、进程存活及上传状态：

```bash
cd /media/midea/F496AC8096AC4548/closed_loop_act
watch -n 5 '.venv/bin/python scripts/training_status.py'
```

后台监控每 30 秒向 `.cache/background_training.log` 追加 `LIVE` 心跳，完整状态写入 `reports/20261023T104759Z/training_progress.json`，从真实 checkpoint 中导出损失到 `results/20261023T104759Z/act_pretraining_loss.csv`。它不改训练参数、不重启优化器。`background_job.json` 的时间是进入阶段的时间，不是训练更新的时间；`run_manifest.json` 的阶段结果通常在阶段结束后写入，不能将其中旧的 `NOT_RUN` 当作当前没有进程。

队列为 baseline（ACT 10000 更新 → 32 回合评估 → 共享真实交互 → 普通 Critic 最多 10000 更新）、diagnostic（保真度/配对/真实 Critic→ACT）、short-ablation（B1/B2/B3 各 2000 Critic + 200 Actor 更新，B0–B3 各 32 回合真实评估）。门槛未通过就停止并显示原因。普通 Critic 循环尚无逐步遥测时面板会明确说明，不伪造更新进度。pilot/confirm/generalization 不在自动队列中。

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
