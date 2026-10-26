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

## 推荐：前台顺序继续

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
tail -f .cache/continue_training.log
watch -n 10 '.venv/bin/python -c "import json,pathlib; p=pathlib.Path(\"checkpoints/20261023T104759Z/act0.json\"); d=json.loads(p.read_text()) if p.exists() else {}; print({k:d.get(k) for k in (\"status\",\"completed_updates\",\"requested_updates\")})"'
```

`act0.json` 每 100 次更新保存进度；模型与优化器在 `act0.pt`。普通 Critic 的 10,000 次更新目前在阶段结束时保存 `critic0.pt` 和 `baseline_training.json`，没有逐步进度 JSON。仿真评估时 CSV 每回合刷新：

```bash
cat reports/PROJECT_STATUS.json
tail -f results/20261023T104759Z/act0_validation.csv
tail -f results/20261023T104759Z/paired_screening.csv
tail -f results/20261023T104759Z/paired_confirmation.csv
tail -f results/20261023T104759Z/short_B3_validation.csv
cat reports/20261023T104759Z/decision.json
```

文件只在相应阶段启动后出现。阶段实际命令、退出码和源文件 hash 在 `reports/20261023T104759Z/*command.json` 或 `*_command.json`，仿真输出在同目录 `.log`，各类别的真实物理步数在 `run_manifest.json` 与阶段 `.accounting.json`。查看这些文件时，无需改动任何协议或阈值。
