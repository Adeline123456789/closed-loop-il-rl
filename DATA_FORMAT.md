# 本地 FORGE 示教格式

这些是当前 Isaac Lab FORGE 中重新采集的专家示教，不是旧 HF 数据的格式转换。完整数据在 `data/act_demonstrations/`。20 个采集批次共 376 个回合，367 个稳定成功；train 256/251、validation 56/55、test 64/61（总数/成功数）。按完整采集批次分区，开发校准回合另存且不混入。

`.npy` 是 NumPy 数组文件：文件头保存维度、数值类型和排列方式，后面保存数组本身。它不只是文件名或图片路径。`rgb.npy` 本身就包含按时间排列的全部 RGB 图像。使用无损、未视频压缩的数组方便逐帧与力、动作对应；不需要同时存一份 MP4 才算有视觉数据。

下面维度实测自 `00_train_1100/train-1100-000000/`。`T=149`，15 Hz，对应约 9.93 秒。其他回合以其文件实际维度为准。除 RGB 为 uint8、三个标志为 bool，其余都是 float32。

| 文件 | 实测维度 | 内容 |
|---|---|---|
| `rgb.npy` | `(149,256,256,3)` | 149 张单固定相机 RGB 图像，0–255；`rgb[t]` 就是一张图片 |
| `state.npy` | `(149,8)` | 动作前的观测：末端位置 xyz（米）、末端四元数 wxyz、两夹指位置之和（夹爪开度，米）；含 stock 观测噪声 |
| `wrench.npy` | `(149,6)` | 动作前的因果平滑六维载荷 `[Fx,Fy,Fz,Tx,Ty,Tz]`，前三项 N，后三项 N·m |
| `motion.npy` | `(149,6)` | 动作前的末端线速度与角速度（有限差分），m/s、rad/s |
| `previous_command.npy` | `(149,4)` | 动作前已知的上一条原生四维指令 |
| `action.npy` | `(149,4)` | 当前提交、EMA 前的归一化 `[x,y,z,yaw]` 指令；不是关节角，也不是直接以 m/s 为单位的速度 |
| `ema_action.npy` | `(149,7)` | stock 七维布局的 EMA 后动作；含受约束 roll/pitch 和第七维占位；第七维不是夹爪控制 |
| `target_position.npy` | `(149,3)` | 当前动作实际下发的末端位置目标（米） |
| `target_quaternion.npy` | `(149,4)` | 当前动作实际下发的末端姿态目标（wxyz） |
| `raw_wrench.npy` | `(149,6)` | 当前动作执行后最后物理子步的未平滑六维载荷 |
| `substep_wrench.npy` | `(149,9,6)` | 前一控制边界值加 8 个物理子步的载荷；物理频率 120 Hz，跨步统计不能重复算边界 |
| `observed_after_state.npy` | `(149,8)` | 当前动作后的状态，在自动 reset 前捕获；用于审计，不可当作动作前输入 |
| `stable_success.npy` | `(149,)` | 独立几何成功连续三控制步的判据 |
| `terminated.npy` / `truncated.npy` | 各 `(149,)` | 当前转移的终止/截断标志 |

**当前没有保存七个机械臂关节角/关节速度序列。** `state` 是末端姿态和夹爪开度，不是完整关节本体状态。它满足当前协议的八维 state 输入；若以后需要关节空间学习或完整重放，应增加明确命名的关节数组并升级数据契约，不能从现有八维 state 猜测补全。仿真诊断使用的完整快照是另一种产物，不能把这些示教数组称为完整 PhysX 快照。

载荷按已经审计的 incoming-joint 参考系/锚点保存，不应凭 stock 变量名中的 `world` 就当世界坐标系载荷。契约证据见 `reports/20261023T104759Z/sensor_contract.md`、`contract_probe_02/contract_audit.json` 与 `p0_acceptance.json`。只有动作前的 RGB/state/wrench/motion/previous_command 进入相应观测；动作后的字段单独用于审计。训练取最近四步历史，以固定四步动作块作 BC 目标。

`manifest.json` 保存回合分区、成功状态、每个数组的相对路径/SHA256，以及相机、控制 dt、动作阶段、专家权重版本与 P0 证据 hash。完整原始回合保留；正向 BC 排除失败回合，并截到首次稳定成功，避免把后续保持/退出动作作为继续插入目标。

## 直接查看或转成视频

在项目根目录运行：

```bash
.venv/bin/python scripts/inspect_recording.py \
  data/act_demonstrations/00_train_1100/train-1100-000000 \
  --output data/preview/train-1100-000000
```

打印所有维度，并导出 `rgb.mp4` 和第 0、74、148 帧 PNG。MP4 只是便于观看的有损副本；训练仍读取原始数组。

```python
import numpy as np
from PIL import Image
from pathlib import Path
p = Path('data/act_demonstrations/00_train_1100/train-1100-000000')
rgb = np.load(p / 'rgb.npy', mmap_mode='r', allow_pickle=False)
wrench = np.load(p / 'wrench.npy', allow_pickle=False)
state = np.load(p / 'state.npy', allow_pickle=False)
Image.fromarray(rgb[74]).save('frame_74.png')
print(rgb.shape, wrench[74], state[74])
```

单回合 RGB 约 29.3 MB 是 149 张未压缩图片的正常大小。完整原始数据与视频保留在本机；GitHub 上传契约、清单 hash、结果 CSV、日志及少量审计图。

## 在 FORGE 中重建原采集轨迹

2026-10-27 已验证 seed 1100、1101 各自的第一个 8 环境批次，共 16 回合、2,384 帧：所有非图像字段逐值完全一致，包括力的每个物理子步、EMA、动作目标和成功标记。重渲染 RGB 与原记录仍有差异，运行中恢复快照也未通过原容差。完整证据见 [根因报告](reports/replay_root_cause_20261027/FINAL_REPORT.md)。

根因之一是采集时的 `Expert.action()`：rl-games 即使最终选择确定性的均值动作，也先采样并推进 CUDA 随机数状态。直接读 `action.npy` 而省掉这个调用，会改变 FORGE 后续观测噪声和随机化。修复脚本保持原初始化顺序、seed、8 环境批量、专家及 RNN 调用顺序，实际下发的动作仍来自文件。

回放需要原 `.venv` 和专家权重。在项目根目录查看本次已完成的真实调用命令和结果（不消耗仿真步）：

```bash
cat reports/replay_root_cause_20261027/06_dataset_original.command.json
cat reports/replay_root_cause_20261027/restore_components_summary.json
```

已完成的完整 149 步证据位于 `reports/replay_root_cause_20261027/06_dataset_original/result.json` 和 `08_dataset_seed1101/result.json`。每次完整批次含 reset 共消耗 1,326 个环境控制步当量。工程预算总上限 5,000；包括后续定位实验已计费或保留 4,906.875，完整复跑需要新的有界预算记录，不能自动绕过上限。

后续修复了快照复制改变张量内存布局、重复写回未变化物理状态两个实现问题，原地保存/恢复已实现物理零差异。真正回退到旧接触快照仍失败，具体证据见 [恢复根因报告](reports/replay_root_cause_20261027/RESTORE_ROOT_CAUSE.md)。此问题发生在回放恢复流程，原始采集没有使用此快照代码。不要据此删除原力数据，也不能据此宣称原数据已通过真实传感器标定。

当前脚本仅接受原训练采集 manifest 的第一个同步批次。后续批次需要执行之前的完整采集前缀；本次未实现或验证。不能把单个任意回合改名后从 seed 直接启动。`MEASURED` 只说明测量完成，是否一致必须查看每个字段的误差；`all_nonvisual_fields_bitwise_equal` 按有限数值的零最大绝对误差判定，不是文件字节比较。`--omit-expert-rng` 仅用于复现失败的负对照。
