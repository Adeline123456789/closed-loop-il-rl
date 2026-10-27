# FORGE 数据回放根因与修复，2026-10-27

本报告保留第一次调查截至 `ad1503d` 的结果；后续发现的张量布局错误、状态写回副作用及修复验收见 [恢复根因报告](RESTORE_ROOT_CAUSE.md)。最新累计预算见 `restore_components_summary.json`，不覆盖以下历史计数。

结论：现有数据可以用于离线读取/训练，已修复并实测通过原采集轨迹的非视觉重建。不能据此宣称任意时间点快照分支或视觉闭环回放已经解决。历史 D0 的 NO_GO 保留。

## 已定位并修复的根因

原采集器的顺序是：建立原宽度 FORGE 环境 → 初始化原专家 → 按原 seed reset → 每步调用专家 → 下发四维 EMA 前动作 → 更新 RNN done 状态。专家和 FORGE 共用 CUDA 随机数流。

本机 pinned rl-games 的 `algos_torch/models.py` 推理路径先执行 `distr.sample()`，`algos_torch/players.py` 再按 `is_deterministic` 选择均值。动作虽为确定性均值，调用仍推进 CUDA RNG。原回放若只提交动作文件、不执行原专家调用，就改变后续噪声及随机化。seed 相同不足以保证随机数调用序列相同。

`scripts/replay_demonstrations.py` 重建原初始化/批量/专家调用顺序，同时实际提交原文件动作；不注入原图像或原力来制造匹配。逐个验证数组 SHA256，使用原 stock 摄像头和物理参数。专家权重也由现有 `Expert` 的固定 SHA256 检查。`src/forcecritic/data/reconstruction.py` 拒绝单独回合、错序、后续批次伪装成首批、混合采集批次及 held-out 分区。

## 真实 FORGE 结果

| 实验 | 范围 | 结果 |
|---|---|---|
| `06_dataset_original` | seed 1100，8 回合 × 149 帧 | 所有非视觉字段最大绝对误差 0；RGB 最大差 38/255 |
| `08_dataset_seed1101` | seed 1101，8 回合 × 149 帧 | 所有非视觉字段最大绝对误差 0；RGB 最大差 33/255；确认专家调用推进 CUDA RNG |
| `09_dataset_without_rng_fixed` | 同一 seed 1100、原动作、原 8 环境，省略专家 RNG 调用，64 帧 | 第 0 步动作后观测开始不同，第 30 步原始载荷开始不同，第 36 步成功标记不同（均为零基步号） |

两次完整批次共 **16 回合、2,384 帧**。比较包含状态、历史动作、专家重算动作、平滑/原始/逐子步载荷、速度、EMA 后动作、目标位姿、动作后观测和全部终止/成功标记。原数组 `action` 用作输入；另外独立比对专家重新产生的动作。误差为 0 指逐值完全相等，非重新编码文件的字节相等。结论仅覆盖这两个训练首批，不能外推为全部 376 回合均已仿真重播。

负对照中 `observed_after_state` 首步最大差 0.00119057；64 步范围内 state 最大数值差 0.0482550、raw_wrench 最大数值差 19.6881、stable_success 最大差 1。这些向量含不同物理量，不能把跨维最大数值差统称为米或牛顿。负对照的 previous_command 和 ema_action 仍完全一致，排除了把提交动作改错作为此处差异的解释。

## 另一个独立问题：运行中快照恢复仍失败

`02_legacy` 与 `03_skip_properties` 的独立新进程 original 轨迹，在全部 16 个控制步/128 个物理子步的记录上完全一致；这表明这些条件下的冷启动物理轨迹可以重现。

同进程恢复快照则失败：恢复后 incoming-joint force 已不同，第一次物理推进后关节位置出现约 0.00011188 rad 差异。去掉对未改变质量/惯量/材质的重复写入仍未解决。恢复公开位姿、速度、控制目标、缓存张量和 RNG 不能证明恢复了 PhysX 内部所有状态。未直接观测到的 solver/contact 内部状态只是剩余候选，不能宣称已经证明唯一底层原因。

因此本次修复选择重新建立原采集进程来重建已有轨迹；**没有把 `restore()` 标记为可用，也没有继续配对反事实分支训练**。快照恢复为 FAILED；其更深底层原因仍 INCONCLUSIVE。历史 native crash 未封账尾部仍保留，不因新实验而清零。

## 图像确定性仍未解决

冷进程相同物理轨迹最后观测的 RGB 仍相差 1–2 个灰度值。原数据与新渲染的全轨迹最大差分别为 38、33。没有用原图替换在线图，也没有修改历史像素/动作容差。

对照 `04/05` 与 `10/11` 尝试新版上游建议的 RTPT/caches 设置：在已安装 Sim 5.0 中，实际 `/rtx/rendermode` 始终读回 `RaytracedLighting`，包括环境创建后再设置。新版配方没有在本运行时实现预期渲染模式；不能把它算作修复。配置保留为实验记录，未加入默认数据回放。

视觉闭环精确重放：FAILED/未获准；新的 ACT 闭环改进与科学 D0 重新验收：NOT_RUN。记录中的 RGB 本身可正常读取用于离线训练或视频查看。

## 预算、失败尝试与软件验证

独立工程总上限 5,000 个环境控制步当量；最终已计费或保留 **4,275.5**。每次初始化、reset、prefix、branch、interaction 均由 before-step journal 记录，8 个物理子步折合 1 控制步，8 环境逐个计费。详细逐运行数值见 `summary.json` 与 `*.command.json`。

`01_legacy` 是脚本错误：旧记录没有 candidate_action，未获得 journal，保守保留整个 500 步预留。`07_dataset_without_rng` 是负对照脚本的未初始化 RNN mask 错误，保留失败记录并计费 75 步；修正后由 `09` 完成实测。Isaac launcher 在异常时可能返回进程码 0，supervisor 现同时检查 `result.json.status == MEASURED`，不把退出码 0 当成功。MEASURED 也不等于 fidelity PASSED。

完整测试：`PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q`，**78 passed in 8.73s**。这些软件测试不替代以上真实模拟器结果。原始 PhysX 张量审计文件留在被忽略的 checkpoints 中，SHA256/大小列于 summary；Git 只保存代码、配置、数值误差、journal 与日志。

使用入口和现有预算内的复核命令见根目录 `DATA_FORMAT.md`。依赖保持本机原版本：Isaac Sim 5.0、Isaac Lab v2.2.1（`0f00ca2b4b2d54d5f90006a92abb1b00a72b2f20`）、Torch 2.7、RTX 3090；未升级驱动、物理引擎或修改数据。

## 文档证据及适用范围

本次使用 web-debug-search 与 run-experiment 工作流。网络资料只作调试线索；版本适配与有效性以本机源码、实际 settings 读回和真实运行结果为准，未经实测的建议不能视为修复。

- [Isaac Lab v2.2.1 reproducibility 源文档](https://raw.githubusercontent.com/isaac-sim/IsaacLab/v2.2.1/docs/source/features/reproducibility.rst)：说明相同设置下的可复现性及运行时修改物理属性的限制；不证明任意公开状态快照足够。
- [上游 RTX 确定性设置源码](https://raw.githubusercontent.com/isaac-sim/IsaacLab/develop/source/isaaclab_physx/isaaclab_physx/renderers/isaac_rtx_renderer_utils.py)：新版 RTPT/caches 配方，仅作为兼容性实验来源，本机没有成功切换目标模式。
- 根因直接证据是本机 rl-games 调用路径与 `06/08/09` 的真实成对对照，未依赖社区猜测。
