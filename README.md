# 力觉 Critic → ACT：最小可行性验证

**协议：`force_critic_act_v2_diagnostic_first`。当前只有任务书与配置；算法实现、真实 Isaac Lab 训练和实验成绩均为 NOT_RUN。**

本协议已覆盖旧的三组件 residual/distillation MVP。旧内容只保留在 Git 历史中，不再执行。

## 现在验证什么？

示教初始化 ACT → ACT 输出完整动作块 → FORGE 接触交互与六维力 → 收益/接触代价 Critic → 对 ACT 动作求导 → 直接更新 ACT → 新交互。

只有 ACT-CVAE 与双收益/代价 Critic 两组可训练组件。不实现残差教师、成功轨迹蒸馏、双可恢复性头、prior-gap 权重或额外方向预测网络。

完整规格：[CODEX_TASK.md](CODEX_TASK.md)。实现约束：[AGENTS.md](AGENTS.md)。启动提示：[START_CODEX.txt](START_CODEX.txt)。预算草案：[configs/minimal_validation.yaml](configs/minimal_validation.yaml)。状态：[reports/PROJECT_STATUS.json](reports/PROJECT_STATUS.json)。

## 先诊断，后训练

1. **P0 环境/接口验收**：HF schema、stock FORGE、原生动作、六维力帧、EMA、相机与 reset。1 个真实仿真环境通过后再并行。
2. **P1 基础策略**：仅 PegInsert 成功示教预训练 ACT，新增仿真交互训练普通 Critic。失败集用于审计/诊断，不把所有失败片段标为不可恢复。
3. **P2 最小诊断**：先 8 个检查点验证原始连续执行/快照恢复/前缀重放；通过后 24 个独立检查点检验 A0、A+、A- 的有限效果；再检查 Critic 梯度能否变成 ACT 的实际改善。
4. **P3 三组同数据控制**：Direct-value、Direct-filter、Calibrated-filter；ACT0 仅作评估参照。先短步更新验证，出现可信信号才运行完整两轮闭环。
5. **P4 扩展**：先 1 seed/100 测试回合；支持继续才 3 seeds。冻结模型测初始位姿、摩擦和力噪声/延迟，不先上多个任务。

| 组别 | Actor 的 Q 更新状态 | Critic 监督 |
|---|---|---|
| B1 Direct-value | 所有合法训练状态 | 同一批 TD 与端点回报 |
| B2 Direct-filter | 训练配对确认的可信可改善状态 | 与 B1 相同 |
| B3 Calibrated-filter | 与 B2 相同 | B2 + 有限尺度配对价值差损失 |

B3-B2 检验校准，B2-B1 检验筛选。所有组都拿到同样配对数据，不能把多采样收益当校准贡献。失败接触处的方向错误是待检验假设，不是既定事实。

## 两个数据集与力

- https://huggingface.co/datasets/angledusgar/forge-v1
- https://huggingface.co/datasets/angledusgar/forge-failure-v1
- 发布者 stock 专家：https://huggingface.co/angledusgar/forge-peginsert-v1

数据卡的 ft 是已平滑六维 wrench，ft_noisy 只有三维力，记录为15Hz。它们不能代替新仿真的接触快照、分支回放或在线RL数据。

没有真机可以验证仿真可行性，但需要能运行所选 Isaac Sim 和相机的 GPU。首轮仅 stock x/y/z/yaw 控制，完整六维力观测；原生第7维是成功预测，不是夹爪。具体以锁定源码和采集语义审计为准。

## 开源起点

- Isaac Lab / FORGE：https://github.com/isaac-sim/IsaacLab/tree/v2.2.1/source/isaaclab_tasks/isaaclab_tasks/direct/forge
- ACT 原始实现：https://github.com/tonyzhaozh/act
- ACT 工程参考：https://github.com/huggingface/lerobot
- 直接 Q actor 更新参考：https://github.com/sfujim/TD3_BC
- AC3 文献：https://arxiv.org/html/2508.11143v2 。作者仓库 https://github.com/flyfaerss/ac3 的 README 当前仅一行，不假定有可运行完整代码。

版本起点可用 Isaac Lab v2.2.1 / Isaac Sim 5.0.0 / Python 3.11，按对应文档审计安装。它不是已经确认的数据采集栈。保留来源和许可证，不混装所有旧依赖。

## 启动 Codex

在已有 GPU / Isaac 环境的机器克隆本仓库，运行 Codex，然后把 START_CODEX.txt 发给它。使用新分支 `codex/force-critic-act-v2`。CODEX_TASK 中列出的项目 CLI 尚待 Codex 实现，不能现在就声称命令可运行。

真正的验收测试包括：固定数据与 ACT、改变 Critic 的动作梯度，ACT 参数更新应改变；Critic 自身参数不更新，但动作路径必须可微。仅重新做 BC 不算完成闭环。

## 结果纪律

这是一项可失败的验证，不预设正结果。回放不可信或动作效果无法识别就停止校准；普通直接更新已经足够好时优先保留简单方案。所有图表由真实 CSV/JSON 生成；保留失败/负结果和所有交互成本。无 GPU、资产或接口缺失写 BLOCKED，不用 mock 和离线误差冒充插接成绩。

代码、配置、小型原始结果和报告上传 GitHub；不上传密钥、HF原始数据、Isaac缓存和巨大 replay。本次文档更新不表示 Codex 或GPU训练已在后台启动。
