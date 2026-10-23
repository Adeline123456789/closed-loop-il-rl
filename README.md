# PRD-ACT-SAC：三组件 MVP 最小验证

**当前状态：已提交实现任务书和执行规范；算法代码与 Isaac Lab 训练尚未运行，没有实验成绩。**

本项目冻结在 `prd_mvp_v1_prior_gap`：ACT-CVAE、模式条件 SAC（双奖励 Q）、双可恢复性头，以及不训练的幂等方向投影。不采用后续反事实力觉 KL、教师条件互信息、刚度辨识或额外网络。

完整规范见 [CODEX_TASK.md](CODEX_TASK.md)。Codex 开始前先读 [AGENTS.md](AGENTS.md)。启动提示见 [START_CODEX.txt](START_CODEX.txt)。

## 最小验证路线

1. 审计硬件、数据、动作、六维力坐标、控制频率、EMA 和 reset，先启动真正的 `Isaac-Forge-PegInsert-Direct-v0`。
2. 用成功集 PegInsert 子集预训练 ACT；失败集只作失败数据审计/诊断，不能把失败轨迹全部当作不可恢复标签。
3. 冻结 ACT，训练模式条件 SAC 与双可恢复性头，得到一个共同教师。
4. 从完全相同的教师轨迹和 ACT 起点分叉：不蒸馏、全量蒸馏、prior-gap 选择性蒸馏。
5. 评估原始/全量/选择性三个 ACT-only 策略，再给三条完整分支相同的 residual 适配预算，完成一轮真正双向闭环。
6. 一个训练种子、100 个测试回合先筛查；有可信信号后才扩展到三个训练种子和冻结模型的分布外评估。

最初只做 PegInsert，不同时训练 GearMesh/NutThread。判定依据是实际稳定插接成功率、接触载荷、恢复结果和 residual 指标，不是离线重构损失或总 reward 单项。

## 环境审计的额外省时检查

数据发布者同时提供 stock PegInsert 的 PPO 专家 checkpoint：
https://huggingface.co/angledusgar/forge-peginsert-v1

环境审计时先核对其 `rl_games` 版本、观测归一化和配置兼容性；兼容后通过官方 player 做一个小规模 rollout，检查 stock 任务是否正常。这个 PPO 仅用于环境验收或必要时补采可复现示教，不是本算法新增的第四个组件，也不替代 SAC 教师。

模型卡特别提示：该 checkpoint 对应 stock PegInsert，不对应发布者另行修改的更困难插接配置。不要因困难变体失败就判定环境或算法无效。也不要将模型卡报告的成功率当作本项目实测结果。

## 必须先解决的接口

候选 Isaac Lab v2.2.1 FORGE 的 7 维动作不是“6 维速度 + 夹爪”：第 7 维是成功预测，roll/pitch 策略动作被约束。首轮保持 stock 的 x/y/z/yaw 可控空间，明确六维力观测不等于六自由度控制。成功判据使用环境几何条件，不使用动作第 7 维。

数据卡的 `ft` 为 6 维；`ft_noisy` 是 3 维。实际字段、参考坐标、平滑、延迟及记录动作位于 EMA 前后必须读数据和源码核验，不能补零冒充六维测量。

无真机可以做仿真可行性实验；仍需能运行所选 Isaac Sim、资产和相机渲染的 GPU 环境。没有该环境就记录 BLOCKED，不用 mock、离线 MSE 或 CPU 单测替代机器人实验。

## 结果与提交约定

所有实验表从真实 `episode_metrics.csv` 生成，并附配置、commit、依赖版本、种子、数据 revision、真实执行命令和退出状态。未执行写 NOT_RUN，环境缺失写 BLOCKED；保留失败和负结果。

GitHub 提交代码、配置、审计报告和真实小型结果文件；不上传密钥、HF 原始数据、Isaac 缓存或巨大 replay。权重用经授权的 Release/存储并记录校验值。完整要求见 CODEX_TASK.md。

本仓库的任务文件不是已经完成的代码实现，也不表示任何 GPU 训练已经在后台启动。
