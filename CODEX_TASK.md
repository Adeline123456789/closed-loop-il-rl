# Codex 执行任务书：PRD-ACT-SAC 三组件 MVP 的最小可行性验证

状态：设计与实现任务书；尚未实现算法、下载核验数据字节、运行 Isaac Sim 或产生实验成绩。
研究目标：先以最少实验判断核心机制是否值得继续，再扩展泛化；允许负结果，不以制造显著提升为目标。
目标仓库：https://github.com/Adeline123456789/closed-loop-il-rl
版本标识：prd_mvp_v1_prior_gap。此标识是本项目协议，不是既有开源软件版本。

## 0. 不可变范围

只允许三组可训练组件：
1. ACT-CVAE：块级基础策略；先验、后验、图像/历史编码器均属于这一组件。
2. 模式条件 SAC：一个接收 F/R 模式的 residual actor、双奖励 Q 及其 target copies。
3. 轻量双可恢复性头：共享轻量特征，输出 Q_g^F、Q_g^R；target copies 不是新算法组件。

确定性组件：动作适配器、幂等方向投影、限幅、固定恢复基础控制律、有限状态监督器。
禁止新增 f_chi、刚度矩阵在线辨识、QP、世界模型、顶层 Q_Omega、教师条件边缘化网络、反事实力觉 KL、条件互信息估计模块。
保留的步级权重是冻结 SAC 教师与冻结 ACT 先验的分布差距。它是“先验可实现性代理”，混合了欠拟合与反馈差异，不能称为严格可预测性或因果信息增益。

不得为跑通而退化成 ACT 每步重规划、冻结 ACT 永不蒸馏、仅训练成功分类器、只计算离线 MSE、用玩具环境冒充 FORGE、或者用三维力补零冒充六维测量。
不得承诺算法有效；代码正确性、训练收敛、仿真性能、真机有效性分别报告。

## 1. 已核对来源；先锁版本，不混装所有项目

环境与安装：
- Isaac Lab: https://github.com/isaac-sim/IsaacLab
- 已检查存在 FORGE 源码的候选 tag：v2.2.1
- FORGE 目录：https://github.com/isaac-sim/IsaacLab/tree/v2.2.1/source/isaaclab_tasks/isaaclab_tasks/direct/forge
- 动作和力觉：https://github.com/isaac-sim/IsaacLab/blob/v2.2.1/source/isaaclab_tasks/isaaclab_tasks/direct/forge/forge_env.py
- 任务配置：https://github.com/isaac-sim/IsaacLab/blob/v2.2.1/source/isaaclab_tasks/isaaclab_tasks/direct/forge/forge_env_cfg.py
- 安装：https://isaac-sim.github.io/IsaacLab/v2.2.1/source/setup/installation/pip_installation.html
- FORGE 项目：https://noseworm.github.io/forge/

候选复现栈：Isaac Lab v2.2.1 + Isaac Sim 5.0.0 + Python 3.11，按上述对应版本文档安装；这是已确认兼容文档的起点，不是已证明等于数据采集栈。优先采用能重现数据语义的可用版本并锁实际 commit。不要无条件使用当前 main。已有其他可工作的栈不得擅自破坏；在独立环境验证。

算法代码：
- ACT 原始实现：https://github.com/tonyzhaozh/act
- 推荐复用的 ACT 实现：https://github.com/huggingface/lerobot
- ACT 文件：https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/modeling_act.py
- ACT 文档：https://huggingface.co/docs/lerobot/en/act
- 简洁 SAC 参考：https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
- ResFiT 官方 off-policy residual 参考：https://github.com/amazon-far/residual-offpolicy-rl
- ResiP：https://github.com/ankile/robust-rearrangement
- ResiP 项目：https://residual-assembly.github.io/

建议只实现一个一致的 PyTorch 训练循环，借用 ACT 与 SAC 必要代码，保留许可证/来源。ResFiT 用于核对 residual replay 和冻结基座接口；ResiP 是 PPO/其他任务的设计参考，不把其环境或整套依赖装进来。不要把四个仓库硬拼为主工程。

数据：
- 成功：https://huggingface.co/datasets/angledusgar/forge-v1
- 失败：https://huggingface.co/datasets/angledusgar/forge-failure-v1
- Codex 使用方式参考：https://developers.openai.com/codex/cli

记录每个依赖 SHA/tag、HF revision、Isaac Sim/Python/Torch/CUDA/驱动和资产版本到 reports/environment.lock.json。网址中的 main 只供发现文件，实验必须锁实际 SHA。核对第三方许可证；不要无授权更改其许可证。

## 2. 两个 HF 数据集的用途与边界

数据卡说明二者来自 Isaac Lab FORGE 的 PPO 专家仿真轨迹，不是真机采集。包含三个任务；首轮只过滤 Isaac-Forge-PegInsert-Direct-v0。
数据卡给出的主要字段：state[8]、actions[7]、ft[6]、ft_noisy[3]、image、wrist_image；ft 声称为平滑后的腕部六维 wrench，ft_noisy 只有带噪三维力；15 fps。实际读取 revision 后核对，不能只信卡片。
成功集卡片报告约 984 条 PegInsert 成功；失败集卡片报告三个任务共 51 条，当前 UI 又显示 53。最终数量、schema 与任务分布必须按实际 metadata/parquet 统计，不硬编码任意一方。
失败的标签来自 dataset provenance，不保证存在 success 列。失败轨迹不是逐帧“不可恢复”标签。数据未提供完整关节角、保证可用的仿真快照、恢复策略 rollout、奖励或每步完整终止事件。

用途：
- 成功 PegInsert：ACT 预训练、离线验证、数据适配。
- 失败 PegInsert：检查输入/标签/力曲线，做可选的行为策略结局预测 sanity check；不得将其动作无筛选加入正向 BC。
- 双可恢复性和 SAC：主要依赖新增的真实仿真交互转移。不得将失败集的所有状态标为 g_F=0 或 g_R=0，不得将文件结束一律伪造为危险终止，不在缺失奖励/状态语义时伪造 offline RL replay。
- Dataset loss 下降不能证明插接成功，最终性能必须来自在线 FORGE rollout。

划分：按 episode/source run 分组，优先用采集批次避免泄漏；至少 episode-level 70/15/15，全部相邻块留在同一 split。若 run id 不存在则说明不能完全排除批次相关性。成功/失败集 episode index 可能冲突，键用 (dataset_id, revision, task, episode_id)。只用训练集计算归一化量。
先读 meta/info、tasks、episodes 与少量 parquet，输出 reports/data_audit.json/md，再按 PegInsert 需要选择下载，不先拉全部三任务视频。v2.1 卡片与实际格式/新 LeRobot API 不一致时写兼容 reader，不静默升级原始数据。

## 3. 阶段 P0：环境/数据/控制契约审计（不算消融）

先写 scripts/doctor.py，记录 GPU、驱动、CUDA、Isaac Sim 启动、网络/资产、磁盘、数据读取、Git 认证。无需真机，但必须有能运行所选 Isaac Sim 和相机渲染的 GPU 环境。先 1 env，再 8 env；没有 GPU 时完成 CPU 单测/文档并明确 BLOCKED，不把 CPU 伪环境结果算 FORGE。不得自动租赁付费机器或修改系统驱动。

第一目标：stock Isaac-Forge-PegInsert-Direct-v0。数据卡专门警告采集 checkout 存在另一个困难插接变体，但采集时恢复 stock 参数。检查并记录：
hand_init_pos=[0,0,0.047]
hand_init_pos_noise=[0.02,0.02,0.01]
hand_init_orn_noise=[0,0,0.785]
held_asset_pos_noise=[0.003,0,0.003]
fixed_asset_init_tilt_deg=0（仅在相应字段确实存在时配置）
held_asset_tilt_deg=0（同上）
不要将不存在的自定义字段直接传给 stock config。先成功启动 stock，再在后续泛化阶段定义独立变体，不暗改同名任务。

### 3.1 动作契约：必须先实现/测试适配器

已检查的 v2.2.1 FORGE：actions 为 7 维；前 6 项映射到相对固定零件的位姿目标，不是六维末端速度；roll/pitch 两项被清零；第 7 项用于成功预测奖励，不是夹爪；夹爪通过底层固定闭合。
首轮保留 stock 可控维度 x/y/z/yaw，使用完整六维力反馈，不宣称“六自由度纠偏”。不在首轮解锁 roll/pitch。
- 查明 HF actions 记录的是提交给 env.step 的原始动作，还是 env 内 EMA 后动作。数据卡“executed”不足以辨认。查 source/metadata，可输出证据或未知。
- 查明位置单位、角度单位、yaw wrap、物体相对坐标、env_origin、EMA、限幅顺序。
- 统一 ACT 输出、SAC 残差、监督目标的控制空间。native normalized pose target 与物理 twist 不可直接相加。
- 在原生目标转换为运动误差的确定性适配层投影，并映射回底层目标。若使用速度形式，明确 u=local pose error/dt 是控制命令而非实测速度。
- 投影应在可控子空间构建，而不是先在 6D 投影再丢掉不可控轴。令 S 为缩放六维运动空间的正交可控基（本例 6x4），对 S^T w 与 S^T e 构造 n/t，输出 S G(u4)；报告可控空间的功率关系。
- ACT 蒸馏最终被接受的同类目标指令，不把测量速度当动作标签。原始提交动作、EMA 后目标、投影后目标、实际测量速度分别保存。
- EMA 仅作用一次。执行目标蒸馏到 ACT 后不能再多做一次 EMA。若需要统一“post-EMA target”接口，离线数据要可证明转换；不能证明时记录阻塞，补采同格式少量仿真示教，明确数据协议变更，禁止静默猜测。
- 第 7 维从运动损失、运动 Q 和残差动作中排除。adapter 写固定占位，并统一关闭原 FORGE success_pred_error 奖励；成功只用环境几何条件计算。不能用专家第 7 维做真值或让策略通过它自行宣告成功。

### 3.2 力觉/观测契约

数据卡称 ft 是 EE frame；所选源码仍须核对 reference point、旋转、力矩平移及正负号。不能仅重命名 world_smooth 就称 EE wrench。不同版本/collector 可能有变换差异。
报告 free-space 基线、已知方向轻接触响应、卸载响应、ft/ft_noisy shape、平滑系数与相位延迟。不得用 |f| 标量复制成六维，不用 noisy force 三维补零力矩。
归一化保留物理单位；用于功率的 w=[f;tau/Lc]、V=[v;Lc*omega] 以同一参考点/坐标系计算。力与力矩阈值分别以 N / N m 报告。Lc 来自实际资产尺度并记录。
真实仿真完整接触/零件位姿可用于奖励、成功判据及离线诊断，不能偷塞给只在部署中拿不到这些信息的 actor。

首轮观测推荐一张固定 RGB + state + 过去 H=4 步 ft/运动/动作历史；不要双相机或大视觉模型。若数据中缺少可用零件相对位姿，不能为了省视觉而从仿真真值补给 state-only actor。state-only 仅作审计，除非该非特权观测确实能完成任务并被所有组共享。
匹配相机位姿/内参/裁剪。若 HF 相机参数缺失，显式记录域差异；必要时先补采可复现 sim demos，不把视觉/坐标不匹配解释为算法失败。

### 3.3 时间与成功判据

保留 stock 控制节奏并从 cfg 实测 physics_dt、decimation；数据为 15 Hz。第一版 SAC 每环境决策步更新动作，ACT K=M=8，约 1.875 Hz；这是相对快慢闭环，不是 100/1000Hz 实机力控。不插值 15Hz 数据后宣称取得新的高频接触观测。
K=M；禁止 temporal ensemble、跨模式/跨 reset/chunk 混合。记录 phase 与 remaining chunk 供 SAC/Q/g 访问。ACT 块初之后不再读取新图像/力生成该块后续动作；SAC 可以读新反馈。
成功固定为官方几何插入条件并持续 3 个控制步（若官方已有更严格保持，沿用），记录首次成功时间；绝不以 Q 值、reward 阈值或动作第7维判断。额外分别记录 ever_success 和 final_success，主指标为声明的稳定完成。所有组用同一成功/超限/回合预算定义。
审查 Factory/FORGE reset 是否会重置整个并行批次；保留 terminal_observation，禁止 bootstrap 到自动 reset 后的初始状态，禁止 chunk 跨越 reset。批次同步 reset 条件下用 active mask/填充直到批次结束，避免其他 env 无声丢样本。

P0 退出条件：真实 headless FORGE 启动并完成 1->8 env smoke；数据 schema/任务过滤验证；动作、力帧、EMA 和 reset 测试通过。未通过不开始大训练。

## 4. 固定算法，实现后不继续加理论补丁

### 4.1 ACT 与选择性分布蒸馏

先以成功训练片段预训练 ACT。CVAE 后验训练、先验推理；部署不能读取目标块进入后验。先验可为标准 N(0,I)。以整块潜变量采样保持块内模态一致，不逐步平均多模态样本；用于评分必须能够从先验采样，不能一直 z=0 后声称估计了动作分布。部署模式在所有组一致。若存在显式动作分支 c，必须块初选定，不得事后按成功动作贴标签；首轮不增加独立模态分类器。

按旧版定义：
T_jl = frozen teacher 的最终可执行动作分布（在记录的实际状态处采样 SAC 并经过相同 adapter/投影）；
P_jl^- = frozen ACT 在块初 H_j 的先验动作边缘（映射到相同监督动作空间，不输入未来历史）；
d_jl = KL(T_jl^sigma || P_jl^{-,sigma})。

使用固定的动作白化尺度及共同高斯小核，prior samples=16、teacher samples=16 作为首轮起点；logsumexp 估计混合密度。评分采样与建密度采样分离。投影多对一，不把原始 residual 高斯 log_prob 当作最终动作密度。保持所有样本，禁止先平均动作再做单高斯 KL。
beta_jl = stopgrad[epsilon + (1-epsilon)*exp(-max(dhat_jl,0)/T)]，epsilon=0.1 作为起点；T 用训练/评分校准集 gap 的稳健尺度确定并冻结，不能看测试成功率调。记录 beta 分位数、ESS、所有无效值及被选片段相位分布。
这只是 prior-gap 代理，严禁写成条件互信息、反事实判据或无偏可预测概率。

omega_j：同一 F 尝试中有效、安全并最终稳定成功的片段。不要让某次失败后的重试成功倒灌为先前不良推进的正向标签。首轮不额外叠加多个 Q/恢复性权重。
从 beta 抽 Bernoulli 动作 mask；后验只看块初信息、mask、选中动作，不偷看被遮蔽的未来动作。共享 z 重构所选步；空 mask 跳过。
L_distill = -E_q sum_{selected l} log p_psi(u_exec_l|H,z) + KL(q_phi(z|H,U_selected,mask)||p_psi(z|H)).
L_ACT = L_expert + lambda_D L_distill，起点 lambda_D=1；expert/online batches 1:1。标注固定 decoder likelihood（例如训练集尺度的 Gaussian）；不把 L1 直接称为 Gaussian NLL。旧 ACT/teacher/gap/mask 权重 stop_gradient；ACT 不接 -grad Q。

### 4.2 SAC 与版本化 replay

奖励 Q 采用 SAC 标准 entropy-regularized target；Q 的动作约定为原始 residual，确定性 adapter/投影视为环境的一部分，entropy 对 residual proposal 而非投影后动作。另存最终物理动作给 g 头和蒸馏。
state 含历史、F/R、块 phase、基础当前/剩余计划、必要 controller memory，防止计划混叠。不要将跨版本计划直接复用为正确的 next state。
F 基础指令来自 ACT；R 丢弃原 ACT 块，基础指令为固定顺应卸载/有限回抽，同一 SAC actor 接收 mode=R。
保留官方几何进度和接触惩罚，去除 success-prediction 奖励；R 有到达已卸载可重新规划集合的奖励、时间/载荷成本。所有比较组完全一致。明确各奖励公式/系数写进 config。
ACT 每个外层轮次冻结。先只用本 ACT 版本在线 replay；HF 数据不满足转移契约时不直接入 TD。
采样/更新起点：每个 vector_step 1 个 gradient update，batch=256；global_transitions += num_envs。分别记录 vector_steps、transitions、critic_updates，不将 num_envs 倍数漏记。
常规起点 actor/critic lr=3e-4、gamma_reward=0.99、target tau=0.005，初始化小残差。只是待验证默认值，非声称最优。

### 4.3 双可恢复性

Q_g^m(s,u,b) 是先执行物理动作 u、随后持续目标模式 pi_m、在 b 步内到目标且不先入危险集的概率，m∈{F,R}。概率头不加入 entropy，也不做 expectile；gamma_g=1，剩余预算 b 明确输入。
目标：到 G_m ->1；危险终止或预算耗尽 ->0；其余 E_{u'~pi_m} target Q_g^m(s'_m,u',b-1)。BCE 用 soft targets。
使用真实 mode 对应的转移；next action 从当前目标 mode 重采样，而不是 replay 的行为 next action。F->R 主动切换处为 F counterfactual continuation bootstrap，不把 R 后续轨迹当 F 续行，也不把切换贴失败。必须保存切换前 F 的 remaining-plan/controller state。
G_R 必须同时包括卸载、足够回退/分离且仍保有可重试的抓持/位置；只有 force≈0 不算（掉落也会没力）。这些条件可在仿真定义成目标标签，actor 不读取未来/真值答案。
先收集实际 R rollout。失败数据不提供 g_R 真值。先固定恢复监督器并以 shadow mode 训练 g；达到预先规定验证和数据覆盖要求再启用概率切换，否则明确报告 g 未校准/未激活。
不训练 Q_Omega：continue_score=g_F(s,b)；retry_score≈g_R(s,b)*mean_restart_gF(b-d)，d 来自恢复数据；同一预算、gamma_g=1。加滞回/时间上限。该近似是一次恢复后重试，不宣称全局最优。

### 4.4 幂等方向投影

在功率保持尺度和可控子空间中：n=-w/||w||；e=延迟对齐后的已下发指令-实测速度；r=(I-nn^T)e；t=r/||r||，仅有效时构造。
G(v)=v-[n^T v-kappa]_+n-chi(t^T v)t；kappa>=0，chi∈{0,1}；对基础+残差的完整指令作用，最后统一非负缩放限幅。固定方向下幂等。无接触/退化方向必须有稳定分支，不除近零量。
chi 最简用接触载荷与正交误差持续阈值，计数器/方向缓存防抖；它不是第四网络。数值测试幂等性、加载上界、负向卸载和可控子空间保持。
不要宣称系统被动性/真机安全已证明。功率是指令端口性质；继续保留独立硬限幅与保护。

## 5. 最小实验：一个预训练起点、一个共享教师、三条后续分支

先做 smoke，再做 pilot；全流程第一任务只 PegInsert。
建议 pilot 配额（研究预算，不是效果保证）：
- 先使用成功训练 split 中固定 200 episodes，ACT 10,000 updates；若基础策略完全失败，先审计/增加到所有训练示教并最多 20,000 updates，不能故意把 ACT 弄差。
- actor observation 默认单 RGB + state/force；相机匹配失败先报告，不把 OOD 视觉误差算方法失败。
- 8 env 起跑，可内存允许时增至16/32且各组一致。
- seed=0；教师在线预算 200,000 individual transitions，必要时预先登记上限500,000。失败原因先定位，不能无限加预算挑成功。
- 所有训练中调参只用32个 validation reset seeds；最终 pilot 用另外100个固定测试初始条件。测试用结果不可反调阈值。

分组（A0为评估参考，真正核心比较 A1/A2/A3）：
A0 原始 ACT-only：预训练后直接评估，无额外训练，保留统一底层硬安全；无学习 residual/恢复性能主张。
A1 NoDistill：冻结 ACT0，模式 SAC+双g+幂等投影；这是共享教师起点。
A2 Uniform：复制同一 ACT0；同一教师执行数据、omega、专家混合、训练updates，beta恒1，完整CVAE蒸馏。
A3 Selective：复制同一 ACT0；除 beta 来自 prior-gap 外，与 A2 完全一致。

步骤：
1. 训练并评估 A0。
2. 在 ACT0 下训练 A1。先确认“有效纠偏教师”存在；教师没有比基础行为更好的安全纠偏就不解释蒸馏失败为核心理论失败。
3. 冻结 A1 教师，收集固定200次尝试（成功失败全记录），A2/A3共用完全相同数据。若有效成功轨迹很少，停止蒸馏判定并报告 bottleneck。
4. A2/A3 各5,000次更新；同 batch 顺序、同源数据、同优化器设置，仅 masking 不同。
5. 零额外RL成本：评估 ACT0-only、ACT_uniform-only、ACT_selective-only，硬安全逻辑相同，检验是否真的内化经验。
6. 完成一轮双向闭环：A1/A2/A3 从相同 residual+双g checkpoint 各继续100,000 transitions。A1 不改 ACT；A2/A3 使用新ACT、清空或正确重建版本化 replay，给相同适配预算。不能拿旧 residual 不适配直接永久叠到新ACT上。
7. 用相同测试初始条件/扰动序列评价完整三分支。不得让选择性组多拿交互/更多恢复重试。

统计与决策：
- pilot 仅判断是否值得继续，不称证明。100回合可能区分不了小差距。
- 若教师有效且 A3 相对 A2/A1 有合理成功率、力安全或 residual 节约信号，才重复 seeds=[0,1,2]，每seed 200测试回合。三训练seed仍属初步证据。
- 实用 go 条件事先登记，例如 A3 vs A2 成功率+5个百分点且载荷不更差，或成功率差不低于-3个百分点而 residual 能量降低15%且恢复/超限不恶化。这些是项目决策阈值，不等于显著性证明。区间太宽则记 inconclusive。
- 如果 A2>A1 而 A3≈/低于A2，只能支持“蒸馏有用”，不支持选择性权重；记录欠拟合代理可能误筛，不能自动升级算法重跑后冒称原版本成功。
- 若 beta 几乎全低/全高、prior采样塌缩、教师不改善、传感器/控制不匹配，分别报告；不要用总 reward 掩盖。

不再为每个辅助模块都训练一套 ablation。A1/A2/A3 都保留相同 g 和投影，第一轮只检验蒸馏主线。可做低成本 frozen-policy force-shuffle 评估诊断（只打乱学习策略的力输入，不破坏保护通道）；明确这是使用力的诊断，不是“重新训练无力消融”的证据。双g/投影各自必要性待核心通过后再研究。

## 6. 先通过可行性，再做小规模泛化

先不新训练。冻结最优验证checkpoint，对A1/A2/A3做相同评估：
1. ID：原训练分布的新初始条件。
2. 初始位姿 OOD：仅一个维度扩大至训练范围外，先轻度偏移再加姿态；记录真实范围，不直接复用失败率100%的极端改装。
3. 接触/传感 OOD：分别改变一个训练范围外的摩擦值或力噪声/1-step延迟，再考虑组合。每组100回合筛查。
改变参数前审计 FORGE 已有 randomization 支持：训练中已经含有的摩擦/噪声值不能再称OOD。未见随机seed不等于参数泛化。每个变体独立config，与stock不混名。
通过后才选 GearMesh 第二任务。该任务单独训练验证叫“跨任务可复现性”；只有用 PegInsert 训练好的同一模型直接无微调执行才叫零样本跨任务泛化，不能混用术语。NutThread和6DoF动作解锁不在第一轮范围。

## 7. 指标、日志、审计与产物

主指标：稳定插入成功率；回合按固定总预算含恢复计数，不能通过更长回合提高成功率。
辅指标：首次成功/稳定完成时间、超载发生率、力峰值/95分位(N)、力矩峰值(N m)、卡滞率、R触发次数/成功率、误退出率、raw residual与实际纠偏能量、projection干预幅度、ACT-only成功率、g头Brier/校准曲线/删失比例、beta分布。
恢复成功率分母是实际进入R的次数，无R则NA不是100%；失败不能从时间平均中悄悄删掉。成功时间可条件统计但明确分母。报告ever_success与stable_success区别。
reset seeds、训练种子、场景初始参数、动作/force trace、完整命令、退出码、walltime、GPU显存、有效transition数量全部保存。相同seed不保证不同策略后续随机过程对齐，尽量预采样按时刻的扰动序列。
每seed给成功率及Wilson区间；配对评估差异、训练seed间波动分别报告，不把回合当独立训练seed。不得挑最佳seed。
不写预期数字当结果；所有表由 episode_metrics.csv 自动生成。未执行=NOT_RUN，环境缺失=BLOCKED，程序错误=FAILED，可评价但不支持假设=NEGATIVE/INCONCLUSIVE。

建议结构：
closed-loop-il-rl/
  AGENTS.md
  CODEX_TASK.md
  README.md
  pyproject.toml
  configs/{data,env,algorithm,experiments}/
  src/prd/{data,envs,models,training,evaluation}/
  scripts/{doctor,audit_dataset,smoke_forge,train_act,train_teacher,collect_teacher,distill_act,adapt_residual,evaluate,summarize}.py
  tests/
  reports/{data_audit,action_contract,sensor_contract,environment,results}/
  results/<run_id>/{config.yaml,manifest.json,episode_metrics.csv,train_metrics.csv,status.json}
  artifacts_manifest.json
  THIRD_PARTY_NOTICES.md

必须单测：split不泄漏、HF task过滤、mask后验不见未来、先验/后验推理区分、完整latent保持、动作roundtrip/第7维隔离、EMA单次、投影幂等/功率/退化方向、F切R删失目标、预算终止、向量终止obs与reset隔离、versioned replay、单环境步和global transitions计数、只有当前phase/mode合法。
提供 no-Isaac CPU unit tests；真实 Isaac smoke 独立标记。Mock通过不称仿真通过。
构造简单多模态/未来泄漏合成单测是代码验证，不作为机器人实验结果。

## 8. Codex 必须交付可运行命令，不只给伪代码

实现一个入口，至少支持：
python -m prd.cli audit
python -m prd.cli smoke --task peg_insert
python -m prd.cli run --protocol pilot --seed 0
python -m prd.cli run --protocol confirm --seeds 0 1 2
python -m prd.cli evaluate --suite generalization --no-training
python -m prd.cli report --run-dir <actual_run>
这些是要实现的接口，不是当前已经存在的命令。README 中同时给所选 Isaac Lab launcher 的真实调用方式；camera路径需要对应 --enable_cameras。不能将示意命令标为已验证。
训练进程前台执行、定期checkpoint；中断后可恢复到明确状态，不承诺在任务结束后无管理地继续工作。

## 9. GitHub 提交规则

本仓库可能最初为空。已有内容先读，不覆盖用户修改。初始化后工作分支建议 codex/prd-mvp；每阶段提交：audit -> adapter/tests -> algorithms -> pilot真实结果 -> confirmation -> generalization。
可以向用户指定仓库推送实现和真实结果，但不能伪造已跑、删去负结果或强推覆盖。
Git中提交源码、configs、lock/manifest、CSV/JSON、小图、报告、精简视频索引。不上传HF原始数据、Isaac缓存、tokens、完整本地env、巨大replay；权重大文件以Release或经授权的存储上传，写SHA256和可验证位置，不发本机绝对路径冒充公开下载。
CPU CI运行单测，GPU仿真需实际可用的自托管/开发主机，不以普通CI通过声称GPU训练通过。
无认证/推送失败就保留commit和完整报错，不宣称上传成功。

最终Codex回复必须列明：实际完成阶段、commit/branch、执行过的命令和退出码、实际数据规模、原始结果位置、四组比较、哪些假设被支持/不支持、未完成原因。实现完但无GPU时，结论只能是“代码待真实仿真验证”。

## 10. 第一条执行指令

先读本任务书并冻结算法范围。现在从 P0 环境与数据审计开始，随后按序实现、测试并运行 pilot；每个阶段失败先定位，达到预算上限就诚实记录。未通过pilot不自动开展大规模泛化，不增加网络，不把最后两版理论补丁带回。只有真实运行后才能填写实验结果并推送。
