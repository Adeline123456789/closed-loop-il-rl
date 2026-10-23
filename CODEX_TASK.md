# Codex 实施任务书：力觉 Critic 直接更新 ACT 的最小验证

协议：`force_critic_act_v2_diagnostic_first`。这是待执行的研究协议，不是已验证算法。
目标仓库：https://github.com/Adeline123456789/closed-loop-il-rl
当前状态：已增加 CPU 可验证的学习与控制组件及 P0 实测审计；P0 尚未验收，没有本项目机器人训练结果。具体执行状态见 reports/PROJECT_STATUS.json 和对应 implementation_audit.md。旧版 `prd_mvp_v1_prior_gap` 已作废，Git 历史保留，禁止继续按旧版执行。

## 0. 先固定研究对象，不再换题

保留的主计算图：
示教 -> ACT 初始化 -> 完整动作块 -> FORGE 接触交互 -> 收益/力代价 Critic -> 对动作求导 -> 直接更新 ACT -> 新交互。

只实现两组可训练组件：
- ACT-CVAE（ACT 本身就是在线 Actor，不是永远冻结的基础策略）。
- 两个 Critic 副本，每个输出收益 Q_R 和接触代价 Q_C；target copies 不是新的研究模块。

不实现独立 residual SAC、双可恢复性头、prior-gap 蒸馏、教师条件互信息、方向投影安全证明、刚度回归、世界模型、QP、可学习恢复监督器。确定性的控制适配、动作限幅、硬载荷保护和必要 reset 保留。

第一任务只做 stock `Isaac-Forge-PegInsert-Direct-v0`。先诊断，再有限训练；诊断失败不自动扩大模型或预算。负结果允许且必须提交。

核心假设不是“AC3 已证明梯度方向错误”，而是：在本任务部分困难接触状态上，普通 Critic 对有限可执行动作变化的判断有误；真实配对执行能提供可信监督；修正后梯度能改善 ACT 的实际输出。
校准的默认对象是有限尺度价值差，不宣称接触切换处存在真实微分。无新增配对信号时仍可判断普通 Critic->ACT 路径是否可用；校准无效不等于整个直接强化闭环无效。

## 1. 来源与复用边界

所有 main 链接只供定位；安装前锁实际 commit、许可证及依赖。不把全部仓库混装成一个环境。

### 必要工程参考
1. NVIDIA Isaac Lab / FORGE
   https://github.com/isaac-sim/IsaacLab
   https://github.com/isaac-sim/IsaacLab/tree/v2.2.1/source/isaaclab_tasks/isaaclab_tasks/direct/forge
   https://isaac-sim.github.io/IsaacLab/v2.2.1/source/setup/installation/pip_installation.html
   https://isaac-sim.github.io/IsaacLab/v2.2.1/source/features/reproducibility.html
   候选兼容栈：Isaac Lab v2.2.1 / Isaac Sim 5.0.0 / Python 3.11；此组合来自对应版本文档，不保证等于数据采集栈。已有能工作的环境优先审计，不盲目重装驱动。
2. ACT
   https://github.com/tonyzhaozh/act
   https://github.com/huggingface/lerobot
   https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/modeling_act.py
   复用 ACT/CVAE，增加六维力历史的输入适配；RL 调用可微 forward，不调用带 no_grad 的部署 predict_action 接口。
3. TD3 / TD3+BC
   https://github.com/sfujim/TD3
   https://github.com/sfujim/TD3_BC
   复用 replay、target update、直接 Q actor update。旧 MuJoCo/D4RL/Python 环境不需要安装。我们是动作块及收益/代价头的适配版本，不宣称原版 TD3+BC 复现，也不将隐式 CVAE 当有可计算密度的 SAC 高斯 actor。
4. FORGE 数据与发布者专家
   https://huggingface.co/datasets/angledusgar/forge-v1
   https://huggingface.co/datasets/angledusgar/forge-failure-v1
   https://huggingface.co/angledusgar/forge-peginsert-v1
   PPO checkpoint 仅用于兼容性检查与必要时补采示教，不是新增学习模块。

### 文献定位，不作为可用代码的保证
- AC3: https://arxiv.org/html/2508.11143v2
  作者链接 https://github.com/flyfaerss/ac3 ，本次读取 README 只有 `# ac3`，不能依赖其存在完整训练代码；Codex 再检查实际文件，缺失就明确记录。
- MAGE / How to Learn a Useful Critic?: https://arxiv.org/abs/2004.14309
  只作价值导数研究参考，不额外实现动力学网络。
- ACT–Flow：成功轨迹 BC 回流不是本协议的核心更新路径，禁止用它替代 Critic 对 ACT 的动作梯度。

## 2. P0：先审计数据与控制（不计为方法消融）

### 2.1 硬件和运行环境
写 doctor，记录 GPU 型号/显存/驱动、Torch/CUDA、Isaac Sim/Lab SHA、资产、相机、磁盘、Git 认证。先 1 env headless+camera，再 8 env。没有支持当前 Isaac Sim 的 GPU 或资产时记 BLOCKED；CPU/mock 单测通过不等于 FORGE 通过。不得自动付费租机器、修改系统驱动、绕过许可、上传凭据。

### 2.2 HF 数据审计
先 metadata+少量 parquet/视频，不下载所有任务再过滤。只取 PegInsert。数据卡显示：state[8]、actions[7]、ft[6]、ft_noisy[3]，15 Hz，两相机；ft 已平滑，不称完全原始力。失败卡片 51 与 UI 53 数量不一致，按固定 revision 的 metadata/parquet 实测，不硬编码。

成功集用于 ACT 预训练。失败集用于接触/动作/结局审计及离线失败分布诊断，不把每帧标成不可恢复，不把失败动作全部纳入正向 BC。
数据缺少完整机器人/接触/求解器快照，不能从 HF 文件直接恢复同一接触状态；配对分支必须在新仿真 rollout 中保存或重放。
数据没有经过验证的块级奖励/终止/计划时，不伪造为离策略 TD 训练集。相邻 4 个示教动作可以作 BC 目标，不能假定它们原本由一个开环块生成并直接作为块级因果转移。

按完整回合分组拆分 70/15/15（有采集 run id 时按 run 分组）。同一回合全部片段、前缀和派生分支在同一分区。episode key=(dataset,revision,task,episode_id)，防止成功/失败编号冲突。只用训练数据估计标准化量。

### 2.3 stock 任务与动作契约
数据卡警告采集 checkout 还有另一个困难变体；首轮必须 stock，不静默开启 ±10 度 peg tilt。核对采集时恢复的 stock 配置：
hand_init_pos=[0,0,0.047], hand_init_pos_noise=[0.02,0.02,0.01],
hand_init_orn_noise=[0,0,0.785], held_asset_pos_noise=[0.003,0,0.003]。
只有源码有相应 tilt 字段才设置为0，不传不存在参数。

所查 v2.2.1 FORGE 的七维动作不是六维速度+夹爪：前六项映射相对零件的位姿目标，roll/pitch 被约束，第七项是成功预测。重新读所选源码确认。
主实验使用四个实际可控量 x/y/z/yaw，ACT 输出 Kx4 的原生归一化指令。adapter 插入被固定的两个轴与第7维占位。关闭成功预测奖励，成功只用独立几何判据。六维力输入不等于六自由度控制。
动作记录在 EMA 前还是后、坐标/单位/yaw wrap/限幅顺序，必须查清。提交给 env.step 的动作、EMA 后目标、实际下发目标、实测运动分别记录。不得把实测速度当策略动作，不重复 EMA。
不能确认 HF 动作含义或相机配置时，先完成小样本回放；确实无法对齐则记录 BLOCKED_DATA_ALIGNMENT，并用兼容专家在当前环境补采少量明确语义示教。必须区分 HF 预训练与补采数据，不称完全由 HF 复现。

### 2.4 力和时间契约
ft 六维包含力和力矩；ft_noisy 仅三维，禁止补零冒充六维。核对传感器参考点、坐标、正负号、平滑与延迟。包括力矩平移项，不能仅旋转 force 后直接保留 torque。
观测首轮：单固定 RGB + state + 最近 H=4 步六维力/运动/指令历史。力同时进入 ACT 和 Critic，所有组相同。可记录抓取后无接触基线并减偏置，但绝不让六维力历史偷用未来。物理阈值与代价使用正确单位，不直接用标准化网络输入计算 N/Nm 阈值。
若数据视觉信息不匹配，不暗中给 actor 真实孔位/接触模式；仿真真值只允许用于奖励、终止、状态恢复与诊断。

K=M=4；先不 temporal ensemble、不动态块长，完整块一次产生，除硬保护/成功外执行到块尾再重规划。policy FPS 从实际 dt/decimation 读取并匹配数据；15Hz 与 4步块不称千赫兹力控。训练与采集的块执行规则必须相同。
回合有限时域 T 由任务配置确定，剩余预算输入 Critic。块提前终止 k<K 时记录原计划、有效执行 mask、k、gamma^k 和 final_observation。不得 bootstrap 到自动 reset 后状态。审查 FORGE 全批次同步 reset，诊断优先 num_envs=1，防止其他并行环境被隐式重置。

P0 报告：data_audit.md/json、action_contract.md、sensor_contract.md、environment.lock.json、expert_smoke.csv。专家卡片报告的成绩不能当项目实测。

## 3. P1：可用 ACT 起点与普通 Critic

### 3.1 ACT
成功训练集先取固定200条回合，训练10,000次更新；验证32个固定初始条件。若无任何有效行为，先审计接口，可扩到全部训练示教/20,000更新上限；仍失败则 BLOCKED_BASELINE，不能算校准失败，也不刻意弱化 ACT 制造空间。
单相机、已有视觉骨干；首轮可冻结视觉骨干，ACT transformer/decoder 与低维输入层在线可训练，所有组一致。BC 保留 CVAE 后验重构+KL。
部署、Q策略目标和配对冻结续行统一使用 z=0 的确定性先验推理（不是后验）。这是最小版，不宣称解决多模态采样问题。后续采样先验实验另立协议。

### 3.2 网络与损失
A=ACT_psi(h,z=0)，不是 frozen_ACT+bounded_residual。
两个 Critic 各输出 (Q_R,Q_C)。输入是可观察历史、剩余时间、完整原生动作块与必要的已知控制器记忆。状态特征在 Actor 更新时作为固定条件；不允许 Actor 改变 Critic 的观察编码以“刷Q”。首轮 Critic 用固定 ACT0 视觉特征+自身低维层，Critic 优化器不修改 ACT 参数。

r 使用所选官方几何进度/成功项，去除第7维预测奖励和原接触惩罚以避免与独立成本重复；导出每项公式与系数。终端成功奖励只记一次。
c = sum_i relu(|f_i|/F_lim_i-1)^2 + sum_i relu(|tau_i|/Tau_lim_i-1)^2 + unsafe_penalty*unsafe。
阈值根据任务/资产和开发验证数据预先登记；没有可信阈值就标注代价尚未校准，不能看测试调。硬保护独立于学习。若能获取物理子步峰值，成本/安全检查不能只看低频平滑力；否则报告采样带宽限制。

R=sum_l gamma^l r_l, C=sum_l gamma^l c_l, gamma=0.99起点。
y_R=R+gamma^k*bootstrap*mean_i Q_R_target(h',ACT_target(h'))；C同理。
概率/安全保证不在本项目范围。初轮固定归一化后 lambda_cost=1，固定 alpha_BC/eta_step；不新增乘子学习器或Q加权示教，以免归因混乱。
L_actor= -mean_valid(min_i Q_R_i(h,A)-lambda_cost*max_i Q_C_i(h,A)) + alpha_BC*L_demo_CVAE + eta_step*||A_psi(h,0)-stopgrad A_old(h,0)||^2。
state mask全部为零时只做BC/保持项，不除以零。先拟合并固定 Critic，再小步改ACT，不需对模拟器求导。same-z保持项是移动参照，不永久锁初始ACT。

建议起点：critic lr=3e-4，actor lr=1e-5，batch=128（OOM可64且所有组一致），target tau=0.005。不直接继承旧 TD3_BC 的lambda自适应尺度又称相同目标；用训练尺度固定归一化，记录所有值。

### 3.3 共享数据
先用ACT0及小幅完整动作块探索收集100次仿真尝试或50,000个独立环境步，以先到者为限。不足以形成成功/失败价值范围时先报告，不强行把全零Critic做校准。
统一保存块计划、实际执行、成本、结果和policy_version。行为动作分布不同可用于动作条件TD；next action来自被评价的目标策略。多步Monte Carlo标签只属于指定的冻结续行策略，不能跨策略版本无限复用。
普通Critic先训练最多10,000更新，训练/验证按parent episode分组。HF失败不能伪造回报入buffer。

## 4. P2：最小诊断，先回答三个“能不能”

### D0 回放保真度（先8个检查点）
从新仿真采样8个独立检查点，包含正常接触/停滞/模式切换附近。至少记录所有刚体/关节、速度、夹持、控制积分/EMA、force滤波、history、episode计数、物理参数/随机种子与外部噪声时间表。不能声称用户快照包含全部PhysX内部状态，必须实测。
- O：到达检查点后不重置，执行A0及冻结续行。
- R：快照恢复后执行相同测试，重复至少3次。
- P：从明确初始条件重放固定动作前缀，再测试；R不可靠时作为替代。
重复初始seed+固定前缀仍需验证，不把同一seed自动当相同物理状态。对代表性A+/A-也检查O/R/P，不只检查A0。
测轨迹位姿/速度/力偏差、回报与成本差和成功差。A/A只说明重复性，不能排除每次恢复都到错误状态。采用预先登记的等价范围，而不是p>0.05就算通过。
将收益/成本按开发集固定尺度归一化；工程默认最小有效差delta_effect=0.1个尺度单位，回放容差delta_replay=0.025；物理位姿/force容差必须按资产与采样定义。它们是工程门槛，不是通用统计保证。测试前锁protocol_effect_thresholds.json。
R失败用P；都失败则NO_GO_REPLAY并停止方向校准；可以保留普通直接Q基线，不虚构结果。

### D1 有限动作效果是否可辨识（先24个独立检查点）
固定24个parent episode检查点：16困难、8正常。按parent episode分成12训练/6验证/6诊断测试并分层；检查点是pilot不是足够统计证据。全部派生分支随其父episode进入同一split。
困难定义来自事先规定的接触载荷+低进度/偏差，不使用被测试Critic的好坏选样。不得把“本次失败”叫不可恢复。
每点选择2个方向模板（如卸载、侧移或yaw；只能在stock可控空间）；尺度从开发集冻结，默认候选为原生归一化动作幅度0.02和0.05，adapter必须导出对应米/弧度且检查硬边界，若不合理先校准再锁定。
A+=A0+epsilon*d，A-=A0-epsilon*d。方向单位范数、时序模板写进manifest。裁剪后不对称时用真实候选差，不能仍按2epsilon解释微分。
每个方向/尺度先做3次配对；另做3次A0。24*(2*2*2*3+3)=648条续行；不等于648次从零训练。先筛查不声称显著。
用同一冻结ACT续行，外部噪声按物理时间/通道对齐。完整剩余时域返回G_R/G_C为主要标签；短窗结果只叫短时效果，不用被检验的Q补尾巴充当真值。
计算 Delta=mean(G+ - G-)、I+=mean(G+-G0)、I-=mean(G--G0)。A+胜A-不代表优于A0。
先分析有限效应及配对区间，不先除epsilon。重复不独立时按episode/外部噪声块聚类；完全确定重复不能冒充统计样本。多方向搜索后独立复测，预留至少部分被拒绝点复查。
最多对8个候选条件/检查点追加独立重复至8次；诊断总预算上限1500条续行或300,000环境步（包括prefix/restore burn-in），先到者停并报告INCONCLUSIVE，不自动加预算。
输出三类覆盖：局部斜率候选可信；有限效应可信但导数不可用；当前预算不可辨识。主要校准仅依赖第二类也可参与的有限效应，不要求把跨分支差值当导数。
训练筛选标记仅由训练分支生成。测试分支只用于评估，禁止用测试效应选择actor样本/调参。恢复容差经验估计不是严格偏差上界。

### D2 “正确方向”能否成为ACT更新
对训练中可信改善点，诊断副本进行：候选动作直接执行；Q-only小步更新；Q+BC+step小步更新。固定输入与z，记录真实DeltaA、与改善方向夹角、动作饱和、Q梯度/BC梯度范数及实际回报变化。
同时在固定的正常成功验证集评估遗忘。输出是参数更新的实际结果，不能用Q上涨代替。
可做诊断性局部监督拟合检验表达能力，但其副本不参与主实验、不把它算作Critic更新效果。
禁止声称反向梯度投影保证输出动作只沿该方向；本最小版不实现方向子空间反传投影，统一用普通autograd+小步长+验证。
必须区分：Critic排序/尺度错；BC抵消；输出限幅；参数共享遗忘；信息不足。如果原Critic判断已准确或普通缩步足够，则停止把校准当主创新。

## 5. 固定最小校准形式，不再同时试多套算法

默认有限尺度配对价值差，不实现Sobolev二阶损失、模型梯度或随机平滑新策略。
对k in {R,C}：
L_point = mean_{b in {0,+,-},i} Huber(Q_k_i(h,A_b)-G_k_b)
L_pair = mean_{accepted pairs,i,k} Huber((Q_k_i(h,A+)-Q_k_i(h,A-)) - stopgrad(mean(G_k+ - G_k-)))
L_Q = L_TD + lambda_point*L_point + lambda_pair*L_pair。
起点lambda_point=lambda_pair=1；各量先用同一训练尺度归一化；loss归一化按样本数固定。方差权重可不做，首轮只有接受/拒绝，避免再调多个权重。

重要：两端点标签已含差分信息，L_pair是强调相对效应的归纳偏置，不宣称创造新监督。它能否改善实际Actor更新是待测结论。
每条MC/pair标签携带frozen_continuation_policy_sha；策略更新后旧MC不能继续当Q新策略真值。当前轮固定目标策略评价，下一轮用同规则重采新配对；普通真实转移可重用于当前目标TD。训练与评估return定义、budget一致。
观察近似相同而隐藏状态不同，不能给策略隐藏状态来消除问题；按历史条件不确定性报告。

## 6. P3：三组最小消融 + ACT0参照

先同一个ACT0、同一个普通Critic checkpoint，复制网络/优化器和随机种子。
B0 ACT0-only：仅评估，不额外训练。
B1 Direct-value：全部合法训练状态用于Q actor更新；Critic=TD+相同点值MC；lambda_pair=0。
B2 Direct-filter：与B1完全相同的Critic值训练，只用训练配对中预先确认的可改善可信状态作Q actor更新；其余仍可训练Critic/BC；lambda_pair=0。
B3 Calibrated-filter：与B2相同训练数据、actor状态mask、优化器、更新次数与预算；仅lambda_pair=1。

首轮B2/B3使用同一份冻结白名单（具体parent/snapshot，不靠测试标签或新增网络泛化mask）。不使用梯度方向投影；比较的是样本筛选与配对对比损失。以后若推广邻域过滤，另写协议，不能偷偷更换。
B1也拿到所有分支转移和端点MC，点值监督方式与B2/B3完全一致。额外计算量单独报告。B3比B2的差才可归于配对项；B2比B1检验筛选。不得拿不同teacher/更多data当校准贡献。

### 最省钱的先后顺序
1. 固定共享数据各再训Critic 2,000 updates，ACT从同一checkpoint只做最多200小步更新（不同时放大许多参数）。在32 validation starts及未见诊断检查点检验效果。
2. 只有B3出现可重复的预测/ACT实际改进信号且没有明显超载，才开始下面的完整闭环pilot；否则报告negative/inconclusive。离线成绩不当闭环成功率。
3. 完整pilot先seed=0，最多2轮；每组每轮最多50,000 individual simulator transitions，含普通rollout、配对采集、prefix、校准验证。B1/B2/B3具有相同总预算，配对采集协议相同。后续策略不同，轨迹自然可不同，不能再宣称同数据；第一轮共享数据才有严格同数据归因。
4. 每轮先冻结当前ACT采集并训练当前策略Critic，再短步更新ACT，评价，下一轮重新采集。新配对的续行策略必须为本轮冻结ACT；保持旧标签版本不混用。因可辨识性不足无法提供新mask时不无限搜干净状态，按预算停止。
5. 最终100个独立test reset条件，三组相同初始配置与预采样扰动序列；测试不调超参。稳定成功定义预注册：官方几何成功连续3个控制步（若官方更严格则沿用），任务固定总预算包含任何退出/重新接近。第7维不能宣告成功。
6. 信号值得继续才跑seeds=[0,1,2]，每seed200测试回合。三seed仍是有限证据。

原来的R-small/R-wide不在第一轮重新训练列表。诊断可用现有ACT和候选动作检查小残差边界是否排除了已验证改善动作；这不是残差RL性能证明。若以后宣称“比残差更好”，必须额外实现相同观测/动作权限/预算的R-small/R-wide或残差+相同校准对照。

## 7. Go / No-go：先锁工程门槛，不根据测试改结论

状态：NOT_RUN / BLOCKED / FAILED_IMPLEMENTATION / GO_PILOT / NO_GO_CALIBRATION / INCONCLUSIVE。
1. P0失败：BLOCKED，不谈算法有效性。
2. O/R/P无法提供可信比较：NO_GO_CALIBRATION。可以保留Direct基线，不转成伪配对实验。
3. 困难状态中可信有限效果覆盖不足：工程初筛以>=25%且至少4个独立困难检查点为继续门槛；这是项目预算决策，不是统计证明。必须报告区间和全部被拒绝比例；不足/样本太少记INCONCLUSIVE或停止，不只展示干净子集。
4. 普通Critic在这些点基本准确、但ACT仍坏：优先检查BC权重/梯度幅度/参数共享；不强推校准。方向错误不是先验事实。
5. 只改善训练差值误差、没改善独立检查点和实际ACT动作：不能GO。
6. pilot实践性信号：预登记B3相对B2稳定成功率+5个百分点且超限/峰值无明显恶化；或成功率差不低于-3个百分点、超限率相对降低>=20%且基准确有足够超限事件。低事件数、区间过宽则INCONCLUSIVE。阈值是继续投入准则，不等于显著性。
7. B1/B2已经同样好：保留更简单直接Q闭环，结论是校准不必要，不是所有idea失败。

所有门槛、尺度与选择规则须在test评估前写入protocol_lock.json。不得无限调参直到某组赢。

## 8. P4：只在pilot/确认有信号后做泛化

先不重新训练，冻结checkpoint：
- ID新初始条件100回合；
- 一个训练范围外的初始位置/姿态区间100回合；
- 一个训练范围外的摩擦条件100回合；
- 力传感1控制步延迟或预登记噪声条件100回合。
每次只变一因子，先检查FORGE已有随机化支持。新seed不是OOD，训练覆盖的摩擦不能标OOD。三组使用相同变体/seed。更困难但物理不可行的任务不算正常泛化测试。
可做冻结策略force-shuffle诊断（只打乱学习输入，不破坏真实保护/成本通道），明确它不是重训无力baseline。
只有以上通过才考虑GearMesh第二任务。重新训练后有效叫跨任务方法验证，不叫PegInsert零样本跨任务泛化。无真机结果不写sim2real成功。

## 9. 指标与原始数据

最少输出：
- fidelity.csv：O/R/P、同动作差、力/位姿轨迹误差、版本与恢复方法；
- pairs.csv：parent_id, split, continuation_sha, A0/A+/A-, epsilon/方向/实际单位, seed, G_R,G_C, success, unsafe, env_steps, cutoff；
- critic_diagnostics.csv：point_error, contrast_error, predicted_sign, observed_sign/区间, support状态；
- actor_step.csv：前后动作、实际DeltaA、梯度范数、BC冲突、候选效果与更新效果；
- episode_metrics.csv：稳定成功、ever/final success、时间、峰值力/力矩、超限、终止原因、steps；
- run_manifest.json：git SHA、HF revisions、环境/资产版本、命令、返回码、配置hash、时间、GPU/显存、计数器。

以parent episode分组汇总；不把同一snapshot多分支当很多独立训练seed。首次成功时间和失败时间分开注明分母；不只统计成功轨迹峰值。区间可用固定方案的配对/聚类bootstrap，独立种子单独报告。3次筛查重复不足以证明显著。
训练曲线横轴是所有individual env transitions，包括分支/前缀恢复和验证；另记录纯训练步、evaluation步与梯度更新数，不能藏掉配对采集成本。

## 10. 必须实现的接口与测试

建议工程：
  src/forcecritic/{data,envs,models,diagnostics,training,evaluation}/
  configs/minimal_validation.yaml
  scripts/{doctor,audit_data,smoke_forge,pretrain_act,collect_base,check_replay,probe_pairs,check_actor_step,run_pilot,evaluate,report}.py
  tests/
  reports/<run_id>/
  results/<run_id>/

CLI须真正实现：
python -m forcecritic.cli doctor
python -m forcecritic.cli audit-data
python -m forcecritic.cli run --stage baseline --config configs/minimal_validation.yaml
python -m forcecritic.cli run --stage diagnostic --config configs/minimal_validation.yaml
python -m forcecritic.cli run --stage pilot --require-go reports/<run>/decision.json
python -m forcecritic.cli evaluate --suite generalization --no-training
python -m forcecritic.cli report --run-dir results/<run>
CLI 已实现参数与前置条件检查；后续机器人阶段调度仍待 P0 验收后实现/验证，不能将阻塞返回视为阶段完成。README应给实际Isaac launcher/camera flags；不要把示例标为已验证。

最重要autograd单测：固定replay和ACT，改变Critic动作梯度，actor更新确实改变；Critic参数冻结不更新，但dQ/dA非零。不能在Q(actor_action)外层no_grad或detach动作。证明online actor收到Q项而不仅BC项。
其他单测：分割不泄漏、HF task过滤、动作第7维排除、roll/pitch约束、EMA单次、force帧变换、z=0推理不读未来、K步执行及提前终止、gamma^k、terminal_observation、防跨reset、零mask稳定、MC续行policy版本、paired数据各组一致、测试标签不进筛选、env_steps计数含所有分支。
CPU单测与真实Isaac smoke分开，Mock只验证软件。禁止以mock输出“插接成功率”。

## 11. GitHub提交与任务执行

已授权目标仓库；新工作分支`codex/force-critic-act-v2`。先读现有改动，不覆盖用户新代码，不force push。旧文档只通过Git历史保留，根目录新协议唯一有效。
分阶段提交audit -> env/data adapter/tests -> direct actor-critic -> diagnostics -> pilot results -> confirmation/generalization（条件满足才执行）。
提交代码/config、单测、dependency lock、CSV/JSON、小图与报告，生成结果表必须可回溯原始回合文件。未跑写NOT_RUN；GPU缺失写BLOCKED；负结果正常提交。
不得上传原始HF数据、大replay、Isaac缓存、tokens或整个conda环境。大权重按经授权的Release/存储上传并记录SHA256；失败则报告本地路径和缺失状态，不造下载地址。
只能运行实际可用的前台会话/用户已有受管理作业；当前消息结束不表示训练会自动后台继续。无需询问是否新增大模块，遇预算/可信性门槛直接按协议停止并写结论。

最终Codex回报：实际完成阶段、branch/commit、真实执行命令/退出码、环境是否启动、数据数量、所有组结果/区间、成本计数、停止原因、是否支持继续投入。实现完成但未运行仿真只能称“代码待仿真验证”。

## 12. 现在开始

先完成P0，随后P1及P2。当前交付若被 P0 阻塞，保留真实停止报告，不将后续未完成调度宣称为已实现。不要先把所有训练脚本空壳写完，也不要直接运行全套泛化。在可运行GPU环境中按门槛推进；条件不满足就以真实审计/诊断产物完成本轮交付。目标是尽早判断是否值得做，不是强行制造正结果。
