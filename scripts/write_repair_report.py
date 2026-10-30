"""Render current measured baseline and repair evidence without obsolete placeholders."""
import csv
import json
from pathlib import Path

report=Path('reports/20261023T104759Z')
results=Path('results/20261023T104759Z')
final=json.loads((report/'diagnostic_repair_result.json').read_text())
inputs=json.loads((report/'replay_input_audit.json').read_text())
manifest=json.loads((report/'run_manifest.json').read_text())
baseline=json.loads((report/'baseline_training.json').read_text())
episodes=list(csv.DictReader((results/'act0_validation.csv').open()))
successes=sum(int(r['stable_success']) for r in episodes)
text=f'''# 实测结果与诊断停止报告

配对校准状态：**NO_GO_REPLAY / NO_GO_CALIBRATION**。没有完成 D1、D2 或 B1/B2/B3。普通直接 Q 对照按任务书允许的独立路径执行，实时状态和结果见本文末尾；不能将其当作配对消融完成。

| 阶段 | 实测进度和结果 |
|---|---|
| 实际 FORGE smoke | 1 / 8 环境均有通过记录，用户已接受 EULA |
| ACT0 示教预训练 | 10,000 次更新；已有权重未覆盖 |
| ACT0 真实评估 | {successes}/{len(episodes)} 稳定插接，批次成功率 {100*successes/len(episodes):.2f}% |
| 共享交互 | {baseline['real_complete_episodes']} 回合；14,304 控制步 |
| 普通 Critic | {baseline['critic_updates']} 次更新；loss 不是插接成功率 |
| 修复前 D0 | 原始 JSON 故障前已保存 2 点、42 条续行、36 次比较；已恢复并独立归档 |
| 修复后 D0 | 保存 1 条 O、3 条 R、3 条 P；R/P 各 0/3 通过。只测到首个父状态的 A0，未完成 8 点或全部 3 个候选 |
| D1 / D2 / B1–B3 | NOT_RUN；前置回放门槛未通过 |

## 软件修复和真实验证

NumPy 布尔值/标量在 JSON 写入前转换为相应 Python 数值类型，NaN 和未知对象仍然拒绝。快照额外保存/恢复关节控制目标，修正 Isaac Lab `reset_to` 用实测关节状态替换控制目标的问题；重置前清除残留速度/力矩目标，并重置间隔随机事件倒计时。独立真实探针确认修复后的控制目标差和事件时钟差为零；**物理回放仍未通过**。

诊断现在每条轨迹单独保存，每 5 秒报告真实完成的物理步。守护器检查生产进程：启动超时 300 秒，连续无物理进展 180 秒抓栈并中断，留 30 秒收尾；日志刷新不算物理进展。子进程停滞及退出码保留已用真实软件子进程测试，不能当作机器人成功证据。

修复后一次正式诊断仍发生原生段错误（退出 -11），没有找到足以宣称该原生故障已根治的证据。原始日志/失败事件完整保留。新的逐物理步写前日志用于防止未来原生故障再次绕过 `finally` 而丢失账本；没有把这个新机制说成已经恢复了过去未记录的数据。

## 为什么 D0 阻止继续训练

已保存的 7 条真实轨迹足以重新计算 A0 的六次 O/R/P 比较，全部未通过冻结容差。门槛要求所有比较通过；增加其它候选或父状态不能抵消已失败的必要比较。因此记录提前 NO_GO，其余候选/检查点明确 NOT_RUN。这不是完成全部 D0，更不是算法有效性结论。恢复用 CPU 计算已有轨迹之间的误差，没有用 CPU 梯度替代真实交互。

两次独立新进程的初始策略输入中，state 与 history 的最大差均为 0，但 RGB 的最大像素差为 {inputs['fields']['rgb']['max_abs']:.0f}、平均绝对差 {inputs['fields']['rgb']['mean_abs']:.6f}。同一份输入重复推理差为 0；两次真实输入对应的动作最大差为 {inputs['gpu_action_difference']:.6f}。保存输入的额外推理确认图像变化会改变动作。尚不能把全部物理差异归因于图像，或宣称独立进程已解决配对问题。

两个 fresh full-rollout 工程探针的前缀动作本身不同，**不是匹配动作的 O/P 保真度实验**，不能拿它们给 D0 判通过。相机渲染设置未被静默修改，ACT0 的 6/32 保留为原条件下真实观测，不保证重新运行相同 seed 得到同一结果。

## 账本与后续限制

原生退出后恢复到最后落盘的下界：restore 168、branch 987、prefix 148 控制步。未落盘尾部未知，不能记为 0，也没有凭耗时猜上界。独立工程探针另行计入账本且避免重复计数。未来正式 diagnostic 重启会被守护器阻止，直到该预算缺口被核实；不会自动增加预算或跳过实验门槛。

本轮按 CODEX_TASK.md 的 D0 停止标准结束方向校准。若以后修订渲染/恢复协议，需要记录新版本、重新核对数据与评估条件，再开展新的验证；不能把修订后的环境结果拼进本轮配对统计。

## 可核查产物

- `act0_validation.csv`：原 32 回合真实评估。
- `fidelity_before_repair.csv/json`：写盘故障前的 36 次比较。
- `results/20261023T104759Z/fidelity.csv` 与 `diagnostic_repair_result.json`：修复后 6 次已保存真实比较和数据哈希；逐比较无法精确归属的总环境步字段留空。
- `diagnostic_1793004526346629461.log` / `_outcome.json`：原生段错误。
- `diagnostic_native_recovery.json`：已恢复步数下界与未知尾部。
- `replay_probe_*.json/log`、`replay_input_audit.json`：工程排错结果、源代码哈希及输入差异。
- 原始数组、轨迹与权重保留本地 `data/`、`checkpoints/`，没有作为 GitHub 仓库大文件上传。

观测命令：`watch -n 5 '.venv/bin/python scripts/training_status.py'`。配对消融保持门槛停止，普通直接 Q 对照单独显示。
'''
def read(path):
    return json.loads(path.read_text()) if path.exists() else {}
direct=read(report/'direct_q_result.json')
job=read(report/'direct_q_job.json')
training=read(report/'direct_q_progress.json')
render=read(report/'render_repair_latest.json')
text+='\n## 本次继续执行：普通直接 Q 与渲染排错\n\n'
text+='CODEX_TASK.md 第 D0 节允许在回放失败后保留普通直接 Q 基线。本对照不使用任何未验证 MC/pair 标签，不冒充 B1/B2/B3。初始模型、数据、优化参数、随机种子和评估种子先写入 `direct_q_plan.json`。\n\n'
text+=f"Critic 更新 {training.get('critic_completed',0)}/2000，ACT 更新 {training.get('actor_completed',0)}/200。作业记录：{job.get('status','未启动评估')} / {job.get('phase','')}；是否仍存活以实时监控为准。\n\n"
if direct:
    text+='| 普通对照 | 真实回合 | 稳定成功 | 批次成功率 | Wilson 95% 区间 | 平均代价 |\n|---|---|---|---|---|---|\n'
    for name,a in direct['arms'].items():
        low,high=a['wilson95']
        text+=f"| {name} | {a['episodes']} | {a['successes']} | {100*a['success_rate']:.2f}% | {100*low:.1f}%–{100*high:.1f}% | {a['mean_cost']:.4f} |\n"
    text+='\n这是真实仿真中的批次结果。更新包含 Q、额外 BC 和 step 正则项，不能把组间差异单独归因于 Q，也不能据此证明配对校准有效。原 ACT0 6/32 结果属于原评估批次，未覆盖。\n'
    text+='\n本批固定 ACT0 与更新 ACT 的成功次数只相差 1 回合；样本量小，区间较宽，结论为效果证据不足（INCONCLUSIVE），不据此进入 pilot/确认/泛化。两组触发物理保护均为 3 回合；更新组平均代价下降，但平均折扣回报也下降，完整数值保留在 `direct_q_result.json`。\n'
    audit=read(report/'direct_q_integrity.json')
    if audit.get('status')=='PASSED':
        text+='\n最终产物一致性核验通过：200 次 ACT 更新记录、两个模型哈希、各组 32 个预注册种子、64 条真实 CSV、正常进程退出和全部物理步账本均已核对。见 `direct_q_integrity.json`。这是产物核验通过，不是算法有效性门槛通过。\n'
else:
    text+='真实评估尚未全部完成；没有将梯度、loss 或训练完成当作插接成功率。\n'
    for name in ('direct_B0','direct_Q'):
        path=results/f'{name}_validation.csv'
        measured=list(csv.DictReader(path.open())) if path.exists() else []
        text+=f"\n- {name}：已完成 {len(measured)}/32 回合，已完成回合中稳定成功 {sum(int(v['stable_success']) for v in measured)} 次。未完成批次不报告最终成功率。\n"
text+='\n使用原基线相机设置，两组均使用相同的控制目标/随机事件重置修复。各组在独立 Isaac 进程执行，逐物理步写前日志、180 秒无进展中断和退出状态均保留。真实 CSV 每回合刷新。\n'
text+=f"\n渲染工程检查最新记录：`{render.get('result','无')}`，状态 `{render.get('status','未运行')}`。AA、历史清空、性能预设和抖动设置探针均未证明像素/动作完全一致；没有作为 D0 通过证据，也没有把探针相机设置应用到普通对照。\n"
(report/'FINAL_REPORT.md').write_text(text)
(report/'DIAGNOSTIC_REPAIR.md').write_text(text)
