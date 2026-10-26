"""Generate staged conclusions only from measured CSV/JSON."""
import csv
import json
from pathlib import Path
from forcecritic.staged.common import locked_plan,CONFIG
from forcecritic.staged.results import read_rows,arm_metrics,matched_decision
from forcecritic.runtime import atomic_json,digest
from forcecritic.staged.audit import validate_completed_batch,gradient_evidence

p=locked_plan();report=Path(p['report']);result=Path(p['results'])
job=json.loads((report/'job.json').read_text()) if (report/'job.json').exists() else {}
audit=json.loads((report/'a2_reproduction_audit.json').read_text()) if (report/'a2_reproduction_audit.json').exists() else {}
a0=read_rows(p['historical_A0_csv']);a1=read_rows(result/'eval_A1_initial.csv');a2=read_rows(p['historical_A2_csv'])
extension1=read_rows(result/'eval_A1_extension.csv');extension2=read_rows(result/'eval_A2_extension.csv')
completed=lambda name: (report/(name+'.supervisor.json')).exists() and json.loads((report/(name+'.supervisor.json')).read_text()).get('returncode')==0
training=json.loads((report/'A1_training.json').read_text())
matching=json.loads((report/'training_matching_audit.json').read_text())
if matching['status']!='PASSED':raise ValueError('Full training matching audit incomplete')
if digest(training['checkpoint'])!=training['checkpoint_sha']:raise ValueError('A1 checkpoint changed')
admission={}
for name in ('eval_A1_initial','eval_A1_extension','eval_A2_extension'):
    if completed(name):
        sha=training['checkpoint_sha'] if 'A1' in name else p['inputs']['checkpoints/20261023T104759Z/direct_Q.pt']
        admission[name]=validate_completed_batch(report,result,name,p,sha)
gradients=gradient_evidence(result/'A1_updates.csv','results/20261023T104759Z/direct_Q_actor_updates.csv')
objective_probe=json.loads((report/'objective_probe.json').read_text()) if (report/'objective_probe.json').exists() else {'status':'NOT_RUN'}
physics_archives=json.loads((report/'physics_ledger_archives.json').read_text()) if (report/'physics_ledger_archives.json').exists() else {}
atomic_json(report/'gradient_evidence.json',gradients)
atomic_json(report/'evaluation_admission.json',admission)
decision=dict(status='NOT_RUN',reason='Matched A1 evaluation incomplete',extension_allowed=False)
if len(a1)==32 and completed('eval_A1_initial') and audit.get('status')=='PASSED':
    decision=matched_decision(a1,a2,p['initial_evaluation_seeds'],1,p)
    look_path=report/'phase1_look1.json'
    if look_path.exists():
        if json.loads(look_path.read_text())!=decision:raise ValueError('Initial decision changed')
    else:atomic_json(look_path,decision)
    if len(extension1)==32 and len(extension2)==32 and completed('eval_A1_extension') and completed('eval_A2_extension'):
        a1+=extension1;a2+=extension2
        decision=matched_decision(a1,a2,p['initial_evaluation_seeds']+p['extension_seeds'],2,p)
        look_path=report/'phase1_look2.json'
        if look_path.exists():
            if json.loads(look_path.read_text())!=decision:raise ValueError('Final decision changed')
        else:atomic_json(look_path,decision)
arms={k:arm_metrics(v) for k,v in [('A0_historical',a0),('A1',a1),('A2',a2)]}
if not completed('eval_A1_initial'):
    arms['A1'].update(status='INCONCLUSIVE',complete=False,success_rate=None,wilson95=None,reason='Evaluation still incomplete; counts are progress only')
if job.get('status') in ('FAILED_IMPLEMENTATION','BLOCKED'):
    decision=dict(status=job['status'],reason=job.get('error'),extension_allowed=False)
summary=dict(protocol=p['protocol'],config_sha=digest(CONFIG),phase1=decision,arms=arms,
    extension_progress=dict(A1=len(extension1),A2=len(extension2)),
    phase2=dict(status='NOT_RUN',reason=f"Phase1 is {decision['status']}; requires GO. Historical arms include force and cannot stand in for no-force B2"),
    phase3=dict(status='NOT_RUN',reason='Requires phase1/phase2 support and trustworthy replay; historical D0 NO_GO retained'),
    phase4=dict(status='NOT_RUN',reason='Requires phase3 GO'),phase5=dict(status='NOT_RUN',reason='Requires confirmation GO'),
    provenance=dict(historical_A2_already_observed=True,a2_first_update_audit=audit.get('status','NOT_RUN')),
    gradient_evidence=gradients,final_policy_objective_probe=objective_probe,evaluation_admission=admission,training_matching_audit=matching,
    physical_step_journal_archives=physics_archives,
    execution_manifest=json.loads((report/'execution_manifest.json').read_text()))
atomic_json(report/'phase1_decision.json',decision);atomic_json(report/'stage_results.json',summary)
with (result/'phase1_summary.csv').open('w',newline='') as stream:
    fields=['arm','status','episodes','successes','success_rate','mean_return','mean_cost','mean_peak_force_N',
            'max_peak_force_N','mean_peak_torque_Nm','max_peak_torque_Nm','overload_episode_rate','overload_step_rate',
            'protection_stop_rate','mean_first_stable_success_seconds','evaluation_control_steps']
    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
    for arm,values in arms.items():writer.writerow(dict(arm=arm,**{k:values.get(k) for k in fields if k!='arm'}))
project_path=Path('reports/PROJECT_STATUS.json')
project=json.loads(project_path.read_text())
project['active_protocol']=p['protocol']
project['active_stage_report']=str(report/'stage_results.json')
project['staged_v3']={k:summary[k] for k in ('phase1','phase2','phase3','phase4','phase5')}
project['staged_v3_execution']=job
atomic_json(project_path,project)
text='# 分阶段实验 v3：实测报告\n\n'
text+=f"第一阶段：**{decision['status']}**。{decision['reason']}。\n\n"
text+=f"当前执行状态：{job.get('status','NOT_RUN')} / {job.get('phase','NOT_RUN')}。\n\n"
text+='A0/A2 历史测试结果在本计划前已见；本轮补匹配 A1。新追加只按预锁规则执行，不追溯声称历史结果是新盲测。第一阶段两组都有力历史，不能直接作为第二阶段无力组。\n\n'
text+='| 组 | 已记录回合 | 稳定成功 | 成功率 | 95% Wilson 区间 | 平均回报 | 平均代价 | 保护停止率 |\n|---|---:|---:|---:|---|---:|---:|---:|\n'
for name,v in arms.items():
    if not v['episodes']:
        text+=f'| {name} | 0 | NOT_RUN | — | — | — | — | — |\n';continue
    if v['success_rate'] is None:
        text+=f"| {name}（未完成） | {v['episodes']} | {v.get('successes',0)} | 未报告 | — | — | — | — |\n";continue
    low,high=v['wilson95']
    text+=f"| {name} | {v['episodes']} | {v['successes']} | {100*v['success_rate']:.2f}% | {100*low:.1f}–{100*high:.1f}% | {v['mean_return']:.4f} | {v['mean_cost']:.4f} | {100*v['protection_stop_rate']:.2f}% |\n"
text+='\n未完成批次仅为进度，不作为最终成功率或阶段判断。完整峰值力/力矩、超限、失败类型和成功时间在 `stage_results.json`；原始回合在同名 results 目录。A0 只有历史 32 回合，不与追加后的 64 回合冒充严格同样本比较。\n'
text+='\n原 ACT0 在 20000–20031 起点的 6/32 仍保留。表中 A0 为另一批 21000–21031 的 5/32；不同批次不相互覆盖，不把旧 D0 崩溃解释为此前已完成 ACT0 评估失效。\n'
text+='\n| 组 | 峰值力均值 / 最大值 N | 峰值力矩均值 / 最大值 Nm | 超限回合率 | 首次稳定成功时间均值 s（仅成功） | 失败原因计数 |\n|---|---|---|---|---|---|\n'
for name,v in arms.items():
    if v.get('success_rate') is None:continue
    text+=f"| {name} | {v['mean_peak_force_N']:.3f} / {v['max_peak_force_N']:.3f} | {v['mean_peak_torque_Nm']:.3f} / {v['max_peak_torque_Nm']:.3f} | {100*v['overload_episode_rate']:.2f}% | {v['mean_first_stable_success_seconds']} | {v['failure_reasons']} |\n"
if 'success_difference' in decision:
    text+=f"\nA2−A1：{100*decision['success_difference']:.2f} 个百分点；配对 seed 区间 {decision['paired_seed_bootstrap_interval']}（比例单位），精确不一致对检验 p={decision['exact_discordant_p']:.6g}；第 {decision['look']}/2 次判断。\n"
text+='\n## 六项问题\n\n'
text+=f"1. 直接 Q 额外价值：{decision['status']}；以 A1/A2 匹配对照判定，不能用旧 A0/A2 差异替代。\n"
text+='2. 六维力额外价值：NOT_RUN，当前阶段所有组含力，尚无独立无力对照。\n'
text+='3. 配对校准额外价值：NOT_RUN；历史 D0 失败保留，不能制造同状态标签。\n'
text+=f"4. 支持/不支持：历史模型和数据哈希审计已保存，首次更新重现状态 {audit.get('status','NOT_RUN')}；这仅支持实现/匹配检查，不等于任务收益。\n"
text+='5. 多种子和泛化：当前未获准；须依次通过力觉、可信配对校准和确认门槛。\n'
text+='6. 未运行原因：第二阶段等待第一阶段 GO；第三阶段还受回放可信性及历史未落盘预算尾部约束；第四、五阶段等待前序证据。\n'
text+='\n配置、模型 SHA、批次次序、训练/评估命令、退出码、原始 CSV、逐物理步日志和阶段报告分别保存。停止不等于停止交付；失败和 INCONCLUSIVE 同样上传。\n'
text+=f"\n实现证据：200 次 A2 更新的 Q 梯度范数范围 {gradients['q_gradient_norm_range']}，Q/BC 范数比中位数 {gradients['q_to_bc_gradient_norm_median']:.4f}；负内积比例 {gradients['q_bc_negative_dot_fraction']:.3f}。每次均有非零实际 ACT 动作变化。这不证明 Q 方向正确；历史 A2 未记录步长项完整梯度，不能仅用范数断言没有抵消，也不以此擅自增加训练。\n"
text+=f"\n匹配审计：{matching['batches_verified']}/200 批次一致，{matching['replay_train_rows']} 条训练 replay、{matching['demonstration_episodes']} 个示教回合、{matching['demonstration_windows']} 个窗口；优化器设置、归一化和冻结视觉编码器一致。A2−A1 参数差 L2={matching['parameter_changes']['A2_vs_A1']['l2']:.6f}，两种更新确实得到不同 ACT。历史 A2 未保存最终 RNG；首次更新数值精确复现，后续采样次序与随机调用数量由源代码和全批次记录核对。\n"
if objective_probe.get('status')=='PASSED':
    norms=objective_probe['weighted_parameter_gradient_norm']
    text+=f"\n最终 A2 的固定训练批次只读检查：Q/BC/步长梯度范数为 {norms['q']:.4f}/{norms['bc']:.4f}/{norms['step']:.4f}，Q 与步长梯度余弦 {objective_probe['q_step_cosine']:.4f}；合梯度在 Q 方向的投影保留 {100*objective_probe['total_gradient_projection_onto_q_relative_to_q']:.2f}%。约束明显抵消部分 Q 梯度，但没有完全清零。这是一个最终状态的局部梯度检查，不是历史全部更新重现，也不等同于 Adam 实际步向或物理改善；没有新增优化步、仿真或权重调节。真实效果仍只由 episode CSV 判定。\n"
text+='\n## 本轮执行预算与返回码\n\n'
text+='| 批次 | OS 返回码 | 初始化/重置控制步等价数 | 评估控制步 | 总数（含失败尝试） |\n|---|---:|---:|---:|---:|\n'
total=0
for name,record in summary['execution_manifest'].get('new_evaluations',{}).items():
    counts=(record.get('accounting') or {}).get('individual_control_step_equivalents',{})
    amount=sum(counts.values());total+=amount
    text+=f"| {name} | {record['returncode']} | {counts.get('restore','UNKNOWN')} | {counts.get('evaluation','UNKNOWN')} | {amount if counts else 'UNKNOWN'} |\n"
text+=f'\n已结束新批次合计 {total} 个 individual control-step equivalents；仍运行的批次见实时物理步日志。历史复用回合不重复收费。A1 新增 200 次优化；另有 1 次预注册 A2 首步实现复算，模型丢弃且从未部署。没有新增 Critic 训练。\n'
text+='\nA1 训练的早期 detached 启动未保留 OS 退出码（null），不补造 0；完成依据是完整 200 行训练记录、最终 checkpoint 和匹配审计。所有本轮仿真由管理进程保存实际退出码。旧 D0 原生崩溃有未知尾部，仍保留原预算限制，不把未知交互计为零。\n'
text+=f'\n逐物理步证据：已验证并归档 {len(physics_archives)} 个完整批次的 before/after 日志，见 `physics_ledger_archives.json` 及同目录 `.physics.jsonl.gz`；可用 `gzip -dc 文件名.physics.jsonl.gz` 查看。归档仅含时间戳和阶段计数，不含图像、原始传感器数组或模型权重。\n'
(report/'FINAL_REPORT.md').write_text(text)
print(decision['status'],decision.get('extension_allowed'))
