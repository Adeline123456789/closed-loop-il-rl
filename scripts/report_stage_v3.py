"""Generate staged conclusions only from measured CSV/JSON."""
import json
from pathlib import Path
from forcecritic.staged.common import locked_plan,CONFIG
from forcecritic.staged.results import read_rows,arm_metrics,matched_decision
from forcecritic.runtime import atomic_json,digest

p=locked_plan();report=Path(p['report']);result=Path(p['results'])
job=json.loads((report/'job.json').read_text()) if (report/'job.json').exists() else {}
audit=json.loads((report/'a2_reproduction_audit.json').read_text()) if (report/'a2_reproduction_audit.json').exists() else {}
a0=read_rows(p['historical_A0_csv']);a1=read_rows(result/'eval_A1_initial.csv');a2=read_rows(p['historical_A2_csv'])
extension1=read_rows(result/'eval_A1_extension.csv');extension2=read_rows(result/'eval_A2_extension.csv')
completed=lambda name: (report/(name+'.supervisor.json')).exists() and json.loads((report/(name+'.supervisor.json')).read_text()).get('returncode')==0
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
if job.get('status') in ('FAILED_IMPLEMENTATION','BLOCKED'):
    decision=dict(status=job['status'],reason=job.get('error'),extension_allowed=False)
summary=dict(protocol=p['protocol'],config_sha=digest(CONFIG),phase1=decision,arms=arms,
    extension_progress=dict(A1=len(extension1),A2=len(extension2)),
    phase2=dict(status='NOT_RUN',reason='Requires phase1 GO; historical arms include force and cannot stand in for no-force B2'),
    phase3=dict(status='NOT_RUN',reason='Requires phase1/phase2 support and trustworthy replay; historical D0 NO_GO retained'),
    phase4=dict(status='NOT_RUN',reason='Requires phase3 GO'),phase5=dict(status='NOT_RUN',reason='Requires confirmation GO'),
    provenance=dict(historical_A2_already_observed=True,a2_first_update_audit=audit.get('status','NOT_RUN')))
atomic_json(report/'phase1_decision.json',decision);atomic_json(report/'stage_results.json',summary)
project_path=Path('reports/PROJECT_STATUS.json')
project=json.loads(project_path.read_text())
project['active_protocol']=p['protocol']
project['active_stage_report']=str(report/'stage_results.json')
project['staged_v3']={k:summary[k] for k in ('phase1','phase2','phase3','phase4','phase5')}
project['staged_v3_execution']=job
atomic_json(project_path,project)
text='# 分阶段实验 v3：实测报告\n\n'
text+=f"第一阶段：**{decision['status']}**。{decision['reason']}。\n\n"
text+='A0/A2 历史测试结果在本计划前已见；本轮补匹配 A1。新追加只按预锁规则执行，不追溯声称历史结果是新盲测。第一阶段两组都有力历史，不能直接作为第二阶段无力组。\n\n'
text+='| 组 | 已记录回合 | 稳定成功 | 成功率 | 95% Wilson 区间 | 平均回报 | 平均代价 | 保护停止率 |\n|---|---:|---:|---:|---|---:|---:|---:|\n'
for name,v in arms.items():
    if not v['episodes']:
        text+=f'| {name} | 0 | NOT_RUN | — | — | — | — | — |\n';continue
    low,high=v['wilson95']
    text+=f"| {name} | {v['episodes']} | {v['successes']} | {100*v['success_rate']:.2f}% | {100*low:.1f}–{100*high:.1f}% | {v['mean_return']:.4f} | {v['mean_cost']:.4f} | {100*v['protection_stop_rate']:.2f}% |\n"
text+='\n未完成批次仅为进度，不作为最终成功率或阶段判断。完整峰值力/力矩、超限、失败类型和成功时间在 `stage_results.json`；原始回合在同名 results 目录。A0 只有历史 32 回合，不与追加后的 64 回合冒充严格同样本比较。\n'
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
(report/'FINAL_REPORT.md').write_text(text)
print(decision['status'],decision.get('extension_allowed'))
