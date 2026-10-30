"""Summarize saved real FORGE probes without advancing the simulator."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

root = Path('reports/replay_root_cause_20261027')
raw = Path('checkpoints/replay_root_cause_20261027')
thresholds = json.loads(Path('reports/20261023T104759Z/protocol_effect_thresholds.json').read_text())['fidelity_tolerances']


def load(label, name='original'):
    # These are locally produced audit files, never user-supplied pickle input.
    return torch.load(raw/label/(name+'.pt'), map_location='cpu', weights_only=False)


def maximum(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return float(np.abs(a.astype(float)-b.astype(float)).max()) if a.size else 0.


def compare(a,b):
    rows=list(zip(a['rows'],b['rows']))
    errors={k:max(maximum(x['physical'][k],y['physical'][k]) for x,y in rows)
            for k in a['rows'][0]['physical']}
    errors.update(force_N=max(maximum(x['substep_wrench'][...,:3],y['substep_wrench'][...,:3]) for x,y in rows),
                  torque_Nm=max(maximum(x['substep_wrench'][...,3:],y['substep_wrench'][...,3:]) for x,y in rows),
                  controller=max(maximum(x['controller'],y['controller']) for x,y in rows))
    equal_length=len(a['rows'])==len(b['rows'])
    return dict(original_steps=len(a['rows']),replayed_steps=len(b['rows']),errors=errors,
                first_command_torque_error=maximum(a['traces'][0]['before']['robot/target_effort'],
                                                   b['traces'][0]['before']['robot/target_effort']),
                first_physics_step_errors={k:maximum(v,b['traces'][0]['after'][k])
                                           for k,v in a['traces'][0]['after'].items()},
                all_measured_physical_values_equal=equal_length and not any(errors.values()),
                physical_tolerances_passed=equal_length and all(v<=thresholds[k] for k,v in errors.items()),
                scope='Fixed 16-step command witness only, not full-horizon scientific D0')


baseline=load('12_identity_none')
comparisons=[]
for path in sorted(root.glob('*identity*/result.json')):
    label=path.parent.name
    if int(label.split('_')[0])<12: continue
    d=json.loads(path.read_text())
    if d['status']!='MEASURED': continue
    b=load(label)
    comparisons.append(dict(label=label,mutation=d['mutation'],**compare(baseline,b),
                            before_errors={k:maximum(v,b['before'][k]) for k,v in baseline['before'].items()},
                            report_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
rollbacks=[]
for label in ['19_rollback_minimal_writes','22_rollback_layout_fixed']:
    if not (root/label/'result.json').exists(): continue
    d=json.loads((root/label/'result.json').read_text())
    if d['status']!='MEASURED': continue
    for p in sorted((raw/label).glob('restore_*.pt')):
        rollbacks.append(dict(label=label,repeat=p.stem,**compare(load(label),load(label,p.stem))))
charges=[]
for path in sorted(root.glob('*.command.json')):
    d=json.loads(path.read_text());account=d.get('accounting',{})
    exact=d.get('status')=='FINISHED' and account.get('status')=='CHARGED_STEPS_RECOVERED'
    charge=sum(account['physics_substeps'].values())/8 if exact else d['reserved_control_steps']
    charges.append(dict(label=path.stem,charge=charge,reservation_retained=not exact))
evidence=[]
for path in sorted(raw.glob('*/*.pt')):
    if int(path.parent.name.split('_')[0])>=12:
        evidence.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size))
summary=dict(comparisons=comparisons,rollbacks=rollbacks,thresholds=thresholds,
             budget=dict(cap=5000,charged_or_reserved=sum(x['charge'] for x in charges),runs=charges),
             raw_evidence=evidence)
(root/'restore_components_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for c in comparisons+rollbacks:
    print(c['label'],c.get('repeat',''),json.dumps({k:c['errors'][k] for k in ('position_m','force_N','torque_Nm')}),
          'first torque',c['first_command_torque_error'],'physical pass',c['physical_tolerances_passed'])
print('charged or reserved:',summary['budget']['charged_or_reserved'])

table='\n'.join(f"| {c['label']} | {c['errors']['position_m']*1000:.6f} | {c['errors']['force_N']:.6f} | "
                f"{c['first_command_torque_error']:.9g} | {'PASSED' if c['physical_tolerances_passed'] else 'FAILED'} |"
                for c in comparisons)
rollback_table='\n'.join(f"| {c['repeat']} | {c['errors']['position_m']*1000:.6f} | {c['errors']['force_N']:.6f} | "
                         f"{c['first_command_torque_error']:.9g} | FAILED |"
                         for c in rollbacks if c['label']=='22_rollback_layout_fixed')
report=f'''# 接触恢复根因定位与修复验收

用户追问“是否是仿真搭建问题”后的独立组件排查，2026-10-27。数值表由本地真实 FORGE 审计张量自动生成；历史失败不覆盖。

结论：确实存在我们回放实现中的错误，并非仅仅数据要重采。已修复两个实现问题，原地保存/恢复的物理轨迹已完全一致；但真正从未来回退到旧接触状态仍失败。现有公开状态接口不是完整接触检查点。原始示教采集没有经过这段快照代码，不能据此判定原六维力无效。

## 1. 确认的实现根因及因果对照

**质量矩阵的内存布局丢失。** FORGE 控制器使用 9×9 质量矩阵的 7×7 切片，stride 为 `(81,9,1)`。旧快照和恢复使用普通 `clone()`，得到 `(49,7,1)`。所有矩阵元素仍相等，但真实 GPU 控制计算的第一条力矩差为 1.448392868×10⁻⁵ N·m。这发生在第一次物理推进前，不能归因于推进后的接触噪声。

实验 18 已跳过全部物理状态写入，仍出现该误差；实验 20 只复制质量矩阵，就重现相同的物理轨迹差异（16 步最大位置差约 0.370 mm，力差约 3.139 N）。这隔离了张量布局变化对控制输出的影响；没有进一步声称已经识别具体 CUDA 内核内部实现。

**不必要的物理状态写回。** 旧恢复对所有资产调用 `reset_to`，即使某些状态根本未变。实验 15/16/17 分别把当前关节位置、被夹物体位姿、关节速度原样写回，后续接触轨迹均改变。关节位置/速度写回还会清空当前 incoming-joint-force 查询缓冲；这只是恢复接口的即时副作用，不能将其解释为旧数据里的物理力突然变成零。后续真实物理步的载荷差异另行测量。

| 对照 | 16 步内最大位置差 mm | 最大力差 N | 首条控制力矩差 N·m | 原物理容差 |
|---|---:|---:|---:|---|
{table}

`12` 是不作修改的新进程基准；`14` 只刷新运动学；`13` 是旧版完整原地恢复；`18` 只修复不必要写回；`20` 单独注入质量矩阵 clone；`21` 同时修复状态写回和张量布局。所有实验使用相同 development seed 60000、相同 8 步前缀和相同后续动作。`21` 的全部实测位置、旋转、关节、速度、六维力和控制器字段最大误差均为 **0**。不将此 16 步固定指令见证当作完整视觉闭环或科学 D0 验收。

## 2. 已落地的修复

- `clone_tensor_layout` 在快照保存与恢复两端保留 tensor stride，包括广播轴；回归测试检查切片矩阵、转置、广播、标量和空数组。
- `write_changed_scene_state` 仅写入确实变化的状态，避免重设未移动资产和错误替换 PD 目标；物理属性也只在变化时写入。
- 旧快照缺少布局版本时直接拒绝，要求从原前缀重新生成；不把已丢失的信息猜测补回。
- 快照显式标记为公开场景/控制器状态；恢复回执保留 `contact_state_restored=false` 和必须通过 fidelity admission 的要求。

这些修改限于诊断恢复流程。没有修改示教数组、动作语义、奖励、物理参数或容差，也没有重新采集或训练。

## 3. 真正回退后的剩余障碍

修复后实验 22 从原状态继续执行，再回退到旧快照，做三次重复。第一条控制力矩均精确一致；差异在物理推进后仍出现。第一重复的首个物理子步关节位置最大差约 0.000111878 rad。

| 恢复重复 | 最大位置差 mm | 最大力差 N | 首条控制力矩差 N·m | 原物理容差 |
|---|---:|---:|---:|---|
{rollback_table}

因此不能声称任意接触状态回退已经修好。**软件复制错误已修复；公开状态回写仍不能等价重建接触过程。** 对究竟是哪一项引擎内部缓存/状态占主导，仍为 INCONCLUSIVE；本次没有接触到可完整导出/导入内部接触历史的接口。

[对应 Omni Physics 107.3 的官方限制文档](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/current_limitations.html) 说明，接触中的仿真从保存状态续跑可能不确定，因为部分内部接触状态跨步保留且不能从 USD 恢复；官方给出的路径是从头启动仿真。本项目 `get_state/reset_to` 同样只暴露公开位姿/速度等量，不能据此推断恢复了这些内部量。文档支持能力边界，本机组件对照支持实际失败位置；它们不构成对内部具体缓存的唯一归因。

## 4. 对仿真搭建和六维力的判断

实际配置保持 stock：120 Hz、每控制步 8 个物理子步、TGS、最大位置迭代 192、最大速度迭代 1、GPU partition 1。运行时检查机器人/被夹物/固定物的 fixed 和 spatial tendon 数均为 0。因此不能把官方“tendon 加非零 TGS 速度迭代导致力异常”的条目直接套到本任务。机械臂关节的刚度/阻尼为 0，夹指为 7500/173；与 stock 力矩控制配置相符。完整只读信息在 `16_identity_held_pose/result.json` 的 setup_audit。

[PhysX 5.5.1 incoming-joint-force 文档](https://nvidia-omniverse.github.io/PhysX/physx/5.5.1/docs/Articulations.html#link-incoming-joint-force) 定义的是关节参考系中父链接传给子链接的六维载荷，按最后一个物理步平均。前次本机审计已核对参考系和因果平滑。它是有明确物理语义的仿真信号；可重复不等于真实硬件标定，也不能把恢复后不等价解释为所有采集力都失真。

当前没有证据要求删除整批示教，也没有证据证明仿真已经满足所有真实精密操作精度要求。改变精度、摩擦或时间步需要新的物理验收，不能为了回放通过随意调参。

## 5. 当前可用路径、停止点和预算

已有数据继续保留。原采集轨迹使用独立新进程、原批量/初始化、完整动作前缀及原 RNG 调用顺序重建；上一轮 16 回合、2,384 帧的非视觉精确重建证据仍有效。旧数据不被冒充任意接触检查点。新数据若需要配对分支，必须先通过实际所选回放后端的原容差验收；不能先大量重采再假设问题消失。

任意接触快照回退：**FAILED / NO_GO**；原地保存恢复：**PASSED（固定见证范围）**；RGB 精确回放：上一轮 FAILED，本轮未修改 renderer；配对校准和新训练：**NOT_RUN**。本次不扩大科学实验、不改历史 D0 结果。

本轮新增计费 {summary['budget']['charged_or_reserved']-4275.5:.3f} 个环境控制步当量。累计计费或保留 **{summary['budget']['charged_or_reserved']:.3f}/5000**，含初始化/reset、前缀、恢复、分支以及第一次失败尝试的 500 步保守预留。未扩预算。历史 native crash 的未知尾部仍独立保留。所有新模拟器命令退出码为 0 且结果 MEASURED；不把 MEASURED 当保真通过。

软件验证：`PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q`，**87 passed in 2.01s**。仅重算本报告：`.venv/bin/python scripts/analyze_restore_components.py`，不推进仿真。逐运行命令、日志、误差与 journal 均保存；原始审计张量留本机 checkpoints，SHA256 见 `restore_components_summary.json`。

## 文档检索边界

本次仅依赖官方资料和本机运行。搜索覆盖 official/general-web 与 GitHub，英文两条查询；未使用社区帖子作为结论。107.3 限制页标注更新于 2026-01-14，与本机扩展 107.3 系列匹配；5.5.1 力接口文档与既有锁定审计一致。新版 5.8 文档仅作线索，不用于断言本机支持新功能。没有未经执行的修复建议冒充成功。

> **Evidence boundary:** These GitHub, Q&A, community, and general-web results are for debugging and discovery only. They are not paper-citation evidence and must not be added to the bibliography or used alone to support a research claim. Use the project's literature and citation-verification workflow for that purpose.
'''
(root/'RESTORE_ROOT_CAUSE.md').write_text(report)
