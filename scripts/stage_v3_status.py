"""Read-only staged progress; counts episode rows and completed physics, never heartbeats."""
import csv
import json
import os
from pathlib import Path
import time
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/stage_v3_20261026'
RESULT=ROOT/'results/stage_v3_20261026'

def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}

def alive(pid):
    try:return Path(f'/proc/{int(pid)}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
    except (OSError,ValueError,TypeError):return False

def count(name):
    try:
        with (RESULT/(name+'.csv')).open() as f:return len(list(csv.DictReader(f)))
    except OSError:return 0

def display():
    job=read(REPORT/'job.json');training=read(REPORT/'A1_training.json')
    progress=read(REPORT/job.get('phase','none')/'diagnostic_live_progress.json')
    decision=read(REPORT/'phase1_decision.json')
    phase=job.get('phase','尚未启动');current=count(phase)
    last=progress.get('last_physics_step');age=time.time()-last if last else None
    counts=progress.get('accounting',{}).get('individual_control_step_equivalents',{})
    lines=[f'更新时间 UTC：{datetime.now(timezone.utc).isoformat()}',
        f"新协议 v3 第一阶段：{job.get('status','NOT_RUN')} / {phase}",
        f"管理进程存活：{alive(job.get('pid'))}；仿真子进程存活：{alive(job.get('child_pid'))}",
        '1 ACT0：复用已审计 10000 次更新 checkpoint；不重新预训练',
        f"2 A1 BC＋步长约束：{training.get('actor_updates',0)}/200 更新；A2 Q＋相同 BC/步长：历史 200/200 复用",
        f"3 A2 首步复算审计：{read(REPORT/'a2_reproduction_audit.json').get('status','NOT_RUN')}（实现检查，不是成功率）",
        f"4 初始真实评估：A1 {count('eval_A1_initial')}/32；A2 历史 32/32 复用",
        f"5 仅一次条件追加：A1 {count('eval_A1_extension')}/32；A2 {count('eval_A2_extension')}/32",
        '  只在初始结果 INCONCLUSIVE 且实现审计通过时追加；不增加训练',
        f"阶段判断：{decision.get('status','NOT_RUN')}；判断次数 {decision.get('look',0)}/2",
        '6 第二阶段（无力/六维力）：NOT_RUN，要求第一阶段 GO',
        '7 第三阶段（配对校准）：NOT_RUN，还须力觉证据及 D0 可信回放；旧失败保留',
        '8 多种子 / 泛化：NOT_RUN，要求前序证据通过']
    if job.get('status')=='RUNNING':
        lines.append(f'当前批次真实完成 {current}/32；已计控制步 {sum(counts.values())}/10000（含初始化/重置）')
        lines.append('距最近已完成物理步：'+(f'{age:.1f} 秒' if age is not None else '尚未确认'))
        if current>=3 and age is not None and age<180 and job.get('started'):
            per_episode=(time.time()-job['started'])/current
            minutes=per_episode*(32-current)/60
            lines.append(f'按当前速度，本批剩余约 {minutes*.75:.0f}–{minutes*1.25:.0f} 分钟；条件追加每批约 {per_episode*32/60:.0f} 分钟')
        else:lines.append('完成时间：有效回合不足或缺少新物理步，暂不估算')
        if not alive(job.get('pid')) or not alive(job.get('child_pid')):lines.append('注意：运行状态与进程不一致，请检查日志，不能视为仍在执行')
        lines.append('守护：启动上限 5 分钟；连续 3 分钟无完成物理步则抓栈并停止，保留计数')
    if job.get('error'):lines.append('失败原因：'+job['error'])
    if job.get('log'):lines.append('执行日志：'+job['log'])
    publication=read(ROOT/'.cache/stage_v3_publication.json')
    lines.append(f"最新上传：{publication.get('time_utc','尚无')}；退出码 {publication.get('returncode')}；{publication.get('head','')[:8]}")
    return '\n'.join(lines)

if __name__=='__main__':print(display())
