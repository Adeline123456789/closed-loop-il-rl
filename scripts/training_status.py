"""Live, read-only training view; optional background heartbeat and loss export."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = '20261023T104759Z'
REPORT = ROOT / 'reports' / RUN
RESULT = ROOT / 'results' / RUN
CHECKPOINT = ROOT / 'checkpoints' / RUN


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def atomic(path, data):
    temp = path.with_suffix('.monitor.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def alive(pid):
    try:
        return Path(f'/proc/{int(pid)}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except (OSError, ValueError, TypeError):
        return False


def rows(name):
    path = RESULT / name
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def detailed_progress(stage):
    baseline=read(REPORT/'baseline_training.json')
    live=read(REPORT/'baseline_live_progress.json')
    initial=read(CHECKPOINT/'fidelity_initial/fidelity.json')
    additional=read(CHECKPOINT/'fidelity_additional/fidelity.json')
    initial_n=len(initial.get('checkpoints',[]));extra_n=len(additional.get('checkpoints',[]))
    screening=len(rows('paired_screening.csv'));confirmation=len(rows('paired_confirmation.csv'))
    transfer=4 if (CHECKPOINT/'d2_transfer.pt').exists() else 0
    forgetting=sum(len(rows(f'd2_{arm}_normal_validation.csv')) for arm in ('baseline','q_only','q_bc_step'))
    seeds=sum(int(r['stable_success']) for r in rows('act0_validation.csv'))
    if stage=='diagnostic':
        if (REPORT/'d2_transfer.json').exists():detail='D2 已测量，正在判断是否允许最小消融'
        elif transfer:detail='D2：已完成4条更新对照轨迹，正在检查原有成功行为是否退化'
        elif (REPORT/'paired_confirmation.json').exists() or confirmation:detail='D1：对初筛候选做独立重复确认'
        elif screening:detail='D1：固定24个检查点，比较原动作和正/负扰动动作'
        elif extra_n or (CHECKPOINT/'fidelity_additional').exists():detail=f'D0 扩展检查：当前第 {min(extra_n+1,16)}/16 个点'
        elif read(REPORT/'d0_initial.json').get('status')=='PASSED':detail='初始D0已通过，正在采样额外16个检查点'
        elif (CHECKPOINT/'fidelity_initial').exists():detail=f'D0：初始8个检查点已选齐；当前第 {min(initial_n+1,8)}/8 个点，整点完成 {initial_n}/8'
        else:detail='D0：采样初始8个检查点（4个困难、4个正常）；尚无采样计数文件'
    elif stage=='short-ablation':
        current=read(REPORT/'short_live_progress.json').get('current',{})
        detail='最小消融：'+str(current.get('arm','准备'))+' / '+{'critic':'训练Critic','actor':'更新ACT','evaluation':'真实仿真评估','prepare':'加载相同起点和数据'}.get(current.get('phase'),'等待首次遥测')
    else:detail='诊断与消融按前置门槛推进；没有结果的项目不计为完成'
    return dict(shared_episodes=baseline.get('real_complete_episodes',live.get('completed_episodes',0)),
        shared_steps=baseline.get('accounting',{}).get('individual_control_step_equivalents',{}).get('interaction',live.get('individual_transitions',0)),
        critic_updates=baseline.get('critic_updates',live.get('completed_updates',0) if live.get('phase')=='critic_training' else 0),
        initial_points=initial_n,extra_points=extra_n,screening_runs=screening,confirmation_runs=confirmation,
        transfer_runs=transfer,forgetting_runs=forgetting,forgetting_target=3*seeds,
        arms=read(REPORT/'short_live_progress.json').get('arms',{}),diagnostic_detail=detail)


def snapshot():
    job = read(REPORT / 'background_job.json')
    progress = read(CHECKPOINT / 'act0.json')
    stages = read(REPORT / 'run_manifest.json').get('stages', {})
    live = read(REPORT / 'baseline_live_progress.json')
    running = alive(job.get('pid'))
    done = progress.get('completed_updates', 0)
    target = progress.get('requested_updates', 10000)
    evaluations = rows('act0_validation.csv')
    phase = job.get('stage', 'UNKNOWN')
    if phase == 'baseline' and job.get('status') == 'RUNNING' and running:
        if done < target:
            phase = 'ACT 示教预训练'
        elif len(evaluations) < 32:
            phase = 'ACT0 真实仿真评估'
        elif not (CHECKPOINT / 'shared_replay.pt').exists():
            phase = '共享仿真交互 / 基线门槛检查'
        elif not (CHECKPOINT / 'critic0.pt').exists():
            phase = '普通 Critic 拟合 / 数据范围门槛检查（循环未输出逐步计数）'
        else:
            phase = '基线阶段收尾'
    elif job.get('status') == 'RUNNING' and not running:
        phase = '管理进程已退出；旧 RUNNING 状态不能证明仍在训练'
    queue = {}
    order = ('baseline', 'diagnostic', 'short-ablation')
    for stage in order:
        event = stages.get(stage, {})
        status = event.get('status', 'NOT_RUN')
        if job.get('stage') == stage and job.get('status') == 'RUNNING' and running:
            status = 'RUNNING'
        elif (status == 'NOT_RUN' and running and job.get('status') == 'RUNNING'
              and job.get('stage') in order and order.index(stage)>order.index(job['stage'])):
            status = 'WAITING_GATE'
        queue[stage] = status
    # Heartbeats prove only that this observer is alive. Track producer evidence
    # independently, so a quiet/stuck simulator cannot masquerade as progress.
    stage = job.get('stage', 'baseline')
    logs = sorted(REPORT.glob(f'{stage}_*.log'), key=lambda p:p.stat().st_mtime)
    evidence = [p for p in (CHECKPOINT/'act0.json', REPORT/'baseline_validation.json',REPORT/'baseline_live_progress.json',
                           REPORT/'baseline_training.json',REPORT/'short_live_progress.json',
                           CHECKPOINT/'fidelity_initial/fidelity.json',CHECKPOINT/'fidelity_additional/fidelity.json',CHECKPOINT/'shared_replay.pt',
                           CHECKPOINT/'critic0.pt') if p.exists()]
    evidence += [p for p in RESULT.glob('*.csv')
                 if p.name not in ('act_pretraining_loss.csv','recollection_episodes.csv','summary.csv','paired_comparisons.csv')]
    if logs:evidence.append(logs[-1])
    latest = max(evidence, key=lambda p:p.stat().st_mtime) if evidence else None
    evidence_age = round(time.time()-latest.stat().st_mtime,1) if latest else None
    warning = None
    if running and job.get('status')=='RUNNING' and evidence_age is not None and evidence_age>300:
        warning=f'已 {evidence_age/60:.1f} 分钟没有新的可核实执行记录；疑似停滞或缺少阶段内遥测，不能估算完成时间。'
    if (stage=='baseline' and running and job.get('status')=='RUNNING'
            and len(evaluations)==32 and not (CHECKPOINT/'shared_replay.pt').exists() and logs):
        text=logs[-1].read_text(errors='replace')
        tail=text.rsplit('[INFO]: Parsing configuration from:',1)[-1]
        if 'Setting seed: 0' in tail and '[INFO]: Base environment:' not in tail:
            phase='切换到 8 环境：初始化完成尚未得到日志确认'
    if stage=='baseline' and running and job.get('status')=='RUNNING' and live:
        phase={'act_pretraining':'ACT 权重加载 / 预训练恢复',
               'act_validation':'ACT0 真实仿真评估',
               'shared_environment_initialization':'独立进程：8 环境初始化',
               'shared_collection':'共享真实仿真交互',
               'critic_training':'在线 Critic 训练'}.get(live.get('phase'),phase)
    return dict(time_utc=datetime.now(timezone.utc).isoformat(), phase=phase,
                supervisor_alive=running, supervisor_status=job.get('status'),
                act_updates=done, act_target=target, act_percent=round(100*done/target, 1),
                checkpoint_age_seconds=round(time.time()-(CHECKPOINT/'act0.json').stat().st_mtime,1)
                    if (CHECKPOINT/'act0.json').exists() else None,
                real_baseline_episodes=len(evaluations),
                real_baseline_successes=sum(str(r.get('stable_success','')).lower() in ('1','true') for r in evaluations),
                queue=queue, stop_reason=stages.get(job.get('stage'),{}).get('reason')
                    if job.get('status') in ('STOPPED','FAILED') else None,
                diagnostic_rows={n:len(rows(n+'.csv')) for n in ('fidelity','paired_screening','paired_confirmation','actor_step')},
                ablation_episodes={a:len(rows(f'short_{a}_validation.csv')) for a in ('B0','B1','B2','B3')},
                milestones=[p.name for p in sorted((CHECKPOINT/'act0_milestones').glob('*.pt'))],
                publication=read(ROOT/'.cache/background_publication.json'),
                execution_evidence_age_seconds=evidence_age, progress_warning=warning,
                latest_execution_evidence=str(latest.relative_to(ROOT)) if latest else None,
                latest_stage_log=str(logs[-1].relative_to(ROOT)) if logs else None,
                baseline_live=live,detail=detailed_progress(stage))


def display(data):
    view=data['detail']
    phase_name={'diagnostic':'可信性诊断（D0 / D1 / D2）','short-ablation':'最小消融（B0 / B1 / B2 / B3）'}.get(data['phase'],data['phase'])
    translated={'RUNNING':'进行中','PASSED':'通过','WAITING_GATE':'等待前置门槛',
        'NOT_RUN':'尚未开始','FAILED_IMPLEMENTATION':'执行失败','INCONCLUSIVE':'证据不足',
        'NO_GO_REPLAY':'回放门槛未通过','NO_GO_CALIBRATION':'不进入校准'}
    lines=[f"更新时间 UTC：{data['time_utc']}",
        f"当前：{phase_name}；管理进程存活：{data['supervisor_alive']}",
        '序号  阶段 / 已完成数量 / 状态',
        f" 1    ACT 示教预训练：{data['act_updates']}/{data['act_target']} 次更新",
        f" 2    ACT0 仿真评估：{data['real_baseline_episodes']}/32 回合；成功 {data['real_baseline_successes']} 回合",
        f" 3    共享真实交互：{view['shared_episodes']} 回合（每批8、上限100）；{view['shared_steps']} 控制步",
        f" 4    普通 Critic：{view['critic_updates']}/10000 次更新",
        f" 5    D0 初始回放检查：{view['initial_points']}/8 点（已落盘 {21*view['initial_points']}/168 条轨迹）",
        f" 6    D0 扩展回放检查：{view['extra_points']}/16 点（已落盘 {21*view['extra_points']}/336 条轨迹）",
        f" 7    D1 动作配对初筛：{view['screening_runs']}/648 条真实轨迹完成",
        f" 8    D1 独立确认：{view['confirmation_runs']} 条轨迹；最多192，实际目标由初筛决定",
        f" 9    D2 ACT 更新验证：{view['transfer_runs']}/4 条对照轨迹；遗忘评估 {view['forgetting_runs']}/{view['forgetting_target']}",
        '10    最小消融（Critic更新 / ACT更新 / 真实评估）：']
    lines.append(f"      B0 固定ACT：无需更新 / 无需更新 / {data['ablation_episodes']['B0']}/32")
    for arm in ('B1','B2','B3'):
        counts=view['arms'].get(arm,{})
        lines.append(f"      {arm}：{counts.get('critic_completed',0)}/2000 / {counts.get('actor_completed',0)}/200 / {data['ablation_episodes'][arm]}/32")
    lines.append('当前子步骤：'+view['diagnostic_detail'])
    if data['queue'].get('diagnostic')=='RUNNING':
        lines.append('D0 每个点含21条轨迹；当前运行到整点结束才落盘，点内精确计数未提供。')
    lines.append('状态：'+' → '.join(f"{name}({translated.get(data['queue'][key],data['queue'][key])})"
        for key,name in [('baseline','基线'),('diagnostic','可信性诊断'),('short-ablation','最小消融')]))
    lines.append(f"日志/产物更新距今 {data['execution_evidence_age_seconds']} 秒；有日志不等于新增成功结果。")
    if data.get('progress_warning'):lines.append('注意：'+data['progress_warning'])
    if data.get('stop_reason'):lines.append('停止原因：'+data['stop_reason'])
    publication=data['publication']
    lines.append(f"上传：{publication.get('time_utc','无')}；退出码 {publication.get('returncode')}；{publication.get('head','')[:8]}")
    lines.append('D0=回放是否可信；D1=动作改动是否有真实效果；D2=Critic能否有效更新ACT。')
    lines.append('B1=直接Q更新；B2=加可信状态筛选；B3=再加配对价值差校准。详见 EXPERIMENT_PROGRESS.md')
    return '\n'.join(lines)


def export_loss():
    import torch
    pinned = CHECKPOINT / '.progress_snapshot.pt'
    pinned.unlink(missing_ok=True)
    try:
        os.link(CHECKPOINT/'act0.pt', pinned)
        data = torch.load(pinned, map_location='cpu', weights_only=False, mmap=True)
        history = data['history']
        target = RESULT / 'act_pretraining_loss.csv'
        temp = target.with_suffix('.monitor.tmp')
        with temp.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=['update','bc_cvae_loss'])
            writer.writeheader();writer.writerows(history)
        temp.replace(target)
        return dict(update=data['update'], last=history[-1]['bc_cvae_loss'],
                    mean_last_100=sum(r['bc_cvae_loss'] for r in history[-100:])/len(history[-100:])) if history else {}
    finally:
        pinned.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--daemon',action='store_true')
    parser.add_argument('--interval',type=float,default=30)
    args=parser.parse_args()
    if not args.daemon:
        data=snapshot()
        cached=read(REPORT/'training_progress.json')
        if cached.get('loss'):
            data['loss']=cached['loss']
        if cached.get('act_updates')==data['act_updates'] and 'pretraining_eta_minutes' in cached:
            data['pretraining_eta_minutes']=cached['pretraining_eta_minutes']
        print(display(data));return
    if args.interval<1:parser.error('interval must be >= 1 second')
    lock=(ROOT/'.cache/training_monitor.lock').open('w')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('Monitor already running.');return
    previous=None;loss={};rate=None
    while True:
        data=snapshot()
        if (CHECKPOINT/'act0.pt').exists() and loss.get('update')!=data['act_updates']:
            try:
                loss=export_loss()
            except Exception as error:
                data['loss_read_error']=repr(error)
        data['loss']=loss
        stamp=(CHECKPOINT/'act0.json').stat().st_mtime if (CHECKPOINT/'act0.json').exists() else None
        if previous and stamp and data['act_updates']>previous[0] and stamp>previous[1]:
            rate=(data['act_updates']-previous[0])/(stamp-previous[1])
        if stamp:previous=(data['act_updates'],stamp)
        if rate and data['act_updates']<data['act_target']:
            data['pretraining_eta_minutes']=(data['act_target']-data['act_updates'])/rate/60
        atomic(REPORT/'training_progress.json',data)
        line=f"{data['time_utc']} LIVE {data['phase']} | ACT {data['act_updates']}/{data['act_target']} ({data['act_percent']}%) | real eval {data['real_baseline_episodes']}/32 | queue {data['queue']}"
        if loss:line+=f" | loss@{loss['update']}={loss['last']:.5f}"
        if data.get('progress_warning'):line+=' | '+data['progress_warning']
        live=data.get('baseline_live',{})
        if live.get('phase')=='shared_collection':
            line+=f" | shared episodes={live.get('completed_episodes',0)} steps={live.get('individual_transitions',0)}"
        if live.get('phase')=='critic_training':
            line+=f" | critic={live.get('completed_updates',0)}/{live.get('requested_updates',10000)}"
        with (ROOT/'.cache/background_training.log').open('a') as stream:stream.write(line+'\n')
        print(line,flush=True)
        if not data['supervisor_alive']:
            return
        time.sleep(args.interval)


if __name__=='__main__':
    main()
