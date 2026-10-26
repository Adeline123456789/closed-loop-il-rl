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


def snapshot():
    job = read(REPORT / 'background_job.json')
    progress = read(CHECKPOINT / 'act0.json')
    stages = read(REPORT / 'run_manifest.json').get('stages', {})
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
    evidence = [p for p in (CHECKPOINT/'act0.json', REPORT/'baseline_validation.json',
                           REPORT/'baseline_training.json', CHECKPOINT/'shared_replay.pt',
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
                latest_stage_log=str(logs[-1].relative_to(ROOT)) if logs else None)


def display(data):
    lines = [f"更新时间 UTC：{data['time_utc']}",
        f"当前：{data['phase']} | 后台进程存活：{data['supervisor_alive']}",
        f"ACT 更新：{data['act_updates']:,} / {data['act_target']:,} ({data['act_percent']}%)；checkpoint 距今 {data['checkpoint_age_seconds']} 秒"]
    loss = data.get('loss', {})
    if loss:
        lines.append(f"CVAE loss（第 {loss['update']} 次）：{loss['last']:.5f}；最近 100 次均值：{loss['mean_last_100']:.5f}（不是成功率）")
    eta = data.get('pretraining_eta_minutes')
    if eta is not None:
        lines.append(f'按近期实测速度，剩余预训练约 {eta:.1f} 分钟；不含后续仿真/Critic/消融')
    evaluation = f"真实 ACT0 评估：{data['real_baseline_episodes']} / 32 回合，已完成回合成功 {data['real_baseline_successes']}"
    if data['real_baseline_episodes']==32:
        evaluation+=f"；本批次成功率 {100*data['real_baseline_successes']/32:.2f}%"
    else:
        evaluation+='；未跑完不报最终成功率'
    lines += [evaluation,
        f"最近实际执行记录距今：{data['execution_evidence_age_seconds']} 秒（监控心跳不算实验进度）",
        '执行日志：'+str(data['latest_stage_log']),
        '后续队列：' + ' → '.join(f'{k} [{v}]' for k,v in data['queue'].items()),
        'diagnostic = O/R/P 保真度 → 配对信号 → 真实 Critic→ACT 更新；short-ablation = B1/B2/B3 各 2000 Critic / 200 Actor 更新，B0–B3 各 32 回合评估',
        '前置门槛失败会停止并记录原因；pilot/confirm/generalization 不在自动队列中。',
        '诊断已记录行数：'+str(data['diagnostic_rows']),
        '消融已完成评估回合：'+str(data['ablation_episodes']),
        '独立 checkpoint：'+', '.join(data['milestones'])]
    if data.get('progress_warning'):
        lines.insert(2, '注意：'+data['progress_warning'])
    if data.get('stop_reason'):
        lines.append('停止原因：'+data['stop_reason'])
    publication=data['publication']
    lines.append(f"最近上传：{publication.get('time_utc','尚无记录')}；退出码 {publication.get('returncode')}；提交 {publication.get('head','')[:8]}")
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
        with (ROOT/'.cache/background_training.log').open('a') as stream:stream.write(line+'\n')
        print(line,flush=True)
        if not data['supervisor_alive']:
            return
        time.sleep(args.interval)


if __name__=='__main__':
    main()
