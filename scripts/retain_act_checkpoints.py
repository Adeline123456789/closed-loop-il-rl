"""Retain immutable ACT milestones without interrupting an active optimizer.

The trainer atomically replaces act0.pt every 100 updates. Pin its current inode
before inspecting it, and trust the update inside the payload rather than the
separately replaced progress JSON. Hard links survive later atomic replacements.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def pin_file(source, target):
    try:
        os.link(source, target)
    except OSError:
        # Open pins the source inode even if the trainer replaces its pathname.
        with open(source, 'rb') as src, open(target, 'xb') as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)


def retain(checkpoint, *, interval=1000, initial=False):
    import torch
    checkpoint = Path(checkpoint)
    folder = checkpoint.parent / 'act0_milestones'
    folder.mkdir(exist_ok=True)
    pinned = folder / '.pinned.pt'
    pinned.unlink(missing_ok=True)
    pin_file(checkpoint, pinned)
    try:
        # Trusted local optimizer output; mmap avoids allocating/copying all tensors.
        payload = torch.load(pinned, map_location='cpu', weights_only=False, mmap=True)
        update = int(payload['update'])
        if not initial and update % interval:
            return None
        target = folder / f'act0_update_{update:06d}.pt'
        if target.exists():
            return {'update': update, 'path': str(target), 'already_retained': True}
        os.replace(pinned, target)
        h = hashlib.sha256()
        with target.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        record = dict(update=update, path=str(target), sha256=h.hexdigest(),
                      bytes=target.stat().st_size, reason='activation_snapshot' if initial else 'scheduled_milestone',
                      contains=['actor','optimizer','sampler_rng','torch_rng','cuda_rng','python_rng','numpy_rng','history','identity'])
        atomic_json(target.with_suffix('.json'), record)
        return record
    finally:
        pinned.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--producer-pid', type=int, required=True)
    parser.add_argument('--interval', type=int, default=1000)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('positive interval required')
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    lock = args.checkpoint.with_suffix('.retention.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Existing checkpoint retention process; no duplicate started.', flush=True)
        return
    report = dict(status='RUNNING', interval=args.interval, producer_pid=args.producer_pid,
                  retained=[], unavailable_before_activation=[])
    activated = False
    seen = None
    while True:
        progress_path = args.checkpoint.with_suffix('.json')
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        update = progress.get('completed_updates')
        if args.checkpoint.exists() and update is not None and update != seen:
            if not activated or update % args.interval == 0 or progress.get('status') == 'COMPLETED':
                record = retain(args.checkpoint, interval=args.interval, initial=not activated)
                if record:
                    if not activated:
                        folder = args.checkpoint.parent / 'act0_milestones'
                        report['unavailable_before_activation'] = [u for u in range(args.interval, record['update'], args.interval)
                            if not (folder / f'act0_update_{u:06d}.pt').exists()]
                    report['retained'].append(record)
                    print(json.dumps(record), flush=True)
                    activated = True
            seen = update
            report['latest_observed_update'] = update
            atomic_json(args.report, report)
        proc = Path(f'/proc/{args.producer_pid}/stat')
        try:
            alive = proc.exists() and proc.read_text().split(') ', 1)[1].split()[0] != 'Z'
        except FileNotFoundError:
            alive = False
        if progress.get('status') == 'COMPLETED' or not alive:
            report['status'] = 'COMPLETED' if progress.get('status') == 'COMPLETED' else 'PRODUCER_EXITED'
            atomic_json(args.report, report)
            return
        time.sleep(1)


if __name__ == '__main__':
    main()
