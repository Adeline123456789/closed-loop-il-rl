"""Producer progress is emitted only after work; observers cannot refresh it."""
import os
import json
import time
from pathlib import Path
import torch
from .artifacts import atomic_json


class DiagnosticProgress:
    def __init__(self, ledger, report, directory, previous_continuations=0):
        import faulthandler
        import signal
        # An on-demand watchdog stack is sufficient. Do not dump a healthy
        # native simulator's threads periodically while it is executing work.
        faulthandler.cancel_dump_traceback_later()
        faulthandler.register(signal.SIGUSR1, all_threads=True)
        self.ledger = ledger
        self.path = Path(report)/'diagnostic_live_progress.json'
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.original = ledger.original
        self.journal = (self.directory/'physics_ledger.jsonl').open('xb',buffering=0)
        self.last_write = 0.
        self.data = dict(pid=os.getpid(), started=time.time(), phase='D0_sampling',
                         journal_path=str(self.directory/'physics_ledger.jsonl'),
                         completed_continuations=0, previous_continuations=previous_continuations,
                         phase_completed={}, status='RUNNING', last_physics_step=None)
        ledger.progress = self
        def measured_step(sim, *args, **kwargs):
            # Persist every charged substep BEFORE native code. A native crash
            # can leave at most the last recorded step incomplete, not erase
            # five seconds of accounting with an unexecuted finally block.
            self.journal.write((json.dumps(dict(event='before',time=time.time(),
                counts=dict(ledger.counts)))+'\n').encode())
            value = self.original(sim, *args, **kwargs)
            self.journal.write((json.dumps(dict(event='after',time=time.time(),
                counts=dict(ledger.counts)))+'\n').encode())
            self.data['last_physics_step'] = time.time()
            if time.monotonic()-self.last_write >= 5:
                self.save()
            return value
        ledger.original = measured_step
        self.save()

    def save(self):
        self.data.update(updated=time.time(), accounting=self.ledger.record())
        atomic_json(self.path, self.data)
        self.last_write = time.monotonic()

    def mark(self, **values):
        self.data.update(values)
        self.save()

    def begin(self):
        if self.data['previous_continuations'] + self.data['completed_continuations'] >= 1500:
            raise RuntimeError('BUDGET_EXHAUSTED before diagnostic continuation including earlier attempts')
        self.mark(trajectory_step=0)

    def finish(self, trajectory):
        number = self.data['completed_continuations']+1
        path = self.directory/f'trajectory_{number:04d}.pt'
        temp = path.with_suffix('.tmp')
        torch.save(dict(context=dict(self.data), trajectory=trajectory), temp)
        temp.replace(path)
        self.data['completed_continuations'] = number
        phase = self.data['phase']
        self.data['phase_completed'][phase] = self.data['phase_completed'].get(phase, 0)+1
        self.mark(last_trajectory=str(path), trajectory_step=trajectory['steps'])

    def close(self):
        self.ledger.original = self.original
        self.journal.close()


def progress(ledger, **values):
    observer = getattr(ledger, 'progress', None)
    if observer is not None:
        observer.mark(**values)
