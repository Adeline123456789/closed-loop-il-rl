"""Causal history builder. Ground-truth task state is not an accepted actor input."""
from collections import deque
import numpy as np


class ObservationHistory:
    def __init__(self):
        self.frames=deque(maxlen=4)

    def reset(self):
        self.frames.clear()

    def append(self,wrench,motion,previous_command):
        # motion is measured 6-D displacement/velocity; previous_command is distinct 4-D control.
        wrench,motion,command=map(np.asarray,(wrench,motion,previous_command))
        if wrench.shape!=(6,) or motion.shape!=(6,) or command.shape!=(4,):
            raise ValueError('expected actual six-axis wrench, six-axis motion, four-axis command')
        frame=np.concatenate((wrench,motion,command)).astype(np.float32)
        if not np.isfinite(frame).all():raise ValueError('nonfinite sensor history')
        # Replicate the first observed frame, never read a future frame for padding.
        if not self.frames:self.frames.extend([frame.copy() for _ in range(4)])
        else:self.frames.append(frame.copy())

    def observation(self,rgb,state):
        if len(self.frames)!=4:raise ValueError('history is uninitialized')
        state=np.asarray(state,dtype=np.float32)
        if state.shape!=(8,):raise ValueError('EEF pose + gripper state only')
        rgb=np.asarray(rgb)
        if rgb.ndim!=3 or rgb.shape[-1]!=3:raise ValueError('single RGB image required')
        return {'rgb':rgb.copy(),'state':state.copy(),'history':np.stack(self.frames)}


class StableSuccess:
    def __init__(self,hold=3):
        if hold<3:raise ValueError('must hold geometric success for at least three control steps')
        self.hold=hold;self.count=0;self.ever=False;self.final=False
    def step(self,geometric_success):
        self.final=bool(geometric_success);self.ever |= self.final
        self.count=self.count+1 if self.final else 0
        return self.count>=self.hold
    def reset(self):self.count=0;self.ever=False;self.final=False
