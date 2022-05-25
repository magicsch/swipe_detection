import numpy as np
from collections import deque
from classifier_utils import Direction
from typing import Optional


class LMState():
    # def __init__(self) -> None:
    #     self.fps = 0
    #     self.seq = None
    #     self.last_pos = None

    # make this create seq and update  its length
    def resize_seq(self, seq, fps) -> deque:
        self.fps = int(fps)
        if seq is None:
            seq = deque(maxlen=self.fps)
        d = seq.maxlen - self.fps
        if abs(d) > 2:
            arr = np.array(seq)
            return deque(
                arr[-(self.fps-1):], self.fps) if np.sign(d) else deque(arr, self.fps)
        return seq

    def update(self, fps, pos) -> None:
        if not hasattr(self, 'seq'):
            self.seq = deque(maxlen=10)
        self.seq = self.resize_seq(self.seq, fps)
        self.last_pos = pos
        self.seq.append(pos)
