from pickle import NEWTRUE
from sre_parse import State
import numpy as np
from collections import deque
from classifier_utils import Direction
from typing import Optional
import matplotlib.pyplot as plt


class LMState(State):
    def __init__(self) -> None:
        super(State, self).__init__(self)
        self._avg_speed = 0
        self._speeds = None
        self._direction = None

    def _resize_seq(self, seq, fps) -> deque:
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
        self._seq = self._resize_seq(self._seq, fps)
        self._speeds = self._resize_seq(self._speeds, fps)
        self._seq.append(pos)
        self._avg_speed = np.linalg.norm(self._seq)
        if len(self._seq) > 2:
            curr_speed = (np.linalg.norm(
                self._seq[-1]-self._seq[-2])) / self.fps
            self._speeds.append(curr_speed)
            if curr_speed > 0.5:
                print("moving")

    @property
    def seq(self) -> deque:
        return self._seq

    @property
    def direction(self) -> Optional[Direction]:
        dx, dy = np.ptp(self._seq, axis=0)
        sgnx, sgny = np.sign([dx, dy])
        if dx > dy:
            self._direction = Direction.right if sgnx == -1 else Direction.left
        elif dy >= dx:
            self._direction = Direction.up if sgny == -1 else Direction.down
        return self._direction

    @property
    def peak(self) -> bool:
        """ Return True if speed has a at least 80% falling edge """
        if self._speeds:
            spds = np.array(self._speeds)
            max = np.max(spds)
            thresh = spds/(max)
            return True if any(thresh > 1) else False
        return False

    @property
    def speed(self) -> float:
        return self._avg_speed
