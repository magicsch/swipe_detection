import numpy as np
from collections import deque
from classifier_utils import Direction
from typing import Optional
import matplotlib.pyplot as plt


class State:
    def __init__(self) -> None:
        self.fps = 0
        self._seq = None

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
        self._seq.append(pos)

    @property
    def seq(self) -> deque:
        return self._seq
