import numpy as np
from collections import deque
from classifier_utils import Direction, Position, SWIPE_DEF_DICT, Swipe
from typing import Optional
from lm_state import LMState


class WristState(LMState):
    def __init__(self, thresh=.15) -> None:
        super().__init__()
        self._direction = None
        self._pos_thresh = thresh
        self.last_event = None
        # state machine
        self.img_pos_seq = deque(maxlen=20)
        self._sm = deque(maxlen=3)

    def update(self, fps, norm_pos, img_pos) -> Optional[Swipe]:
        super().update(fps, norm_pos)
        self.img_pos_seq.append(img_pos)
        return self.update_sm()

    @property
    def direction(self) -> Optional[Direction]:
        if self.seq and np.linalg.norm(np.ptp(self.img_pos_seq, axis=0)) >= self._pos_thresh:
            seq = np.array(self.img_pos_seq)
            dx, dy = np.ptp(seq, axis=0)
            return Direction.horizontal if dx >= dy else Direction.vertical

    @property
    def moving(self) -> bool:
        if self.seq:
            seq = np.array(self.seq)[-len(self.seq)//2:]
            return True if np.linalg.norm(np.ptp(seq, axis=0)) >= self._pos_thresh else False
        return False

    @property
    def position(self) -> Position:
        if not self.last_pos is None:
            pos = self.last_pos
            if np.linalg.norm(pos) > self._pos_thresh:
                p = abs(pos)
                if p[0] > p[1]:
                    return Position.right if np.sign(pos[0]) == -1 else Position.left
                if p[0] < p[1]:
                    return Position.up if np.sign(pos[1]) == -1 else Position.down
        return Position.middle

    def update_sm(self) -> Optional[Position]:
        if not self._sm:
            self._sm.append(self.position)
        elif self._sm and not self._sm[-1] == self.position:
            self._sm.append(self.position)
            self.last_event = self.detect_event()
        return self.detect_swipe()

    def detect_event(self) -> Optional[Swipe]:
        for k, v in SWIPE_DEF_DICT.items():
            for el in v:
                if list(self._sm)[-(len(el)):] == el:
                    self._sm.clear()
                    return k

    def detect_swipe(self) -> Optional[Swipe]:
        if self.last_event and not self.moving:
            res = self.last_event
            self.last_event = None
            return res
