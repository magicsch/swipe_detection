import numpy as np
from collections import deque
from classifier_utils import Direction, Position, Swipe, SWIPE_DEF_DICT
from typing import Optional
from lm_state import LMState


class WristState(LMState):
    def __init__(self, thresh=.15) -> None:
        super().__init__()
        self._pos_thresh = thresh
        self.img_pos_seq = deque(maxlen=20)
        # state machines
        self._pos_sm = deque(maxlen=3)
        self._dir_sm = deque(maxlen=3)

    def update(self, fps, norm_pos, img_pos) -> Optional[Swipe]:
        super().update(fps, norm_pos)
        self.img_pos_seq.append(img_pos)
        return self.update_sm()

    @property
    def direction(self) -> Optional[Direction]:
        # make it not affected by distance from cam
        if self.img_pos_seq:
            seq = np.array(self.img_pos_seq)[-len(self.img_pos_seq)//2:]
            dx, dy = np.ptp(seq, axis=0)
            if dx >= dy and dx >= .2:
                return Direction.left if self._increasing(seq[:, 0]) else Direction.right
            elif dx < dy and dy >= .2:
                return Direction.down if self._increasing(seq[:, 1]) else Direction.up

    def _increasing(self, arr):
        return True if arr[-1]-np.mean(arr) > 0 else False

    @property
    def position(self) -> Position:
        pos = self.seq[-1]
        # pretty weird and complex
        dir_ = self._dir_sm[-2]
        if dir_ == Direction.right or dir_ == Direction.left:
            return Position.right if np.sign(pos[0]) == -1 else Position.left
        elif dir_ == Direction.up or dir_ == Direction.down:
            return Position.up if np.sign(pos[1]) == -1 else Position.down
        return Position.middle

    def update_sm(self) -> Optional[Position]:
        if not self._dir_sm:
            self._dir_sm.append(self.direction)
        elif self._dir_sm and not self._dir_sm[-1] == self.direction:
            # if here event occurred
            self._dir_sm.append(self.direction)
            return self.detect_event()

    def detect_event(self) -> Optional[Swipe]:
        for k, v in SWIPE_DEF_DICT.items():
            if list(self._dir_sm) == v and self.position == k:
                return k
