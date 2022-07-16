from turtle import position
import numpy as np
from collections import deque
from classifier_utils import Direction, Position, Swipe, SWIPE_DEF_DICT, Side
from mp_utils import PoseLandmark
from actom import Actom
from typing import Optional
from lm_state import LMState
from rules import directions_rule, position_rule


class WristState(LMState):
    def __init__(self, thresh=.15) -> None:
        self._pos_thresh = thresh
        self.seq = deque(maxlen=20)
        self._pos_states = deque(maxlen=3)
        self._dir_states = deque(maxlen=2)

    def update(self, fps, lms, side=Side.right) -> Optional[Swipe]:
        r_sh, l_sh = lms[(PoseLandmark.RIGHT_SHOULDER,
                          PoseLandmark.LEFT_SHOULDER), :2]
        self.shoulder_width = np.linalg.norm(r_sh-l_sh)

        self.wrist = lms[PoseLandmark.RIGHT_WRIST if side ==
                         Side.right else PoseLandmark.LEFT_WRIST, :2]
        self.nose = lms[PoseLandmark.NOSE, :2]
        self.shoulder = r_sh if side == Side.right else l_sh
        self.elbow = lms[PoseLandmark.RIGHT_ELBOW if side ==
                         Side.right else PoseLandmark.LEFT_ELBOW, :2]
        self.norm_pos = self._elbow_normalize()

        super().update(fps, self.wrist)

        ang = np.arctan2(*(self.wrist-self.elbow))
        self.wrist_ang = np.rad2deg(ang)
        self.update_pos_states()

        return self.update_dir_states()

    @property
    def direction(self) -> Optional[Direction]:
        # make it not affected by distance from cam
        if self.seq:
            seq = np.array(self.seq)[-len(self.seq)//2:]
            dx, dy = np.ptp(seq, axis=0)
            if dx >= dy and dx >= .1:
                return Direction.left if self._increasing(seq[:, 0]) else Direction.right
            elif dx < dy and dy >= .1:
                return Direction.down if self._increasing(seq[:, 1]) else Direction.up

    def _increasing(self, arr):
        return True if arr[-1]-np.mean(arr) > 0 else False

    # @property
    # def position(self) -> Position:
    #     if self.right_wrist[1] < self.nose_pos[1]:
    #         return Position.up
    #     elif self.right_wrist[1] > self.right_hip[1]:
    #         return Position.down
    #     elif np.linalg.norm(self.last_pos) > 1:
    #         return Position.right if np.sign(self.last_pos[0]) == -1 else Position.left
    #     return Position.middle

    @property
    def position(self) -> Position:
        if np.linalg.norm(self.norm_pos) > self.shoulder_width*3:
            if abs(self.wrist_ang) >= 140:
                return Position.up
            elif abs(self.wrist_ang) <= 40:
                return Position.down
            else:
                return Position.right if np.sign(self.wrist_ang) == -1 else Position.left
        return Position.middle

    # @property
    # def position(self) -> Position:
    #     if np.linalg.norm(self.last_pos) > 0.65:
    #         p = abs(self.last_pos)
    #         if p[0] > p[1]:
    #             return Position.right if np.sign(self.last_pos[0]) == -1 else Position.left
    #         if p[0] < p[1]:
    #             return Position.up if np.sign(self.last_pos[1]) == -1 else Position.down
    #     return Position.middle

    def update_pos_states(self) -> Optional[Position]:
        if not self._pos_states:
            self._pos_states.append(self.position)
        elif self._pos_states and not self._pos_states[-1] == self.position:
            self._pos_states.append(self.position)

    # def update_pos_states(self):
    #     if not self._pos_states:
    #         self._pos_states.append(Actom(self.position))
    #     elif self._pos_states and self._pos_states[-1] == self.position:
    #         self._pos_states[-1].duration += 1
    #     elif self._pos_states and not self._pos_states[-1] == self.position:
    #         # if here event occurred
    #         self._dir_states.append(Actom(self.position))

    def update_dir_states(self):
        if not self._dir_states:
            self._dir_states.append(Actom(self.direction))
        elif self._dir_states and self._dir_states[-1] == self.direction:
            self._dir_states[-1].duration += 1
        elif self._dir_states and not self._dir_states[-1] == self.direction:
            # if here event occurred
            self._dir_states.append(Actom(self.direction))
        return self.detect_event()

    # def detect_event(self) -> Optional[Swipe]:
    #     for k, v in SWIPE_DEF_DICT.items():
    #         dirs, *poss = v
    #         types = [el.type for el in self._dir_states]
    #         durations = [el.duration for el in self._dir_states]
    #         if types == dirs and durations[-2] <= durations[-1] and self.position == k:
    #             for pos in poss:
    #                 l = len(pos)
    #                 states = np.array(self._pos_states)[-l:]
    #                 if pos == list(states):
    #                     # self._dir_states.clear()
    #                     return k

    # def denoise_pos_states(self):
    #     result = []
    #     mx = max(self._pos_states)
    #     for id, el in enumerate(self._pos_states):
    #         if el.duration > .1*mx.duration:
    #             result.append(el.type)
    #     return result

    def detect_event(self) -> Optional[Swipe]:
        for k, v in SWIPE_DEF_DICT.items():
            dirs, *poss = v
            types = [el.type for el in self._dir_states]
            durations = [el.duration for el in self._dir_states]
            if types == dirs and durations[-2]/2 <= durations[-1]:
                for pos in poss:
                    l = len(pos)
                    states = np.array(self._pos_states)[-l:]
                    if pos == list(states):
                        self._dir_states.clear()
                        return k

    # def detect_event(self) -> Optional[Swipe]:
    #     for k, v in SWIPE_DEF_DICT.items():
    #         dirs, *poss = v
    #         types = [el.type for el in self._dir_states]
    #         durations = [el.duration for el in self._dir_states]
    #         if types == dirs and durations[-2] <= durations[-1] and self.position == k:
    #             print(self._dir_states)
    #             print('----------------')
    #             self._dir_states.clear()
    #             return k

    # def detect_event(self):
    #     if directions_rule(self._dir_states) and position_rule(self._pos_states):

    def _get_normalization_factors(self, lms) -> tuple:
        items = [PoseLandmark.RIGHT_SHOULDER,
                 PoseLandmark.LEFT_SHOULDER]
        r_sh, l_sh = lms[items, :2]
        sh_width = np.linalg.norm(r_sh-l_sh)
        return sh_width

    def _elbow_normalize(self) -> tuple:
        scl_factor = np.array((self.shoulder_width,)*2)
        return (self.wrist - self.elbow)/scl_factor
