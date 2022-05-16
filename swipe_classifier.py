import time
import numpy as np
from cv2 import cv2
from collections import deque
from movenet import Movenet
from utils import *
from operator import itemgetter
from typing import Optional


class SwipeClassifier:
    def __init__(self) -> None:
        self._last_states = deque(maxlen=3)
        self._r_wrist_seq = deque(maxlen=10)
        self._l_wrist_seq = deque(maxlen=10)
        self._nose_seq = deque(maxlen=10)
        self._movenet = Movenet()
        self.shoulder_width = 0
        self._thresh = .3
        self.fps = 0
        self.frame_count = 0
        self.start_time = None
        self._elbow_circle_raidus = 0

    def _wrapper(func):
        """
            Calc FPS and set sequence lengths
        """
        def wrap_func(*args, **kwargs):
            if not args[0].start_time:
                args[0].start_time = time.time()
            seq_len = int(args[0].fps)
            if abs(args[0]._r_wrist_seq.maxlen - seq_len) > 2:
                args[0]._r_wrist_seq = deque(maxlen=seq_len)
                args[0]._l_wrist_seq = deque(maxlen=seq_len)
                args[0]._nose_seq = deque(maxlen=seq_len//2)
            result = func(*args, **kwargs)
            args[0].frame_count += 1
            args[0].fps = args[0].frame_count//(time.time()-args[0].start_time)
            if kwargs['debug_img']:
                cv2.putText(args[1], f'{args[0].fps} fps', (50, int(args[1].shape[1]*0.9)),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, RED, 5)
            return result
        return wrap_func

    @_wrapper
    def classify_swipe(self, frame, debug_img=False):
        out = None
        keypoints = self._movenet.infer(frame.copy())
        shoulder_width, nose = self.get_normalization_factors(
            keypoints)
        self.shoulder_width = shoulder_width
        self._elbow_circle_raidus = self.shoulder_width*4
        self._nose_seq.append(nose)

        if self.person_valid(keypoints, self._nose_seq):

            n_right, n_left = self.normalize_wrists(
                keypoints, shoulder_width)

            self._r_wrist_seq.append(n_right)
            self._l_wrist_seq.append(n_left)
            res = self.wrist_position(n_right)
            out = self.update_state(res)

            frame = self.debug_draw(frame, keypoints)

        if debug_img:
            return out, frame
        return out

    def debug_draw(self, img, kps):
        # items = itemgetter('right_elbow')(KEYPOINT_DICT)
        # r_e = kps[items, :2]*img.shape[0]
        frame = Movenet.draw_keypoints(img, kps, self._thresh)
        return frame

    def update_state(self, position) -> Optional[Position]:
        if not self._last_states:
            self._last_states.append(position)
        elif self._last_states and not self._last_states[-1] == position:
            self._last_states.append(position)
            return self.detect_swipe()
        return None

    def detect_swipe(self) -> Optional[Position]:
        for k, v in SWIPE_DEF_DICT.items():
            for el in v:
                if list(self._last_states)[-(len(el)):] == el:
                    return k

    def move_displacement(self, seq) -> np.array:
        """
            Returns displacement
            horizontal : 1
            vertical : 0
        """
        return np.ptp(seq, axis=0)

    @ staticmethod
    def person_valid(keypoints_with_scores, seq, epsilon=0.03, thresh=.3) -> bool:
        if len(seq) == 0:
            return True
        # is person moving
        if np.ptp(seq, axis=0)[1] <= epsilon:
            kp_keys = ['nose', 'right_shoulder',
                       'left_shoulder', 'right_elbow',
                       'left_elbow', 'right_wrist', 'left_wrist']
            for key in kp_keys:
                if keypoints_with_scores[KEYPOINT_DICT[key]][2] <= thresh:
                    return False
            return True
        return False

    @ staticmethod
    def get_normalization_factors(kps) -> tuple:
        items = itemgetter('right_shoulder', 'left_shoulder',
                           'nose')(KEYPOINT_DICT)
        r_sh, l_sh, nose = kps[items, :2]
        sh_width = np.linalg.norm(r_sh-l_sh)
        return sh_width, nose

    @ staticmethod
    def normalize_wrists(kps, scl_factor) -> tuple:
        """
            Scale and position normalization
        """
        items = itemgetter('right_wrist', 'right_elbow',
                           'left_wrist', 'left_elbow')(KEYPOINT_DICT)
        r_wr, r_el, l_wr, l_el = kps[items, :2]
        scl = np.array((scl_factor,)*2)
        return (r_wr - r_el)/scl, (l_wr - l_el)/scl

    def wrist_position(self, pos) -> Position:
        if np.linalg.norm(pos) > self._elbow_circle_raidus:
            p = abs(pos)
            if p[0] < p[1]:
                return Position.right if np.sign(pos[1]) == -1 else Position.left
            if p[0] > p[1]:
                return Position.up if np.sign(pos[0]) == -1 else Position.down
        return Position.middle
