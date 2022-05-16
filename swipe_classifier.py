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
        self._r_last_states = deque(maxlen=3)
        self._l_last_states = deque(maxlen=3)
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
        self._move_thresh = 0

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
                cv2.putText(args[1], f'{args[0].fps:.0f} fps', (50, int(args[1].shape[1]*0.9)),
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
        self._move_thresh = shoulder_width*.2
        self._elbow_circle_raidus = self.shoulder_width*3.5
        self._nose_seq.append(nose)

        if self.person_valid(keypoints, self._nose_seq):

            n_right, n_left = self.normalize_wrists(
                keypoints, shoulder_width)

            # make it detect swipes even when last frame
            # person not valid

            self._r_wrist_seq.append(n_right)
            pos = self.wrist_position(n_right)
            out_r = self.update_state(
                self._r_last_states, pos, self._r_wrist_seq)

            self._l_wrist_seq.append(n_left)
            pos = self.wrist_position(n_left)
            out_l = self.update_state(
                self._l_last_states, pos, self._l_wrist_seq)

            out = out_r if out_r else out_l

            frame = self.debug_draw(frame, keypoints) if debug_img else frame

        return out, frame if debug_img else out

    def debug_draw(self, img, kps) -> np.array:
        frame = Movenet.draw_keypoints(img, kps, self._thresh)
        return frame

    def update_state(self, states, position, positions_seq) -> Optional[Position]:
        if not states:
            states.append(position)
        elif states and not states[-1] == position:
            states.append(position)
            return self.detect_swipe(states, positions_seq)
        return None

    def detect_swipe(self, states, positions_seq) -> Optional[Position]:
        for k, v in SWIPE_DEF_DICT.items():
            multip, *list_ = v
            for el in list_:
                if list(states)[-(len(el)):] == el and self.move_displacement(positions_seq)[1] >= multip*self.shoulder_width:
                    # print(self.move_displacement(self._r_wrist_seq)[1])
                    # print(multip*self.shoulder_width)
                    return k

    def move_displacement(self, seq) -> tuple:
        """
            Returns the bigger displacement, horizontal or vertical
        """
        disp = np.ptp(seq, axis=0)
        return (0, disp[0]) if disp[0] > disp[1] else (1, disp[1])

    def person_valid(self, keypoints_with_scores, seq, thresh=.3) -> bool:
        if len(seq) == 0:
            return True
        # is person moving
        if self.move_displacement(seq)[1] <= self._move_thresh and self.shoulder_width >= 0.12:
            items = itemgetter('nose', 'right_shoulder',
                               'left_shoulder', 'right_elbow',
                               'left_elbow', 'right_wrist', 'left_wrist')(KEYPOINT_DICT)
            for score in keypoints_with_scores[items, 2]:
                if score <= thresh:
                    return False
            return True
        return False

    @staticmethod
    def get_normalization_factors(kps) -> tuple:
        items = itemgetter('right_shoulder', 'left_shoulder',
                           'nose')(KEYPOINT_DICT)
        r_sh, l_sh, nose = kps[items, :2]
        sh_width = np.linalg.norm(r_sh-l_sh)
        return sh_width, nose

    @staticmethod
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
