from functools import wraps
import time
import numpy as np
from cv2 import cv2
from collections import deque
from geometry_utils import plane_normal, vec_direction
from classifier_utils import Position, Swipe, SWIPE_DEF_DICT, Side
from mp_utils import PoseLandmark, RED
from typing import Optional, Union
from wrist_state import WristState
import mediapipe as mp
from lm_state import LMState
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


class SwipeClassifier:
    def __init__(self, detection_confidence=.7, tracking_confidence=.7) -> None:
        self._r_last_states = deque(maxlen=3)
        self._l_last_states = deque(maxlen=3)
        self._rw_state = WristState()
        self._lw_state = WristState()
        self._nose_state = LMState()
        self._pose = mp_pose.Pose(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            smooth_landmarks=True,
            model_complexity=1
        )
        self.shoulder_width = 0
        self.fps = 0
        self.frame_count = 0
        self.start_time = None

    def _fps_wrapper(func):
        """
            Calcs fps and displays them on frame if debug mode
        """
        @wraps(func)
        def wrap_func(*args, **kwargs):
            if not args[0].start_time:
                args[0].start_time = time.time()
            result = func(*args, **kwargs)
            args[0].frame_count += 1
            args[0].fps = args[0].frame_count//(time.time()-args[0].start_time)
            if kwargs['debug_img']:
                cv2.putText(args[1], f'{args[0].fps:.0f} fps', (int(args[1].shape[1]*.75), 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 5)
            return result
        return wrap_func

    @_fps_wrapper
    def classify_swipe(self, frame, debug_img=False) -> Union[np.array, tuple]:
        out = None
        results = self._pose.process(frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            frame = self._debug_draw(
                frame, results.pose_landmarks) if debug_img else frame
            lms = np.array(
                [(lm.x, lm.y, lm.visibility) for lm in landmarks]
            )
            r_sh, l_sh = lms[(PoseLandmark.RIGHT_SHOULDER,
                              PoseLandmark.LEFT_SHOULDER), :2]
            self.shoulder_width = np.linalg.norm(r_sh-l_sh)
            # self.shoulder_width, nose = self._get_normalization_factors(
            #     lms)
            self._nose_state.update(self.fps, lms[PoseLandmark.NOSE, :2])

            if self._person_valid(self._nose_state.seq):
                # n_right, n_left = self._elbow_normalize(
                #     lms, self.shoulder_width)

                out_r = self._rw_state.update(self.fps, lms)
                out_l = self._lw_state.update(self.fps, lms)

                print(self._rw_state.position)

                out = out_r if out_r else out_l

        return out, frame if debug_img else out

    def _debug_draw(self, img, lms) -> np.array:
        mp_drawing.draw_landmarks(
            img,
            lms,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        return img

    def _person_valid(self, nose_seq) -> bool:
        if len(nose_seq) == 0:
            return True
        if (self._move_displacement(nose_seq)[1] <= self.shoulder_width*.5) and (.2 >= self.shoulder_width >= .10):
            return True
        return False

    def _move_displacement(self, seq) -> tuple:
        dx, dy = np.ptp(seq, axis=0)
        return (0, dx) if dx > dy else (1, dy)

    # def _get_normalization_factors(self, lms) -> tuple:
    #     items = [PoseLandmark.RIGHT_SHOULDER,
    #              PoseLandmark.LEFT_SHOULDER,
    #              PoseLandmark.NOSE]
    #     r_sh, l_sh, nose = lms[items, :2]
    #     sh_width = np.linalg.norm(r_sh-l_sh)
    #     return sh_width, nose

    # def _elbow_normalize(self, lms, scl_factor=None) -> tuple:
    #     if not scl_factor:
    #         scl_factor = self.shoulder_width
    #     items = [PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_ELBOW,
    #              PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_ELBOW]
    #     r_wr, r_e, l_wr, l_e = lms[items, :2]
    #     scl = np.array((scl_factor,)*2)
    #     return (r_wr - r_e)/scl, (l_wr - l_e)/scl
