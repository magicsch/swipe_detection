import time
import numpy as np
import cv2
from collections import deque
from movenet import Movenet
from utils import *


class SwipeClassifier:
    def __init__(self) -> None:
        self._norm_r_seq = deque(maxlen=10)
        self._norm_l_seq = deque(maxlen=10)
        self._nose_seq = deque(maxlen=10)
        self._movenet = Movenet()
        self._thresh = .3
        self.fps = 0
        self.frame_count = 0
        self.start_time = None

    def classify_swipe(self, frame, debug_img=False):
        if not self.start_time:
            self.start_time = time.time()
        out = Swipe.none
        seq_len = int(self.fps)
        if abs(self._norm_r_seq.maxlen - seq_len) > 2:
            self._norm_r_seq = deque(maxlen=seq_len)
            self._norm_l_seq = deque(maxlen=seq_len)
            self._nose_seq = deque(maxlen=seq_len//2)

        keypoints = self._movenet.infer(frame)
        shoulder_width, shoulder_nose_height, nose = self.get_normalization_factors(
            keypoints)
        self._nose_seq.append(nose)

        if self.person_valid(keypoints, self._nose_seq):

            n_right, n_left = self.normalize_kps(
                keypoints, shoulder_width, shoulder_nose_height)

            self._norm_r_seq.append(n_right)
            self._norm_l_seq.append(n_left)

            a = 4
            r_arm_r_stride = shoulder_width*a
            r_arm_l_stride = shoulder_width*a
            l_arm_r_stride = shoulder_width*a
            l_arm_l_stride = shoulder_width*a
            up_stride = shoulder_width*(a+5)
            down_stride = shoulder_width*(a+5)

            r_var = np.var(self._norm_r_seq)
            l_var = np.var(self._norm_l_seq)

            if l_var > r_var:
                out = self.detect_swipe(
                    self._norm_l_seq, l_arm_l_stride, l_arm_r_stride, up_stride, down_stride)
                if out is not Swipe.none:
                    self._norm_l_seq.clear()
            elif r_var > l_var:
                out = self.detect_swipe(
                    self._norm_r_seq, r_arm_r_stride, r_arm_l_stride, up_stride, down_stride)
                if out is not Swipe.none:
                    self._norm_r_seq.clear()

            frame = self._movenet.draw_keypoints(
                frame, keypoints, threshold=self._thresh)

        self.fps = self.frame_count//(time.time()-self.start_time)
        self.frame_count += 1
        cv2.putText(frame, f'{self.fps} fps', (50, int(frame.shape[1]*0.9)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

        if debug_img:
            return out, frame
        return out

    @staticmethod
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

    @staticmethod
    def detect_swipe(seq, left_stride, right_stride, up_stride, down_stride) -> Swipe:
        disp = np.ptp(seq, axis=0)
        h_pos = SwipeClassifier.wrist_position(seq, axis=0)
        v_pos = SwipeClassifier.wrist_position(seq, axis=1)
        if disp[0] < disp[1] and h_pos == Swipe.right and disp[1] > right_stride:
            return Swipe.right
        elif disp[0] < disp[1] and h_pos == Swipe.left and disp[1] > left_stride:
            return Swipe.left
        elif disp[1] < disp[0] and v_pos == Swipe.up and disp[0] > up_stride:
            return Swipe.up
        elif disp[1] < disp[0] and v_pos == Swipe.down and disp[0] > down_stride:
            return Swipe.down
        else:
            return Swipe.none

    @staticmethod
    def sig_edge(seq) -> bool:
        seq = np.array(seq)
        return np.all(seq[:-2] == seq[0]) and seq[-1] != seq[0]

    @staticmethod
    def get_normalization_factors(keypoints_with_scores) -> tuple:
        right_shoulder_kp = keypoints_with_scores[KEYPOINT_DICT['right_shoulder'], :2]
        left_shoulder_kp = keypoints_with_scores[KEYPOINT_DICT['left_shoulder'], :2]
        nose_kp = keypoints_with_scores[KEYPOINT_DICT['nose'], :2]
        shoulder_width = np.linalg.norm(right_shoulder_kp-left_shoulder_kp)
        shoulders_midpoint = SwipeClassifier.midpoint(
            right_shoulder_kp, left_shoulder_kp)
        shoulder_nose_height = np.linalg.norm(shoulders_midpoint-nose_kp)
        return shoulder_width, shoulder_nose_height, nose_kp

    @staticmethod
    def normalize_kps(keypoints_with_scores, hor_scale_factor, vert_scale_factor) -> tuple:
        right_wrist_kp = keypoints_with_scores[KEYPOINT_DICT['right_wrist'], :2]
        right_elbow_kp = keypoints_with_scores[KEYPOINT_DICT['right_elbow'], :2]
        left_wrist_kp = keypoints_with_scores[KEYPOINT_DICT['left_wrist'], :2]
        left_elbow_kp = keypoints_with_scores[KEYPOINT_DICT['left_elbow'], :2]
        # nose_kp = keypoints_with_scores[KEYPOINT_DICT['nose'], :2]
        # Position normalization with elbow kps
        norm_r = right_wrist_kp - right_elbow_kp
        norm_l = left_wrist_kp - left_elbow_kp
        # Scale normalization
        scl = np.array([hor_scale_factor, hor_scale_factor])
        norm_r /= scl
        norm_l /= scl
        return norm_r, norm_l

    @staticmethod
    def wrist_position(seq, axis=0) -> Swipe:
        """ Works with elbow normalized coords """
        seq = np.array(seq)
        if axis == 0:
            if np.sign(seq[-1, 1]) == 1:
                return Swipe.left
            elif np.sign(seq[-1, 1]) == -1:
                return Swipe.right
        if axis == 1:
            if np.sign(seq[-1, 0]) == 1:
                return Swipe.down
            elif np.sign(seq[-1, 0]) == -1:
                return Swipe.up
        else:
            return Swipe.none

    @ staticmethod
    def midpoint(p1, p2) -> np.array:
        return np.array([(p1[0]+p2[0])/2, (p1[1]+p2[1])/2])
