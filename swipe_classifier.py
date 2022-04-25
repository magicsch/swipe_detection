from ast import Return
from matplotlib.axis import Axis
import yaml
import numpy as np
import cv2
from collections import deque
from movenet import Movenet
from utils import *
from yaml.loader import SafeLoader


class SwipeClassifier:
    def __init__(self) -> None:
        self._config = self._load_config()
        self._r_wrist_seq = deque(maxlen=self._config['sequnece_length'])
        self._l_wrist_seq = deque(maxlen=self._config['sequnece_length'])
        self._r_swipe_seq = deque(maxlen=5)
        self._l_swipe_seq = deque(maxlen=5)
        self._nose_seq = deque(
            maxlen=self._config['sequnece_length']//2)
        self._movenet = Movenet()

    def _load_config(self):
        with open('config.yaml', 'r') as f:
            return yaml.load(f, Loader=SafeLoader)

    def classify_swipe(self, frame, debug_img=False) -> Swipe:
        keypoints = self._movenet.infer(frame)

        shoulder_width, shoulder_nose_height, nose = self.get_normalization_factors(
            keypoints)
        # print(nose)
        self._nose_seq.append(nose)

        if self.person_valid(keypoints, self._nose_seq):
            frame = self._movenet.draw_keypoints(
                frame, keypoints, threshold=.3)

            # shoulder_width, shoulder_nose_height, nose = self.get_normalization_factors(
            #     keypoints)
            # print(nose)
            # self._nose_seq.append(nose)

            norm_right_wrist, norm_left_wrist = self.normalize_kps(
                keypoints, shoulder_width, shoulder_nose_height)

            self._r_wrist_seq.append(norm_right_wrist)
            self._l_wrist_seq.append(norm_left_wrist)

            r_arm_r_stride = shoulder_width * \
                self._config['right_arm_right_swipe_stride']
            r_arm_l_stride = shoulder_width * \
                self._config['right_arm_left_swipe_stride']
            l_arm_r_stride = shoulder_width * \
                self._config['left_arm_right_swipe_stride']
            l_arm_l_stride = shoulder_width * \
                self._config['left_arm_left_swipe_stride']
            up_stride = shoulder_nose_height*self._config['up_swipe_stride']
            down_stride = shoulder_nose_height * \
                self._config['down_swipe_stride']

            r_wrist_var = np.var(self._r_wrist_seq)
            l_wrist_var = np.var(self._l_wrist_seq)

            if l_wrist_var > r_wrist_var:
                l_out = self.detect_swipe(
                    self._l_wrist_seq, l_arm_r_stride, l_arm_l_stride, up_stride, down_stride)
                self._l_swipe_seq.append(l_out)
                if self.sig_edge(self._l_swipe_seq):
                    out = l_out
                else:
                    out = Swipe.none
            elif r_wrist_var >= l_wrist_var:
                r_out = self.detect_swipe(
                    self._r_wrist_seq, r_arm_r_stride, r_arm_l_stride, up_stride, down_stride)
                self._r_swipe_seq.append(r_out)
                if self.sig_edge(self._r_swipe_seq):
                    out = r_out
                else:
                    out = Swipe.none

            if debug_img:
                return out, frame
            return out

        if debug_img:
            return Swipe.none, frame
        return Swipe.none

    @staticmethod
    def person_valid(keypoints_with_scores, shoulder_seq, epsilon=.04, thresh=.3) -> bool:
        if len(shoulder_seq) == 0:
            return True
        # is person moving
        if np.ptp(shoulder_seq, axis=0)[1] <= epsilon:
            kp_keys = ['nose', 'right_shoulder',
                       'left_shoulder', 'right_elbow',
                       'left_elbow', 'right_wrist', 'left_wrist']
            for key in kp_keys:
                if keypoints_with_scores[KEYPOINT_DICT[key]][2] < thresh:
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
        # Position normalization
        norm_right_wrist = right_wrist_kp - right_elbow_kp
        norm_left_wrist = left_wrist_kp - left_elbow_kp
        # Scale normalization
        norm_right_wrist[0] /= vert_scale_factor
        norm_right_wrist[1] /= hor_scale_factor
        norm_left_wrist[0] /= vert_scale_factor
        norm_left_wrist[1] /= hor_scale_factor
        return norm_right_wrist, norm_left_wrist

    @staticmethod
    def wrist_position(seq, axis=0) -> Swipe:
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
        return np.array([(p1[0]+p2[0])//2, (p1[1]+p2[1])//2])

    @staticmethod
    def negative_edge(seq):
        seq = np.array(seq)
        if len(seq) > 5:
            if seq[-2] == Direction.right and seq[-1] == Direction.none:
                return Swipe.right
            elif seq[-2] == Direction.left and seq[-1] == Direction.none:
                return Swipe.left
            elif seq[-2] == Direction.up and seq[-1] == Direction.none:
                return Swipe.up
            elif seq[-2] == Direction.down and seq[-1] == Direction.none:
                return Swipe.down
