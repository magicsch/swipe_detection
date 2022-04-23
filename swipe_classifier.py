from sys import displayhook
from matplotlib.ft2font import HORIZONTAL
import numpy as np
import cv2
from fastdtw import fastdtw
from collections import deque

from rsa import sign
from utils import *


class SwipeClassifier:
    def __init__(self, seq_length=20) -> None:
        self._right_wrist_norm_seq = deque(maxlen=seq_length)
        self._left_wrist_norm_seq = deque(maxlen=seq_length)

    def classify_swipe(self):
        pass

    def _record_kp(self, norm_right_wrist, norm_left_wrist) -> None:
        """
            Appends input vals to internal sequences
        """
        self._right_wrist_norm_seq.append(norm_right_wrist)
        self._left_wrist_norm_seq.append(norm_left_wrist)

    @staticmethod
    def get_normalization_factors(keypoints_with_scores) -> tuple:
        """
            Returns shoulder width and shoulder-nose height in
            movenet coords, not pixels coords
        """
        right_shoulder_kp = keypoints_with_scores[KEYPOINT_DICT['right_shoulder'], :2]
        left_shoulder_kp = keypoints_with_scores[KEYPOINT_DICT['left_shoulder'], :2]
        nose_kp = keypoints_with_scores[KEYPOINT_DICT['nose'], :2]
        shoulder_width = np.linalg.norm(right_shoulder_kp-left_shoulder_kp)
        shoulders_midpoint = SwipeClassifier.midpoint(
            right_shoulder_kp, left_shoulder_kp)
        shoulder_nose_height = np.linalg.norm(shoulders_midpoint-nose_kp)
        return shoulder_width, shoulder_nose_height

    @staticmethod
    def normalize_kps(keypoints_with_scores, hor_scale_factor, vert_scale_factor) -> tuple:
        """
            Return position and scale normalized coordinates of
            right and left wrists
        """
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
    def ortho_displacement(seq, vert_epsilon=.4, hor_epsilon=.7) -> OrthoDisplacement:
        """ Returns orthogonal displacement direction """
        seq = np.array(seq)
        if len(seq) > 10:
            disp = np.ptp(seq, axis=0)
            if disp[0] < disp[1] > hor_epsilon:
                return OrthoDisplacement.Horizontal
            elif disp[1] < disp[0] > vert_epsilon:
                return OrthoDisplacement.Vertical
            else:
                return OrthoDisplacement.No_displacement

    @staticmethod
    def displacement_direction(seq) -> DisplacementDirection:
        disp = SwipeClassifier.ortho_displacement(seq)
        seq = np.array(seq)
        if disp == OrthoDisplacement.Horizontal:
            if np.all(np.sign(seq[-1, 1]) == 1):
                return DisplacementDirection.Left
            elif np.sign(seq[-1, 1]) == -1:
                return DisplacementDirection.Right
        elif disp == OrthoDisplacement.Vertical:
            if np.sign(seq[-1, 0]) == 1:
                return DisplacementDirection.Down
            elif np.sign(seq[-1, 0]) == -1:
                return DisplacementDirection.Up
        else:
            return DisplacementDirection.No_displacement

    @ staticmethod
    def midpoint(p1, p2) -> np.array:
        return np.array([(p1[0]+p2[0])//2, (p1[1]+p2[1])//2])
