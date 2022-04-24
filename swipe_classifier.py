import yaml
import numpy as np
import cv2
from fastdtw import fastdtw
from collections import deque
from movenet import Movenet
from utils import *
from scipy.ndimage import gaussian_filter


class SwipeClassifier:
    def __init__(self, seq_length=10) -> None:
        self._r_wrist_seq = deque(maxlen=seq_length)
        self._l_wrist_seq = deque(maxlen=seq_length)
        self._output_seq = deque(maxlen=5)
        self._movenet = Movenet()
        # self._example_files = [
        #     'gesture_examples/right_swipe/example.npy',
        #     'gesture_examples/left_swipe/example1.npy'
        # ]
        # self._r_swipe_files = [
        #     'gesture_examples/right_swipe/example.npy',
        #     'gesture_examples/left_swipe/example1.npy'
        # ]
        # self._l_swipe_files = [
        #     'gesture_examples/right_swipe/example.npy',
        #     'gesture_examples/left_swipe/example1.npy'
        # ]
        # self._u_swipe_files = [
        #     'gesture_examples/right_swipe/example.npy',
        #     'gesture_examples/left_swipe/example1.npy'
        # ]
        # self._d_swipe_files = [
        #     'gesture_examples/right_swipe/example.npy',
        #     'gesture_examples/left_swipe/example1.npy'
        # ]
        # # self._examples = SwipeClassifier.load_examples(self._example_files)
        # self._r_swipe = SwipeClassifier.load(self._r_swipe_files)
        # self._l_swipe = SwipeClassifier.load(self._l_swipe_files)
        # self._u_swipe = SwipeClassifier.load(self._u_swipe_files)
        # self._d_swipe = SwipeClassifier.load(self._d_swipe_files)

    # @staticmethod
    # def load(files_list):
    #     ex = []
    #     for filename in files_list:
    #         with open(filename, 'rb') as f:
    #             ex.append(np.load(f))
    #     return ex

    def classify_swipe(self, frame) -> Swipe:
        keypoints = self._movenet.infer(frame)
        shoulder_width, shoulder_nose_height = self.get_normalization_factors(
            keypoints)
        norm_right_wrist, norm_left_wrist = self.normalize_kps(
            keypoints, shoulder_width, shoulder_nose_height)
        self._r_wrist_seq.append(norm_right_wrist)
        self._l_wrist_seq.append(norm_left_wrist)

        disp = np.ptp(self._r_wrist_seq, axis=0)

        if disp[0] < disp[1] and disp[1] > shoulder_width*7.5:
            out = self.wrist_position(self._r_wrist_seq, axis=0)
        elif disp[1] < disp[0] and disp[0] > shoulder_nose_height*0.8:
            out = self.wrist_position(self._r_wrist_seq, axis=1)
        else:
            out = Swipe.none

        # processing

        return out

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

    # @staticmethod  # .4  and .7
    # def ortho_displacement(seq, vert_epsilon=.5, hor_epsilon=0.25) -> Ortho:
    #     """ Returns orthogonal displacement direction """
    #     seq = np.array(seq)
    #     if len(seq) > 10:
    #         disp = np.ptp(seq, axis=0)
    #         if disp[0] < disp[1] > hor_epsilon:
    #             return Ortho.horizontal
    #         elif disp[1] < disp[0] > vert_epsilon:
    #             return Ortho.vertical
    #         else:
    #             return Ortho.none

    # @staticmethod
    # def displacement_direction(seq) -> Direction:
    #     disp = SwipeClassifier.ortho_displacement(seq)
    #     disp_thresh = 0.9
    #     seq = np.array(seq)
    #     disp_vals = np.ptp(seq, axis=0)
    #     if disp == Ortho.horizontal:
    #         if np.sign(seq[-1, 1]) == 1:
    #             return Direction.left
    #         elif np.sign(seq[-1, 1]) == -1:
    #             return Direction.right
    #     elif disp == Ortho.vertical:
    #         if np.sign(seq[-1, 0]) == 1:
    #             return Direction.down
    #         elif np.sign(seq[-1, 0]) == -1:
    #             return Direction.up
    #     else:
    #         return Direction.none

    # @staticmethod  # .4  and .7
    # def ortho_position(seq) -> Ortho:
    #     seq = np.array(seq)
    #     if len(seq) > 10:
    #         disp = np.ptp(seq, axis=0)
    #         if disp[0] < disp[1]:
    #             return Ortho.horizontal
    #         elif disp[1] < disp[0]:
    #             return Ortho.vertical
    #         else:
    #             return Ortho.none

    # @staticmethod
    # def detect_swipe(seq):
    #     seq = np.array(seq)
    #     dir = SwipeClassifier.displacement_direction(seq)
    #     if dir is Direction.right and np.sign(seq[-1, 1]) == -1:
    #         return Swipe.right
    #     elif dir is Direction.left and np.sign(seq[-1, 1]) == 1:
    #         return Swipe.left
    #     elif dir is Direction.up and np.sign(seq[-1, 0]) == -1:
    #         return Swipe.up
    #     elif dir is Direction.down and np.sign(seq[-1, 0]) == 1:
    #         return Swipe.down
    #     return Swipe.none

    # @staticmethod
    # def negative_edge(seq):
    #     seq = np.array(seq)
    #     if len(seq) > 5:
    #         if seq[-2] == Direction.right and seq[-1] == Direction.none:
    #             return Swipe.right
    #         elif seq[-2] == Direction.left and seq[-1] == Direction.none:
    #             return Swipe.left
    #         elif seq[-2] == Direction.up and seq[-1] == Direction.none:
    #             return Swipe.up
    #         elif seq[-2] == Direction.down and seq[-1] == Direction.none:
    #             return Swipe.down

    # this give bad results
    # def detect_swipe(self, seq,  right_thresh=2.0, left_thresh=3.5, up_thresh=3, down_thresh=3) -> Swipe:
    #     seq = np.array(seq)
    #     dir = SwipeClassifier.displacement_direction(seq)
    #     if dir is DisplacementDirection.Right:
    #         for ex in self._r_swipe:
    #             dist, _ = fastdtw(seq[:, 1], ex[:, 1])
    #             if dist <= right_thresh:
    #                 return Swipe.Right
    #     elif dir is DisplacementDirection.Left:
    #         for ex in self._l_swipe:
    #             dist, _ = fastdtw(seq[:, 1], ex[:, 1])
    #             if dist <= left_thresh:
    #                 return Swipe.Left
    #     elif dir is DisplacementDirection.Up:
    #         for ex in self._u_swipe:
    #             dist, _ = fastdtw(seq[:, 1], ex[:, 0])
    #             if dist <= up_thresh:
    #                 return Swipe.Up
    #     elif dir is DisplacementDirection.Down:
    #         for ex in self._d_swipe:
    #             dist, _ = fastdtw(seq[:, 1], ex[:, 0])
    #             if dist <= down_thresh:
    #                 return Swipe.Down
    #     return Swipe.No_swipe

    @ staticmethod
    def midpoint(p1, p2) -> np.array:
        return np.array([(p1[0]+p2[0])//2, (p1[1]+p2[1])//2])
