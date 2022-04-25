from enum import Enum

KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16}


# class Ortho(Enum):
#     none = 0
#     horizontal = 1
#     vertical = 2


# class Direction(Enum):
#     none = 0
#     right = 1
#     left = 2
#     up = 3
#     down = 4


class Swipe(Enum):
    none = 0
    right = 1
    left = 2
    up = 3
    down = 4
