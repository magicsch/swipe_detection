from enum import Enum

BLUE = (255, 0, 0)
GREE = (0, 0, 255)
RED = (0, 0, 255)

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
    'right_ankle': 16
}

KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (12, 14): 'c',
}


class Swipe(Enum):
    right = 1
    left = 2
    up = 3
    down = 4


class Position(Enum):
    middle = 0
    right = 1
    left = 2
    up = 3
    down = 4


SWIPE_DEF_DICT = {
    Swipe.up: [[Position.down, Position.middle, Position.up],
               [Position.down, Position.up],
               [Position.middle, Position.up]],
    Swipe.down: [[Position.up, Position.middle, Position.down],
                 [Position.up, Position.down]],
    Swipe.right: [[Position.middle, Position.right],
                  [Position.left, Position.middle, Position.right],
                  [Position.left, Position.right]],
    Swipe.left: [[Position.middle, Position.left],
                 [Position.right, Position.middle, Position.left],
                 [Position.right, Position.left]]}
