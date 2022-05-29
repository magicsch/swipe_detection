import enum
from collections import namedtuple
import numpy as np

from utils import SWIPE_DEF_DICT


class Swipe(enum.IntEnum):
    right = 1
    left = 2
    up = 3
    down = 4


class Direction(enum.IntEnum):
    right = 1
    left = 2
    up = 3
    down = 4


class Position(enum.IntEnum):
    middle = 0
    right = 1
    left = 2
    up = 3
    down = 4


class Side(enum.IntEnum):
    right = 1
    left = 2


# # Swipes defined in terms of position states
# SWIPE_POS_DEF_DICT = {
#     Swipe.up: [
#         [Position.down, Position.middle, Position.up],
#         [Position.down, Position.up],
#         [Position.middle, Position.up]],
#     Swipe.down: [
#         [Position.up, Position.middle, Position.down],
#         [Position.up, Position.down]],
#     Swipe.right: [
#         [Position.middle, Position.right],
#         [Position.left, Position.middle, Position.right],
#         [Position.left, Position.right]],
#     Swipe.left: [
#         [Position.middle, Position.left],
#         [Position.right, Position.middle, Position.left],
#         [Position.right, Position.left]]}

# Swipes defined in terms of movement direction states
SWIPE_DEF_DICT = {
    Swipe.up: ([Direction.up, None],
               [Position.down, Position.middle, Position.up],
               [Position.down, Position.up],
               [Position.middle, Position.up]),
    Swipe.down: ([Direction.down, None],
                 [Position.up, Position.middle, Position.down],
                 [Position.up, Position.down]),
    Swipe.right: ([Direction.right, None],
                  [Position.middle, Position.right],
                  [Position.left, Position.middle, Position.right],
                  [Position.left, Position.right]),
    Swipe.left: ([Direction.left, None],
                 [Position.middle, Position.left],
                 [Position.right, Position.middle, Position.left],
                 [Position.right, Position.left])}
