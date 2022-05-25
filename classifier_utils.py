import enum
from collections import namedtuple
import numpy as np


class Swipe(enum.IntEnum):
    right = 1
    left = 2
    up = 3
    down = 4


class Direction(enum.IntEnum):
    horizontal = 1
    vertical = 2
    # up = 3
    # down = 4


class Position(enum.IntEnum):
    middle = 0
    right = 1
    left = 2
    up = 3
    down = 4


# A change of position is an event
# Events that are swipes defined
SWIPE_DEF_DICT = {
    Swipe.up: [
        [Position.down, Position.middle, Position.up],
        [Position.down, Position.up],
        [Position.middle, Position.up]],
    Swipe.down: [
        [Position.up, Position.middle, Position.down],
        [Position.up, Position.down]],
    Swipe.right: [
        [Position.middle, Position.right],
        [Position.left, Position.middle, Position.right],
        [Position.left, Position.right]],
    Swipe.left: [
        [Position.middle, Position.left],
        [Position.right, Position.middle, Position.left],
        [Position.right, Position.left]]}
