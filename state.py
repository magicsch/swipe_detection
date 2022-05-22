import numpy as np
from collections import deque
from classifier_utils import Direction


class State:
    def __init__(self, fps) -> None:
        self.fps = fps
        self.positions = deque(maxlen=self.fps)
        self.speed = 0
        self.direction = None

    def update(self, fps, pos):
        pass
