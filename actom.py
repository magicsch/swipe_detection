from typing import Optional, Union


class Actom:
    """
        Atomic/primitive action.
        Basic element that comprises an action
    """
    __slots__ = ('type', 'duration')

    def __init__(self, actom_type, duration=1) -> None:
        self.type = actom_type
        self.duration = duration

    def __len__(self):
        return self.duration

    def __str__(self) -> str:
        return f'<{self.type},{self.duration}>'

    def __repr__(self) -> str:
        return f'Actom({self.type.name if self.type else self.type},{self.duration})'

    def __add__(self, other):
        if isinstance(other, Actom):
            return self.duration + other.duration
        else:
            return self.duration + other

    def __iadd__(self, other):
        if isinstance(other, Actom):
            self.duration += other.duration
        else:
            self.duration += other
        return self

    def __lt__(self, other):
        return self.duration < other.duration

    def __eq__(self, other):
        if isinstance(other, Actom):
            return self.type == other.type
        else:
            return self.type == other
