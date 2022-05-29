from classifier_utils import SWIPE_DEF_DICT



def directions_rule(dir_states):
    for k, v in SWIPE_DEF_DICT.items():
        dirs, *_ = v
        types = [el.type for el in dir_states]
        durations = [el.duration for el in dir_states]
        # if last movement has been at least as long as the rest of the wrist at the end
        return True if types == dirs and durations[-2] <= durations[-1] else False


def position_rule(pos_state):
    for k, v in SWIPE_DEF_DICT.items():
        _, *positions = v
        types = [el.type for el in pos_state]
        durations = [el.duration for el in pos_state]
        
        return True if types == positions else False




