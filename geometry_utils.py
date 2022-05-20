import numpy as np
from scipy.spatial.transform import Rotation as R


def plane_normal(a, b, c) -> tuple:
    v1 = c - a
    v2 = b - a
    cp = np.cross(v1, v2)
    mat = R.from_rotvec(cp)
    return mat.as_euler('xyz', degrees=True)


def vec_direction(vec) -> tuple:
    vec = vec/np.linalg.norm(vec)
    mat = R.from_rotvec(vec)
    return mat.as_euler('xyz', degrees=True)
