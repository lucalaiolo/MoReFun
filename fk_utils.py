"""
Forward kinematics for Human3.6M expmap data.

Code copied (with light cleanup) from
  https://github.com/una-dinosauria/human-motion-prediction
specifically src/forward_kinematics.py and src/data_utils.py. Used here to
convert 99-column expmap frames into (32, 3) joint positions in millimetres.
"""

import numpy as np


# ── data_utils.expmap2rotmat ─────────────────────────────────────────────────

def expmap2rotmat(r):
    theta = np.linalg.norm(r)
    r0 = np.divide(r, theta + np.finfo(np.float32).eps)
    r0x = np.array([[0, -r0[2],  r0[1]],
                    [0,      0, -r0[0]],
                    [0,      0,      0]])
    r0x = r0x - r0x.T
    R = np.eye(3, 3) + np.sin(theta) * r0x + (1 - np.cos(theta)) * (r0x).dot(r0x)
    return R


# ── forward_kinematics._some_variables ───────────────────────────────────────

def _some_variables():
    parent = np.array([0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 1, 12, 13, 14, 15, 13,
                       17, 18, 19, 20, 21, 20, 23, 13, 25, 26, 27, 28, 29, 28, 31]) - 1

    offset = np.array(
        [0.000000, 0.000000, 0.000000, -132.948591, 0.000000, 0.000000, 0.000000, -442.894612, 0.000000, 0.000000,
         -454.206447, 0.000000, 0.000000, 0.000000, 162.767078, 0.000000, 0.000000, 74.999437, 132.948826, 0.000000,
         0.000000, 0.000000, -442.894413, 0.000000, 0.000000, -454.206590, 0.000000, 0.000000, 0.000000, 162.767426,
         0.000000, 0.000000, 74.999948, 0.000000, 0.100000, 0.000000, 0.000000, 233.383263, 0.000000, 0.000000,
         257.077681, 0.000000, 0.000000, 121.134938, 0.000000, 0.000000, 115.002227, 0.000000, 0.000000, 257.077681,
         0.000000, 0.000000, 151.034226, 0.000000, 0.000000, 278.882773, 0.000000, 0.000000, 251.733451, 0.000000,
         0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 99.999627, 0.000000, 100.000188, 0.000000, 0.000000,
         0.000000, 0.000000, 0.000000, 257.077681, 0.000000, 0.000000, 151.031437, 0.000000, 0.000000, 278.892924,
         0.000000, 0.000000, 251.728680, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 99.999888,
         0.000000, 137.499922, 0.000000, 0.000000, 0.000000, 0.000000])
    offset = offset.reshape(-1, 3)

    rotInd = [[5, 6, 4], [8, 9, 7], [11, 12, 10], [14, 15, 13], [17, 18, 16],
              [], [20, 21, 19], [23, 24, 22], [26, 27, 25], [29, 30, 28], [],
              [32, 33, 31], [35, 36, 34], [38, 39, 37], [41, 42, 40], [],
              [44, 45, 43], [47, 48, 46], [50, 51, 49], [53, 54, 52],
              [56, 57, 55], [], [59, 60, 58], [], [62, 63, 61], [65, 66, 64],
              [68, 69, 67], [71, 72, 70], [74, 75, 73], [], [77, 78, 76], []]

    expmapInd = np.split(np.arange(4, 100) - 1, 32)
    return parent, offset, rotInd, expmapInd


# ── forward_kinematics.fkl ───────────────────────────────────────────────────

def fkl(angles, parent, offset, rotInd, expmapInd):
    """Convert one 99-vector of expmap angles into 32 xyz positions."""
    assert len(angles) == 99
    njoints = 32
    xyzStruct = [dict() for _ in range(njoints)]

    for i in np.arange(njoints):
        if not rotInd[i]:
            xangle, yangle, zangle = 0, 0, 0
        else:
            xangle = angles[rotInd[i][0] - 1]
            yangle = angles[rotInd[i][1] - 1]
            zangle = angles[rotInd[i][2] - 1]

        r = angles[expmapInd[i]]
        thisRotation = expmap2rotmat(r)
        thisPosition = np.array([xangle, yangle, zangle])

        if parent[i] == -1:
            xyzStruct[i]['rotation'] = thisRotation
            xyzStruct[i]['xyz'] = np.reshape(offset[i, :], (1, 3)) + thisPosition
        else:
            xyzStruct[i]['xyz'] = (offset[i, :] + thisPosition).dot(xyzStruct[parent[i]]['rotation']) \
                                  + xyzStruct[parent[i]]['xyz']
            xyzStruct[i]['rotation'] = thisRotation.dot(xyzStruct[parent[i]]['rotation'])

    xyz = np.array([xyzStruct[i]['xyz'] for i in range(njoints)]).squeeze()
    xyz = xyz[:, [0, 2, 1]]   # axis swap as in the original repo (Z up)
    return xyz


def expmap_seq_to_xyz(angles_seq: np.ndarray) -> np.ndarray:
    """
    Vectorised wrapper.

    Args:
        angles_seq: (T, 99) expmap frames.
    Returns:
        (T, 32, 3) joint positions in millimetres.
    """
    parent, offset, rotInd, expmapInd = _some_variables()
    out = np.empty((angles_seq.shape[0], 32, 3), dtype=np.float32)
    for t in range(angles_seq.shape[0]):
        out[t] = fkl(angles_seq[t], parent, offset, rotInd, expmapInd)
    return out


# Bone connections used for visualisation (from src/viz.py).
VIZ_I  = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1
VIZ_J  = np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]) - 1
VIZ_LR = np.array([1, 1, 1, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0, 1,  1,  1], dtype=bool)
