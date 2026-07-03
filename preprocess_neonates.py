"""
Preprocess neonate 3D keypoints for AuxFormer fine-tuning.

What this does
--------------
1.  Loads each .npz (one per video) with `kpts3d` of shape (T, 70, 3).
2.  Computes a per-clip rotation that aligns the body's hip→shoulder axis with
    H3.6M's +y axis (standing upright).
3.  Maps the 70 MHR keypoints to the 22 H3.6M joints used by AuxFormer.
    Derived joints (hip_root, spine, neck etc.) are constructed by averaging
    or interpolating the relevant input keypoints.
4.  Retargets each bone to H3.6M's reference length while keeping the baby's
    joint angles (direction preserved, magnitude rewritten).
5.  Downsamples each clip to 25 fps to match the H3.6M training rate.
6.  Saves one .npz per video with `xyz` of shape (T, 22, 3) in mm.

Output
------
data/neonates_processed/
    big_neonate.npz      kpts of shape (T, 22, 3)
    small_neonate.npz    kpts of shape (T, 22, 3)
    meta.json            per-clip info (n_frames, fps_in, fps_out, ...)

The output joint order is the same one AuxFormer uses internally (the 22
joints kept after `joints_to_drop`).

Usage
-----
    python preprocess_neonates.py
    python preprocess_neonates.py --input-dir /mnt/user-data/uploads --out-dir data/neonates_processed
"""

import argparse
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Joint conventions
# ---------------------------------------------------------------------------

# Input: 70-joint MHR skeleton (just the parts we need).
MHR_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle",
    "left_big_toe","left_small_toe","left_heel",
    "right_big_toe","right_small_toe","right_heel",
    "right_thumb4","right_thumb3","right_thumb2","right_thumb_third_joint",
    "right_forefinger4","right_forefinger3","right_forefinger2","right_forefinger_third_joint",
    "right_middle_finger4","right_middle_finger3","right_middle_finger2","right_middle_finger_third_joint",
    "right_ring_finger4","right_ring_finger3","right_ring_finger2","right_ring_finger_third_joint",
    "right_pinky_finger4","right_pinky_finger3","right_pinky_finger2","right_pinky_finger_third_joint",
    "right_wrist",
    "left_thumb4","left_thumb3","left_thumb2","left_thumb_third_joint",
    "left_forefinger4","left_forefinger3","left_forefinger2","left_forefinger_third_joint",
    "left_middle_finger4","left_middle_finger3","left_middle_finger2","left_middle_finger_third_joint",
    "left_ring_finger4","left_ring_finger3","left_ring_finger2","left_ring_finger_third_joint",
    "left_pinky_finger4","left_pinky_finger3","left_pinky_finger2","left_pinky_finger_third_joint",
    "left_wrist",
    "left_olecranon","right_olecranon","left_cubital_fossa","right_cubital_fossa",
    "left_acromion","right_acromion","neck",
]
MHR = {n: i for i, n in enumerate(MHR_NAMES)}


# Output: the 22 H3.6M joints AuxFormer keeps, with H3.6M index in column 0
# and a human-readable name in column 1.
H36M_KEPT = [
    ( 2, "right_knee"),
    ( 3, "right_ankle"),
    ( 4, "right_foot"),
    ( 5, "right_toe"),
    ( 7, "left_knee"),
    ( 8, "left_ankle"),
    ( 9, "left_foot"),
    (10, "left_toe"),
    (12, "spine2"),
    (13, "neck"),
    (14, "head"),
    (15, "head_top"),
    (17, "left_shoulder"),
    (18, "left_elbow"),
    (19, "left_wrist"),
    (21, "left_finger"),
    (22, "left_thumb"),
    (25, "right_elbow"),
    (26, "right_wrist"),
    (27, "right_hand"),
    (29, "right_thumb"),
    (30, "extra1"),
]

# H3.6M bone offsets (parent->child) in mm — from forward_kinematics.py
H36M_PARENT = np.array([
    -1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14,
    12, 16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30,
])

H36M_BONE_LEN_MM = {
    # bone (parent_h36m_idx, child_h36m_idx) -> length in mm
    (0, 1):  132.9,    # hip_root -> right_hip
    (1, 2):  442.9,    # right_hip -> right_knee
    (2, 3):  454.2,    # right_knee -> right_ankle
    (3, 4):  162.8,    # right_ankle -> right_foot
    (4, 5):   75.0,    # right_foot -> right_toe
    (0, 6):  132.9,    # hip_root -> left_hip
    (6, 7):  442.9,
    (7, 8):  454.2,
    (8, 9):  162.8,
    (9, 10):  75.0,
    (0, 11):   0.1,    # hip_root -> spine1
    (11, 12): 233.4,   # spine1 -> spine2
    (12, 13): 257.1,   # spine2 -> neck
    (13, 14): 121.1,   # neck -> head
    (14, 15): 115.0,   # head -> head_top
    (12, 16): 257.1,   # spine2 -> thorax_dup
    (16, 17): 151.0,   # thorax_dup -> left_shoulder
    (17, 18): 278.9,
    (18, 19): 251.7,
    (19, 20):   0.0,
    (20, 21): 100.0,
    (19, 22): 100.0,
    (12, 24): 257.1,
    (24, 25): 151.0,
    (25, 26): 278.9,
    (26, 27): 251.7,
    (27, 28):   0.0,
    (28, 29): 100.0,
    (27, 30): 137.5,
}


# ---------------------------------------------------------------------------
# Step 1 — rotate the baby upright
# ---------------------------------------------------------------------------


def rotation_matrix_from_axis(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rotation matrix that takes unit vector `src` onto unit vector `dst`."""
    src = src / np.linalg.norm(src)
    dst = dst / np.linalg.norm(dst)
    v = np.cross(src, dst)
    s = np.linalg.norm(v)
    c = float(np.dot(src, dst))
    if s < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    return np.eye(3) + K + K @ K * ((1 - c) / (s ** 2))


def rotate_upright(kpts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute one rotation matrix per clip from the average hip→shoulder vector,
    then apply it to every frame. Returns (rotated_kpts, R).
    """
    # Average across all frames so noise doesn't dominate.
    shoulder_mid = (kpts[:, MHR["left_shoulder"]] + kpts[:, MHR["right_shoulder"]]) / 2
    hip_mid = (kpts[:, MHR["left_hip"]] + kpts[:, MHR["right_hip"]]) / 2
    body_up = (shoulder_mid - hip_mid).mean(axis=0)
    body_up /= np.linalg.norm(body_up)

    # Target: +y is up in H3.6M's convention.
    target = np.array([0.0, 1.0, 0.0])
    R = rotation_matrix_from_axis(body_up, target)
    return kpts @ R.T, R


# ---------------------------------------------------------------------------
# Step 2 — map 70 MHR joints to the 22 H3.6M joints
# ---------------------------------------------------------------------------


def map_to_h36m_22(kpts: np.ndarray) -> np.ndarray:
    """
    Build a (T, 32, 3) skeleton with all 32 H3.6M joints filled in, then return
    the (T, 22, 3) subset AuxFormer expects.

    Mapping (input keypoint(s) -> H3.6M joint):
        hip_root  (0)   = midpoint(left_hip, right_hip)
        right_hip (1)   = right_hip
        right_knee(2)   = right_knee
        right_ankle(3)  = right_ankle
        right_foot(4)   = right_heel
        right_toe (5)   = right_big_toe
        left_hip (6)    = left_hip
        left_knee(7)    = left_knee
        left_ankle(8)   = left_ankle
        left_foot(9)    = left_heel
        left_toe(10)    = left_big_toe
        spine1 (11)     = lerp(hip_root, spine2, 1/3)
        spine2 (12)     = midpoint(left_shoulder, right_shoulder)  (a.k.a. thorax)
        neck   (13)     = neck
        head   (14)     = midpoint(nose, ears)
        head_top(15)    = lerp away from head by neck->head direction
        thorax_dup(16)  = spine2
        left_shoulder(17) = left_shoulder
        left_elbow (18) = left_elbow
        left_wrist (19) = left_wrist
        left_hand  (20) = left_middle_finger_third_joint (palm-ish)
        left_finger(21) = left_middle_finger4 (fingertip)
        left_thumb (22) = left_thumb4
        right_shoulder_branch(23) = right_shoulder      (unused, kept for parent chain)
        right_shoulder(24) = right_shoulder
        right_elbow(25) = right_elbow
        right_wrist(26) = right_wrist
        right_hand (27) = right_middle_finger_third_joint
        right_finger(28) = right_middle_finger4
        right_thumb(29) = right_thumb4
        extra1(30)      = right_middle_finger3
        extra2(31)      = unused
    """
    T = kpts.shape[0]
    h36m = np.zeros((T, 32, 3), dtype=np.float32)

    # Convenience accessors
    g = lambda name: kpts[:, MHR[name]]
    mid = lambda a, b: (g(a) + g(b)) * 0.5
    lerp = lambda a, b, t: a * (1 - t) + b * t

    hip_root = mid("left_hip", "right_hip")
    spine2_pt = mid("left_shoulder", "right_shoulder")
    head_pt = (g("nose") + g("left_ear") + g("right_ear")) / 3.0
    neck_pt = g("neck")

    # extend head -> head_top by neck->head direction (approximately)
    neck_to_head = head_pt - neck_pt
    head_top_pt = head_pt + neck_to_head * 0.5

    h36m[:, 0]  = hip_root
    h36m[:, 1]  = g("right_hip")
    h36m[:, 2]  = g("right_knee")
    h36m[:, 3]  = g("right_ankle")
    h36m[:, 4]  = g("right_heel")
    h36m[:, 5]  = g("right_big_toe")
    h36m[:, 6]  = g("left_hip")
    h36m[:, 7]  = g("left_knee")
    h36m[:, 8]  = g("left_ankle")
    h36m[:, 9]  = g("left_heel")
    h36m[:, 10] = g("left_big_toe")
    h36m[:, 11] = lerp(hip_root, spine2_pt, 1.0/3.0)
    h36m[:, 12] = spine2_pt
    h36m[:, 13] = neck_pt
    h36m[:, 14] = head_pt
    h36m[:, 15] = head_top_pt
    h36m[:, 16] = (spine2_pt + (g("left_shoulder") + g("right_shoulder")) * 0.5) * 0.5
    # put thorax_dup halfway between spine2 and the shoulder midpoint  
    h36m[:, 17] = g("left_shoulder")
    h36m[:, 18] = g("left_elbow")
    h36m[:, 19] = g("left_wrist")
    h36m[:, 20] = g("left_middle_finger_third_joint")
    h36m[:, 21] = g("left_middle_finger4")
    h36m[:, 22] = g("left_thumb4")
    h36m[:, 23] = g("right_shoulder")
    h36m[:, 24] = g("right_shoulder")
    h36m[:, 25] = g("right_elbow")
    h36m[:, 26] = g("right_wrist")
    h36m[:, 27] = g("right_middle_finger_third_joint")
    h36m[:, 28] = g("right_middle_finger4")
    h36m[:, 29] = g("right_thumb4")
    h36m[:, 30] = g("right_middle_finger3")
    h36m[:, 31] = g("right_middle_finger4")

    return h36m


# ---------------------------------------------------------------------------
# Step 3 — retarget bone lengths to H3.6M template
# ---------------------------------------------------------------------------


def retarget_to_h36m_lengths(h36m_full: np.ndarray) -> np.ndarray:
    """
    Walk the H3.6M kinematic tree, replacing each bone's length with H3.6M's
    reference length while keeping the original direction. Joint angles are
    therefore preserved; only bone lengths change.

    Input units: meters. Output units: millimeters (matches H3.6M raw scale).
    """
    T, J, _ = h36m_full.shape
    out = np.zeros_like(h36m_full, dtype=np.float32)
    # Place the root at the origin per frame
    out[:, 0] = 0.0

    # BFS order so parents are placed before children
    order = []
    visited = [False] * J
    stack = [0]
    visited[0] = True
    while stack:
        i = stack.pop(0)
        order.append(i)
        for child in range(J):
            if not visited[child] and H36M_PARENT[child] == i:
                visited[child] = True
                stack.append(child)

    for i in order:
        p = H36M_PARENT[i]
        if p < 0:
            continue
        # Direction from parent to child in the source (in meters)
        d = h36m_full[:, i] - h36m_full[:, p]                # (T, 3)
        d_norm = np.linalg.norm(d, axis=1, keepdims=True)    # (T, 1)
        d_norm = np.maximum(d_norm, 1e-8)
        dir_unit = d / d_norm
        L = H36M_BONE_LEN_MM.get((p, i), float(d_norm.mean() * 1000))
        out[:, i] = out[:, p] + dir_unit * L

    return out


def select_22(h36m_full: np.ndarray) -> np.ndarray:
    """Pick the 22 joints AuxFormer keeps."""
    idx = [h for h, _ in H36M_KEPT]
    return h36m_full[:, idx]


# ---------------------------------------------------------------------------
# Step 4 — downsample to 25 fps
# ---------------------------------------------------------------------------


def downsample_to_25fps(kpts: np.ndarray, fps_in: float) -> tuple[np.ndarray, float]:
    """Resample by integer ratio when possible, else by linear interpolation."""
    if abs(fps_in - 25.0) < 0.1:
        return kpts.astype(np.float32), 25.0
    if abs(fps_in - 50.0) < 0.1:
        return kpts[::2].astype(np.float32), 25.0
    if abs(fps_in - 30.0) < 0.1:
        # Linear interp from 30 to 25.
        T_in = kpts.shape[0]
        t_in = np.arange(T_in) / 30.0
        T_out = int(np.floor(t_in[-1] * 25.0))
        t_out = np.arange(T_out) / 25.0
        out = np.zeros((T_out,) + kpts.shape[1:], dtype=np.float32)
        for j in range(kpts.shape[1]):
            for d in range(kpts.shape[2]):
                out[:, j, d] = np.interp(t_out, t_in, kpts[:, j, d])
        return out, 25.0
    if abs(fps_in - 15.0) < 0.1:
        # 15 fps is below 25 fps. Upsample by linear interpolation.
        T_in = kpts.shape[0]
        t_in = np.arange(T_in) / 15.0
        T_out = int(np.floor(t_in[-1] * 25.0))
        t_out = np.arange(T_out) / 25.0
        out = np.zeros((T_out,) + kpts.shape[1:], dtype=np.float32)
        for j in range(kpts.shape[1]):
            for d in range(kpts.shape[2]):
                out[:, j, d] = np.interp(t_out, t_in, kpts[:, j, d])
        return out, 25.0
    # General case: linear interp from any FPS to 25.
    T_in = kpts.shape[0]
    t_in = np.arange(T_in) / fps_in
    T_out = int(np.floor(t_in[-1] * 25.0))
    t_out = np.arange(T_out) / 25.0
    out = np.zeros((T_out,) + kpts.shape[1:], dtype=np.float32)
    for j in range(kpts.shape[1]):
        for d in range(kpts.shape[2]):
            out[:, j, d] = np.interp(t_out, t_in, kpts[:, j, d])
    return out, 25.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def process_one(npz_path: Path, name: str) -> dict:
    raw = np.load(npz_path)
    kpts = raw["kpts3d"].astype(np.float32)        # (T, 70, 3), meters
    fps_in = float(raw["fps"])
    T_in = kpts.shape[0]

    # 1. rotate upright
    kpts_rot, R = rotate_upright(kpts)

    # 2. build 32-joint H3.6M tensor
    h36m_full = map_to_h36m_22(kpts_rot)

    # 3. retarget bone lengths to H3.6M (output in mm)
    h36m_retgt = retarget_to_h36m_lengths(h36m_full)

    # 4. select the 22 joints
    h36m_22 = select_22(h36m_retgt)                # (T, 22, 3), mm

    # 5. downsample to 25 fps
    h36m_25, fps_out = downsample_to_25fps(h36m_22, fps_in)

    return {
        "name": name,
        "xyz": h36m_25,
        "rotation": R,
        "fps_in": fps_in,
        "fps_out": fps_out,
        "T_in": T_in,
        "T_out": int(h36m_25.shape[0]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=str, default="/mnt/user-data/uploads")
    p.add_argument("--out-dir", type=str, default="data/neonates_processed")
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "big_neonate":   in_dir / "big_neonate_keypoints.npz",
        "small_neonate": in_dir / "small_neonate_keypoints.npz",
    }

    meta = {"clips": {}}
    for name, fp in files.items():
        if not fp.exists():
            print(f"  [preprocess] skipping {name} (not found at {fp})")
            continue
        info = process_one(fp, name)
        out_path = out_dir / f"{name}.npz"
        np.savez(out_path, xyz=info["xyz"], R=info["rotation"])
        meta["clips"][name] = {
            "out_path": str(out_path),
            "T_in": info["T_in"],
            "T_out": info["T_out"],
            "fps_in": info["fps_in"],
            "fps_out": info["fps_out"],
            "duration_s": info["T_out"] / info["fps_out"],
        }
        print(f"  [preprocess] {name}: {info['T_in']} frames @{info['fps_in']:.0f} -> "
              f"{info['T_out']} frames @{info['fps_out']:.0f} fps, "
              f"xyz shape {info['xyz'].shape}, "
              f"y-range [{info['xyz'][...,1].min():.1f},{info['xyz'][...,1].max():.1f}]")

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[preprocess] wrote everything to {out_dir}/")


if __name__ == "__main__":
    main()
