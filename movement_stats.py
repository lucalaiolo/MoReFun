"""
movement_stats.py

Movement quantification for neonate skeletons, agnostic to the joint
convention used.

Implements the seven distance-based / kinematic quantifiers from the NeoVault
pilot study (Pigueiras-del-Real et al., Healthcare 2025) plus additional
statistics from the infant-movement literature.

The two optical-flow methods in the paper (Lucas-Kanade, Farnebäck) are
intentionally omitted: they operate on pixel images (we work with 3D
keypoints), and the paper itself found them unreliable for preterm infants
(CV 26 %–87 %, versus 5–15 % for the distance-based methods).

Inputs
------
xyz : (T, J, 3) float array in mm at fixed fps (default 25).
      J can be anything — pass the matching region map.

Region maps
-----------
    REGIONS_H36M22   — the 22-joint kept subset from preprocess_neonates.py
    regions_mhr(MHR) — the raw 70-joint MHR skeleton (pass in the MHR dict)

Notes on cross-skeleton comparability
-------------------------------------
Distance-based methods sum over the joints in the region, so their absolute
magnitude scales with joint count. Values are comparable across windows,
subjects, and epochs *within one skeleton convention*, not across.
Set `per_joint=True` on `full_report` to divide by |region| for a
skeleton-agnostic magnitude.

Usage
-----
    from movement_stats import full_report, REGIONS_H36M22
    report = full_report(xyz, regions_map=REGIONS_H36M22, fps=25)
    # report[region][method][window_s] = {min, max, avg, std, median, cv_pct, n}
"""

from __future__ import annotations
import numpy as np
from typing import Callable


# ---------------------------------------------------------------------------
# Region maps
# ---------------------------------------------------------------------------
# 22-joint kept H3.6M subset (index order from H36M_KEPT in preprocess_neonates.py):
#  0 right_knee     1 right_ankle    2 right_foot    3 right_toe
#  4 left_knee      5 left_ankle     6 left_foot     7 left_toe
#  8 spine2         9 neck          10 head         11 head_top
# 12 left_shoulder 13 left_elbow    14 left_wrist   15 left_finger  16 left_thumb
# 17 right_elbow  18 right_wrist   19 right_hand   20 right_thumb  21 extra1
REGIONS_H36M22: dict[str, list[int]] = {
    "whole_body": list(range(22)),
    "head":       [9, 10, 11],
    "torso":      [8, 9],
    "left_arm":   [12, 13, 14, 15, 16],
    "right_arm":  [17, 18, 19, 20, 21],
    "left_leg":   [4, 5, 6, 7],
    "right_leg":  [0, 1, 2, 3],
    "upper_body": list(range(8, 22)),
    "lower_body": list(range(8)),
}


def regions_mhr(MHR: dict[str, int]) -> dict[str, list[int]]:
    """Region map for the raw 70-joint MHR skeleton. Pass in the MHR dict
    from preprocess_neonates so indices aren't hardcoded here."""
    def pick(*names): return [MHR[n] for n in names]
    return {
        "whole_body": list(range(70)),
        "body_only":  pick(
            "nose","left_eye","right_eye","left_ear","right_ear",
            "left_shoulder","right_shoulder","left_elbow","right_elbow",
            "left_wrist","right_wrist",
            "left_hip","right_hip","left_knee","right_knee",
            "left_ankle","right_ankle",
            "left_big_toe","left_small_toe","left_heel",
            "right_big_toe","right_small_toe","right_heel","neck",
        ),
        "head":       pick("nose","left_eye","right_eye","left_ear","right_ear","neck"),
        "torso":      pick("left_shoulder","right_shoulder","left_hip","right_hip","neck"),
        "left_arm":   pick("left_shoulder","left_elbow","left_wrist"),
        "right_arm":  pick("right_shoulder","right_elbow","right_wrist"),
        "left_leg":   pick("left_hip","left_knee","left_ankle","left_heel",
                           "left_big_toe","left_small_toe"),
        "right_leg":  pick("right_hip","right_knee","right_ankle","right_heel",
                           "right_big_toe","right_small_toe"),
    }


def _select(xyz: np.ndarray, region: str,
            regions: dict[str, list[int]]) -> np.ndarray:
    idx = regions[region]
    if max(idx) >= xyz.shape[1]:
        raise IndexError(
            f"Region '{region}' expects at least {max(idx)+1} joints, "
            f"got xyz with {xyz.shape[1]}. Wrong region map for this skeleton?"
        )
    return xyz[:, idx, :]


# ---------------------------------------------------------------------------
# NeoVault frame-to-frame quantifiers
#
# Each returns one scalar per pair of consecutive frames, summed over joints
# in the selected region — i.e. a (T-1,) or (T-2,) sequence. Call `.sum()`
# on the output to obtain the "total movement in this segment" NeoVault
# reports as one entry in its Values list.
# ---------------------------------------------------------------------------

def euclidean(xyz, region="whole_body", regions=REGIONS_H36M22):
    """Σ over joints of L2 displacement between consecutive frames."""
    p = _select(xyz, region, regions)
    d = np.diff(p, axis=0)                          # (T-1, J, 3)
    return np.linalg.norm(d, axis=2).sum(axis=1)    # (T-1,)


def manhattan(xyz, region="whole_body", regions=REGIONS_H36M22):
    """Σ over joints & dims of |Δx| + |Δy| + |Δz|."""
    p = _select(xyz, region, regions)
    return np.abs(np.diff(p, axis=0)).sum(axis=(1, 2))


def chebyshev(xyz, region="whole_body", regions=REGIONS_H36M22):
    """Σ over joints of max(|Δx|, |Δy|, |Δz|)."""
    p = _select(xyz, region, regions)
    return np.abs(np.diff(p, axis=0)).max(axis=2).sum(axis=1)


def minkowski(xyz, region="whole_body", regions=REGIONS_H36M22, p=3.0):
    """Σ over joints of L_p displacement. p=2 → Euclidean, p=1 → Manhattan."""
    pos = _select(xyz, region, regions)
    d = np.abs(np.diff(pos, axis=0))                # (T-1, J, 3)
    per_joint = (d ** p).sum(axis=2) ** (1.0 / p)   # (T-1, J)
    return per_joint.sum(axis=1)                    # (T-1,)


def mahalanobis(xyz, region="whole_body", regions=REGIONS_H36M22):
    """
    Σ over joints of √(dᵀ S⁻¹ d), where d is the frame-to-frame displacement
    vector and S is the sample covariance of all displacement vectors in the
    segment. Normalizes by the neonate's dominant movement directions.
    Works for any spatial dimensionality (2D or 3D).
    """
    pos = _select(xyz, region, regions)
    d = np.diff(pos, axis=0)                        # (T-1, J, D)
    D = d.shape[-1]
    flat = d.reshape(-1, D)
    cov = np.cov(flat, rowvar=False) + 1e-6 * np.eye(D)
    inv = np.linalg.inv(cov)
    q = np.einsum("tji,ik,tjk->tj", d, inv, d)
    return np.sqrt(np.maximum(q, 0.0)).sum(axis=1)  # (T-1,)


def differential_acceleration(xyz, region="whole_body",
                              regions=REGIONS_H36M22, fps=25.0):
    """
    Σ over joints & dims of |a| = |second time derivative of position| (mm/s²).
    Matches the paper's Σ|V_{i,i+1} − V_{i-1,i}| formulation up to a constant.
    """
    pos = _select(xyz, region, regions)
    v = np.diff(pos, axis=0) * fps                  # (T-1, J, 3), mm/s
    a = np.diff(v,   axis=0) * fps                  # (T-2, J, 3), mm/s²
    return np.abs(a).sum(axis=(1, 2))               # (T-2,)


def angular_displacement(xyz, region="whole_body", regions=REGIONS_H36M22):
    """
    Σ over joints of the angle (radians) between the joint's position vector
    (relative to the region centroid) at t and at t+1. Centroid reference
    makes the angles invariant to global translation.
    """
    pos = _select(xyz, region, regions).astype(np.float64)
    center = pos.mean(axis=1, keepdims=True)
    v = pos - center                                # (T, J, D)
    n = np.maximum(np.linalg.norm(v, axis=2, keepdims=True), 1e-8)
    u = v / n
    dot = np.clip((u[:-1] * u[1:]).sum(axis=2), -1.0, 1.0)
    return np.arccos(dot).sum(axis=1)               # (T-1,) in radians


DISTANCE_METHODS: dict[str, Callable] = {
    "euclidean":                 euclidean,
    "manhattan":                 manhattan,
    "chebyshev":                 chebyshev,
    "minkowski":                 minkowski,
    "mahalanobis":               mahalanobis,
    "differential_acceleration": differential_acceleration,
    "angular_displacement":      angular_displacement,
}


# ---------------------------------------------------------------------------
# Additional statistics — one scalar per segment.
#
# These are joint-count-agnostic in magnitude (they average or normalize),
# so they compare cleanly across skeletons.
# ---------------------------------------------------------------------------

def path_length(xyz, region="whole_body", regions=REGIONS_H36M22):
    """Total path length (mm) summed over joints. Cumulative Euclidean."""
    return float(euclidean(xyz, region, regions).sum())


def mean_speed(xyz, region="whole_body", regions=REGIONS_H36M22, fps=25.0):
    """Mean joint speed (mm/s), averaged over joints and time."""
    pos = _select(xyz, region, regions)
    v = np.linalg.norm(np.diff(pos, axis=0), axis=2) * fps
    return float(v.mean())


def active_time_ratio(xyz, region="whole_body", regions=REGIONS_H36M22,
                      fps=25.0, thresh_mm_s=5.0):
    """
    Fraction of frames in which the region's mean joint speed exceeds
    `thresh_mm_s`. Used to separate active bouts from quiescent periods.
    Threshold is intentionally low for neonates.
    """
    pos = _select(xyz, region, regions)
    v = np.linalg.norm(np.diff(pos, axis=0), axis=2) * fps
    return float((v.mean(axis=1) > thresh_mm_s).mean())


def jerk_smoothness_ldlj(xyz, region="whole_body", regions=REGIONS_H36M22,
                         fps=25.0):
    """
    Log Dimensionless Jerk (LDLJ) — standard motion-smoothness measure
    (Hogan & Sternad 2009). More negative → jerkier movement. Sensitive
    to the "cramped-synchronized" pattern flagged by Prechtl-style GMA.
    """
    pos = _select(xyz, region, regions)
    T = pos.shape[0]
    if T < 4:
        return float("nan")
    duration = (T - 1) / fps
    v = np.diff(pos, axis=0) * fps
    a = np.diff(v,   axis=0) * fps
    j = np.diff(a,   axis=0) * fps                  # (T-3, J, 3), mm/s³
    speeds = np.linalg.norm(v, axis=2)              # (T-1, J)
    peak = float(speeds.max())
    if peak < 1e-6:
        return float("nan")
    integral = float((j ** 2).sum()) / fps
    return float(-np.log((duration ** 3 / peak ** 2) * integral + 1e-12))


def left_right_asymmetry(xyz, side="arm", regions=REGIONS_H36M22):
    """
    Ratio of path length between right and left of a paired limb.
    1.0 = symmetric; large deviations suggest lateralized motor patterns.
    `side` ∈ {"arm", "leg"} (or any key with matching left_/right_ entries).
    """
    L = path_length(xyz, "left_"  + side, regions)
    R = path_length(xyz, "right_" + side, regions)
    return float(R / L) if L > 1e-6 else float("nan")


def dominant_frequency(xyz, region="whole_body", regions=REGIONS_H36M22,
                       fps=25.0):
    """
    Peak frequency (Hz) of the region's mean-joint-speed signal.
    Distinguishes fast, "cramped" oscillations from slower "fidgety"
    or writhing movements typical of healthy neonates.
    """
    pos = _select(xyz, region, regions)
    v = np.linalg.norm(np.diff(pos, axis=0), axis=2).mean(axis=1)
    if len(v) < 4:
        return float("nan")
    v = v - v.mean()
    freqs = np.fft.rfftfreq(len(v), d=1.0 / fps)
    spec = np.abs(np.fft.rfft(v))
    spec[0] = 0.0                                   # kill DC
    return float(freqs[spec.argmax()])


def movement_range(xyz, region="whole_body", regions=REGIONS_H36M22):
    """
    Axis-aligned bounding-box volume (mm³ in 3D, mm² in 2D) enclosing all
    joint positions over the segment. Captures workspace exploration,
    complementary to total path length.
    """
    pos = _select(xyz, region, regions).reshape(-1, xyz.shape[-1])
    return float(np.prod(pos.max(axis=0) - pos.min(axis=0)))


EXTRA_METHODS: dict[str, Callable] = {
    "path_length":          path_length,
    "mean_speed":           mean_speed,
    "active_time_ratio":    active_time_ratio,
    "jerk_smoothness_ldlj": jerk_smoothness_ldlj,
    "dominant_frequency":   dominant_frequency,
    "movement_range":       movement_range,
}


# ---------------------------------------------------------------------------
# Windowing + NeoVault-style aggregation
# ---------------------------------------------------------------------------

def windowed(fn: Callable, xyz: np.ndarray, window_s: float,
             fps: float = 25.0, region: str = "whole_body",
             regions: dict[str, list[int]] = REGIONS_H36M22,
             per_joint: bool = False, **kwargs) -> np.ndarray:
    """
    Apply `fn` on non-overlapping windows of `window_s` seconds. Returns one
    value per window. Per-frame sequences are summed within their window;
    scalar-valued functions are used as-is.

    If per_joint=True, distance-summed values are divided by |region| for
    skeleton-agnostic magnitude. No effect on already-normalized stats.
    """
    win = int(round(window_s * fps))
    T = xyz.shape[0]
    n_win = T // win
    out = np.zeros(n_win, dtype=np.float64)
    denom = len(regions[region]) if per_joint else 1.0
    for i in range(n_win):
        seg = xyz[i * win: (i + 1) * win]
        # left_right_asymmetry uses `side` instead of `region`
        if fn is left_right_asymmetry:
            val = fn(seg, side=region, regions=regions)
        else:
            val = fn(seg, region=region, regions=regions, **kwargs)
        v = float(np.sum(val)) if np.ndim(val) else float(val)
        out[i] = v / denom
    return out


def summarize(values: np.ndarray) -> dict:
    """NeoVault-style summary of a per-window values array."""
    if len(values) == 0:
        return {"n": 0}
    mean = float(np.mean(values))
    std  = float(np.std(values, ddof=0))
    return {
        "n":      int(len(values)),
        "min":    float(values.min()),
        "max":    float(values.max()),
        "avg":    mean,
        "std":    std,
        "median": float(np.median(values)),
        "cv_pct": float(std / mean * 100) if abs(mean) > 1e-9 else float("nan"),
    }


def full_report(xyz: np.ndarray, fps: float = 25.0,
                windows_s: tuple[float, ...] = (30, 60, 120, 180),
                regions_map: dict[str, list[int]] = REGIONS_H36M22,
                region_names: list[str] | None = None,
                include_extras: bool = True,
                per_joint: bool = False) -> dict:
    """
    Full NeoVault-style matrix: for every (region × method × window),
    return the summary dict.

    Return shape: report[region][method][window_s] = summary.
    """
    if region_names is None:
        region_names = list(regions_map)
    all_methods = dict(DISTANCE_METHODS)
    if include_extras:
        all_methods.update(EXTRA_METHODS)

    report = {}
    for reg in region_names:
        report[reg] = {}
        for name, fn in all_methods.items():
            report[reg][name] = {}
            for w in windows_s:
                vals = windowed(fn, xyz, w, fps=fps, region=reg,
                                regions=regions_map, per_joint=per_joint)
                report[reg][name][w] = summarize(vals)
    return report


# ---------------------------------------------------------------------------
# CLI: run over the .npz files produced by preprocess_neonates.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("--in-dir",    default="data/neonates_processed")
    p.add_argument("--out",       default="data/neonates_processed/movement_stats.json")
    p.add_argument("--fps",       type=float, default=25.0)
    p.add_argument("--skeleton",  choices=["h36m22", "mhr70"], default="h36m22",
                   help="Which region map to use. h36m22 matches the "
                        "preprocessed .npz output; mhr70 matches the raw "
                        "keypoints .npz (kpts3d, 70 joints).")
    p.add_argument("--kpts-key",  default=None,
                   help="Array key inside the .npz. Default: 'xyz' for "
                        "h36m22, 'kpts3d' for mhr70.")
    p.add_argument("--per-joint", action="store_true",
                   help="Divide summed-distance methods by |region| for "
                        "skeleton-agnostic magnitude.")
    args = p.parse_args()

    # Pick the region map + default array key
    if args.skeleton == "h36m22":
        regions_map = REGIONS_H36M22
        kpts_key = args.kpts_key or "xyz"
        scale_to_mm = 1.0                       # already in mm
    else:  # mhr70
        from preprocess_neonates import MHR
        regions_map = regions_mhr(MHR)
        kpts_key = args.kpts_key or "kpts3d"
        scale_to_mm = 1000.0                    # raw is in meters

    in_dir = Path(args.in_dir)
    report_all = {}
    for npz in sorted(in_dir.glob("*.npz")):
        data = np.load(npz)
        if kpts_key not in data.files:
            continue
        xyz = data[kpts_key].astype(np.float32) * scale_to_mm
        print(f"[movement_stats] {npz.name}: {xyz.shape[0]} frames, "
              f"{xyz.shape[1]} joints ({args.skeleton})")
        report_all[npz.stem] = full_report(
            xyz, fps=args.fps,
            regions_map=regions_map,
            per_joint=args.per_joint,
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report_all, f, indent=2)
    print(f"[movement_stats] wrote {args.out}")
