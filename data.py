"""
Human3.6M data loading from raw expmap text files.

Expected layout (Stanford preprocessed H3.6M, ashesh / una-dinosauria format):

    {data_root}/S1/walking_1.txt
    {data_root}/S1/walking_2.txt
    {data_root}/S1/greeting_1.txt
    ...
    {data_root}/S5/...
    {data_root}/S6/...

Each .txt file is a comma-separated table with 99 columns per row, one row per
frame at 50 fps. Columns 0-2 are root translation; columns 3-98 are expmap
joint rotations (3 per joint, for 32 joints).

This loader runs forward kinematics on every file at startup, caches the
resulting (T, 22, 3) joint-position arrays in memory, then serves
sliding-window samples.

Note on subjects: H3.6M only releases S1, S5, S6, S7, S8, S9, S11. Subjects
S2, S3, S4, S10 are held back by the dataset authors; their absence is
expected.
"""

import math
import os
import glob
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from fk_utils import expmap_seq_to_xyz


# Keep only these 22 joints from the full 32-joint H3.6M skeleton.
H36M_22_JOINT_INDICES = [
    0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19, 21, 24, 25, 26, 27, 29, 31
]


# ---------------------------------------------------------------------------
# Real Human3.6M dataset (raw expmap txt files)
# ---------------------------------------------------------------------------


class Human36MDataset(Dataset):
    """
    Sliding-window dataset over preprocessed Human3.6M sequences.

    Reads one .txt file per action-subaction, runs forward kinematics to
    obtain (T, 32, 3) joint positions, selects only the 22 joints listed in
    H36M_22_JOINT_INDICES, optionally downsamples to 25 fps, then slides a
    window of length (past_len + future_len) across each sequence with the
    given stride.

    Args:
        data_root: directory holding S1/, S5/, ... subfolders of .txt files.
        subjects: list of subjects to include (e.g. ['S1', 'S5']).
        past_len: number of past frames per window.
        future_len: number of future frames per window.
        stride: sliding-window stride (1 = every frame).
        center: subtract the root joint position at the last past frame.
        downsample: keep every k-th frame after FK (raw data is 50 fps; the
                    paper expects 25 fps, so downsample=2 by default).
    """

    def __init__(self, data_root: str, subjects: List[str],
                 past_len: int = 10, future_len: int = 25,
                 stride: int = 1, center: bool = True, downsample: int = 2):
        self.past_len = past_len
        self.future_len = future_len
        self.window = past_len + future_len
        self.stride = stride
        self.center = center
        self.joint_indices = H36M_22_JOINT_INDICES

        self.sequences: List[np.ndarray] = []
        self.index: List[Tuple[int, int]] = []

        for subj in subjects:
            patt = os.path.join(data_root, subj, "*.txt")
            files = sorted(glob.glob(patt))
            if not files:
                # Some preprocessed copies use a flat layout.
                patt2 = os.path.join(data_root, f"{subj}_*.txt")
                files = sorted(glob.glob(patt2))

            for f in files:
                expmap = np.loadtxt(f, delimiter=',').astype(np.float32)
                if expmap.ndim == 1 or expmap.shape[1] != 99:
                    print(f"[data] skipping {f}: unexpected shape {expmap.shape}")
                    continue
                if downsample > 1:
                    expmap = expmap[::downsample]

                # Run FK now and cache the xyz positions.
                xyz = expmap_seq_to_xyz(expmap)              # (T, 32, 3)
                xyz = xyz[:, self.joint_indices, :]          # (T, 22, 3)

                if xyz.shape[0] < self.window:
                    continue
                seq_id = len(self.sequences)
                self.sequences.append(xyz)
                for t in range(0, xyz.shape[0] - self.window + 1, self.stride):
                    self.index.append((seq_id, t))

        if not self.sequences:
            raise FileNotFoundError(
                f"No Human3.6M sequences found at {data_root} for subjects {subjects}. "
                f"Expected layout: {data_root}/S1/walking_1.txt, etc. "
                f"Set use_synthetic=True in the config to skip real data."
            )

        n_frames = sum(s.shape[0] for s in self.sequences)
        print(f"[data] loaded {len(self.sequences)} sequences "
              f"({n_frames} frames after downsample) → {len(self.index)} windows")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        seq_id, start = self.index[idx]
        seq = self.sequences[seq_id]
        clip = seq[start:start + self.window].copy()         # (T+L, 22, 3)

        if self.center:
            ref = clip[self.past_len - 1, 0:1, :].copy()     # (1, 3)
            clip = clip - ref[None, :, :]

        past = clip[:self.past_len]                          # (T, 22, 3)
        future = clip[self.past_len:]                        # (L, 22, 3)
        return torch.from_numpy(past), torch.from_numpy(future)


# ---------------------------------------------------------------------------
# Synthetic dataset (smoke testing)
# ---------------------------------------------------------------------------


class SyntheticMotionDataset(Dataset):
    """Slow sinusoids per joint. Only useful for testing the training loop."""

    def __init__(self, num_samples: int = 1024, num_joints: int = 22,
                 past_len: int = 10, future_len: int = 25, seed: int = 0):
        self.num_samples = num_samples
        self.past_len = past_len
        self.future_len = future_len
        self.rng = np.random.default_rng(seed)

        T = past_len + future_len
        self.clips = np.zeros((num_samples, T, num_joints, 3), dtype=np.float32)
        for i in range(num_samples):
            freqs = self.rng.uniform(0.05, 0.3, size=(num_joints, 3))
            phases = self.rng.uniform(0, 2 * math.pi, size=(num_joints, 3))
            amps = self.rng.uniform(50, 300, size=(num_joints, 3))
            base = self.rng.uniform(-500, 500, size=(num_joints, 3))
            t = np.arange(T)[:, None, None]
            self.clips[i] = base + amps * np.sin(2 * math.pi * freqs * t + phases)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        clip = self.clips[idx]
        return torch.from_numpy(clip[:self.past_len]), torch.from_numpy(clip[self.past_len:])


# ---------------------------------------------------------------------------
# Loader factory
# ---------------------------------------------------------------------------


def build_dataloaders(cfg, batch_size: int, num_workers: int = 4):
    if cfg.data.use_synthetic:
        train_set = SyntheticMotionDataset(
            num_samples=512, num_joints=len(H36M_22_JOINT_INDICES),
            past_len=cfg.data.past_len, future_len=cfg.data.future_len, seed=0,
        )
        val_set = SyntheticMotionDataset(
            num_samples=64, num_joints=len(H36M_22_JOINT_INDICES),
            past_len=cfg.data.past_len, future_len=cfg.data.future_len, seed=1,
        )
    else:
        train_set = Human36MDataset(
            data_root=cfg.data.data_root,
            subjects=cfg.data.train_subjects,
            past_len=cfg.data.past_len,
            future_len=cfg.data.future_len,
            stride=1,
            downsample=cfg.data.downsample,
        )
        val_set = Human36MDataset(
            data_root=cfg.data.data_root,
            subjects=cfg.data.test_subjects,
            past_len=cfg.data.past_len,
            future_len=cfg.data.future_len,
            stride=10,
            downsample=cfg.data.downsample,
        )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader