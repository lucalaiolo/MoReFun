"""
Extract encodings for the neonate clips and visualize them with UMAP.

What this does
--------------
1.  Loads the preprocessed neonate clips and a checkpoint (either the original
    pretrained AuxFormer or a fine-tuned one from finetune_auxformer_neonates).
2.  Slides a window over each clip with the same convention used in fine-tuning
    (past=25, future=25 raw, fed as zeros, total T=50; only past is real signal).
3.  Extracts AuxFormer's mask_forward features and pools to `mean_past`
    (one F-dim vector per window).
4.  Computes a per-window velocity scalar = mean joint speed over the past
    window in mm/frame.
5.  Runs UMAP and writes a 2D scatter where:
       marker shape  = which video the window came from
       marker color  = window velocity (viridis colourmap)

Outputs (under <out-dir>)
-------------------------
    encodings.npy        (N_windows, F)  float32
    labels.npz           {video: array of strings, velocity: array of floats,
                          window_start: array of ints, ...}
    umap_coords.npy      (N_windows, 2) float32
    umap.png             the scatter plot

Usage
-----
    python umap_neonates.py                                          # default: pretrained checkpoint
    python umap_neonates.py --ckpt finetune_neonates/best.pth.tar    # use fine-tuned weights
    python umap_neonates.py --stride 5                               # denser sampling
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_auxformer import build_model, load_cfg, setup_seed
from finetune_auxformer_neonates import load_clips
from extract_encodings import extract_features_batch, pool_features


# ---------------------------------------------------------------------------
# Sliding window over a single clip
# ---------------------------------------------------------------------------


def iter_windows(xyz: np.ndarray, past: int, raw_future: int, stride: int):
    """Yield (start_t, past_xyz, vel) for every valid window in a single clip.
    `vel` is the mean joint speed over the past window (mm/frame)."""
    seq_len = past + raw_future
    T = xyz.shape[0]
    if T < seq_len:
        return
    for t in range(0, T - seq_len + 1, stride):
        past_xyz = xyz[t : t + past]                  # (past, 22, 3)
        # speed = mean over joints and adjacent frames
        diffs = np.linalg.norm(past_xyz[1:] - past_xyz[:-1], axis=-1)   # (past-1, 22)
        v = float(diffs.mean())                       # mm/frame
        yield t, past_xyz, v


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", type=str, default="data/neonates_processed")
    p.add_argument("--ckpt", type=str, default="auxformer/ckpt/pretrain_h36m_ckpt_long.pth.tar",
                   help="Path to a checkpoint. Pass the pretrained one for baseline encodings, "
                        "or a fine-tuned best.pth.tar to see how the encoder shifted.")
    p.add_argument("--out-dir", type=str, default="umap_neonates")
    p.add_argument("--stride", type=int, default=10,
                   help="Sliding-window stride in frames. Smaller -> more points "
                        "in UMAP but more correlated.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--scale", type=float, default=100.0,
                   help="Same scaling factor used during training.")
    p.add_argument("--umap-neighbors", type=int, default=30)
    p.add_argument("--umap-min-dist", type=float, default=0.15)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    setup_seed(args.seed)
    device = (torch.device(args.device) if args.device else
              torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Load clips
    clips = load_clips(Path(args.processed_dir))
    if not clips:
        sys.exit(f"No clips in {args.processed_dir}")
    print(f"[umap] clips: {list(clips.keys())}")

    # Load model
    cfg = load_cfg("long")
    model = build_model(cfg, device)
    ckpt_path = Path(args.ckpt)
    print(f"[umap] loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    past = cfg.past_length            # 25
    raw_future = 25                   # raw future frames the AuxFormer dataloader uses
    seq_len = past + raw_future       # 50

    # Iterate windows over all clips, batch into the model
    all_encs, video_labels, velocities, starts = [], [], [], []
    pending = []                       # batch buffer
    for video_name, xyz in clips.items():
        for t, past_xyz, v in iter_windows(xyz, past=past, raw_future=raw_future,
                                            stride=args.stride):
            pending.append((video_name, t, past_xyz, v))
            if len(pending) >= args.batch_size:
                _flush(model, pending, all_encs, video_labels, velocities, starts,
                        device, args.scale, past)
                pending = []
    if pending:
        _flush(model, pending, all_encs, video_labels, velocities, starts,
                device, args.scale, past)

    encs = np.concatenate(all_encs, axis=0)
    video_labels = np.array(video_labels, dtype=object)
    velocities = np.array(velocities, dtype=np.float32)
    starts = np.array(starts, dtype=np.int64)
    print(f"[umap] extracted {len(encs)} windows, encoding dim = {encs.shape[1]}")
    print(f"  velocity range: [{velocities.min():.2f}, {velocities.max():.2f}] mm/frame")
    for v in np.unique(video_labels):
        n = int((video_labels == v).sum())
        print(f"  {v}: {n} windows")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "encodings.npy", encs)
    np.savez(out_dir / "labels.npz",
              video=video_labels, velocity=velocities, window_start=starts)

    # UMAP
    import umap
    from sklearn.preprocessing import StandardScaler
    print(f"[umap] running UMAP (n_neighbors={args.umap_neighbors}, "
          f"min_dist={args.umap_min_dist})...")
    encs_scaled = StandardScaler().fit_transform(encs)
    reducer = umap.UMAP(n_neighbors=args.umap_neighbors,
                         min_dist=args.umap_min_dist,
                         n_components=2,
                         random_state=args.seed,
                         verbose=False)
    coords = reducer.fit_transform(encs_scaled)
    np.save(out_dir / "umap_coords.npy", coords)

    # Scatter
    fig, ax = plt.subplots(figsize=(11, 8))
    # Cap velocity for colormap so a few outliers don't dominate.
    v_lo, v_hi = np.percentile(velocities, [2, 98])
    norm = matplotlib.colors.Normalize(vmin=v_lo, vmax=v_hi)
    cmap = plt.get_cmap("viridis")

    markers = ['o', 's', 'D', '^', 'v', 'P', 'X']
    unique_videos = sorted(np.unique(video_labels).tolist())
    for vi, v in enumerate(unique_videos):
        m = video_labels == v
        sc = ax.scatter(coords[m, 0], coords[m, 1],
                         c=velocities[m], cmap=cmap, norm=norm,
                         s=25, alpha=0.7,
                         marker=markers[vi % len(markers)],
                         edgecolors="black", linewidths=0.25,
                         label=v)

    cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("mean joint speed (mm / frame in 25 fps)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ckpt_name = ckpt_path.name
    ax.set_title(f"AuxFormer encodings of neonates ({ckpt_name})  "
                  f"n = {len(encs)}")
    ax.legend(loc="best", fontsize=10, markerscale=2.0,
               handletextpad=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "umap.png", dpi=150)
    plt.close(fig)

    # Also write a summary
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "checkpoint": str(ckpt_path),
            "n_windows_total": int(len(encs)),
            "encoding_dim": int(encs.shape[1]),
            "stride": args.stride,
            "videos": {v: int((video_labels == v).sum())
                        for v in unique_videos},
            "velocity_stats": {"min": float(velocities.min()),
                                "p50": float(np.median(velocities)),
                                "max": float(velocities.max())},
            "umap": {"n_neighbors": args.umap_neighbors,
                      "min_dist": args.umap_min_dist},
        }, f, indent=2)
    print(f"[umap] wrote everything to {out_dir}/")


def _flush(model, pending, all_encs, video_labels, velocities, starts,
            device, scale, past):
    """Process one batch worth of windows through the model."""
    names = [x[0] for x in pending]
    ts = [x[1] for x in pending]
    past_arr = np.stack([x[2] for x in pending])                  # (B, past, 22, 3)
    vs = [x[3] for x in pending]

    past_t = torch.from_numpy(past_arr / scale).float().to(device)
    past_t = past_t.permute(0, 2, 1, 3)                            # (B, 22, past, 3)
    # zero-padded future of length raw_future = 25 to make T = 50 = past + 25.
    # But the model's past_timestep is 25 and future_timestep is 13.
    # Total T expected = 38.
    # We need to feed a tensor of (B, 22, 38, 3) where future part is zeros.
    B, N, _, _ = past_t.shape
    full_T = 25 + 13     # 38
    all_traj = torch.zeros(B, N, full_T, 3, device=device)
    all_traj[:, :, :25] = past_t

    with torch.no_grad():
        feats = extract_features_batch(model, all_traj)            # (B, 22, 38, F)
        pooled = pool_features(feats, past_len=25, mode="mean_past")   # (B, F)

    all_encs.append(pooled.cpu().numpy().astype(np.float32))
    video_labels.extend(names)
    velocities.extend(vs)
    starts.extend(ts)


if __name__ == "__main__":
    main()
