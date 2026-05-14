"""
Extract AuxFormer encodings for every H3.6M test sample.

Pipeline (mirrors run_auxformer.py --mode test):
  1. Load the pretrained AuxFormer weights into the patched model.
  2. Walk the 15 H3.6M test actions (subject S5, sliding-window samples).
  3. For each batch, call `mask_forward` and grab the (B, N, T, F) feature
     tensor that lives just before the prediction head.
  4. Pool it down to a per-sample vector (default = average over the past
     frames and all joints), or save the raw tensor.
  5. Write three files into the output directory:
       encodings.npy  — float32 array, shape depends on --pool
       labels.npy     — object array of action names, one per sample
       meta.json      — task, checkpoint, pooling mode, shape, feature dim

Usage:
    # short-term encodings, mean-pooled over past frames + joints
    python extract_encodings.py --task short

    # long-term, no pooling — keeps full (N, T, F) tensor per sample
    python extract_encodings.py --task long --pool none

    # custom output dir, custom data root
    python extract_encodings.py --task short --pool mean_joints \
        --data-root /path/to/h3.6m/dataset \
        --out-dir encodings/h36m_short_per_frame/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from auxformer.dataset.dataloader import H36motion3D
from auxformer.model.model import AuxFormer
from run_auxformer import (ACTIONS, DEFAULT_CKPT, build_model, load_cfg,
                            setup_seed)


# --------------------------------------------------------------------------
# Pooling
# --------------------------------------------------------------------------


def pool_features(features: torch.Tensor, past_len: int, mode: str) -> torch.Tensor:
    """
    features: (B, N, T, F)
    past_len: number of observed timestamps; the rest are future placeholders
    mode:
        - 'none'        -> (B, N, T, F)
        - 'mean'        -> (B, F)               mean over N, T
        - 'mean_past'   -> (B, F)               mean over N and past frames only
        - 'mean_joints' -> (B, T, F)            mean over joints, keep time
        - 'mean_time'   -> (B, N, F)            mean over time, keep joints
        - 'mean_time_past' -> (B, N, F)         mean over past frames only, keep joints
        - 'flatten'     -> (B, N*T*F)           flat per-sample vector
        - 'flatten_past'-> (B, N*past_len*F)    flat, observed part only
    """
    if mode == "none":
        return features
    if mode == "mean":
        return features.mean(dim=(1, 2))
    if mode == "mean_past":
        return features[:, :, :past_len].mean(dim=(1, 2))
    if mode == "mean_joints":
        return features.mean(dim=1)
    if mode == "mean_time":
        return features.mean(dim=2)
    if mode == "mean_time_past":
        return features[:, :, :past_len].mean(dim=2)
    if mode == "flatten":
        return features.flatten(start_dim=1)
    if mode == "flatten_past":
        return features[:, :, :past_len].flatten(start_dim=1)
    raise ValueError(f"unknown pool mode: {mode}")


# --------------------------------------------------------------------------
# The core hook: call mask_forward and stop before the head
# --------------------------------------------------------------------------


def extract_features_batch(model: AuxFormer, all_traj: torch.Tensor) -> torch.Tensor:
    """
    Run mask_forward exactly the way predict() does, return raw (B, N, T, F)
    features. No head applied.
    """
    B, N, T = all_traj.shape[0], all_traj.shape[1], all_traj.shape[2]
    all_traj = all_traj.view(B, N, T, 3)

    # Same mask as in predict(): 1 for observed past frames, 0 for future.
    ordinary_mask = torch.zeros((B, N, T), device=all_traj.device)
    ordinary_mask[:, :, : model.past_timestep] = 1.0

    if model.multi_output:
        # mask_forward returns a list of L tensors when multi_output=True and
        # all_out=True. The last one is the "final" encoding (what predict()
        # feeds into the head).
        feats = model.mask_forward(all_traj, ordinary_mask, all_out=True)[-1]
    else:
        feats = model.mask_forward(all_traj, ordinary_mask)

    return feats   # (B, N, T, F)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="Extract AuxFormer encodings on H3.6M test.")
    p.add_argument("--task", choices=["short", "long"], default="short")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Checkpoint name in auxformer/ckpt/ (no .pth.tar). "
                        "Defaults to the official pretrained file for the task.")
    p.add_argument("--data-root", type=str, default=None,
                   help="Path to h3.6m/dataset/. Falls back to H36M_DATA_ROOT.")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Where to write encodings.npy / labels.npy / meta.json. "
                        "Default: encodings/h36m_{task}_{pool}/")
    p.add_argument("--pool", default="mean_past",
                   choices=["none", "mean", "mean_past", "mean_joints",
                            "mean_time", "mean_time_past", "flatten",
                            "flatten_past"],
                   help="How to summarize the (B, N, T, F) feature tensor.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override config's batch size for extraction.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.data_root is not None:
        os.environ["H36M_DATA_ROOT"] = args.data_root
    if args.ckpt is None:
        args.ckpt = DEFAULT_CKPT[args.task]
    if args.out_dir is None:
        args.out_dir = f"encodings/h36m_{args.task}_{args.pool}"

    device = (torch.device(args.device) if args.device else
              torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    cfg = load_cfg(args.task)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    setup_seed(args.seed)

    # 1. Load pretrained model
    model = build_model(cfg, device)
    ckpt_path = Path(__file__).resolve().parent / "auxformer" / "ckpt" / (args.ckpt + ".pth.tar")
    if not ckpt_path.exists():
        sys.exit(f"checkpoint not found: {ckpt_path}\n"
                 "  Run `python download_auxformer_ckpts.py` first.")
    print(f"[extract] loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[extract] device: {device}, params: {n_params:.2f}M, "
          f"past={cfg.past_length}, future={cfg.future_length}, "
          f"feature dim F={cfg.nf}")
    print(f"[extract] pool mode: {args.pool}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Walk the test split
    all_encodings = []
    all_labels = []
    all_sample_idx = []

    n_total = 0
    for act in ACTIONS:
        ds = H36motion3D(
            actions=act,
            input_n=cfg.past_length,
            output_n=cfg.future_length,
            split=1,
            scale=cfg.scale,
            path_to_data=os.environ.get("H36M_DATA_ROOT"),
        )
        loader = torch.utils.data.DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
            num_workers=2,
        )

        action_count = 0
        with torch.no_grad():
            for data in loader:
                data = [d.to(device) for d in data]
                loc, _vel, loc_end, _loc_end_ori, item_idx = data

                # Same construction as predict(): future is a zero placeholder.
                loc_end_fake = torch.zeros_like(loc_end)
                all_traj = torch.cat([loc, loc_end_fake], dim=2)

                feats = extract_features_batch(model, all_traj)
                pooled = pool_features(feats, cfg.past_length, args.pool)

                all_encodings.append(pooled.cpu().numpy())
                B = pooled.shape[0]
                all_labels.extend([act] * B)
                all_sample_idx.extend(item_idx.cpu().numpy().tolist())
                action_count += B

        print(f"  {act:<16} {action_count:5d} samples")
        n_total += action_count

    # 3. Concatenate and save
    encodings = np.concatenate(all_encodings, axis=0)
    labels = np.array(all_labels, dtype=object)
    sample_idx = np.asarray(all_sample_idx, dtype=np.int64)

    np.save(out_dir / "encodings.npy", encodings)
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "sample_idx.npy", sample_idx)

    meta = {
        "task": args.task,
        "checkpoint": args.ckpt,
        "pool_mode": args.pool,
        "n_samples": int(n_total),
        "encoding_shape_per_sample": list(encodings.shape[1:]),
        "encoding_dtype": str(encodings.dtype),
        "feature_dim_F": int(cfg.nf),
        "past_length": int(cfg.past_length),
        "future_length": int(cfg.future_length),
        "n_joints": int(encodings.shape[1]) if args.pool in ("mean_time", "mean_time_past") else 22,
        "actions": ACTIONS,
        "data_root": os.environ.get("H36M_DATA_ROOT", "data/h3.6m/dataset"),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print()
    print(f"[extract] wrote {n_total} samples to {out_dir}/")
    print(f"  encodings.npy  shape={encodings.shape}  dtype={encodings.dtype}")
    print(f"  labels.npy     shape={labels.shape}     (15 unique actions)")
    print(f"  sample_idx.npy shape={sample_idx.shape}")
    print(f"  meta.json")
    print()
    # Quick sanity print
    by_action = {act: int((labels == act).sum()) for act in ACTIONS}
    print("[extract] per-action counts:")
    for act, c in by_action.items():
        print(f"  {act:<16} {c}")


if __name__ == "__main__":
    main()
