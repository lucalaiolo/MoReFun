"""
Fine-tune AuxFormer on preprocessed neonate keypoints.

Inputs
------
data/neonates_processed/{big_neonate,small_neonate}.npz   from preprocess_neonates.py
auxformer/ckpt/pretrain_h36m_ckpt_long.pth.tar             pretrained weights

What this does
--------------
1.  Loads the preprocessed neonate clips (T, 22, 3) in millimetres.
2.  Builds sliding windows of length 50 (= past 25 + future 25), stride 5, the
    same window the AuxFormer long-term model uses on H3.6M. The output target
    is the second half of the window with the AuxFormer downsampling logic
    (every other frame plus the last one => 13 frames).
3.  Splits each clip 80/20 into train/val by time (so the val portion is the
    final 20% of each clip, no leakage from sliding windows).
4.  Loads the pretrained long-term AuxFormer checkpoint, optionally freezes
    everything except the last encoder block and the heads, and fine-tunes
    with a low learning rate.
5.  Logs train + val loss and val MPJPE every epoch, saves the best checkpoint
    by val MPJPE, and produces training_curves.png at the end.

Usage
-----
    python finetune_auxformer_neonates.py
    python finetune_auxformer_neonates.py --epochs 30 --lr 1e-5 --freeze
    python finetune_auxformer_neonates.py --device cpu --epochs 3   # quick smoke
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from auxformer.model.model import AuxFormer
from run_auxformer import build_model, load_cfg, setup_seed


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class NeonateWindows(Dataset):
    """
    Sliding-window dataset over neonate clips.

    Each window is (50 frames, 22 joints, 3 coordinates) in millimetres. The
    past portion is the first 25 frames. The target ("ground-truth future")
    uses the H3.6M downsample rule: frames 26, 28, 30 ... 48, plus frame 49,
    giving 13 frames.

    The model input is the concatenation of past + zero-padded future, with
    the past divided by `scale` to match AuxFormer's expected units.
    """

    def __init__(self, clips: dict[str, np.ndarray], past: int = 25,
                  future: int = 13, stride: int = 5, scale: float = 100.0,
                  time_split: str = "train", val_frac: float = 0.2):
        self.past = past
        self.future = future
        self.scale = scale
        self.seq_len = past + 2 * future          # 25 + 26 = 51 frames? no, see below
        # The dataloader picks frames [1+input_n : input_n+output_n : 2] then
        # appends frame -1. That requires past=25 + 25 raw future frames = 50.
        self.raw_seq_len = past + 25              # 50

        self.windows = []                          # list of (clip_id, start_t)
        self.clip_ids = []
        self.clip_data = {}                        # clip_id -> (T, 22, 3) tensor

        for clip_id, (name, xyz) in enumerate(clips.items()):
            T = xyz.shape[0]
            if time_split == "train":
                T_end = int(T * (1 - val_frac))
                t_lo, t_hi = 0, T_end - self.raw_seq_len
            elif time_split == "val":
                T_start = int(T * (1 - val_frac))
                t_lo, t_hi = T_start, T - self.raw_seq_len
            else:
                raise ValueError(time_split)

            if t_hi < t_lo:
                continue
            for t in range(t_lo, t_hi + 1, stride):
                self.windows.append((clip_id, t))
            self.clip_ids.append(name)
            self.clip_data[clip_id] = torch.from_numpy(xyz.astype(np.float32))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        clip_id, t = self.windows[idx]
        xyz = self.clip_data[clip_id][t : t + self.raw_seq_len]   # (50, 22, 3)
        xyz_scaled = xyz / self.scale
        xyz_scaled = xyz_scaled.permute(1, 0, 2)                    # (22, 50, 3)

        past_xyz = xyz_scaled[:, : self.past]                       # (22, 25, 3)

        # H3.6M downsample-mode target
        future_idx = list(range(1 + self.past, self.past + 25, 2)) + [self.raw_seq_len - 1]
        # past=25, so range(26, 50, 2) = 26,28,...,48 (12 frames) + 49 = 13
        future_xyz = xyz_scaled[:, future_idx]                      # (22, 13, 3)

        return past_xyz, future_xyz


def load_clips(processed_dir: Path) -> dict[str, np.ndarray]:
    clips = {}
    for fp in sorted(processed_dir.glob("*.npz")):
        if fp.name == "meta.json":
            continue
        data = np.load(fp)
        clips[fp.stem] = data["xyz"]
    return clips


# ---------------------------------------------------------------------------
# Freezing helpers
# ---------------------------------------------------------------------------


def freeze_backbone(model: AuxFormer, unfreeze_n_blocks: int = 1) -> dict:
    """
    Freeze most of AuxFormer. Keeps trainable:
        - the last `unfreeze_n_blocks` encoder blocks
        - the decoder
        - the prediction head
        - the auxiliary heads
        - mask/agent/pos embeddings (they need to re-learn for the new domain)

    Returns a dict with statistics about what got frozen.
    """
    for p in model.parameters():
        p.requires_grad = False

    # Re-enable: embeddings, encoder's last n blocks, decoder, heads.
    trainables = []
    for name, p in model.named_parameters():
        if any(name.startswith(k) for k in [
            "mask_embed", "agent_embed", "pos_embed",
            "decoder_pos_embed", "decoder_agent_embed",
            "patch_embed", "head", "aux_head",
        ]):
            p.requires_grad = True
            trainables.append(name)
        elif "decoder" in name:
            p.requires_grad = True
            trainables.append(name)

    # Encoder blocks live as ModuleList; unfreeze the last `unfreeze_n_blocks`.
    if hasattr(model, "transformer_blocks"):
        blocks = model.transformer_blocks
        for i in range(max(0, len(blocks) - unfreeze_n_blocks), len(blocks)):
            for p in blocks[i].parameters():
                p.requires_grad = True
                trainables.extend(f"transformer_blocks.{i}.*" for _ in [0])

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": n_total,
        "trainable_params": n_trainable,
        "frac_trainable": n_trainable / n_total,
        "unfrozen_groups": sorted(set(trainables)),
    }


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def compute_loss(model_outputs: tuple, future_target: torch.Tensor,
                  past_target: torch.Tensor, cfg) -> dict:
    """
    Compute the same loss AuxFormer trained with:
        - prediction loss vs ground-truth future
        - mask reconstruction loss
        - denoising loss
    Returns a dict with components and the total.
    """
    loc_pred, mask_pred, mask_gt, denoised_pred, mask = model_outputs

    if cfg.multi_output:
        pred_loss = sum(
            torch.mean(torch.norm(item - future_target, dim=-1)) / cfg.encoder_depth
            for item in loc_pred
        )
    else:
        pred_loss = torch.mean(torch.norm(loc_pred - future_target, dim=-1))

    mask_loss = torch.sum(torch.norm(mask_pred - mask_gt, dim=-1, p=2)) / torch.sum(mask)

    if cfg.denoise_mode == "past":
        denoise_loss = torch.mean(torch.norm(denoised_pred - past_target, dim=-1))
    elif cfg.denoise_mode == "future":
        denoise_loss = torch.mean(torch.norm(denoised_pred - future_target, dim=-1))
    else:
        denoise_loss = torch.tensor(0.0, device=future_target.device)

    total = pred_loss + mask_loss + denoise_loss
    return {"total": total, "pred": pred_loss.detach(),
             "mask": mask_loss.detach(), "denoise": denoise_loss.detach()}


@torch.no_grad()
def evaluate(model, loader, device, scale, eval_frames=None):
    """Return mean MPJPE over the validation set per horizon, in mm."""
    if eval_frames is None:
        eval_frames = [0, 1, 3, 4, 6, 12]
    model.eval()
    sum_err = np.zeros(len(eval_frames))
    n = 0
    for past_xyz, future_xyz in loader:
        past_xyz = past_xyz.to(device)
        future_xyz = future_xyz.to(device)
        B = past_xyz.shape[0]
        fake_future = torch.zeros_like(future_xyz)
        all_traj = torch.cat([past_xyz, fake_future], dim=2)
        pred = model.predict(all_traj)           # (B, 22, future_len, 3)
        err = torch.norm(pred - future_xyz, dim=-1)   # (B, 22, future_len)
        err = err.mean(dim=1)                          # (B, future_len)
        err = err * scale                              # back to mm
        for i, f in enumerate(eval_frames):
            sum_err[i] += err[:, f].sum().item()
        n += B
    return sum_err / max(n, 1)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_training_curves(history: dict, out_path: Path, mpjpe_horizons_ms):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = np.arange(1, len(history["train_total"]) + 1)
    axes[0].plot(epochs, history["train_total"], label="train (total)", color="#3a7bd5")
    axes[0].plot(epochs, history["val_total"],   label="val (total)",   color="#f5a623")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Total loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_pred"], label="prediction", color="#3a7bd5")
    axes[1].plot(epochs, history["train_mask"], label="mask recon",  color="#7ed321")
    axes[1].plot(epochs, history["train_denoise"], label="denoise",  color="#d0021b")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("training loss components")
    axes[1].set_title("Training loss breakdown")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    mpjpe = np.array(history["val_mpjpe"])    # (epochs, H)
    cmap = plt.get_cmap("viridis", len(mpjpe_horizons_ms))
    for i, h in enumerate(mpjpe_horizons_ms):
        axes[2].plot(epochs, mpjpe[:, i], color=cmap(i), label=f"{h} ms")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("MPJPE (mm)")
    axes[2].set_title("Val MPJPE per horizon")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", type=str, default="data/neonates_processed")
    p.add_argument("--ckpt", type=str, default="pretrain_h36m_ckpt_long",
                   help="Pretrained checkpoint name under auxformer/ckpt/")
    p.add_argument("--out-dir", type=str, default="finetune_neonates")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5,
                   help="Lower than the pretraining LR (5e-4). Default 1e-5 "
                        "is a safe choice for fine-tuning on small data.")
    p.add_argument("--stride", type=int, default=5,
                   help="Sliding-window stride during training. Smaller = more "
                        "samples per epoch but more correlated.")
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="Last fraction of each clip held out for val.")
    p.add_argument("--freeze", action="store_true",
                   help="Freeze most of the backbone, fine-tune only the last "
                        "encoder block + decoder + heads.")
    p.add_argument("--unfreeze-n", type=int, default=1,
                   help="Number of encoder blocks to leave trainable when --freeze.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    setup_seed(args.seed)
    device = (torch.device(args.device) if args.device else
              torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[finetune] device = {device}")

    # Load clips
    processed_dir = Path(args.processed_dir)
    clips = load_clips(processed_dir)
    if not clips:
        sys.exit(f"No clips found in {processed_dir}. Run preprocess_neonates.py first.")
    print(f"[finetune] loaded clips: {list(clips.keys())}")
    for n, x in clips.items():
        print(f"  {n}: {x.shape[0]} frames, duration {x.shape[0]/25:.1f} s")

    # Datasets
    train_ds = NeonateWindows(clips, past=25, future=13, stride=args.stride,
                                scale=100.0, time_split="train",
                                val_frac=args.val_frac)
    val_ds = NeonateWindows(clips, past=25, future=13, stride=args.stride,
                              scale=100.0, time_split="val",
                              val_frac=args.val_frac)
    print(f"[finetune] train windows: {len(train_ds)}, val windows: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              drop_last=False, num_workers=0)

    # Model — long-term H3.6M config (past=25, future=13, encoder_depth=4)
    cfg = load_cfg("long")
    model = build_model(cfg, device)
    ckpt_path = Path("auxformer/ckpt") / f"{args.ckpt}.pth.tar"
    if not ckpt_path.exists():
        sys.exit(f"Pretrained checkpoint not found at {ckpt_path}")
    print(f"[finetune] loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    if args.freeze:
        info = freeze_backbone(model, unfreeze_n_blocks=args.unfreeze_n)
        print(f"[finetune] frozen mode: {info['trainable_params']/1e3:.1f}K trainable "
              f"of {info['total_params']/1e3:.1f}K total "
              f"({100*info['frac_trainable']:.1f}%)")
    else:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[finetune] full fine-tune: {n_train/1e3:.1f}K params trainable")

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=args.lr, weight_decay=cfg.weight_decay)

    # Eval frames: pick six points spanning short to long horizon (in 13-frame target).
    # In the downsampled target, frame i corresponds to (1+25 + i*2)/25 seconds since start of future.
    # Or: simply report at indices [0,1,3,4,6,12] which in real ms are [80, 160, 320, 400, 560, 1000] for H3.6M.
    eval_frames = [0, 1, 3, 4, 6, 12]
    horizons_ms = [80, 160, 320, 400, 560, 1000]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_total": [], "train_pred": [], "train_mask": [],
               "train_denoise": [], "val_total": [], "val_mpjpe": []}

    best_mpjpe = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, pre, msk, den, n_batches = 0.0, 0.0, 0.0, 0.0, 0
        for past_xyz, future_xyz in train_loader:
            past_xyz = past_xyz.to(device)
            future_xyz = future_xyz.to(device)
            all_traj = torch.cat([past_xyz, future_xyz], dim=2)
            outputs = model(all_traj)
            loss_dict = compute_loss(outputs, future_xyz, past_xyz, cfg)

            opt.zero_grad()
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            opt.step()

            tot += float(loss_dict["total"])
            pre += float(loss_dict["pred"])
            msk += float(loss_dict["mask"])
            den += float(loss_dict["denoise"])
            n_batches += 1

        train_total = tot / n_batches

        # Val: same loss + MPJPE
        model.eval()
        vtot, vn = 0.0, 0
        with torch.no_grad():
            for past_xyz, future_xyz in val_loader:
                past_xyz = past_xyz.to(device)
                future_xyz = future_xyz.to(device)
                all_traj = torch.cat([past_xyz, future_xyz], dim=2)
                outputs = model(all_traj)
                lossv = compute_loss(outputs, future_xyz, past_xyz, cfg)
                vtot += float(lossv["total"])
                vn += 1
        val_total = vtot / max(vn, 1)
        mpjpe = evaluate(model, val_loader, device, scale=100.0, eval_frames=eval_frames)

        history["train_total"].append(train_total)
        history["train_pred"].append(pre / n_batches)
        history["train_mask"].append(msk / n_batches)
        history["train_denoise"].append(den / n_batches)
        history["val_total"].append(val_total)
        history["val_mpjpe"].append(mpjpe.tolist())

        mean_mpjpe = float(np.mean(mpjpe))
        msg = f"[ep {epoch:3d}/{args.epochs}]  train {train_total:6.3f}  val {val_total:6.3f}  " \
              f"mpjpe {mean_mpjpe:6.2f} mm   per-horizon {np.round(mpjpe,1).tolist()}"
        print(msg)

        if mean_mpjpe < best_mpjpe:
            best_mpjpe = mean_mpjpe
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "val_mpjpe_mean": mean_mpjpe},
                       out_dir / "best.pth.tar")

    # Save final + history + curves
    torch.save({"state_dict": model.state_dict(), "epoch": args.epochs,
                "val_mpjpe_mean": float(np.mean(history["val_mpjpe"][-1]))},
               out_dir / "last.pth.tar")
    with open(out_dir / "history.json", "w") as f:
        json.dump({"history": history,
                   "horizons_ms": horizons_ms,
                   "best_val_mpjpe_mean": best_mpjpe,
                   "config": vars(args)}, f, indent=2)
    plot_training_curves(history, out_dir / "training_curves.png", horizons_ms)
    print(f"\n[finetune] best val MPJPE = {best_mpjpe:.2f} mm")
    print(f"[finetune] outputs in {out_dir}/")


if __name__ == "__main__":
    main()
