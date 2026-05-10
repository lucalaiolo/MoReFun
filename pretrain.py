"""
Stage 1: pretraining.

Two proxy tasks per batch, run in parallel:

  Task A -- past self-reconstruction
    - Apply velocity-based mask to the past clip.
    - Reconstruct the original past from the masked version.
    - Loss: MSE between reconstruction and ground-truth past.
    - Touches: encoder + head.

  Task B -- past-guided future reconstruction
    - Encode the COMPLETE past (no mask) into H_P.
    - Apply velocity-based mask to the future clip.
    - Reconstruct the original future from H_P + masked future.
    - Loss: MSE between reconstruction and ground-truth future.
    - Touches: encoder + decoder + head.

The combined loss is:

    L_pretrain = MSE_past + alpha * MSE_future

with alpha = 1.0 by default.

This stage NEVER asks the model to predict from past alone. The future is
always partially visible during pretraining; only at fine-tuning does it
become fully unknown.
"""

import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import Config
from model import MoReFun
from data import build_dataloaders
from masking import velocity_based_mask, apply_mask


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pretrain_loss(model: MoReFun, past: torch.Tensor, future: torch.Tensor,
                  mask_rate: float, alpha: float):
    """
    Compute the combined past+future reconstruction loss for one batch.

    Args:
        past: (B, T, J, K)
        future: (B, L, J, K)
        mask_rate: fraction of joint-frames to mask in each clip.
        alpha: weight on the future reconstruction loss.

    Returns:
        (total_loss, loss_past, loss_future) -- all scalar tensors.
    """
    # --- Task A: past reconstruction ---
    keep_past = velocity_based_mask(past, mask_rate)
    past_masked = apply_mask(past, keep_past)
    past_rec = model.reconstruct_past(past_masked)
    loss_past = F.mse_loss(past_rec, past)

    # --- Task B: future reconstruction (past is unmasked here) ---
    keep_future = velocity_based_mask(future, mask_rate)
    future_masked = apply_mask(future, keep_future)
    future_rec = model.reconstruct_future(past, future_masked)
    loss_future = F.mse_loss(future_rec, future)

    total = loss_past + alpha * loss_future
    return total, loss_past, loss_future


@torch.no_grad()
def evaluate_pretrain(model: MoReFun, loader, device, mask_rate: float, alpha: float):
    """Run the pretraining loss over the validation set."""
    model.eval()
    total = 0.0
    total_past = 0.0
    total_future = 0.0
    n = 0
    for past, future in loader:
        past = past.to(device, non_blocking=True)
        future = future.to(device, non_blocking=True)
        loss, lp, lf = pretrain_loss(model, past, future, mask_rate, alpha)
        bs = past.shape[0]
        total += loss.item() * bs
        total_past += lp.item() * bs
        total_future += lf.item() * bs
        n += bs
    return total / n, total_past / n, total_future / n


def main(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Output directory.
    Path(cfg.pretrain.ckpt_dir).mkdir(parents=True, exist_ok=True)

    # Data.
    train_loader, val_loader = build_dataloaders(cfg, cfg.pretrain.batch_size, num_workers=4)
    print(f"[pretrain] train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    # Model.
    model = MoReFun(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[pretrain] trainable params: {n_params/1e6:.2f}M")

    # Optimiser + cosine annealing.
    optim = Adam(model.parameters(), lr=cfg.pretrain.lr,
                 weight_decay=cfg.pretrain.weight_decay)
    sched = CosineAnnealingLR(
        optim, T_max=cfg.pretrain.epochs * len(train_loader),
        eta_min=cfg.pretrain.lr * cfg.pretrain.min_lr_ratio,
    )

    best_val = float("inf")
    step = 0

    for epoch in range(cfg.pretrain.epochs):
        model.train()
        t0 = time.time()
        for past, future in train_loader:
            past = past.to(device, non_blocking=True)
            future = future.to(device, non_blocking=True)

            loss, lp, lf = pretrain_loss(
                model, past, future, cfg.mask.mask_rate, cfg.pretrain.alpha
            )

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            sched.step()

            if step % cfg.pretrain.log_every == 0:
                print(
                    f"[pretrain] ep {epoch:3d} step {step:6d}  "
                    f"loss {loss.item():.4f}  past {lp.item():.4f}  future {lf.item():.4f}  "
                    f"lr {optim.param_groups[0]['lr']:.2e}"
                )
            step += 1

        # Validation.
        if (epoch + 1) % cfg.pretrain.val_every == 0:
            val_total, val_past, val_future = evaluate_pretrain(
                model, val_loader, device, cfg.mask.mask_rate, cfg.pretrain.alpha
            )
            dt = time.time() - t0
            print(
                f"[pretrain] === ep {epoch:3d} done in {dt:.1f}s  "
                f"val total {val_total:.4f}  past {val_past:.4f}  future {val_future:.4f} ==="
            )

            # Save best checkpoint and a rolling 'last' copy.
            if val_total < best_val:
                best_val = val_total
                ckpt = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "val": val_total,
                    "cfg": cfg.__dict__,
                }
                torch.save(ckpt, os.path.join(cfg.pretrain.ckpt_dir, "best.pt"))
                print(f"[pretrain] saved new best ({val_total:.4f})")

            torch.save({"epoch": epoch, "model": model.state_dict()},
                       os.path.join(cfg.pretrain.ckpt_dir, "last.pt"))


if __name__ == "__main__":
    cfg = Config()
    main(cfg)
