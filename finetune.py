"""
Stage 2: fine-tuning for future motion prediction.

Loads the pretrained encoder/decoder/head from stage 1 and trains them on
the actual prediction task: given the complete past, predict the entire
future. The future input to the decoder is a tensor of zeros that picks up
only the position encodings inside the joint embedder.

Loss is straight MPJPE-like MSE (Eq. 10 motion term in the paper). The
text branch is omitted -- Human3.6M has no captions, and the paper's
caption experiments are on FineMotion.
"""

import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import Config
from model import MoReFun
from data import build_dataloaders
from eval import evaluate_mpjpe, format_mpjpe_table


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def finetune_loss(model: MoReFun, past: torch.Tensor, future: torch.Tensor):
    """
    Forward pass + MSE on predicted future motion.

    Args:
        past: (B, T, J, K)
        future: (B, L, J, K)
    Returns:
        scalar loss (mean MSE over all frames, joints, batch).
    """
    L = future.shape[1]
    pred = model(past, future_len=L)
    return F.mse_loss(pred, future), pred


def load_pretrained(model: MoReFun, ckpt_path: str, device):
    """Load weights from a pretraining checkpoint, with key compatibility check."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[finetune] missing keys: {missing}")
    if unexpected:
        print(f"[finetune] unexpected keys: {unexpected}")
    print(f"[finetune] loaded pretrain epoch {ckpt.get('epoch', '?')} "
          f"with val {ckpt.get('val', float('nan')):.4f}")


def main(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    Path(cfg.finetune.ckpt_dir).mkdir(parents=True, exist_ok=True)

    # Data.
    train_loader, val_loader = build_dataloaders(cfg, cfg.finetune.batch_size, num_workers=4)
    print(f"[finetune] train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    # Model.
    model = MoReFun(cfg).to(device)

    # Load pretrained weights.
    if os.path.exists(cfg.finetune.pretrain_ckpt):
        load_pretrained(model, cfg.finetune.pretrain_ckpt, device)
    else:
        print(f"[finetune] WARNING: pretrain checkpoint {cfg.finetune.pretrain_ckpt} "
              f"not found; training from scratch.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[finetune] trainable params: {n_params/1e6:.2f}M")

    # Optimiser.
    optim = Adam(model.parameters(), lr=cfg.finetune.lr,
                 weight_decay=cfg.finetune.weight_decay)
    sched = CosineAnnealingLR(
        optim, T_max=cfg.finetune.epochs * len(train_loader),
        eta_min=cfg.finetune.lr * cfg.finetune.min_lr_ratio,
    )

    best_val_mpjpe = float("inf")
    step = 0

    for epoch in range(cfg.finetune.epochs):
        model.train()
        t0 = time.time()
        for past, future in train_loader:
            past = past.to(device, non_blocking=True)
            future = future.to(device, non_blocking=True)

            loss, _ = finetune_loss(model, past, future)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            sched.step()

            if step % cfg.finetune.log_every == 0:
                print(
                    f"[finetune] ep {epoch:3d} step {step:6d}  "
                    f"loss {loss.item():.4f}  "
                    f"lr {optim.param_groups[0]['lr']:.2e}"
                )
            step += 1

        # Validation -- use real MPJPE in mm rather than MSE.
        if (epoch + 1) % cfg.finetune.val_every == 0:
            per_frame = evaluate_mpjpe(
                model, val_loader, device, future_len=cfg.data.future_len
            )
            avg = per_frame.mean().item()
            dt = time.time() - t0
            print(f"[finetune] === ep {epoch:3d} done in {dt:.1f}s  "
                  f"avg MPJPE {avg:.2f} mm ===")
            print(format_mpjpe_table(per_frame))

            if avg < best_val_mpjpe:
                best_val_mpjpe = avg
                ckpt = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "mpjpe": avg,
                    "per_frame_mpjpe": per_frame.tolist(),
                    "cfg": cfg.__dict__,
                }
                torch.save(ckpt, os.path.join(cfg.finetune.ckpt_dir, "best.pt"))
                print(f"[finetune] saved new best ({avg:.2f} mm)")

            torch.save({"epoch": epoch, "model": model.state_dict()},
                       os.path.join(cfg.finetune.ckpt_dir, "last.pt"))


if __name__ == "__main__":
    cfg = Config()
    main(cfg)
