"""
Velocity-based joint masking (Eq. 4 of the paper).

The mask preferentially deletes high-motion joint-frames so that pretraining
loss concentrates on the entries that actually carry information about what
the action is. Spine/hip joints rarely move and rarely get masked; wrists,
ankles, and fingers move a lot and get masked often.
"""

import torch


def velocity_based_mask(x: torch.Tensor, mask_rate: float = 0.75) -> torch.Tensor:
    """
    Build a binary keep-mask using per-joint speed.

    Args:
        x: motion tensor of shape (B, T, J, K) with K coordinate components.
        mask_rate: fraction of joint-frames to delete (0..1).

    Returns:
        keep: tensor of shape (B, T, J) with values in {0, 1}, where 1 means
              "keep this joint-frame" and 0 means "delete it".

    Notes:
        - Speed at frame t is computed as ||x[t] - x[t-1]||_2 over the K axis.
          Frame 0 has no previous frame, so we always keep it.
        - The threshold is chosen per-batch-element so that exactly
          mask_rate of the maskable entries (frames 1..T-1) get a 0.
        - The mask is applied identically to all K coordinates of a joint-frame:
          if (t, j) is masked, all its coordinates are zeroed.
    """
    B, T, J, K = x.shape
    assert 0.0 <= mask_rate <= 1.0

    # Speed: (B, T-1, J). Norm over the coordinate axis.
    velocity = x[:, 1:, :, :] - x[:, :-1, :, :]  # (B, T-1, J, K)
    speed = torch.linalg.norm(velocity, dim=-1)  # (B, T-1, J)

    # Per-batch threshold so that mask_rate of (T-1)*J entries are above it.
    flat = speed.reshape(B, -1)  # (B, (T-1)*J)
    n_total = flat.shape[1]
    n_mask = int(round(mask_rate * n_total))

    if n_mask == 0:
        # Keep everything.
        keep_motion = torch.ones_like(speed)
    elif n_mask >= n_total:
        # Mask everything maskable.
        keep_motion = torch.zeros_like(speed)
    else:
        # We want to mask the top n_mask entries by speed (so keep the bottom).
        # topk on -speed gives the k smallest speeds; those are kept.
        n_keep = n_total - n_mask
        # Indices of entries to keep, per batch element.
        _, keep_idx = torch.topk(flat, k=n_keep, dim=1, largest=False, sorted=False)
        keep_flat = torch.zeros_like(flat)
        keep_flat.scatter_(1, keep_idx, 1.0)
        keep_motion = keep_flat.reshape(B, T - 1, J)

    # Pad with a "keep" column at frame 0 so the mask aligns with x.
    keep_first = torch.ones(B, 1, J, device=x.device, dtype=keep_motion.dtype)
    keep = torch.cat([keep_first, keep_motion], dim=1)  # (B, T, J)
    return keep


def apply_mask(x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """
    Zero out the masked joint-frames.

    Args:
        x: (B, T, J, K)
        keep: (B, T, J) with values in {0, 1}

    Returns:
        Masked tensor of the same shape as x.
    """
    return x * keep.unsqueeze(-1)


def random_mask(x: torch.Tensor, mask_rate: float = 0.75) -> torch.Tensor:
    """
    Random masking baseline (Table 8 row A in the paper).

    Useful for ablations: this is what the paper compares against to show
    that velocity-based masking is better.
    """
    B, T, J, _ = x.shape
    n_total = (T - 1) * J
    n_mask = int(round(mask_rate * n_total))

    keep_first = torch.ones(B, 1, J, device=x.device)
    if n_mask == 0:
        keep_rest = torch.ones(B, T - 1, J, device=x.device)
    elif n_mask >= n_total:
        keep_rest = torch.zeros(B, T - 1, J, device=x.device)
    else:
        # Random permutation per batch element.
        n_keep = n_total - n_mask
        scores = torch.rand(B, n_total, device=x.device)
        _, keep_idx = torch.topk(scores, k=n_keep, dim=1, largest=False, sorted=False)
        keep_flat = torch.zeros(B, n_total, device=x.device)
        keep_flat.scatter_(1, keep_idx, 1.0)
        keep_rest = keep_flat.reshape(B, T - 1, J)

    return torch.cat([keep_first, keep_rest], dim=1)
