"""
Evaluation: Mean Per-Joint Position Error (MPJPE).

Standard metric for 3D motion prediction. Reported per-frame in millimetres.
Common timestamps at 25 fps are 80, 160, 320, 400, 560, 1000 ms, which
correspond to frames 2, 4, 8, 10, 14, 25.
"""

import torch


# Frame indices at 25 fps for the standard timestamp grid.
TIMESTAMPS_MS = [80, 160, 320, 400, 560, 720, 880, 1000]
FRAME_INDICES_25FPS = [2, 4, 8, 10, 14, 18, 22, 25]


def mpjpe(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean Per-Joint Position Error per frame.

    Args:
        pred: (B, L, J, K)
        target: (B, L, J, K)
    Returns:
        per-frame MPJPE of shape (L,) in the same units as the inputs (mm).
    """
    # Per-joint Euclidean distance, then mean over joints and batch.
    err = torch.linalg.norm(pred - target, dim=-1)  # (B, L, J)
    return err.mean(dim=(0, 2))  # (L,)


@torch.no_grad()
def evaluate_mpjpe(model, loader, device, future_len: int):
    """
    Run the model on a validation loader and return per-frame MPJPE.

    Args:
        model: a MoReFun instance.
        loader: validation dataloader yielding (past, future) tensors.
        device: torch device.
        future_len: how many future frames to predict.

    Returns:
        per-frame MPJPE tensor of length future_len, on CPU.
    """
    model.eval()
    total_err = torch.zeros(future_len, device=device)
    total_count = 0

    for past, future in loader:
        past = past.to(device, non_blocking=True)
        future = future.to(device, non_blocking=True)
        # Crop or pad to requested length.
        future = future[:, :future_len]

        pred = model(past, future_len=future_len)
        err = torch.linalg.norm(pred - future, dim=-1)  # (B, L, J)
        total_err += err.sum(dim=(0, 2))
        total_count += err.shape[0] * err.shape[2]

    return (total_err / total_count).cpu()


def format_mpjpe_table(per_frame: torch.Tensor, fps: int = 25) -> str:
    """Pretty-print MPJPE at the standard timestamp grid."""
    lines = ["Time (ms)   MPJPE (mm)"]
    L = per_frame.shape[0]
    for ms, idx in zip(TIMESTAMPS_MS, FRAME_INDICES_25FPS):
        if idx <= L:
            lines.append(f"  {ms:>5d}      {per_frame[idx - 1].item():>7.2f}")
    avg = per_frame.mean().item()
    lines.append(f"  avg        {avg:>7.2f}")
    return "\n".join(lines)
