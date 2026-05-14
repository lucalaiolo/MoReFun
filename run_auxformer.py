"""
Run AuxFormer on Human3.6M.

This is a thin wrapper around the original AuxFormer test_h36m.py, with:
  - configurable H3.6M path (the same one MoReFun uses, defaulting to
    `data/h3.6m/dataset` and override-able via --data-root or H36M_DATA_ROOT)
  - device-aware model (no hardcoded .cuda() so it also runs on CPU/MPS)
  - the pretrained checkpoint paths from auxformer/ckpt/

Usage:
    # short-term (predict 400 ms ahead): uses pretrain_h36m_ckpt.pth.tar
    python run_auxformer.py --mode test --task short

    # long-term (predict up to 1000 ms): uses pretrain_h36m_ckpt_long.pth.tar
    python run_auxformer.py --mode test --task long

    # custom data location
    python run_auxformer.py --mode test --task short --data-root /path/to/h3.6m/dataset

    # train from scratch
    python run_auxformer.py --mode train --task short
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import optim

# Make `auxformer.*` imports resolvable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from auxformer.dataset.dataloader import H36motion3D
from auxformer.model.model import AuxFormer


ACTIONS = ["walking", "eating", "smoking", "discussion", "directions",
           "greeting", "phoning", "posing", "purchases", "sitting",
           "sittingdown", "takingphoto", "waiting", "walkingdog",
           "walkingtogether"]

# Default checkpoint names per task. These are the official files released
# alongside the paper. Download them with `python download_auxformer_ckpts.py`.
DEFAULT_CKPT = {
    "short": "pretrain_h36m_ckpt",
    "long": "pretrain_h36m_ckpt_long",
}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_cfg(task: str) -> argparse.Namespace:
    """Read auxformer/cfg/h36m_{task}.yml and return as a Namespace."""
    cfg_path = Path(__file__).resolve().parent / "auxformer" / "cfg" / f"h36m_{task}.yml"
    with open(cfg_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    ns = argparse.Namespace(**cfg_dict)
    return ns


def build_model(cfg, device) -> AuxFormer:
    model = AuxFormer(
        in_dim=2,
        h_dim=cfg.nf,
        past_timestep=cfg.past_length,
        future_timestep=cfg.future_length,
        mask_ratio=cfg.mask_ratio,
        decoder_dim=cfg.decoder_dim,
        num_heads=8,
        encoder_depth=cfg.encoder_depth,
        decoder_depth=cfg.decoder_depth,
        decoder_dim_per_head=cfg.dim_per_head,
        same_head=cfg.same_head,
        range_mask_ratio=cfg.range_mask_ratio,
        mlp_head=cfg.mlp_head,
        mask_past=cfg.mask_past,
        mask_range=cfg.mask_range,
        multi_output=cfg.multi_output,
        decoder_masking=cfg.decoder_masking,
        pred_all=cfg.pred_all,
        mlp_dim=cfg.mlp_dim,
        dim_per_head=cfg.dim_per_head,
        noise_dev=cfg.noise_dev,
        part_noise=cfg.part_noise,
        denoise_mode=cfg.denoise_mode,
        part_noise_ratio=cfg.part_noise_ratio,
        add_joint_token=cfg.add_joint_token,
        n_agent=22,
        concat_vel=cfg.concat_vel,
        only_recons_past=cfg.only_recons_past,
        add_residual=cfg.add_residual,
        denoise=cfg.denoise,
        regular_masking=cfg.regular_masking,
        multi_same_head=cfg.multi_same_head,
        range_noise_dev=cfg.range_noise_dev,
    )
    return model.to(device)


# ---------------------------------------------------------------------------
# Evaluation (same logic as original test_h36m.py, with device awareness)
# ---------------------------------------------------------------------------


def evaluate_action(model, loader, cfg, device):
    output_n = cfg.future_length
    if output_n == 25:
        eval_frame = [1, 3, 7, 9, 13, 24]
    elif output_n == 15:
        eval_frame = [3, 14]
    elif output_n == 10:
        eval_frame = [1, 3, 7, 9]
    elif output_n == 13:
        eval_frame = [0, 1, 3, 4, 6, 12]
    else:
        raise ValueError(f"Unsupported future_length {output_n}")

    t_3d = np.zeros(len(eval_frame))
    counter = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data = [d.to(device) for d in data]
            loc, vel, loc_end, loc_end_ori, _ = data
            batch_size, n_nodes = loc.shape[0], loc.shape[1]
            pred_length = loc_end.shape[2]

            loc_end_fake = torch.zeros_like(loc_end)
            all_traj = torch.cat([loc, loc_end_fake], dim=2)
            loc_pred = model.predict(all_traj)                # (B, N, T_future, 3)

            pred_3d = loc_end_ori.clone()
            loc_pred = loc_pred.transpose(1, 2)
            loc_pred = loc_pred.contiguous().view(batch_size, pred_length, n_nodes * 3)

            joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
            index_to_ignore = np.concatenate(
                (joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2)
            )
            joint_equal = np.array([13, 19, 22, 13, 27, 30])
            index_to_equal = np.concatenate(
                (joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2)
            )

            pred_3d[:, :, loader.dataset.dim_used] = loc_pred
            pred_3d[:, :, index_to_ignore] = pred_3d[:, :, index_to_equal]
            pred_p3d = pred_3d.contiguous().view(batch_size, pred_length, -1, 3)
            targ_p3d = loc_end_ori.contiguous().view(batch_size, pred_length, -1, 3)

            for k, j in enumerate(eval_frame):
                err = torch.norm(
                    targ_p3d[:, j].contiguous().view(-1, 3)
                    - pred_p3d[:, j].contiguous().view(-1, 3),
                    p=2, dim=1,
                )
                t_3d[k] += err.mean().item() * batch_size
            counter += batch_size

    t_3d *= cfg.scale
    return t_3d / counter, eval_frame


def cmd_test(args, cfg, device):
    """Evaluate a pretrained AuxFormer model on all 15 H3.6M actions."""
    setup_seed(args.seed)

    model = build_model(cfg, device)
    ckpt_path = Path(__file__).resolve().parent / "auxformer" / "ckpt" / (args.ckpt + ".pth.tar")
    if not ckpt_path.exists():
        print(f"[run_auxformer] checkpoint not found: {ckpt_path}")
        print("  Run `python download_auxformer_ckpts.py` first, or set --ckpt to a "
              "different file in auxformer/ckpt/.")
        sys.exit(1)
    print(f"[run_auxformer] loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    print(f"[run_auxformer] params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"[run_auxformer] device: {device}")
    print(f"[run_auxformer] H3.6M root: {os.environ.get('H36M_DATA_ROOT', 'data/h3.6m/dataset')}")

    horizons_ms = {
        "short": [80, 160, 320, 400],
        "long":  [80, 160, 320, 400, 560, 1000],
    }[args.task]

    avg = np.zeros(len(horizons_ms))
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
            ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=2,
        )
        mpjpe, _ = evaluate_action(model, loader, cfg, device)
        cells = "  ".join(f"{m:7.2f}" for m in mpjpe)
        print(f"  {act:<16} | {cells} | avg {mpjpe.mean():7.2f}")
        avg += mpjpe

    avg /= len(ACTIONS)
    print()
    header = "  ".join(f"{h:>5d}ms" for h in horizons_ms)
    cells = "  ".join(f"{m:7.2f}" for m in avg)
    print(f"  {'AVG':<16} | {cells} | avg {avg.mean():7.2f}")
    print(f"  ({header})")


def cmd_train(args, cfg, device):
    """Train AuxFormer from scratch with the original combined loss."""
    setup_seed(args.seed)

    print("[run_auxformer] training from scratch.")
    print(f"  task: {args.task}, past={cfg.past_length}, future={cfg.future_length}")
    print(f"  device: {device}")
    print(f"  H3.6M root: {os.environ.get('H36M_DATA_ROOT', 'data/h3.6m/dataset')}")

    train_ds = H36motion3D(
        actions="walking" if args.debug else "all",
        input_n=cfg.past_length, output_n=cfg.future_length,
        split=0, scale=cfg.scale, path_to_data=os.environ.get("H36M_DATA_ROOT"),
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=2,
    )

    test_loaders = {}
    for act in ACTIONS:
        ds = H36motion3D(
            actions=act, input_n=cfg.past_length, output_n=cfg.future_length,
            split=1, scale=cfg.scale, path_to_data=os.environ.get("H36M_DATA_ROOT"),
        )
        test_loaders[act] = torch.utils.data.DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=2,
        )

    model = build_model(cfg, device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    print(f"  params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    ckpt_dir = Path(__file__).resolve().parent / "auxformer" / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_name = args.save_name or f"trained_h36m_{args.task}"
    best_loss = float("inf")
    lr_now = cfg.lr

    for epoch in range(cfg.epochs):
        if cfg.apply_decay and epoch % cfg.epoch_decay == 0 and epoch > 0:
            lr_now = lr_now * cfg.lr_gamma
            for g in optimizer.param_groups:
                g["lr"] = lr_now

        model.train()
        running, count = 0.0, 0
        for data in train_loader:
            data = [d.to(device) for d in data]
            loc, vel, loc_end, _, _ = data
            all_traj = torch.cat([loc, loc_end], dim=2)

            loc_pred, mask_pred, mask_gt, denoised_pred, mask = model(all_traj)

            if cfg.multi_output:
                loss = sum(
                    torch.mean(torch.norm(item - loc_end, dim=-1)) / cfg.encoder_depth
                    for item in loc_pred
                )
            else:
                loss = torch.mean(torch.norm(loc_pred - loc_end, dim=-1))

            loss = loss + torch.sum(torch.norm(mask_pred - mask_gt, dim=-1, p=2)) / torch.sum(mask)
            if cfg.denoise_mode == "past":
                loss = loss + torch.mean(torch.norm(denoised_pred - loc, dim=-1))
            elif cfg.denoise_mode == "future":
                loss = loss + torch.mean(torch.norm(denoised_pred - loc_end, dim=-1))
            elif cfg.denoise_mode == "all":
                loss = loss + torch.mean(torch.norm(denoised_pred - all_traj, dim=-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item() * loc.shape[0]
            count += loc.shape[0]

        print(f"[train] epoch {epoch+1:3d}/{cfg.epochs}  loss {running / count:.4f}  lr {lr_now:.2e}")

        # eval each test_interval epochs
        if (epoch + 1) % cfg.test_interval == 0:
            avg = None
            for act in ACTIONS:
                mpjpe, _ = evaluate_action(model, test_loaders[act], cfg, device)
                avg = mpjpe if avg is None else avg + mpjpe
            avg = avg / len(ACTIONS)
            mean_mpjpe = float(avg.mean())
            print(f"  [eval] avg MPJPE = {mean_mpjpe:.3f}  per-horizon = {avg.tolist()}")
            if mean_mpjpe < best_loss:
                best_loss = mean_mpjpe
                torch.save(
                    {"epoch": epoch, "state_dict": model.state_dict(),
                     "optimizer": optimizer.state_dict()},
                    ckpt_dir / f"{save_name}_best.pth.tar",
                )
                print(f"  [eval] saved best -> {ckpt_dir / (save_name + '_best.pth.tar')}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="Run AuxFormer on Human3.6M.")
    p.add_argument("--mode", choices=["train", "test"], default="test")
    p.add_argument("--task", choices=["short", "long"], default="short",
                   help="short = predict 400 ms (encoder_depth=3); "
                        "long = predict 1000 ms (encoder_depth=4)")
    p.add_argument("--data-root", type=str, default=None,
                   help="Path to h3.6m/dataset/ (overrides H36M_DATA_ROOT env var).")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Name of checkpoint in auxformer/ckpt/ (without .pth.tar). "
                        "Defaults to the official pretrained file for the task.")
    p.add_argument("--save-name", type=str, default=None,
                   help="(train mode) name for the saved checkpoint.")
    p.add_argument("--device", type=str, default=None,
                   help="Force device, e.g. cuda, cpu. Auto by default.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug", action="store_true",
                   help="(train mode) use only 'walking' action.")
    args = p.parse_args()

    if args.data_root is not None:
        os.environ["H36M_DATA_ROOT"] = args.data_root

    if args.ckpt is None:
        args.ckpt = DEFAULT_CKPT[args.task]

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_cfg(args.task)

    if args.mode == "test":
        cmd_test(args, cfg, device)
    else:
        cmd_train(args, cfg, device)


if __name__ == "__main__":
    main()
