"""
Top-level training script.

Runs pretraining (40 epochs) then fine-tuning (20 epochs), exactly as the
paper describes. Use this for a full training run on Human3.6M.

Usage:
    python train.py                 # use real H3.6M data at cfg.data.data_root
    python train.py --synthetic     # smoke-test with synthetic data
    python train.py --skip-pretrain # finetune only (e.g. when resuming)
"""

import argparse

from config import Config
import pretrain as stage1
import finetune as stage2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic data instead of real Human3.6M.")
    p.add_argument("--skip-pretrain", action="store_true",
                   help="Skip stage 1 (pretraining).")
    p.add_argument("--skip-finetune", action="store_true",
                   help="Skip stage 2 (fine-tuning).")
    p.add_argument("--data-root", type=str, default=None,
                   help="Override the default Human3.6M directory.")
    p.add_argument("--epochs-pretrain", type=int, default=None)
    p.add_argument("--epochs-finetune", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()

    if args.synthetic:
        cfg.data.use_synthetic = True
    if args.data_root:
        cfg.data.data_root = args.data_root
    if args.epochs_pretrain is not None:
        cfg.pretrain.epochs = args.epochs_pretrain
    if args.epochs_finetune is not None:
        cfg.finetune.epochs = args.epochs_finetune

    if not args.skip_pretrain:
        print("=" * 60)
        print("STAGE 1: PRETRAINING")
        print("=" * 60)
        stage1.main(cfg)

    if not args.skip_finetune:
        print()
        print("=" * 60)
        print("STAGE 2: FINE-TUNING")
        print("=" * 60)
        stage2.main(cfg)


if __name__ == "__main__":
    main()
