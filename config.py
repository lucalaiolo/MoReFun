"""
Configuration for MoReFun on Human3.6M.

All numbers come from the paper (Sec. 4.2 + supplementary Sec. 8) unless noted.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    # Human3.6M setup, after dropping detail joints and downsampling to 25 fps.
    num_joints: int = 22
    coord_dim: int = 3  # x, y, z

    # Sequence lengths. Paper uses 10 past frames and predicts up to 25 future frames
    # (1000 ms at 25 fps). Short-term eval reports up to 400 ms = 10 future frames.
    past_len: int = 10
    future_len: int = 25

    # Training/test split (Human3.6M convention).
    train_subjects: List[str] = field(default_factory=lambda: ["S1", "S5", "S6", "S7", "S8"])
    test_subjects: List[str] = field(default_factory=lambda: ["S9", "S11"])

    # Where the preprocessed .npz files live. See data.py for expected layout.
    data_root: str = "./data/h36m"

    # Set to True to skip real data and run on a synthetic generator (useful for
    # smoke tests before you have Human3.6M downloaded).
    use_synthetic: bool = False

    downsample: int = 2


@dataclass
class ModelConfig:
    # Channel widths. Paper: 128 in PME and FMD, 256 in FSU (we omit FSU here).
    channels: int = 128

    # Attention. Paper: 8 heads, 32 channels per head -> projected width 256
    # internally, then projected back to `channels`.
    num_heads: int = 8
    head_dim: int = 32

    # Depth of PME and FMD. Paper sweeps {2, 3, 4} and picks 3 (Table 11).
    num_blocks: int = 3

    # FFN expansion in the bottleneck. The paper's "bottle neck" naming suggests
    # contraction; the parameter count (1.66M total) is consistent with a
    # 2x expansion rather than the standard 4x.
    ffn_mult: int = 2

    # Dropout. Paper does not specify; small value works well empirically.
    dropout: float = 0.1


@dataclass
class MaskConfig:
    # Fraction of joint-frames to mask, ranked by per-joint speed.
    # Paper sweeps {0.25, 0.50, 0.75} and picks 0.75 (Table 8).
    mask_rate: float = 0.75


@dataclass
class PretrainConfig:
    # Loss weighting between past and future reconstructions (paper alpha=1).
    alpha: float = 1.0

    epochs: int = 40
    batch_size: int = 24
    lr: float = 5e-4
    weight_decay: float = 0.0

    # Cosine annealing down to this fraction of the initial lr.
    min_lr_ratio: float = 0.01

    # Validation cadence and checkpointing.
    val_every: int = 1
    log_every: int = 50
    ckpt_dir: str = "./checkpoints/pretrain"


@dataclass
class FinetuneConfig:
    epochs: int = 20
    batch_size: int = 24
    lr: float = 5e-4
    weight_decay: float = 0.0
    min_lr_ratio: float = 0.01

    val_every: int = 1
    log_every: int = 50
    ckpt_dir: str = "./checkpoints/finetune"

    # Where to load pretrained weights from.
    pretrain_ckpt: str = "./checkpoints/pretrain/best.pt"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)

    # Reproducibility.
    seed: int = 42

    # Device.
    device: str = "cuda"
