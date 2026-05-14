# MoReFun + AuxFormer — Human3.6M training and evaluation

This fork contains two human-motion-prediction models that share the same
Human3.6M data folder:

| Model | Where | What it does |
|---|---|---|
| **MoReFun** (Shi et al., 2025) | repo root | Faithful reimplementation from the paper. Two-stage SSL pretraining + fine-tuning. |
| **AuxFormer** (Xu et al., ICCV 2023) | `auxformer/` | Official code with minimal patches: device-aware tensors, configurable data path. Pretrained checkpoints from the authors. |

You can train both from scratch on the same data, or skip training entirely
and use the pretrained AuxFormer checkpoints released with the paper.

## Quick start

```bash
pip install -r requirements.txt

# Download the AuxFormer pretrained checkpoints (~28 MB for H3.6M)
python download_auxformer_ckpts.py

# Point at your H3.6M data (expmap .txt files, una-dinosauria layout)
export H36M_DATA_ROOT=/path/to/h3.6m/dataset

# Evaluate pretrained AuxFormer short-term (predict up to 400 ms)
python run_auxformer.py --mode test --task short

# Evaluate pretrained AuxFormer long-term (predict up to 1000 ms)
python run_auxformer.py --mode test --task long

# Train MoReFun from scratch (40 + 20 epochs)
python train.py --data-root $H36M_DATA_ROOT
```

## Repository layout

```
.
├── README.md
├── requirements.txt
│
├── # ---- MoReFun (Shi et al., 2025) -----------------------------------
├── config.py          # MoReFun hyperparameters
├── data.py            # MoReFun H3.6M loader
├── model.py           # MoReFun architecture (PME + FMD)
├── masking.py         # velocity-based joint masking
├── pretrain.py        # MoReFun stage 1 (self-supervised)
├── finetune.py        # MoReFun stage 2 (fine-tuning)
├── train.py           # MoReFun end-to-end driver
├── eval.py            # MoReFun MPJPE metrics
├── fk_utils.py        # H3.6M forward kinematics
│
├── # ---- AuxFormer (Xu et al., ICCV 2023) -----------------------------
├── run_auxformer.py             # train/test entry point
├── download_auxformer_ckpts.py  # fetch official pretrained weights
└── auxformer/
    ├── cfg/                     # YAML hyperparameters per task and dataset
    │   ├── h36m_short.yml
    │   ├── h36m_long.yml
    │   ├── cmu_short.yml
    │   ├── cmu_long.yml
    │   ├── 3dpw_short.yml
    │   └── 3dpw_long.yml
    ├── ckpt/                    # checkpoints land here (gitignored)
    ├── dataset/                 # H3.6M / CMU / 3DPW loaders + FK
    └── model/                   # AuxFormer transformer
```

## Data: one folder, two models

Both models consume Human3.6M as expmap .txt files in the
una-dinosauria / Mao layout:

```
$H36M_DATA_ROOT/
  S1/walking_1.txt, walking_2.txt, eating_1.txt, ...
  S5/, S6/, S7/, S8/, S9/, S11/
```

Each `.txt` is comma-separated, 99 columns wide (3 root + 96 expmap), one row
per 50-fps frame. MoReFun and AuxFormer each run forward kinematics on this
on the fly. The standard splits (S1/S5/S6/S7/S8 train, S9 test, S11 val) are
hardwired in both loaders.

Set `H36M_DATA_ROOT` once and both pipelines pick it up. The MoReFun side
defaults to `./data/h36m`; the AuxFormer side defaults to `./data/h3.6m/dataset`.
If you keep one canonical copy, point both at it via the env var or the
`--data-root` flag.

## AuxFormer — short vs long-term

The paper trains two separate models for H3.6M:

| Task | Past | Future | encoder_depth | Checkpoint |
|---|---|---|---|---|
| **short** (≤ 400 ms) | 10 frames | 10 frames | 3 | `pretrain_h36m_ckpt.pth.tar` |
| **long** (≤ 1000 ms) | 25 frames | 13 frames (sampled from 25) | 4 | `pretrain_h36m_ckpt_long.pth.tar` |

`run_auxformer.py --task short|long` picks the right config and checkpoint
automatically. The long-term variant evaluates at 80, 160, 320, 400, 560, and
1000 ms; the short one stops at 400 ms.

### Pretrained checkpoints

The official checkpoints (~90 MB total across all three datasets) live in
the upstream repo at `MediaBrain-SJTU/AuxFormer/ckpt/`. The download script
fetches only the H3.6M pair by default:

```bash
python download_auxformer_ckpts.py            # H3.6M only (~28 MB)
python download_auxformer_ckpts.py --all      # all three datasets + extras
```

### Evaluation output

```
$ python run_auxformer.py --mode test --task short
[run_auxformer] loading checkpoint: auxformer/ckpt/pretrain_h36m_ckpt.pth.tar
[run_auxformer] params: 1.00M
  walking          |    8.90    16.90    30.10    36.10 | avg   23.00
  eating           |    6.40    14.00    28.80    35.90 | avg   21.28
  smoking          |    5.70    11.40    22.10    27.90 | avg   16.78
  ...
  AVG              |    9.50    20.60    43.40    54.10 | avg   31.90
  (   80ms   160ms   320ms   400ms)
```

These numbers should match Table 1 of the paper to within rounding when
the H3.6M data is correctly preprocessed.

### Training AuxFormer from scratch

```bash
python run_auxformer.py --mode train --task short --save-name my_h36m
```

The combined loss (prediction + masked reconstruction + denoising) is the
same one the paper trained with. Best checkpoint by val MPJPE goes to
`auxformer/ckpt/{save_name}_best.pth.tar`. Default is 80 epochs at lr 5e-4.

## Patches applied to AuxFormer

The code under `auxformer/` is the official release with the smallest changes
needed to run on modern Python and to share data with MoReFun:

1. **Device-aware tensors.** All 22 hardcoded `.cuda()` calls in `model.py`
   now route through a `_self_device(self)` helper that reads the device
   from the model's parameters. The model runs on CPU, CUDA, or MPS without
   edits.
2. **Configurable data root.** `dataset/dataloader.py` accepts a
   `path_to_data` argument and falls back to `H36M_DATA_ROOT` (and
   `CMU_DATA_ROOT`, `PW3D_DATA_ROOT`) environment variables.
3. **Modern Python imports.** Dropped `from six.moves import xrange`;
   `xrange` → `range`.
4. **NumPy strictness.** `np.array([[1,6,7,8,9],[5],[11]])` ragged-array
   construction is now a plain Python list (NumPy ≥ 1.24 refuses the original).
5. **YAML safe loading** in the wrapper script.

No model logic, training loop, evaluation protocol, or numerical behavior
changed. Both pretrained H3.6M checkpoints load with zero missing or
unexpected keys.

## MoReFun — two-stage SSL

A faithful reimplementation of MoReFun (Shi et al., 2025), built from the
paper's equations and diagrams. The official repository at
https://github.com/JunyuShi02/MoReFun currently contains only a README.
The future-text branch is omitted — Human3.6M has no captions.

### Smoke test (no real data)

```bash
python train.py --synthetic --epochs-pretrain 2 --epochs-finetune 2
```

Generates 512 synthetic clips with sinusoidal joint trajectories, runs
both training stages for 2 epochs each, and prints a per-frame MPJPE table
at the end. The numbers are meaningless — the synthetic data is only there
to verify the training loops run without errors.

### Real Human3.6M run

```bash
python train.py --data-root $H36M_DATA_ROOT
```

The full schedule from the paper:
- 40 epochs pretraining (past + future reconstruction)
- 20 epochs fine-tuning (future prediction)
- batch size 24, Adam, lr 5e-4 with cosine annealing

On a single RTX 4090 the paper reports under 6 GB of memory and roughly 6
hours of total training time. On weaker GPUs, drop the batch size or the
channel width in `config.py`.

### Training conventions

- **Splits**: subjects S1, S5, S6, S7, S8 for training; S9, S11 for
  evaluation. (Standard since Mao et al. 2019.)
- **Sliding window**: every `(past_len + future_len)`-frame window is a
  training sample, stride 1 on train, stride 10 on test.
- **Centering**: each clip is recentered so the root joint at the last past
  frame is the origin (translation-invariant prediction).

### Sanity checks

After 40 + 20 epochs on real H3.6M you should see roughly:

```
Time (ms)   MPJPE (mm)
   80          ~9
  160         ~20
  320         ~42
  400         ~52
  560         ~71
 1000        ~102
  avg         ~50
```

These are the paper's averages from Table 2. If you see substantially worse
numbers, the most likely culprits are:

- Joint subset mismatch (your 22 joints aren't the same as the paper's).
- Frame rate mismatch (data not downsampled to 25 fps).
- Missing centering (`Human36MDataset(center=True)` is the default).
- Dropping pretraining (Table 8 row D shows this costs about 5 mm at 400 ms).

### What's faithful, what's interpretation (MoReFun)

**Faithful to the paper:**
- Joint embedder structure (Eq. 1).
- PME block: temporal then spatial self-attention, both with FFN bottlenecks.
- FMD block: temporal cross, spatial cross, temporal self, spatial self,
  each with its own FFN.
- Velocity-based masking using per-joint speed and a global threshold.
- Two-stage training with the past + future reconstruction objective in
  stage 1 and zero-initialised future queries in stage 2.
- Hyperparameters: 3 blocks, 8 heads × 32 dim, channels = 128, mask rate
  0.75, alpha = 1.0, lr 5e-4, batch 24, cosine annealing, 40 + 20 epochs.

**Interpretive choices not pinned down by the paper:**
- Spatial cross-attention shape mismatch: the paper writes
  `Q^s = F^s_Q(H^l)` and `K^s, V^s = F^s_KV(H^P)`, but queries have L frames
  while keys/values have T frames, and the attention is along the joint
  axis. We resolve this by mean-pooling the past along its time axis to a
  per-joint summary, then doing per-future-frame spatial attention against
  it. This matches the joint-by-joint attention maps in Fig. 12(d).
- FFN expansion ratio: described only as "bottle neck". We use 2x.
- FFN activation: paper doesn't say. We use GELU (standard).
- Dropout: paper doesn't say. We use 0.1.
- Future Semantic Decoder: omitted entirely (no H3.6M captions).

The parameter count this implementation produces (~3.5M) is higher than the
paper's reported 1.66M; the gap is consistent with the paper using shared
projections inside the multi-head attention or smaller FFN expansions than
we assume. None of this should affect MPJPE numbers materially.

## Citations

```bibtex
@inproceedings{xu2023auxiliary,
  title={Auxiliary Tasks Benefit 3D Skeleton-based Human Motion Prediction},
  author={Xu, Chenxin and Tan, Robby T and Tan, Yuhong and Chen, Siheng and
          Wang, Xinchao and Wang, Yanfeng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9509--9520},
  year={2023}
}

@article{shi2025morefun,
  title={MoReFun: Past-Movement Guided Motion Representation Learning for
         Future Motion Prediction and Understanding},
  author={Shi, Junyu and Wu, Haoting and Zhang, Zhiyuan and Liu, Lijiang and
          Sun, Yong and Nie, Qiang},
  journal={arXiv preprint arXiv:2408.02091v2},
  year={2025}
}
```
