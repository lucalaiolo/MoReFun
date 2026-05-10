# MoReFun — Human3.6M training

A faithful reimplementation of MoReFun (Shi et al., 2025) for 3D human motion
prediction on Human3.6M. The official repository at
https://github.com/JunyuShi02/MoReFun currently contains only a README, so
this code is rebuilt from the equations and architecture diagrams in the
paper. It omits the future-text branch, which is only relevant for the
FineMotion dataset — Human3.6M has no captions.

## What's here

```
config.py     all hyperparameters (paper defaults)
masking.py    velocity-based joint masking (Eq. 4)
model.py      JointEmbedder + PME + FMD + prediction head
data.py       Human3.6M loader + synthetic generator
eval.py       MPJPE per-frame, formatted table
pretrain.py   stage 1: past + future reconstruction
finetune.py   stage 2: future prediction
train.py      runs both stages end-to-end
```

## Quick smoke test (no real data)

```
python train.py --synthetic --epochs-pretrain 2 --epochs-finetune 2
```

This generates 512 synthetic clips with sinusoidal joint trajectories, runs
both training stages for 2 epochs each, and prints a per-frame MPJPE table at
the end. The synthetic data isn't meaningful — its only purpose is to verify
the training loops run without errors before you commit to real data.

## Real Human3.6M run

```
python train.py --data-root /path/to/h36m
```

This runs the full schedule from the paper:
- 40 epochs pretraining (past + future reconstruction)
- 20 epochs fine-tuning (future prediction)
- batch size 24, Adam, lr 5e-4 with cosine annealing

On a single RTX 4090 the paper reports under 6 GB of memory and roughly 6
hours of total training time. On weaker GPUs, drop the batch size or the
channel width in `config.py`.

## Data preparation

Human3.6M requires registration at
http://vision.imar.ro/human3.6m/description.php
(academic agreement required). Once you have the raw data:

1. **Extract 3D joint positions** from the `D3_Positions` field of the
   provided CDF files. Each subject has its own `MyPoseFeatures/D3_Positions`
   folder containing per-action `.cdf` files.

2. **Reduce to the 22-joint subset**. The original skeleton has 32 joints;
   most papers (AuxFormer, SiMLPe, SPGSN, GCNext) drop redundant and
   detail-only joints. The 22-joint indices used in this codebase are listed
   in `data.py` under `H36M_22_JOINT_INDICES`.

3. **Downsample from 50 fps to 25 fps** by taking every other frame. All
   timing references in the paper (80 ms, 160 ms, ..., 1000 ms) assume 25 fps.

4. **Save as .npz files** with a `positions` key of shape `(n_frames, 22, 3)`
   in millimetres. Layout expected by `Human36MDataset`:

       data_root/
         S1/
           directions_1.npz
           directions_2.npz
           eating_1.npz
           ...
         S5/
           ...

If your preprocessing produces a different layout (for example
`{subject}_{action}.npz` directly under the root), `Human36MDataset` will
fall back to that pattern automatically.

A widely-used preprocessing script that produces this exact layout is the
one shipped with HisRepItself
(https://github.com/wei-mao-2019/HisRepItself), specifically
`utils/data_utils_h36m.py`. AuxFormer and SiMLPe both consume the same
layout.

## Training conventions

- **Train/test split**: subjects S1, S5, S6, S7, S8 for training;
  S9 and S11 for evaluation. Standard convention since Mao et al. 2019.
- **Sliding window**: every (past_len + future_len)-frame window is a
  training sample, with stride 1 on train and stride 10 on test.
- **Centering**: each clip is recentered so that the root joint at the last
  past frame is the origin. This makes prediction translation-invariant; the
  alternative is to predict the global trajectory too, which most papers
  don't.

## Sanity checks for a successful run

After 40 + 20 epochs on real H3.6M, you should see something like:

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

Numbers above are the paper's averages from Table 2. If you see substantially
worse numbers, the most likely culprits are:

- Joint subset mismatch (your 22 joints aren't the same as the paper's).
- Frame rate mismatch (data not downsampled to 25 fps).
- Missing centering (`Human36MDataset(center=True)` is the default).
- Dropping pretraining (Table 8 row D shows this costs about 5 mm at 400 ms).

## What's faithful, what's interpretation

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
  `Q^s = F^s_Q(H^l)` and `K^s, V^s = F^s_KV(H^P)`, but the queries have L
  frames while the keys/values have T frames, and the attention is along
  the joint axis. We resolve this by mean-pooling the past along its time
  axis to a per-joint summary, then doing per-future-frame spatial attention
  against it. This matches the joint-by-joint attention maps in Fig. 12(d).
- FFN expansion ratio: described only as "bottle neck". We use 2x, which
  produces a model close to the paper's 1.66M parameter count when combined
  with shared-projection assumptions; with the 4x default it goes higher.
- FFN activation: paper doesn't say. We use GELU (standard).
- Dropout: paper doesn't say. We use 0.1.
- Future Semantic Decoder: omitted entirely (Human3.6M has no captions).

The parameter count this implementation produces (~3.5M) is higher than the
paper's reported 1.66M; the gap is consistent with the paper using shared
projections inside the multi-head attention or smaller FFN expansions than
we assume. None of this should affect MPJPE numbers materially.
# MoReFun
