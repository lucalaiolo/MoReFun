# AuxFormer — Human3.6M training, evaluation, and neonate fine-tuning

This repository packages the official
[AuxFormer](https://github.com/MediaBrain-SJTU/AuxFormer)
(Xu et al., ICCV 2023) code with minimal patches — device-aware tensors,
configurable data paths, modern Python — plus tooling on top for
downloading pretrained checkpoints, extracting encodings, and fine-tuning
on neonate keypoints.

You can either evaluate the released pretrained AuxFormer checkpoints
out of the box, or train from scratch on the same H3.6M data.

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
```

## Repository layout

```
.
├── README.md
├── requirements.txt
│
├── run_auxformer.py               # train/test entry point
├── download_auxformer_ckpts.py    # fetch official pretrained weights
│
├── extract_encodings.py           # dump AuxFormer features per test sample
├── evaluate_encodings.py          # linear/k-NN probes + UMAP of encodings
│
├── preprocess_neonates.py         # MHR keypoints → H3.6M-compatible 22 joints
├── finetune_auxformer_neonates.py # fine-tune AuxFormer on preprocessed neonates
├── umap_neonates.py               # UMAP of neonate encodings by velocity
│
└── auxformer/
    ├── cfg/                       # YAML hyperparameters per task and dataset
    │   ├── h36m_short.yml
    │   ├── h36m_long.yml
    │   ├── cmu_short.yml
    │   ├── cmu_long.yml
    │   ├── 3dpw_short.yml
    │   └── 3dpw_long.yml
    ├── ckpt/                      # checkpoints land here (gitignored)
    ├── dataset/                   # H3.6M / CMU / 3DPW loaders + FK
    └── model/                     # AuxFormer transformer
```

## Data

AuxFormer consumes Human3.6M as expmap .txt files in the
una-dinosauria / Mao layout:

```
$H36M_DATA_ROOT/
  S1/walking_1.txt, walking_2.txt, eating_1.txt, ...
  S5/, S6/, S7/, S8/, S9/, S11/
```

Each `.txt` is comma-separated, 99 columns wide (3 root + 96 expmap), one row
per 50-fps frame. Forward kinematics run on the fly. The standard splits
(S1/S5/S6/S7/S8 train, S9 test, S11 val) are hardwired in the loader.

Set `H36M_DATA_ROOT` and the pipeline picks it up (it defaults to
`./data/h3.6m/dataset` otherwise), or pass `--data-root` explicitly.

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
needed to run on modern Python:

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

## Extracting and probing AuxFormer encodings

`extract_encodings.py` runs the pretrained model over the H3.6M test set and
saves the pre-head feature tensor for every sample, optionally pooled:

```bash
# short-term encodings, mean-pooled over past frames + joints
python extract_encodings.py --task short

# long-term, no pooling — keeps full (N, T, F) tensor per sample
python extract_encodings.py --task long --pool none
```

`evaluate_encodings.py` then probes what those encodings capture:

```bash
python evaluate_encodings.py                                 # defaults
python evaluate_encodings.py --enc-dir encodings/h36m_long_mean_past
```

It trains a linear probe and a k-NN baseline for action classification,
computes intra/inter-class distances and a silhouette score, and writes a
UMAP scatter into the encodings directory.

## Fine-tuning on neonate keypoints

Three scripts turn raw MHR-format neonate keypoints into AuxFormer inputs
and fine-tune the long-term H3.6M model on them.

```bash
# 1. MHR (70 joints, ~60 fps) → H3.6M (22 joints, 25 fps) + retargeting
python preprocess_neonates.py \
    --input-dir /path/to/neonate_npz \
    --out-dir data/neonates_processed

# 2. Fine-tune the long-term H3.6M checkpoint on the preprocessed clips
python finetune_auxformer_neonates.py --epochs 30 --lr 1e-5 --freeze

# 3. UMAP the fine-tuned encodings, coloured by joint velocity
python umap_neonates.py --ckpt finetune_neonates/best.pth.tar
```

`preprocess_neonates.py` aligns each clip's hip→shoulder axis with H3.6M's
+y axis, maps the 70 MHR keypoints to the 22 H3.6M joints (constructing
derived joints like `hip_root`, `spine`, and `neck` by averaging the
appropriate input joints), retargets bone lengths to the H3.6M reference
skeleton while keeping the baby's joint angles, and downsamples to 25 fps.

`finetune_auxformer_neonates.py` slides length-50 windows (past 25 + future
25) with stride 5, splits each clip 80/20 in time (no leakage across
sliding windows), and can optionally freeze everything except the last
encoder block and the heads. It saves the best checkpoint by validation
MPJPE and writes `training_curves.png`.

## Citation

```bibtex
@inproceedings{xu2023auxiliary,
  title={Auxiliary Tasks Benefit 3D Skeleton-based Human Motion Prediction},
  author={Xu, Chenxin and Tan, Robby T and Tan, Yuhong and Chen, Siheng and
          Wang, Xinchao and Wang, Yanfeng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9509--9520},
  year={2023}
}
```
