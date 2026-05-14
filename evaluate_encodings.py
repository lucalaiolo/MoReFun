"""
Probe AuxFormer encodings for action information.

Loads encodings.npy and labels.npy from `extract_encodings.py`, then:
  1. trains a linear probe (logistic regression) to predict the action label,
     reporting accuracy + confusion matrix
  2. trains a k-NN baseline for reference
  3. computes intra- and inter-class distances (the MoReFun paper's Table 5
     metrics) and the silhouette score
  4. runs UMAP and writes a 2D scatter coloured by action

A note on stride-1 windowing: adjacent H3.6M test samples share 49 of 50
frames (long config), which means a random 80/20 split will leak near-duplicate
neighbours into the test set and produce inflated accuracy numbers. The
`--subsample` flag thins out the dataset to fight this. The default keeps
every 25th sample (~1 s of motion between consecutive samples at 25 fps), so
classifier numbers are honest.

Usage:
    python evaluate_encodings.py                                 # defaults
    python evaluate_encodings.py --enc-dir encodings/h36m_long_mean_past
    python evaluate_encodings.py --subsample 1                   # use all samples (inflated)
    python evaluate_encodings.py --no-umap                       # skip UMAP

All outputs (figures + a summary .json) go inside the encodings directory.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, silhouette_score)


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


def load_encodings(enc_dir: Path):
    enc = np.load(enc_dir / "encodings.npy")
    labels = np.load(enc_dir / "labels.npy", allow_pickle=True)
    sample_idx = np.load(enc_dir / "sample_idx.npy")
    with open(enc_dir / "meta.json") as f:
        meta = json.load(f)
    return enc, labels, sample_idx, meta


# --------------------------------------------------------------------------
# Subsampling
# --------------------------------------------------------------------------


def subsample_stratified(enc, labels, sample_idx, every_k: int):
    """Keep every k-th sample within each action. Stride-1 windows are highly
    correlated; thinning makes downstream evaluation honest."""
    if every_k <= 1:
        return enc, labels, sample_idx

    keep_mask = np.zeros(len(labels), dtype=bool)
    for act in np.unique(labels):
        idx = np.where(labels == act)[0]
        keep_mask[idx[::every_k]] = True
    return enc[keep_mask], labels[keep_mask], sample_idx[keep_mask]


# --------------------------------------------------------------------------
# Classification probes
# --------------------------------------------------------------------------


def linear_probe(X_train, y_train, X_test, y_test, classes):
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=0)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred, labels=classes)
    report = classification_report(y_test, pred, labels=classes,
                                    zero_division=0, output_dict=True)
    return acc, cm, report, pred


def knn_probe(X_train, y_train, X_test, y_test, k=5):
    clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    clf.fit(X_train, y_train)
    return accuracy_score(y_test, clf.predict(X_test))


# --------------------------------------------------------------------------
# Clustering metrics
# --------------------------------------------------------------------------


def cluster_metrics(X, labels):
    """MoReFun-style inter- vs intra-class distances, plus silhouette."""
    classes = np.unique(labels)
    class_means = np.stack([X[labels == c].mean(axis=0) for c in classes])

    inter = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            inter.append(np.linalg.norm(class_means[i] - class_means[j]))
    inter_mean = float(np.mean(inter))

    intra = []
    for i, c in enumerate(classes):
        members = X[labels == c]
        if len(members) < 2:
            continue
        d = np.linalg.norm(members - class_means[i], axis=1)
        intra.append(float(d.mean()))
    intra_mean = float(np.mean(intra))

    # Silhouette is slow for big N; subsample to 5000 if necessary.
    if len(X) > 5000:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(X), size=5000, replace=False)
        sil = float(silhouette_score(X[sel], labels[sel]))
    else:
        sil = float(silhouette_score(X, labels))

    return {"inter_class_distance": inter_mean,
            "intra_class_distance": intra_mean,
            "ratio_inter_over_intra": inter_mean / max(intra_mean, 1e-9),
            "silhouette": sil}


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------


def plot_confusion(cm, classes, out_path, title=""):
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)

    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cm_norm[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if v > 0.5 else "black")

    plt.colorbar(im, ax=ax, fraction=0.045)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_action_accuracy(report, classes, out_path, title=""):
    f1 = [report[c]["f1-score"] for c in classes]
    recall = [report[c]["recall"] for c in classes]
    order = np.argsort(f1)
    classes_ord = [classes[i] for i in order]
    f1_ord = [f1[i] for i in order]
    rec_ord = [recall[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(classes_ord))
    ax.barh(y - 0.2, f1_ord, height=0.4, label="F1", color="#3a7bd5")
    ax.barh(y + 0.2, rec_ord, height=0.4, label="Recall", color="#f5a623")
    ax.set_yticks(y)
    ax.set_yticklabels(classes_ord, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("score")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_umap(coords, labels, classes, out_path, title=""):
    cmap = plt.get_cmap("tab20", len(classes))
    fig, ax = plt.subplots(figsize=(11, 8))
    for i, c in enumerate(classes):
        m = labels == c
        ax.scatter(coords[m, 0], coords[m, 1], s=6, alpha=0.5,
                    color=cmap(i), label=c, edgecolors="none")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9,
              markerscale=2.5, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description="Linear probe + UMAP for AuxFormer encodings.")
    p.add_argument("--enc-dir", type=str, default="encodings/h36m_long_mean_past",
                   help="Where encodings.npy / labels.npy live.")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Where to write figures and analysis.json. "
                        "Defaults to --enc-dir/analysis/.")
    p.add_argument("--subsample", type=int, default=25,
                   help="Keep every k-th sample within each action class "
                        "(stride-1 windows are highly correlated). Default 25 "
                        "≈ 1 s of motion between consecutive samples at 25 fps. "
                        "Pass 1 to disable subsampling (will inflate accuracy).")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--no-umap", action="store_true")
    p.add_argument("--umap-neighbors", type=int, default=30)
    p.add_argument("--umap-min-dist", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    enc_dir = Path(args.enc_dir)
    out_dir = Path(args.out_dir) if args.out_dir else enc_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    enc, labels, sample_idx, meta = load_encodings(enc_dir)
    title_tag = f"{meta.get('task', '?')}/{meta.get('pool_mode', '?')}"

    print(f"[probe] loaded {enc.shape[0]} samples × {enc.shape[1]} dims "
          f"from {enc_dir}")
    print(f"[probe] task = {title_tag}, "
          f"actions = {len(np.unique(labels))}")

    if args.subsample > 1:
        enc, labels, sample_idx = subsample_stratified(
            enc, labels, sample_idx, args.subsample)
        print(f"[probe] subsampled to {enc.shape[0]} (every {args.subsample}th window).")

    classes = sorted(np.unique(labels).tolist())

    # ---- classification ----
    X_train, X_test, y_train, y_test = train_test_split(
        enc, labels, test_size=args.test_frac,
        stratify=labels, random_state=args.seed,
    )

    # Standardize features before linear probe — small gain, common practice.
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"[probe] linear probe: training on {len(X_train)}, testing on {len(X_test)}")
    lr_acc, cm, report, _ = linear_probe(X_train_s, y_train, X_test_s, y_test, classes)
    print(f"  linear probe accuracy   = {lr_acc*100:.2f}%")

    knn_acc = knn_probe(X_train_s, y_train, X_test_s, y_test, k=5)
    print(f"  k-NN (k=5) accuracy     = {knn_acc*100:.2f}%")

    # Chance level
    _, counts = np.unique(y_test, return_counts=True)
    chance = counts.max() / counts.sum()
    print(f"  chance (majority class) = {chance*100:.2f}%")

    # ---- clustering metrics on the *training* set, scaled ----
    cluster_stats = cluster_metrics(X_train_s, y_train)
    print(f"[probe] inter-class dist  = {cluster_stats['inter_class_distance']:.2f}")
    print(f"        intra-class dist  = {cluster_stats['intra_class_distance']:.2f}")
    print(f"        ratio (inter/intra) = {cluster_stats['ratio_inter_over_intra']:.2f}")
    print(f"        silhouette score  = {cluster_stats['silhouette']:.3f}")

    # ---- plots ----
    print(f"[probe] writing figures to {out_dir}/")
    plot_confusion(cm, classes, out_dir / "confusion_matrix.png",
                    title=f"AuxFormer ({title_tag}) — linear probe   "
                          f"acc = {lr_acc*100:.1f}%")
    plot_per_action_accuracy(report, classes,
                              out_dir / "per_action_scores.png",
                              title=f"Per-action F1 / recall ({title_tag})")

    # ---- UMAP ----
    umap_payload = None
    if not args.no_umap:
        import umap
        print(f"[probe] running UMAP on {len(enc)} samples (n_neighbors="
              f"{args.umap_neighbors}, min_dist={args.umap_min_dist})...")
        reducer = umap.UMAP(n_neighbors=args.umap_neighbors,
                             min_dist=args.umap_min_dist,
                             n_components=2,
                             random_state=args.seed,
                             verbose=False)
        enc_scaled = StandardScaler().fit_transform(enc)
        coords = reducer.fit_transform(enc_scaled)
        plot_umap(coords, labels, classes, out_dir / "umap.png",
                   title=f"UMAP of AuxFormer encodings ({title_tag})  "
                         f"  n={len(enc)}")
        np.save(out_dir / "umap_coords.npy", coords)
        umap_payload = {"n_neighbors": args.umap_neighbors,
                        "min_dist": args.umap_min_dist,
                        "n_samples": int(len(enc))}

    # ---- summary ----
    summary = {
        "encoding_dir": str(enc_dir),
        "task": meta.get("task"),
        "pool_mode": meta.get("pool_mode"),
        "n_samples_total": int(enc.shape[0]),
        "feature_dim": int(enc.shape[1]),
        "n_classes": int(len(classes)),
        "subsample_factor": args.subsample,
        "linear_probe_accuracy": float(lr_acc),
        "knn5_accuracy": float(knn_acc),
        "chance_accuracy": float(chance),
        **cluster_stats,
    }
    if umap_payload:
        summary["umap"] = umap_payload
    with open(out_dir / "analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[probe] summary written to {out_dir / 'analysis.json'}")


if __name__ == "__main__":
    main()
