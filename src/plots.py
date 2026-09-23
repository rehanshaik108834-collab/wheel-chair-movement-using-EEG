"""Per-subject figures: training-curve grids, confusion-matrix grids, per-class recall heatmaps."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score

TRAIN_COLOR, VAL_COLOR = "#1d3557", "#e76f51"


def plot_curves_grid(histories, metric, title, path=None, second="val", second_label="validation",
                     mark_best=True):
    """histories: {subject: [history, ...]} (one history per fold or seed).

    metric: "loss" or "acc". Each panel shows every fold/seed as a thin line.
    second: history key prefix of the second curve ("val", or "mon" for a monitor-only test curve).
    mark_best: mark the epoch whose weights were kept (best validation loss).
    """
    subjects = list(histories)
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharey=True)
    for ax, s in zip(axes.flat, subjects):
        for i, h in enumerate(histories[s]):
            tr = np.array(h[f"train_{metric}"]) * (100 if metric == "acc" else 1)
            va = np.array(h[f"{second}_{metric}"]) * (100 if metric == "acc" else 1)
            ep = np.arange(1, len(tr) + 1)
            ax.plot(ep, tr, color=TRAIN_COLOR, lw=0.9, alpha=0.6, label="train" if i == 0 else None)
            ax.plot(ep, va, color=VAL_COLOR, lw=0.9, alpha=0.6, label=second_label if i == 0 else None)
            if mark_best:
                b = h["best_epoch"]
                ax.plot(b + 1, va[b], "o", color=VAL_COLOR, ms=4, label="kept epoch" if i == 0 else None)
        if metric == "acc":
            ax.axhline(25, color="#e63946", ls="--", lw=0.8)
        ax.set_title(s)
        ax.set_xlabel("Epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy (%)" if metric == "acc" else "Cross-entropy loss")
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig


def plot_confusion_grid(y_true, y_pred, class_names, title, cmap="Blues", path=None):
    """y_true / y_pred: {subject: array}. One row-normalized confusion matrix per subject."""
    subjects = list(y_true)
    short = [c.replace("_hand", "").replace("_", " ") for c in class_names]
    fig, axes = plt.subplots(3, 3, figsize=(12, 11.5))
    for ax, s in zip(axes.flat, subjects):
        cm = confusion_matrix(y_true[s], y_pred[s], labels=range(len(class_names)))
        cm_pct = 100 * cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=100)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, f"{cm[i, j]}\n{cm_pct[i, j]:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if cm_pct[i, j] > 55 else "black")
        acc = (np.asarray(y_true[s]) == np.asarray(y_pred[s])).mean()
        ax.set_title(f"{s}: accuracy {acc:.1%}")
        ax.set_xticks(range(len(short)), short, fontsize=8)
        ax.set_yticks(range(len(short)), short, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
    fig.suptitle(title + "\n(counts and % of each true class)", fontsize=13)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig


def per_class_recall(y_true, y_pred, class_names):
    """Table (subjects x classes) of recall = % of each class's trials that were classified correctly."""
    import pandas as pd
    rows = {s: recall_score(y_true[s], y_pred[s], labels=range(len(class_names)), average=None) * 100
            for s in y_true}
    table = pd.DataFrame(rows, index=class_names).T
    table.loc["mean"] = table.mean()
    return table.round(1)


def plot_recall_heatmap(table, title, cmap="Blues", path=None):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(table.values, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9, color="white" if v > 55 else "black")
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.axhline(table.shape[0] - 1.5, color="black", lw=1)
    ax.set_title(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig
