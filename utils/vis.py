"""结果可视化：损失曲线、HER 代理分布、稳定性与可合成性、结构示意图。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(
    train_losses: Sequence[float],
    val_losses: Optional[Sequence[float]],
    out_path: str | Path,
    title: str = "Training loss",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="train total")
    if val_losses is not None and len(val_losses) > 0:
        plt.plot(val_losses, label="val total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_her_performance(
    dg_values: Sequence[float],
    labels: Optional[Sequence[str]],
    out_path: str | Path,
    title: str = r"$\Delta G_H$ proxy distribution (eV)",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    if labels is None:
        plt.hist(np.asarray(dg_values), bins=20, alpha=0.85, color="#2c7fb8")
    else:
        uniq = list(dict.fromkeys(labels))
        for lab in uniq:
            vals = [v for v, l in zip(dg_values, labels) if l == lab]
            plt.hist(np.asarray(vals), bins=16, alpha=0.55, label=str(lab))
        plt.legend()
    plt.axvline(0.0, color="crimson", linestyle="--", linewidth=1.2, label="ideal ~0 eV")
    plt.xlabel(r"$\Delta G_H$ proxy (eV)")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_stability_dual(
    ours: Dict[str, np.ndarray],
    baseline: Dict[str, np.ndarray],
    out_path: str | Path,
    title: str = "Stability & synthesizability (ours vs baseline slice)",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, rec, lab in zip(
        axes,
        (ours, baseline),
        ("Ours (generated)", "Baseline (dataset slice)"),
    ):
        x = rec.get("step", np.arange(len(rec["stab"])))
        ax.plot(x, rec["stab"], label="stability aggregate", color="#1b9e77")
        ax.plot(x, rec["synth"], label="synthesis score", color="#d95f02")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(lab)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Score [0,1]")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_stability_synthesis_curves(
    records: Dict[str, np.ndarray],
    out_path: str | Path,
    title: str = "Stability & synthesizability",
) -> None:
    """
    records: dict with keys 'step' or index default, 'stab', 'synth' (each 1D same length)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    x = records.get("step", np.arange(len(records["stab"])))
    plt.plot(x, records["stab"], label="stability aggregate", color="#1b9e77")
    plt.plot(x, records["synth"], label="synthesis score", color="#d95f02")
    plt.xlabel("Sample index (or step)")
    plt.ylabel("Score [0,1]")
    plt.ylim(0.0, 1.05)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_generated_structures_panel(
    image_paths: List[str | Path],
    out_path: str | Path,
    title: str = "Generated structures (2D projections)",
) -> None:
    """将若干 PNG 合成一张拼图；若路径不存在则跳过。

    注意：``tight_layout`` 与 ``suptitle`` 同时用时必须传 ``rect=``，否则子图易被挤成空白。
    """
    from matplotlib import image as mpimg

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    paths = [p for p in image_paths if Path(p).is_file()]
    if not paths:
        plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "No structure images found", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, facecolor="white")
        plt.close()
        return
    n = min(len(paths), 12)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.3, rows * 2.3), squeeze=False)
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        if i < n:
            img = mpimg.imread(str(paths[i]))
            ax.imshow(img, interpolation="nearest")
        ax.set_axis_off()
    fig.suptitle(title, fontsize=12)
    # 为标题留出顶部边距，避免子图被压到不可见
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(out_path, dpi=160, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
