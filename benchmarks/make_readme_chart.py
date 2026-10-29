# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""The README's charts, from the results of October 2026: assets/epoch-*.png and assets/epoch-loss-*.png, light and dark.

One group per model and method, one bar per arm: the mean training tokens per second of three runs, and a line
over the range of the three. Run from the repository's root: python benchmarks/make_readme_chart.py
"""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results" / "2026-10"
ASSETS = ROOT / "assets"

# Rows top to bottom, and the result-file label of each arm in each row
ROWS = [
    ("Gemma 4 E4B, LoRA", "speed-gb10.jsonl", "gemma"),
    ("Qwen 3.5 9B, LoRA", "speed-gb10.jsonl", "qwen"),
    ("Gemma 4 E4B, QLoRA", "qlora-gb10.jsonl", "gemma-qlora"),
    ("Llama 3.1 8B, LoRA", "speed-gb10.jsonl", "llama"),
    ("Llama 3.1 8B, QLoRA", "qlora-gb10.jsonl", "llama-qlora"),
]
ARMS = [  # (legend, label suffix in the results)
    ("TRL, default settings", "trl"),
    ("Unsloth, default", "unsloth"),
    ("Unsloth, fastest (no checkpointing)", "unsloth-no-checkpointing"),
    ("Meep²", "auto"),
]
# Colours per arm, in this order; checked for colour-vision separation on each theme's surface
THEMES = {
    "light": {"surface": "#ffffff", "ink": "#1f2328", "muted": "#59636e", "grid": "#d8dee4",
              "arms": ["#8F5D04", "#D14E95", "#7A4ABA", "#008CA0"]},
    "dark": {"surface": "#0d1117", "ink": "#e6edf3", "muted": "#9198a1", "grid": "#30363d",
             "arms": ["#B07A1E", "#E0569F", "#8E66D6", "#1AA3B8"]},
}


def runs(file: str, label: str) -> list:
    return [json.loads(line)["tokens_per_s"] for line in open(RESULTS / file)
            if json.loads(line)["arm"] == label]


def draw(theme: str) -> Path:
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(11, 6.8))
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    bar_w = 0.19
    for r, (row, file, prefix) in enumerate(ROWS):
        unsloth_fastest = None
        for a, (name, suffix) in enumerate(ARMS):
            values = runs(file, f"{prefix}-{suffix}")
            mean = statistics.mean(values)
            x = r + (a - 1.5) * bar_w
            ax.bar(x, mean, width=bar_w, color=t["arms"][a], edgecolor=t["surface"], linewidth=2,
                   label=name if r == 0 else None, zorder=3)
            ax.plot([x, x], [min(values), max(values)], color=t["ink"], linewidth=1.2, zorder=4)
            if suffix == "unsloth-no-checkpointing":
                unsloth_fastest = mean
            if suffix == "auto":
                ratio = mean / unsloth_fastest
                note = f"{mean:,.0f}\n{ratio:.1f}×" if ratio >= 1.1 else f"{mean:,.0f}\nlevel"
                ax.text(x, max(values) + 30, note, va="bottom", ha="center", fontsize=10.5, color=t["ink"],
                        fontweight="bold", linespacing=1.15)
    ax.set_xticks(range(len(ROWS)))
    ax.set_xticklabels([row.replace(", ", "\n") for row, _, _ in ROWS], fontsize=11, color=t["ink"])
    ax.set_xlim(-0.55, len(ROWS) - 0.45)
    ax.set_ylim(0, 2250)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.tick_params(axis="y", colors=t["muted"], labelsize=10)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.grid(axis="y", color=t["grid"], linewidth=1, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.set_ylabel("Training tokens per second", fontsize=10.5, color=t["muted"], labelpad=8)
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=4, frameon=False, fontsize=10,
                       handlelength=1.2, columnspacing=1.4)
    for text in legend.get_texts():
        text.set_color(t["ink"])
    fig.suptitle("LoRA and QLoRA on one NVIDIA GB10 (DGX Spark), 30 steps", x=0.012, y=0.975, ha="left",
                 fontsize=14, color=t["ink"], fontweight="bold")
    fig.text(0.012, 0.012, "Mean of three runs; the line spans the three. Above Meep²: its speed against Unsloth's fastest "
             "setting. Same batches in the same order in every arm.\nMeasured 25–28 Oct 2026: Meep² 1375994 "
             "(transformers 5.20.2, TRL 1.16.0); Unsloth 2026.10.7 (transformers 5.17.0, TRL 0.29.1).",
             fontsize=9, color=t["muted"], ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    ASSETS.mkdir(exist_ok=True)
    path = ASSETS / f"benchmarks-{theme}.png"
    fig.savefig(path, dpi=150, facecolor=t["surface"])
    plt.close(fig)
    return path


def epoch(label: str) -> dict:
    """One epoch run of results/2026-10/epoch.jsonl, by its label."""
    for line in open(RESULTS / "epoch.jsonl"):
        d = json.loads(line)
        if d["arm"] == label:
            return d
    raise SystemExit(f"epoch.jsonl has no run {label}")


def shared(label: str) -> float:
    """The held-out loss of an epoch run's adapter, evaluated through TRL's evaluation (epoch-shared-evaluation.jsonl)."""
    for line in open(RESULTS / "epoch-shared-evaluation.jsonl"):
        d = json.loads(line)
        if d["arm"] == label:
            return d["eval_loss"]
    raise SystemExit(f"epoch-shared-evaluation.jsonl has no {label}")


MODELS = [("Gemma 4 E4B", "gemma"), ("Qwen 3.5 9B", "qwen")]
ENGINES = [("TRL, default settings", "trl", "trl", 0), ("Unsloth, fastest (no checkpointing)", "unsloth-fastest", "unsloth", 2),
           ("Meep²", "meep", "meep", 3)]
FOOT = ("One epoch of no_robots: 9,100 conversations, 1,137 steps of 8, the same batches in the same order in every arm. "
        "Held-out loss on 400 other conversations,\nevery adapter measured by the same evaluation (TRL's). Measured 29 Oct 2026 on "
        "one NVIDIA GB10:\nMeep² 7971242 (transformers 5.20.2, TRL 1.16.0); Unsloth 2026.10.7 (transformers 5.17.0, TRL 0.29.1).")


def style(ax, t):
    ax.tick_params(colors=t["muted"], labelsize=10)
    ax.grid(axis="y", color=t["grid"], linewidth=1, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])


def draw_epoch(theme: str) -> Path:
    """Minutes to train one epoch, per model and engine, with each adapter's held-out loss under its bar."""
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(11, 6.6))
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    w = 0.27
    for m, (model, prefix) in enumerate(MODELS):
        unsloth = epoch(f"{prefix}-unsloth-fastest")["quality"][-1]["training_seconds"] / 60
        for e, (name, label, short, colour) in enumerate(ENGINES):
            run = epoch(f"{prefix}-{label}")
            minutes = run["quality"][-1]["training_seconds"] / 60
            x = m + (e - 1) * w
            ax.bar(x, minutes, width=w, color=t["arms"][colour], edgecolor=t["surface"], linewidth=2,
                   label=name if m == 0 else None, zorder=3)
            top = f"{minutes:.0f} min" + (f"\n{unsloth / minutes:.1f}× faster\nthan Unsloth" if short == "meep" else "")
            ax.text(x, minutes + 2, top, ha="center", va="bottom", fontsize=10.5 if short == "meep" else 10,
                    color=t["ink"], fontweight="bold" if short == "meep" else "normal", linespacing=1.15)
            ax.text(x, -4, f"loss\n{shared(f'{prefix}-{short}-eval-trl'):.3f}", ha="center", va="top", fontsize=9,
                    color=t["muted"], linespacing=1.1)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([m for m, _ in MODELS], fontsize=12, color=t["ink"])
    ax.tick_params(axis="x", length=0, pad=34)
    ax.set_xlim(-0.6, len(MODELS) - 0.4)
    ax.set_ylim(0, 165)
    ax.set_ylabel("Minutes to train one epoch (lower is faster)", fontsize=10.5, color=t["muted"], labelpad=8)
    style(ax, t)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_color(t["ink"])
    fig.suptitle("One epoch of LoRA on one DGX Spark: the same held-out loss, in a fraction of the time", x=0.012, y=0.975,
                 ha="left", fontsize=14, color=t["ink"], fontweight="bold")
    fig.text(0.012, 0.012, FOOT, fontsize=8.5, color=t["muted"], ha="left")
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    path = ASSETS / f"epoch-{theme}.png"
    fig.savefig(path, dpi=150, facecolor=t["surface"])
    plt.close(fig)
    return path


def draw_epoch_loss(theme: str) -> Path:
    """Held-out loss against minutes over the epoch: Meep² and TRL as curves, Unsloth's adapter at the end of its epoch."""
    t = THEMES[theme]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=False)
    fig.patch.set_facecolor(t["surface"])
    for ax, (model, prefix) in zip(axes, MODELS):
        ax.set_facecolor(t["surface"])
        for name, label, short, colour in ENGINES:
            run = epoch(f"{prefix}-{label}")
            pts = run["quality"]
            end = pts[-1]["training_seconds"] / 60
            if short == "unsloth":
                ax.plot([end], [shared(f"{prefix}-unsloth-eval-trl")], marker="D", markersize=9, color=t["arms"][colour],
                        markeredgecolor=t["surface"], markeredgewidth=2, linestyle="none", zorder=4,
                        label="Unsloth, fastest: its adapter at the end" if prefix == "gemma" else None)
                continue
            ax.plot([q["training_seconds"] / 60 for q in pts], [q["eval_loss"] for q in pts], color=t["arms"][colour],
                    linewidth=2.5, marker="o", markersize=5, markeredgecolor=t["surface"], zorder=3,
                    label=name if prefix == "gemma" else None)
            ax.annotate(f"{end:.0f} min", (end, pts[-1]["eval_loss"]), textcoords="offset points", xytext=(6, 8),
                        fontsize=10.5, color=t["ink"], fontweight="bold" if short == "meep" else "normal")
        ax.set_title(model, loc="left", fontsize=12, color=t["ink"], fontweight="bold")
        ax.set_xlim(0, 160)
        ax.set_xlabel("Minutes of training", fontsize=10, color=t["muted"])
        ax.tick_params(colors=t["muted"], labelsize=9.5)
        ax.grid(color=t["grid"], linewidth=1, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(t["grid"])
    axes[0].set_ylabel("Held-out loss (lower is better)", fontsize=10, color=t["muted"])
    legend = fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=3, frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_color(t["ink"])
    fig.suptitle("Held-out loss over one epoch", x=0.012, y=0.98, ha="left", fontsize=14, color=t["ink"], fontweight="bold")
    fig.text(0.012, 0.012, "Every 100 steps over 1,137. Meep² and TRL from their own evaluation, which agree; Unsloth's adapter "
             "measured by TRL's evaluation, since its own reads lower on these models.\nMeasured 29 Oct 2026 on one NVIDIA GB10.",
             fontsize=8.5, color=t["muted"], ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.86))
    path = ASSETS / f"epoch-loss-{theme}.png"
    fig.savefig(path, dpi=150, facecolor=t["surface"])
    plt.close(fig)
    return path


if __name__ == "__main__":
    for name in THEMES:
        print(draw_epoch(name))
        print(draw_epoch_loss(name))
