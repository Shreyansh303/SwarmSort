"""Fig. 1 of the paper: the experimental pipeline shared by both generations.

    python paper/figures/scripts/m_pipeline.py      # -> paper/figures/m_pipeline.png

Numbers in the boxes come from results/dataset_report.json, results/gen2/build_report.json,
results/gen2/test_india_results.json and src/search.py / src/pso.py.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "m_pipeline.png"

INK = "#1f2933"
EDGE = {"data": "#3b5b7f", "a": "#6b7280", "b": "#2563eb", "c": "#ea580c", "proxy": "#6b5b95", "train": "#555f6b",
        "eval": "#7a5c12"}
FILL = {"data": "#e9eff6", "a": "#f3f4f6", "b": "#dbeafe", "c": "#ffedd5", "proxy": "#efecf6", "train": "#eef0f2",
        "eval": "#f8f2e1"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.2})
fig = plt.figure(figsize=(3.5, 4.55))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 130)
ax.axis("off")


def box(x0, y0, x1, y1, kind, title, body):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.6",
                                linewidth=0.9, edgecolor=EDGE[kind], facecolor=FILL[kind]))
    cx, top = (x0 + x1) / 2, y1 - 2.6
    ax.text(cx, top, title, ha="center", va="top", fontweight="bold", color=INK, fontsize=6.6)
    ax.text(cx, top - 4.6, body, ha="center", va="top", color=INK, linespacing=1.35)


def arrow(x0, y0, x1, y1, both=False):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="<|-|>" if both else "-|>", mutation_scale=6,
                                 linewidth=0.8, color=INK, shrinkA=0, shrinkB=0))


def line(xs, ys):
    ax.plot(xs, ys, color=INK, linewidth=0.8, solid_capstyle="butt")


# Data
box(2, 103, 48, 128, "data", "Generation 1 (studio)",
    "Kaggle Garbage Detection\n6 classes, 416 px\ntrain / val / test:\n7,324 / 2,094 / 1,046 images")
box(52, 103, 98, 128, "data", "Generation 2 (mixed)",
    "Gen 1 subsample + TACO + HITL\n7 classes (OTHER added), 640 px\n6,292 / 1,841 / 1,514 images\n"
    "+ DWSD India: 700 test-only")
line([25, 25], [103, 99.5])
line([75, 75], [103, 99.5])
line([17, 83], [99.5, 99.5])
for x in (17, 50, 83):
    arrow(x, 99.5, x, 95)

# Arms
box(2, 78, 32, 95, "a", "Arm A: defaults", "Ultralytics values\nno search")
box(35, 78, 65, 95, "b", "Arm B: random", "32 i.i.d. samples\nseed 123")
box(68, 78, 98, 95, "c", "Arm C: PSO", "8 particles × 4 rounds\nseed 42")

# Proxy fitness shared by B and C
box(35, 53, 98, 71, "proxy", "Proxy fitness (32 trainings per arm)",
    "12 epochs on a stratified 40% train subset\nfitness = val mAP@50\nchecked once (Gen 1) against 40-epoch runs")
arrow(50, 78, 50, 71, both=True)
arrow(83, 78, 83, 71, both=True)

# Full training
box(2, 31, 98, 46, "train", "Full training of each arm's chosen setting",
    "100 epochs, SGD, batch 16, seed 42, identical recipe per generation\nbest.pt selected on val")
arrow(17, 78, 17, 46)
arrow(66.5, 53, 66.5, 46)
ax.text(68.5, 49.5, "best setting of B and of C", ha="left", va="center", color=INK, fontsize=5.6,
        style="italic")

# Evaluation
box(2, 2, 98, 24, "eval", "Test evaluation (test split used once)",
    "mAP@50 and mAP@50-95 from the Ultralytics validator\npaired image bootstrap: 1,000 resamples, 95% CIs\n"
    "for every model and every pairwise difference\nGen 2: all, studio, real-world and India test sets")
arrow(50, 31, 50, 24)

fig.savefig(OUT, dpi=400)
print(f"Wrote {OUT}")
