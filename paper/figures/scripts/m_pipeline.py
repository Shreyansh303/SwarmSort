"""Fig. 1 of the paper: the experimental pipeline shared by both generations.

    python paper/figures/scripts/m_pipeline.py      # -> paper/figures/m_pipeline.png

Numbers in the boxes come from results/dataset_report.json, results/gen2/build_report.json,
results/gen2/test_india_results.json and src/search.py / src/pso.py.

The figure is drawn at its printed size (one IEEE column, 3.5 in wide), and all coordinates are in
points, so the font sizes below are the sizes in print.
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

# Same serif family as the other figures of the paper.
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"]})
TITLE_PT, BODY_PT, NOTE_PT = 7.7, 7.2, 7.0      # printed sizes
LINE = 1.16                                     # body line spacing
PAD_TOP, TITLE_GAP, PAD_BOTTOM = 3.0, 1.4, 3.0  # pt
W, MARGIN, GAP = 252.0, 1.5, 4.0                # column width, outer margin, gap between boxes


def box_height(lines):
    return PAD_TOP + TITLE_PT * 1.1 + TITLE_GAP + lines * BODY_PT * LINE + PAD_BOTTOM


# Vertical layout, top to bottom (all in pt).
H_DATA, H_ARM, H_PROXY, H_TRAIN, H_EVAL = (box_height(n) for n in (4, 2, 3, 2, 4))
FORK, ARM_GAP, LABEL_GAP, EVAL_GAP = 11.0, 9.0, 10.0, 7.0
H = 2 * MARGIN + H_DATA + FORK + H_ARM + ARM_GAP + H_PROXY + LABEL_GAP + H_TRAIN + EVAL_GAP + H_EVAL

fig = plt.figure(figsize=(W / 72, H / 72))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def box(x0, x1, top, height, kind, title, body):
    ax.add_patch(FancyBboxPatch((x0, top - height), x1 - x0, height, boxstyle="round,pad=0,rounding_size=3.5",
                                linewidth=0.9, edgecolor=EDGE[kind], facecolor=FILL[kind]))
    cx, y = (x0 + x1) / 2, top - PAD_TOP
    ax.text(cx, y, title, ha="center", va="top", fontweight="bold", color=INK, fontsize=TITLE_PT)
    ax.text(cx, y - TITLE_PT * 1.1 - TITLE_GAP, body, ha="center", va="top", color=INK, fontsize=BODY_PT,
            linespacing=LINE)
    return top - height


def arrow(x0, y0, x1, y1, both=False):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="<|-|>" if both else "-|>", mutation_scale=6,
                                 linewidth=0.8, color=INK, shrinkA=0, shrinkB=0))


def line(xs, ys):
    ax.plot(xs, ys, color=INK, linewidth=0.8, solid_capstyle="butt")


L, R = MARGIN + 0.5, W - MARGIN - 0.5
HALF = (L + R) / 2
ARM_W = (R - L - 2 * GAP) / 3
A0, B0, C0 = L, L + ARM_W + GAP, L + 2 * (ARM_W + GAP)
XA, XB, XC = A0 + ARM_W / 2, B0 + ARM_W / 2, C0 + ARM_W / 2

# Data
top = H - MARGIN
bottom = box(L, HALF - GAP / 2, top, H_DATA, "data", "Generation 1 (studio)",
             "Kaggle Garbage Detection\n6 classes, 416 px\ntrain / val / test:\n7,324 / 2,094 / 1,046 images")
box(HALF + GAP / 2, R, top, H_DATA, "data", "Generation 2 (mixed)",
    "Gen 1 subsample + TACO + HITL\n7 classes (OTHER added), 640 px\n6,292 / 1,841 / 1,514 images\n"
    "+ DWSD India: 700 test-only")
bar = bottom - FORK * 0.4
line([(L + HALF) / 2] * 2, [bottom, bar])
line([(HALF + R) / 2] * 2, [bottom, bar])
line([XA, XC], [bar, bar])
for x in (XA, XB, XC):
    arrow(x, bar, x, bottom - FORK)

# Arms
top = bottom - FORK
box(A0, A0 + ARM_W, top, H_ARM, "a", "Arm A: defaults", "Ultralytics values\nno search")
box(B0, B0 + ARM_W, top, H_ARM, "b", "Arm B: random", "32 i.i.d. samples\nseed 123")
arms_bottom = box(C0, C0 + ARM_W, top, H_ARM, "c", "Arm C: PSO", "8 particles × 4 rounds\nseed 42")

# Proxy fitness shared by B and C
top = arms_bottom - ARM_GAP
proxy_bottom = box(B0, R, top, H_PROXY, "proxy", "Proxy fitness (32 trainings per arm)",
                   "12 epochs on a stratified 40% train subset\nfitness = val mAP@50\n"
                   "checked once (Gen 1) against 40-epoch runs")
arrow(XB, arms_bottom, XB, top, both=True)
arrow(XC, arms_bottom, XC, top, both=True)

# Full training
top = proxy_bottom - LABEL_GAP
train_bottom = box(L, R, top, H_TRAIN, "train", "Full training of each arm's chosen setting",
                   "100 epochs, SGD, batch 16, seed 42, identical recipe per generation\nbest.pt selected on val")
arrow(XA, arms_bottom, XA, top)
x_mid = (B0 + R) / 2
arrow(x_mid, proxy_bottom, x_mid, top)
ax.text(x_mid + 3, (proxy_bottom + top) / 2, "best setting of B and of C", ha="left", va="center", color=INK,
        fontsize=NOTE_PT, style="italic")

# Evaluation
top = train_bottom - EVAL_GAP
box(L, R, top, H_EVAL, "eval", "Test evaluation (test split used once)",
    "mAP@50 and mAP@50-95 from the Ultralytics validator\npaired image bootstrap: 1,000 resamples, 95% CIs\n"
    "for every model and every pairwise difference\nGen 2: all, studio, real-world and India test sets")
arrow(W / 2, train_bottom, W / 2, top)

fig.savefig(OUT, dpi=400)
print(f"Wrote {OUT} ({W:.0f} x {H:.0f} pt)")
