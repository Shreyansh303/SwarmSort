"""Figures for the Results section, drawn from the result files, plus the derived numbers the text uses.

    python paper/figures/scripts/r_results_figures.py

Writes four 300-dpi PNGs to paper/figures/:
  r_proxy.png          proxy vs medium val mAP@50 of the 8 proxy-check settings (Gen 1)
  r_convergence.png    best-so-far proxy val mAP@50 vs evaluation, Gen 1 and Gen 2 side by side
  r_gen1_per_class.png per-class test AP@50 of the three Gen 1 models
  r_domains.png        Gen 2 test mAP@50 with 95% CIs per test domain and arm

and prints the derived statistics quoted in the text: proxy ranks, search summaries (per-round mean and SD,
evaluations below the defaults' proxy score), the cross-generation Spearman correlation on the
configurations that both generations evaluated, and the recorded GPU time per run (Table 8).

Inputs (read only): results/proxy_check.json, results/search_{pso,random}.json, results/gen2/search_{pso,random}.json,
results/{baseline,arm_b,arm_c}.json, results/gen2/arm_{a,b,c}.json,
results/test_results.json, results/gen2/test_{all,studio,real_world,india}_results.json.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "results"
OUT = REPO / "paper" / "figures"

# Arm colors and labels shared by every figure of the paper.
ARM = {
    "A": {"label": "A: Defaults", "color": "#6b7280", "marker": "o"},
    "B": {"label": "B: Random search", "color": "#2563eb", "marker": "^"},
    "C": {"label": "C: PSO", "color": "#ea580c", "marker": "s"},
}
INK = "#1f2937"  # text and neutral marks
MUTED = "#6b7280"
GRID = "#e5e7eb"
COL_W = 3.5  # IEEE column width, inches
PAGE_W = 7.16  # IEEE text width, inches

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.labelcolor": INK,
    "text.color": INK,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def style_axes(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def ranks(values):
    """1 = lowest, as scipy's rankdata (no ties occur in these data)."""
    order = np.argsort(values)
    r = np.empty(len(values), dtype=int)
    r[order] = np.arange(1, len(values) + 1)
    return r


# --------------------------------------------------------------------------------------------------------------
# Fig. 3: proxy validity
# --------------------------------------------------------------------------------------------------------------
def fig_proxy():
    check = load(RES / "proxy_check.json")["checks"]["proxy_e12_s416 vs medium_e40_s416"]
    ids, prox, med = check["configs"], np.array(check["proxy_map50"]), np.array(check["medium_map50"])
    rho = spearmanr(prox, med).statistic
    print("== Proxy check (Gen 1)")
    print(f"stored rho {check['rho']}, recomputed {rho:.4f}, n {check['n']}")
    for i, p, m, rp, rm in zip(ids, prox, med, ranks(prox), ranks(med)):
        print(f"  {i:<9} proxy {p:.5f} (rank {rp})  medium {m:.5f} (rank {rm})")

    fig, ax = plt.subplots(figsize=(COL_W, 2.55))
    style_axes(ax, "both")
    lhs = [k for k, i in enumerate(ids) if i != "defaults"]
    ax.scatter(prox[lhs], med[lhs], s=30, facecolor="white", edgecolor=INK, linewidth=1.0, zorder=3,
               label="LHS setting")
    d = ids.index("defaults")
    ax.scatter(prox[d], med[d], s=34, marker=ARM["A"]["marker"], color=ARM["A"]["color"], edgecolor="white",
               linewidth=0.8, zorder=4, label="Ultralytics defaults")
    # Label offsets in points. The defaults marker sits next to lhs4, so its label gets a leader line and cannot be
    # read as the label of lhs4.
    offsets = {"lhs1": (5, -3), "lhs2": (5, -9), "lhs3": (-24, 4), "lhs5": (5, -9), "lhs7": (-6, -12),
               "lhs6": (-14, 6), "lhs4": (-6, 6)}
    for k, i in enumerate(ids):
        if i == "defaults":
            ax.annotate(i, (prox[k], med[k]), xytext=(4, -24), textcoords="offset points", fontsize=8, color=INK,
                        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6, shrinkA=0, shrinkB=3))
            continue
        dx, dy = offsets.get(i, (5, 3))
        ax.annotate(i, (prox[k], med[k]), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlim(0.04, 0.54)
    ax.set_ylim(0.25, 0.68)
    ax.set_xlabel("Proxy val mAP@50 (12 epochs, 40% of train)")
    ax.set_ylabel("Medium val mAP@50\n(40 epochs, full train)")
    ax.text(0.97, 0.06, rf"Spearman $\rho$ = {check['rho']:.3f}, $n$ = {check['n']}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8)
    ax.legend(loc="upper left", frameon=False, handletextpad=0.3, borderaxespad=0.2)
    fig.savefig(OUT / "r_proxy.png")
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------------
# Fig. 4: search convergence (and the derived search statistics)
# --------------------------------------------------------------------------------------------------------------
def search_stats(tag, log, defaults=None):
    ev = log["evaluations"]
    s = np.array([e["map50"] for e in ev])
    best = int(np.argmax(s))
    line = (f"  {tag:<11} best {s[best]:.5f} ({ev[best]['label']}, evaluation {best + 1}) "
            f"mean {s.mean():.5f} SD {s.std(ddof=1):.5f} min {s.min():.5f} "
            f"time {sum(e['seconds'] for e in ev):.1f} s")
    if defaults is not None:
        line += f"  below defaults {(s < defaults).sum()} of {len(s)}"
    print(line)
    if ev[0].get("round"):
        for r in sorted({e["round"] for e in ev}):
            rs = np.array([e["map50"] for e in ev if e["round"] == r])
            print(f"    round {r}: mean {rs.mean():.5f} SD {rs.std(ddof=1):.5f} max {rs.max():.5f}")
    return s


def same_params(a, b):
    return all(np.isclose(a["params"][k], b["params"][k], rtol=1e-9, atol=0) for k in a["params"])


def fig_convergence():
    g1 = {"B": load(RES / "search_random.json"), "C": load(RES / "search_pso.json")}
    g2 = {"B": load(RES / "gen2" / "search_random.json"), "C": load(RES / "gen2" / "search_pso.json")}
    defaults = load(RES / "proxy_check.json")["runs"]["proxy_e12_s416_defaults"]["map50"]
    n_particles = g1["C"]["settings"]["n_particles"]

    print("== Search (proxy val mAP@50)")
    print(f"  Gen 1 defaults on the same proxy: {defaults}")
    scores = {}
    for gen, logs, ref in (("Gen 1", g1, defaults), ("Gen 2", g2, None)):
        for arm, log in logs.items():
            scores[(gen, arm)] = search_stats(f"{gen} {arm}", log, ref)

    print("== Cross-generation rank agreement on identical configurations")
    for arm in ("B", "C"):
        e1, e2 = g1[arm]["evaluations"], g2[arm]["evaluations"]
        idx = [i for i in range(min(len(e1), len(e2))) if same_params(e1[i], e2[i])]
        r = spearmanr([e1[i]["map50"] for i in idx], [e2[i]["map50"] for i in idx])
        rounds = sorted({e1[i]["round"] for i in idx}) if e1[0].get("round") else None
        print(f"  {arm}: {len(idx)} identical configurations (indices {idx[0]}-{idx[-1]}"
              f"{', rounds ' + str(rounds) if rounds else ''}); Spearman {r.statistic:.4f}, p {r.pvalue:.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.45))
    panels = (("(a) Generation 1: studio data, 6 classes, 416 px", "Gen 1", g1, (0.08, 0.565)),
              ("(b) Generation 2: studio + real-world data, 7 classes, 640 px", "Gen 2", g2, (0.05, 0.378)))
    for ax, (title, gen, logs, ylim) in zip(axes, panels):
        style_axes(ax)
        n = len(logs["C"]["evaluations"])
        for b in range(n_particles, n, n_particles):
            ax.axvline(b + 0.5, color=MUTED, linestyle=":", linewidth=0.8, zorder=1)
        for r in range(n // n_particles):
            ax.text(r * n_particles + n_particles / 2 + 0.5, ylim[1], f"Round {r + 1}", ha="center",
                    va="top", fontsize=8, color=MUTED)
        if gen == "Gen 1":
            ax.axhline(defaults, color=ARM["A"]["color"], linestyle="--", linewidth=1.0, zorder=2)
        for arm in ("B", "C"):
            s = scores[(gen, arm)]
            x = np.arange(1, len(s) + 1)
            ax.scatter(x, s, s=14, marker=ARM[arm]["marker"], color=ARM[arm]["color"], alpha=0.35, linewidths=0,
                       zorder=3)
            bsf = np.maximum.accumulate(s)
            ax.step(x, bsf, where="post", color=ARM[arm]["color"], linewidth=1.6, zorder=4)
        ax.set_xlim(0.5, n + 0.5)
        ax.set_ylim(*ylim)
        ax.set_xticks([1, 8, 16, 24, 32])
        ax.set_xlabel("Evaluation (proxy training); dotted lines separate PSO rounds")
        ax.set_title(title, loc="left", pad=4)
    axes[0].set_ylabel("Proxy val mAP@50")
    handles = [plt.Line2D([], [], color=ARM[a]["color"], marker=ARM[a]["marker"], markersize=5, linewidth=1.6,
                          label=f"{ARM[a]['label']} (best: Gen 1 {scores[('Gen 1', a)].max():.4f}, "
                                f"Gen 2 {scores[('Gen 2', a)].max():.4f})")
               for a in ("B", "C")]
    handles.append(plt.Line2D([], [], color=ARM["A"]["color"], linestyle="--", linewidth=1.0,
                              label=f"A: Defaults on the same proxy ({defaults:.4f}; Gen 1 only)"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04),
               handlelength=2.2, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.5)
    fig.savefig(OUT / "r_convergence.png")
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------------
# Fig. 6: Gen 1 per-class test AP@50
# --------------------------------------------------------------------------------------------------------------
def fig_gen1_per_class():
    test = load(RES / "test_results.json")
    classes = list(test["models"]["A"]["per_class"].keys())
    boxes = test["boxes_per_class"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.45))
    style_axes(ax, "x")
    y = np.arange(len(classes))[::-1]
    print("== Gen 1 per-class test AP@50")
    for yi, c in zip(y, classes):
        vals = [test["models"][a]["per_class"][c]["ap50"] for a in "ABC"]
        print(f"  {c:<14} " + "  ".join(f"{a} {v:.5f}" for a, v in zip("ABC", vals)))
        ax.plot([min(vals), max(vals)], [yi, yi], color=GRID, linewidth=3.0, solid_capstyle="round", zorder=1)
        # Small vertical dodge per arm so that near-equal values (e.g. METAL A/C, PAPER B/C) stay visible.
        for (a, v), dy in zip(zip("ABC", vals), (0.2, 0.0, -0.2)):
            ax.scatter(v, yi + dy, s=26, marker=ARM[a]["marker"], color=ARM[a]["color"], edgecolor="white",
                       linewidth=0.6, zorder=3)
        if c == "PLASTIC":
            ax.annotate(f"{vals[2]:.3f}", (vals[2], yi - 0.2), xytext=(-5, -1), textcoords="offset points",
                        ha="right", va="center", fontsize=8)
            ax.annotate(f"{vals[0]:.3f}", (vals[0], yi + 0.2), xytext=(6, -1), textcoords="offset points",
                        ha="left", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c} ({boxes[c]:,})" for c in classes])
    ax.set_xlim(0.56, 0.80)
    ax.set_ylim(-0.6, len(classes) - 0.4)
    ax.set_xlabel("Test AP@50 (studio test split, 1,046 images)")
    handles = [plt.Line2D([], [], color=ARM[a]["color"], marker=ARM[a]["marker"], linestyle="none", markersize=5,
                          label=ARM[a]["label"]) for a in "ABC"]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3, frameon=False,
              handletextpad=0.2, columnspacing=1.0)
    fig.savefig(OUT / "r_gen1_per_class.png")
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------------
# Fig. 5: Gen 2 test mAP@50 per domain
# --------------------------------------------------------------------------------------------------------------
def fig_domains():
    tags = [("all", "All"), ("studio", "Studio"), ("real_world", "Real-world"), ("india", "India (DWSD)")]
    runs = {t: load(RES / "gen2" / f"test_{t}_results.json") for t, _ in tags}
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    style_axes(ax)
    width = 0.26
    x = np.arange(len(tags))
    print("== Gen 2 test mAP@50 [95% CI] per domain")
    for j, a in enumerate("ABC"):
        vals = np.array([runs[t]["models"][a]["metrics"]["map50"] for t, _ in tags])
        ci = np.array([runs[t]["models"][a]["bootstrap"]["map50_ci95"] for t, _ in tags])
        err = np.vstack([vals - ci[:, 0], ci[:, 1] - vals])
        xs = x + (j - 1) * width
        ax.bar(xs, vals, width, color=ARM[a]["color"], edgecolor="white", linewidth=0.8, label=ARM[a]["label"],
               zorder=2)
        ax.errorbar(xs, vals, yerr=err, fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.6, capthick=0.7,
                    zorder=3)
        for xi, v, top in zip(xs, vals, ci[:, 1]):
            ax.text(xi, top + 0.012, f"{v:.3f}", rotation=90, ha="center", va="bottom", fontsize=8)
        print(f"  {a}: " + "  ".join(f"{t} {v:.3f} [{lo:.3f}, {hi:.3f}]"
                                     for (t, _), v, (lo, hi) in zip(tags, vals, ci)))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{name}\n({runs[t]['images']:,} img.)" for t, name in tags])
    ax.set_ylim(0, 0.85)
    ax.set_yticks(np.arange(0, 0.81, 0.2))
    ax.set_ylabel("Test mAP@50 (95% CI)")
    ax.legend(loc="upper right", frameon=False, handlelength=1.2, handletextpad=0.4, borderaxespad=0.1)
    fig.savefig(OUT / "r_domains.png")
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------------
# Table 8: recorded GPU training time (no figure; printed for the text)
# --------------------------------------------------------------------------------------------------------------
def compute_cost():
    """Sums of the recorded `seconds` fields: training wall-clock only, idle and setup time excluded."""
    check = load(RES / "proxy_check.json")["runs"]
    rows = {
        "Gen 1": {
            "Proxy-validity check": sum(v["seconds"] for v in check.values()),
            "PSO search": sum(e["seconds"] for e in load(RES / "search_pso.json")["evaluations"]),
            "Random search": sum(e["seconds"] for e in load(RES / "search_random.json")["evaluations"]),
            "Arm A training": load(RES / "baseline.json")["seconds"],
            "Arm B training": load(RES / "arm_b.json")["seconds"],
            "Arm C training": load(RES / "arm_c.json")["seconds"],
        },
        "Gen 2": {
            "PSO search": sum(e["seconds"] for e in load(RES / "gen2" / "search_pso.json")["evaluations"]),
            "Random search": sum(e["seconds"] for e in load(RES / "gen2" / "search_random.json")["evaluations"]),
            "Arm A training": load(RES / "gen2" / "arm_a.json")["seconds"],
            "Arm B training": load(RES / "gen2" / "arm_b.json")["seconds"],
            "Arm C training": load(RES / "gen2" / "arm_c.json")["seconds"],
        },
    }
    print("== Recorded GPU time (Tesla T4)")
    for gen, r in rows.items():
        for name, sec in r.items():
            print(f"  {gen} {name:<22} {sec:9.1f} s  {sec / 3600:5.2f} h")
        total = sum(r.values())
        print(f"  {gen} {'Total':<22} {total:9.1f} s  {total / 3600:5.2f} h")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    compute_cost()
    fig_proxy()
    fig_convergence()
    fig_gen1_per_class()
    fig_domains()
    for name in ("r_proxy.png", "r_convergence.png", "r_gen1_per_class.png", "r_domains.png"):
        print("wrote", (OUT / name).relative_to(REPO), (OUT / name).stat().st_size, "bytes")
