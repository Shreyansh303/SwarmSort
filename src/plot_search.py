"""Convergence plot of the Phase 4 searches: best val mAP@50 found so far vs evaluation number.

    python src/plot_search.py        # reads results/search_pso.json and/or results/search_random.json

Either log may be missing (but not both). Each arm is drawn as its best-so-far curve with faint markers for the
individual evaluations; dotted lines mark the PSO round boundaries and a dashed line marks the Ultralytics
defaults trained with the same proxy (from results/proxy_check.json). Also prints a short summary table.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ARMS = {"pso": ("PSO (Arm C)", "#2a78d6", "o"), "random": ("Random search (Arm B)", "#eb6834", "^")}


def load_log(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def summarize(log):
    """Numbers for one arm. A failed run is logged with mAP@50 0.0, so it never becomes the best."""
    evals = log["evaluations"]
    scores = [e["map50"] for e in evals]
    ok = [e for e in evals if not e.get("failed")]
    best = max(ok, key=lambda e: e["map50"]) if ok else None  # first of equal maxima, as search.py picks it
    return {"evaluations": len(evals), "scores": scores,
            "best_so_far": np.maximum.accumulate(scores).tolist() if scores else [],
            "best_map50": best["map50"] if best else None,
            "best_eval": best["index"] + 1 if best else None,  # 1-based, as on the plot's x-axis
            "best_params": best["params"] if best else None}


def defaults_reference(path, proxy_epochs, imgsz):
    """Proxy val mAP@50 of the Ultralytics defaults from the Phase 2 check, or None if not available."""
    check = load_log(path)
    run = (check or {}).get("runs", {}).get(f"proxy_e{proxy_epochs}_s{imgsz}_defaults")
    return run["map50"] if run else None


def plot(summaries, reference, n_particles, out):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    n_max = max(max(s["evaluations"] for s in summaries.values()), 1)
    for b in range(n_particles, n_max, n_particles):
        ax.axvline(b + 0.5, color="#999999", linestyle=":", linewidth=1)
    if reference is not None:
        ax.axhline(reference, color="#555555", linestyle="--", linewidth=1.2,
                   label=f"defaults, same proxy ({reference:.3f})")
    for method, s in summaries.items():
        name, color, marker = ARMS[method]
        x = np.arange(1, s["evaluations"] + 1)
        ax.scatter(x, s["scores"], color=color, marker=marker, s=22, alpha=0.3, linewidths=0)
        ax.step(x, s["best_so_far"], where="post", color=color, linewidth=2,
                label=f"{name}: best {s['best_map50']:.3f}" if s["best_map50"] is not None else name)
    ax.set_xlim(0.5, n_max + 0.5)
    ax.set_xticks([1, *range(n_particles, n_max + 1, n_particles)])
    ax.set_xlabel(f"evaluation (proxy trainings; dotted lines = PSO rounds of {n_particles})")
    ax.set_ylabel("val mAP@50 (best so far; faint markers = each evaluation)")
    ax.set_title("Hyperparameter search convergence at equal budget")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def print_table(summaries, reference):
    print(f"{'arm':<8} {'evals':>5} {'best mAP@50':>11} {'at eval':>7}  best params")
    for method, s in summaries.items():
        best = f"{s['best_map50']:.4f}" if s["best_map50"] is not None else "-"
        print(f"{method:<8} {s['evaluations']:>5} {best:>11} {s['best_eval'] or '-':>7}  {s['best_params']}")
    if reference is not None:
        print(f"defaults (same proxy): {reference:.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pso", default=str(REPO / "results" / "search_pso.json"))
    ap.add_argument("--random", default=str(REPO / "results" / "search_random.json"))
    ap.add_argument("--proxy-check", default=str(REPO / "results" / "proxy_check.json"))
    ap.add_argument("--out", default=str(REPO / "results" / "plots" / "search_convergence.png"))
    args = ap.parse_args(argv)

    logs = {m: load_log(p) for m, p in (("pso", args.pso), ("random", args.random))}
    logs = {m: log for m, log in logs.items() if log is not None}
    if not logs:
        sys.exit(f"No search log found: neither {args.pso} nor {args.random} exists. "
                 f"Download them from the Phase 4 Kaggle notebooks into results/ first.")
    settings = next(iter(logs.values()))["settings"]
    n_particles = logs["pso"]["settings"]["n_particles"] if "pso" in logs else 8
    reference = defaults_reference(args.proxy_check, settings["proxy_epochs"], settings["imgsz"])
    summaries = {m: summarize(log) for m, log in logs.items()}
    plot(summaries, reference, n_particles, args.out)
    print_table(summaries, reference)
    print(f"Wrote {args.out}")
    return summaries, reference


if __name__ == "__main__":
    main()
