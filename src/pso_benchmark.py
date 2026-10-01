"""Sanity check of the hand-written PSO on two standard benchmark functions before it tunes YOLOv8n.

    python src/pso_benchmark.py      # -> results/plots/pso_benchmarks.png

Sphere (one smooth bowl) and Rastrigin (a bowl covered in local minima), both in 6 dimensions like the
hyperparameter space, on [-5.12, 5.12]^6 with the minimum 0 at the origin. The PSO maximizes, so it is given
-f. 20 particles x 50 rounds, repeated for 10 seeds; the plot shows the best value found so far after each round.
"""
import argparse
from pathlib import Path

import numpy as np

from pso import PSO

REPO = Path(__file__).resolve().parent.parent
DIM, BOUND = 6, 5.12


def sphere(x):
    return float(np.sum(x ** 2))


def rastrigin(x):
    return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def minimize(f, seed, n_particles=20, n_iters=50):
    """Run the PSO on f; return the best-so-far f value after each round and the optimizer itself."""
    pso = PSO(np.full(DIM, -BOUND), np.full(DIM, BOUND), n_particles, n_iters, seed)
    for _ in range(n_iters):
        pso.tell([-f(x) for x in pso.ask()])
    return -np.array(pso.history["gbest_f"]), pso


def plot(curves, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (name, runs) in zip(axes, curves.items()):
        rounds = np.arange(1, runs.shape[1] + 1)
        for run in runs:
            ax.plot(rounds, run, color="#4C72B0", alpha=0.25, linewidth=1)
        ax.plot(rounds, np.median(runs, axis=0), color="#1F3A68", linewidth=2.2,
                label=f"median of {len(runs)} seeds")
        ax.set_yscale("log")
        ax.set_title(f"{name} (6-D): final median {np.median(runs[:, -1]):.3g}")
        ax.set_xlabel("round (20 particles evaluated per round)")
        ax.set_ylabel("best f(x) found so far (log scale, optimum 0)")
        ax.grid(True, which="major", alpha=0.3)
        ax.legend()
    fig.suptitle("Hand-written PSO on benchmark functions (minimization)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default=str(REPO / "results" / "plots" / "pso_benchmarks.png"))
    args = ap.parse_args()

    curves = {}
    for name, f in (("Sphere", sphere), ("Rastrigin", rastrigin)):
        curves[name] = np.array([minimize(f, seed)[0] for seed in range(args.seeds)])
        final = curves[name][:, -1]
        print(f"{name:<10} best f after 50 rounds: median {np.median(final):.3g}, "
              f"min {final.min():.3g}, max {final.max():.3g} (start median {np.median(curves[name][:, 0]):.3g})")
    plot(curves, Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
