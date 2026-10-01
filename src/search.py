"""Hyperparameter search for YOLOv8n with equal budgets: PSO (Arm C) or random search (Arm B).

    python src/search.py --method pso       # 8 particles x 4 rounds = 32 proxy trainings -> results/search_pso.json
    python src/search.py --method random    # 32 uniform samples (seed 123)              -> results/search_random.json
    python src/search.py --method pso --proxy-epochs 20   # if the Phase 2 proxy check escalated to 20 epochs

Fitness = val mAP@50 of one proxy training, with exactly the proxy settings of src/proxy_check.py (40% subset
configs/data_proxy.yaml, imgsz 416, val=False, SGD, seed 42, epoch-scaled close_mosaic/warmup via train_run).
The best evaluation is written to results/best_<method>.json, which src/train_final.py --config accepts.

Resuming by deterministic replay: the log is rewritten after every evaluation. On restart the algorithm is run
again from its seed; every evaluation already in the log reuses the logged fitness instead of training (after
checking that the recomputed position equals the logged one), and training continues at the first missing
evaluation. A log made with different settings is refused.
"""
import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pso import PSO
from search_space import HIGH, IS_LOG, LOW, NAMES, decode, sample
from training import BATCH, MODEL, REPO, SEED, device_name, train_run, write_json

PSO_COEFFS = {"w_start": 0.9, "w_end": 0.4, "c1": 1.5, "c2": 1.5, "v_max_frac": 0.2, "v_init_frac": 0.1}


def proxy_fitness(params, name, args):
    """The only place that trains: one proxy run, called exactly as proxy_check.py calls its proxy runs."""
    import ultralytics

    result = train_run(params, args.data, args.proxy_epochs, args.imgsz, name, args.project, val=False,
                       model=args.model, device=args.device, plots=False)
    return {**result, "device": device_name(), "ultralytics": ultralytics.__version__,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def finite_result(result, name):
    """A NaN/inf metric means a failed training: it is logged as fitness 0.0 with "failed": true, so the log stays
    valid JSON, replay feeds the optimizer the same value, and a failed run never becomes the best."""
    failed = not math.isfinite(result["map50"])
    result = {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in result.items()}
    if failed:
        print(f"WARNING: {name} returned a non-finite mAP@50; recorded as a failed run with fitness 0.0")
        result.update(map50=0.0, failed=True)
    return result


class RandomSearch:
    """Random search behind the same ask/tell interface as PSO: one round holding all samples."""

    def __init__(self, n_samples, seed):
        self.points, self.n_iters = sample(n_samples, seed), 1

    def ask(self):
        return self.points.copy()

    def tell(self, fitness):
        pass


def make_settings(args):
    settings = {"method": args.method, "seed": args.seed, "proxy_epochs": args.proxy_epochs, "imgsz": args.imgsz,
                "data": Path(args.data).as_posix(), "model": args.model, "smoke": args.smoke, "val": False,
                "train_seed": SEED, "batch": BATCH, "optimizer": "SGD",
                "search_space": {"names": NAMES, "low": LOW.tolist(), "high": HIGH.tolist(), "log10": IS_LOG.tolist()}}
    if args.method == "pso":
        settings.update(n_particles=args.n_particles, n_iters=args.n_iters, pso=PSO_COEFFS)
    else:
        settings.update(n_samples=args.n_samples)
    return settings


def load_or_create_log(path, settings):
    if not path.exists():
        return {"description": f"{settings['method']} search; fitness = val mAP@50 of a proxy training",
                "settings": settings, "evaluations": [], "complete": False}
    log = json.loads(path.read_text())
    old = log.get("settings", {})
    diff = sorted(k for k in settings.keys() | old.keys() if old.get(k) != settings.get(k))
    if diff:
        sys.exit(f"Refusing to resume {path}: it was made with different settings ({', '.join(diff)}). "
                 f"Use the same arguments or another --log.")
    if [e["index"] for e in log["evaluations"]] != list(range(len(log["evaluations"]))):
        sys.exit(f"Refusing to resume {path}: its evaluation indices are not 0, 1, 2, ... in order.")
    return log


def run_search(args, fitness):
    settings = make_settings(args)
    log_path = Path(args.log)
    log = load_or_create_log(log_path, settings)
    done = log["evaluations"]
    print(f"Log: {log_path} ({len(done)} evaluations already done)")
    if args.method == "pso":
        opt = PSO(LOW, HIGH, args.n_particles, args.n_iters, args.seed, **PSO_COEFFS)
    else:
        opt = RandomSearch(args.n_samples, args.seed)
    write_json(log_path, log)

    index = 0
    for rnd in range(opt.n_iters):
        round_fitness = []
        for p, x in enumerate(opt.ask()):
            label = f"r{rnd + 1}_p{p + 1}" if args.method == "pso" else f"s{p + 1:02d}"
            if index < len(done):  # replay: reuse the logged result
                entry = done[index]
                if not np.allclose(entry["vector"], x, rtol=0, atol=1e-6):
                    sys.exit(f"Refusing to resume {log_path}: evaluation {index} ({label}) was logged at "
                             f"{entry['vector']}, but replaying seed {args.seed} gives {np.round(x, 6).tolist()}.")
                print(f"replay {label}: mAP@50 {entry['map50']:.4f} (from the log)")
            else:
                params = decode(x)
                name = f"{args.method}_e{args.proxy_epochs}_{label}"
                print(f"\n=== evaluation {index + 1}: {name} {params}")
                result = finite_result(fitness(params, name, args), name)
                entry = {"index": index, "label": label, "round": rnd + 1 if args.method == "pso" else None,
                         "params": params, "vector": [round(float(v), 6) for v in x], **result}
                done.append(entry)
                write_json(log_path, log)
                print(f"{label}: mAP@50 {entry['map50']:.4f}")
            round_fitness.append(entry["map50"])
            index += 1
        opt.tell(round_fitness)

    log["complete"] = True
    ok = [e for e in done if not e.get("failed")]
    if not ok:
        write_json(log_path, log)
        sys.exit(f"All {len(done)} evaluations in {log_path} failed (non-finite mAP@50); no best file written.")
    best = max(ok, key=lambda e: e["map50"])  # first of equal maxima, as in the PSO's gbest
    log["summary"] = {"evaluations": len(done), "best_index": best["index"], "best_label": best["label"],
                      "best_map50": best["map50"], "best_params": best["params"],
                      "best_so_far": np.maximum.accumulate([e["map50"] for e in done]).tolist()}
    if args.method == "pso":
        log["summary"]["gbest_per_round"] = opt.history["gbest_f"]
    write_json(log_path, log)
    write_json(args.best, {"method": args.method, "source_log": log_path.as_posix(), "index": best["index"],
                           "label": best["label"], "map50": best["map50"], "map50_95": best["map50_95"],
                           "params": best["params"], "vector": best["vector"], "proxy_epochs": args.proxy_epochs,
                           "imgsz": args.imgsz, "seed": args.seed, "smoke": args.smoke})
    print(f"\nBest of {len(done)}: {best['label']} mAP@50 {best['map50']:.4f} {best['params']}\n"
          f"Wrote {log_path} and {args.best}")
    return log


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["pso", "random"], required=True)
    ap.add_argument("--seed", type=int, help="default 42 for pso, 123 for random")
    ap.add_argument("--n-particles", type=int, default=8)
    ap.add_argument("--n-iters", type=int, default=4, help="PSO rounds, the first is the random initial swarm")
    ap.add_argument("--n-samples", type=int, default=32, help="random-search budget")
    ap.add_argument("--proxy-epochs", type=int, default=12, help="20 if the Phase 2 proxy check escalated")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--data", default=str(REPO / "configs" / "data_proxy.yaml"),
                    help="40%% subset yaml from src/training.py")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", help="e.g. 0 or cpu (default: Ultralytics picks)")
    ap.add_argument("--log", help="default results/search_<method>.json (smoke: results/smoke_search_<method>.json)")
    ap.add_argument("--best", help="default results/best_<method>.json (smoke: results/smoke_best_<method>.json)")
    ap.add_argument("--project", help="run dirs; default results/runs (smoke: results/runs_smoke)")
    ap.add_argument("--smoke", action="store_true", help="PSO 2x2 or random 3, 2-epoch proxies: pipeline check")
    args = ap.parse_args(argv)
    if args.seed is None:
        args.seed = 42 if args.method == "pso" else 123
    if args.smoke:  # same smoke proxy as proxy_check.py
        args.n_particles, args.n_iters, args.n_samples, args.proxy_epochs = 2, 2, 3, 2
    prefix = "smoke_" if args.smoke else ""  # smoke runs never write to the real result files
    args.log = args.log or str(REPO / "results" / f"{prefix}search_{args.method}.json")
    args.best = args.best or str(REPO / "results" / f"{prefix}best_{args.method}.json")
    args.project = args.project or str(REPO / "results" / ("runs_smoke" if args.smoke else "runs"))
    return args


def main(argv=None, fitness=proxy_fitness):
    return run_search(parse_args(argv), fitness)


if __name__ == "__main__":
    main()
