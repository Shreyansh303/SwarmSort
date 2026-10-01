"""Full training of one hyperparameter configuration (Arm A now; the Arm B/C winners later).

    python src/train_final.py                    # Arm A: Ultralytics defaults, 100 epochs @416 -> results/baseline.json
    python src/train_final.py --config best_pso.json --name arm_c --out results/arm_c.json

--config takes a JSON file holding the searched hyperparameters, either at the top level or under "params".
Validation runs every epoch; the reported metrics are the val-set metrics of best.pt.
"""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import ultralytics

from search_space import DEFAULTS, NAMES
from training import BATCH, MODEL, REPO, SEED, device_name, train_run, write_json


def load_params(path):
    if path is None:
        return dict(DEFAULTS)
    cfg = json.loads(Path(path).read_text())
    cfg = cfg.get("params", cfg)
    missing = set(NAMES) - cfg.keys()
    if missing:
        raise SystemExit(f"{path} lacks hyperparameters {sorted(missing)}")
    return {k: float(cfg[k]) for k in NAMES}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON with the hyperparameters (default: Ultralytics defaults)")
    ap.add_argument("--name", default="baseline", help="run name, also used for results/runs/<name>")
    ap.add_argument("--out", help="default results/baseline.json (smoke: results/smoke_baseline.json)")
    ap.add_argument("--data", default=str(REPO / "configs" / "data.yaml"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=416, help="dataset images are natively 416x416")
    ap.add_argument("--project", help="run dirs; default results/runs (smoke: results/runs_smoke)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", help="e.g. 0 or cpu (default: Ultralytics picks)")
    ap.add_argument("--smoke", action="store_true", help="3 epochs: checks the pipeline only")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 3
    # smoke runs never write to the real result files
    prefix = "smoke_" if args.smoke else ""
    args.out = args.out or str(REPO / "results" / f"{prefix}baseline.json")
    args.project = args.project or str(REPO / "results" / ("runs_smoke" if args.smoke else "runs"))
    print(f"Output: {args.out}")
    print(f"Run dir: {Path(args.project) / args.name}")

    params = load_params(args.config)
    print(f"Training '{args.name}' for {args.epochs} epochs at imgsz {args.imgsz} with {params}")
    result = train_run(params, args.data, args.epochs, args.imgsz, args.name, args.project, val=True,
                       model=args.model, device=args.device)

    out = Path(args.out)
    curves = out.with_name(out.stem + "_epochs.csv")  # per-epoch metrics, small enough to commit
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(result["run_dir"]) / "results.csv", curves)
    write_json(out, {
        "name": args.name,
        "config_source": args.config or "ultralytics defaults",
        "params": params,
        "epochs": args.epochs, "imgsz": args.imgsz, "batch": BATCH, "seed": SEED,
        "data": Path(args.data).as_posix(),
        "metrics_split": "val (best.pt)",
        **result,
        "epoch_curves": curves.name,
        "device": device_name(),
        "ultralytics": ultralytics.__version__,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "smoke": args.smoke,
    })
    print(f"\nval mAP@50 {result['map50']:.4f}  mAP@50-95 {result['map50_95']:.4f}  "
          f"({result['seconds'] / 60:.1f} min)\nWrote {out} and {curves}")


if __name__ == "__main__":
    main()
