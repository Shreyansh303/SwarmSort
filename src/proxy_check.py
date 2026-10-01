"""Proxy-validity check: does the cheap proxy training rank configurations like a longer training does?

    python src/proxy_check.py                     # proxy: 12 epochs on the 40% subset; medium: 40 epochs on 100%
    python src/proxy_check.py --proxy-epochs 20   # escalation: reuses the finished medium runs

Eight configurations (7 Latin-hypercube samples in search units + the Ultralytics defaults) are trained with the
proxy and the medium settings, all at imgsz 416 (the images' native size). The Spearman rank correlation of their
val mAP@50 decides validity (pass: rho >= 0.6).

The log is rewritten after every finished run and finished runs are skipped on restart, so an interrupted
session loses at most the run in progress. Runs are keyed by their settings (e.g. proxy_e12_s416_lhs1), so an
escalated proxy lands in the same log next to the old one and the medium runs are not repeated.
"""
import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import ultralytics
from scipy.stats import spearmanr

from search_space import DEFAULTS, NAMES, decode, latin_hypercube
from training import BATCH, MODEL, REPO, SEED, device_name, train_run, write_json


def make_configs(n_samples, sample_seed):
    configs = [{"id": f"lhs{i + 1}", "vector": [round(float(x), 6) for x in v], "params": decode(v)}
               for i, v in enumerate(latin_hypercube(n_samples, sample_seed))]
    return configs + [{"id": "defaults", "vector": None, "params": dict(DEFAULTS)}]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(REPO / "configs" / "data.yaml"), help="full train split (medium runs)")
    ap.add_argument("--proxy-data", default=str(REPO / "configs" / "data_proxy.yaml"),
                    help="40%% subset yaml from src/training.py (proxy runs)")
    ap.add_argument("--proxy-epochs", type=int, default=12)
    ap.add_argument("--proxy-imgsz", type=int, default=416)
    ap.add_argument("--medium-epochs", type=int, default=40)
    ap.add_argument("--medium-imgsz", type=int, default=416)
    ap.add_argument("--n-samples", type=int, default=7, help="Latin-hypercube configs; the defaults are added")
    ap.add_argument("--sample-seed", type=int, default=SEED)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--log", help="default results/proxy_check.json (smoke: results/smoke_proxy_check.json)")
    ap.add_argument("--project", help="run dirs; default results/runs (smoke: results/runs_smoke)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", help="e.g. 0 or cpu (default: Ultralytics picks)")
    ap.add_argument("--smoke", action="store_true", help="2- and 3-epoch runs: pipeline check only")
    args = ap.parse_args()
    if args.smoke:
        args.proxy_epochs, args.medium_epochs = 2, 3
    # smoke runs never write to the real result files
    args.log = args.log or str(REPO / "results" / ("smoke_proxy_check.json" if args.smoke else "proxy_check.json"))
    args.project = args.project or str(REPO / "results" / ("runs_smoke" if args.smoke else "runs"))
    print(f"Log: {args.log}")
    print(f"Run dirs: {args.project}")

    configs = make_configs(args.n_samples, args.sample_seed)
    # Epochs and imgsz are not settings: they are part of each run's name, so an escalated proxy shares the log
    settings = {"sampling": "latin hypercube (random-cd)", "n_samples": args.n_samples,
                "sample_seed": args.sample_seed, "train_seed": SEED, "batch": BATCH, "optimizer": "SGD",
                "model": args.model, "data": Path(args.data).as_posix(), "proxy_data": Path(args.proxy_data).as_posix(),
                "search_space": NAMES, "smoke": args.smoke}
    log_path = Path(args.log)
    if log_path.exists():
        log = json.loads(log_path.read_text())
        diff = sorted(k for k in settings.keys() | log.get("settings", {}).keys()
                      if log.get("settings", {}).get(k) != settings.get(k))
        if diff or log["configs"] != configs:
            print(f"Refusing to resume {log_path}: it was made with different settings "
                  f"({', '.join(diff) or 'configs'}). Use the same arguments or another --log.")
            sys.exit(1)
        print(f"Resuming {log_path}: {len(log['runs'])} finished runs")
    else:
        log = {"description": "Spearman rank correlation of val mAP@50 between proxy and medium trainings",
               "settings": settings, "configs": configs, "runs": {}, "checks": {}}

    stages = [("proxy", args.proxy_data, args.proxy_epochs, args.proxy_imgsz),
              ("medium", args.data, args.medium_epochs, args.medium_imgsz)]
    tags = {}
    for kind, data, epochs, imgsz in stages:  # all cheap proxy runs first, then the medium runs
        tags[kind] = tag = f"{kind}_e{epochs}_s{imgsz}"
        for cfg in configs:
            name = f"{tag}_{cfg['id']}"
            if name in log["runs"]:
                print(f"skip {name}: already in the log (val mAP@50 {log['runs'][name]['map50']:.4f})")
                continue
            print(f"\n=== {name}: {cfg['params']}")
            result = train_run(cfg["params"], data, epochs, imgsz, name, args.project, val=False,
                               model=args.model, device=args.device, plots=False)
            log["runs"][name] = {"config": cfg["id"], "kind": kind, "epochs": epochs, "imgsz": imgsz,
                                 "data": Path(data).as_posix(), **result, "device": device_name(),
                                 "ultralytics": ultralytics.__version__,
                                 "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            write_json(log_path, log)

    proxy = [log["runs"][f"{tags['proxy']}_{c['id']}"]["map50"] for c in configs]
    medium = [log["runs"][f"{tags['medium']}_{c['id']}"]["map50"] for c in configs]
    rho, p = spearmanr(proxy, medium)
    defined = not math.isnan(rho)  # undefined when one list is constant
    check = {"configs": [c["id"] for c in configs], "proxy_map50": proxy, "medium_map50": medium,
             "rho": round(float(rho), 4) if defined else None, "p_value": round(float(p), 4) if defined else None,
             "n": len(configs), "threshold": args.threshold,
             "devices": sorted({log["runs"][f"{t}_{c['id']}"]["device"] for t in tags.values() for c in configs}),
             "ultralytics": ultralytics.__version__,
             "verdict": ("pass" if rho >= args.threshold else "fail") if defined else "fail (rho undefined)"}
    log["checks"][f"{tags['proxy']} vs {tags['medium']}"] = check
    write_json(log_path, log)

    print(f"\n{'config':<10}{'proxy mAP50':>12}{'medium mAP50':>14}")
    for c, a, b in zip(configs, proxy, medium):
        print(f"{c['id']:<10}{a:>12.4f}{b:>14.4f}")
    print(f"Spearman rho = {check['rho']}, p = {check['p_value']}, n = {check['n']} -> {check['verdict'].upper()} "
          f"(threshold {args.threshold})\nWrote {log_path}")


if __name__ == "__main__":
    main()
