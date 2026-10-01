"""Shared training code for Phase 2+: one YOLOv8n training run, the stratified proxy subset, and a safe JSON writer.

Build the proxy subset (40% of the train list, stratified by source class):
    python src/training.py --data configs/data.yaml
writes configs/lists/train_proxy.txt and configs/data_proxy.yaml (same val/test/names, train = the subset).
"""
import argparse
import csv
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
SPLIT_CSV = REPO / "configs" / "split.csv"
SEED = 42
BATCH = 16
MODEL = "yolov8n.pt"


def train_run(params, data, epochs, imgsz, name, project, val, model=MODEL, device=None, plots=True):
    """Train YOLOv8n once and return its final validation metrics, wall-clock time and run directory.

    params: the searched hyperparameters (see search_space.py). Everything else is fixed here so that every arm,
    proxy and full run trains under identical rules:
      - optimizer="SGD": the default "auto" ignores lr0/momentum and switches AdamW <-> MuSGD by run length
      - close_mosaic = 10% and warmup_epochs = 3% of the epochs (Ultralytics' 10 and 3 for 100 epochs), so mosaic
        (a searched knob) and LR warmup cover the same share of a 12-epoch proxy as of a full run. Ultralytics 8.4
        counts warmup in batches (round(warmup_epochs x batches per epoch)) with no minimum iteration count.
      - the training subset is chosen through `data`, never through Ultralytics' `fraction` (not a random sample)
    With val=False Ultralytics still validates the last epoch; with val=True the reported metrics are best.pt's.
    """
    from ultralytics import YOLO
    # Ultralytics imports this module lazily during the first training of a process, after seeding, and its import
    # draws from Python's `random`; that made the first run of a process differ from later ones (and resumed
    # runs differ from uninterrupted ones). Importing it here, before seeding, keeps every run identical.
    import ultralytics.utils.events  # noqa: F401

    args = dict(data=str(data), epochs=epochs, imgsz=imgsz, batch=BATCH, seed=SEED, deterministic=True,
                optimizer="SGD", close_mosaic=round(0.1 * epochs), warmup_epochs=round(0.03 * epochs, 4),
                val=val, plots=plots, project=str(Path(project).resolve()), name=name, exist_ok=True, **params)
    if device is not None:
        args["device"] = device
    start = time.time()
    yolo = YOLO(model)
    metrics = yolo.train(**args)
    seconds = time.time() - start
    trainer = yolo.trainer
    return {
        "map50": round(float(metrics.box.map50), 5),
        "map50_95": round(float(metrics.box.map), 5),
        "precision": round(float(metrics.box.mp), 5),
        "recall": round(float(metrics.box.mr), 5),
        "seconds": round(seconds, 1),
        "run_dir": Path(trainer.save_dir).as_posix(),
        # proof that the requested optimizer and lr0 were really used
        "optimizer": type(trainer.optimizer).__name__,
        "initial_lr": trainer.optimizer.param_groups[0]["initial_lr"],
        "train_args": {k: v for k, v in args.items() if k not in ("project", "name", "exist_ok")},
    }


def write_json(path, obj):
    """Write JSON atomically, so a session killed mid-write never leaves a corrupt log behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def device_name():
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def build_proxy_subset(data_yaml, out_yaml, out_list, fraction=0.4, seed=SEED, split_csv=SPLIT_CSV):
    """Write a stratified random `fraction` of the train list and a data yaml that trains on it.

    Each source class contributes fraction x its size (largest-remainder rounding so the total is exactly
    round(fraction x N)); images are drawn with a seeded permutation, so the subset is the same on every machine.
    """
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    train_list = Path(cfg["train"])
    if not train_list.is_absolute():
        train_list = Path(cfg["path"]) / train_list
    paths = [line.strip() for line in train_list.read_text().splitlines() if line.strip()]
    with open(split_csv, newline="") as f:
        class_of = {row["file"]: row["source_class"] for row in csv.DictReader(f)}
    by_class = defaultdict(list)
    for p in paths:
        by_class[class_of[Path(p).name]].append(p)

    quota = {c: fraction * len(v) for c, v in by_class.items()}
    counts = {c: int(q) for c, q in quota.items()}
    leftover = round(fraction * len(paths)) - sum(counts.values())
    for c in sorted(quota, key=lambda c: (counts[c] - quota[c], c))[:leftover]:
        counts[c] += 1

    rng = np.random.default_rng(seed)
    chosen = set()
    for c in sorted(by_class):
        members = sorted(by_class[c])
        chosen.update(members[i] for i in rng.permutation(len(members))[:counts[c]])
    subset = [p for p in paths if p in chosen]  # keep the original list order

    out_list, out_yaml = Path(out_list), Path(out_yaml)
    out_list.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_list.write_text("".join(p + "\n" for p in subset))
    out_yaml.write_text(yaml.safe_dump({**cfg, "train": out_list.resolve().as_posix()}, sort_keys=False))

    full = Counter(class_of[Path(p).name] for p in paths)
    part = Counter(class_of[Path(p).name] for p in subset)
    return {c: {"full": full[c], "subset": part[c], "full_share": full[c] / len(paths),
                "subset_share": part[c] / len(subset)} for c in sorted(full)}, len(paths), len(subset)


def main():
    ap = argparse.ArgumentParser(description="Build the stratified proxy training subset.")
    ap.add_argument("--data", default=str(REPO / "configs" / "data.yaml"), help="full data yaml")
    ap.add_argument("--fraction", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--split-csv", default=str(SPLIT_CSV))
    ap.add_argument("--out-yaml", default=str(REPO / "configs" / "data_proxy.yaml"))
    ap.add_argument("--out-list", default=str(REPO / "configs" / "lists" / "train_proxy.txt"))
    args = ap.parse_args()

    stats, n_full, n_sub = build_proxy_subset(args.data, args.out_yaml, args.out_list, args.fraction,
                                              args.seed, args.split_csv)
    print(f"Proxy subset: {n_sub} of {n_full} train images ({n_sub / n_full:.1%})")
    for c, s in stats.items():
        print(f"  {c:<14} {s['subset']:>5}/{s['full']:<5} share {s['subset_share']:.2%} (full {s['full_share']:.2%})")
    print(f"Wrote {args.out_list} and {args.out_yaml}")


if __name__ == "__main__":
    main()
