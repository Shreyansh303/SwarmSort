"""Shared training code for Phase 2+: one YOLOv8n training run, the stratified proxy subset, and a safe JSON writer.

Build the proxy subset (40% of the train list, stratified by source class):
    python src/training.py --data configs/data.yaml
writes configs/lists/train_proxy.txt and configs/data_proxy.yaml (same val/test/names, train = the subset).
"""
import argparse
import csv
import json
import math
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


def run_args(params, data, epochs, imgsz, name, project, val, device=None, plots=True):
    """The Ultralytics arguments train_run passes for these inputs (a copy of its rules, used to check a resume;
    tests/test_resume.py checks that the two stay the same)."""
    args = dict(data=str(data), epochs=epochs, imgsz=imgsz, batch=BATCH, seed=SEED, deterministic=True,
                optimizer="SGD", close_mosaic=round(0.1 * epochs), warmup_epochs=round(0.03 * epochs, 4),
                val=val, plots=plots, project=str(Path(project).resolve()), name=name, exist_ok=True, **params)
    if device is not None:
        args["device"] = device
    return args


NOT_CHECKED_ON_RESUME = ("project", "name", "exist_ok", "device")  # where and on which device, not how


def resume_mismatches(saved, wanted):
    """The training arguments in which a checkpoint's saved args differ from `wanted` (run_args), as text lines."""
    diff = []
    for k, v in wanted.items():
        if k in NOT_CHECKED_ON_RESUME:
            continue
        if k not in saved:
            diff.append(f"{k}: missing in the checkpoint, wanted {v!r}")
            continue
        s = saved[k]
        if k == "data":
            same = Path(str(s)).resolve() == Path(str(v)).resolve()
        elif isinstance(v, bool) or isinstance(s, bool) or isinstance(v, str) or isinstance(s, str):
            same = s == v
        else:
            same = math.isclose(float(s), float(v), rel_tol=1e-9, abs_tol=1e-12)
        if not same:
            diff.append(f"{k}: checkpoint {s!r}, wanted {v!r}")
    return diff


def trim_epochs_csv(path, epochs_done):
    """Drop rows of an Ultralytics results.csv beyond `epochs_done` (a session killed between writing the row and
    last.pt leaves one row too many) and return the remaining rows as dicts."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="") as f:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    keep = [r for r in rows if int(float(r["epoch"])) <= epochs_done]
    if len(keep) < len(rows):
        lines = path.read_text().splitlines(keepends=True)
        path.write_text("".join(lines[:1 + len(keep)]))
        print(f"Dropped {len(rows) - len(keep)} row(s) beyond epoch {epochs_done} from {path}")
    return keep


def csv_training_seconds(rows):
    """Training time in an Ultralytics results.csv. Its time column counts from the start of each session and
    restarts at a resume, so the last value of every session is added up."""
    total, prev = 0.0, None
    for r in rows:
        t = float(r.get("time") or 0)
        if prev is not None and t < prev:
            total += prev
        prev = t
    return total + (prev or 0.0)


def resume_run(params, data, epochs, imgsz, name, project, val, model=MODEL, device=None, plots=True):
    """Continue an interrupted train_run from <project>/<name>/weights/last.pt. Returns train_run's fields plus
    resumed (True), resumed_from_epoch (epochs finished before) and resume_seconds (this session).

    Ultralytics' resume (YOLO(last.pt).train(resume=True)) restores every training argument from the checkpoint; only
    the data yaml, the device and the run folder are passed. Before that, the saved arguments are compared with what
    train_run would pass now (run_args), and again with what the resumed trainer really uses, so a resume never
    continues a run made with another recipe or other hyperparameters. `model` is not needed: last.pt holds the
    weights. If last.pt has no optimizer state (Ultralytics strips it once all epochs are done) or all epochs are
    done, only the final validation of best.pt is missing, and it is run again.
    "seconds" = the training time already in results.csv + this session's time.
    """
    from ultralytics import YOLO
    import ultralytics.utils.events  # noqa: F401  (imported before seeding, as in train_run)
    from ultralytics.nn.tasks import torch_safe_load

    run_dir = Path(project).resolve() / name
    last, best = run_dir / "weights" / "last.pt", run_dir / "weights" / "best.pt"
    wanted = run_args(params, data, epochs, imgsz, name, project, val, device=device, plots=plots)
    ckpt, _ = torch_safe_load(last)
    diff = resume_mismatches(ckpt.get("train_args") or {}, wanted)
    if diff:
        raise RuntimeError(f"Cannot resume {last}: it was trained with other arguments:\n  " + "\n  ".join(diff)
                           + f"\nUse the same arguments, or delete {run_dir} to start a new training.")
    done = ckpt.get("epoch", -1) + 1
    finished = ckpt.get("optimizer") is None or done >= epochs
    rows = trim_epochs_csv(run_dir / "results.csv", epochs if finished else done)
    before = csv_training_seconds(rows)
    dev = {} if device is None else {"device": device}
    start = time.time()
    if finished:
        print(f"{last}: all {epochs} epochs are done; validating best.pt again (the final step that was missing)")
        if not best.exists():
            raise FileNotFoundError(f"{best} not found: cannot finish {run_dir}")
        import torch

        metrics = YOLO(str(best)).val(data=str(data), imgsz=imgsz, batch=BATCH * 2, split="val", plots=plots,
                                      half=torch.cuda.is_available() and str(device) != "cpu",
                                      project=str(run_dir), name="final_val", exist_ok=True, **dev)
        optimizer, initial_lr = wanted["optimizer"], wanted["lr0"]
        resumed_from = epochs
    else:
        print(f"Resuming {last}: {done} of {epochs} epochs done, continuing at epoch {done + 1}")
        yolo = YOLO(str(last))
        metrics = yolo.train(resume=True, data=str(data), save_dir=str(run_dir), **dev)
        trainer = yolo.trainer
        diff = resume_mismatches(vars(trainer.args), wanted)
        if diff:  # cannot happen unless Ultralytics' resume changes; then the result must not pass as this config
            raise RuntimeError("The resumed training used other arguments:\n  " + "\n  ".join(diff))
        optimizer = type(trainer.optimizer).__name__
        initial_lr = trainer.optimizer.param_groups[0]["initial_lr"]
        resumed_from = done
    seconds = time.time() - start
    result = {
        "map50": round(float(metrics.box.map50), 5),
        "map50_95": round(float(metrics.box.map), 5),
        "precision": round(float(metrics.box.mp), 5),
        "recall": round(float(metrics.box.mr), 5),
        "seconds": round(before + seconds, 1),
        "run_dir": run_dir.as_posix(),
        "optimizer": optimizer,
        "initial_lr": initial_lr,
        "train_args": {k: v for k, v in wanted.items() if k not in ("project", "name", "exist_ok")},
        "resumed": True,
        "resumed_from_epoch": resumed_from,
        "resume_seconds": round(seconds, 1),
    }
    if finished:
        result["metrics_from"] = "best.pt validated again on resume (the training had finished)"
    return result


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
