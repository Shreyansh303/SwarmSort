"""Phase 5: score the trained arms once on the held-out test split, and draw the report figures.

    python src/evaluate.py                                # A, B and C on the test split -> results/test_*
    python src/evaluate.py --models A=results/runs/baseline/weights/best.pt --split val   # checks use val only

Steps:
  1. Checkpoint check: each best.pt must be a 6-class model trained for 100 epochs at imgsz 416 with its arm's
     hyperparameters (A: results/baseline.json, B: results/best_random.json, C: results/best_pso.json).
  2. Metrics: Ultralytics' own validator with the training-time validation settings (imgsz 416, conf 0.001,
     iou 0.7, rect batches of 32, max_det from the checkpoint). It also records every image's statistics.
  3. Speed: median batch-1 latency over 100 images of the split (10 warm-up predictions first).
  4. Paired bootstrap: resample the images with replacement (the same images for every model) to get 95% CIs
     of mAP@50 and mAP@50-95 and of every pairwise difference. A difference whose CI contains 0 is not significant.
  5. Outputs: <out-dir>/<split>_results.json, <out-dir>/<split>_comparison.md and <out-dir>/plots/.

The test split is meant to be scored exactly once, for the final comparison. Use --split val for any trial run.
"""
import argparse
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from ultralytics.models.yolo.detect import DetectionValidator  # noqa: E402
from ultralytics.utils.metrics import DetMetrics, ap_per_class  # noqa: E402

from search_space import NAMES  # noqa: E402
from training import REPO, write_json  # noqa: E402

# ---------------------------------------------------------------------------------------------------------------
# Settings and the three arms
# ---------------------------------------------------------------------------------------------------------------
IMGSZ = 416        # native image size, as in every training
EPOCHS = 100       # full-training length every checkpoint must have
VAL_BATCH = 32     # the trainer validates with 2 x batch 16 and rect batches; all images are 416x416 anyway
SAMPLE_CONF = 0.25  # confidence threshold for drawn boxes and for the latency runs (Ultralytics' predict default)
LATENCY_IMAGES, LATENCY_WARMUP = 100, 10
REL_TOL = 1e-6     # tolerance of the hyperparameter comparison
CONSISTENCY_TOL = 1e-4

ARMS = {
    "A": {"name": "Defaults", "model": "results/runs/baseline/weights/best.pt", "config": "results/baseline.json",
          "val_json": "results/baseline.json", "epochs_csv": "results/baseline_epochs.csv",
          "notebook": "notebooks/phase2a_baseline.ipynb", "download": "results/runs/baseline/weights/best.pt"},
    "B": {"name": "Random search", "model": "results/runs/arm_b/weights/best.pt", "config": "results/best_random.json",
          "val_json": "results/arm_b.json", "epochs_csv": "results/arm_b_epochs.csv",
          "notebook": "notebooks/phase5a_train_arm_b.ipynb", "download": "arm_b_best.pt"},
    "C": {"name": "PSO", "model": "results/runs/arm_c/weights/best.pt", "config": "results/best_pso.json",
          "val_json": "results/arm_c.json", "epochs_csv": "results/arm_c_epochs.csv",
          "notebook": "notebooks/phase5b_train_arm_c.ipynb", "download": "arm_c_best.pt"},
}

# one fixed colour per model and per class (validated categorical palette; B and C as in search_convergence.png)
MODEL_COLORS = {"A": "#1baf7a", "B": "#eb6834", "C": "#2a78d6"}
EXTRA_COLORS = ["#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def display_name(label):
    """'A' -> 'Defaults', 'B' -> 'Random search', 'C' -> 'PSO'; other labels show as they are."""
    return ARMS[label]["name"] if label in ARMS else label


def arm_title(label):
    return f"{label}: {display_name(label)}" if label in ARMS else label


def model_colors(labels):
    extra = iter(EXTRA_COLORS * 4)
    return {label: MODEL_COLORS.get(label) or next(extra) for label in labels}


def rel(path):
    """Path relative to the repo when possible, for readable logs."""
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


# ---------------------------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------------------------
def parse_model_arg(text):
    """'LABEL=path' -> (label, Path). Used as the argparse type of --models."""
    label, sep, path = text.partition("=")
    if not sep or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError(f"expected LABEL=path (e.g. A=results/runs/baseline/weights/best.pt), "
                                         f"got '{text}'")
    return label.strip(), Path(path.strip())


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", action="extend", type=parse_model_arg, metavar="LABEL=path",
                    help="models to compare (default: A, B and C at their results/runs/.../best.pt)")
    ap.add_argument("--split", choices=("test", "val"), default="test")
    ap.add_argument("--data", default=str(REPO / "configs" / "data.yaml"))
    ap.add_argument("--out-dir", default=str(REPO / "results"), help="plots go to <out-dir>/plots/")
    ap.add_argument("--device", help="e.g. 0 or cpu (default: Ultralytics picks)")
    ap.add_argument("--bootstrap", type=int, default=1000, help="number of paired bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0, help="seed of the bootstrap and of the image choices")
    ap.add_argument("--skip-checks", action="store_true", help="skip the checkpoint config check")
    args = ap.parse_args(argv)
    pairs = args.models or [(label, REPO / arm["model"]) for label, arm in ARMS.items()]
    labels = [label for label, _ in pairs]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        ap.error(f"--models: label(s) {duplicates} given more than once")
    if args.bootstrap < 1:
        ap.error("--bootstrap must be at least 1")
    args.models = dict(pairs)
    return args


# ---------------------------------------------------------------------------------------------------------------
# Checkpoint checks
# ---------------------------------------------------------------------------------------------------------------
def missing_model_message(label, path):
    """What to download and where to put it, for a model file that does not exist."""
    msg = f"Model {label} not found: {rel(path)}"
    arm = ARMS.get(label)
    if arm is None:
        return msg + " (check the path given with --models)."
    msg += (f"\n  Download {arm['download']} from the Output tab of the finished Kaggle version of "
            f"{arm['notebook']} and put it at {arm['model']}.")
    if arm["download"].endswith(".pt"):
        msg += (f"\n  The .pt file downloads as a .zip: rename it to best.pt; do not unzip it "
                f"(a torch checkpoint is itself a zip archive).")
    return msg


def require_model_files(models):
    missing = [missing_model_message(label, path) for label, path in models.items() if not Path(path).is_file()]
    if missing:
        raise SystemExit("\n".join(missing))


def sha256_12(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def load_checkpoint_info(path):
    """Class names and train_args that Ultralytics stores inside a .pt checkpoint."""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)  # our own checkpoints; they hold a pickled model
    model = ckpt.get("ema") or ckpt.get("model")
    names = getattr(model, "names", None) or {}
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    return {"names": names, "train_args": dict(ckpt.get("train_args") or {})}


def data_class_names(data_yaml):
    names = yaml.safe_load(Path(data_yaml).read_text())["names"]
    return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)


def expected_params(label):
    """The 6 hyperparameters arm A/B/C must have been trained with, or None for other labels."""
    if label not in ARMS:
        return None
    cfg = json.loads((REPO / ARMS[label]["config"]).read_text())
    return {k: float(cfg["params"][k]) for k in NAMES}


def check_checkpoint(info, class_names, expected=None):
    """List of problems with one checkpoint (empty = fine). expected: the arm's 6 hyperparameters, or None."""
    problems = []
    if info["names"] != list(class_names):
        problems.append(f"it has {len(info['names'])} classes {info['names'][:8]}{' ...' * (len(info['names']) > 8)}"
                        f", but data.yaml has {len(class_names)}: {list(class_names)}")
    train_args = info["train_args"]
    for key, want in (("epochs", EPOCHS), ("imgsz", IMGSZ)):
        if train_args.get(key) != want:
            problems.append(f"it was trained with {key}={train_args.get(key)}, expected {want}")
    if expected is not None:
        for key in NAMES:
            got = train_args.get(key)
            if got is None or not math.isclose(float(got), expected[key], rel_tol=REL_TOL):
                problems.append(f"{key}={got}, but the arm's config has {key}={expected[key]}")
    return problems


def run_checks(models, infos, class_names):
    report = []
    for label, path in models.items():
        problems = check_checkpoint(infos[label], class_names, expected_params(label))
        if problems:
            source = f" ({ARMS[label]['config']})" if label in ARMS else ""
            report.append(f"Model {label} ({rel(path)}) failed the checkpoint check{source}:\n    - "
                          + "\n    - ".join(problems))
    if report:
        raise SystemExit("\n".join(report) + "\nIs the right best.pt in the right slot? (e.g. yolov8n.pt or another "
                         "arm's model in its place). Use --skip-checks only for deliberate experiments.")


# ---------------------------------------------------------------------------------------------------------------
# Validation with Ultralytics' own validator, keeping each image's statistics
# ---------------------------------------------------------------------------------------------------------------
class RecordingDetMetrics(DetMetrics):
    """Ultralytics' DetMetrics that also keeps every image's statistics.

    Ultralytics passes one image at a time to update_stats() and only keeps the concatenation, which loses the image
    each prediction came from. The bootstrap needs that, so each image's dict is also kept here, unchanged.
    """

    def __init__(self, names=None):
        super().__init__(names)
        self.per_image = []

    def update_stats(self, stat):
        super().update_stats(stat)
        self.per_image.append({"image": stat["im_name"], "tp": np.asarray(stat["tp"], dtype=bool),
                               "conf": np.asarray(stat["conf"]), "pred_cls": np.asarray(stat["pred_cls"]),
                               "target_cls": np.asarray(stat["target_cls"])})


class RecordingValidator(DetectionValidator):
    """Ultralytics' DetectionValidator; the only change is the recording metrics object above."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = RecordingDetMetrics()


def validate(path, data, split, max_det, device, run_dir):
    """Validate one checkpoint like the trainer's final validation of best.pt; returns the finished validator."""
    args = dict(model=str(path), data=str(data), split=split, imgsz=IMGSZ, batch=VAL_BATCH, rect=True, iou=0.7,
                max_det=max_det, plots=True, project=str(run_dir.parent), name=run_dir.name, exist_ok=True,
                mode="val", task="detect")
    # conf stays unset, as in training: metrics then use conf 0.001 and the confusion matrix conf 0.25
    if device is not None:
        args["device"] = device
    validator = RecordingValidator(args=args)
    validator(model=str(path))
    return validator


def inference_precision(args):
    """'fp16' or 'fp32', from a finished validator's args (Ultralytics records the precision it actually used)."""
    half = getattr(args, "quantize", None) in (16, "16", "fp16") or bool(getattr(args, "half", False))
    return "fp16" if half else "fp32"


def device_label(device):
    """GPU name, or the CPU name (Ultralytics' readable name; platform.processor() as a fallback)."""
    import torch
    from ultralytics.utils.torch_utils import get_cpu_info

    if device.type == "cuda":
        return torch.cuda.get_device_name(device.index or 0)
    return f"CPU {get_cpu_info() or platform.processor() or platform.machine()}"


def summarize_validator(validator):
    """Metrics, per-class APs and the normalized confusion matrix from a finished validator."""
    box = validator.metrics.box
    names = [validator.names[i] for i in range(len(validator.names))]
    per_class = {name: {"ap50": None, "ap50_95": None} for name in names}
    for i, c in enumerate(box.ap_class_index):
        per_class[names[int(c)]] = {"ap50": round(float(box.ap50[i]), 5), "ap50_95": round(float(box.ap[i]), 5)}
    matrix = validator.confusion_matrix.matrix  # rows = predicted, columns = true; last row/column = background
    normalized = matrix / (matrix.sum(0, keepdims=True) + 1e-9)  # as Ultralytics' plot: each true class sums to 1
    return {
        "metrics": {"map50": round(float(box.map50), 5), "map50_95": round(float(box.map), 5),
                    "precision": round(float(box.mp), 5), "recall": round(float(box.mr), 5)},
        "per_class": per_class,
        "confusion_matrix": {"labels": [*names, "background"], "rows": "predicted", "columns": "true",
                             "counts": matrix.astype(int).tolist(),
                             "normalized": np.round(normalized, 4).tolist()},
    }


# ---------------------------------------------------------------------------------------------------------------
# mAP from per-image statistics, consistency check and paired bootstrap (pure functions, unit-tested)
# ---------------------------------------------------------------------------------------------------------------
def stack_stats(per_image):
    """Concatenate per-image stats into flat arrays plus each image's start offset (for fast resampling)."""
    n_pred = [len(s["conf"]) for s in per_image]
    n_target = [len(s["target_cls"]) for s in per_image]
    return {
        "tp": np.concatenate([np.asarray(s["tp"], dtype=bool).reshape(-1, 10) for s in per_image]),
        "conf": np.concatenate([np.asarray(s["conf"], dtype=float) for s in per_image]),
        "pred_cls": np.concatenate([np.asarray(s["pred_cls"], dtype=float) for s in per_image]),
        "target_cls": np.concatenate([np.asarray(s["target_cls"], dtype=float) for s in per_image]),
        "pred_start": np.concatenate(([0], np.cumsum(n_pred))).astype(int),
        "target_start": np.concatenate(([0], np.cumsum(n_target))).astype(int),
    }


def block_indices(starts, idx):
    """Row indices of the images in idx (repeats allowed), image after image, given each image's start offset."""
    idx = np.asarray(idx, dtype=int)
    counts = starts[idx + 1] - starts[idx]
    offsets = np.repeat(starts[idx] - np.cumsum(counts) + counts, counts)
    return offsets + np.arange(counts.sum())


def map_from_stats(flat, idx=None):
    """(mAP@50, mAP@50-95) of the images idx (all images if None), computed by Ultralytics' ap_per_class."""
    if idx is None:
        tp, conf, pred_cls, target_cls = flat["tp"], flat["conf"], flat["pred_cls"], flat["target_cls"]
    else:
        p, t = block_indices(flat["pred_start"], idx), block_indices(flat["target_start"], idx)
        tp, conf, pred_cls, target_cls = flat["tp"][p], flat["conf"][p], flat["pred_cls"][p], flat["target_cls"][t]
    ap = ap_per_class(tp, conf, pred_cls, target_cls)[5]  # (classes present, 10 IoU thresholds)
    if not len(ap):
        return 0.0, 0.0
    return float(ap[:, 0].mean()), float(ap.mean())  # as Ultralytics' Metric.map50 and Metric.map


def consistency_check(flat, map50, map50_95, tol=CONSISTENCY_TOL):
    """Recompute both mAPs from the recorded per-image stats; they must equal Ultralytics' numbers."""
    m50, m = map_from_stats(flat)
    diff = max(abs(m50 - map50), abs(m - map50_95))
    return {"map50_ultralytics": map50, "map50_recomputed": m50, "map50_95_ultralytics": map50_95,
            "map50_95_recomputed": m, "max_abs_diff": diff, "tolerance": tol, "passed": bool(diff <= tol)}


def bootstrap_maps(flats, n_resamples, seed, progress=False):
    """Paired bootstrap: each resample draws image indices with replacement, the SAME ones for every model.

    Returns {label: array (n_resamples, 2)} of (mAP@50, mAP@50-95).
    """
    sizes = {len(f["pred_start"]) - 1 for f in flats.values()}
    if len(sizes) != 1:
        raise ValueError(f"models were scored on different numbers of images: {sizes}")
    n_images = sizes.pop()
    rng = np.random.default_rng(seed)
    out = {label: np.empty((n_resamples, 2)) for label in flats}
    start = time.time()
    for b in range(n_resamples):
        idx = rng.integers(0, n_images, size=n_images)
        for label, flat in flats.items():
            out[label][b] = map_from_stats(flat, idx)
        if progress and ((b + 1) % 100 == 0 or b + 1 == n_resamples):
            print(f"  bootstrap {b + 1}/{n_resamples} ({time.time() - start:.0f} s)", flush=True)
    return out


def percentile_ci(values, level=95):
    lo, hi = np.percentile(values, [(100 - level) / 2, 100 - (100 - level) / 2])
    return [float(lo), float(hi)]


METRICS = (("map50", "mAP@50"), ("map50_95", "mAP@50-95"))


def pairwise_differences(samples, observed):
    """For every pair (a, b) in the given order: b - a per metric, with its 95% CI and P(b - a > 0).

    samples: {label: (n, 2) bootstrap array}; observed: {label: {"map50": .., "map50_95": ..}} on all images.
    """
    pairs = []
    for a, b in combinations(list(samples), 2):
        row = {"pair": f"{b}-{a}", "a": a, "b": b}
        for k, (key, _) in enumerate(METRICS):
            d = samples[b][:, k] - samples[a][:, k]
            lo, hi = percentile_ci(d)
            row[key] = {"observed_diff": round(observed[b][key] - observed[a][key], 5),
                        "mean_diff": float(d.mean()), "ci95": [lo, hi], "p_gt_0": float((d > 0).mean()),
                        "significant": not (lo <= 0 <= hi)}
        row["verdict"] = verdict(row)
        pairs.append(row)
    return pairs


def verdict(row):
    """Plain-English verdict for one pair; a CI that contains 0 means 'not significant'."""
    a, b = display_name(row["a"]), display_name(row["b"])
    parts = []
    for key, name in METRICS:
        r = row[key]
        if r["significant"]:
            better = b if r["ci95"][0] > 0 else a
            parts.append(f"{better} is significantly better on {name}")
        else:
            parts.append(f"{name}: not significant")
    if not row["map50"]["significant"] and not row["map50_95"]["significant"]:
        return f"Not significant: both 95% CIs contain 0, so this split cannot tell {b} and {a} apart."
    return "; ".join(parts) + "."


# ---------------------------------------------------------------------------------------------------------------
# Real-time speed and image choices
# ---------------------------------------------------------------------------------------------------------------
def choose(items, k, seed):
    """k items chosen deterministically (seeded) from a sorted copy of items."""
    items = sorted(items)
    if len(items) <= k:
        return items
    picks = np.random.default_rng(seed).choice(len(items), size=k, replace=False)
    return [items[i] for i in sorted(picks)]


def measure_latency(model, image_paths, device, run_dir):
    """Median batch-1 latency (preprocess + inference + NMS) on in-memory images, after warm-up predictions."""
    import cv2
    import torch

    images = [cv2.imread(str(p)) for p in image_paths]
    kwargs = dict(imgsz=IMGSZ, conf=SAMPLE_CONF, verbose=False, save=False, project=str(run_dir.parent),
                  name=run_dir.name, exist_ok=True)
    if device is not None:
        kwargs["device"] = device
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    for im in images[:LATENCY_WARMUP]:
        model.predict(im, **kwargs)
    times = []
    for im in images:
        sync()
        t0 = time.perf_counter()
        model.predict(im, **kwargs)
        sync()
        times.append(time.perf_counter() - t0)
    ms = float(np.median(times) * 1000)
    return {"ms_per_img": round(ms, 2), "fps": round(1000 / ms, 1), "images": len(images),
            "warmup": LATENCY_WARMUP, "batch": 1, "imgsz": IMGSZ, "conf": SAMPLE_CONF}


def choose_samples(image_classes, class_names, seed):
    """One image per class (in class order), drawn with the seed from the images whose labels contain it."""
    rng = np.random.default_rng(seed)
    chosen = []
    for c, name in enumerate(class_names):
        taken = {path for _, path in chosen}
        candidates = sorted(p for p, classes in image_classes.items() if c in classes and p not in taken)
        if not candidates:
            print(f"Note: no {name} image in this split; the sample grid skips that class")
            continue
        chosen.append((name, candidates[int(rng.integers(len(candidates)))]))
    return chosen


def read_ground_truth(image_path, width, height):
    """YOLO label file of an image -> (classes, xyxy pixel boxes)."""
    from ultralytics.data.utils import img2label_paths

    label = Path(img2label_paths([str(image_path)])[0])
    rows = [line.split() for line in label.read_text().splitlines() if line.strip()] if label.exists() else []
    cls = np.array([int(float(r[0])) for r in rows], dtype=int)
    xywh = np.array([[float(v) for v in r[1:5]] for r in rows]).reshape(-1, 4) * [width, height, width, height]
    xyxy = np.concatenate([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], axis=1)
    return cls, xyxy


def predict_boxes(model, image_path, device):
    kwargs = dict(imgsz=IMGSZ, conf=SAMPLE_CONF, verbose=False, save=False)
    if device is not None:
        kwargs["device"] = device
    boxes = model.predict(str(image_path), **kwargs)[0].boxes
    return boxes.cls.cpu().numpy().astype(int), boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()


# ---------------------------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------------------------
def save(fig, out, **kwargs):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, **kwargs)
    plt.close(fig)


def plot_confusion(summaries, split, out):
    """Normalized confusion matrices side by side (columns = true class, each column sums to 1)."""
    labels = list(summaries)
    fig, axes = plt.subplots(1, len(labels), figsize=(3.9 * len(labels) + 1.6, 5.8), squeeze=False,
                             layout="constrained")
    for k, (ax, label) in enumerate(zip(axes[0], labels)):
        cm = summaries[label]["confusion_matrix"]
        array, ticks = np.array(cm["normalized"]), cm["labels"]
        im = ax.imshow(array, cmap="Blues", vmin=0, vmax=1)
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array[i, j] >= 0.005:
                    ax.text(j, i, f"{array[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if array[i, j] > 0.45 else "black")
        ax.set_xticks(range(len(ticks)), ticks, rotation=90, fontsize=7)
        ax.set_yticks(range(len(ticks)), ticks if k == 0 else [], fontsize=7)  # row names on the first panel only
        ax.set_xlabel("True", fontsize=8)
        if k == 0:
            ax.set_ylabel("Predicted", fontsize=8)
        ax.set_title(arm_title(label), fontsize=10)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.75)
    fig.suptitle(f"Normalized confusion matrices, {split} split (boxes at conf >= 0.25, IoU 0.45)", fontsize=11)
    save(fig, out, dpi=160, bbox_inches="tight")


def plot_per_class(summaries, split, out):
    """Grouped bar chart of per-class AP@50, one bar colour per model."""
    labels = list(summaries)
    names = list(next(iter(summaries.values()))["per_class"])
    colors = model_colors(labels)
    width = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(max(7.5, 1.2 * len(names) + 0.5 * len(labels)), 4.6))
    x = np.arange(len(names))
    for k, label in enumerate(labels):
        values = [summaries[label]["per_class"][n]["ap50"] or 0.0 for n in names]
        pos = x - 0.4 + width * (k + 0.5)
        ax.bar(pos, values, width, color=colors[label], edgecolor="white", linewidth=1, label=arm_title(label))
        for xi, v in zip(pos, values):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6, color="#333333")
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(f"AP@50 ({split} split)")
    ax.set_title(f"Per-class AP@50 on the {split} split")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), fontsize=8, ncol=len(labels), frameon=False)
    fig.tight_layout()
    save(fig, out, dpi=160)


def draw_boxes(ax, cls, xyxy, class_names, conf=None):
    for i, (c, (x1, y1, x2, y2)) in enumerate(zip(cls, xyxy)):
        color = CLASS_COLORS[int(c) % len(CLASS_COLORS)]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.2))
        name = class_names[int(c)] if int(c) < len(class_names) else str(int(c))
        text = name if conf is None else f"{name} {conf[i]:.2f}"
        inside = y1 < 0.04 * max(ax.get_ylim())  # a box touching the top edge gets its label inside it
        ax.text(x1, y1, text, fontsize=4.5, color="white", va="top" if inside else "bottom", ha="left", clip_on=True,
                bbox=dict(facecolor=color, edgecolor="none", pad=0.6, alpha=0.9))


def plot_samples(samples, columns, class_names, split, out):
    """Rows = sample images, columns = ground truth then one per model. columns: [(title, {path: boxes})]."""
    from PIL import Image

    n_rows, n_cols = len(samples), len(columns) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows + 0.4), squeeze=False)
    for r, (cls_name, path) in enumerate(samples):
        img = np.asarray(Image.open(path).convert("RGB"))
        h, w = img.shape[:2]
        panels = [("Ground truth", (*read_ground_truth(path, w, h), None))]
        panels += [(title, preds[path]) for title, preds in columns]
        for c, (title, (cls, xyxy, conf)) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img)
            draw_boxes(ax, cls, xyxy, class_names, conf)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{cls_name} sample", fontsize=8)
    fig.suptitle(f"{split} images, one per class (predictions at conf >= {SAMPLE_CONF})", fontsize=10)
    fig.tight_layout()
    save(fig, out, dpi=120, pil_kwargs={"quality": 85})


def plot_training_curves(sources, out):
    """Val mAP@50 per epoch of each arm's full training, from Ultralytics' results.csv files.

    sources: {label: csv path}. Missing files are skipped with a note. Returns the labels that were drawn.
    """
    import pandas as pd

    curves = {}
    for label, path in sources.items():
        if not Path(path).is_file():
            print(f"Note: {rel(path)} not found; arm {label} is left out of {Path(out).name}")
            continue
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        curves[label] = (df["epoch"].to_numpy(), df["metrics/mAP50(B)"].to_numpy())
    if not curves:
        print(f"Note: no training curves found; {Path(out).name} not written")
        return []
    colors = model_colors(list(curves))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for label, (epoch, map50) in curves.items():
        best = int(np.argmax(map50))
        ax.plot(epoch, map50, color=colors[label], linewidth=2,
                label=f"{arm_title(label)} (best {map50[best]:.3f} at epoch {int(epoch[best])})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val mAP@50")
    ax.set_title("Validation mAP@50 per epoch of the full trainings")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, out, dpi=160)
    return list(curves)


# ---------------------------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------------------------
def fmt_ci(value, ci, signed=False):
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(value)} [{f.format(ci[0])}, {f.format(ci[1])}]"


def comparison_markdown(report):
    """Report-ready markdown: main table, pairwise differences, per-class AP@50."""
    split, models = report["split"], report["models"]
    n_boot = report["settings"]["bootstrap_resamples"]
    lines = [f"## Final comparison on the {split} split",
             "",
             f"{report['images']} images, {report['boxes']} boxes. 95% CIs: percentile, {n_boot} paired bootstrap "
             f"resamples of the images (seed {report['settings']['seed']}). P and R at Ultralytics' max-F1 "
             f"confidence. ms/img: median batch-1 latency at imgsz {IMGSZ} on {report['device']}.",
             "",
             f"| Arm | val mAP@50 (training) | {split} mAP@50 [95% CI] | {split} mAP@50-95 [95% CI] | P | R | ms/img |",
             "|---|---|---|---|---|---|---|"]
    for label, m in models.items():
        val = report["val_metrics"].get(label)
        val_map50 = f"{val['map50']:.3f}" if val else "-"
        boot = m["bootstrap"]
        lines.append(f"| {arm_title(label)} | {val_map50} "
                     f"| {fmt_ci(m['metrics']['map50'], boot['map50_ci95'])} "
                     f"| {fmt_ci(m['metrics']['map50_95'], boot['map50_95_ci95'])} "
                     f"| {m['metrics']['precision']:.3f} | {m['metrics']['recall']:.3f} "
                     f"| {m['speed']['ms_per_img']:.1f} |")
    if report["pairwise"]:
        lines += ["", "### Pairwise differences (paired bootstrap)", "",
                  "| Pair | Δ mAP@50 [95% CI] | P(Δ>0) | Δ mAP@50-95 [95% CI] | P(Δ>0) | Verdict |",
                  "|---|---|---|---|---|---|"]
        for p in report["pairwise"]:
            pair = f"{display_name(p['b'])} − {display_name(p['a'])} ({p['b']}−{p['a']})"
            cells = [f"{fmt_ci(p[k]['observed_diff'], p[k]['ci95'], signed=True)} | {p[k]['p_gt_0']:.2f}"
                     for k, _ in METRICS]
            lines.append(f"| {pair} | {cells[0]} | {cells[1]} | {p['verdict']} |")
        lines += ["", "Δ is the difference of the two arms' mAPs on the whole split (as in the main table); the 95% "
                      "CI and P(Δ>0) come from the paired bootstrap (P(Δ>0): share of resamples in which the first "
                      "arm of the pair scored higher). A difference is significant only if its CI excludes 0."]
    else:
        lines += ["", "(Pairwise differences need at least 2 models.)"]
    names = list(next(iter(models.values()))["per_class"])
    lines += ["", f"### Per-class AP@50 ({split} split)", "",
              "| Class | " + " | ".join(arm_title(label) for label in models) + " |",
              "|---|" + "---|" * len(models)]
    for name in names:
        cells = [models[label]["per_class"].get(name, {}).get("ap50") for label in models]
        lines.append(f"| {name} | " + " | ".join("-" if v is None else f"{v:.3f}" for v in cells) + " |")
    return "\n".join(lines) + "\n"


def load_val_metrics(labels):
    """Each arm's val metrics from its training JSON (results/baseline.json, arm_b.json, arm_c.json), if present."""
    out = {}
    for label in labels:
        path = REPO / ARMS[label]["val_json"] if label in ARMS else None
        if path is not None and path.is_file():
            d = json.loads(path.read_text())
            out[label] = {k: d.get(k) for k in ("map50", "map50_95", "precision", "recall")}
            out[label]["source"] = ARMS[label]["val_json"]
    return out


# ---------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------
def score_model(label, path, info, args, run_dir):
    """Validate one model, check its recorded per-image stats against Ultralytics' mAP, and summarize it."""
    max_det = int(info["train_args"].get("max_det", 300))
    print(f"\n[{label}] validating {rel(path)} (max_det {max_det}, as in its training)")
    validator = validate(path, args.data, args.split, max_det, args.device, run_dir)
    check = consistency_check(stack_stats(validator.metrics.per_image), float(validator.metrics.box.map50),
                              float(validator.metrics.box.map))  # unrounded Ultralytics values
    print(f"[{label}] consistency check: mAP@50 {check['map50_ultralytics']:.6f} (Ultralytics) vs "
          f"{check['map50_recomputed']:.6f} (per-image stats), mAP@50-95 {check['map50_95_ultralytics']:.6f} vs "
          f"{check['map50_95_recomputed']:.6f}; max |diff| {check['max_abs_diff']:.2e} "
          f"-> {'PASS' if check['passed'] else 'FAIL'}")
    if not check["passed"]:
        raise SystemExit(f"Consistency check failed for {label}: the recorded per-image statistics do not "
                         f"reproduce Ultralytics' mAP, so the bootstrap would be wrong.")
    result = {"name": display_name(label), "path": rel(path), "sha256_12": sha256_12(path), "max_det": max_det,
              "conf": float(validator.args.conf), "run_dir": rel(run_dir), **summarize_validator(validator),
              "consistency_check": check}
    return result, validator


def pair_images(per_image_by_model):
    """Put every model's per-image stats in the same image order (the first model's) and stack them."""
    order = [s["image"] for s in next(iter(per_image_by_model.values()))]
    flats = {}
    for label, per_image in per_image_by_model.items():
        by_name = {s["image"]: s for s in per_image}
        if len(by_name) != len(per_image) or sorted(by_name) != sorted(order):
            raise SystemExit(f"model {label} was not scored on the same uniquely named images as the first model")
        flats[label] = stack_stats([by_name[name] for name in order])
    return order, flats


def main(argv=None):
    import ultralytics
    from ultralytics import YOLO

    args = parse_args(argv)
    start = time.time()
    split, models = args.split, args.models
    out_dir = Path(args.out_dir)
    plots = out_dir / "plots"
    class_names = data_class_names(args.data)
    print(f"Evaluating {', '.join(f'{label}={rel(p)}' for label, p in models.items())} on the {split} split")
    if split != "test" and out_dir.resolve() == (REPO / "results").resolve():
        print(f"Warning: a --split {split} trial is writing into results/; pass a scratch --out-dir to keep "
              f"results/ for the final test run.")

    # 1. checkpoint files and configs, before any long work
    require_model_files(models)
    infos = {label: load_checkpoint_info(path) for label, path in models.items()}
    if args.skip_checks:
        print("Checkpoint config check skipped (--skip-checks)")
    else:
        run_checks(models, infos, class_names)
        print("Checkpoint check passed: 6 classes, 100 epochs, imgsz 416 and each arm's hyperparameters")

    # 2. metrics, with every image's statistics
    run_dirs = {label: (out_dir / "runs" / f"eval_{split}_{label}").resolve() for label in models}
    results, per_image, validators = {}, {}, {}
    for label, path in models.items():
        results[label], validators[label] = score_model(label, path, infos[label], args, run_dirs[label])
        per_image[label] = validators[label].metrics.per_image
    image_order, flats = pair_images(per_image)
    first = validators[next(iter(models))]
    eval_device = device_label(first.device)
    full_path = {Path(f).name: f for f in first.dataloader.dataset.im_files}
    image_classes = {full_path[s["image"]]: set(s["target_cls"].astype(int).tolist()) for s in first.metrics.per_image}

    # 3. real-time speed, on the same images for every model
    yolo = {label: YOLO(str(path)) for label, path in models.items()}
    latency_images = choose([full_path[name] for name in image_order], LATENCY_IMAGES, args.seed)
    for label in models:
        results[label]["speed"] = {**measure_latency(yolo[label], latency_images, args.device, run_dirs[label]),
                                   "device": eval_device}
        print(f"[{label}] {results[label]['metrics']}  latency {results[label]['speed']['ms_per_img']} ms/img "
              f"({results[label]['speed']['fps']} FPS) on {eval_device}")

    # 4. paired bootstrap
    print(f"\nPaired bootstrap: {args.bootstrap} resamples of {len(image_order)} images, seed {args.seed}")
    samples = bootstrap_maps(flats, args.bootstrap, args.seed, progress=True)
    for label, s in samples.items():
        results[label]["bootstrap"] = {"map50_ci95": percentile_ci(s[:, 0]), "map50_95_ci95": percentile_ci(s[:, 1]),
                                       "map50_mean": float(s[:, 0].mean()), "map50_95_mean": float(s[:, 1].mean())}
    pairwise = pairwise_differences(samples, {label: r["metrics"] for label, r in results.items()}) \
        if len(models) >= 2 else []

    # 5. figures
    plot_confusion(results, split, plots / f"{split}_confusion.png")
    plot_per_class(results, split, plots / f"{split}_per_class.png")
    samples_grid = choose_samples(image_classes, class_names, args.seed)
    columns = [(arm_title(label), {p: predict_boxes(yolo[label], p, args.device) for _, p in samples_grid})
               for label in models]
    plot_samples(samples_grid, columns, class_names, split, plots / f"{split}_samples.jpg")
    curves_name = "training_curves.png" if split == "test" else f"{split}_training_curves.png"  # trials keep apart
    plot_training_curves({label: REPO / arm["epochs_csv"] for label, arm in ARMS.items()}, plots / curves_name)

    # 6. JSON and markdown
    report = {
        "split": split,
        "data": rel(args.data),
        "images": len(image_order),
        "boxes": int(flats[next(iter(models))]["target_cls"].size),
        "boxes_per_class": dict(zip(class_names, first.metrics.nt_per_class.astype(int).tolist())),
        "settings": {"imgsz": IMGSZ, "batch": VAL_BATCH, "rect": True, "conf": float(first.args.conf), "iou": 0.7,
                     "max_det": {label: r["max_det"] for label, r in results.items()},
                     "precision": {label: inference_precision(v.args) for label, v in validators.items()},
                     "confusion_matrix": "conf 0.25, IoU 0.45 (Ultralytics' validator default)",
                     "latency": {"images": len(latency_images), "warmup": LATENCY_WARMUP, "batch": 1,
                                 "conf": SAMPLE_CONF},
                     "bootstrap_resamples": args.bootstrap, "seed": args.seed, "ci": "95% percentile",
                     "samples_conf": SAMPLE_CONF, "checks_skipped": args.skip_checks},
        "models": results,
        "val_metrics": load_val_metrics(list(ARMS)),
        "pairwise": pairwise,
        "ultralytics": ultralytics.__version__,
        "device": eval_device,
        "processor": platform.processor(),
        "seconds": round(time.time() - start, 1),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    markdown = comparison_markdown(report)
    out_json, out_md = out_dir / f"{split}_results.json", out_dir / f"{split}_comparison.md"
    write_json(out_json, report)
    out_md.write_text(markdown, encoding="utf-8")
    print("\n" + markdown)
    print(f"Wrote {rel(out_json)}, {rel(out_md)} and {rel(plots)}/ in {(time.time() - start) / 60:.1f} min")
    return report


if __name__ == "__main__":
    main()
