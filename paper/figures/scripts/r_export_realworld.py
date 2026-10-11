"""Export the real-world sample bundle for the Results figures (run on Kaggle by
paper/figures/kaggle/paper_realworld_samples.ipynb; the real-world test photos exist only there).

    python paper/figures/scripts/r_export_realworld.py --data <cfg>/data_test_real_world.yaml \
        --weights app/models/gen2/arm_a.pt app/models/gen2/arm_b.pt app/models/gen2/arm_c.pt \
        --out /kaggle/working/realworld_samples --expect 468 779

Every test image listed by --data is predicted by the three Gen 2 models (A: defaults, B: random search, C: PSO)
with the settings of r_samples.py (confidence 0.25, NMS IoU 0.7, imgsz 640, CPU by default). Each model's boxes are
matched to the ground truth greedily: predictions in order of confidence (highest first), each to the unmatched
ground-truth box of the SAME class with the highest IoU, a true positive if that IoU is at least 0.5. Unmatched
predictions are false positives, unmatched ground-truth boxes false negatives. A class-agnostic count (the same
matching, any class) separates "found but mislabeled" from "missed". An image is FULLY CORRECT for a model when it
has at least one ground-truth box, TP == n_gt and FP == 0. Per-image F1 = 2 TP / (2 TP + FP + FN).

Two fixed selections (no image is chosen by eye):
  A  random rows, the rule of r_samples.py (its choose() is called): images with >= 3 ground-truth boxes, sorted by
     file name, put in a random order with random.Random(0).sample, first 2. This is the real-world row of the
     sample figure.
  B  best cases, deliberately favorable: images with >= 3 ground-truth boxes that are fully correct for all three
     models. If fewer than 3 qualify, the rule is relaxed to >= 2 boxes, then to fully correct for at least 2 of the
     3 models. The candidates of the level used are sorted by file name, put in the seed-0 order as in A, then
     stably sorted by n_gt (most first); the first 3 are taken. The level used and the candidates at every level
     are recorded, with each model's totals over the whole test set, so a caption can state how favorable it is.

Output folder (--out, recreated):
  per_image_stats.json  every image: source, n_gt and per model n_pred, TP, FP, FN, class-agnostic TP, fully
                        correct, F1; per-model totals (fully correct images, mean per-image F1).
  selection.json        the settings, rules, level used, totals, and for each selected image its source, statistics
                        and the predicted boxes (normalized x0 y0 x1 y1, class, confidence), so the local figure
                        draws exactly the boxes that were scored.
  images/, labels/      the selected images (A and B) and their YOLO labels. An image whose long side is above
                        --max-side (1280) is downscaled to it (JPEG quality 92) and marked in selection.json; the
                        labels and boxes are normalized, so they still fit.
and a zip of the folder's contents next to it (realworld_samples.zip), to unzip into paper/figures/data/realworld/.
"""
import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r_samples as rs  # noqa: E402  (also puts the repo's src/ on the path)
import build_dataset as bd  # noqa: E402
from verify_dataset import IMG_EXTS  # noqa: E402

IOU_MIN = 0.5
N_BEST = 3
LEVELS = [  # selection B: tried in order until at least N_BEST images qualify
    {"level": 1, "min_gt": 3, "min_models_fully_correct": 3},
    {"level": 2, "min_gt": 2, "min_models_fully_correct": 3},
    {"level": 3, "min_gt": 2, "min_models_fully_correct": 2},
]
OUT_NAMES = {"images", "labels", "selection.json", "per_image_stats.json"}
JPEG_QUALITY = 92
ZIP_NAME = "realworld_samples.zip"


# ---------------------------------------------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------------------------------------------
def iou(a, b):
    """IoU of two boxes (x0, y0, x1, y1). Normalized coordinates give the same IoU as pixels."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def greedy_tp(gt, preds, same_class=True, iou_min=IOU_MIN):
    """Number of true positives. gt: [(cls, x0, y0, x1, y1)]; preds: [(cls, x0, y0, x1, y1, conf)]. Predictions are
    taken by confidence, highest first (ties: input order); each is matched to the unmatched ground-truth box (of the
    same class if same_class) with the highest IoU (ties: the first one), and counts if that IoU >= iou_min."""
    used = [False] * len(gt)
    tp = 0
    for pi in sorted(range(len(preds)), key=lambda i: -preds[i][5]):
        p = preds[pi]
        best, best_iou = None, -1.0
        for gi, g in enumerate(gt):
            if used[gi] or (same_class and g[0] != p[0]):
                continue
            v = iou(g[1:5], p[1:5])
            if v > best_iou:
                best, best_iou = gi, v
        if best is not None and best_iou >= iou_min:
            used[best] = True
            tp += 1
    return tp


def image_stats(gt, preds):
    tp = greedy_tp(gt, preds)
    fp, fn = len(preds) - tp, len(gt) - tp
    denom = 2 * tp + fp + fn
    return {"n_gt": len(gt), "n_pred": len(preds), "tp": tp, "fp": fp, "fn": fn,
            "tp_any_class": greedy_tp(gt, preds, same_class=False),
            "fully_correct": len(gt) >= 1 and tp == len(gt) and fp == 0,
            "f1": round(2 * tp / denom, 6) if denom else None}


# ---------------------------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------------------------
def test_images(data_yaml, split):
    """The image paths of `split` in a YOLO data yaml: a .txt list (relative lines from its folder), a folder of
    images, or a list of either; relative entries are taken from the yaml's 'path'."""
    data_yaml = Path(data_yaml)
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    if split not in d:
        raise SystemExit(f"{data_yaml} has no '{split}' entry")
    root = Path(d.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    paths = []
    for entry in d[split] if isinstance(d[split], list) else [d[split]]:
        e = Path(entry) if Path(entry).is_absolute() else root / entry
        if e.is_dir():
            paths += sorted(q for q in e.rglob("*") if q.suffix.lower() in IMG_EXTS)
        elif e.suffix == ".txt" and e.is_file():
            for line in e.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    q = Path(line.strip())
                    paths.append(q if q.is_absolute() else (e.parent / q).resolve())
        else:
            raise SystemExit(f"{data_yaml}: '{split}' entry {e} is neither a folder nor a .txt list")
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} listed test images do not exist, e.g. {missing[:3]}")
    dup = [n for n, k in Counter(p.name for p in paths).items() if k > 1]
    if dup:
        raise SystemExit(f"test image file names must be unique (they are copied into one folder): {dup[:5]}")
    return paths


def sources_by_file(split_csv, paths):
    """{file name: (source, how it was found)}: the 'source' column of the build's split.csv, else the file-name
    prefix the build writes (<source>__<name>.jpg), else unknown."""
    table = {}
    if split_csv and Path(split_csv).is_file():
        with open(split_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("file") and r.get("source"):
                    table[r["file"]] = r["source"]
    out = {}
    for p in paths:
        if p.name in table:
            out[p.name] = (table[p.name], "split.csv")
        elif "__" in p.name:
            out[p.name] = (p.name.split("__", 1)[0], "file name prefix")
        else:
            out[p.name] = (None, "unknown")
    return out


def sha12(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def prepare_out(out):
    if out.exists():
        extra = set(p.name for p in out.iterdir()) - OUT_NAMES
        if extra:
            raise SystemExit(f"--out {out} exists and holds files this script did not write: {sorted(extra)[:5]}")
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir()


# ---------------------------------------------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------------------------------------------
def seeded_order(paths):
    """r_samples' order: sorted by file name, then random.Random(SEED).sample."""
    cands = sorted(paths, key=lambda p: p.name)
    return random.Random(rs.SEED).sample(cands, len(cands))


def select_best(paths, stats, arms):
    """Selection B. Returns (chosen paths, level used, [level records with candidate counts])."""
    per_level = []
    for lv in LEVELS:
        cands = [p for p in paths if stats[p.name]["n_gt"] >= lv["min_gt"]
                 and sum(stats[p.name]["models"][a]["fully_correct"] for a in arms) >= lv["min_models_fully_correct"]]
        per_level.append((lv, cands))
    # the first level with enough candidates; if none has, the last (most relaxed) level, with what it has
    lv, cands = next(((lv, c) for lv, c in per_level if len(c) >= N_BEST), per_level[-1])
    ranked = sorted(seeded_order(cands), key=lambda p: -stats[p.name]["n_gt"])  # stable: ties keep the seeded order
    return ranked[:N_BEST], lv["level"], [{**v, "candidates": len(c)} for v, c in per_level]


def export_image(src, dst, max_side):
    """Copy the image unchanged, or downscale it to a long side of max_side. Returns (original size, saved size)."""
    with Image.open(src) as im:
        size = im.size
        if max(size) <= max_side:
            shutil.copyfile(src, dst)
            return size, size
        s = max_side / max(size)
        new = (max(1, round(size[0] * s)), max(1, round(size[1] * s)))
        im.convert("RGB").resize(new, Image.LANCZOS).save(dst, "JPEG", quality=JPEG_QUALITY)
        return size, new


def zip_folder(folder, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(folder).as_posix())


# ---------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="YOLO data yaml of the real-world test set")
    ap.add_argument("--weights", nargs=3, required=True, metavar=("A", "B", "C"), help="Gen 2 arm A, B, C weights")
    ap.add_argument("--out", required=True, help="output folder (recreated); the zip is written next to it")
    ap.add_argument("--split", default="test")
    ap.add_argument("--split-csv", help="the build's split.csv, for each image's source (default: split.csv next "
                    "to --data, if there is one)")
    ap.add_argument("--expect", nargs=2, type=int, metavar=("IMAGES", "BOXES"),
                    help="stop unless the test set has exactly this many images and ground-truth boxes")
    ap.add_argument("--device", default=rs.PREDICT["device"])
    ap.add_argument("--max-side", type=int, default=1280, help="long side of exported images, in pixels")
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    t0 = time.time()
    out = Path(args.out).resolve()
    zip_path = out.parent / ZIP_NAME
    paths = test_images(args.data, args.split)
    gts = {p.name: rs.ground_truth(p) for p in paths}
    n_boxes = sum(len(g) for g in gts.values())
    print(f"{args.data}: {len(paths)} {args.split} images, {n_boxes} ground-truth boxes", flush=True)
    if args.expect and [len(paths), n_boxes] != args.expect:
        raise SystemExit(f"expected {args.expect[0]} images and {args.expect[1]} boxes (the paper's real-world test "
                         f"set), got {len(paths)} and {n_boxes}")
    if not paths:
        raise SystemExit("no test images")
    split_csv = args.split_csv or (Path(args.data).parent / "split.csv")
    sources = sources_by_file(split_csv, paths)

    arms = [a for a, _ in rs.ARMS]
    predict = {**rs.PREDICT, "device": args.device}
    weights, preds = {}, {a: {} for a in arms}
    for arm, w in zip(arms, args.weights):
        model = YOLO(w)
        names = [model.names[i] for i in range(len(model.names))]
        if names != rs.CLASSES:
            raise SystemExit(f"{w}: class names {names}, expected the 7 Gen 2 classes {rs.CLASSES}")
        weights[arm] = {"path": Path(w).as_posix(), "sha256_12": sha12(w)}
        ta = time.time()
        for i, (p, res) in enumerate(zip(paths, model.predict([str(p) for p in paths], stream=True, **predict))):
            if Path(res.path).name != p.name:
                raise RuntimeError(f"prediction order mismatch: {res.path} vs {p}")
            b = res.boxes
            preds[arm][p.name] = [(int(c), *(round(float(v), 6) for v in xy), round(float(s), 6))
                                  for xy, c, s in zip(b.xyxyn.cpu().numpy(), b.cls.cpu().numpy(),
                                                      b.conf.cpu().numpy())]
            if (i + 1) % 100 == 0 or i + 1 == len(paths):
                print(f"  {arm}: {i + 1}/{len(paths)} images predicted ({time.time() - ta:.0f} s)", flush=True)

    stats = {}
    for p in paths:
        src, how = sources[p.name]
        stats[p.name] = {"file": p.name, "source": src, "n_gt": len(gts[p.name]),
                         "gt_classes": dict(sorted(Counter(rs.CLASSES[g[0]] for g in gts[p.name]).items())),
                         "models": {a: image_stats(gts[p.name], preds[a][p.name]) for a in arms}}
    scored = [s for s in stats.values() if s["n_gt"] >= 1]
    totals = {"images": len(paths), "boxes": n_boxes, "images_scored": len(scored),
              "images_without_boxes": len(paths) - len(scored), "per_model": {}}
    for a in arms:
        m = [s["models"][a] for s in scored]
        totals["per_model"][a] = {
            "fully_correct": sum(x["fully_correct"] for x in m),
            "fully_correct_share": round(sum(x["fully_correct"] for x in m) / max(len(m), 1), 4),
            "mean_f1": round(sum(x["f1"] for x in m) / max(len(m), 1), 4),
            "tp": sum(x["tp"] for x in m), "fp": sum(x["fp"] for x in m), "fn": sum(x["fn"] for x in m),
            "tp_any_class": sum(x["tp_any_class"] for x in m)}
    n_ok = [sum(s["models"][a]["fully_correct"] for a in arms) for s in scored]
    totals["fully_correct_all_three"] = sum(k == 3 for k in n_ok)
    totals["fully_correct_at_least_two"] = sum(k >= 2 for k in n_ok)
    totals["fully_correct_none"] = sum(k == 0 for k in n_ok)

    sel_a, n_cands_a = rs.choose(paths)
    sel_b, level, levels = select_best(paths, stats, arms)
    if len(sel_b) < N_BEST:
        print(f"WARNING: only {len(sel_b)} images qualify for the best cases even at level {level}", flush=True)

    prepare_out(out)
    exported = {}
    for p in dict.fromkeys(sel_a + sel_b):  # A first, no duplicates
        orig, saved = export_image(p, out / "images" / p.name, args.max_side)
        label = bd.label_path_for(p)
        dst = (out / "labels" / p.name).with_suffix(".txt")
        if label.is_file():
            shutil.copyfile(label, dst)
        else:
            dst.write_text("")
        exported[p.name] = {"original_size": list(orig), "saved_size": list(saved), "downscaled": orig != saved}

    def entry(p, extra=None):
        s = stats[p.name]
        return {"file": p.name, "source": s["source"], "source_from": sources[p.name][1], "n_gt": s["n_gt"],
                "gt_classes": s["gt_classes"], **(extra or {}), **exported[p.name],
                "models": {a: {**s["models"][a], "boxes": [list(b) for b in preds[a][p.name]]} for a in arms}}

    order_a = {p.name: i for i, p in enumerate(seeded_order(
        [p for p in paths if len(gts[p.name]) >= rs.MIN_BOXES]))}
    settings = {
        "data": Path(args.data).as_posix(), "split": args.split, "split_csv": Path(split_csv).as_posix()
        if Path(split_csv).is_file() else None,
        "weights": weights,
        "predict": {k: predict[k] for k in ("conf", "iou", "imgsz", "device")},
        "matching": f"greedy: predictions by confidence (highest first), each to the unmatched ground-truth box of "
                    f"the same class with the highest IoU, a TP if IoU >= {IOU_MIN}; FP = unmatched predictions, "
                    f"FN = unmatched ground truth; tp_any_class: the same matching ignoring the class",
        "fully_correct": "n_gt >= 1 and TP == n_gt and FP == 0",
        "f1": "per image 2 TP / (2 TP + FP + FN); mean over the images with n_gt >= 1",
        "image_export": f"selected images copied unchanged if the long side is <= {args.max_side} px, else "
                        f"downscaled to {args.max_side} px (JPEG quality {JPEG_QUALITY})",
    }
    selection = {
        **settings,
        "totals": totals,
        "selection_a": {
            "purpose": "random real-world row of the sample figure (fig:samples)",
            "rule": f"images with n_gt >= {rs.MIN_BOXES}, sorted by file name, random.Random({rs.SEED}).sample "
                    f"order, first {rs.N_PER_DOMAIN} (r_samples.choose)",
            "candidates": n_cands_a,
            "images": [entry(p, {"seeded_rank": order_a[p.name]}) for p in sel_a]},
        "selection_b": {
            "purpose": "best cases (deliberately favorable; state the rule, level and totals in the caption)",
            "rule": f"level 1: n_gt >= 3 and fully correct for all 3 models; if fewer than {N_BEST} qualify, "
                    f"level 2: n_gt >= 2 and all 3 models; then level 3: n_gt >= 2 and at least 2 of the 3 models. "
                    f"The candidates of the level used are sorted by file name, put in random.Random({rs.SEED})"
                    f".sample order, stably sorted by n_gt (most first); first {N_BEST}",
            "levels": levels, "level_used": level,
            "images": [entry(p, {"models_fully_correct": sum(stats[p.name]["models"][a]["fully_correct"]
                                                             for a in arms)}) for p in sel_b]},
        "overlap_a_b": sorted({p.name for p in sel_a} & {p.name for p in sel_b}),
        "seconds": round(time.time() - t0, 1),
    }
    (out / "per_image_stats.json").write_text(json.dumps(
        {**settings, "totals": totals, "images": [stats[p.name] for p in paths]}, indent=1), encoding="utf-8")
    (out / "selection.json").write_text(json.dumps(selection, indent=1), encoding="utf-8")
    zip_folder(out, zip_path)

    print(f"\nPer model over {len(scored)} images with ground truth:")
    for a, label in rs.ARMS:
        t = totals["per_model"][a]
        print(f"  {label:<18} fully correct {t['fully_correct']:>4} ({100 * t['fully_correct_share']:.1f}%), "
              f"mean F1 {t['mean_f1']:.3f}, TP {t['tp']}, FP {t['fp']}, FN {t['fn']}")
    print(f"  fully correct for all three: {totals['fully_correct_all_three']}, for at least two: "
          f"{totals['fully_correct_at_least_two']}")
    print(f"Selection A ({n_cands_a} candidates): "
          + ", ".join(f"{p.name} [{sources[p.name][0] or 'source unknown'}]" for p in sel_a))
    print("Selection B levels: " + "; ".join(f"level {lv['level']} {lv['candidates']}" for lv in levels)
          + f"; used level {level}")
    for p in sel_b:
        print(f"  {p.name} [{sources[p.name][0] or 'source unknown'}] n_gt {stats[p.name]['n_gt']}")
    down = [n for n, e in exported.items() if e["downscaled"]]
    print(f"Exported {len(exported)} images ({len(down)} downscaled to {args.max_side} px) to {out}")
    print(f"Wrote {zip_path} ({zip_path.stat().st_size / 1e6:.2f} MB) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
