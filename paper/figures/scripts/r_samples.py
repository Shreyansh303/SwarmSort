"""Sample detections of the three Generation 2 models next to the ground truth (Results, Fig. samples).

    .venv/Scripts/python.exe paper/figures/scripts/r_samples.py
    .venv/Scripts/python.exe paper/figures/scripts/r_samples.py --add-domain Real-world <build>/lists/test_real_world.txt

Writes paper/figures/r_samples.jpg (300 dpi, JPEG quality 92, 7.16 in wide) and prints the chosen images, their ground-truth box
counts and the number of boxes each model predicts on each of them. If the Kaggle real-world bundle is in
paper/figures/data/realworld/ (see "Real-world bundle" below), it also writes paper/figures/r_best.jpg.

Rows are test images, columns are the ground truth and the three Gen 2 models (A: defaults, B: random search,
C: PSO; weights in app/models/gen2/). Needs Ultralytics, which is installed in the project venv only.

Selection rule (fixed in advance, the same for every domain): the domain's test images with at least MIN_BOXES
ground-truth boxes are sorted by file name and put in a random order with random.Random(SEED).sample (a separate
generator per domain, seed 0); the first N_PER_DOMAIN images of that order are shown.

Predictions: every model sees the same images with the same settings (confidence 0.25, NMS IoU 0.7, imgsz 640,
CPU). The ground truth is what the Gen 2 evaluation scored, made with the repo's own code (src/build_dataset.py):
  Studio        the Gen 1 test images (configs/split.csv, split == test) in "GARBAGE CLASSIFICATION/", read, mapped
                (parse_label, clip_yolo) and re-encoded (JPEG quality 92) by build_dataset's map_yolo_item and
                process_item, as the build of the garbage_detection source does for its pinned test images.
  India (DWSD)  build(..., only=["dwsd"]) on the local DSWD/ folder, as notebook g2_6 built the India test set:
                every 8-connected region of one mask value is one box (regions under 50 mask pixels dropped),
                classes from the dwsd entry of configs/gen2/sources.yaml, images without boxes dropped. Touching
                items of one class therefore form one box.
Both are written to a temporary folder (seed 42, max side 640, as in results/gen2/build_report.json) that is
deleted at the end. The script checks that it gets the test-set sizes of the paper (1,046 / 7,413 and 700 / 4,502).

More rows: a domain is any list of test images in YOLO format (labels at Ultralytics' label path, images/ ->
labels/, class indices of the seven Gen 2 classes). --add-domain NAME LIST adds one from a text file with one image
path per line, e.g. lists/test_real_world.txt of a local Gen 2 build (TACO and HITL); added rows go between Studio
and India, the order of the domains in the paper. Rows of other domains do not change when one is added.

Real-world bundle: the real-world test photos exist only on Kaggle. Notebook
paper/figures/kaggle/paper_realworld_samples.ipynb runs r_export_realworld.py there and gives realworld_samples.zip;
unzipped into paper/figures/data/realworld/ (--realworld) it holds the selected images, their labels and
selection.json. When that folder exists (and --no-realworld is not given):
  r_samples.jpg  gets a "Real-world" row between Studio (and any --add-domain rows) and India: selection A of the
                 bundle, chosen on Kaggle by this script's own choose() rule over all 468 real-world test images.
  r_best.jpg     (--best-out) is drawn too: the same columns, one row per best-case image. First selection B of the
                 bundle (multi-object best cases), then selection C (single-object images that all three models get
                 fully correct); both are deliberately favorable, rule-based choices, and each row is marked on the
                 left with its number of objects. The rules, the level used, every model's totals and the fully
                 correct counts by objects per image are in selection.json and printed here for the caption. A
                 bundle without selection C (exported before it was added) gives selection B alone, as before.
Bundle rows draw the boxes recorded on Kaggle, the ones the selection was scored on. The models are also run here on
the bundle images and any difference from the recorded boxes is printed. Without the folder the output is the
figure of the paper as before.
"""
import argparse
import csv
import json
import random
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
import build_dataset as bd  # noqa: E402
from verify_dataset import parse_label  # noqa: E402

OUT = REPO / "paper" / "figures" / "r_samples.jpg"
BEST_OUT = REPO / "paper" / "figures" / "r_best.jpg"
REALWORLD = REPO / "paper" / "figures" / "data" / "realworld"  # the unzipped Kaggle bundle (r_export_realworld.py)
SOURCES = REPO / "configs" / "gen2" / "sources.yaml"
SPLIT_CSV = REPO / "configs" / "split.csv"
WEIGHTS = REPO / "app" / "models" / "gen2"
BUILD_SEED, MAX_SIDE = 42, 640  # settings of the Gen 2 build (results/gen2/build_report.json)

SEED, MIN_BOXES, N_PER_DOMAIN = 0, 3, 2
PREDICT = dict(conf=0.25, iou=0.7, imgsz=640, device="cpu", verbose=False)

ARMS = [("A", "A: Defaults"), ("B", "B: Random search"), ("C", "C: PSO")]
CLASSES = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC", "OTHER"]
SHORT = ["BIO", "CARD", "GLASS", "METAL", "PAPER", "PLAST", "OTHER"]
# One color per class, in class-index order (a categorical palette checked for color-vision deficiency on adjacent
# slots); every box also carries its class name, so color is never the only cue.
CLASS_COLOR = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK, MUTED, PAD_GREY = "#1f2937", "#6b7280", (114, 114, 114)
PAGE_W = 7.16  # IEEE text width, inches
LABEL_PT = 6.0  # box labels, at print size

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "font.size": 8,
    "savefig.dpi": 300,
})


# ---------------------------------------------------------------------------------------------------------------
# Domains: each returns the paths of its test images; their YOLO labels sit at Ultralytics' label path
# ---------------------------------------------------------------------------------------------------------------
def quiet(*_):
    pass


def studio_images(tmp):
    """The Gen 1 test images, mapped and written exactly as the Gen 2 build writes its pinned studio test images."""
    classes, (src,) = bd.load_config(SOURCES, only=["garbage_detection"])
    assert classes == CLASSES, classes
    root = bd.locate_yolo(src, REPO)
    stats = bd.new_stats()
    names, items = bd.read_yolo(src, root, classes, stats)
    with open(SPLIT_CSV, newline="") as f:
        test = {r["file"] for r in csv.DictReader(f) if r["split"] == "test"}
    items = [it for it in items if Path(it["uid"]).name in test]  # build pins by file name, as done here
    for it in items:
        it["format"], it["pinned"] = src["format"], "test"
    items = [it for it in items if bd.map_yolo_item(it, src, names, classes, stats)]
    bd.assign_file_names(items)
    out = bd.prepare_out_dir(Path(tmp) / "studio", [root.resolve()])
    paths = []
    for it in items:
        rec, reason, _ = bd.process_item(it, src, out, MAX_SIDE)
        if rec is not None:
            paths.append(out / "images" / rec["file"])
    return paths


def dwsd_images(tmp):
    """The India test set, built from the local DSWD/ masks by build_dataset.build(--only dwsd)."""
    out = Path(tmp) / "dwsd"
    report = bd.build(SOURCES, out, seed=BUILD_SEED, max_side=MAX_SIDE, search_root=REPO, only=["dwsd"], log=quiet)
    assert report["classes"] == CLASSES, report["classes"]
    return read_list(out / "lists" / "test_india.txt")


def read_list(path):
    return [Path(line.strip()) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def ground_truth(img_path):
    """[(cls, x0, y0, x1, y1)] normalized, read and clipped as build_dataset's map_yolo_item does."""
    label = bd.label_path_for(Path(img_path))
    if not label.is_file():
        return []
    boxes, _ = parse_label(label, len(CLASSES))
    out = []
    for b in boxes:
        c = bd.clip_yolo(b)
        if c is not None:
            cls, cx, cy, w, h = c
            out.append((cls, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def choose(paths):
    """The selection rule: >= MIN_BOXES boxes, sorted by name, seeded random order, first N_PER_DOMAIN."""
    cands = sorted((p for p in paths if len(ground_truth(p)) >= MIN_BOXES), key=lambda p: p.name)
    order = random.Random(SEED).sample(cands, len(cands))
    return order[:N_PER_DOMAIN], len(cands)


def predict_boxes(model, path):
    """[(cls, x0, y0, x1, y1, conf)] normalized, lowest confidence first."""
    res = model.predict(str(path), **PREDICT)[0]
    xyxyn = res.boxes.xyxyn.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    conf = res.boxes.conf.cpu().numpy()
    return sorted([(int(c), *map(float, b), float(s)) for b, c, s in zip(xyxyn, cls, conf)], key=lambda t: t[-1])


# ---------------------------------------------------------------------------------------------------------------
# The Kaggle real-world bundle (r_export_realworld.py): images chosen and predicted there
# ---------------------------------------------------------------------------------------------------------------
def load_bundle(folder):
    """selection.json of an unzipped bundle, or None if there is none."""
    sel = Path(folder) / "selection.json"
    if not sel.is_file():
        return None
    return json.loads(sel.read_text(encoding="utf-8"))


def same_boxes(a, b, tol=0.01):
    """Same classes and boxes (coordinates and confidence within tol), in any order."""
    if len(a) != len(b):
        return False
    a, b = sorted(a), sorted(b)
    return all(x[0] == y[0] and max(abs(u - v) for u, v in zip(x[1:], y[1:])) <= tol for x, y in zip(a, b))


def bundle_rows(name, folder, selection, models):
    """Figure rows from one selection of the bundle, drawing the boxes recorded on Kaggle."""
    rows = []
    for e in selection["images"]:
        path = Path(folder) / "images" / e["file"]
        gt = ground_truth(path)
        if len(gt) != e["n_gt"]:
            raise SystemExit(f"{path}: {len(gt)} ground-truth boxes, but selection.json says {e['n_gt']}")
        preds = {arm: sorted((tuple(b) for b in e["models"][arm]["boxes"]), key=lambda t: t[-1])
                 for arm, _ in ARMS}
        stats = "; ".join(f"{arm} TP {e['models'][arm]['tp']} FP {e['models'][arm]['fp']} FN {e['models'][arm]['fn']}"
                          for arm, _ in ARMS)
        print(f"    {e['file']} [{e.get('source') or 'source unknown'}]: ground truth {len(gt)} boxes; {stats}")
        if not e.get("downscaled"):  # a downscaled image cannot give exactly the same boxes
            differ = [arm for arm, m in models.items() if not same_boxes(predict_boxes(m, path), preds[arm])]
            print("      local re-prediction: " + (f"DIFFERS for {', '.join(differ)} (the recorded Kaggle boxes "
                                                     f"are drawn)" if differ else "same boxes as recorded"))
        rows.append((name, path, gt, preds))
    return rows


def print_bundle_totals(bundle):
    t = bundle["totals"]
    print(f"  Kaggle totals over {t['images_scored']} real-world test images: "
          + ", ".join(f"{arm} fully correct {t['per_model'][arm]['fully_correct']}, mean F1 "
                      f"{t['per_model'][arm]['mean_f1']:.3f}" for arm, _ in ARMS)
          + f"; all three {t['fully_correct_all_three']}")
    for k, g in t.get("fully_correct_by_objects", {}).items():
        print(f"    {k} object(s) per image: {g['images']} images; fully correct "
              + ", ".join(f"{arm} {g[arm]}" for arm, _ in ARMS) + f", all three {g['all_three']}")


def object_label(n):
    return "1 object" if n == 1 else f"{n} objects"


# ---------------------------------------------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------------------------------------------
def text_color(hex_color):
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r ** 2.2 + 0.7152 * g ** 2.2 + 0.0722 * b ** 2.2
    return INK if lum > 0.35 else "white"


def letterbox(img):
    """Pad to a square with the YOLO letterbox grey; returns the array and the offset/scale of the original."""
    w, h = img.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), PAD_GREY)
    ox, oy = (s - w) // 2, (s - h) // 2
    canvas.paste(img, (ox, oy))
    return np.asarray(canvas), (ox, oy, w, h, s)


def tag_extent(text, renderer):
    """Window extent of a text including its background tag (pad 0.15 em on each side)."""
    pad = 0.15 * text.get_fontsize() * renderer.points_to_pixels(1.0)
    return text.get_window_extent(renderer).padded(pad)


def place_labels(ax, labels, reserved):
    """Put each label at the top-left corner of its box: above it, else inside it, else below it, shifted left
    where it would leave the panel. A label that would cover one already placed is left out (its box stays)."""
    renderer = ax.figure.canvas.get_renderer()
    panel = ax.get_window_extent(renderer)
    data_per_px = (ax.get_xlim()[1] - ax.get_xlim()[0]) / panel.width
    placed = list(reserved)
    for text, color, X0, Y0, X1, Y1 in labels:
        for y, va in ((Y0, "bottom"), (Y0, "top"), (Y1, "top")):
            t = ax.text(X0, y, text, fontsize=LABEL_PT, color=text_color(color), va=va, ha="left", zorder=5,
                        bbox=dict(boxstyle="square,pad=0.15", facecolor=color, edgecolor="none", alpha=0.9))
            bb = tag_extent(t, renderer)
            if bb.x1 > panel.x1:
                t.set_x(X0 - (bb.x1 - panel.x1) * data_per_px)
                bb = tag_extent(t, renderer)
            fits = bb.y0 >= panel.y0 and bb.y1 <= panel.y1 and bb.x0 >= panel.x0
            if fits and not any(bb.overlaps(p) for p in placed):
                placed.append(bb)
                break
            t.remove()


def draw_panel(ax, arr, frame, boxes, count_text):
    """boxes: [(cls, x0, y0, x1, y1, conf or None)], normalized to the original image."""
    ox, oy, w, h, s = frame
    ax.imshow(arr, extent=(0, s, s, 0), interpolation="antialiased")
    ax.set_xlim(0, s)
    ax.set_ylim(s, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(MUTED)
        spine.set_linewidth(0.5)
    count = ax.text(0.985 * s, 0.985 * s, count_text, fontsize=LABEL_PT, color=INK, ha="right", va="bottom",
                    zorder=6, bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))
    labels = []
    # boxes: largest drawn first, so small boxes stay visible; labels: highest confidence (ground truth: largest
    # box) placed first
    for cls, x0, y0, x1, y1, conf in sorted(boxes, key=lambda b: -(b[3] - b[1]) * (b[4] - b[2])):
        X0, Y0, X1, Y1 = ox + x0 * w, oy + y0 * h, ox + x1 * w, oy + y1 * h
        color = CLASS_COLOR[cls]
        ax.add_patch(Rectangle((X0, Y0), X1 - X0, Y1 - Y0, fill=False, edgecolor=color, linewidth=0.9))
        text = SHORT[cls] if conf is None else f"{SHORT[cls]} {conf:.2f}".replace(" 0.", " .")
        labels.append((-(conf if conf is not None else 0.0), text, color, X0, Y0, X1, Y1))
    labels.sort(key=lambda l: l[0])  # stable: equal keys keep the largest-box-first order
    place_labels(ax, [l[1:] for l in labels], [tag_extent(count, ax.figure.canvas.get_renderer())])


def make_figure(rows, out_path, row_labels=None):
    """rows: [(domain, image path, gt boxes, {arm: boxes})]. The left margin names each domain once, centered on its
    rows; with row_labels (one text per row) it names every row instead (a wider gap still separates domains)."""
    n_rows, n_cols = len(rows), 1 + len(ARMS)
    left, right, top, bottom = 0.30, 0.03, 0.20, 0.30  # inches
    gap, domain_gap = 0.04, 0.10
    panel = (PAGE_W - left - right - (n_cols - 1) * gap) / n_cols
    breaks = sum(1 for i in range(1, n_rows) if rows[i][0] != rows[i - 1][0])
    height = top + bottom + n_rows * panel + (n_rows - 1 - breaks) * gap + breaks * domain_gap
    fig = plt.figure(figsize=(PAGE_W, height), dpi=300)  # label overlap is measured at the saved resolution
    headers = ["Ground truth"] + [label for _, label in ARMS]
    y = height - top
    spans, centers = {}, []
    for r, (domain, path, gt, preds) in enumerate(rows):
        if r and domain != rows[r - 1][0]:
            y -= domain_gap - gap
        y -= panel
        with Image.open(path) as im:
            arr, frame = letterbox(im.convert("RGB"))
        for c in range(n_cols):
            x = left + c * (panel + gap)
            ax = fig.add_axes([x / PAGE_W, y / height, panel / PAGE_W, panel / height])
            if c == 0:
                draw_panel(ax, arr, frame, [(*b, None) for b in gt], "1 box" if len(gt) == 1 else f"{len(gt)} boxes")
            else:
                boxes = preds[ARMS[c - 1][0]]
                draw_panel(ax, arr, frame, boxes, f"{len(boxes)} det.")
            if r == 0:
                ax.set_title(headers[c], fontsize=8, color=INK, pad=3)
        lo, hi = spans.get(domain, (y, y + panel))
        spans[domain] = (min(lo, y), max(hi, y + panel))
        centers.append(y + panel / 2)
        y -= gap
    if row_labels is None:
        for domain, (lo, hi) in spans.items():
            fig.text((left - 0.12) / PAGE_W, (lo + hi) / 2 / height, domain, rotation=90, ha="center", va="center",
                     fontsize=8, color=INK)
    else:
        for text, yc in zip(row_labels, centers, strict=True):
            fig.text((left - 0.12) / PAGE_W, yc / height, text, rotation=90, ha="center", va="center", fontsize=8,
                     color=INK)
    handles = [Patch(facecolor="none", edgecolor=col, linewidth=1.2, label=f"{full} ({short})" if short != full
                     else full) for full, short, col in zip(CLASSES, SHORT, CLASS_COLOR)]
    fig.legend(handles=handles, loc="lower center", ncol=len(CLASSES), frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5 + (left - right) / 2 / PAGE_W, 0.04 / height), handlelength=1.4,
               columnspacing=1.1, handletextpad=0.4)
    fig.savefig(out_path, dpi=300, facecolor="white", pil_kwargs={"quality": 92})
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--add-domain", nargs=2, action="append", default=[], metavar=("NAME", "LIST"),
                    help="add a row domain from a text file of test image paths (YOLO labels next to them)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--realworld", default=str(REALWORLD), help="the unzipped Kaggle real-world bundle (used if it "
                    "has a selection.json)")
    ap.add_argument("--no-realworld", action="store_true", help="ignore the real-world bundle")
    ap.add_argument("--best-out", default=str(BEST_OUT), help="the best-case figure (drawn from the bundle)")
    args = ap.parse_args(argv)

    from ultralytics import YOLO  # project venv only

    bundle = None if args.no_realworld else load_bundle(args.realworld)
    if bundle is not None and any(name == "Real-world" for name, _ in args.add_domain):
        raise SystemExit("the real-world bundle already adds a 'Real-world' row: drop --add-domain Real-world or "
                         "pass --no-realworld")
    domains = [("Studio", studio_images, (1046, 7413))]
    domains += [(name, (lambda tmp, p=lst: read_list(p)), None) for name, lst in args.add_domain]
    if bundle is not None:
        domains += [("Real-world", None, None)]  # rows chosen and predicted on Kaggle (selection A)
    domains += [("India (DWSD)", dwsd_images, (700, 4502))]

    models = {arm: YOLO(str(WEIGHTS / f"arm_{arm.lower()}.pt")) for arm, _ in ARMS}
    for arm, m in models.items():
        assert [m.names[i] for i in range(len(m.names))] == CLASSES, (arm, m.names)

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, get_images, expected in domains:
            if get_images is None:
                sel = bundle["selection_a"]
                print(f"{name} (Kaggle bundle {args.realworld}): {bundle['totals']['images']} test images, "
                      f"{bundle['totals']['boxes']} boxes\n  {sel['candidates']} images with >= {MIN_BOXES} boxes; "
                      f"chosen (seed {SEED}): " + ", ".join(e["file"] for e in sel["images"]))
                rows += bundle_rows(name, args.realworld, sel, models)
                continue
            paths = get_images(tmp)
            n_boxes = sum(len(ground_truth(p)) for p in paths)
            print(f"{name}: {len(paths)} test images, {n_boxes} boxes")
            if expected is not None and (len(paths), n_boxes) != expected:
                raise SystemExit(f"{name}: expected {expected[0]} images and {expected[1]} boxes as in the paper")
            chosen, n_cands = choose(paths)
            print(f"  {n_cands} images with >= {MIN_BOXES} boxes; chosen (seed {SEED}): "
                  + ", ".join(p.name for p in chosen))
            for p in chosen:
                gt = ground_truth(p)
                preds = {arm: predict_boxes(m, p) for arm, m in models.items()}
                print(f"    {p.name}: ground truth {len(gt)} boxes; predicted "
                      + ", ".join(f"{arm} {len(preds[arm])}" for arm, _ in ARMS))
                rows.append((name, p, gt, preds))
        make_figure(rows, args.out)  # inside the with block: the images live in the temporary folder
    print("Wrote", args.out)

    if bundle is not None:
        sel = bundle["selection_b"]
        levels = "; ".join(f"level {lv['level']} (n_gt >= {lv['min_gt']}, fully correct for >= "
                           f"{lv['min_models_fully_correct']} models) {lv['candidates']}" for lv in sel["levels"])
        print(f"Best cases (selection B, deliberately favorable): candidates {levels}; level used "
              f"{sel['level_used']}")
        print_bundle_totals(bundle)
        sel_c = bundle.get("selection_c")  # missing in a bundle exported before selection C was added
        if sel_c is None and not sel["images"]:
            raise SystemExit("selection B of the bundle is empty: no best-case figure")
        rows = bundle_rows("Real-world", args.realworld, sel, models)
        row_labels = None  # without selection C: one "Real-world" label, as before
        if sel_c is not None:
            print(f"Single-object successes (selection C, deliberately favorable): {sel_c['candidates']} candidates "
                  f"({sel_c['rule']})")
            rows += bundle_rows("Real-world, one object", args.realworld, sel_c, models)
            if not rows:
                raise SystemExit("selections B and C of the bundle are empty: no best-case figure")
            row_labels = [object_label(len(gt)) for _, _, gt, _ in rows]
        make_figure(rows, args.best_out, row_labels)
        print("Wrote", args.best_out)


if __name__ == "__main__":
    main()
