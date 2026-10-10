"""Pure helpers of the SwarmSort demo app: generations, bin rules, image handling, inference, drawing, tables.

Nothing here imports Streamlit, so the unit tests can import this file directly. app/app.py builds the page.
"""
import hashlib
import io
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Ultralytics wraps PIL's Image.open, and the first file that fails to open makes it pip-install a HEIC plugin.
# A live demo must never install packages, so auto-install is switched off (read when Ultralytics is imported).
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

APP_DIR = Path(__file__).resolve().parent
REPO = APP_DIR.parent
MODELS_DIR = APP_DIR / "models"
RESULTS_JSON = REPO / "results" / "test_results.json"  # Generation 1
GEN2_RESULTS_DIR = REPO / "results" / "gen2"
TEST_LIST = REPO / "configs" / "lists" / "test.txt"  # written per machine by src/verify_dataset.py
DATASET_DIR = REPO / "GARBAGE CLASSIFICATION"

IMGSZ = 416      # Generation 1 training image size; Ultralytics letterboxes every photo to the model's size
MAX_SIDE = 1280  # larger photos are shrunk first (the model sees 416 or 640 px anyway), so drawing stays fast

# Generation 1: studio photos, 6 classes, imgsz 416. Generation 2: studio + real-world photos, 7 classes (adds
# OTHER), imgsz 640. Each generation has its own three arms, weights folder and test results. The imgsz here is
# only a fallback: the app reads the real one from each checkpoint's train_args.
GENERATIONS = {
    "gen2": {"label": "Generation 2 (real-world data, 7 classes)", "short": "Generation 2",
             "dir": MODELS_DIR / "gen2", "imgsz": 640, "results": GEN2_RESULTS_DIR / "test_all_results.json"},
    "gen1": {"label": "Generation 1 (studio data, 6 classes)", "short": "Generation 1",
             "dir": MODELS_DIR, "imgsz": IMGSZ, "results": RESULTS_JSON},
}
DEFAULT_GENERATION = "gen2"
DOMAINS = ["all", "studio", "real_world", "india"]  # Generation 2 test sets: results/gen2/test_<domain>_results.json

# All classes in the models' index order: the 6 Generation 1 classes, then Generation 2's OTHER
CLASS_NAMES = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC", "OTHER"]
# Box colour per class. Only BIODEGRADABLE (green bin) is green; the dry classes and OTHER use colours that are
# neither green nor blue, so a box colour is never mistaken for a bin colour.
CLASS_COLORS = {
    "BIODEGRADABLE": "#237a33",  # green
    "CARDBOARD": "#8b5a2b",      # brown
    "GLASS": "#a3399e",          # purple
    "METAL": "#707070",          # grey
    "PAPER": "#e6c200",          # yellow (dark label text)
    "PLASTIC": "#cc3a24",        # red-orange
    "OTHER": "#262626",          # near-black, like the reject bin
}
OTHER_COLOR = "#000000"  # a class name this app does not know

# The three trained arms of each generation (byte copies of the trained best.pt; see app/models/README.md)
MODELS = {
    "A": {"name": "Defaults", "file": "arm_a.pt"},
    "B": {"name": "Random search", "file": "arm_b.pt"},
    "C": {"name": "PSO", "file": "arm_c.pt"},
}

# Bin rules after India's Solid Waste Management Rules 2016, which ask households to keep biodegradable,
# non-biodegradable and domestic hazardous waste apart: green = wet / biodegradable, blue = dry / recyclable, and a
# reject bin (black in this app) for OTHER: dry waste that cannot be recycled, plus domestic hazardous bits that
# need their own drop-off. Cities handle the reject stream differently, so the label says to check local rules.
BINS = {"green": "Green bin (wet / compost)", "blue": "Blue bin (dry recyclables)",
        "reject": "Reject bin (non-recyclable / check local rules)"}
BIN_RULES = {
    "BIODEGRADABLE": {"bin": "green", "tip": "Compost"},
    "PAPER": {"bin": "blue", "tip": "Paper recycling; keep it dry and clean"},
    "CARDBOARD": {"bin": "blue", "tip": "Paper recycling; flatten boxes"},
    "PLASTIC": {"bin": "blue", "tip": "Plastic recycling; rinse containers"},
    "METAL": {"bin": "blue", "tip": "Metal recycling; rinse cans"},
    "GLASS": {"bin": "blue", "tip": "Glass recycling; handle with care, keep separate if broken"},
    "OTHER": {"bin": "reject", "tip": "Not recyclable (cigarette butts, dirty mixed litter, textiles); take "
                                      "batteries, medicines or sharps to a domestic hazardous waste drop-off"},
}


class ImageError(ValueError):
    """A file that cannot be read as an image. The message is meant for the user."""


# ---------------------------------------------------------------------------------------------------------------
# Test results (results/test_results.json); every reader returns None or [] if the file is missing
# ---------------------------------------------------------------------------------------------------------------
def load_results(path=RESULTS_JSON):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def test_map50(results, label):
    try:
        return float(results["models"][label]["metrics"]["map50"])
    except (TypeError, KeyError, ValueError):
        return None


def model_label(label, results=None):
    """'A: Defaults (test mAP@50 0.676)', or 'A: Defaults' when the score is unknown."""
    text = f"{label}: {MODELS[label]['name']}"
    score = test_map50(results, label)
    return text if score is None else f"{text} (test mAP@50 {score:.3f})"


def default_model(results=None):
    """The model with the highest test mAP@50; 'A' when the results are missing."""
    scores = {label: test_map50(results, label) for label in MODELS}
    scores = {label: s for label, s in scores.items() if s is not None}
    return max(scores, key=scores.get) if scores else "A"


def models_tied(results):
    """True if no pairwise mAP@50 difference is significant (each bootstrap CI contains 0); None if unknown."""
    try:
        pairs = results["pairwise"]
        return bool(pairs) and not any(p["map50"]["significant"] for p in pairs)
    except (TypeError, KeyError):
        return None


def comparison_rows(results):
    """One row per model: test mAP@50 and mAP@50-95 with their 95% bootstrap CIs."""
    def fmt(value, ci):
        return f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"

    rows = []
    for label in MODELS:
        try:
            m = results["models"][label]
            rows.append({"Model": f"{label}: {MODELS[label]['name']}",
                         "Test mAP@50 [95% CI]": fmt(m["metrics"]["map50"], m["bootstrap"]["map50_ci95"]),
                         "Test mAP@50-95 [95% CI]": fmt(m["metrics"]["map50_95"], m["bootstrap"]["map50_95_ci95"])})
        except (TypeError, KeyError, IndexError):
            continue
    return rows


def pairwise_note(results):
    """Plain summary of the test mAP@50 pairwise verdicts, e.g. 'B and C significantly beat A; C vs B is a tie.'

    Uses each pair's paired-bootstrap verdict ('significant'); the sign of observed_diff (b minus a) picks the
    winner. None if the results have no pairwise comparisons.
    """
    try:
        pairs = [(p["a"], p["b"], p["map50"]) for p in results["pairwise"]]
    except (TypeError, KeyError):
        return None
    if not pairs:
        return None
    beaten, ties = {}, []
    for a, b, m in pairs:
        if m.get("significant"):
            winner, loser = (b, a) if m.get("observed_diff", 0) > 0 else (a, b)
            beaten.setdefault(loser, []).append(winner)
        else:
            ties.append(f"{b} vs {a}")
    if not beaten:
        return "The three arms are statistically tied (every paired bootstrap 95% CI of their differences contains 0)."
    parts = []
    for loser, winners in beaten.items():
        winners = sorted(winners)
        names = winners[0] if len(winners) == 1 else ", ".join(winners[:-1]) + " and " + winners[-1]
        parts.append(f"{names} significantly beat{'s' if len(winners) == 1 else ''} {loser}")
    parts += [f"{t} is a tie" for t in ties]
    return "; ".join(parts) + "."


def load_domain_results(results_dir=GEN2_RESULTS_DIR):
    """{domain: Generation 2 test results, or None if missing} for every domain in DOMAINS."""
    return {d: load_results(Path(results_dir) / f"test_{d}_results.json") for d in DOMAINS}


def domain_rows(domain_results):
    """One row per arm: test mAP@50 [95% CI] on each domain that has results ('n/a' where an arm is missing)."""
    present = {d: r for d, r in (domain_results or {}).items() if r}
    if not present:
        return []
    rows = []
    for label in MODELS:
        row = {"Model": f"{label}: {MODELS[label]['name']}"}
        for domain, results in present.items():
            images = results.get("images")
            column = f"{domain} ({images} images)" if images else domain
            try:
                m = results["models"][label]
                ci = m["bootstrap"]["map50_ci95"]
                row[column] = f"{m['metrics']['map50']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
            except (TypeError, KeyError, IndexError):
                row[column] = "n/a"
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------------------------
def model_path(generation, label):
    return GENERATIONS[generation]["dir"] / MODELS[label]["file"]


def missing_weights_message(generation, label):
    """What the page says when a model file is not on disk."""
    path = model_path(generation, label)
    try:
        shown = path.relative_to(REPO).as_posix()
    except ValueError:
        shown = str(path)
    text = (f"Model file not found: {shown}. It is a copy of the trained {GENERATIONS[generation]['short']} "
            "model (see app/models/README.md for its source).")
    others = [GENERATIONS[g]["short"] for g in GENERATIONS if g != generation]
    if others:
        text += f" Meanwhile, pick {others[0]} in the sidebar."
    return text


def sha256_12(path):
    """First 12 hex digits of a file's sha256, as recorded in the results files and app/models/README.md."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def test_images(test_list=TEST_LIST, dataset_dir=DATASET_DIR):
    """Test-split images that exist on this machine; [] on a fresh clone without the dataset."""
    if not (Path(test_list).is_file() and Path(dataset_dir).is_dir()):
        return []
    lines = Path(test_list).read_text(encoding="utf-8").splitlines()
    return [p for p in (Path(line.strip()) for line in lines if line.strip()) if p.is_file()]


def caused_by(error, kind):
    """True if the error or any error it was raised from is a `kind` (Ultralytics' Image.open re-raises others)."""
    while error is not None:
        if isinstance(error, kind):
            return True
        error = error.__cause__ or error.__context__
    return False


def prepare_image(source, max_side=MAX_SIDE):
    """Bytes, a path or a file object -> upright RGB PIL image, at most max_side pixels on its long side."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    try:
        with Image.open(source) as raw:
            img = ImageOps.exif_transpose(raw)  # phone photos store their rotation in EXIF
            if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
                img = img.convert("RGBA")
                img = Image.alpha_composite(Image.new("RGBA", img.size, "white"), img)  # transparent -> white
            img = img.convert("RGB")  # palette and grayscale -> RGB too
    except Exception as e:  # unknown format, truncated or damaged file, decompression bomb, ...
        if caused_by(e, Image.DecompressionBombError):
            raise ImageError("This image is too large to process. Please use a smaller photo.") from e
        raise ImageError("This file could not be read as an image (it may be damaged or in an unsupported "
                         "format). Please try a JPG, PNG or WebP photo.") from e
    img.thumbnail((max_side, max_side))  # keeps the aspect ratio; smaller images are left as they are
    return img


# ---------------------------------------------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------------------------------------------
def model_imgsz(model, fallback=IMGSZ):
    """The image size the model was trained at: the checkpoint's train_args['imgsz'], else the fallback."""
    ckpt = getattr(model, "ckpt", None)
    train_args = ckpt.get("train_args") if isinstance(ckpt, dict) else None
    size = train_args.get("imgsz") if isinstance(train_args, dict) else None
    if isinstance(size, (list, tuple)):
        size = max(size) if size else None
    try:
        size = int(size)
    except (TypeError, ValueError):
        return fallback
    return size if size > 0 else fallback


def model_class_names(model):
    """The model's class names in index order, or [] if it has none."""
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names) if isinstance(names, (list, tuple)) else []


def load_model(path, fallback_imgsz=IMGSZ):
    """Load a YOLO checkpoint and run one warm-up prediction at its training size, so the first photo is quick."""
    from ultralytics import YOLO

    model = YOLO(str(path))
    size = model_imgsz(model, fallback_imgsz)
    model.predict(np.zeros((size, size, 3), dtype=np.uint8), imgsz=size, verbose=False)
    return model


def detect(model, image, conf=0.35, iou=0.7, imgsz=IMGSZ):
    """Run the model on one RGB PIL image -> (detections sorted by confidence, inference time in ms).

    Each detection is {"class": name, "conf": float, "box": (x1, y1, x2, y2) in image pixels}.
    The time covers Ultralytics' preprocessing, the network and NMS.
    """
    start = time.perf_counter()
    result = model.predict(image, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
    ms = (time.perf_counter() - start) * 1000
    boxes = result.boxes
    detections = [{"class": result.names[int(c)], "conf": float(p), "box": tuple(xyxy)}
                  for c, p, xyxy in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist())]
    detections.sort(key=lambda d: d["conf"], reverse=True)
    return detections, ms


# ---------------------------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------------------------
def class_color(name):
    return CLASS_COLORS.get(name, OTHER_COLOR)


def label_text_color(color):
    """'black' or 'white', whichever reads better on the colour '#rrggbb' (WCAG relative luminance)."""
    channels = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return "black" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.179 else "white"


def draw_detections(image, detections):
    """Copy of the image with a box and a 'CLASS 0.87' label per detection, in the class's colour."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    side = max(out.size)
    line = max(2, round(side / 250))
    font = ImageFont.load_default(size=max(12, round(side / 32)))
    for d in sorted(detections, key=lambda d: d["conf"]):  # most confident last, so it is drawn on top
        x1, y1, x2, y2 = d["box"]
        color = class_color(d["class"])
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line)
        text = f"{d['class']} {d['conf']:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        w, h, pad = right - left, bottom - top, line
        tx = max(0, min(x1, out.width - w - 2 * pad))       # keep the label inside the image
        ty = y1 - h - 2 * pad if y1 >= h + 2 * pad else y1  # above the box, or inside it at the top edge
        draw.rectangle((tx, ty, tx + w + 2 * pad, ty + h + 2 * pad), fill=color)
        draw.text((tx + pad - left, ty + pad - top), text, fill=label_text_color(color), font=font)
    return out


def detection_rows(detections):
    """Table rows: #, item, confidence, bin, disposal tip."""
    unknown = {"bin": None, "tip": "Check your local rules"}
    rows = []
    for i, d in enumerate(detections, start=1):
        rule = BIN_RULES.get(d["class"], unknown)
        rows.append({"#": i, "Item": d["class"], "Confidence": round(d["conf"], 2),
                     "Bin": BINS.get(rule["bin"], "Unknown"), "Disposal tip": rule["tip"]})
    return rows


def bin_summary(detections):
    """{bin label: number of items} for the bins that got at least one item, in BINS order."""
    counts = Counter(BIN_RULES[d["class"]]["bin"] for d in detections if d["class"] in BIN_RULES)
    return {label: counts[key] for key, label in BINS.items() if counts[key]}
