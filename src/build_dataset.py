"""Generation 2: merge several waste-detection datasets (YOLO, COCO, Supervisely, Labelme) into one YOLO dataset.

    python src/build_dataset.py --sources configs/gen2/sources.yaml --out data_gen2 --seed 42
    python src/build_dataset.py --sources configs/gen2/sources.yaml --out /kaggle/working/gen2_data \
        --search-root /kaggle/input --max-side 1280

Relocate mode: a later notebook gets a finished build as a read-only input (e.g. /kaggle/input/<notebook>/gen2_data).
The lists and yamls inside it point to where it was built, so rewrite them into a writable folder:

    python src/build_dataset.py --relocate-from /kaggle/input/<notebook>/gen2_data --out /kaggle/working/gen2_cfg

This writes data.yaml, data_test_*.yaml, lists/*.txt and split.csv (with a 'path' column) into --out, every path
pointing into the read-only build. Nothing is written into the build folder.

Steps:
  1. Read the sources config: target classes, and per source its format, where to find it, and how its classes map
     to the target classes (a target name, drop_annotation, drop_and_mask or drop_image; an unmapped class is an
     error).
  2. Read every enabled source. YOLO: images under any images/ folder, labels in the matching labels/ folder
     (Ultralytics' rule). COCO: one annotations json and an image folder; crowd and invalid boxes are skipped
     and counted. Supervisely: <dataset>/ann/<image>.json next to <dataset>/img/<image>, plus the project
     meta.json; the class can come from an object tag (e.g. Material). Labelme: one json per image with
     polygon/rectangle shapes; each shape's bounding box is one box. A source that is not found is an error naming
     the Kaggle dataset to attach, unless it is optional (then it is skipped with a warning).
  3. Optional per source: force_split (every image to one split) and subsample (a seeded, class-stratified sample
     of a pinned source's train/valid images; test is never subsampled).
  4. Write every kept image as JPEG quality 92 to <out>/images/<source>__<name>.jpg, with EXIF orientation
     applied (and the EXIF dropped, so no loader rotates it twice) and the long side at most --max-side. Boxes
     mapped to drop_and_mask are painted mid-grey (114, the YOLO letterbox colour). The boxes are written,
     normalized, to <out>/labels/.
  5. Split: duplicates across all sources are grouped with verify_dataset's rules (same bytes, same Roboflow
     source image within a source, near-identical dHash or thumbnail). Groups are then split 70/20/10, stratified
     by (source, dominant class), and a group always stays in one split. A source with keep_split_csv keeps its
     Gen 1 assignment; its duplicates in other sources follow it, and test always wins over valid and train.
  6. Outputs: split.csv, lists/{train,valid,test}.txt, lists/test_<domain>.txt, lists/test_src_<source>.txt,
     data.yaml, data_test_<domain>.yaml, data_test_src_<source>.yaml and build_report.json.

The source folders are only read. The output is the same for the same seed, inputs and settings.
"""
import base64
import argparse
import csv
import hashlib
import io
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_dataset import (  # noqa: E402
    CANDIDATE_MAX_BITS, DHASH_SIZE, IMG_EXTS, MAX_ROOT_DEPTH, MIN_THUMB_CORR, NEAR_DUP_MAX_BITS, PROTECTED_BY,
    SPLIT_FRACS, SPLITS, THUMB_SIZE, assign_near_dup_groups, count_shared_key_pairs, dhash_bits, duplicate_groups,
    load_names, parse_label)

REPO = Path(__file__).resolve().parent.parent
JPEG_QUALITY = 92
DROP_ANN, DROP_IMG, DROP_MASK = "drop_annotation", "drop_image", "drop_and_mask"
DROP_TARGETS = (DROP_ANN, DROP_IMG, DROP_MASK)
MASK_GREY = (114, 114, 114)  # YOLO's letterbox colour: a masked object reads as "no image here", not background
SPLIT_PRIORITY = {"test": 2, "valid": 1, "train": 0}  # a duplicate group pinned to several splits joins the highest
ORIENT_SWAPS = {5, 6, 7, 8}  # EXIF orientations that swap width and height
ASPECT_TOL = 0.02  # recorded vs actual image size: same aspect ratio within 2%
SKIP_DIRS = {"images", "labels", "__pycache__", "node_modules", "site-packages"}
FORMATS = ("yolo", "coco", "supervisely", "labelme")
PIXEL_FORMATS = ("coco", "supervisely", "labelme")  # boxes in pixels of a recorded image size
SOURCE_KEYS = {"name", "enabled", "format", "locate", "locate_match", "names_file", "annotations", "images_dir",
               "class_field", "box_frame", "class_map", "allow_unused_map_keys", "class_from_tag", "tag_overrides",
               "untagged", "domain", "licence", "kaggle", "optional", "keep_split_csv", "force_split", "subsample",
               "max_images", "keep_empty"}
TOP_KEYS = {"classes", "sources"}
SLUG = re.compile(r"^[A-Za-z0-9_-]+$")
OUTPUT_FILES = ("split.csv", "build_report.json", "labels.cache")
CONTENT_DEPTH = 4  # how deep below a located folder its ann/ folders or json files are looked for


class BuildError(Exception):
    """A problem with the config or the inputs; the message says what to fix."""


class SourceMissing(BuildError):
    """A source's files were not found under the search root (skipped with a warning if the source is optional)."""


# ---------------------------------------------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------------------------------------------
def as_list(value):
    return [] if value is None else [value] if isinstance(value, str) else list(value)


def load_config(path):
    """Read and check the sources yaml. Returns (classes, enabled sources as dicts with defaults filled in)."""
    path = Path(path)
    if not path.is_file():
        raise BuildError(f"Sources config {path} not found")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = set(cfg) - TOP_KEYS
    if unknown:
        raise BuildError(f"{path}: unknown top-level keys {sorted(unknown)}; allowed: {sorted(TOP_KEYS)}")
    classes = [str(c) for c in cfg.get("classes") or []]
    if not classes or len(set(classes)) != len(classes):
        raise BuildError(f"{path}: 'classes' must be a non-empty list of unique target class names")
    if set(DROP_TARGETS) & set(classes):
        raise BuildError(f"{path}: {list(DROP_TARGETS)} cannot be class names")
    sources, seen = [], set()
    for i, src in enumerate(cfg.get("sources") or []):
        where = f"{path}: source #{i + 1}"
        if not isinstance(src, dict):
            raise BuildError(f"{where} must be a mapping")
        unknown = set(src) - SOURCE_KEYS
        if unknown:
            raise BuildError(f"{where}: unknown keys {sorted(unknown)}; allowed: {sorted(SOURCE_KEYS)}")
        name = str(src.get("name", ""))
        if not SLUG.match(name):
            raise BuildError(f"{where}: 'name' must use only letters, digits, '_' and '-' (got '{name}')")
        if name.lower() in seen:
            raise BuildError(f"{path}: source name '{name}' is used twice")
        seen.add(name.lower())
        if not src.get("enabled", True):
            continue  # disabled sources may hold unfinished placeholders
        sources.append(check_source(src, name, f"{path}: source '{name}'", classes))
    if not sources:
        raise BuildError(f"{path}: no enabled sources")
    return classes, sources


def check_targets(mapping, classes, where, what):
    bad = sorted({v for v in mapping.values() if v not in classes and v not in DROP_TARGETS})
    if bad:
        raise BuildError(f"{where}: {what} targets {bad} are not in classes {classes} (or {', '.join(DROP_TARGETS)})")


def is_pos_int(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 1


def check_source(src, name, where, classes):
    """Validate one enabled source entry and return it with defaults filled in."""
    fmt = src.get("format")
    s = {"name": name, "format": fmt,
         "locate": [str(v) for v in as_list(src.get("locate"))], "locate_match": src.get("locate_match", "exact"),
         "names_file": src.get("names_file"), "annotations": src.get("annotations"),
         "images_dir": src.get("images_dir"), "class_field": src.get("class_field", "name"),
         "box_frame": src.get("box_frame", "auto"), "class_map": src.get("class_map"),
         "allow_unused_map_keys": bool(src.get("allow_unused_map_keys", False)),
         "class_from_tag": src.get("class_from_tag"), "tag_overrides": src.get("tag_overrides") or {},
         "untagged": src.get("untagged"), "domain": str(src.get("domain", "")), "licence": src.get("licence"),
         "kaggle": src.get("kaggle"), "optional": bool(src.get("optional", False)),
         "keep_split_csv": src.get("keep_split_csv"), "force_split": src.get("force_split"),
         "subsample": src.get("subsample"), "max_images": src.get("max_images"),
         "keep_empty": bool(src.get("keep_empty", False))}
    if fmt not in FORMATS:
        raise BuildError(f"{where}: 'format' must be one of {', '.join(FORMATS)} (got {fmt!r})")
    if fmt != "coco" and not s["locate"]:
        raise BuildError(f"{where}: a {fmt} source needs 'locate' (its folder name)")
    if fmt == "coco" and not s["annotations"]:
        raise BuildError(f"{where}: a coco source needs 'annotations' (the json file name)")
    misplaced = [k for k, fmts in (("names_file", ("yolo",)), ("annotations", ("coco",)), ("images_dir", ("coco",)),
                                   ("class_field", ("coco",)), ("box_frame", PIXEL_FORMATS),
                                   ("class_from_tag", ("supervisely",)), ("tag_overrides", ("supervisely",)),
                                   ("untagged", ("supervisely",))) if k in src and fmt not in fmts]
    if misplaced:
        raise BuildError(f"{where}: {misplaced} cannot be used with format {fmt}")
    if s["locate_match"] not in ("exact", "contains"):
        raise BuildError(f"{where}: 'locate_match' must be exact or contains")
    if s["class_field"] not in ("name", "supercategory"):
        raise BuildError(f"{where}: 'class_field' must be name or supercategory")
    if s["box_frame"] not in ("auto", "display", "raw"):
        raise BuildError(f"{where}: 'box_frame' must be auto, display or raw")
    if not SLUG.match(s["domain"]):
        raise BuildError(f"{where}: 'domain' is required (e.g. studio or real_world), letters/digits/_/- only")
    if not isinstance(s["licence"], str) or not s["licence"].strip():
        raise BuildError(f"{where}: 'licence' is required (a string, e.g. 'CC BY 4.0')")
    if s["kaggle"] is not None and not isinstance(s["kaggle"], str):
        raise BuildError(f"{where}: 'kaggle' must be a string (the Kaggle dataset to attach, e.g. owner/slug)")
    if not isinstance(s["class_map"], dict) or not s["class_map"]:
        raise BuildError(f"{where}: 'class_map' must map each source class to a target class or one of "
                         f"{', '.join(DROP_TARGETS)}")
    s["class_map"] = {str(k): str(v) for k, v in s["class_map"].items()}
    check_targets(s["class_map"], classes, where, "class_map")
    if fmt == "supervisely":
        if s["class_from_tag"] is not None:
            if not isinstance(s["class_from_tag"], str) or not s["class_from_tag"]:
                raise BuildError(f"{where}: 'class_from_tag' must be a tag name (e.g. Material)")
            if s["untagged"] is None:
                raise BuildError(f"{where}: with class_from_tag, 'untagged' must say what an object without that "
                                 f"tag becomes (a target class or one of {', '.join(DROP_TARGETS)})")
        elif s["untagged"] is not None:
            raise BuildError(f"{where}: 'untagged' needs 'class_from_tag'")
        if s["untagged"] is not None:
            s["untagged"] = str(s["untagged"])
            check_targets({"untagged": s["untagged"]}, classes, where, "untagged")
        if not isinstance(s["tag_overrides"], dict) or not all(isinstance(v, dict) and v
                                                                for v in s["tag_overrides"].values()):
            raise BuildError(f"{where}: 'tag_overrides' must map a tag name to {{tag value: target}}")
        s["tag_overrides"] = {str(t): {str(k): str(v) for k, v in m.items()} for t, m in s["tag_overrides"].items()}
        for t, m in s["tag_overrides"].items():
            check_targets(m, classes, where, f"tag_overrides.{t}")
    if s["max_images"] is not None and not is_pos_int(s["max_images"]):
        raise BuildError(f"{where}: 'max_images' must be a positive integer")
    if s["force_split"] is not None:
        if s["force_split"] not in SPLITS:
            raise BuildError(f"{where}: 'force_split' must be one of {SPLITS}")
        if s["keep_split_csv"]:
            raise BuildError(f"{where}: use either 'force_split' or 'keep_split_csv', not both")
    sub = s["subsample"]
    if sub is not None:
        if not isinstance(sub, dict) or not sub:
            raise BuildError(f"{where}: 'subsample' must be a mapping like {{train: 3000, valid: 900, seed: 42}}")
        if "test" in sub:
            raise BuildError(f"{where}: 'subsample' cannot include test: test images are never subsampled")
        unknown = set(sub) - {"train", "valid", "seed"}
        if unknown:
            raise BuildError(f"{where}: unknown 'subsample' keys {sorted(unknown)}; allowed: train, valid, seed")
        if not all(is_pos_int(sub[k]) for k in ("train", "valid") if k in sub) or not {"train", "valid"} & set(sub):
            raise BuildError(f"{where}: 'subsample' needs train and/or valid as positive integers")
        if "seed" in sub and (not isinstance(sub["seed"], int) or isinstance(sub["seed"], bool)):
            raise BuildError(f"{where}: 'subsample.seed' must be an integer")
        if not s["keep_split_csv"]:
            raise BuildError(f"{where}: 'subsample' works on a source pinned by keep_split_csv")
    return s


def check_class_map(src, source_names):
    """Every source class must be mapped and every mapped name must exist in the source (unless the source allows
    unused names, e.g. spelling variants of a class whose exact spelling is not known in advance)."""
    cmap = src["class_map"]
    missing = [n for n in source_names if n not in cmap]
    extra = [] if src["allow_unused_map_keys"] else sorted(set(cmap) - set(source_names))
    problems = []
    if missing:
        problems.append(f"unmapped classes {missing}")
    if extra:
        problems.append(f"class_map names not in the source {extra}")
    if problems:
        return f"source '{src['name']}': " + "; ".join(problems) + f" (source classes: {list(source_names)})"
    return None


# ---------------------------------------------------------------------------------------------------------------
# Locating sources under --search-root (Kaggle nests datasets as /kaggle/input/datasets/<owner>/<slug>/...)
# ---------------------------------------------------------------------------------------------------------------
def norm_name(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def name_matches(name, wanted, mode="exact"):
    """wanted: normalized names. exact: the folder's normalized name is one of them; contains: it contains one."""
    n = norm_name(name)
    return n in wanted if mode == "exact" else any(w and w in n for w in wanted)


def missing_hint(src):
    """The end of a 'not found' message: which Kaggle dataset to attach."""
    if src.get("kaggle"):
        return f". Attach the Kaggle dataset '{src['kaggle']}' to the notebook (or fix --search-root / locate)"
    return ". Check --search-root and the source's locate setting"


def walk_dirs(start, skip=()):
    """Folders at depth 0..MAX_ROOT_DEPTH below start, breadth first, sorted; image/label folders and hidden
    folders are not entered."""
    skip = {Path(p).resolve() for p in skip}
    level = [start]
    for _ in range(MAX_ROOT_DEPTH + 1):
        next_level = []
        for folder in level:
            yield folder
            try:
                children = sorted(d for d in folder.iterdir() if d.is_dir())
            except OSError:
                continue
            next_level += [d for d in children if not d.name.startswith(".") and d.name.lower() not in SKIP_DIRS
                           and d.resolve() not in skip]
        level = next_level


def has_yolo_images(folder):
    if (folder / "images").is_dir():
        return True
    try:
        return any((d / "images").is_dir() for d in folder.iterdir() if d.is_dir())
    except OSError:
        return False


def outermost(paths):
    """Drop paths that lie inside another path of the list."""
    paths = sorted(set(paths))
    return [p for p in paths if not any(q != p and p.is_relative_to(q) for q in paths)]


def dirs_below(folder, depth):
    """folder and its subfolders down to `depth` levels, breadth first, sorted; hidden folders skipped."""
    level = [folder]
    for _ in range(depth + 1):
        next_level = []
        for d in level:
            yield d
            try:
                next_level += sorted(c for c in d.iterdir() if c.is_dir() and not c.name.startswith("."))
            except OSError:
                continue
        level = next_level


def has_supervisely(folder):
    """A Supervisely dataset (ann/ next to img/) at or below folder."""
    return any(d.name == "ann" and (d.parent / "img").is_dir() for d in dirs_below(folder, CONTENT_DEPTH))


def has_json(folder):
    for d in dirs_below(folder, CONTENT_DEPTH):
        try:
            if any(p.suffix.lower() == ".json" and p.is_file() for p in d.iterdir()):
                return True
        except OSError:
            continue
    return False


def locate_named(src, search_root, skip, ok, what):
    """The one outermost folder under search_root whose name matches src['locate'] and that passes ok()."""
    wanted = {norm_name(n) for n in src["locate"]}
    for n in src["locate"]:  # an explicit path wins
        p = Path(n) if Path(n).is_absolute() else search_root / n
        if p.is_dir() and ok(p):
            return p.resolve()
    found = outermost([d for d in walk_dirs(search_root, skip)
                       if name_matches(d.name, wanted, src["locate_match"]) and ok(d)])
    if not found:
        raise SourceMissing(f"source '{src['name']}': no folder named {src['locate']}"
                            f"{' (or containing one of these names)' if src['locate_match'] == 'contains' else ''}"
                            f" with {what} under {search_root}" + missing_hint(src))
    if len(found) > 1:
        raise BuildError(f"source '{src['name']}': several folders named {src['locate']} with {what} under "
                         f"{search_root}:\n  " + "\n  ".join(map(str, found)))
    return found[0].resolve()


def locate_yolo(src, search_root, skip=()):
    return locate_named(src, search_root, skip, has_yolo_images, "images/ subfolders")


def locate_coco(src, search_root, skip=()):
    rel = Path(src["annotations"])
    if rel.is_absolute():
        if not rel.is_file():
            raise SourceMissing(f"source '{src['name']}': annotations file {rel} not found" + missing_hint(src))
        return rel.resolve()
    wanted = {norm_name(n) for n in src["locate"]}
    found = []
    for d in walk_dirs(search_root, skip):
        if (d / rel).is_file() and (not wanted or any(name_matches(p.name, wanted, src["locate_match"])
                                                      for p in [d, *d.parents])):
            found.append((d / rel).resolve())
    found = sorted(set(found))
    where = f"'{rel.as_posix()}'{' inside ' + str(src['locate']) if wanted else ''} under {search_root}"
    if not found:
        raise SourceMissing(f"source '{src['name']}': no annotation file {where}" + missing_hint(src))
    if len(found) > 1:
        raise BuildError(f"source '{src['name']}': several annotation files {where}:\n  "
                         + "\n  ".join(map(str, found)))
    return found[0]


def resolve_repo_path(p, config_dir):
    p = Path(p)
    if p.is_absolute():
        return p
    for base in (REPO, config_dir):
        if (base / p).exists():
            return (base / p).resolve()
    return (REPO / p).resolve()


# ---------------------------------------------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------------------------------------------
def orient_point(x, y, orientation):
    """Map a normalized point of the stored (raw) image to the image after ImageOps.exif_transpose."""
    return {1: (x, y), 2: (1 - x, y), 3: (1 - x, 1 - y), 4: (x, 1 - y),
            5: (y, x), 6: (1 - y, x), 7: (1 - y, 1 - x), 8: (y, 1 - x)}.get(orientation, (x, y))


def same_aspect(a, b):
    p, q = a[0] * b[1], a[1] * b[0]
    return abs(p - q) <= ASPECT_TOL * max(p, q)


def resolve_frame(recorded, raw_size, orientation, box_frame):
    """Which image the COCO boxes refer to: 'display' (after EXIF rotation) or 'raw' (stored pixels).

    Returns (frame, warnings). auto: without a rotation both are the same; with a 90-degree rotation the recorded
    width/height tell which one was annotated; when they cannot tell (square image, 180-degree turn or mirror)
    'display' is assumed, as TACO annotates the EXIF-rotated photo. Raises ValueError when the recorded size has
    another aspect ratio than the image it refers to.
    """
    warnings = Counter()
    swapped = orientation in ORIENT_SWAPS
    display = (raw_size[1], raw_size[0]) if swapped else tuple(raw_size)
    frame = box_frame
    if frame == "auto":
        if orientation in (None, 1) or orientation not in range(1, 9):
            frame = "display"
        else:
            m_disp, m_raw = same_aspect(recorded, display), same_aspect(recorded, raw_size)
            if m_disp and not m_raw:
                frame = "display"
            elif m_raw and not m_disp:
                frame = "raw"
            else:
                frame = "display"
                warnings["orientation_frame_assumed_display"] += 1
    expected = display if frame == "display" else tuple(raw_size)
    if not same_aspect(recorded, expected):
        raise ValueError(f"recorded size {recorded[0]}x{recorded[1]} does not match the {frame} image "
                         f"{expected[0]}x{expected[1]}")
    if tuple(recorded) != tuple(expected):
        warnings["recorded_size_rescaled"] += 1  # same aspect, other scale: normalized boxes are unaffected
    return frame, warnings


def coco_to_yolo(anns, recorded, raw_size, orientation, box_frame):
    """COCO [x, y, w, h] pixel boxes -> YOLO (cls, cx, cy, w, h) normalized to the EXIF-transposed image.

    anns: (cls, bbox) pairs whose class is already a target index. Boxes are normalized by the recorded size (so a
    recorded size that only differs in scale is harmless), rotated if they refer to the raw pixels, and clipped.
    Returns (boxes, warnings Counter).
    """
    frame, warnings = resolve_frame(recorded, raw_size, orientation, box_frame)
    W, H = recorded
    boxes = []
    for cls, bbox in anns:
        try:
            x, y, w, h = (float(v) for v in bbox)
        except (TypeError, ValueError):
            warnings["invalid_bbox"] += 1
            continue
        if not all(np.isfinite([x, y, w, h])) or w <= 0 or h <= 0:
            warnings["invalid_bbox"] += 1
            continue
        x0, y0, x1, y1 = x / W, y / H, (x + w) / W, (y + h) / H
        if frame == "raw" and orientation in range(2, 9):
            pts = [orient_point(px, py, orientation) for px, py in ((x0, y0), (x1, y1))]
            x0, x1 = sorted(p[0] for p in pts)
            y0, y1 = sorted(p[1] for p in pts)
        c = [min(max(v, 0.0), 1.0) for v in (x0, y0, x1, y1)]
        if c[2] - c[0] <= 0 or c[3] - c[1] <= 0:
            warnings["box_outside_image"] += 1
            continue
        if max(abs(a - b) for a, b in zip(c, (x0, y0, x1, y1))) > 1e-6:
            warnings["clipped_boxes"] += 1
        boxes.append((cls, (c[0] + c[2]) / 2, (c[1] + c[3]) / 2, c[2] - c[0], c[3] - c[1]))
    return boxes, warnings


def clip_yolo(box):
    """Clip a normalized YOLO box to the image; None if nothing is left."""
    cls, cx, cy, w, h = box
    x0, y0, x1, y1 = max(cx - w / 2, 0.0), max(cy - h / 2, 0.0), min(cx + w / 2, 1.0), min(cy + h / 2, 1.0)
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    return cls, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0


def dominant_class(boxes, classes):
    if not boxes:
        return "background"
    counts = Counter(b[0] for b in boxes)
    return classes[max(counts, key=lambda c: (counts[c], -c))]


# ---------------------------------------------------------------------------------------------------------------
# Reading sources into items (one item = one source image with its mapped annotations)
# ---------------------------------------------------------------------------------------------------------------
def sanitize(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "img"


def label_path_for(img_path):
    """Ultralytics' rule: the last 'images' folder becomes 'labels', the suffix becomes .txt."""
    parts = list(img_path.parts)
    idx = max(i for i, p in enumerate(parts) if p == "images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_yolo(src, root, classes, stats):
    if src["names_file"]:
        names_path = Path(src["names_file"]) if Path(src["names_file"]).is_absolute() else root / src["names_file"]
        if not names_path.is_file():
            raise BuildError(f"source '{src['name']}': names_file {names_path} not found")
        names = yaml.safe_load(names_path.read_text(encoding="utf-8"))["names"]
        names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    else:
        try:
            names = load_names(root)
        except SystemExit as e:
            raise BuildError(f"source '{src['name']}': {e}") from None
    names = [str(n) for n in names]
    images = []
    for d in sorted(p for p in root.rglob("images") if p.is_dir()):
        images += [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    items = []
    for p in sorted(set(images)):
        items.append({"source": src["name"], "uid": p.relative_to(root).as_posix(), "path": p,
                      "label": label_path_for(p)})
    return names, items


def map_yolo_item(item, src, names, classes, stats):
    """Parse the item's label file and map its classes. Returns False if the image is dropped."""
    target_idx = {c: i for i, c in enumerate(classes)}
    if not item["label"].is_file():
        stats["images_dropped"]["missing_label_file"] += 1
        return False
    boxes, issues = parse_label(item["label"], len(names))
    for k, v in issues.items():
        stats["label_issues"][k] += v
    mapped, masks = [], []
    for cls, cx, cy, w, h in boxes:
        target = src["class_map"][names[cls]]
        if target == DROP_IMG:
            stats["images_dropped"][f"{DROP_IMG}:{names[cls]}"] += 1
            return False
        if target in (DROP_ANN, DROP_MASK):
            stats["annotations_dropped"][f"{target}:{names[cls]}"] += 1
            if target == DROP_MASK:
                masks.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
            continue
        box = clip_yolo((target_idx[target], cx, cy, w, h))
        if box is None:
            stats["annotations_dropped"]["box_outside_image"] += 1
            continue
        if max(abs(a - b) for a, b in zip(box[1:], (cx, cy, w, h))) > 1e-6:
            stats["warnings"]["clipped_boxes"] += 1
        mapped.append(box)
    item["boxes"], item["mask_boxes"], item["had_boxes"] = mapped, masks, bool(boxes)
    return True


def read_coco(src, ann_path, classes, stats):
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    cats = {c["id"]: str(c.get(src["class_field"]) or c["name"]) for c in data.get("categories", [])}
    names = sorted(set(cats.values()))
    img_dir = (ann_path.parent / src["images_dir"]).resolve() if src["images_dir"] else ann_path.parent
    if not img_dir.is_dir():
        raise BuildError(f"source '{src['name']}': images_dir {img_dir} not found")
    anns = defaultdict(list)
    for a in data.get("annotations", []):
        anns[a.get("image_id")].append(a)
    items = []
    for im in data.get("images", []):
        p = img_dir / im["file_name"]
        items.append({"source": src["name"], "uid": Path(im["file_name"]).as_posix(), "path": p,
                      "recorded": (im.get("width"), im.get("height")), "anns": anns.get(im["id"], []),
                      "cats": cats})
    items.sort(key=lambda it: it["uid"])
    return names, items, img_dir


def map_coco_item(item, src, classes, stats):
    target_idx = {c: i for i, c in enumerate(classes)}
    mapped, masks = [], []
    for a in item["anns"]:
        name = item["cats"].get(a.get("category_id"))
        if name is None:
            stats["annotations_dropped"]["unknown_category_id"] += 1
            continue
        target = src["class_map"][name]
        if target == DROP_IMG:
            stats["images_dropped"][f"{DROP_IMG}:{name}"] += 1
            return False
        if a.get("iscrowd"):
            stats["annotations_dropped"]["crowd"] += 1
            continue
        if target in (DROP_ANN, DROP_MASK):
            stats["annotations_dropped"][f"{target}:{name}"] += 1
            if target == DROP_MASK:
                masks.append((-1, a.get("bbox")))
            continue
        mapped.append((target_idx[target], a.get("bbox")))
    # coco_anns / mask_anns: (class, [x, y, w, h]) in pixels of the recorded image size; shared by every pixel format
    item["coco_anns"], item["mask_anns"], item["had_boxes"] = mapped, masks, bool(item["anns"])
    if not item["path"].is_file():
        stats["images_dropped"]["missing_image_file"] += 1
        return False
    return True


def points_bbox(points):
    """[x, y, w, h] around a list of [x, y] points, or None if there are fewer than two valid points."""
    try:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if len(pts) < 2 or not np.isfinite(pts).all():
        return None
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def find_image(folder, name, stem=None):
    """folder/name, or a file in folder with the same stem and an image extension."""
    if name and (folder / name).is_file():
        return folder / name
    stem = stem if stem is not None else Path(name).stem
    try:
        hits = sorted(p for p in folder.iterdir() if p.is_file() and p.stem == stem and p.suffix.lower() in IMG_EXTS)
    except OSError:
        return None
    return hits[0] if hits else None


def tag_value(tag):
    """(name, value) of a Supervisely tag; tags are {"name": ..., "value": ...} (older exports: plain strings)."""
    if isinstance(tag, dict):
        return str(tag.get("name", "")), None if tag.get("value") is None else str(tag.get("value"))
    return str(tag), None


def read_supervisely(src, root, classes, stats):
    """A Supervisely project: <dataset>/ann/<image>.json next to <dataset>/img/<image>, and meta.json.

    Each object's class: the first tag_overrides hit (tag -> value -> target), else the class_from_tag tag's value
    looked up in class_map (no such tag: 'untagged'), else (without class_from_tag) the object's classTitle.
    Returns (source class names for check_class_map, items).
    """
    ann_dirs = sorted(d for d in root.rglob("ann") if d.is_dir() and (d.parent / "img").is_dir())
    if not ann_dirs:
        raise SourceMissing(f"source '{src['name']}': no Supervisely ann/ + img/ folders in {root}" + missing_hint(src))
    tag_name, overrides = src["class_from_tag"], src["tag_overrides"]
    meta_values = defaultdict(set)  # tag name -> values declared in meta.json
    meta_classes = set()
    metas = [p for p in sorted(root.rglob("meta.json")) if p.parent.name not in ("ann", "img")]
    for p in metas:
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for t in meta.get("tags", []) if isinstance(meta, dict) else []:
            meta_values[str(t.get("name"))].update(str(v) for v in t.get("values") or [])
        meta_classes.update(str(c.get("title")) for c in (meta.get("classes", []) if isinstance(meta, dict) else []))
    if not metas:
        stats["warnings"]["supervisely_meta_json_not_found"] += 1
    seen = set()
    items = []
    for ann_dir in ann_dirs:
        img_dir = ann_dir.parent / "img"
        for ap in sorted(p for p in ann_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"):
            try:
                ann = json.loads(ap.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                stats["images_dropped"]["unreadable_annotation_json"] += 1
                continue
            img_name = ap.name[:-5]  # Supervisely names the json <image file name>.json
            img = find_image(img_dir, img_name, Path(img_name).stem if Path(img_name).suffix else img_name)
            size = ann.get("size") or {}
            objects = []
            for obj in ann.get("objects", []):
                geom = obj.get("geometryType", "")
                bbox = points_bbox((obj.get("points") or {}).get("exterior")) if geom in ("rectangle", "polygon") \
                    else None
                problem = None if bbox else (f"unsupported_geometry:{geom}" if geom not in ("rectangle", "polygon")
                                             else "invalid_bbox")
                tags = dict(tag_value(t) for t in obj.get("tags", []))
                override = next(((f"{t}={tags[t]}", m[tags[t]]) for t, m in overrides.items()
                                 if tags.get(t) in m), None)
                name = None
                if override is None:
                    name = tags.get(tag_name) if tag_name else str(obj.get("classTitle", ""))
                    if name is not None:
                        seen.add(name)
                    else:
                        stats["warnings"][f"objects_without_{tag_name}_tag"] += 1
                objects.append({"name": name, "override": override, "bbox": bbox, "problem": problem})
            path = img or img_dir / img_name
            uid = (ann_dir.parent.relative_to(root) / path.name).as_posix()  # <dataset>/<image>, without img/
            items.append({"source": src["name"], "uid": uid, "path": path,
                          "recorded": (size.get("width"), size.get("height")), "objects": objects})
    for t, m in overrides.items():  # a typo in an override value would silently never match
        unknown = sorted(set(m) - meta_values[t]) if meta_values.get(t) else []
        if unknown:
            raise BuildError(f"source '{src['name']}': tag_overrides.{t} values {unknown} are not values of tag "
                             f"'{t}' in meta.json ({sorted(meta_values[t])})")
    declared = meta_values.get(tag_name, set()) if tag_name else meta_classes
    items.sort(key=lambda it: it["uid"])
    return sorted(seen | declared), items


def label_lookup(src):
    """Loose class-name matching for Labelme labels: case, spaces and punctuation are ignored, and a trailing
    instance number ('bottle 2') is tried without the number. Returns a function label -> class_map key (or the
    label itself when nothing matches, so check_class_map reports it as unmapped)."""
    keys = {norm_name(k): k for k in src["class_map"]}

    def lookup(label):
        n = norm_name(label)
        return keys.get(n) or keys.get(re.sub(r"\d+$", "", n)) or label
    return lookup


def read_labelme(src, root, classes, stats):
    """Labelme: one json per image with 'shapes' (polygon, rectangle or circle) and 'imagePath'. Each shape becomes
    one box (its bounding box). The image is found from imagePath (relative to the json), else by the json's stem,
    else decoded from the embedded imageData. Returns (source class names, items)."""
    lookup = label_lookup(src)
    seen, items, by_image = set(), [], {}
    for jp in sorted(root.rglob("*.json")):
        try:
            ann = json.loads(jp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            stats["warnings"]["unreadable_json_skipped"] += 1
            continue
        if not isinstance(ann, dict) or not isinstance(ann.get("shapes"), list):
            stats["warnings"]["non_labelme_json_skipped"] += 1
            continue
        ipath = str(ann.get("imagePath") or "").replace("\\", "/")
        img = None
        if ipath:
            cand = (jp.parent / ipath)
            img = cand if cand.is_file() else find_image(jp.parent, Path(ipath).name, Path(ipath).stem)
        img = img or find_image(jp.parent, "", jp.stem)
        image_bytes = None
        if img is None and ann.get("imageData"):
            try:
                image_bytes = base64.b64decode(ann["imageData"])
                stats["warnings"]["image_from_embedded_imageData"] += 1
            except (ValueError, TypeError):
                image_bytes = None
        if img is not None:
            img = img.resolve()
            if img in by_image:
                stats["warnings"]["second_json_for_same_image_skipped"] += 1
                continue
            by_image[img] = jp
        objects = []
        for sh in ann["shapes"]:
            kind = sh.get("shape_type") or "polygon"
            pts = sh.get("points")
            if kind == "circle":  # centre and one point on the circle
                try:
                    (cx, cy), (px, py) = pts[:2]
                    r = float(np.hypot(px - cx, py - cy))
                    pts = [[cx - r, cy - r], [cx + r, cy + r]]
                except (TypeError, ValueError):
                    pts = None
            bbox = points_bbox(pts) if kind in ("polygon", "rectangle", "circle") else None
            problem = None if bbox else (f"unsupported_shape:{kind}" if kind not in ("polygon", "rectangle", "circle")
                                         else "invalid_bbox")
            name = lookup(str(sh.get("label", "")))
            seen.add(name)
            objects.append({"name": name, "override": None, "bbox": bbox, "problem": problem})
        path = img if img is not None else jp.parent / (Path(ipath).name if ipath else jp.stem + ".jpg")
        try:
            uid = path.resolve().relative_to(root).as_posix()
        except ValueError:
            uid = jp.relative_to(root).with_suffix(Path(path).suffix or ".jpg").as_posix()
        items.append({"source": src["name"], "uid": uid, "path": path, "image_bytes": image_bytes,
                      "recorded": (ann.get("imageWidth"), ann.get("imageHeight")), "objects": objects})
    if not items:
        raise SourceMissing(f"source '{src['name']}': no Labelme json files (with 'shapes') in {root}"
                            + missing_hint(src))
    items.sort(key=lambda it: it["uid"])
    return sorted(seen), items


def map_objects_item(item, src, classes, stats):
    """Map a Supervisely or Labelme item's objects to target classes. Returns False if the image is dropped."""
    target_idx = {c: i for i, c in enumerate(classes)}
    mapped, masks = [], []
    for obj in item["objects"]:
        if obj["override"]:
            label, target = obj["override"]
        elif obj["name"] is None:
            label, target = "untagged", src["untagged"]
        else:
            label, target = obj["name"], src["class_map"][obj["name"]]
        if target == DROP_IMG:
            stats["images_dropped"][f"{DROP_IMG}:{label}"] += 1
            return False
        if obj["bbox"] is None:
            stats["annotations_dropped"][obj["problem"]] += 1
            continue
        if target in (DROP_ANN, DROP_MASK):
            stats["annotations_dropped"][f"{target}:{label}"] += 1
            if target == DROP_MASK:
                masks.append((-1, obj["bbox"]))
            continue
        mapped.append((target_idx[target], obj["bbox"]))
    item["coco_anns"], item["mask_anns"], item["had_boxes"] = mapped, masks, bool(item["objects"])
    if item.get("image_bytes") is None and not Path(item["path"]).is_file():
        stats["images_dropped"]["missing_image_file"] += 1
        return False
    return True


# ---------------------------------------------------------------------------------------------------------------
# Writing images
# ---------------------------------------------------------------------------------------------------------------
def paint_masks(img, boxes):
    """Paint normalized (x0, y0, x1, y1) regions of an RGB image mid-grey, in place, covering every touched pixel."""
    W, H = img.size
    draw = ImageDraw.Draw(img)
    for x0, y0, x1, y1 in boxes:
        l, t = max(0, int(np.floor(x0 * W))), max(0, int(np.floor(y0 * H)))
        r, b = min(W, int(np.ceil(x1 * W))), min(H, int(np.ceil(y1 * H)))
        if r > l and b > t:
            draw.rectangle([l, t, r - 1, b - 1], fill=MASK_GREY)


def process_item(item, src, out_dir, max_side):
    """Decode, EXIF-transpose, (for COCO) convert boxes, downscale, hash and write one image and its label.

    Returns (record or None, drop reason or None, warnings Counter).
    """
    warnings = Counter()
    try:
        data = item["image_bytes"] if item.get("image_bytes") is not None else item["path"].read_bytes()
        with Image.open(io.BytesIO(data)) as im:
            raw_size = im.size
            orientation = im.getexif().get(0x0112, 1)
            orientation = orientation if orientation in range(1, 9) else 1
            scale = min(1.0, max_side / max(raw_size))
            if scale < 0.5:  # let the JPEG decoder skip detail we would throw away anyway
                im.draft("RGB", (max(1, int(raw_size[0] * scale)), max(1, int(raw_size[1] * scale))))
            im.load()
            img = ImageOps.exif_transpose(im)
            img = img.convert("RGB")
    except Exception as e:
        return None, "unreadable_image", Counter({f"unreadable: {type(e).__name__}": 1})
    if orientation != 1:
        warnings["exif_rotated_images"] += 1
    if "coco_anns" in item:
        rec_w, rec_h = item["recorded"]
        if not rec_w or not rec_h:
            rec_w, rec_h = (raw_size[1], raw_size[0]) if orientation in ORIENT_SWAPS else raw_size
            warnings["recorded_size_missing"] += 1
        try:
            boxes, w2 = coco_to_yolo(item["coco_anns"], (rec_w, rec_h), raw_size, orientation, src["box_frame"])
            masks, _ = coco_to_yolo(item.get("mask_anns", []), (rec_w, rec_h), raw_size, orientation,
                                    src["box_frame"])
        except ValueError:
            return None, "recorded_size_mismatch", warnings
        warnings.update(w2)
        masks = [(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2) for _, cx, cy, w, h in masks]
    else:
        boxes, masks = item["boxes"], item.get("mask_boxes", [])
    if not boxes and (item["had_boxes"] or not src["keep_empty"]):
        return None, "no_boxes_left" if item["had_boxes"] else "no_annotations", warnings
    if masks:
        paint_masks(img, masks)
        warnings["masked_regions"] += len(masks)
    disp = (raw_size[1], raw_size[0]) if orientation in ORIENT_SWAPS else raw_size
    if scale < 1.0:
        target = (max(1, round(disp[0] * scale)), max(1, round(disp[1] * scale)))
        img = img.resize(target, Image.Resampling.LANCZOS)
        warnings["downscaled_images"] += 1
    elif img.size != tuple(disp):
        img = img.resize(disp, Image.Resampling.LANCZOS)
    gray = img.convert("L")
    bits = dhash_bits(gray, DHASH_SIZE)
    hashed = bool(bits.any() and not bits.all())
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    (out_dir / "images" / item["file"]).write_bytes(buf.getvalue())
    (out_dir / "labels" / item["file"]).with_suffix(".txt").write_text(
        "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in boxes))
    src_name = item["path"].name
    rec = {"name": item["file"], "file": item["file"], "source": item["source"], "uid": item["uid"],
           "boxes": boxes, "size": img.size, "md5": hashlib.md5(data).hexdigest(),
           # Roboflow copies share a stem; only within one source, as generic stems could collide across sources
           "stem": f"{item['source']}/{src_name.split('.rf.')[0]}" if ".rf." in src_name else None,
           "dhash256": bits if hashed else None,
           "thumb": np.asarray(gray.resize((THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS), dtype=np.uint8)
           if hashed else None,
           "near_dup": None, "pinned": item.get("pinned")}
    return rec, None, warnings


def assign_file_names(items):
    """<source>__<name>.jpg, unique even on case-insensitive file systems; colliding names get a path hash."""
    for it in items:
        base = Path(it["uid"]).stem if it["format"] == "yolo" else str(Path(it["uid"]).with_suffix(""))
        it["file"] = f"{it['source']}__{sanitize(base)}.jpg"
    counts = Counter(it["file"].lower() for it in items)
    for it in items:
        if counts[it["file"].lower()] > 1:
            tag = hashlib.md5(it["uid"].encode()).hexdigest()[:8]
            it["file"] = it["file"][:-4] + f"_{tag}.jpg"
    if len({it["file"].lower() for it in items}) != len(items):
        raise BuildError("Could not make unique output file names")


# ---------------------------------------------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------------------------------------------
def split_records(recs, classes, seed):
    """Assign rec['split'] for every record; returns (groups, excluded records, pin conflicts).

    Pinned records (keep_split_csv) keep their split. The rest of a pinned record's duplicate group joins the
    highest-priority pinned split (test > valid > train). Free groups are split 70/20/10 within each
    (source, dominant class) stratum, counting the pinned members of that stratum. If one group ends up in two
    splits (only possible when pinned images disagree), the lower-priority copies are excluded, as in Gen 1.
    """
    recs = sorted(recs, key=lambda r: r["file"])
    groups = sorted((sorted(g, key=lambda r: r["file"]) for g in duplicate_groups(recs)),
                    key=lambda g: g[0]["file"])
    stratum = {r["file"]: f"{r['source']}/{r['dominant_class']}" for r in recs}
    totals, counts, free = Counter(), defaultdict(Counter), defaultdict(list)
    conflicts = 0
    for g in groups:
        pins = {r["pinned"] for r in g if r["pinned"]}
        if pins:
            conflicts += len(pins) > 1
            target = max(pins, key=SPLIT_PRIORITY.get)
            for r in g:
                r["split"] = r["pinned"] or target
                counts[stratum[r["file"]]][r["split"]] += 1
                totals[stratum[r["file"]]] += 1
        else:
            s = stratum[g[0]["file"]]
            free[s].append(g)
            totals[s] += len(g)
    rng = random.Random(seed)
    for s in sorted(free):
        gs = free[s]
        rng.shuffle(gs)
        for g in gs:
            k = max(SPLITS, key=lambda k: SPLIT_FRACS[k] * totals[s] - counts[s][k])
            counts[s][k] += len(g)
            for r in g:
                r["split"] = k
    excluded = []
    for g in groups:
        present = {r["split"] for r in g}
        for r in g:
            if any(o in present for o in PROTECTED_BY[r["split"]]):
                excluded.append(r)
    return groups, excluded, conflicts


def item_dominant(item, classes):
    ids = [b[0] for b in item["boxes"]] if "boxes" in item else [c for c, _ in item["coco_anns"]]
    return dominant_class([(c,) for c in ids], classes)


def largest_remainder(sizes, n):
    """Split n over strata in proportion to their sizes; the total is exactly n (n <= sum of sizes)."""
    total = sum(sizes.values())
    quota = {k: n * v / total for k, v in sizes.items()}
    counts = {k: int(q) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: (counts[k] - quota[k], k))[:n - sum(counts.values())]:
        counts[k] += 1
    return counts


def subsample_items(items, src, classes, seed):
    """Keep a seeded sample of a pinned source's train and valid images, stratified by dominant class.

    subsample = {train: N, valid: M, seed: S}: each pinned split with a quota smaller than its size keeps exactly
    that many images, every dominant class keeping its share (largest-remainder rounding). Test images, unpinned
    images and splits without a quota are kept as they are. Returns (kept items sorted by uid, report).
    """
    cfg = src["subsample"]
    seed = cfg.get("seed", seed)
    by_split = defaultdict(list)
    for it in items:
        by_split[it.get("pinned")].append(it)
    kept, report = [], {"seed": seed}
    for split in sorted(by_split, key=str):
        members = by_split[split]
        want = cfg.get(split) if split in ("train", "valid") else None
        if want is None or want >= len(members):
            kept += members
            if want is not None:
                report[split] = {"requested": want, "before": len(members), "after": len(members)}
            continue
        strata = defaultdict(list)
        for it in members:
            strata[item_dominant(it, classes)].append(it)
        counts = largest_remainder({c: len(v) for c, v in strata.items()}, want)
        rng = random.Random(f"{seed}:{src['name']}:{split}")
        chosen = []
        for c in sorted(strata):
            chosen += rng.sample(sorted(strata[c], key=lambda it: it["uid"]), counts[c])
        kept += chosen
        report[split] = {"requested": want, "before": len(members), "after": len(chosen),
                         "dominant_class_before": {c: len(strata[c]) for c in sorted(strata)},
                         "dominant_class_after": {c: counts[c] for c in sorted(strata)}}
    return sorted(kept, key=lambda it: it["uid"]), report


# ---------------------------------------------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------------------------------------------
def prepare_out_dir(out, protected):
    out = out.resolve()
    for p in protected:
        if out == p or out.is_relative_to(p) or p.is_relative_to(out):
            raise BuildError(f"--out {out} overlaps the source folder {p}; choose another output folder")
    if out.exists() and any(out.iterdir()):
        if not (out / "build_report.json").is_file():
            raise BuildError(f"--out {out} exists, is not empty and is not a previous build; choose an empty folder")
        for sub in ("images", "labels", "lists"):
            if (out / sub).exists():
                shutil.rmtree(out / sub)
        for f in [*out.glob("data*.yaml"), *(out / n for n in OUTPUT_FILES)]:
            if f.is_file():
                f.unlink()
    for sub in ("images", "labels", "lists"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    return out


def write_yaml(path, out, classes, lists):
    cfg = {"path": out.as_posix(), "train": lists["train"].as_posix(), "val": lists["valid"].as_posix(),
           "test": lists["test"].as_posix(), "nc": len(classes), "names": list(classes)}
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def new_stats():
    return {"images_found": 0, "images_sampled": 0, "images_dropped": Counter(), "annotations_dropped": Counter(),
            "label_issues": Counter(), "warnings": Counter()}


def build(sources_yaml, out, seed=42, max_side=1280, search_root=None, max_images=None, workers=8, log=print):
    sources_yaml = Path(sources_yaml).resolve()
    classes, sources = load_config(sources_yaml)
    search_root = Path(search_root).resolve() if search_root else REPO
    if not search_root.is_dir():
        raise BuildError(f"--search-root {search_root} is not a folder")
    out = Path(out).resolve()

    # 1. locate and read every source, check every class map before any image work
    per_source, problems, roots, skipped = {}, [], [], {}
    for src in sources:
        stats = new_stats()
        try:
            if src["format"] == "coco":
                ann = locate_coco(src, search_root, skip=[out])
                names, items, root = read_coco(src, ann, classes, stats)
                location = ann
            else:
                locate = {"yolo": locate_yolo,
                          "supervisely": lambda s, r, skip: locate_named(s, r, skip, has_supervisely,
                                                                         "Supervisely ann/ and img/ folders"),
                          "labelme": lambda s, r, skip: locate_named(s, r, skip, has_json, "json files")}
                root = locate[src["format"]](src, search_root, skip=[out])
                reader = {"yolo": read_yolo, "supervisely": read_supervisely, "labelme": read_labelme}
                names, items = reader[src["format"]](src, root, classes, stats)
                location = root
        except SourceMissing as e:
            if not src["optional"]:
                raise
            skipped[src["name"]] = str(e)
            log(f"WARNING: optional source '{src['name']}' skipped: {e}")
            continue
        roots.append(root)
        problem = check_class_map(src, names)
        if problem:
            problems.append(problem)
        for it in items:
            it["format"] = src["format"]
        per_source[src["name"]] = {"src": src, "names": names, "items": items, "stats": stats,
                                   "location": location}
        log(f"{src['name']}: {len(items)} images in {location}")
    if problems:
        raise BuildError("Class map errors:\n  " + "\n  ".join(problems))
    if not per_source:
        raise BuildError("No sources found: " + "; ".join(skipped.values()))
    out = prepare_out_dir(out, [r.resolve() for r in roots])

    # 2. sample, pin, map and subsample
    all_items = []
    pin_report, sub_report = {}, {}
    for name, ps in per_source.items():
        src, items, stats = ps["src"], ps["items"], ps["stats"]
        stats["images_found"] = len(items)
        cap = min(v for v in (src["max_images"], max_images, len(items)) if v is not None)
        if cap < len(items):
            items = random.Random(f"{seed}:{name}").sample(items, cap)
            items.sort(key=lambda it: it["uid"])
        stats["images_sampled"] = len(items)
        if src["keep_split_csv"]:
            csv_path = resolve_repo_path(src["keep_split_csv"], sources_yaml.parent)
            if not csv_path.is_file():
                raise BuildError(f"source '{name}': keep_split_csv {csv_path} not found")
            with open(csv_path, newline="") as f:
                rows = list(csv.DictReader(f))
            if not rows or not {"split", "file"} <= set(rows[0]):
                raise BuildError(f"{csv_path} needs 'split' and 'file' columns")
            pinned = {r["file"]: r["split"] for r in rows}
            if set(pinned.values()) - set(SPLITS):
                raise BuildError(f"{csv_path} has split values other than {SPLITS}")
            basenames = Counter(Path(it["uid"]).name for it in ps["items"])
            if any(n > 1 for n in basenames.values()):
                raise BuildError(f"source '{name}': keep_split_csv needs unique image file names in the source")
            hits = 0
            for it in items:
                it["pinned"] = pinned.get(Path(it["uid"]).name)
                hits += it["pinned"] is not None
            pin_report[name] = {"csv": csv_path.as_posix(), "csv_rows": len(rows), "pinned_images": hits,
                                "images_not_in_csv": len(items) - hits,
                                "csv_rows_not_in_source": len(set(pinned) - set(basenames))}
            if hits < len(items):
                stats["warnings"]["images_not_in_keep_split_csv"] += len(items) - hits
        if src["force_split"]:
            for it in items:
                it["pinned"] = src["force_split"]
        mapped = []
        for it in items:
            if src["format"] == "yolo":
                ok = map_yolo_item(it, src, ps["names"], classes, stats)
            elif src["format"] == "coco":
                ok = map_coco_item(it, src, classes, stats)
            else:
                ok = map_objects_item(it, src, classes, stats)
            if ok:
                mapped.append(it)
        if src["subsample"]:
            mapped, sub_report[name] = subsample_items(mapped, src, classes, seed)
        all_items += mapped
    assign_file_names(all_items)

    # 3. write images and labels
    log(f"Writing {len(all_items)} images to {out / 'images'} ...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda it: process_item(it, per_source[it["source"]]["src"], out, max_side),
                                all_items))
    recs = []
    for it, (rec, reason, warns) in zip(all_items, results):
        stats = per_source[it["source"]]["stats"]
        for k, v in warns.items():
            if k.startswith("unreadable"):
                continue
            target = "annotations_dropped" if k in ("invalid_bbox", "box_outside_image") else "warnings"
            stats[target][k] += v
        if reason:
            stats["images_dropped"][reason] += 1
        else:
            rec["domain"] = per_source[it["source"]]["src"]["domain"]
            rec["dominant_class"] = dominant_class(rec["boxes"], classes)
            recs.append(rec)
    if not recs:
        raise BuildError("No images left after mapping; check the class maps and the source folders")

    # 4. duplicates and split
    log(f"Comparing {len(recs)} images for duplicates ...")
    near_matches = assign_near_dup_groups(sorted(recs, key=lambda r: r["file"]))
    groups, excluded, conflicts = split_records(recs, classes, seed)
    excluded_names = {r["file"] for r in excluded}
    for r in excluded:
        (out / "images" / r["file"]).unlink()
        (out / "labels" / r["file"]).with_suffix(".txt").unlink()
    kept = sorted((r for r in recs if r["file"] not in excluded_names),
                  key=lambda r: (SPLITS.index(r["split"]), r["file"]))

    # 5. outputs
    gid = {}
    for i, g in enumerate(groups):
        for r in g:
            gid[r["file"]] = i
    with open(out / "split.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "file", "source", "domain", "dominant_class", "source_class", "group", "source_file"])
        for r in kept:
            w.writerow([r["split"], r["file"], r["source"], r["domain"], r["dominant_class"],
                        f"{r['source']}/{r['dominant_class']}", gid[r["file"]], r["uid"]])
    img_dir = out / "images"
    lists = {}
    for s in SPLITS:
        lists[s] = out / "lists" / f"{s}.txt"
        lists[s].write_text("".join((img_dir / r["file"]).as_posix() + "\n" for r in kept if r["split"] == s))
    write_yaml(out / "data.yaml", out, classes, lists)
    warnings = [f"split '{s}' is empty" for s in SPLITS if not any(r["split"] == s for r in kept)]
    variants = {}
    for kind, key, values in (("", "domain", sorted({r["domain"] for r in kept})),
                              ("src_", "source", sorted({r["source"] for r in kept}))):
        for v in values:
            members = [r for r in kept if r["split"] == "test" and r[key] == v]
            if not members:
                warnings.append(f"no test images for {key} '{v}'; data_test_{kind}{v}.yaml not written")
                continue
            lst = out / "lists" / f"test_{kind}{v}.txt"
            lst.write_text("".join((img_dir / r["file"]).as_posix() + "\n" for r in members))
            write_yaml(out / f"data_test_{kind}{v}.yaml", out, classes, {**lists, "test": lst})
            variants[f"{kind}{v}"] = {"yaml": f"data_test_{kind}{v}.yaml", "test_images": len(members)}

    warnings = [f"optional source '{n}' was not found and is NOT in this build: {why}"
                for n, why in skipped.items()] + warnings
    report = make_report(per_source, kept, groups, near_matches, excluded, conflicts, pin_report, classes,
                         variants, warnings, dict(seed=seed, max_side=max_side, search_root=search_root.as_posix(),
                                                  max_images=max_images, sources_yaml=sources_yaml.as_posix(),
                                                  out=out.as_posix()), sub_report, skipped)
    (out / "build_report.json").write_text(json.dumps(report, indent=2))
    return report


def count_block(recs, classes):
    boxes = Counter(classes[b[0]] for r in recs for b in r["boxes"])
    imgs = Counter(classes[c] for r in recs for c in {b[0] for b in r["boxes"]})
    return {"images": len(recs), "boxes": sum(boxes.values()), "background_images": sum(not r["boxes"] for r in recs),
            "images_per_class": {c: imgs.get(c, 0) for c in classes},
            "boxes_per_class": {c: boxes.get(c, 0) for c in classes}}


def make_report(per_source, kept, groups, near_matches, excluded, conflicts, pin_report, classes, variants,
                warnings, settings, sub_report=None, skipped=None):
    multi = [g for g in groups if len(g) > 1]
    rules = Counter(rule for _, _, rule in near_matches)
    sources = {}
    for name, ps in per_source.items():
        src, st = ps["src"], ps["stats"]
        mine = [r for r in kept if r["source"] == name]
        sources[name] = {
            "format": src["format"], "domain": src["domain"], "licence": src["licence"],
            "location": Path(ps["location"]).as_posix(), "class_map": src["class_map"],
            "images_found": st["images_found"], "images_sampled": st["images_sampled"],
            "images_kept": len(mine),
            "images_excluded_as_cross_split_duplicates": sum(r["source"] == name for r in excluded),
            "images_dropped": dict(sorted(st["images_dropped"].items())),
            "annotations_dropped": dict(sorted(st["annotations_dropped"].items())),
            "label_issues": dict(sorted(st["label_issues"].items())),
            "warnings": dict(sorted(st["warnings"].items())),
            "keep_split_csv": pin_report.get(name),
            "force_split": src["force_split"],
            "subsample": (sub_report or {}).get(name),
            "kaggle": src["kaggle"], "optional": src["optional"],
            "splits": {s: count_block([r for r in mine if r["split"] == s], classes) for s in SPLITS},
        }
    domains = sorted({r["domain"] for r in kept})
    return {
        "classes": classes,
        "settings": {**settings, "jpeg_quality": JPEG_QUALITY, "split_fractions": SPLIT_FRACS,
                     "near_duplicate_rules": {"dhash_size": DHASH_SIZE, "near_dup_max_bits": NEAR_DUP_MAX_BITS,
                                              "candidate_max_bits": CANDIDATE_MAX_BITS,
                                              "min_thumb_corr": MIN_THUMB_CORR}},
        "totals": {s: count_block([r for r in kept if r["split"] == s], classes) for s in SPLITS},
        "domains": {d: {s: count_block([r for r in kept if r["split"] == s and r["domain"] == d], classes)
                        for s in SPLITS} for d in domains},
        "sources": sources,
        "skipped_sources": dict(skipped or {}),
        "duplicates": {
            "groups": len(multi), "images_in_groups": sum(len(g) for g in multi),
            "cross_source_groups": sum(len({r["source"] for r in g}) > 1 for g in multi),
            "matched_pairs": {"identical_md5": count_shared_key_pairs(kept, "md5"),
                              "dhash": rules["dhash"], "thumb_corr": rules["thumb_corr"]},
            "pinned_groups_with_conflicting_splits": conflicts,
            "excluded_cross_split_copies": sorted(r["file"] for r in excluded),
            "examples": [[r["file"] for r in g] for g in multi[:50]],
        },
        "test_variants": variants,
        "warnings": warnings,
    }


def print_summary(report, log=print):
    classes = report["classes"]
    log("\nImages / boxes per split: " + ", ".join(
        f"{s} {v['images']}/{v['boxes']}" for s, v in report["totals"].items()))
    log("Boxes per class (train / valid / test):")
    for c in classes:
        log(f"  {c:<16}" + " / ".join(f"{report['totals'][s]['boxes_per_class'][c]:>6}" for s in SPLITS))
    for name, s in report["sources"].items():
        log(f"{name} [{s['domain']}, {s['licence']}]: found {s['images_found']}, sampled {s['images_sampled']}, "
            f"kept {s['images_kept']} (" + ", ".join(f"{k} {v['images']}" for k, v in s["splits"].items()) + ")")
        for split, sub in (s.get("subsample") or {}).items():
            if split != "seed":
                log(f"    subsampled {split}: {sub['before']} -> {sub['after']} (requested {sub['requested']})")
        if s["images_dropped"]:
            log(f"    images dropped: {s['images_dropped']}")
        if s["annotations_dropped"]:
            log(f"    annotations dropped: {s['annotations_dropped']}")
        if s["warnings"]:
            log(f"    warnings: {s['warnings']}")
    d = report["duplicates"]
    log(f"Duplicate groups: {d['groups']} ({d['images_in_groups']} images, {d['cross_source_groups']} across "
        f"sources); excluded cross-split copies: {len(d['excluded_cross_split_copies'])}")
    log("Test variants: " + ", ".join(f"{k} ({v['test_images']})" for k, v in report["test_variants"].items()))
    for w in report["warnings"]:
        log(f"Warning: {w}")
    for name in report.get("skipped_sources", {}):
        log(f"\n*** WARNING: optional source '{name}' was NOT found and is missing from this build. ***")


# ---------------------------------------------------------------------------------------------------------------
# Relocate: point the lists and yamls of a finished (read-only) build at its current location
# ---------------------------------------------------------------------------------------------------------------
def relocate(built_dir, cfg_dir, log=print):
    """Write data.yaml, data_test_*.yaml, lists/*.txt and split.csv into cfg_dir, every image path rewritten to
    <built_dir>/images/<file>. built_dir is only read. Returns a summary dict."""
    built, cfg = Path(built_dir).resolve(), Path(cfg_dir).resolve()
    for need in ("build_report.json", "split.csv", "data.yaml", "lists", "images", "labels"):
        if not (built / need).exists():
            raise BuildError(f"--relocate-from {built} is not a finished build: {need} is missing")
    if cfg == built or cfg.is_relative_to(built) or built.is_relative_to(cfg):
        raise BuildError(f"--out {cfg} must be outside the build folder {built} (which is only read)")
    (cfg / "lists").mkdir(parents=True, exist_ok=True)
    images = built / "images"
    present = {p.name for p in images.iterdir()}
    n_lines, missing = 0, []
    for lst in sorted((built / "lists").glob("*.txt")):
        names = [Path(line.strip().replace("\\", "/")).name for line in lst.read_text().splitlines() if line.strip()]
        missing += [n for n in names if n not in present]
        (cfg / "lists" / lst.name).write_text("".join((images / n).as_posix() + "\n" for n in names))
        n_lines += len(names)
    if missing:
        raise BuildError(f"{len(missing)} listed images are not in {images}, e.g. {sorted(set(missing))[:5]}")
    yamls = []
    for y in sorted(built.glob("data*.yaml")):
        data = yaml.safe_load(y.read_text()) or {}
        targets = {k: cfg / "lists" / Path(str(data[k]).replace("\\", "/")).name
                   for k in ("train", "val", "test") if k in data}
        absent = [str(data[k]) for k, t in targets.items() if not t.is_file()]
        if absent:
            if y.name == "data.yaml" or y.name.startswith("data_test_"):
                raise BuildError(f"{y.name}: lists {absent} are not in {built / 'lists'}")
            log(f"Skipped {y.name}: lists {absent} are not in {built / 'lists'}")  # e.g. a proxy made elsewhere
            continue
        data.update({k: t.as_posix() for k, t in targets.items()})
        data["path"] = built.as_posix()
        (cfg / y.name).write_text(yaml.safe_dump(data, sort_keys=False))
        yamls.append(y.name)
    with open(built / "split.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "file" not in rows[0]:
        raise BuildError(f"{built / 'split.csv'} has no 'file' column")
    fields = list(rows[0]) + (["path"] if "path" not in rows[0] else [])
    with open(cfg / "split.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "path": (images / r["file"]).as_posix()})
    log(f"Relocated {len(yamls)} yamls, {len(list((built / 'lists').glob('*.txt')))} lists ({n_lines} lines) and "
        f"split.csv ({len(rows)} rows) into {cfg}; images stay in {images}")
    return {"built_dir": built.as_posix(), "cfg_dir": cfg.as_posix(), "yamls": yamls, "list_lines": n_lines,
            "split_rows": len(rows)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default=str(REPO / "configs" / "gen2" / "sources.yaml"))
    ap.add_argument("--out", required=True, help="output folder (empty, or a previous build to replace); with "
                    "--relocate-from: the folder for the rewritten yamls, lists and split.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-side", type=int, default=1280, help="long side of written images, in pixels")
    ap.add_argument("--search-root", help="where the source folders are searched (default: the project root)")
    ap.add_argument("--max-images", type=int, help="at most this many images per source (quick tests)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--relocate-from", help="a finished build (may be read-only); rewrite its yamls, lists and "
                    "split.csv into --out instead of building")
    args = ap.parse_args(argv)
    if args.relocate_from:
        try:
            return relocate(args.relocate_from, args.out)
        except BuildError as e:
            sys.exit(f"Error: {e}")
    try:
        report = build(args.sources, args.out, args.seed, args.max_side, args.search_root, args.max_images,
                       args.workers)
    except BuildError as e:
        sys.exit(f"Error: {e}")
    print_summary(report)
    print(f"\nWrote {Path(args.out).resolve() / 'data.yaml'} and build_report.json")
    return report


if __name__ == "__main__":
    main()
