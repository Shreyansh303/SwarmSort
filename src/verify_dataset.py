"""Phase 1: verify the YOLO waste-detection dataset, fix its split, and generate configs/data.yaml.

Usage:
    python src/verify_dataset.py --root "GARBAGE CLASSIFICATION" --split new   # create configs/split.csv
    python src/verify_dataset.py --root /kaggle/input                         # reuse configs/split.csv

--split provided   use the dataset's own train/valid/test folders
--split new        pool every image and write a stratified, duplicate-aware split to <configs>/split.csv
--split auto       (default) use <configs>/split.csv; exit with an error if it is missing

Never modifies the dataset: splits are expressed as generated image lists.
Exit code 0 = every automated check passed, 1 = at least one failed or the input is unusable
(for example a missing split file in auto mode).
"""
import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

SPLITS = ("train", "valid", "test")
SPLIT_FRACS = {"train": 0.70, "valid": 0.20, "test": 0.10}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REPO = Path(__file__).resolve().parent.parent
SPLIT_FILE_NAME = "split.csv"  # lives in the --configs directory
MAX_ROOT_DEPTH = 7  # Kaggle mounts datasets as /kaggle/input/datasets/<owner>/<slug>/<folder>/train/...

MIN_TRAIN_BOXES = 400
MIN_EVAL_BOXES = 30
MAX_EVAL_LEAK_FRAC = 0.01
MAX_DEGENERATE_FRAC = 0.30
DEGENERATE_AREA = 0.90
MAX_MISSING_LABEL_FRAC = 0.01
SAMPLES_PER_SPLIT = 30
GRID_COLS, GRID_ROWS, TILE = 3, 2, 320

# Near-duplicates. Every pair of images is compared by a 256-bit dHash (16 rows x 16 adjacent-pixel
# comparisons of a grayscale thumbnail). Two images are copies, and are kept in the same split, if
#   rule 1: their hashes differ by <= NEAR_DUP_MAX_BITS bits, or
#   rule 2: their hashes differ by <= CANDIDATE_MAX_BITS bits and the Pearson correlation of their
#           64x64 grayscale thumbnails is >= MIN_THUMB_CORR (catches shifted, watermarked and
#           re-coloured copies whose hashes drift further apart).
# Thresholds come from inspecting this dataset's candidate pairs: confirmed copies were found up to
# 39 bits apart with correlations down to about 0.90. Some look-alike bottles on white backgrounds also
# pass rule 2; grouping them only keeps them in the same split. Results depend on Pillow's resize filter,
# which is why configs/split.csv is committed and reused rather than regenerated.
DHASH_SIZE = 16
NEAR_DUP_MAX_BITS = 10
CANDIDATE_MAX_BITS = 40
THUMB_SIZE = 64
MIN_THUMB_CORR = 0.90
HASH_CHUNK = 256  # rows compared at once; 256 x 10k images x 32 bytes is under 100 MB

# When the provided split leaks, the lower-priority split loses its copy; test is never touched.
PROTECTED_BY = {"train": ("valid", "test"), "valid": ("test",), "test": ()}
# Keys that mark two images as copies: identical bytes, the same Roboflow source image ("stem"),
# or the same near-duplicate group id ("near_dup", assigned by assign_near_dup_groups).
LEAK_METHODS = ("md5", "stem", "near_dup")
PALETTE = [(230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66),
           (0, 114, 178), (213, 94, 0), (204, 121, 167), (140, 140, 140)]


def is_dataset(folder):
    return all((folder / s / "images").is_dir() for s in SPLITS)


def find_root(root):
    """Return the one dataset folder at or below `root`; exit if there are none or several."""
    start = Path(root).resolve()
    if not start.exists():
        sys.exit(f"--root {root} does not exist")
    if not start.is_dir():
        sys.exit(f"--root {root} is a file; pass the dataset folder or one of its parents")
    found, level = [], [start]
    for _ in range(MAX_ROOT_DEPTH):
        next_level = []
        for folder in level:
            if is_dataset(folder):
                found.append(folder)  # a match is not searched further
                continue
            try:
                next_level += [d for d in sorted(folder.iterdir()) if d.is_dir()]
            except OSError:
                pass
        level = next_level
    if len(found) > 1:
        sys.exit("Several datasets found, pass one of them as --root:\n  " + "\n  ".join(map(str, found)))
    if not found:
        seen = sorted(str(p.relative_to(start)) for p in start.glob("*/*/*") if p.is_dir())[:40]
        sys.exit(f"No folder containing {', '.join(s + '/images' for s in SPLITS)} found under {root}. "
                 f"Folders seen (3 levels): {seen or 'none - is the dataset attached as input?'}")
    return found[0]


def load_names(root):
    for cand in (root / "data.yaml", root.parent / "data.yaml"):
        if cand.exists():
            names = yaml.safe_load(cand.read_text())["names"]
            return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    sys.exit(f"No data.yaml with class names found in {root}")


def dhash_bits(gray_img, n):
    small = np.asarray(gray_img.resize((n + 1, n), Image.Resampling.LANCZOS), dtype=np.int16)
    return (small[:, 1:] > small[:, :-1]).flatten()


def thumb_correlation(thumbs_a, thumbs_b):
    """Pearson correlation between matching rows of two stacks of grayscale thumbnails."""
    a = thumbs_a.reshape(len(thumbs_a), -1).astype(np.float32)
    b = thumbs_b.reshape(len(thumbs_b), -1).astype(np.float32)
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    return (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1) + 1e-12)


def assign_near_dup_groups(recs):
    """Compare every pair of hashed images, merge matches by union-find, and store the group id.

    Each image's 'near_dup' key becomes the id of its group (the lowest member index, so it is
    deterministic); equality-based code then treats images with the same id as copies.
    Returns the matched pairs as (rec_a, rec_b, rule) with rule "dhash" or "thumb_corr".
    """
    hashed = [r for r in recs if r["dhash256"] is not None]
    if not hashed:
        return []
    words = np.packbits(np.stack([r["dhash256"] for r in hashed]), axis=1).view(np.uint64)
    thumbs = np.stack([r["thumb"] for r in hashed])
    parent = list(range(len(hashed)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    matches = []
    for start in range(0, len(words), HASH_CHUNK):
        block = words[start:start + HASH_CHUNK]
        dist = np.bitwise_count(block[:, None, :] ^ words[None, start:, :]).sum(axis=2, dtype=np.uint16)
        rows, cols = np.nonzero(dist <= CANDIDATE_MAX_BITS)
        a, b = rows + start, cols + start
        keep = a < b  # each unordered pair once
        a, b, d = a[keep], b[keep], dist[rows[keep], cols[keep]]
        close = d <= NEAR_DUP_MAX_BITS
        corr = np.zeros(len(a), dtype=np.float32)
        if (~close).any():
            corr[~close] = thumb_correlation(thumbs[a[~close]], thumbs[b[~close]])
        for i, j, is_close, c in zip(a, b, close, corr):
            if is_close or c >= MIN_THUMB_CORR:
                matches.append((int(i), int(j), "dhash" if is_close else "thumb_corr"))
    for i, j, _ in matches:
        ri, rj = find(i), find(j)
        parent[max(ri, rj)] = min(ri, rj)
    for i, r in enumerate(hashed):
        r["near_dup"] = f"g{find(i)}"
    return [(hashed[i], hashed[j], rule) for i, j, rule in matches]


def scan_image(path):
    data = path.read_bytes()
    rec = {"name": path.name, "path": path, "md5": hashlib.md5(data).hexdigest(),
           "stem": path.name.split(".rf.")[0] if ".rf." in path.name else None,
           "size": None, "dhash256": None, "thumb": None, "near_dup": None, "error": None}
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            rec["size"] = im.size
            gray = im.convert("L")
            bits = dhash_bits(gray, DHASH_SIZE)
            if bits.any() and not bits.all():  # flat images collide trivially
                rec["dhash256"] = bits
                rec["thumb"] = np.asarray(gray.resize((THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS),
                                          dtype=np.uint8)
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def parse_label(path, nc):
    boxes, issues, seen = [], Counter(), set()
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if tuple(parts) in seen:
            issues["duplicate_lines"] += 1
            continue
        seen.add(tuple(parts))
        try:
            cls_value, vals = float(parts[0]), [float(v) for v in parts[1:]]
        except ValueError:
            issues["malformed_lines"] += 1
            continue
        if not cls_value.is_integer() or not all(math.isfinite(v) for v in vals):
            issues["malformed_lines"] += 1  # class ids like 1.7, or nan/inf coordinates
            continue
        cls = int(cls_value)
        if len(vals) == 4:
            cx, cy, w, h = vals
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            xs, ys = vals[0::2], vals[1::2]
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            w, h = max(xs) - min(xs), max(ys) - min(ys)
            issues["polygon_lines"] += 1
        else:
            issues["malformed_lines"] += 1
            continue
        if not 0 <= cls < nc:
            issues["bad_class_id"] += 1
            continue
        if min(vals) < 0 or max(vals) > 1:
            issues["out_of_range_coords"] += 1  # Ultralytics rejects the whole image for this
        if w <= 0 or h <= 0:
            issues["zero_size_boxes"] += 1
            continue
        boxes.append((cls, cx, cy, w, h))
    return boxes, issues


def scan_folder(root, folder, nc):
    """Scan one physical folder; also return how many label files there have no image."""
    img_dir, lbl_dir = root / folder / "images", root / folder / "labels"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    label_stems = {p.stem for p in lbl_dir.glob("*.txt")} if lbl_dir.is_dir() else set()
    print(f"  {folder}/: hashing and decoding {len(images)} images ...", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        recs = list(pool.map(scan_image, images))
    for rec in recs:
        lbl = lbl_dir / f"{rec['path'].stem}.txt"
        rec["folder"] = folder
        rec["has_label"] = lbl.exists()
        rec["boxes"], rec["issues"] = parse_label(lbl, nc) if rec["has_label"] else ([], Counter())
    return recs, len(label_stems - {r["path"].stem for r in recs})


def source_class(rec, names):
    lower = {n.lower(): n for n in names}
    m = re.match(r"[A-Za-z]+", rec["name"])
    if m and m.group(0).lower() in lower:
        return lower[m.group(0).lower()]
    if rec["boxes"]:
        return names[Counter(b[0] for b in rec["boxes"]).most_common(1)[0][0]]
    return "background"


def duplicate_groups(recs):
    parent = list(range(len(recs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first = {}
    for i, r in enumerate(recs):
        for m in LEAK_METHODS:
            if r[m]:
                key = (m, r[m])
                if key in first:
                    parent[find(i)] = find(first[key])
                else:
                    first[key] = i
    groups = defaultdict(list)
    for i, r in enumerate(recs):
        groups[find(i)].append(r)
    return list(groups.values())


def count_shared_key_pairs(recs, key):
    """Number of image pairs that share the same non-empty value of `key`."""
    sizes = Counter(r[key] for r in recs if r[key])
    return sum(k * (k - 1) // 2 for k in sizes.values())


def new_split(recs, names, seed):
    """Stratify by source class; every duplicate group lands in exactly one split."""
    strata = defaultdict(list)
    for g in duplicate_groups(recs):
        strata[source_class(g[0], names)].append(g)
    rng = random.Random(seed)
    assignment = {}
    for stratum in sorted(strata):
        groups = strata[stratum]
        rng.shuffle(groups)
        total = sum(len(g) for g in groups)
        counts = dict.fromkeys(SPLITS, 0)
        for g in groups:
            s = max(SPLITS, key=lambda k: SPLIT_FRACS[k] * total - counts[k])
            counts[s] += len(g)
            for r in g:
                assignment[r["name"]] = s
    return assignment


def write_split_file(path, assignment, recs, names):
    path.parent.mkdir(parents=True, exist_ok=True)
    by_name = {r["name"]: r for r in recs}
    rows = sorted(assignment.items(), key=lambda kv: (SPLITS.index(kv[1]), kv[0]))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "folder", "file", "source_class"])
        for name, s in rows:
            w.writerow([s, by_name[name]["folder"], name, source_class(by_name[name], names)])


def read_split_file(path, recs):
    if not path.is_file():
        sys.exit(f"Split file {path} not found")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"split", "file"} <= set(reader.fieldnames):
            sys.exit(f"{path} must have a header row with 'split' and 'file' columns "
                     f"(found: {reader.fieldnames or 'nothing'})")
        rows = list(reader)
    incomplete = sum(1 for row in rows if not row["file"] or not row["split"])
    if incomplete:
        sys.exit(f"{path} has {incomplete} rows without a file or split value")
    counts = Counter(row["file"] for row in rows)
    repeated = sorted(name for name, n in counts.items() if n > 1)
    if repeated:
        sys.exit(f"{path} lists {len(repeated)} images more than once, e.g. {', '.join(repeated[:3])}")
    bad = sorted({row["split"] for row in rows} - set(SPLITS))
    if bad:
        sys.exit(f"{path} has invalid split values {bad}; allowed: {', '.join(SPLITS)}")
    assignment = {row["file"]: row["split"] for row in rows}
    names = {r["name"] for r in recs}
    missing, extra = names - assignment.keys(), assignment.keys() - names
    if missing or extra:
        sys.exit(f"{path} does not match the dataset: {len(missing)} images unassigned, "
                 f"{len(extra)} listed images not found")
    return assignment


def find_leaks(recs):
    keys = {s: {m: {r[m] for r in recs[s] if r[m]} for m in LEAK_METHODS} for s in SPLITS}
    pairs, drop = {}, {s: [] for s in SPLITS}
    for s in SPLITS:
        for r in recs[s]:
            if any(r[m] and r[m] in keys[o][m] for o in PROTECTED_BY[s] for m in LEAK_METHODS):
                drop[s].append(r)
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            pairs[f"{a}-{b}"] = {m: sum(1 for r in recs[b] if r[m] and r[m] in keys[a][m])
                                 for m in LEAK_METHODS}
    eval_total = len(recs["valid"]) + len(recs["test"])
    eval_leaked = sum(
        1 for s in ("valid", "test") for r in recs[s]
        if any(r[m] and r[m] in keys[o][m] for o in SPLITS if o != s for m in LEAK_METHODS))
    return pairs, drop, eval_leaked / eval_total if eval_total else 0.0


def draw_tile(rec, names, font):
    im = Image.open(rec["path"]).convert("RGB")
    scale = TILE / max(im.size)
    im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))))
    d = ImageDraw.Draw(im)
    W, H = im.size
    for cls, cx, cy, w, h in rec["boxes"]:
        color = PALETTE[cls % len(PALETTE)]
        x0, y0, x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H
        d.rectangle([x0, y0, x1, y1], outline=color, width=2)
        tb = d.textbbox((0, 0), names[cls], font=font)
        tw, th = tb[2] - tb[0] + 6, tb[3] - tb[1] + 6
        ty = y0 - th if y0 >= th else y0
        d.rectangle([x0, ty, x0 + tw, ty + th], fill=color)
        d.text((x0 + 3, ty + 2), names[cls], fill=(0, 0, 0), font=font)
    tile = Image.new("RGB", (TILE, TILE + 22), (245, 245, 245))
    tile.paste(im, ((TILE - W) // 2, (TILE - H) // 2))
    caption = rec["name"] if len(rec["name"]) <= 46 else rec["name"][:43] + "..."
    ImageDraw.Draw(tile).text((4, TILE + 5), caption, fill=(60, 60, 60), font=font)
    return tile


def save_grids(recs, split, names, out_dir, seed):
    font = ImageFont.load_default(size=12)
    readable = [r for r in recs if not r["error"]]  # corrupt images are reported, not drawn
    sample = random.Random(seed).sample(readable, min(SAMPLES_PER_SPLIT, len(readable)))
    per_grid, paths = GRID_COLS * GRID_ROWS, []
    for old in out_dir.glob(f"{split}_grid_*.jpg"):
        old.unlink()
    for g in range(0, len(sample), per_grid):
        grid = Image.new("RGB", (GRID_COLS * TILE, GRID_ROWS * (TILE + 22)), (255, 255, 255))
        for i, rec in enumerate(sample[g:g + per_grid]):
            grid.paste(draw_tile(rec, names, font), ((i % GRID_COLS) * TILE, (i // GRID_COLS) * (TILE + 22)))
        path = out_dir / f"{split}_grid_{g // per_grid + 1:02d}.jpg"
        grid.save(path, quality=85)
        paths.append(path)
    return paths


def save_pair_sheet(pairs, path, limit=12):
    if path.exists():
        path.unlink()
    pairs = [p for p in pairs if not p[0]["error"] and not p[1]["error"]][:limit]
    if not pairs:
        return None
    half = TILE // 2
    sheet = Image.new("RGB", (2 * half, len(pairs) * half), (255, 255, 255))
    for i, pair in enumerate(pairs):
        for j, rec in enumerate(pair):
            im = Image.open(rec["path"]).convert("RGB")
            im.thumbnail((half, half))
            sheet.paste(im, (j * half, i * half))
    sheet.save(path, quality=85)
    return path


def write_yaml(root, names, kept, list_splits, cfg_dir):
    """Write data.yaml; splits in `list_splits` (or with excluded images) point to image lists."""
    cfg_dir.mkdir(parents=True, exist_ok=True)
    list_dir = cfg_dir / "lists"
    entries = {}
    for s in SPLITS:
        if s in list_splits:
            # Ultralytics names its labels.cache after the first listed image's label folder, so every
            # list shares one cache file in the dataset; it is validated by hash and rebuilt in seconds.
            list_dir.mkdir(exist_ok=True)
            lst = list_dir / f"{s}.txt"
            lst.write_text("".join(r["path"].as_posix() + "\n" for r in kept[s]))
            entries[s] = lst.resolve().as_posix()
        else:
            entries[s] = f"{s}/images"
    cfg = {"path": root.as_posix(), "train": entries["train"], "val": entries["valid"],
           "test": entries["test"], "nc": len(names), "names": names}
    out = cfg_dir / "data.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def ultralytics_check(yaml_path, expected):
    try:
        from ultralytics.data.dataset import YOLODataset
        from ultralytics.data.utils import check_det_dataset
    except ImportError:
        return {"status": "skipped", "detail": "ultralytics not installed"}
    try:
        data = check_det_dataset(str(yaml_path))
        found = {}
        for key, split in (("train", "train"), ("val", "valid"), ("test", "test")):
            ds = YOLODataset(img_path=data[key], data=data, task="detect", augment=False)
            found[split] = len(ds.labels)
    except Exception as e:
        return {"status": "fail", "detail": f"{type(e).__name__}: {e}"}
    ok = found == expected
    return {"status": "pass" if ok else "fail",
            "detail": f"loader accepted {found}" + ("" if ok else f", expected {expected}")}


def split_stats(recs_s, names):
    nc = len(names)
    boxes = [b for r in recs_s for b in r["boxes"]]
    per_cls = Counter(b[0] for b in boxes)
    img_per_cls = Counter(c for r in recs_s for c in {b[0] for b in r["boxes"]})
    issues = Counter()
    for r in recs_s:
        issues.update(r["issues"])
    mismatch = [r for r in recs_s
                if source_class(r, names) in names and r["boxes"]
                and names.index(source_class(r, names)) not in {b[0] for b in r["boxes"]}]
    return {
        "images": len(recs_s),
        "missing_labels": sum(not r["has_label"] for r in recs_s),
        "background_images": sum(not r["boxes"] for r in recs_s),
        "boxes": len(boxes),
        "boxes_per_class": {names[i]: per_cls.get(i, 0) for i in range(nc)},
        "images_per_class": {names[i]: img_per_cls.get(i, 0) for i in range(nc)},
        "image_sizes": dict(Counter(f"{r['size'][0]}x{r['size'][1]}" for r in recs_s if r["size"])
                            .most_common(5)),
        "corrupt_images": [f"{r['name']}: {r['error']}" for r in recs_s if r["error"]],
        "label_issues": dict(issues),
        "filename_class_mismatch": len(mismatch),
    }


def show_path(path):
    return path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else str(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dataset folder, or any parent of it")
    ap.add_argument("--split", default="auto", help="auto | provided | new | path to a split csv")
    ap.add_argument("--out", default=str(REPO / "results"), help="where the report and plots go")
    ap.add_argument("--configs", default=str(REPO / "configs"),
                    help="where data.yaml, image lists and split.csv are written")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-ultralytics", action="store_true", help="skip the Ultralytics loader check")
    args = ap.parse_args()

    root = find_root(args.root)
    names = load_names(root)
    nc = len(names)
    out_dir, cfg_dir = Path(args.out).resolve(), Path(args.configs).resolve()
    split_file = cfg_dir / SPLIT_FILE_NAME
    mode = args.split
    if mode == "auto":
        if not split_file.exists():
            sys.exit(f"--split auto needs {split_file}, which does not exist. Restore the committed "
                     f"configs/split.csv (or point --configs at its folder), pass --split provided to use "
                     f"the dataset's own folders, or create a split with --split new.")
        mode = str(split_file)
    plot_dir = out_dir / "plots" / "label_check"
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: {root}\nClasses ({nc}): {', '.join(names)}")

    provided, orphans = {}, {}
    for s in SPLITS:
        provided[s], orphans[s] = scan_folder(root, s, nc)
    all_recs = sorted((r for s in SPLITS for r in provided[s]), key=lambda r: r["name"])
    if len({r["name"] for r in all_recs}) != len(all_recs):
        sys.exit("Duplicate filenames across folders; split files cannot identify images uniquely")
    print("  comparing every image pair for near-duplicates ...", flush=True)
    near_matches = assign_near_dup_groups(all_recs)

    if mode == "provided":
        recs, split_desc = provided, "provided folders"
    else:
        if mode == "new":
            assignment = new_split(all_recs, names, args.seed)
            write_split_file(split_file, assignment, all_recs, names)
            split_desc = f"new stratified split (seed {args.seed}), saved to {show_path(split_file)}"
        else:
            assignment = read_split_file(Path(mode), all_recs)
            split_desc = f"split file {Path(mode).name}"
        recs = {s: [r for r in all_recs if assignment[r["name"]] == s] for s in SPLITS}
    print(f"Split: {split_desc}")

    print("  checking cross-split duplicates ...", flush=True)
    pairs, drop, eval_leak = find_leaks(recs)
    dropped = {s: {r["name"] for r in drop[s]} for s in SPLITS}
    kept = {s: [r for r in recs[s] if r["name"] not in dropped[s]] for s in SPLITS}  # what the lists hold
    split_report = {s: split_stats(kept[s], names) for s in SPLITS}
    provided_balance = {s: split_stats(provided[s], names)["boxes_per_class"] for s in SPLITS}

    # Pairs worth eyeballing: matched by a perceptual rule but not byte-identical.
    near_pairs = [(a, b) for a, b, _ in near_matches if a["md5"] != b["md5"]]
    dup_sheet = save_pair_sheet(near_pairs, plot_dir / "near_duplicate_pairs.jpg")
    groups = [g for g in duplicate_groups(all_recs) if len(g) > 1]
    rule_counts = Counter(rule for _, _, rule in near_matches)
    matched_pairs = {
        "identical_md5": count_shared_key_pairs(all_recs, "md5"),
        "same_stem": count_shared_key_pairs(all_recs, "stem"),
        f"dhash_le_{NEAR_DUP_MAX_BITS}_bits": rule_counts["dhash"],
        f"dhash_le_{CANDIDATE_MAX_BITS}_bits_and_thumb_corr_ge_{MIN_THUMB_CORR}": rule_counts["thumb_corr"],
    }

    all_boxes = [b for s in SPLITS for r in kept[s] for b in r["boxes"]]
    areas = np.array([w * h for _, _, _, w, h in all_boxes]) if all_boxes else np.zeros(1)
    degenerate = float((areas > DEGENERATE_AREA).mean())
    box_stats = {
        "total_boxes": len(all_boxes),
        "degenerate_frac": round(degenerate, 4),
        "area_quantiles": {q: round(float(np.quantile(areas, q / 100)), 4) for q in (5, 25, 50, 75, 95)},
        "tiny_frac_under_1pct_area": round(float((areas < 0.01).mean()), 4),
    }

    print("  drawing label-check grids ...", flush=True)
    grids = [p for s in SPLITS for p in save_grids(kept[s], s, names, plot_dir, args.seed)]
    list_splits = {s for s in SPLITS if mode != "provided" or drop[s]}
    yaml_path = write_yaml(root, names, kept, list_splits, cfg_dir)

    total_imgs = sum(len(kept[s]) for s in SPLITS)
    missing = sum(v["missing_labels"] for v in split_report.values())
    orphan_total = sum(orphans.values())
    fatal_label = Counter()
    for v in split_report.values():
        fatal_label.update({k: n for k, n in v["label_issues"].items()
                            if k in ("malformed_lines", "bad_class_id", "out_of_range_coords")})
    corrupt = sum(len(v["corrupt_images"]) for v in split_report.values())
    zero_size = sum(v["label_issues"].get("zero_size_boxes", 0) for v in split_report.values())
    zero_warning = (f"{zero_size} zero-size boxes ignored - YOLO drops them in training"
                    if zero_size else None)
    train_min = min(split_report["train"]["boxes_per_class"].values())
    thin = [f"{n} in {s} ({split_report[s]['boxes_per_class'][n]})" for s in ("valid", "test") for n in names
            if split_report[s]["boxes_per_class"][n] < MIN_EVAL_BOXES]
    mismatch = sum(v["filename_class_mismatch"] for v in split_report.values())

    checks = {
        "label_pairing": {"status": "pass" if missing / total_imgs < MAX_MISSING_LABEL_FRAC else "fail",
                          "detail": f"{missing} images without a label file; {orphan_total} label files "
                                    f"without an image in the dataset folders"},
        "label_format": {"status": "pass" if not fatal_label and not corrupt else "fail",
                         "detail": f"fatal label issues {dict(fatal_label) or 'none'}, corrupt images {corrupt}"
                                   + (f"; warning: {zero_warning}" if zero_warning else ""),
                         "warning": zero_warning},
        "split_leakage": {"status": "pass" if eval_leak < MAX_EVAL_LEAK_FRAC else "fail",
                          "detail": f"{eval_leak:.2%} of valid+test images duplicated in another split; "
                                    f"excluded {len(drop['train'])} train and {len(drop['valid'])} valid images"},
        "degenerate_boxes": {"status": "pass" if degenerate < MAX_DEGENERATE_FRAC else "fail",
                             "detail": f"{degenerate:.2%} of boxes cover more than {DEGENERATE_AREA:.0%} of the image"},
        "class_balance": {"status": "pass" if train_min >= MIN_TRAIN_BOXES and not thin else "fail",
                          "detail": f"smallest train class has {train_min} boxes"
                                    + (f"; under {MIN_EVAL_BOXES} boxes: {', '.join(thin)}" if thin else
                                       f"; every class has >= {MIN_EVAL_BOXES} boxes in valid and test")},
    }
    if not args.skip_ultralytics:
        print("  running the Ultralytics dataset loader ...", flush=True)
        expected = {s: len(kept[s]) for s in SPLITS}
        checks["ultralytics_loader"] = ultralytics_check(yaml_path, expected)
    checks["visual_check"] = {"status": "manual", "detail": f"inspect {len(grids)} grids in {show_path(plot_dir)}"}

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_root": root.name,
        "class_names": names,
        "split_used": split_desc,
        "splits": split_report,
        "orphan_labels_by_folder": orphans,
        "provided_split_boxes_per_class": provided_balance,
        "box_stats": box_stats,
        "duplicates": {"groups": len(groups), "images_in_groups": sum(len(g) for g in groups),
                       "matched_pairs": matched_pairs,
                       "near_duplicate_pairs": len(near_pairs),
                       "near_duplicate_sheet": dup_sheet.name if dup_sheet else None},
        "filename_class_mismatch": mismatch,
        "leakage": {"by_pair": pairs, "eval_leak_frac": round(eval_leak, 4),
                    "excluded": {s: [r["name"] for r in drop[s]] for s in ("train", "valid")}},
        "checks": checks,
    }
    (out_dir / "dataset_report.json").write_text(json.dumps(report, indent=2))

    print("\nPer-class boxes (train / valid / test):")
    for n in names:
        print(f"  {n:<16}" + " / ".join(f"{split_report[s]['boxes_per_class'][n]:>5}" for s in SPLITS))
    print("Images: " + " / ".join(f"{s} {split_report[s]['images']}" for s in SPLITS))
    print(f"\nBox area quantiles (fraction of image): {box_stats['area_quantiles']}")
    print(f"Duplicate groups: {len(groups)} ({sum(len(g) for g in groups)} images); matched pairs by rule: "
          + ", ".join(f"{k} {v}" for k, v in matched_pairs.items()))
    print(f"Images whose labels lack their filename class: {mismatch}")
    print("\nChecks:")
    for k, v in checks.items():
        print(f"  [{v['status'].upper():^7}] {k:<18} {v['detail']}")
    if zero_warning:
        print(f"\nWarning: {zero_warning} (not counted in any box statistic)")
    print(f"\nWrote {out_dir / 'dataset_report.json'} and {yaml_path}")
    return 1 if any(v["status"] == "fail" for v in checks.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
