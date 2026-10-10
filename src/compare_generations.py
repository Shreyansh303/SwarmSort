r"""Gen 1 vs Gen 2 on the same test images, restricted to the 6 classes both generations know.

    python src/compare_generations.py subset --data <gen2 cfg>/data_test_real_world.yaml --split test \
        --classes 0-5 --out <work>/g1g2_real_world
    python src/compare_generations.py summary --results results/gen2/test_g1g2_real_world_results.json \
        results/gen2/test_g1g2_studio_results.json --out results/gen2/test_g1g2_summary.md

subset: a derived copy of one split of a data yaml that keeps only the first N classes (indices unchanged, so the
  class ids of every model stay valid). For each image of the split it writes <out>/images/<file> (a hard link to
  the original, else a symbolic link, else a copy) and <out>/labels/<stem>.txt with the label lines of the other
  classes removed (Gen 2's OTHER, class 6). An image whose labels were all removed stays in, with an empty label file:
  it has no shared-class objects, the same for every model. Then <out>/lists/<split>.txt, <out>/data.yaml (nc N,
  the first N names; train, val and test all point at the one list, as Ultralytics needs train and val keys: it is
  for evaluation only) and <out>/subset_report.json (counts of kept and removed boxes).
  The originals are only read. <out> must be new, empty, or an earlier subset (it is then replaced).

summary: the headline table of the evaluate.py reports of the six arms (G1A ... G2C) and the key pairwise verdicts:
  G2X vs G1X for each arm letter X, and the best Gen 2 arm vs the best Gen 1 arm, where "best" is chosen by each
  arm's own validation mAP@50 (the val_json of the arms table), never by the test numbers being compared.
"""
import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

REPORT = "subset_report.json"


class SubsetError(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------------
# subset
# ---------------------------------------------------------------------------------------------------------------
def parse_classes(text):
    """'0-5' or '0,1,2,3,4,5' or '0 1 2' -> [0, ..., 5]. Only the first N classes can be kept (0 .. N-1), so that
    every class keeps its index."""
    text = str(text).strip()
    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", text)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        classes = list(range(lo, hi + 1))
    else:
        try:
            classes = sorted({int(t) for t in re.split(r"[,\s]+", text) if t})
        except ValueError:
            raise SubsetError(f"--classes {text!r}: expected a range such as 0-5 or a list such as 0,1,2") from None
    if not classes or classes != list(range(len(classes))):
        raise SubsetError(f"--classes {text!r}: only the first N classes can be kept (0 to N-1, e.g. 0-5), so that "
                          f"every class keeps its index")
    return classes


def class_names(data):
    names = data["names"]
    return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)


def split_images(data_yaml, split):
    """Image paths of one split of a data yaml whose split entry is a .txt list (as every build here writes)."""
    data_yaml = Path(data_yaml)
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    if split not in data:
        raise SubsetError(f"{data_yaml} has no '{split}' entry")
    lst = Path(str(data[split]))
    if not lst.is_absolute():
        base = Path(str(data.get("path") or data_yaml.parent))
        lst = (base if base.is_absolute() else data_yaml.parent / base) / lst
    if lst.suffix != ".txt" or not lst.is_file():
        raise SubsetError(f"{data_yaml}: the '{split}' entry {lst} is not an image list (.txt file)")
    images = []
    for line in lst.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            p = Path(line[2:]) if line.startswith("./") else Path(line)
            images.append(p if p.is_absolute() else lst.parent / p)
    return data, images


def label_file(image_path):
    """The label file Ultralytics reads for an image (last /images/ -> /labels/, extension -> .txt)."""
    from ultralytics.data.utils import img2label_paths

    return Path(img2label_paths([str(image_path).replace("/", os.sep)])[0])  # separators as Ultralytics' loader


def link_or_copy(src, dest):
    """Hard link, else symbolic link, else copy; returns which one was made."""
    try:
        os.link(src, dest)
        return "hardlink"
    except OSError:
        pass
    try:
        os.symlink(Path(src).resolve(), dest)
        return "symlink"
    except OSError:
        shutil.copy2(src, dest)
        return "copy"


def prepare_out(out):
    out = Path(out)
    if out.exists():
        if not out.is_dir():
            raise SubsetError(f"--out {out} is a file")
        if any(out.iterdir()) and not (out / REPORT).is_file():
            raise SubsetError(f"--out {out} is not empty and is not an earlier subset (no {REPORT}); pick a new folder")
        for sub in ("images", "labels", "lists"):
            shutil.rmtree(out / sub, ignore_errors=True)
        for f in (REPORT, "data.yaml", "labels.cache"):
            (out / f).unlink(missing_ok=True)
    for sub in ("images", "labels", "lists"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    return out


def filter_label_lines(text, keep):
    """(kept lines, Counter of removed class ids) for the text of one YOLO label file."""
    kept, removed = [], Counter()
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            c = int(float(parts[0]))
        except ValueError:
            raise SubsetError(f"bad label line {line!r}") from None
        if c in keep:
            kept.append(line.strip())
        else:
            removed[c] += 1
    return kept, removed


def subset_dataset(data_yaml, split, classes, out, log=print):
    """Write the derived split (see the module docstring); returns the report dict."""
    data_yaml = Path(data_yaml).resolve()
    out_resolved = Path(out).resolve()
    data, images = split_images(data_yaml, split)
    names = class_names(data)
    if len(classes) > len(names):
        raise SubsetError(f"--classes keeps {len(classes)} classes, but {data_yaml.name} has only {len(names)}")
    if not images:
        raise SubsetError(f"the '{split}' list of {data_yaml} is empty")
    by_name = {}
    for p in images:
        if p.name in by_name and by_name[p.name] != p:
            raise SubsetError(f"two images share the file name {p.name}: {by_name[p.name]} and {p}")
        by_name[p.name] = p
    if out_resolved == data_yaml.parent or any(out_resolved == p.parent.parent for p in images[:1]):
        raise SubsetError(f"--out {out_resolved} must be a new folder, not the dataset's own")
    out = prepare_out(out_resolved)
    keep = set(classes)
    kept_per_class, removed_per_class, links = Counter(), Counter(), Counter()
    no_label_file, emptied, already_empty = 0, 0, 0
    lines = []
    for name, src in by_name.items():
        if not src.is_file():
            raise SubsetError(f"listed image not found: {src}")
        dest = out / "images" / name
        links[link_or_copy(src, dest)] += 1
        lbl = label_file(src)
        if lbl.is_file():
            kept, removed = filter_label_lines(lbl.read_text(encoding="utf-8"), keep)
        else:
            kept, removed = [], Counter()
            no_label_file += 1
        if removed and not kept:
            emptied += 1
        elif not kept:
            already_empty += 1
        kept_per_class.update(int(float(line.split()[0])) for line in kept)
        removed_per_class.update(removed)
        (out / "labels" / f"{Path(name).stem}.txt").write_text("".join(f"{line}\n" for line in kept),
                                                                 encoding="utf-8")
        lines.append(dest.as_posix())
    lst = out / "lists" / f"{split}.txt"
    lst.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    cfg = {"path": out.as_posix(), "train": lst.as_posix(), "val": lst.as_posix(), "test": lst.as_posix(),
           "nc": len(classes), "names": names[:len(classes)]}
    (out / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    def by_class(counter):
        return {(names[c] if c < len(names) else str(c)): n for c, n in sorted(counter.items())}

    report = {"source_data": data_yaml.as_posix(), "split": split, "classes": list(classes),
              "names": names[:len(classes)], "source_names": names, "out": out.as_posix(),
              "data_yaml": (out / "data.yaml").as_posix(), "images": len(lines),
              "boxes_kept": sum(kept_per_class.values()), "boxes_kept_per_class": by_class(kept_per_class),
              "boxes_removed": sum(removed_per_class.values()), "boxes_removed_per_class": by_class(removed_per_class),
              "images_emptied": emptied, "images_without_objects_before": already_empty,
              "images_without_label_file": no_label_file, "links": dict(links)}
    (out / REPORT).write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"Subset of {data_yaml.name} ({split}): {report['images']} images, kept classes {classes} "
        f"({', '.join(report['names'])}); {report['boxes_kept']} boxes kept, {report['boxes_removed']} removed "
        f"{report['boxes_removed_per_class']}; {emptied} images had only removed classes and stay in with no objects; "
        f"images as {dict(links)} -> {out / 'data.yaml'}")
    return report


# ---------------------------------------------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------------------------------------------
def pair_diff(report, first, second, key):
    """second - first on one metric from evaluate.py's pairwise list: (observed diff, [lo, hi], significant)."""
    for p in report["pairwise"]:
        if (p["a"], p["b"]) == (first, second):
            r = p[key]
            return r["observed_diff"], list(r["ci95"]), r["significant"]
        if (p["a"], p["b"]) == (second, first):  # stored the other way round: flip the sign
            r = p[key]
            return -r["observed_diff"], [-r["ci95"][1], -r["ci95"][0]], r["significant"]
    raise KeyError(f"no pair {first}/{second} in the report")


def best_by_val(report, prefix):
    """The arm with this label prefix (G1 / G2) with the highest own validation mAP@50 (None if unknown)."""
    vals = {label: v.get("map50") for label, v in report.get("val_metrics", {}).items()
            if label.startswith(prefix) and label in report["models"] and v.get("map50") is not None}
    return max(vals, key=lambda k: (vals[k], k)) if vals else None


def verdict_line(report, first, second):
    names = {label: m["name"] for label, m in report["models"].items()}
    cells = []
    for key, title in (("map50", "mAP@50"), ("map50_95", "mAP@50-95")):
        d, (lo, hi), sig = pair_diff(report, first, second, key)
        if sig:
            better = names[second] if lo > 0 else names[first]
            cells.append(f"Δ {title} {d:+.3f} [{lo:+.3f}, {hi:+.3f}], {better} significantly better")
        else:
            cells.append(f"Δ {title} {d:+.3f} [{lo:+.3f}, {hi:+.3f}], not significant")
    return f"{second} − {first} ({names[second]} − {names[first]}): " + "; ".join(cells)


def summary_markdown(reports):
    """reports: {tag: evaluate.py results JSON}. Returns the headline markdown."""
    tags = list(reports)
    labels = list(next(iter(reports.values()))["models"])
    lines = ["## Gen 1 vs Gen 2 on the 6 shared classes", "",
             "Same test images for all six models, labels restricted to the 6 shared classes, Gen 2's OTHER "
             "predictions dropped; each model at its own training image size. Cells: test mAP@50 [95% paired-"
             "bootstrap CI] / mAP@50-95.", "",
             "| Arm | imgsz | " + " | ".join(f"{t} ({reports[t]['images']} images)" for t in tags) + " |",
             "|---|---|" + "---|" * len(tags)]
    for label in labels:
        first = reports[tags[0]]
        name = first["models"][label]["name"]
        size = first["settings"]["imgsz"]
        size = size.get(label, "-") if isinstance(size, dict) else size
        cells = []
        for t in tags:
            m = reports[t]["models"][label]
            lo, hi = m["bootstrap"]["map50_ci95"]
            cells.append(f"{m['metrics']['map50']:.3f} [{lo:.3f}, {hi:.3f}] / {m['metrics']['map50_95']:.3f}")
        lines.append(f"| {label}: {name} | {size} | " + " | ".join(cells) + " |")
    lines += ["", "### Key verdicts (paired bootstrap, 95% CI; a CI that contains 0 is not significant)"]
    for t in tags:
        r = reports[t]
        lines += ["", f"**{t}**", ""]
        for x in sorted({label[2:] for label in labels if label.startswith(("G1", "G2"))}):
            if f"G1{x}" in r["models"] and f"G2{x}" in r["models"]:
                lines.append(f"- {verdict_line(r, f'G1{x}', f'G2{x}')}")
        b1, b2 = best_by_val(r, "G1"), best_by_val(r, "G2")
        if b1 and b2:
            lines.append(f"- Best vs best (chosen by each arm's own val mAP@50, not by these test numbers): "
                         f"{verdict_line(r, b1, b2)}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("subset", help="derived split with only the first N classes (see above)")
    s.add_argument("--data", required=True, help="data yaml whose split is a .txt image list")
    s.add_argument("--split", choices=("test", "val", "train"), default="test")
    s.add_argument("--classes", required=True, help="classes to keep: the first N, e.g. 0-5")
    s.add_argument("--out", required=True, help="new folder for images/, labels/, lists/, data.yaml")
    m = sub.add_parser("summary", help="headline table and key verdicts of the evaluate.py reports")
    m.add_argument("--results", nargs="+", required=True, help="evaluate.py <split>_<tag>_results.json files")
    m.add_argument("--out", help="also write the markdown here")
    args = ap.parse_args(argv)
    try:  # the summary prints Δ and −; a Windows console's code page cannot encode them
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    if args.command == "subset":
        try:
            return subset_dataset(args.data, args.split, parse_classes(args.classes), args.out)
        except SubsetError as e:
            raise SystemExit(f"Error: {e}")
    reports = {}
    for path in args.results:
        r = json.loads(Path(path).read_text(encoding="utf-8"))
        reports[r.get("tag") or Path(path).stem] = r
    text = summary_markdown(reports)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return text


if __name__ == "__main__":
    main()
