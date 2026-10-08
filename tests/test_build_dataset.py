"""Tests of src/build_dataset.py on tiny synthetic YOLO and COCO datasets built in a temp folder."""
import contextlib
import csv
import io
import json
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import build_dataset as bd  # noqa: E402
import training  # noqa: E402

RED = (255, 0, 0)


def noise_image(w, h, seed, rect=None):
    """Grey noise (so no two images look alike), optionally with a pure red rectangle (x0, y0, x1, y1) in pixels."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(60, 180, size=(h, w, 3), dtype=np.uint8)
    if rect:
        x0, y0, x1, y1 = rect
        arr[y0:y1, x0:x1] = RED
    return Image.fromarray(arr)


def red_box(img):
    """Normalized (x0, y0, x1, y1) of the red pixels of an image."""
    a = np.asarray(img.convert("RGB")).astype(int)
    mask = (a[..., 0] > 190) & (a[..., 1] < 80) & (a[..., 2] < 80)
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h


def yolo_to_xyxy(line):
    c, cx, cy, w, h = line.split()
    cx, cy, w, h = map(float, (cx, cy, w, h))
    return int(c), (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def make_yolo(root, names, images, data_yaml=True):
    """images: {relative image path under root: (PIL image, label text)}; folders are created as needed."""
    for rel, (img, label) in images.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        img.save(p, quality=95)
        lbl = bd.label_path_for(p)
        lbl.parent.mkdir(parents=True, exist_ok=True)
        if label is not None:
            lbl.write_text(label)
    if data_yaml:
        (root / "data.yaml").write_text(yaml.safe_dump({"names": names, "nc": len(names)}))


def make_coco(folder, categories, images, ann_name="annotations.json"):
    """images: list of (file_name, PIL image or None if already saved, recorded (w, h), [(cat_id, bbox, iscrowd)])."""
    data = {"images": [], "annotations": [], "categories": [{"id": i, "name": n, "supercategory": s}
                                                            for i, (n, s) in categories.items()]}
    for k, (fname, img, size, anns) in enumerate(images):
        p = folder / fname
        p.parent.mkdir(parents=True, exist_ok=True)
        if img is not None:
            img.save(p, quality=95)
        data["images"].append({"id": k + 1, "file_name": fname, "width": size[0], "height": size[1]})
        for cat, bbox, crowd in anns:
            data["annotations"].append({"id": len(data["annotations"]) + 1, "image_id": k + 1, "category_id": cat,
                                        "bbox": bbox, "iscrowd": crowd, "area": bbox[2] * bbox[3]})
    (folder / ann_name).write_text(json.dumps(data))


def write_config(path, sources, classes=("BIO", "GLASS", "PLASTIC")):
    path.write_text(yaml.safe_dump({"classes": list(classes), "sources": sources}, sort_keys=False))
    return path


def yolo_source(name="studio_src", locate="YOLO_DS", **kw):
    return {"name": name, "format": "yolo", "locate": [locate], "domain": "studio", "licence": "test",
            "class_map": {"bio": "BIO", "glass": "GLASS", "plastic": "PLASTIC"}, **kw}


def coco_source(name="real_src", **kw):
    return {"name": name, "format": "coco", "annotations": "annotations.json", "locate": ["COCO_DS"],
            "domain": "real_world", "licence": "test", "class_map": {"Bottle": "GLASS", "Bag": "PLASTIC"}, **kw}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def quiet_build(*args, **kwargs):
    kwargs.setdefault("log", lambda *a, **k: None)
    return bd.build(*args, **kwargs)


class BuildTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.search = self.tmp / "input"
        self.search.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def standard_yolo(self, n=20, root=None):
        root = root or self.search / "YOLO_DS"
        imgs = {}
        for i in range(n):
            split = ("train", "valid", "test")[i % 3]
            cls = i % 3
            imgs[f"{split}/images/img{i:03d}.jpg"] = (noise_image(64, 48, i), f"{cls} 0.5 0.5 0.4 0.3\n")
        make_yolo(root, ["bio", "glass", "plastic"], imgs)
        return root

    def standard_coco(self, n=20):
        folder = self.search / "COCO_DS"
        images = []
        for i in range(n):
            images.append((f"batch_{i % 2}/{i:06d}.jpg", noise_image(80, 60, 1000 + i, rect=(10, 10, 30, 40)),
                           (80, 60), [(1 + i % 2, [10, 10, 20, 30], 0)]))
        make_coco(folder, {1: ("Bottle", "Bottle"), 2: ("Bag", "Bag")}, images)
        return folder


class MergeTests(BuildTestCase):
    def test_yolo_and_coco_merge(self):
        self.standard_yolo()
        self.standard_coco()
        cfg = write_config(self.tmp / "s.yaml", [yolo_source(), coco_source()])
        out = self.tmp / "out"
        report = quiet_build(cfg, out, seed=1, search_root=self.search)
        rows = read_csv(out / "split.csv")
        self.assertEqual(len(rows), 40)
        self.assertEqual(Counter(r["source"] for r in rows), {"studio_src": 20, "real_src": 20})
        self.assertTrue(all(r["file"].startswith(r["source"] + "__") for r in rows))
        data = yaml.safe_load((out / "data.yaml").read_text())
        self.assertEqual(data["names"], ["BIO", "GLASS", "PLASTIC"])
        self.assertEqual(data["nc"], 3)
        listed = []
        for key in ("train", "val", "test"):
            lst = Path(data[key])
            self.assertTrue(lst.is_absolute())
            listed += lst.read_text().splitlines()
        self.assertEqual(sorted(Path(p).name for p in listed), sorted(r["file"] for r in rows))
        for p in listed:
            self.assertTrue(Path(p).is_file())
            self.assertTrue((out / "labels" / (Path(p).stem + ".txt")).is_file())
        self.assertEqual(report["totals"]["train"]["images"] + report["totals"]["valid"]["images"]
                         + report["totals"]["test"]["images"], 40)
        self.assertEqual(report["sources"]["real_src"]["licence"], "test")
        # the COCO batch_0/000000.jpg file name keeps its folder, so names cannot collide across batches
        self.assertIn("real_src__batch_0_000000.jpg", {r["file"] for r in rows})

    def test_finds_source_in_kaggle_style_nesting(self):
        self.standard_yolo(root=self.search / "datasets" / "owner" / "slug-name" / "YOLO_DS")
        cfg = write_config(self.tmp / "s.yaml", [yolo_source()])
        report = quiet_build(cfg, self.tmp / "out", search_root=self.search)
        self.assertEqual(report["sources"]["studio_src"]["images_kept"], 20)
        self.assertIn("datasets/owner/slug-name/YOLO_DS", report["sources"]["studio_src"]["location"])

    def test_refuses_foreign_non_empty_out_dir(self):
        self.standard_yolo()
        out = self.tmp / "out"
        out.mkdir()
        (out / "precious.txt").write_text("keep me")
        cfg = write_config(self.tmp / "s.yaml", [yolo_source()])
        with self.assertRaises(bd.BuildError):
            quiet_build(cfg, out, search_root=self.search)
        self.assertTrue((out / "precious.txt").is_file())

    def test_unknown_config_key_is_an_error(self):
        cfg = write_config(self.tmp / "s.yaml", [yolo_source(clas_map={})])
        with self.assertRaisesRegex(bd.BuildError, "clas_map"):
            bd.load_config(cfg)

    def test_seven_classes_with_other(self):
        self.standard_yolo()
        src = yolo_source(class_map={"bio": "BIO", "glass": "GLASS", "plastic": "OTHER"})
        cfg = write_config(self.tmp / "s.yaml", [src], classes=("BIO", "GLASS", "PLASTIC", "A", "B", "C", "OTHER"))
        out = self.tmp / "out"
        report = quiet_build(cfg, out, search_root=self.search)
        self.assertEqual(yaml.safe_load((out / "data.yaml").read_text())["nc"], 7)
        other = sum(report["totals"][s]["boxes_per_class"]["OTHER"] for s in bd.SPLITS)
        self.assertEqual(other, 6)  # images 2, 5, ..., 17 have class 2 = plastic


class MappingTests(BuildTestCase):
    def make_mapping_source(self):
        names = ["keep", "skipbox", "banned", "other"]
        imgs = {}
        for i in range(12):
            imgs[f"train/images/k{i:02d}.jpg"] = (noise_image(64, 48, i), "0 0.5 0.5 0.2 0.2\n")
        imgs["train/images/mixed.jpg"] = (noise_image(64, 48, 100), "0 0.3 0.3 0.2 0.2\n1 0.7 0.7 0.2 0.2\n")
        imgs["train/images/only_skip.jpg"] = (noise_image(64, 48, 101), "1 0.5 0.5 0.2 0.2\n")
        imgs["train/images/has_banned.jpg"] = (noise_image(64, 48, 102), "0 0.3 0.3 0.2 0.2\n2 0.6 0.6 0.2 0.2\n")
        imgs["train/images/other.jpg"] = (noise_image(64, 48, 103), "3 0.5 0.5 0.2 0.2\n")
        make_yolo(self.search / "MAP_DS", names, imgs)

    def test_drop_annotation_and_drop_image(self):
        self.make_mapping_source()
        src = yolo_source(locate="MAP_DS", class_map={"keep": "BIO", "skipbox": "drop_annotation",
                                                      "banned": "drop_image", "other": "PLASTIC"})
        cfg = write_config(self.tmp / "s.yaml", [src])
        out = self.tmp / "out"
        report = quiet_build(cfg, out, search_root=self.search)
        files = {r["file"] for r in read_csv(out / "split.csv")}
        self.assertNotIn("studio_src__has_banned.jpg", files)
        self.assertNotIn("studio_src__only_skip.jpg", files)  # nothing left after drop_annotation
        self.assertIn("studio_src__mixed.jpg", files)
        mixed = (out / "labels" / "studio_src__mixed.txt").read_text().splitlines()
        self.assertEqual(len(mixed), 1)
        self.assertEqual(yolo_to_xyxy(mixed[0])[0], 0)
        self.assertEqual((out / "labels" / "studio_src__other.txt").read_text().split()[0], "2")
        s = report["sources"]["studio_src"]
        self.assertEqual(s["images_dropped"], {"drop_image:banned": 1, "no_boxes_left": 1})
        self.assertEqual(s["annotations_dropped"], {"drop_annotation:skipbox": 2})
        self.assertEqual(s["images_kept"], 14)

    def test_unmapped_class_is_an_error_listing_the_names(self):
        self.make_mapping_source()
        src = yolo_source(locate="MAP_DS", class_map={"keep": "BIO", "other": "GLASS"})
        cfg = write_config(self.tmp / "s.yaml", [src])
        with self.assertRaises(bd.BuildError) as ctx:
            quiet_build(cfg, self.tmp / "out", search_root=self.search)
        msg = str(ctx.exception)
        self.assertIn("unmapped classes", msg)
        self.assertIn("skipbox", msg)
        self.assertIn("banned", msg)
        self.assertFalse((self.tmp / "out").exists())  # checked before any image is written

    def test_unknown_target_class_is_an_error(self):
        cfg = write_config(self.tmp / "s.yaml", [yolo_source(class_map={"bio": "BIOLOGICAL"})])
        with self.assertRaisesRegex(bd.BuildError, "BIOLOGICAL"):
            bd.load_config(cfg)

    def test_coco_crowd_invalid_and_supercategory(self):
        folder = self.search / "COCO_DS"
        images = [("a.jpg", noise_image(100, 50, 1), (100, 50),
                   [(1, [10, 10, 20, 20], 0), (1, [0, 0, 50, 50], 1), (2, [5, 5, 0, 10], 0), (2, [60, 5, 10, 10], 0)])]
        images += [(f"f{i}.jpg", noise_image(100, 50, 10 + i), (100, 50), [(1, [1, 1, 5, 5], 0)]) for i in range(10)]
        make_coco(folder, {1: ("Clear bottle", "Bottle"), 2: ("Plastic bag", "Bag")}, images)
        src = coco_source(class_field="supercategory")
        cfg = write_config(self.tmp / "s.yaml", [src])
        out = self.tmp / "out"
        report = quiet_build(cfg, out, search_root=self.search)
        self.assertEqual(report["sources"]["real_src"]["annotations_dropped"], {"crowd": 1, "invalid_bbox": 1})
        lines = (out / "labels" / "real_src__a.txt").read_text().splitlines()
        self.assertEqual(sorted(int(line.split()[0]) for line in lines), [1, 2])


class BoxTests(BuildTestCase):
    def test_coco_to_yolo_exact(self):
        boxes, warns = bd.coco_to_yolo([(2, [10, 20, 30, 40])], (200, 100), (200, 100), 1, "auto")
        cls, cx, cy, w, h = boxes[0]
        self.assertEqual(cls, 2)
        for got, want in zip((cx, cy, w, h), (25 / 200, 40 / 100, 30 / 200, 40 / 100)):
            self.assertAlmostEqual(got, want, places=12)
        self.assertEqual(warns, Counter())
        # a box running over the edge is clipped and counted
        boxes, warns = bd.coco_to_yolo([(0, [190, 90, 20, 20])], (200, 100), (200, 100), 1, "auto")
        self.assertAlmostEqual(boxes[0][1], 0.975)
        self.assertAlmostEqual(boxes[0][3], 0.05)
        self.assertEqual(warns["clipped_boxes"], 1)

    def test_recorded_size_differs_from_actual(self):
        # recorded 400x200 but the file is 200x100: same aspect, so normalized boxes are unchanged
        boxes, warns = bd.coco_to_yolo([(0, [40, 20, 80, 40])], (400, 200), (200, 100), 1, "auto")
        self.assertEqual([round(v, 9) for v in boxes[0][1:]], [0.2, 0.2, 0.2, 0.2])
        self.assertEqual(warns["recorded_size_rescaled"], 1)
        with self.assertRaises(ValueError):
            bd.coco_to_yolo([(0, [1, 1, 2, 2])], (300, 300), (200, 100), 1, "auto")

    def test_end_to_end_label_values(self):
        folder = self.search / "COCO_DS"
        images = [(f"{i}.jpg", noise_image(160, 80, i), (160, 80), [(1, [16, 8, 48, 24], 0)]) for i in range(10)]
        make_coco(folder, {1: ("Bottle", "Bottle"), 2: ("Bag", "Bag")}, images)
        cfg = write_config(self.tmp / "s.yaml", [coco_source()])
        out = self.tmp / "out"
        quiet_build(cfg, out, search_root=self.search)
        self.assertEqual((out / "labels" / "real_src__0.txt").read_text(), "1 0.250000 0.250000 0.300000 0.300000\n")

    def test_orient_point_matches_pillow_for_every_orientation(self):
        raw = noise_image(90, 50, 7, rect=(10, 5, 30, 20))
        x0, y0, x1, y1 = red_box(raw)
        for orientation in range(1, 9):
            exif = Image.Exif()
            exif[0x0112] = orientation
            buf = io.BytesIO()
            raw.save(buf, format="PNG", exif=exif.tobytes())
            shown = ImageOps.exif_transpose(Image.open(io.BytesIO(buf.getvalue())))
            pts = [bd.orient_point(x, y, orientation) for x, y in ((x0, y0), (x1, y1))]
            want = (min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts))
            for got, exp in zip(red_box(shown), want):
                self.assertAlmostEqual(got, exp, places=6, msg=f"orientation {orientation}")

    def make_exif6(self, recorded_display):
        """One JPEG stored 200x100 with EXIF orientation 6 (shown rotated 90 degrees clockwise, 100x200)."""
        folder = self.search / "COCO_DS"
        folder.mkdir(parents=True, exist_ok=True)
        raw = noise_image(200, 100, 5, rect=(20, 10, 60, 40))
        exif = Image.Exif()
        exif[0x0112] = 6
        raw.save(folder / "rot.jpg", quality=95, exif=exif.tobytes())
        shown = ImageOps.exif_transpose(Image.open(folder / "rot.jpg"))
        self.assertEqual(shown.size, (100, 200))
        if recorded_display:  # annotated on the rotated photo (TACO's convention)
            x0, y0, x1, y1 = red_box(shown)
            bbox = [x0 * 100, y0 * 200, (x1 - x0) * 100, (y1 - y0) * 200]
            size = (100, 200)
        else:  # annotated on the stored pixels
            bbox, size = [20, 10, 40, 30], (200, 100)
        images = [("rot.jpg", None, size, [(1, bbox, 0)])]
        images += [(f"f{i}.jpg", noise_image(80, 60, 50 + i), (80, 60), [(2, [5, 5, 10, 10], 0)]) for i in range(10)]
        make_coco(folder, {1: ("Bottle", "Bottle"), 2: ("Bag", "Bag")}, images)

    def check_exif_alignment(self):
        cfg = write_config(self.tmp / "s.yaml", [coco_source()])
        out = self.tmp / "out"
        report = quiet_build(cfg, out, search_root=self.search)
        img = Image.open(out / "images" / "real_src__rot.jpg")
        self.assertEqual(img.size, (100, 200))
        self.assertNotIn(0x0112, img.getexif())  # written upright, no orientation tag left to re-apply
        _, box = yolo_to_xyxy((out / "labels" / "real_src__rot.txt").read_text())
        for got, want in zip(box, red_box(img)):
            self.assertAlmostEqual(got, want, delta=0.025)
        self.assertEqual(report["sources"]["real_src"]["warnings"].get("exif_rotated_images"), 1)

    def test_exif_rotated_coco_boxes_raw_frame(self):
        self.make_exif6(recorded_display=False)
        self.check_exif_alignment()

    def test_exif_rotated_coco_boxes_display_frame(self):
        self.make_exif6(recorded_display=True)
        self.check_exif_alignment()

    def test_downscale_keeps_normalized_boxes(self):
        folder = self.search / "COCO_DS"
        images = [(f"{i}.jpg", noise_image(400, 200, i, rect=(100, 50, 200, 150)), (400, 200),
                   [(1, [100, 50, 100, 100], 0)]) for i in range(10)]
        make_coco(folder, {1: ("Bottle", "Bottle"), 2: ("Bag", "Bag")}, images)
        cfg = write_config(self.tmp / "s.yaml", [coco_source()])
        out = self.tmp / "out"
        quiet_build(cfg, out, search_root=self.search, max_side=100)
        img = Image.open(out / "images" / "real_src__0.jpg")
        self.assertEqual(img.size, (100, 50))
        self.assertEqual((out / "labels" / "real_src__0.txt").read_text(), "1 0.375000 0.500000 0.250000 0.500000\n")
        _, box = yolo_to_xyxy((out / "labels" / "real_src__0.txt").read_text())
        for got, want in zip(box, red_box(img)):
            self.assertAlmostEqual(got, want, delta=0.03)


class SplitTests(BuildTestCase):
    def test_duplicates_stay_in_one_split(self):
        root = self.standard_yolo(n=30)
        dup = noise_image(64, 48, 999)
        bright = Image.fromarray(np.clip(np.asarray(dup).astype(int) + 6, 0, 255).astype(np.uint8))
        extra = {f"train/images/dup_{k}.jpg": (dup, "0 0.5 0.5 0.4 0.3\n") for k in range(4)}
        extra["valid/images/dup_bright.jpg"] = (bright, "0 0.5 0.5 0.4 0.3\n")
        make_yolo(root, ["bio", "glass", "plastic"], extra)
        # the same picture in a second source, saved at another size
        self.standard_coco(n=10)
        make_coco(self.search / "COCO_DS", {1: ("Bottle", "Bottle"), 2: ("Bag", "Bag")},
                  [(f"batch_{i % 2}/{i:06d}.jpg", None, (80, 60), [(1, [10, 10, 20, 30], 0)]) for i in range(10)]
                  + [("copy.jpg", dup.resize((128, 96)), (128, 96), [(1, [10, 10, 20, 30], 0)])])
        for seed in (0, 1, 2, 3):
            out = self.tmp / f"out{seed}"
            cfg = write_config(self.tmp / "s.yaml", [yolo_source(), coco_source()])
            report = quiet_build(cfg, out, seed=seed, search_root=self.search)
            rows = {r["file"]: r for r in read_csv(out / "split.csv")}
            members = [f"studio_src__dup_{k}.jpg" for k in range(4)] + ["studio_src__dup_bright.jpg",
                                                                        "real_src__copy.jpg"]
            self.assertEqual(len({rows[m]["split"] for m in members}), 1, f"seed {seed}")
            self.assertEqual(len({rows[m]["group"] for m in members}), 1)
            self.assertGreaterEqual(report["duplicates"]["cross_source_groups"], 1)

    def test_keep_split_csv_is_honoured(self):
        self.standard_yolo(n=30)
        dup_of_test = Image.open(self.search / "YOLO_DS" / "valid/images/img001.jpg").copy()
        self.standard_coco(n=10)
        make_coco(self.search / "COCO_DS", {1: ("Bottle", "Bottle"), 2: ("Bag", "Bag")},
                  [(f"batch_{i % 2}/{i:06d}.jpg", None, (80, 60), [(1, [10, 10, 20, 30], 0)]) for i in range(10)]
                  + [("copy.jpg", dup_of_test, (64, 48), [(1, [10, 10, 20, 20], 0)])])
        pinned = {f"img{i:03d}.jpg": ("test" if i % 5 == 1 else "valid" if i % 5 == 2 else "train")
                  for i in range(30)}
        csv_path = self.tmp / "split.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["split", "folder", "file", "source_class"])
            for name, s in pinned.items():
                w.writerow([s, "train", name, "x"])
        cfg = write_config(self.tmp / "s.yaml", [yolo_source(keep_split_csv=str(csv_path)), coco_source()])
        out = self.tmp / "out"
        report = quiet_build(cfg, out, seed=3, search_root=self.search)
        rows = {r["file"]: r for r in read_csv(out / "split.csv")}
        for name, s in pinned.items():
            self.assertEqual(rows[f"studio_src__{Path(name).stem}.jpg"]["split"], s, name)
        self.assertEqual(rows["real_src__copy.jpg"]["split"], "test")  # follows its Gen 1 test duplicate
        self.assertEqual(report["sources"]["studio_src"]["keep_split_csv"]["pinned_images"], 30)

    def test_per_domain_test_lists(self):
        self.standard_yolo()
        self.standard_coco()
        cfg = write_config(self.tmp / "s.yaml", [yolo_source(), coco_source()])
        out = self.tmp / "out"
        quiet_build(cfg, out, seed=1, search_root=self.search)
        rows = read_csv(out / "split.csv")
        for domain in ("studio", "real_world"):
            want = sorted(r["file"] for r in rows if r["split"] == "test" and r["domain"] == domain)
            self.assertTrue(want)
            got = sorted(Path(p).name for p in (out / "lists" / f"test_{domain}.txt").read_text().splitlines())
            self.assertEqual(got, want)
            data = yaml.safe_load((out / f"data_test_{domain}.yaml").read_text())
            self.assertEqual(Path(data["test"]), (out / "lists" / f"test_{domain}.txt").resolve())
            self.assertEqual(data["train"], yaml.safe_load((out / "data.yaml").read_text())["train"])
        self.assertTrue((out / "data_test_src_real_src.yaml").is_file())

    def test_build_is_deterministic(self):
        self.standard_yolo()
        self.standard_coco()
        cfg = write_config(self.tmp / "s.yaml", [yolo_source(), coco_source()])
        a, b = self.tmp / "a", self.tmp / "b"
        ra = quiet_build(cfg, a, seed=7, search_root=self.search, max_images=15)
        rb = quiet_build(cfg, b, seed=7, search_root=self.search, max_images=15)
        self.assertEqual((a / "split.csv").read_bytes(), (b / "split.csv").read_bytes())
        for sub in ("images", "labels"):
            names = sorted(p.name for p in (a / sub).iterdir())
            self.assertEqual(names, sorted(p.name for p in (b / sub).iterdir()))
            for n in names:
                self.assertEqual((a / sub / n).read_bytes(), (b / sub / n).read_bytes(), n)
        for lst in (a / "lists").iterdir():
            self.assertEqual(lst.read_text().replace(a.resolve().as_posix(), ""),
                             (b / "lists" / lst.name).read_text().replace(b.resolve().as_posix(), ""))
        ra["settings"].pop("out"), rb["settings"].pop("out")
        self.assertEqual(json.dumps(ra), json.dumps(rb))
        # a rebuild into the same folder replaces the previous build
        quiet_build(cfg, a, seed=7, search_root=self.search, max_images=15)
        self.assertEqual((a / "split.csv").read_bytes(), (b / "split.csv").read_bytes())
        # another seed gives another split
        rc = quiet_build(cfg, self.tmp / "c", seed=8, search_root=self.search, max_images=15)
        self.assertNotEqual((self.tmp / "c" / "split.csv").read_bytes(), (a / "split.csv").read_bytes())
        self.assertEqual(rc["totals"]["train"]["images"] + rc["totals"]["valid"]["images"]
                         + rc["totals"]["test"]["images"], 30)

    def test_proxy_subset_works_on_the_output(self):
        self.standard_yolo(n=30)
        self.standard_coco(n=30)
        cfg = write_config(self.tmp / "s.yaml", [yolo_source(), coco_source()])
        out = self.tmp / "out"
        quiet_build(cfg, out, seed=1, search_root=self.search)
        stats, n_full, n_sub = training.build_proxy_subset(out / "data.yaml", out / "data_proxy.yaml",
                                                           out / "lists" / "train_proxy.txt", 0.4, 42,
                                                           out / "split.csv")
        self.assertEqual(n_sub, round(0.4 * n_full))
        self.assertTrue(all("/" in c for c in stats))  # strata are source/dominant_class
        self.assertEqual({c.split("/")[0] for c in stats}, {"studio_src", "real_src"})
        proxy = yaml.safe_load((out / "data_proxy.yaml").read_text())
        self.assertEqual(len(Path(proxy["train"]).read_text().splitlines()), n_sub)

    def test_main_reports_errors_without_traceback(self):
        cfg = write_config(self.tmp / "s.yaml", [yolo_source(locate="NOWHERE")])
        with self.assertRaises(SystemExit) as ctx, contextlib.redirect_stdout(io.StringIO()):
            bd.main(["--sources", str(cfg), "--out", str(self.tmp / "out"), "--search-root", str(self.search)])
        self.assertIn("NOWHERE", str(ctx.exception.code))


def make_supervisely(project, images, meta_tags=None):
    """images: {name: (PIL image, [object dicts])}; writes meta.json and ds0/ann + ds0/img under project."""
    meta_tags = meta_tags or {"Material": ["glass", "paper", "plastic"], "Object": ["bottle", "cigarettebutt"]}
    project.mkdir(parents=True, exist_ok=True)
    (project / "meta.json").write_text(json.dumps({
        "classes": [{"title": "litter", "shape": "rectangle"}],
        "tags": [{"name": t, "value_type": "oneof_string", "values": v} for t, v in meta_tags.items()]}))
    for d in ("ann", "img"):
        (project / "ds0" / d).mkdir(parents=True, exist_ok=True)
    for name, (img, objects) in images.items():
        img.save(project / "ds0" / "img" / name, quality=95)
        (project / "ds0" / "ann" / f"{name}.json").write_text(json.dumps(
            {"description": "", "tags": [], "size": {"width": img.size[0], "height": img.size[1]},
             "objects": objects}))


def sly_rect(x0, y0, x1, y1, **tags):
    return {"classTitle": "litter", "geometryType": "rectangle",
            "points": {"exterior": [[x0, y0], [x1, y1]], "interior": []},
            "tags": [{"name": k, "value": v} for k, v in tags.items()]}


def sly_source(**kw):
    return {"name": "hitl", "format": "supervisely", "locate": ["recycling-dataset"], "domain": "real_world",
            "licence": "CC0", "class_from_tag": "Material", "untagged": "OTHER",
            "tag_overrides": {"Object": {"cigarettebutt": "OTHER"}},
            "class_map": {"glass": "GLASS", "paper": "BIO", "plastic": "PLASTIC"}, **kw}


def make_labelme(folder, name, img, shapes, image_path=None, embed=False):
    folder.mkdir(parents=True, exist_ok=True)
    data = {"version": "4.5.6", "flags": {}, "shapes": shapes, "imagePath": image_path or name,
            "imageData": None, "imageHeight": img.size[1], "imageWidth": img.size[0]}
    if embed:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        data["imageData"] = __import__("base64").b64encode(buf.getvalue()).decode()
    else:
        img.save(folder / name, quality=95)
    (folder / (Path(name).stem + ".json")).write_text(json.dumps(data))


def lm_poly(label, points):
    return {"label": label, "points": points, "group_id": None, "shape_type": "polygon", "flags": {}}


def labelme_source(**kw):
    return {"name": "dwsd", "format": "labelme", "locate": ["dwsd"], "locate_match": "contains", "domain": "india",
            "licence": "CC BY 4.0", "force_split": "test",
            "class_map": {"plastic bottles": "PLASTIC", "glass": "GLASS", "cloth": "OTHER"}, **kw}


SEVEN = ("BIO", "GLASS", "PLASTIC", "A", "B", "C", "OTHER")


class SuperviselyTests(BuildTestCase):
    def make_project(self):
        project = self.search / "datasets" / "humansintheloop" / "recycling-dataset" / "Recycling Dataset"
        images = {
            "tag.jpg": (noise_image(200, 100, 1), [sly_rect(20, 10, 60, 50, Material="glass", Object="bottle")]),
            "untagged.jpg": (noise_image(200, 100, 2), [sly_rect(100, 20, 180, 80, Object="bottle")]),
            "butt.jpg": (noise_image(200, 100, 3), [sly_rect(10, 10, 30, 30, Material="paper",
                                                             Object="cigarettebutt")]),
            "poly.jpg": (noise_image(200, 100, 4), [{"classTitle": "litter", "geometryType": "polygon",
                                                     "points": {"exterior": [[50, 20], [150, 40], [100, 90]],
                                                                "interior": []},
                                                     "tags": [{"name": "Material", "value": "plastic"}]}]),
        }
        images.update({f"p{i}.jpg": (noise_image(200, 100, 10 + i), [sly_rect(40, 20, 80, 60, Material="paper")])
                       for i in range(8)})
        make_supervisely(project, images)
        return project

    def test_tag_class_untagged_and_override(self):
        self.make_project()
        cfg = write_config(self.tmp / "s.yaml", [sly_source()], classes=SEVEN)
        out = self.tmp / "out"
        report = quiet_build(cfg, out, search_root=self.search)
        lab = {n: (out / "labels" / f"hitl__Recycling_Dataset_ds0_{n}.txt").read_text().splitlines()
               for n in ("tag", "untagged", "butt", "poly")}
        c, box = yolo_to_xyxy(lab["tag"][0])
        self.assertEqual(c, 1)  # Material glass -> GLASS
        for got, want in zip(box, (0.1, 0.1, 0.3, 0.5)):
            self.assertAlmostEqual(got, want, places=5)
        self.assertEqual(yolo_to_xyxy(lab["untagged"][0])[0], 6)  # no Material tag -> untagged: OTHER
        self.assertEqual(yolo_to_xyxy(lab["butt"][0])[0], 6)  # Object override wins over Material paper
        c, box = yolo_to_xyxy(lab["poly"][0])
        self.assertEqual(c, 2)
        for got, want in zip(box, (0.25, 0.2, 0.75, 0.9)):  # polygon -> its bounding box
            self.assertAlmostEqual(got, want, places=5)
        s = report["sources"]["hitl"]
        self.assertEqual(s["images_kept"], 12)
        self.assertEqual(s["warnings"].get("objects_without_Material_tag"), 1)
        self.assertTrue(s["location"].endswith("humansintheloop/recycling-dataset"))  # outermost match

    def test_unmapped_tag_value_and_override_typo_are_errors(self):
        self.make_project()
        cfg = write_config(self.tmp / "s.yaml", [sly_source(class_map={"glass": "GLASS", "paper": "BIO"})],
                           classes=SEVEN)
        with self.assertRaisesRegex(bd.BuildError, "unmapped classes.*plastic"):
            quiet_build(cfg, self.tmp / "out", search_root=self.search)
        cfg = write_config(self.tmp / "s.yaml", [sly_source(tag_overrides={"Object": {"cigarette": "OTHER"}})],
                           classes=SEVEN)
        with self.assertRaisesRegex(bd.BuildError, "cigarette"):
            quiet_build(cfg, self.tmp / "out", search_root=self.search)

    def test_untagged_is_required_with_class_from_tag(self):
        src = sly_source()
        del src["untagged"]
        with self.assertRaisesRegex(bd.BuildError, "untagged"):
            bd.load_config(write_config(self.tmp / "s.yaml", [src], classes=SEVEN))


class LabelmeTests(BuildTestCase):
    def make_dwsd(self, n=6):
        folder = self.search / "dwsd-india" / "DSWD" / "images"
        make_labelme(folder, "a.jpg", noise_image(200, 100, 1),
                     [lm_poly("Plastic Bottles", [[20, 10], [60, 15], [40, 50]]),
                      lm_poly("glass", [[100, 20], [180, 20], [180, 80], [100, 80]]),
                      {"label": "cloth", "points": [[5, 60], [25, 90]], "shape_type": "rectangle"}])
        for i in range(n):
            make_labelme(folder, f"b{i}.jpg", noise_image(200, 100, 10 + i),
                         [lm_poly("glass 2", [[10, 10], [50, 10], [30, 40]])])
        make_labelme(folder, "embedded.jpg", noise_image(200, 100, 99), [lm_poly("glass", [[0, 0], [100, 50]])],
                     embed=True)
        (folder.parent / "classes.json").write_text(json.dumps({"names": ["glass"]}))  # not a Labelme file
        return folder

    def test_polygons_become_boxes_and_force_split_test(self):
        self.make_dwsd()
        self.standard_yolo()
        cfg = write_config(self.tmp / "s.yaml", [yolo_source(), labelme_source()], classes=SEVEN)
        out = self.tmp / "out"
        report = quiet_build(cfg, out, search_root=self.search)
        lines = sorted((out / "labels" / "dwsd__DSWD_images_a.txt").read_text().splitlines())
        boxes = sorted(yolo_to_xyxy(line) for line in lines)
        self.assertEqual([c for c, _ in boxes], [1, 2, 6])  # GLASS, PLASTIC, OTHER: each shape one instance
        want = {2: (0.1, 0.1, 0.3, 0.5), 1: (0.5, 0.2, 0.9, 0.8), 6: (0.025, 0.6, 0.125, 0.9)}
        for c, box in boxes:
            for got, exp in zip(box, want[c]):
                self.assertAlmostEqual(got, exp, places=5)
        rows = read_csv(out / "split.csv")
        dwsd = [r for r in rows if r["source"] == "dwsd"]
        self.assertEqual(len(dwsd), 8)  # 'glass 2' matches glass; the embedded imageData image is decoded
        self.assertEqual({r["split"] for r in dwsd}, {"test"})
        self.assertEqual({r["split"] for r in rows if r["source"] == "studio_src"}, {"train", "valid", "test"})
        india = sorted(Path(p).name for p in (out / "lists" / "test_india.txt").read_text().splitlines())
        self.assertEqual(india, sorted(r["file"] for r in dwsd))
        self.assertTrue((out / "data_test_india.yaml").is_file())
        self.assertEqual(report["sources"]["dwsd"]["force_split"], "test")
        self.assertEqual(report["sources"]["dwsd"]["warnings"].get("non_labelme_json_skipped"), 1)

    def test_unknown_label_is_an_error_naming_it(self):
        folder = self.make_dwsd()
        make_labelme(folder, "odd.jpg", noise_image(200, 100, 50), [lm_poly("styrofoam", [[1, 1], [9, 9]])])
        cfg = write_config(self.tmp / "s.yaml", [labelme_source()], classes=SEVEN)
        with self.assertRaisesRegex(bd.BuildError, "styrofoam"):
            quiet_build(cfg, self.tmp / "out", search_root=self.search)

    def test_unused_spelling_variants_allowed_only_when_asked(self):
        self.make_dwsd()
        cmap = {"plastic bottles": "PLASTIC", "glass": "GLASS", "cloth": "OTHER", "aluminum foil": "A"}
        cfg = write_config(self.tmp / "s.yaml", [labelme_source(class_map=cmap)], classes=SEVEN)
        with self.assertRaisesRegex(bd.BuildError, "aluminum foil"):
            quiet_build(cfg, self.tmp / "out", search_root=self.search)
        cfg = write_config(self.tmp / "s.yaml", [labelme_source(class_map=cmap, allow_unused_map_keys=True)],
                           classes=SEVEN)
        report = quiet_build(cfg, self.tmp / "out", search_root=self.search)
        self.assertEqual(report["sources"]["dwsd"]["images_kept"], 8)


def make_mask_pair(root, split, n, img, mask):
    """root/<split>/Image/img_<n>.png and root/<split>/Mask/mask_<n>.png (mask: 2-D uint8 array of class values)."""
    for sub in ("Image", "Mask"):
        (root / split / sub).mkdir(parents=True, exist_ok=True)
    img.save(root / split / "Image" / f"img_{n}.png")
    Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L").save(root / split / "Mask" / f"mask_{n}.png")


def mask_source(**kw):
    return {"name": "dwsd", "format": "semantic_mask", "locate": ["DSWD", "DWSD"], "domain": "india",
            "licence": "CC BY 4.0", "force_split": "test", "image_dirs": ["Train/Image", "Test/Image"],
            "mask_dirs": ["Train/Mask", "Test/Mask"], "image_prefix": "img_", "mask_prefix": "mask_",
            "ignore_values": [0, 9], "min_box_area": 50, "connectivity": 8,
            "mask_values": {1: "plastic bottles", 2: "cloth"},
            "class_map": {"plastic bottles": "PLASTIC", "cloth": "OTHER"}, **kw}


class SemanticMaskTests(BuildTestCase):
    def make_dswd(self):
        root = self.search / "dwsd-india-waste" / "DSWD"
        m = np.zeros((100, 200), np.uint8)
        m[10:30, 20:60] = 1      # bottle, 800 px
        m[50:90, 120:180] = 1    # a second, separate bottle region -> its own box
        m[60:66, 10:16] = 1      # 36 px: under min_box_area, dropped
        m[5:15, 150:190] = 9     # an ignore value: no box
        m[70:95, 30:70] = 2      # cloth
        make_mask_pair(root, "Train", 0, noise_image(200, 100, 1), m)
        for i in range(1, 4):
            mi = np.zeros((100, 200), np.uint8)
            mi[20:60, 40:100] = 2
            make_mask_pair(root, "Train" if i < 3 else "Test", i, noise_image(200, 100, 10 + i), mi)
        return root

    def test_regions_become_boxes_tiny_and_ignored_dropped(self):
        self.make_dswd()
        cfg = write_config(self.tmp / "s.yaml", [mask_source()], classes=SEVEN)
        out = self.tmp / "out"
        report = quiet_build(cfg, out, search_root=self.search)
        boxes = sorted(yolo_to_xyxy(line) for line in
                       (out / "labels" / "dwsd__Train_Image_img_0.txt").read_text().splitlines())
        want = [(2, (0.1, 0.1, 0.3, 0.3)), (2, (0.6, 0.5, 0.9, 0.9)), (6, (0.15, 0.7, 0.35, 0.95))]
        self.assertEqual([c for c, _ in boxes], [c for c, _ in want])  # two bottle regions, one cloth, no 9
        for (_, got), (_, exp) in zip(boxes, want):
            for g, e in zip(got, exp):
                self.assertAlmostEqual(g, e, places=5)
        s = report["sources"]["dwsd"]
        self.assertEqual(s["annotations_dropped"], {"tiny_region:plastic bottles": 1})
        self.assertEqual(s["images_kept"], 4)
        self.assertEqual(s["format"], "semantic_mask")
        self.assertTrue(s["location"].endswith("dwsd-india-waste/DSWD"))
        rows = read_csv(out / "split.csv")
        self.assertEqual({r["split"] for r in rows}, {"test"})
        self.assertTrue((out / "data_test_india.yaml").is_file())

    def test_connectivity_four_versus_eight(self):
        m = np.zeros((20, 20), np.uint8)
        m[2:8, 2:8] = 1
        m[8:14, 8:14] = 1  # touches the first square only at a corner
        eight, _, _ = bd.mask_regions(m, {1: "a"}, {0}, 1, 8)
        four, _, _ = bd.mask_regions(m, {1: "a"}, {0}, 1, 4)
        self.assertEqual(eight, [(1, [2.0, 2.0, 12.0, 12.0])])
        self.assertEqual(sorted(four), [(1, [2.0, 2.0, 6.0, 6.0]), (1, [8.0, 8.0, 6.0, 6.0])])

    def test_value_missing_from_mask_values_is_an_error(self):
        root = self.make_dswd()
        m = np.zeros((100, 200), np.uint8)
        m[10:40, 10:40] = 7
        make_mask_pair(root, "Test", 9, noise_image(200, 100, 50), m)
        cfg = write_config(self.tmp / "s.yaml", [mask_source()], classes=SEVEN)
        with self.assertRaisesRegex(bd.BuildError, r"mask values \[7\] are in neither.*mask_9\.png"):
            quiet_build(cfg, self.tmp / "out", search_root=self.search)

    def test_image_mask_stem_mismatch_is_an_error(self):
        root = self.make_dswd()
        noise_image(200, 100, 60).save(root / "Test" / "Image" / "img_77.png")
        Image.fromarray(np.zeros((100, 200), np.uint8)).save(root / "Test" / "Mask" / "mask_78.png")
        cfg = write_config(self.tmp / "s.yaml", [mask_source()], classes=SEVEN)
        with self.assertRaisesRegex(bd.BuildError, r"do not pair up by file stem.*1 images without a mask "
                                                   r"\['img_77\.png'\].*1 masks without an image \['mask_78\.png'\]"):
            quiet_build(cfg, self.tmp / "out", search_root=self.search)

    def test_mask_config_errors(self):
        for bad, msg in (({"mask_dirs": ["Train/Mask"]}, "same length"),
                         ({"mask_values": {1: "a", 300: "b"}}, "integer 0-255"),
                         ({"ignore_values": [0, 1]}, r"both 'mask_values' and 'ignore_values'"),
                         ({"connectivity": 6}, "4 or 8"),
                         ({"min_box_area": 0}, "min_box_area")):
            with self.assertRaisesRegex(bd.BuildError, msg):
                bd.load_config(write_config(self.tmp / "s.yaml", [mask_source(**bad)], classes=SEVEN))
        with self.assertRaisesRegex(bd.BuildError, "cannot be used with format labelme"):
            bd.load_config(write_config(self.tmp / "s.yaml", [labelme_source(mask_values={1: "a"})], classes=SEVEN))


class SourceOptionTests(BuildTestCase):
    def make_pinned_yolo(self):
        """90 train (60 bio, 30 glass), 30 valid (20 plastic, 10 bio), 12 test images, pinned by a split csv."""
        root = self.search / "YOLO_DS"
        imgs, pinned = {}, {}
        plan = [("train", 0)] * 60 + [("train", 1)] * 30 + [("valid", 2)] * 20 + [("valid", 0)] * 10 \
            + [("test", i % 3) for i in range(12)]
        for i, (split, cls) in enumerate(plan):
            imgs[f"{split}/images/img{i:03d}.jpg"] = (noise_image(64, 48, i), f"{cls} 0.5 0.5 0.4 0.3\n")
            pinned[f"img{i:03d}.jpg"] = split
        make_yolo(root, ["bio", "glass", "plastic"], imgs)
        csv_path = self.tmp / "split.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["split", "file"])
            w.writerows((s, n) for n, s in pinned.items())
        return csv_path, pinned

    def test_subsample_counts_stratified_and_test_untouched(self):
        csv_path, pinned = self.make_pinned_yolo()
        src = yolo_source(keep_split_csv=str(csv_path), subsample={"train": 30, "valid": 9, "seed": 5})
        out = self.tmp / "out"
        report = quiet_build(cfg := write_config(self.tmp / "s.yaml", [src]), out, search_root=self.search)
        rows = read_csv(out / "split.csv")
        by = Counter((r["split"], r["dominant_class"]) for r in rows)
        self.assertEqual(by[("train", "BIO")], 20)  # 60:30 -> 20:10
        self.assertEqual(by[("train", "GLASS")], 10)
        self.assertEqual(by[("valid", "PLASTIC")], 6)  # 20:10 -> 6:3
        self.assertEqual(by[("valid", "BIO")], 3)
        test = sorted(r["file"] for r in rows if r["split"] == "test")
        self.assertEqual(test, sorted(f"studio_src__{Path(n).stem}.jpg" for n, s in pinned.items() if s == "test"))
        for r in rows:  # nothing moves between splits
            self.assertEqual(r["split"], pinned[r["file"].split("__")[1]])
        sub = report["sources"]["studio_src"]["subsample"]
        self.assertEqual((sub["train"]["before"], sub["train"]["after"]), (90, 30))
        self.assertEqual(sub["valid"]["dominant_class_after"], {"BIO": 3, "PLASTIC": 6})
        self.assertNotIn("test", sub)
        # the same seed gives the same sample, another seed another one
        again = quiet_build(cfg, self.tmp / "out2", search_root=self.search)
        self.assertEqual((out / "split.csv").read_bytes(), (self.tmp / "out2" / "split.csv").read_bytes())
        self.assertEqual(again["totals"]["train"]["images"], 30)
        src["subsample"]["seed"] = 6
        quiet_build(write_config(self.tmp / "s.yaml", [src]), self.tmp / "out3", search_root=self.search)
        self.assertNotEqual(sorted(r["file"] for r in read_csv(self.tmp / "out3" / "split.csv")),
                            sorted(r["file"] for r in rows))

    def test_subsample_config_errors(self):
        for sub, msg in (({"test": 5}, "never subsampled"), ({"train": 0}, "positive"),
                         ({"train": 5, "extra": 1}, "extra")):
            src = yolo_source(keep_split_csv="x.csv", subsample=sub)
            with self.assertRaisesRegex(bd.BuildError, msg):
                bd.load_config(write_config(self.tmp / "s.yaml", [src]))
        with self.assertRaisesRegex(bd.BuildError, "keep_split_csv"):
            bd.load_config(write_config(self.tmp / "s.yaml", [yolo_source(subsample={"train": 5})]))
        with self.assertRaisesRegex(bd.BuildError, "force_split"):
            bd.load_config(write_config(self.tmp / "s.yaml", [yolo_source(force_split="holdout")]))

    def test_drop_and_mask_paints_grey(self):
        folder = self.search / "COCO_DS"
        images = [(f"{i}.jpg", noise_image(200, 100, i), (200, 100),
                   [(1, [20, 10, 40, 30], 0), (2, [120, 40, 60, 50], 0)]) for i in range(10)]
        make_coco(folder, {1: ("Bottle", "Bottle"), 2: ("Bag", "Bag")}, images)
        cfg = write_config(self.tmp / "s.yaml", [coco_source(class_map={"Bottle": "GLASS", "Bag": "drop_and_mask"})])
        out = self.tmp / "out"
        report = quiet_build(cfg, out, search_root=self.search)
        a = np.asarray(Image.open(out / "images" / "real_src__0.jpg").convert("RGB")).astype(int)
        masked = a[50:80, 130:170]  # inside the Bag box (120..180 x 40..90), away from JPEG ringing at its edge
        self.assertLess(np.abs(masked.mean() - 114), 2)
        self.assertLess(masked.std(), 3)
        self.assertGreater(a[12:38, 22:58].std(), 15)  # the kept box is untouched noise
        lines = (out / "labels" / "real_src__0.txt").read_text().splitlines()
        self.assertEqual([int(line.split()[0]) for line in lines], [1])
        s = report["sources"]["real_src"]
        self.assertEqual(s["annotations_dropped"], {"drop_and_mask:Bag": 10})
        self.assertEqual(s["warnings"]["masked_regions"], 10)

    def test_drop_and_mask_yolo(self):
        root = self.search / "YOLO_DS"
        imgs = {f"train/images/m{i}.jpg": (noise_image(100, 100, i), "0 0.25 0.25 0.3 0.3\n1 0.7 0.7 0.4 0.4\n")
                for i in range(10)}
        make_yolo(root, ["bio", "glass", "plastic"], imgs)
        src = yolo_source(class_map={"bio": "BIO", "glass": "drop_and_mask", "plastic": "PLASTIC"})
        out = self.tmp / "out"
        quiet_build(write_config(self.tmp / "s.yaml", [src]), out, search_root=self.search)
        a = np.asarray(Image.open(out / "images" / "studio_src__m0.jpg").convert("RGB")).astype(int)
        masked = a[60:80, 60:80]  # inside the glass box (50..90), away from its edge
        self.assertLess(np.abs(masked.mean() - 114), 2)
        self.assertLess(masked.std(), 3)
        self.assertEqual(len((out / "labels" / "studio_src__m0.txt").read_text().splitlines()), 1)

    def test_optional_missing_source_warns_and_required_one_errors(self):
        self.standard_yolo()
        missing = labelme_source(locate=["nowhere"], optional=True, kaggle="me/dwsd-india")
        cfg = write_config(self.tmp / "s.yaml", [yolo_source(), missing], classes=SEVEN)
        logs = []
        report = bd.build(cfg, self.tmp / "out", search_root=self.search, log=logs.append)
        self.assertIn("dwsd", report["skipped_sources"])
        self.assertTrue(any("optional source 'dwsd'" in w for w in report["warnings"]))
        self.assertTrue(any("WARNING" in line and "dwsd" in line for line in logs))
        self.assertEqual(report["sources"]["studio_src"]["images_kept"], 20)
        summary = []
        bd.print_summary(report, log=summary.append)
        self.assertTrue(any("NOT found" in line for line in summary))
        missing["optional"] = False
        cfg = write_config(self.tmp / "s.yaml", [yolo_source(), missing], classes=SEVEN)
        with self.assertRaisesRegex(bd.SourceMissing, "me/dwsd-india"):
            quiet_build(cfg, self.tmp / "out2", search_root=self.search)
        taco = coco_source(annotations="annotations.json", locate=["tacotrashdataset"],
                           kaggle="kneroma/tacotrashdataset")
        with self.assertRaisesRegex(bd.BuildError, "Attach the Kaggle dataset 'kneroma/tacotrashdataset'"):
            quiet_build(write_config(self.tmp / "s.yaml", [taco]), self.tmp / "out3", search_root=self.search)


class OnlyTests(BuildTestCase):
    def test_only_builds_just_the_named_sources(self):
        self.standard_yolo()
        LabelmeTests.make_dwsd(self)
        dwsd = labelme_source(optional=True)
        cfg = write_config(self.tmp / "s.yaml", [yolo_source(), dwsd], classes=SEVEN)
        out = self.tmp / "out"
        logs = []
        report = bd.build(cfg, out, search_root=self.search, only=["dwsd"], log=logs.append)
        self.assertEqual(list(report["sources"]), ["dwsd"])
        rows = read_csv(out / "split.csv")
        self.assertEqual({r["source"] for r in rows}, {"dwsd"})
        self.assertEqual({r["split"] for r in rows}, {"test"})
        india = yaml.safe_load((out / "data_test_india.yaml").read_text())
        self.assertEqual(india["names"], list(SEVEN))  # the class list of the config, same order
        self.assertEqual(len((out / "lists" / "test_india.txt").read_text().splitlines()), 8)
        self.assertFalse((out / "data_test_studio.yaml").exists())
        self.assertEqual(report["settings"]["only"], ["dwsd"])
        self.assertTrue(any("--only" in line for line in logs))
        with contextlib.redirect_stdout(io.StringIO()):
            cli = bd.main(["--sources", str(cfg), "--out", str(self.tmp / "out_cli"), "--search-root",
                           str(self.search), "--only", "studio_src"])
        self.assertEqual(list(cli["sources"]), ["studio_src"])

    def test_only_unknown_or_disabled_name_is_an_error(self):
        self.standard_yolo()
        cfg = write_config(self.tmp / "s.yaml", [yolo_source(), coco_source(enabled=False)])
        with self.assertRaisesRegex(bd.BuildError, r"unknown source\(s\) \['dswd'\].*studio_src"):
            quiet_build(cfg, self.tmp / "out", search_root=self.search, only=["dswd"])
        with self.assertRaisesRegex(bd.BuildError, "disabled"):
            quiet_build(cfg, self.tmp / "out", search_root=self.search, only=["real_src"])
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
            bd.main(["--sources", str(cfg), "--out", str(self.tmp / "out"), "--search-root", str(self.search),
                     "--only", "nope"])
        self.assertIn("unknown source", str(cm.exception.code))

    def test_only_makes_an_optional_missing_source_required(self):
        self.standard_yolo()
        missing = labelme_source(locate=["nowhere"], optional=True, kaggle="me/dwsd-india")
        cfg = write_config(self.tmp / "s.yaml", [yolo_source(), missing], classes=SEVEN)
        with self.assertRaisesRegex(bd.SourceMissing, "asked for with --only.*me/dwsd-india"):
            quiet_build(cfg, self.tmp / "out", search_root=self.search, only=["dwsd"])
        report = quiet_build(cfg, self.tmp / "out2", search_root=self.search)  # without --only: skipped
        self.assertIn("dwsd", report["skipped_sources"])


class RelocateTests(BuildTestCase):
    def snapshot(self, folder):
        return {p.relative_to(folder).as_posix(): (p.stat().st_mtime_ns, p.read_bytes())
                for p in sorted(folder.rglob("*")) if p.is_file()}

    def test_relocate_rewrites_paths_and_proxy_works(self):
        self.standard_yolo(n=30)
        self.standard_coco(n=30)
        cfg = write_config(self.tmp / "s.yaml", [yolo_source(), coco_source()])
        quiet_build(cfg, self.tmp / "built", seed=1, search_root=self.search)
        moved = self.tmp / "kaggle_input" / "nb" / "gen2_data"  # as a later notebook sees it
        moved.parent.mkdir(parents=True)
        shutil.move(str(self.tmp / "built"), str(moved))
        before = self.snapshot(moved)
        cfg_dir = self.tmp / "working" / "gen2_cfg"
        summary = bd.relocate(moved, cfg_dir, log=lambda *a: None)
        self.assertEqual(self.snapshot(moved), before)  # the build folder is only read
        root = moved.resolve().as_posix()
        data = yaml.safe_load((cfg_dir / "data.yaml").read_text())
        self.assertEqual(data["path"], root)
        self.assertEqual(data["nc"], 3)
        for key in ("train", "val", "test"):
            self.assertEqual(Path(data[key]).parent, (cfg_dir / "lists").resolve())
        for lst in (cfg_dir / "lists").glob("*.txt"):
            for line in lst.read_text().splitlines():
                self.assertTrue(line.startswith(root + "/images/"), line)
                self.assertTrue(Path(line).is_file())
        self.assertIn("data_test_real_world.yaml", summary["yamls"])
        test_rw = yaml.safe_load((cfg_dir / "data_test_real_world.yaml").read_text())
        self.assertEqual(Path(test_rw["test"]), (cfg_dir / "lists" / "test_real_world.txt").resolve())
        rows = read_csv(cfg_dir / "split.csv")
        self.assertEqual(len(rows), 60)
        self.assertTrue(all(r["path"] == f"{root}/images/{r['file']}" for r in rows))
        stats, n_full, n_sub = training.build_proxy_subset(cfg_dir / "data.yaml", cfg_dir / "data_proxy.yaml",
                                                           cfg_dir / "lists" / "train_proxy.txt", 0.4, 42,
                                                           cfg_dir / "split.csv")
        self.assertEqual(n_sub, round(0.4 * n_full))
        proxy = yaml.safe_load((cfg_dir / "data_proxy.yaml").read_text())
        for line in Path(proxy["train"]).read_text().splitlines():
            self.assertTrue(line.startswith(root + "/images/"))
        self.assertEqual(self.snapshot(moved), before)

    def test_relocate_refuses_bad_folders(self):
        self.standard_yolo()
        built = self.tmp / "built"
        quiet_build(write_config(self.tmp / "s.yaml", [yolo_source()]), built, search_root=self.search)
        with self.assertRaisesRegex(bd.BuildError, "outside"):
            bd.relocate(built, built / "cfg", log=lambda *a: None)
        with self.assertRaisesRegex(bd.BuildError, "not a finished build"):
            bd.relocate(self.search, self.tmp / "cfg", log=lambda *a: None)

    def test_relocate_cli(self):
        self.standard_yolo()
        built = self.tmp / "built"
        quiet_build(write_config(self.tmp / "s.yaml", [yolo_source()]), built, search_root=self.search)
        with contextlib.redirect_stdout(io.StringIO()):
            summary = bd.main(["--relocate-from", str(built), "--out", str(self.tmp / "cfg")])
        self.assertEqual(summary["split_rows"], 20)
        self.assertTrue((self.tmp / "cfg" / "data_test_studio.yaml").is_file())


class ConfigFileTests(unittest.TestCase):
    def test_project_sources_yaml_loads(self):
        classes, sources = bd.load_config(bd.REPO / "configs" / "gen2" / "sources.yaml")
        self.assertEqual(classes, ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC", "OTHER"])
        by = {s["name"]: s for s in sources}
        self.assertEqual(list(by), ["garbage_detection", "taco", "hitl", "dwsd"])
        self.assertEqual(by["garbage_detection"]["keep_split_csv"], "configs/split.csv")
        self.assertEqual(by["garbage_detection"]["subsample"], {"train": 3000, "valid": 900, "seed": 42})
        self.assertEqual({n: s["domain"] for n, s in by.items()},
                         {"garbage_detection": "studio", "taco": "real_world", "hitl": "real_world", "dwsd": "india"})
        self.assertEqual(by["dwsd"]["force_split"], "test")
        self.assertTrue(by["dwsd"]["optional"])
        self.assertEqual(by["dwsd"]["format"], "semantic_mask")
        self.assertEqual(sorted(by["dwsd"]["mask_values"]), list(range(1, 16)))
        self.assertEqual(by["dwsd"]["ignore_values"], [0])
        self.assertEqual(sorted(set(by["dwsd"]["mask_values"].values())), sorted(by["dwsd"]["class_map"]))
        self.assertTrue({"dswd", "dwsd"} <= {bd.norm_name(n) for n in by["dwsd"]["locate"]})
        self.assertFalse(any(by[n]["optional"] for n in ("garbage_detection", "taco", "hitl")))
        self.assertEqual(len(by["taco"]["class_map"]), 60)
        self.assertEqual(sorted(by["hitl"]["class_map"]), ["aluminum", "celluloseacetate", "glass", "metal", "paper",
                                                           "plastic", "polystyrene", "rubber", "tetrapak", "textile"])
        self.assertTrue(all(s["kaggle"] for s in sources))
        for name, lic in (("taco", "CC BY 4.0"), ("hitl", "CC0"), ("dwsd", "CC BY 4.0"),
                          ("garbage_detection", "CC BY 4.0")):
            self.assertTrue(by[name]["licence"].startswith(lic), name)


if __name__ == "__main__":
    unittest.main()
