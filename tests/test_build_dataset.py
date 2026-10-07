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


class ConfigFileTests(unittest.TestCase):
    def test_project_sources_yaml_loads(self):
        classes, sources = bd.load_config(bd.REPO / "configs" / "gen2" / "sources.yaml")
        self.assertEqual(classes, ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"])
        self.assertEqual([s["name"] for s in sources], ["garbage_detection", "plusyaml"])
        self.assertEqual(sources[0]["keep_split_csv"], "configs/split.csv")


if __name__ == "__main__":
    unittest.main()
