"""Tests of src/compare_generations.py: the 6-class test subset and the Gen 1 vs Gen 2 summary."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import compare_generations as cg  # noqa: E402

CLASSES7 = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC", "OTHER"]


def make_build(root, labels):
    """A tiny build like gen2_data: images/, labels/, lists/test.txt, data_test_x.yaml. labels: {stem: text|None}."""
    root = Path(root)
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "lists").mkdir()
    lines = []
    for stem, text in labels.items():
        img = root / "images" / f"{stem}.jpg"
        img.write_bytes(b"\xff\xd8 fake jpeg " + stem.encode())
        if text is not None:
            (root / "labels" / f"{stem}.txt").write_text(text)
        lines.append(img.as_posix())
    (root / "lists" / "test.txt").write_text("\n".join(lines) + "\n")
    data = {"path": root.as_posix(), "train": (root / "lists" / "test.txt").as_posix(),
            "val": (root / "lists" / "test.txt").as_posix(), "test": (root / "lists" / "test.txt").as_posix(),
            "nc": 7, "names": CLASSES7}
    (root / "data_test_x.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return root / "data_test_x.yaml"


LABELS = {
    "mixed": "0 0.5 0.5 0.2 0.2\n6 0.3 0.3 0.1 0.1\n5 0.7 0.7 0.1 0.1\n",
    "only_other": "6 0.5 0.5 0.4 0.4\n6 0.2 0.2 0.1 0.1\n",
    "clean": "3 0.5 0.5 0.2 0.2\n",
    "background": "",
    "no_label_file": None,
}


class TestParseClasses(unittest.TestCase):
    def test_forms(self):
        self.assertEqual(cg.parse_classes("0-5"), [0, 1, 2, 3, 4, 5])
        self.assertEqual(cg.parse_classes("0,1,2"), [0, 1, 2])
        self.assertEqual(cg.parse_classes("2 1 0"), [0, 1, 2])

    def test_only_a_prefix(self):
        for bad in ("1-5", "0,2", "x", "", "0-a"):
            with self.assertRaises(cg.SubsetError):
                cg.parse_classes(bad)


class TestSubset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name).resolve()  # the long path, as the subset writes it
        self.data = make_build(self.dir / "build", LABELS)
        self.before = {p: p.read_bytes() for p in (self.dir / "build").rglob("*") if p.is_file()}

    def tearDown(self):
        self.tmp.cleanup()

    def run_subset(self, out="subset"):
        with contextlib.redirect_stdout(io.StringIO()):
            return cg.subset_dataset(self.data, "test", [0, 1, 2, 3, 4, 5], self.dir / out)

    def test_other_lines_removed_and_images_kept(self):
        report = self.run_subset()
        out = self.dir / "subset"
        self.assertEqual(report["images"], 5)
        self.assertEqual(report["boxes_kept"], 3)
        self.assertEqual(report["boxes_removed"], 3)
        self.assertEqual(report["boxes_removed_per_class"], {"OTHER": 3})
        self.assertEqual(report["boxes_kept_per_class"], {"BIODEGRADABLE": 1, "METAL": 1, "PLASTIC": 1})
        self.assertEqual(report["images_emptied"], 1)  # only_other: stays in with no objects
        self.assertEqual(report["images_without_label_file"], 1)
        self.assertEqual((out / "labels" / "mixed.txt").read_text(), "0 0.5 0.5 0.2 0.2\n5 0.7 0.7 0.1 0.1\n")
        self.assertEqual((out / "labels" / "only_other.txt").read_text(), "")
        self.assertEqual((out / "labels" / "clean.txt").read_text(), LABELS["clean"])
        self.assertEqual((out / "labels" / "no_label_file.txt").read_text(), "")
        listed = (out / "lists" / "test.txt").read_text().splitlines()
        self.assertEqual(sorted(Path(p).name for p in listed), sorted(f"{s}.jpg" for s in LABELS))
        for p in listed:  # images in <out>/images with the same bytes as the originals
            self.assertEqual(Path(p).parent, out / "images")
            self.assertEqual(Path(p).read_bytes(), (self.dir / "build" / "images" / Path(p).name).read_bytes())
        cfg = yaml.safe_load((out / "data.yaml").read_text())
        self.assertEqual(cfg["nc"], 6)
        self.assertEqual(cfg["names"], CLASSES7[:6])
        self.assertEqual(cfg["test"], cfg["val"])
        self.assertEqual(json.loads((out / cg.REPORT).read_text())["images"], 5)

    def test_ultralytics_reads_the_derived_labels(self):
        """The label file Ultralytics would open for a listed image is the filtered copy."""
        self.run_subset()
        out = self.dir / "subset"
        for p in (out / "lists" / "test.txt").read_text().splitlines():
            self.assertEqual(cg.label_file(p), out / "labels" / f"{Path(p).stem}.txt")

    def test_originals_untouched_and_rerun_replaces(self):
        self.run_subset()
        (self.dir / "subset" / "labels.cache").write_text("stale")
        report = self.run_subset()  # an earlier subset is replaced
        self.assertEqual(report["images"], 5)
        self.assertFalse((self.dir / "subset" / "labels.cache").exists())
        after = {p: p.read_bytes() for p in (self.dir / "build").rglob("*") if p.is_file()}
        self.assertEqual(after, self.before)

    def test_refusals(self):
        (self.dir / "busy").mkdir()
        (self.dir / "busy" / "notes.txt").write_text("keep me")
        with self.assertRaises(cg.SubsetError):
            self.run_subset("busy")
        self.assertTrue((self.dir / "busy" / "notes.txt").exists())
        with self.assertRaises(cg.SubsetError):
            self.run_subset("build")
        with self.assertRaises(cg.SubsetError), contextlib.redirect_stdout(io.StringIO()):
            cg.subset_dataset(self.data, "test", list(range(8)), self.dir / "too_many")
        with self.assertRaises(cg.SubsetError):
            cg.split_images(self.data, "missing_split")

    def test_command_line(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            report = cg.main(["subset", "--data", str(self.data), "--split", "test", "--classes", "0-5",
                              "--out", str(self.dir / "cli")])
        self.assertEqual(report["boxes_removed"], 3)
        self.assertIn("1 images had only removed classes", out.getvalue())
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()):
            cg.main(["subset", "--data", str(self.data), "--classes", "1-5", "--out", str(self.dir / "bad")])


def fake_report(tag, maps, val, sizes):
    """An evaluate.py results JSON for the six arms. maps: {label: map50}; pairs a < b in label order."""
    labels = list(maps)
    models = {label: {"name": f"name {label}", "metrics": {"map50": m, "map50_95": m / 2},
                      "bootstrap": {"map50_ci95": [m - 0.02, m + 0.02]}} for label, m in maps.items()}
    pairwise = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            row = {"a": a, "b": b, "pair": f"{b}-{a}"}
            for key, scale in (("map50", 1), ("map50_95", 0.5)):
                d = (maps[b] - maps[a]) * scale
                row[key] = {"observed_diff": d, "ci95": [d - 0.03, d + 0.03], "significant": abs(d) > 0.03}
            pairwise.append(row)
    return {"tag": tag, "images": 468, "models": models, "pairwise": pairwise,
            "val_metrics": {label: {"map50": v} for label, v in val.items()},
            "settings": {"imgsz": sizes}}


class TestSummary(unittest.TestCase):
    def setUp(self):
        maps = {"G1A": 0.30, "G1B": 0.29, "G1C": 0.31, "G2A": 0.50, "G2B": 0.52, "G2C": 0.305}
        val = {"G1A": 0.657, "G1B": 0.643, "G1C": 0.657, "G2A": 0.454, "G2B": 0.473, "G2C": 0.466}
        sizes = {label: 416 if label.startswith("G1") else 640 for label in maps}
        self.report = fake_report("g1g2_real_world", maps, val, sizes)

    def test_pair_diff_both_orientations(self):
        d, ci, sig = cg.pair_diff(self.report, "G1A", "G2A", "map50")
        self.assertAlmostEqual(d, 0.20)
        self.assertTrue(sig)
        d2, ci2, _ = cg.pair_diff(self.report, "G2A", "G1A", "map50")
        self.assertAlmostEqual(d2, -0.20)
        self.assertAlmostEqual(ci2[0], -ci[1])
        self.assertAlmostEqual(ci2[1], -ci[0])
        with self.assertRaises(KeyError):
            cg.pair_diff(self.report, "G1A", "X", "map50")

    def test_best_by_val_not_by_test(self):
        self.assertEqual(cg.best_by_val(self.report, "G1"), "G1C")  # tie 0.657: the later label, deterministic
        self.assertEqual(cg.best_by_val(self.report, "G2"), "G2B")

    def test_markdown(self):
        md = cg.summary_markdown({"g1g2_real_world": self.report})
        self.assertIn("| G1A: name G1A | 416 |", md)
        self.assertIn("| G2B: name G2B | 640 | 0.520 [0.500, 0.540] / 0.260 |", md)
        self.assertIn("G2A − G1A (name G2A − name G1A): Δ mAP@50 +0.200", md)
        self.assertIn("name G2A significantly better", md)
        self.assertIn("G2C − G1C (name G2C − name G1C): Δ mAP@50 -0.005 [-0.035, +0.025], not significant", md)
        self.assertIn("Best vs best (chosen by each arm's own val mAP@50, not by these test numbers): G2B − G1C", md)

    def test_command_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_g1g2_real_world_results.json"
            path.write_text(json.dumps(self.report))
            out = Path(tmp) / "summary.md"
            with contextlib.redirect_stdout(io.StringIO()):
                cg.main(["summary", "--results", str(path), "--out", str(out)])
            self.assertIn("**g1g2_real_world**", out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
