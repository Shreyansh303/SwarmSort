"""Tests of the demo app: bin rules, result tables, image preprocessing, and Streamlit smoke runs of app/app.py."""
import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "app"))
import helpers  # noqa: E402

APP_FILE = REPO / "app" / "app.py"
WEIGHTS = [helpers.model_path(label) for label in helpers.MODELS]
GREEN, BLUE = helpers.BINS["green"], helpers.BINS["blue"]


def fake(cls, conf=0.9):
    return {"class": cls, "conf": conf, "box": (10.0, 10.0, 50.0, 40.0)}


def image_bytes(img, fmt, **kwargs):
    buf = io.BytesIO()
    img.save(buf, fmt, **kwargs)
    return buf.getvalue()


class TestBinRules(unittest.TestCase):
    def test_rules_cover_exactly_the_dataset_classes(self):
        with open(REPO / "configs" / "split.csv", newline="") as f:
            classes = {row["source_class"] for row in csv.DictReader(f)}
        self.assertEqual(len(classes), 6)
        self.assertEqual(set(helpers.BIN_RULES), classes)
        self.assertEqual(set(helpers.CLASS_NAMES), classes)

    def test_only_biodegradable_goes_to_the_green_bin(self):
        green = [name for name, rule in helpers.BIN_RULES.items() if rule["bin"] == "green"]
        self.assertEqual(green, ["BIODEGRADABLE"])
        for rule in helpers.BIN_RULES.values():
            self.assertIn(rule["bin"], helpers.BINS)
            self.assertTrue(rule["tip"])


def rgb(color):
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


def contrast(color, text):
    """WCAG contrast ratio of a '#rrggbb' background and 'black' or 'white' text."""
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (v / 255 for v in rgb(color))]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    text_lum = 1.0 if text == "white" else 0.0
    return (max(lum, text_lum) + 0.05) / (min(lum, text_lum) + 0.05)


class TestBoxColours(unittest.TestCase):
    def test_every_class_has_its_own_colour(self):
        self.assertEqual(set(helpers.CLASS_COLORS), set(helpers.CLASS_NAMES))
        self.assertEqual(len(set(helpers.CLASS_COLORS.values())), len(helpers.CLASS_NAMES))

    def test_biodegradable_box_is_green(self):
        r, g, b = rgb(helpers.class_color("BIODEGRADABLE"))
        self.assertGreater(g, max(r, b))

    def test_dry_class_colours_are_neither_green_nor_blue(self):
        for name, rule in helpers.BIN_RULES.items():
            if rule["bin"] == "blue":
                r, g, b = rgb(helpers.class_color(name))
                self.assertFalse(g > max(r, b), f"{name} box colour is green-dominant")
                self.assertFalse(b > max(r, g), f"{name} box colour is blue-dominant")

    def test_label_text_is_readable_on_every_colour(self):
        for name, color in helpers.CLASS_COLORS.items():
            self.assertGreaterEqual(contrast(color, helpers.label_text_color(color)), 4.5, name)
        self.assertEqual(helpers.label_text_color(helpers.CLASS_COLORS["PAPER"]), "black")  # yellow


class TestSummaries(unittest.TestCase):
    def test_bin_summary_counts(self):
        dets = [fake("PLASTIC"), fake("BIODEGRADABLE"), fake("GLASS"), fake("METAL"), fake("PAPER"),
                fake("CARDBOARD"), fake("BIODEGRADABLE")]
        self.assertEqual(helpers.bin_summary(dets), {GREEN: 2, BLUE: 5})
        self.assertEqual(list(helpers.bin_summary(dets)), [GREEN, BLUE])

    def test_bin_summary_skips_empty_bins(self):
        self.assertEqual(helpers.bin_summary([fake("METAL")]), {BLUE: 1})
        self.assertEqual(helpers.bin_summary([]), {})

    def test_detection_rows(self):
        rows = helpers.detection_rows([fake("GLASS", 0.874), fake("BIODEGRADABLE", 0.5)])
        self.assertEqual([r["#"] for r in rows], [1, 2])
        self.assertEqual(rows[0]["Item"], "GLASS")
        self.assertEqual(rows[0]["Confidence"], 0.87)
        self.assertEqual(rows[0]["Bin"], BLUE)
        self.assertIn("handle with care", rows[0]["Disposal tip"])
        self.assertEqual(rows[1]["Bin"], GREEN)
        self.assertEqual(rows[1]["Disposal tip"], "Compost")


class TestResultsReaders(unittest.TestCase):
    RESULTS = {"models": {label: {"metrics": {"map50": m, "map50_95": m - 0.2},
                                  "bootstrap": {"map50_ci95": [m - 0.04, m + 0.04],
                                                "map50_95_ci95": [m - 0.24, m - 0.16]}}
                          for label, m in (("A", 0.60), ("B", 0.61), ("C", 0.65))},
               "pairwise": [{"map50": {"significant": False}}, {"map50": {"significant": False}}]}

    def test_default_model_is_the_best_test_map50(self):
        self.assertEqual(helpers.default_model(self.RESULTS), "C")
        self.assertEqual(helpers.default_model(None), "A")

    def test_model_label(self):
        self.assertEqual(helpers.model_label("C", self.RESULTS), "C: PSO (test mAP@50 0.650)")
        self.assertEqual(helpers.model_label("B", None), "B: Random search")

    def test_models_tied(self):
        self.assertTrue(helpers.models_tied(self.RESULTS))
        self.assertFalse(helpers.models_tied({"pairwise": [{"map50": {"significant": True}}]}))
        self.assertIsNone(helpers.models_tied(None))

    def test_comparison_rows(self):
        rows = helpers.comparison_rows(self.RESULTS)
        self.assertEqual([r["Model"] for r in rows], ["A: Defaults", "B: Random search", "C: PSO"])
        self.assertEqual(rows[2]["Test mAP@50 [95% CI]"], "0.650 [0.610, 0.690]")
        self.assertEqual(helpers.comparison_rows(None), [])

    def test_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(helpers.load_results(Path(tmp) / "missing.json"))
            self.assertEqual(helpers.test_images(Path(tmp) / "test.txt", Path(tmp)), [])


class TestPrepareImage(unittest.TestCase):
    def test_exif_rotated_jpeg_comes_out_upright(self):
        stored = Image.new("RGB", (40, 20), (255, 0, 0))
        stored.paste((0, 0, 255), (20, 0, 40, 20))  # stored: left half red, right half blue
        exif = Image.Exif()
        exif[0x0112] = 6  # Orientation 6: the viewer must rotate the stored image 90 degrees clockwise
        img = helpers.prepare_image(image_bytes(stored, "JPEG", exif=exif, quality=95))
        self.assertEqual(img.size, (20, 40))  # portrait once upright
        top, bottom = img.getpixel((10, 5)), img.getpixel((10, 35))
        self.assertGreater(top[0], 200)
        self.assertLess(top[2], 60)   # the stored left (red) half is now on top
        self.assertGreater(bottom[2], 200)
        self.assertLess(bottom[0], 60)

    def test_rgba_becomes_rgb_on_white(self):
        rgba = Image.new("RGBA", (30, 30), (0, 0, 0, 0))
        rgba.paste((0, 128, 0, 255), (0, 0, 15, 30))
        img = helpers.prepare_image(image_bytes(rgba, "PNG"))
        self.assertEqual(img.mode, "RGB")
        self.assertEqual(img.getpixel((25, 15)), (255, 255, 255))  # transparent part -> white
        self.assertEqual(img.getpixel((5, 15)), (0, 128, 0))

    def test_grayscale_and_palette_become_rgb(self):
        for mode in ("L", "P"):
            img = helpers.prepare_image(image_bytes(Image.new(mode, (16, 12), 100), "PNG"))
            self.assertEqual((img.mode, img.size), ("RGB", (16, 12)))

    def test_large_image_is_shrunk_keeping_aspect_ratio(self):
        img = helpers.prepare_image(image_bytes(Image.new("RGB", (4000, 3000), (90, 90, 90)), "JPEG"))
        self.assertEqual(img.size, (helpers.MAX_SIDE, helpers.MAX_SIDE * 3 // 4))

    def test_corrupt_and_truncated_files_raise_a_friendly_error(self):
        from ultralytics.utils import checks  # importing Ultralytics wraps Image.open, as in the running app

        self.assertFalse(checks.AUTOINSTALL)  # so a failed open cannot start a pip install
        jpeg = image_bytes(Image.new("RGB", (200, 200), (10, 200, 10)), "JPEG")
        for data in (b"not an image at all", jpeg[:len(jpeg) // 2], b""):
            with self.assertRaises(helpers.ImageError) as ctx:
                helpers.prepare_image(data)
            self.assertIn("could not be read as an image", str(ctx.exception))

    def test_decompression_bomb_says_too_large(self):
        import ultralytics  # noqa: F401  (its Image.open wrapper re-raises a different error; the chain is checked)

        data = image_bytes(Image.new("RGB", (100, 100)), "PNG")
        with mock.patch.object(Image, "MAX_IMAGE_PIXELS", 1000):  # 10,000 px is more than twice the limit
            with self.assertRaises(helpers.ImageError) as ctx:
                helpers.prepare_image(data)
        self.assertIn("too large", str(ctx.exception))


class SpyModel:
    """Stands in for a YOLO model: records each predict call and returns one PLASTIC box. No real inference."""

    def __init__(self, path=None):
        self.path, self.calls = path, []

    def predict(self, image, **kwargs):
        self.calls.append((image, kwargs))
        boxes = SimpleNamespace(cls=np.array([5.0]), conf=np.array([0.9]), xyxy=np.array([[10.0, 10.0, 50.0, 40.0]]))
        return [SimpleNamespace(boxes=boxes, names=dict(enumerate(helpers.CLASS_NAMES)))]


class TestDetect(unittest.TestCase):
    def test_conf_and_iou_reach_predict(self):
        model, img = SpyModel(), Image.new("RGB", (64, 64))
        detections, ms = helpers.detect(model, img, conf=0.6, iou=0.3)
        (image, kwargs), = model.calls
        self.assertIs(image, img)
        self.assertEqual((kwargs["conf"], kwargs["iou"], kwargs["imgsz"]), (0.6, 0.3, helpers.IMGSZ))
        self.assertEqual(detections, [{"class": "PLASTIC", "conf": 0.9, "box": (10.0, 10.0, 50.0, 40.0)}])
        self.assertGreaterEqual(ms, 0)


class TestDrawing(unittest.TestCase):
    def test_box_is_drawn_in_the_class_colour(self):
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        out = helpers.draw_detections(img, [{"class": "METAL", "conf": 0.87, "box": (50.0, 80.0, 150.0, 180.0)}])
        metal = Image.new("RGB", (1, 1), helpers.class_color("METAL")).getpixel((0, 0))
        self.assertEqual(out.getpixel((100, 180)), metal)  # bottom edge of the box
        self.assertEqual(img.getpixel((100, 180)), (255, 255, 255))  # the input is left unchanged


@unittest.skipUnless(all(p.is_file() for p in WEIGHTS),
                     "app/models/arm_a.pt, arm_b.pt and arm_c.pt are needed for the Streamlit smoke tests")
class TestStreamlitApp(unittest.TestCase):
    def run_app(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(APP_FILE), default_timeout=120).run(timeout=120)
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        return at

    def test_app_runs_with_model_selector_and_confidence_slider(self):
        at = self.run_app()
        models = [r for r in at.sidebar.radio if r.label == "Model"]
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].value, helpers.default_model(helpers.load_results()))
        conf = [s for s in at.sidebar.slider if s.label == "Confidence threshold"]
        self.assertEqual(len(conf), 1)
        self.assertAlmostEqual(conf[0].value, 0.35)
        self.assertEqual(len(at.error), 0)

    def test_uploads_of_a_corrupt_file_and_an_empty_scene(self):
        at = self.run_app()
        plain = image_bytes(Image.new("RGB", (640, 480), (120, 160, 200)), "PNG")
        at.file_uploader[0].set_value([("broken.jpg", b"not an image", "image/jpeg"),
                                       ("plain.png", plain, "image/png")]).run(timeout=120)
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        self.assertEqual([s.value for s in at.main.subheader], ["broken.jpg", "plain.png"])
        self.assertEqual(len(at.error), 1)
        self.assertIn("broken.jpg", at.error[0].value)
        self.assertEqual(len(at.warning), 1)
        self.assertIn("No waste items detected", at.warning[0].value)

    def test_sidebar_choices_reach_the_model(self):
        """The chosen model file is loaded and the chosen conf and IoU reach predict (spy models, no inference)."""
        import streamlit as st

        st.cache_resource.clear()  # forget real models cached by other tests, so the spy loader is called
        self.addCleanup(st.cache_resource.clear)  # and do not leave spy models in the cache
        loaded = {}

        def spy_loader(path):
            loaded[Path(path).name] = SpyModel(path)
            return loaded[Path(path).name]

        default = helpers.default_model(helpers.load_results())
        other = "B" if default != "B" else "C"
        with mock.patch.object(helpers, "load_model", side_effect=spy_loader):
            at = self.run_app()
            [s for s in at.sidebar.slider if s.label == "Confidence threshold"][0].set_value(0.6)
            [s for s in at.sidebar.slider if s.label == "IoU threshold (NMS)"][0].set_value(0.3)
            photo = image_bytes(Image.new("RGB", (80, 60), (200, 200, 200)), "PNG")
            at.file_uploader[0].set_value(("photo.png", photo, "image/png")).run(timeout=120)
            [r for r in at.sidebar.radio if r.label == "Model"][0].set_value(other).run(timeout=120)
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        first, second = helpers.model_path(default).name, helpers.model_path(other).name
        self.assertEqual(list(loaded), [first, second])
        for name in (first, second):
            _, kwargs = loaded[name].calls[-1]
            self.assertEqual((kwargs["conf"], kwargs["iou"]), (0.6, 0.3), name)
        self.assertIn("PLASTIC", at.dataframe[0].value["Item"].tolist())

    @unittest.skipUnless(helpers.test_images(), "the local dataset and configs/lists/test.txt are not available")
    def test_random_test_image_gives_a_result(self):
        at = self.run_app()
        [r for r in at.radio if r.label == "Input"][0].set_value("Random test image").run(timeout=120)
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        self.assertIn("Inference time", [m.label for m in at.metric])
        self.assertEqual(len(at.error), 0)


if __name__ == "__main__":
    unittest.main()
