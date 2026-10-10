"""Tests of the demo app: generations, bin rules, result tables, image handling, and Streamlit runs of app/app.py."""
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "app"))
import helpers  # noqa: E402

APP_FILE = REPO / "app" / "app.py"
WEIGHTS = {gen: [helpers.model_path(gen, label) for label in helpers.MODELS] for gen in helpers.GENERATIONS}
HAVE_GEN1 = all(p.is_file() for p in WEIGHTS["gen1"])
HAVE_GEN2 = all(p.is_file() for p in WEIGHTS["gen2"])
GREEN, BLUE, REJECT = helpers.BINS["green"], helpers.BINS["blue"], helpers.BINS["reject"]


def fake(cls, conf=0.9):
    return {"class": cls, "conf": conf, "box": (10.0, 10.0, 50.0, 40.0)}


def image_bytes(img, fmt, **kwargs):
    buf = io.BytesIO()
    img.save(buf, fmt, **kwargs)
    return buf.getvalue()


class TestBinRules(unittest.TestCase):
    def test_rules_cover_exactly_the_seven_classes(self):
        with open(REPO / "configs" / "split.csv", newline="") as f:
            gen1 = {row["source_class"] for row in csv.DictReader(f)}
        self.assertEqual(len(gen1), 6)
        gen2 = yaml.safe_load((REPO / "configs" / "gen2" / "sources.yaml").read_text(encoding="utf-8"))["classes"]
        self.assertEqual(gen2, helpers.CLASS_NAMES)  # same names in the same index order
        self.assertEqual(set(gen2), gen1 | {"OTHER"})
        self.assertEqual(set(helpers.BIN_RULES), set(gen2))

    def test_each_bin_gets_the_right_classes(self):
        by_bin = {key: sorted(n for n, r in helpers.BIN_RULES.items() if r["bin"] == key) for key in helpers.BINS}
        self.assertEqual(by_bin, {"green": ["BIODEGRADABLE"],
                                  "blue": ["CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"],
                                  "reject": ["OTHER"]})
        for rule in helpers.BIN_RULES.values():
            self.assertTrue(rule["tip"])

    def test_reject_bin_wording(self):
        self.assertIn("non-recyclable", REJECT)
        self.assertIn("check local rules", REJECT)
        self.assertIn("hazardous", helpers.BIN_RULES["OTHER"]["tip"])


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

    def test_dry_and_other_colours_are_neither_green_nor_blue(self):
        checked = []
        for name, rule in helpers.BIN_RULES.items():
            if rule["bin"] in ("blue", "reject"):
                r, g, b = rgb(helpers.class_color(name))
                self.assertFalse(g > max(r, b), f"{name} box colour is green-dominant")
                self.assertFalse(b > max(r, g), f"{name} box colour is blue-dominant")
                checked.append(name)
        self.assertIn("OTHER", checked)
        self.assertNotEqual(helpers.class_color("OTHER"), helpers.OTHER_COLOR)  # OTHER has its own colour

    def test_label_text_is_readable_on_every_colour(self):
        for name, color in helpers.CLASS_COLORS.items():
            self.assertGreaterEqual(contrast(color, helpers.label_text_color(color)), 4.5, name)
        self.assertEqual(helpers.label_text_color(helpers.CLASS_COLORS["PAPER"]), "black")  # yellow


class TestSummaries(unittest.TestCase):
    def test_bin_summary_counts(self):
        dets = [fake("PLASTIC"), fake("OTHER"), fake("BIODEGRADABLE"), fake("GLASS"), fake("METAL"), fake("PAPER"),
                fake("CARDBOARD"), fake("BIODEGRADABLE")]
        self.assertEqual(helpers.bin_summary(dets), {GREEN: 2, BLUE: 5, REJECT: 1})
        self.assertEqual(list(helpers.bin_summary(dets)), [GREEN, BLUE, REJECT])

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

    def test_other_and_unknown_rows(self):
        rows = helpers.detection_rows([fake("OTHER"), fake("SOMETHING_NEW")])
        self.assertEqual(rows[0]["Bin"], REJECT)
        self.assertEqual((rows[1]["Bin"], rows[1]["Disposal tip"]), ("Unknown", "Check your local rules"))


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


def fake_results(scores, pairs=()):
    """Results JSON in src/evaluate.py's shape: test mAP@50 per arm, and pairs (a, b, b minus a, significant)."""
    return {"images": 100,
            "models": {k: {"metrics": {"map50": m, "map50_95": m / 2},
                           "bootstrap": {"map50_ci95": [m - 0.02, m + 0.02], "map50_95_ci95": [0, 1]}}
                       for k, m in scores.items()},
            "pairwise": [{"a": a, "b": b, "map50": {"observed_diff": d, "significant": sig}} for a, b, d, sig in pairs]}


class TestGenerations(unittest.TestCase):
    def test_generation_settings(self):
        self.assertEqual(helpers.DEFAULT_GENERATION, "gen2")
        self.assertEqual(list(helpers.GENERATIONS), ["gen2", "gen1"])  # the order of the sidebar choices
        self.assertEqual(helpers.GENERATIONS["gen2"]["label"], "Generation 2 (real-world data, 7 classes)")
        self.assertEqual(helpers.GENERATIONS["gen1"]["label"], "Generation 1 (studio data, 6 classes)")
        self.assertEqual((helpers.GENERATIONS["gen1"]["imgsz"], helpers.GENERATIONS["gen2"]["imgsz"]), (416, 640))
        self.assertEqual(helpers.model_path("gen1", "A"), REPO / "app" / "models" / "arm_a.pt")
        self.assertEqual(helpers.model_path("gen2", "C"), REPO / "app" / "models" / "gen2" / "arm_c.pt")

    def test_default_arm_per_generation_from_fake_jsons(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen1, gen2 = Path(tmp) / "gen1.json", Path(tmp) / "gen2.json"
            gen1.write_text(json.dumps(fake_results({"A": 0.68, "B": 0.67, "C": 0.66})), encoding="utf-8")
            gen2.write_text(json.dumps(fake_results({"A": 0.48, "B": 0.497, "C": 0.50})), encoding="utf-8")
            self.assertEqual(helpers.default_model(helpers.load_results(gen1)), "A")
            self.assertEqual(helpers.default_model(helpers.load_results(gen2)), "C")
            self.assertEqual(helpers.default_model(helpers.load_results(Path(tmp) / "missing.json")), "A")

    def test_default_arm_from_the_real_results(self):
        self.assertEqual(helpers.default_model(helpers.load_results(helpers.GENERATIONS["gen1"]["results"])), "A")
        self.assertEqual(helpers.default_model(helpers.load_results(helpers.GENERATIONS["gen2"]["results"])), "C")

    def test_pairwise_note(self):
        tied = fake_results({}, [("A", "B", -0.004, False), ("A", "C", -0.008, False), ("B", "C", -0.004, False)])
        self.assertIn("statistically tied", helpers.pairwise_note(tied))
        gen2 = fake_results({}, [("A", "B", 0.017, True), ("A", "C", 0.020, True), ("B", "C", 0.003, False)])
        self.assertEqual(helpers.pairwise_note(gen2), "B and C significantly beat A; C vs B is a tie.")
        india = fake_results({}, [("A", "B", -0.003, True), ("A", "C", 0.0, False), ("B", "C", 0.003, True)])
        self.assertEqual(helpers.pairwise_note(india), "A and C significantly beat B; C vs A is a tie.")
        self.assertIsNone(helpers.pairwise_note(None))
        self.assertIsNone(helpers.pairwise_note({"pairwise": []}))

    def test_pairwise_note_from_the_real_results(self):
        gen1 = helpers.pairwise_note(helpers.load_results(helpers.GENERATIONS["gen1"]["results"]))
        gen2 = helpers.pairwise_note(helpers.load_results(helpers.GENERATIONS["gen2"]["results"]))
        self.assertIn("statistically tied", gen1)
        self.assertEqual(gen2, "B and C significantly beat A; C vs B is a tie.")

    def test_domain_rows(self):
        domains = {"all": fake_results({"A": 0.479, "B": 0.497, "C": 0.5}), "studio": None,
                   "india": fake_results({"A": 0.027, "C": 0.027})}
        rows = helpers.domain_rows(domains)
        self.assertEqual([r["Model"] for r in rows], ["A: Defaults", "B: Random search", "C: PSO"])
        self.assertEqual(list(rows[0]), ["Model", "all (100 images)", "india (100 images)"])  # studio is missing
        self.assertEqual(rows[2]["all (100 images)"], "0.500 [0.480, 0.520]")
        self.assertEqual(rows[1]["india (100 images)"], "n/a")
        self.assertEqual(helpers.domain_rows({d: None for d in helpers.DOMAINS}), [])

    def test_domain_rows_from_the_real_results(self):
        rows = helpers.domain_rows(helpers.load_domain_results())
        self.assertEqual(len(rows), 3)
        self.assertEqual([c.split(" ")[0] for c in list(rows[0])[1:]], helpers.DOMAINS)

    def test_imgsz_from_fake_train_args(self):
        def model(ckpt):
            return SimpleNamespace(ckpt=ckpt)

        self.assertEqual(helpers.model_imgsz(model({"train_args": {"imgsz": 640}}), 416), 640)
        self.assertEqual(helpers.model_imgsz(model({"train_args": {"imgsz": [480, 640]}}), 416), 640)
        self.assertEqual(helpers.model_imgsz(model({"train_args": {"imgsz": "320"}}), 416), 320)
        for broken in ({"train_args": {}}, {"train_args": {"imgsz": None}}, {"train_args": {"imgsz": "big"}},
                       {"train_args": {"imgsz": 0}}, {"train_args": {"imgsz": []}}, {}, None):
            self.assertEqual(helpers.model_imgsz(model(broken), 416), 416, broken)
        self.assertEqual(helpers.model_imgsz(SimpleNamespace(), 640), 640)  # no checkpoint at all

    def test_model_class_names(self):
        self.assertEqual(helpers.model_class_names(SimpleNamespace(names={1: "B", 0: "A"})), ["A", "B"])
        self.assertEqual(helpers.model_class_names(SimpleNamespace()), [])

    def test_missing_weights_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(helpers.GENERATIONS["gen2"], {"dir": REPO / "app" / "models" / "nowhere"}):
                text = helpers.missing_weights_message("gen2", "C")
                self.assertFalse(helpers.model_path("gen2", "C").is_file())
            with mock.patch.dict(helpers.GENERATIONS["gen1"], {"dir": Path(tmp)}):
                outside = helpers.missing_weights_message("gen1", "A")  # a folder outside the repo
        self.assertIn("Model file not found: app/models/nowhere/arm_c.pt", text)
        self.assertIn("Generation 2", text)
        self.assertIn("app/models/README.md", text)
        self.assertIn("pick Generation 1 in the sidebar", text)
        self.assertIn("arm_a.pt", outside)
        self.assertIn("pick Generation 2 in the sidebar", outside)


class TestWeights(unittest.TestCase):
    """The committed weights are the evaluated models: their sha256[:12] matches the results files."""

    def check(self, gen):
        results = helpers.load_results(helpers.GENERATIONS[gen]["results"])
        for label in helpers.MODELS:
            self.assertEqual(helpers.sha256_12(helpers.model_path(gen, label)),
                             results["models"][label]["sha256_12"], f"{gen} {label}")

    @unittest.skipUnless(HAVE_GEN1, "Generation 1 weights (app/models/arm_*.pt) are missing")
    def test_gen1_weights_match_the_results(self):
        self.check("gen1")

    @unittest.skipUnless(HAVE_GEN2, "Generation 2 weights (app/models/gen2/arm_*.pt) are missing")
    def test_gen2_weights_match_the_results(self):
        self.check("gen2")

    def test_weights_folders_are_not_gitignored(self):
        lines = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("!app/models/*.pt", lines)
        self.assertIn("!app/models/gen2/*.pt", lines)


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
    """Stands in for a YOLO model: records each predict call and returns one box (PLASTIC by default).

    No real inference. With imgsz it carries a checkpoint whose train_args hold that image size.
    """

    def __init__(self, path=None, cls=5, imgsz=None):
        self.path, self.cls, self.calls = path, cls, []
        self.names = dict(enumerate(helpers.CLASS_NAMES))
        if imgsz:
            self.ckpt = {"train_args": {"imgsz": imgsz}}

    def predict(self, image, **kwargs):
        self.calls.append((image, kwargs))
        boxes = SimpleNamespace(cls=np.array([float(self.cls)]), conf=np.array([0.9]),
                                xyxy=np.array([[10.0, 10.0, 50.0, 40.0]]))
        return [SimpleNamespace(boxes=boxes, names=self.names)]


class TestDetect(unittest.TestCase):
    def test_conf_and_iou_reach_predict(self):
        model, img = SpyModel(), Image.new("RGB", (64, 64))
        detections, ms = helpers.detect(model, img, conf=0.6, iou=0.3, imgsz=640)
        (image, kwargs), = model.calls
        self.assertIs(image, img)
        self.assertEqual((kwargs["conf"], kwargs["iou"], kwargs["imgsz"]), (0.6, 0.3, 640))
        self.assertEqual(detections, [{"class": "PLASTIC", "conf": 0.9, "box": (10.0, 10.0, 50.0, 40.0)}])
        self.assertGreaterEqual(ms, 0)


class TestDrawing(unittest.TestCase):
    def test_box_is_drawn_in_the_class_colour(self):
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        out = helpers.draw_detections(img, [{"class": "METAL", "conf": 0.87, "box": (50.0, 80.0, 150.0, 180.0)}])
        metal = Image.new("RGB", (1, 1), helpers.class_color("METAL")).getpixel((0, 0))
        self.assertEqual(out.getpixel((100, 180)), metal)  # bottom edge of the box
        self.assertEqual(img.getpixel((100, 180)), (255, 255, 255))  # the input is left unchanged


def sidebar_radio(at, label):
    radios = [r for r in at.sidebar.radio if r.label == label]
    assert len(radios) == 1, f"expected one sidebar radio called {label!r}"
    return radios[0]


def sidebar_slider(at, label):
    return [s for s in at.sidebar.slider if s.label == label][0]


def sidebar_text(at):
    return " ".join(c.value for c in at.sidebar.caption)


class AppRunner(unittest.TestCase):
    def assert_no_exception(self, at):
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])

    def run_app(self, gen=helpers.DEFAULT_GENERATION):
        """Run app/app.py headless, switching the sidebar to `gen` if it is not the default."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(APP_FILE), default_timeout=120).run(timeout=120)
        self.assert_no_exception(at)
        self.assertEqual(sidebar_radio(at, "Generation").value, helpers.DEFAULT_GENERATION)
        if gen != helpers.DEFAULT_GENERATION:
            sidebar_radio(at, "Generation").set_value(gen).run(timeout=120)
            self.assert_no_exception(at)
        return at

    def use_spy_models(self, **spy_kwargs):
        """Replace the model loader with SpyModels for this test; returns {repo-relative path: SpyModel}."""
        import streamlit as st

        st.cache_resource.clear()  # forget real models cached by other tests, so the spy loader is called
        self.addCleanup(st.cache_resource.clear)  # and do not leave spy models in the cache
        loaded = {}

        def spy_loader(path, fallback_imgsz):
            key = Path(path).relative_to(REPO).as_posix()
            loaded[key] = SpyModel(path, **spy_kwargs)
            return loaded[key]

        patcher = mock.patch.object(helpers, "load_model", side_effect=spy_loader)
        patcher.start()
        self.addCleanup(patcher.stop)
        return loaded


class TestStreamlitMissingWeights(AppRunner):
    def test_missing_gen2_weights_give_a_clear_message(self):
        with mock.patch.dict(helpers.GENERATIONS["gen2"], {"dir": REPO / "app" / "models" / "nowhere"}):
            at = self.run_app()
        self.assertEqual(len(at.error), 1)
        self.assertIn("Model file not found: app/models/nowhere/arm_", at.error[0].value)
        self.assertIn("pick Generation 1 in the sidebar", at.error[0].value)
        self.assertIn("About this project", [e.label for e in at.expander])  # the rest of the page still shows


@unittest.skipUnless(HAVE_GEN1, "Generation 1 weights (app/models/arm_*.pt) are needed for these smoke tests")
class TestStreamlitGen1(AppRunner):
    def test_gen1_runs_at_416_with_its_default_arm(self):
        at = self.run_app("gen1")
        self.assertEqual(sidebar_radio(at, "Model").value,
                         helpers.default_model(helpers.load_results(helpers.GENERATIONS["gen1"]["results"])))
        self.assertAlmostEqual(sidebar_slider(at, "Confidence threshold").value, 0.35)
        text = sidebar_text(at)
        self.assertIn("image size 416 px", text)
        self.assertIn("statistically tied", text)
        self.assertNotIn("OTHER", text)  # the 6-class model's names
        self.assertEqual(len(at.error), 0)

    def test_uploads_of_a_corrupt_file_and_an_empty_scene(self):
        at = self.run_app("gen1")
        plain = image_bytes(Image.new("RGB", (640, 480), (120, 160, 200)), "PNG")
        at.file_uploader[0].set_value([("broken.jpg", b"not an image", "image/jpeg"),
                                       ("plain.png", plain, "image/png")]).run(timeout=120)
        self.assert_no_exception(at)
        self.assertEqual([s.value for s in at.main.subheader], ["broken.jpg", "plain.png"])
        self.assertEqual(len(at.error), 1)
        self.assertIn("broken.jpg", at.error[0].value)
        self.assertEqual(len(at.warning), 1)
        self.assertIn("No waste items detected", at.warning[0].value)

    def test_sidebar_choices_reach_the_model(self):
        """The chosen model file is loaded, and the chosen conf and IoU and the checkpoint's imgsz reach predict."""
        loaded = self.use_spy_models(imgsz=352)  # an imgsz no real model uses, so it must come from train_args
        default = helpers.default_model(helpers.load_results(helpers.GENERATIONS["gen1"]["results"]))
        other = "B" if default != "B" else "C"
        at = self.run_app("gen1")
        sidebar_slider(at, "Confidence threshold").set_value(0.6)
        sidebar_slider(at, "IoU threshold (NMS)").set_value(0.3)
        photo = image_bytes(Image.new("RGB", (80, 60), (200, 200, 200)), "PNG")
        at.file_uploader[0].set_value(("photo.png", photo, "image/png")).run(timeout=120)
        sidebar_radio(at, "Model").set_value(other).run(timeout=120)
        self.assert_no_exception(at)
        first = helpers.model_path("gen1", default).relative_to(REPO).as_posix()
        second = helpers.model_path("gen1", other).relative_to(REPO).as_posix()
        self.assertEqual(list(loaded)[-2:], [first, second])
        for key in (first, second):
            _, kwargs = loaded[key].calls[-1]
            self.assertEqual((kwargs["conf"], kwargs["iou"], kwargs["imgsz"]), (0.6, 0.3, 352), key)
        self.assertIn("image size 352 px", sidebar_text(at))
        self.assertIn("PLASTIC", at.dataframe[0].value["Item"].tolist())

    @unittest.skipUnless(helpers.test_images(), "the local dataset and configs/lists/test.txt are not available")
    def test_random_test_image_gives_a_result(self):
        at = self.run_app("gen1")
        [r for r in at.radio if r.label == "Input"][0].set_value("Random test image").run(timeout=120)
        self.assert_no_exception(at)
        self.assertIn("Inference time", [m.label for m in at.metric])
        self.assertEqual(len(at.error), 0)


@unittest.skipUnless(HAVE_GEN2, "Generation 2 weights (app/models/gen2/arm_*.pt) are needed for these smoke tests")
class TestStreamlitGen2(AppRunner):
    def test_gen2_is_the_default_and_runs_at_640(self):
        at = self.run_app()
        self.assertEqual(sidebar_radio(at, "Model").value,
                         helpers.default_model(helpers.load_results(helpers.GENERATIONS["gen2"]["results"])))
        text = sidebar_text(at)
        self.assertIn("image size 640 px", text)
        self.assertIn("OTHER", text)  # the 7-class model's names
        self.assertIn("B and C significantly beat A; C vs B is a tie.", text)
        self.assertEqual(len(at.error), 0)

    def test_switching_generation_keeps_each_arm_choice(self):
        at = self.run_app()
        sidebar_radio(at, "Model").set_value("A").run(timeout=120)
        sidebar_radio(at, "Generation").set_value("gen1").run(timeout=120)
        sidebar_radio(at, "Model").set_value("B").run(timeout=120)
        sidebar_radio(at, "Generation").set_value("gen2").run(timeout=120)
        self.assert_no_exception(at)
        self.assertEqual(sidebar_radio(at, "Model").value, "A")
        self.assertIn("image size 640 px", sidebar_text(at))

    def test_other_detection_goes_to_the_reject_bin(self):
        self.use_spy_models(cls=helpers.CLASS_NAMES.index("OTHER"))
        at = self.run_app()
        photo = image_bytes(Image.new("RGB", (80, 60), (200, 200, 200)), "PNG")
        at.file_uploader[0].set_value(("photo.png", photo, "image/png")).run(timeout=120)
        self.assert_no_exception(at)
        table = at.dataframe[0].value
        self.assertEqual(table["Item"].tolist(), ["OTHER"])
        self.assertEqual(table["Bin"].tolist(), [REJECT])
        self.assertIn(REJECT, [m.label for m in at.metric])

    @unittest.skipUnless(helpers.test_images(), "the local dataset and configs/lists/test.txt are not available")
    def test_random_test_image_gives_a_result(self):
        at = self.run_app()
        [r for r in at.radio if r.label == "Input"][0].set_value("Random test image").run(timeout=120)
        self.assert_no_exception(at)
        self.assertIn("Inference time", [m.label for m in at.metric])
        self.assertEqual(len(at.error), 0)


if __name__ == "__main__":
    unittest.main()
