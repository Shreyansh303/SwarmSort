"""Tests of src/evaluate.py's pure parts: no model is loaded and no image is predicted."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from ultralytics.utils.metrics import ap_per_class

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import evaluate  # noqa: E402
from search_space import DEFAULTS, NAMES  # noqa: E402

CLASSES = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]
CLASSES7 = [*CLASSES, "OTHER"]


def fake_stats(n_images=40, seed=0, all_correct=False):
    """Per-image stats shaped like Ultralytics' (TP flags at 10 IoU thresholds, conf, classes); some images empty."""
    rng = np.random.default_rng(seed)
    per_image = []
    for i in range(n_images):
        n_targets, n_preds = int(rng.integers(0, 6)), int(rng.integers(0, 12))
        target_cls = rng.integers(0, 6, n_targets).astype(float)
        pred_cls = rng.integers(0, 6, n_preds).astype(float)
        conf = rng.random(n_preds)
        tp = rng.random((n_preds, 1)) < np.linspace(0.8, 0.2, 10)  # fewer hits at stricter IoU thresholds
        if all_correct:
            tp[:] = True
        per_image.append({"image": f"im{i}.jpg", "tp": tp, "conf": conf, "pred_cls": pred_cls,
                          "target_cls": target_cls})
    return per_image


def ap_on_concatenation(per_image):
    """Reference: Ultralytics' ap_per_class on the plain concatenation, as DetMetrics.process() does it."""
    cat = {k: np.concatenate([s[k] for s in per_image]) for k in ("tp", "conf", "pred_cls", "target_cls")}
    ap = ap_per_class(cat["tp"], cat["conf"], cat["pred_cls"], cat["target_cls"])[5]
    return float(ap[:, 0].mean()), float(ap.mean())


class TestMapFromStats(unittest.TestCase):
    def test_all_images_equal_ap_per_class(self):
        per_image = fake_stats()
        self.assertEqual(evaluate.map_from_stats(evaluate.stack_stats(per_image)), ap_on_concatenation(per_image))

    def test_resampled_images_equal_ap_per_class(self):
        per_image = fake_stats(seed=1)
        idx = np.array([3, 3, 0, 17, 39, 3, 25, 8, 8, 1])
        got = evaluate.map_from_stats(evaluate.stack_stats(per_image), idx)
        self.assertEqual(got, ap_on_concatenation([per_image[i] for i in idx]))

    def test_block_indices(self):
        starts = np.array([0, 2, 2, 5])  # image 0 -> rows 0-1, image 1 -> none, image 2 -> rows 2-4
        self.assertEqual(evaluate.block_indices(starts, [2, 0, 1, 2]).tolist(), [2, 3, 4, 0, 1, 2, 3, 4])

    def test_recording_metrics_match_ultralytics_process(self):
        """The hook records exactly what DetMetrics receives: recomputing from it gives DetMetrics' own mAP."""
        metrics = evaluate.RecordingDetMetrics(names=dict(enumerate(CLASSES)))
        for s in fake_stats(seed=2):
            metrics.update_stats({**{k: s[k] for k in ("tp", "conf", "pred_cls", "target_cls")},
                                  "target_img": np.unique(s["target_cls"]), "im_name": s["image"]})
        metrics.process()
        self.assertEqual(len(metrics.per_image), 40)
        check = evaluate.consistency_check(evaluate.stack_stats(metrics.per_image), metrics.box.map50,
                                           metrics.box.map)
        self.assertTrue(check["passed"])
        self.assertEqual(check["max_abs_diff"], 0.0)

    def test_consistency_check_fails_on_a_mismatch(self):
        flat = evaluate.stack_stats(fake_stats())
        m50, m = evaluate.map_from_stats(flat)
        self.assertTrue(evaluate.consistency_check(flat, m50, m)["passed"])
        self.assertFalse(evaluate.consistency_check(flat, m50 + 1e-3, m)["passed"])


class TestSummarizeValidator(unittest.TestCase):
    def test_class_without_targets(self):
        """Ultralytics only lists classes with targets in ap_class_index; a missing class must map to None."""
        present = np.array([0, 1, 2, 4, 5])  # no METAL (3) in this split
        box = SimpleNamespace(ap_class_index=present, ap50=np.array([0.9, 0.8, 0.7, 0.6, 0.5]),
                              ap=np.array([0.45, 0.4, 0.35, 0.3, 0.25]), map50=0.7, map=0.35, mp=0.75, mr=0.65)
        matrix = np.arange(49, dtype=float).reshape(7, 7)
        fake = SimpleNamespace(names=dict(enumerate(CLASSES)), metrics=SimpleNamespace(box=box),
                               confusion_matrix=SimpleNamespace(matrix=matrix))
        summary = evaluate.summarize_validator(fake)
        per_class = summary["per_class"]
        self.assertEqual(list(per_class), CLASSES)
        self.assertEqual(per_class["METAL"], {"ap50": None, "ap50_95": None})
        self.assertEqual(per_class["PAPER"], {"ap50": 0.6, "ap50_95": 0.3})  # 4th entry, class index 4
        self.assertEqual(per_class["PLASTIC"], {"ap50": 0.5, "ap50_95": 0.25})
        self.assertEqual(summary["metrics"], {"map50": 0.7, "map50_95": 0.35, "precision": 0.75, "recall": 0.65})
        cm = summary["confusion_matrix"]
        self.assertEqual(cm["labels"], [*CLASSES, "background"])
        np.testing.assert_allclose(np.array(cm["normalized"]).sum(0), 1.0, atol=1e-3)  # each true class sums to 1
        markdown = evaluate.comparison_markdown({
            "split": "val", "images": 1, "boxes": 1, "device": "CPU test", "val_metrics": {}, "pairwise": [],
            "settings": {"bootstrap_resamples": 1, "seed": 0},
            "models": {"A": {**summary, "speed": {"ms_per_img": 1.0},
                             "bootstrap": {"map50_ci95": [0.6, 0.8], "map50_95_ci95": [0.3, 0.4]}}}})
        self.assertIn("| METAL | - |", markdown)

    def test_seven_classes_with_other(self):
        """A Gen 2 model with the optional OTHER class: every table, matrix and plot takes the 7th class."""
        box = SimpleNamespace(ap_class_index=np.arange(7), ap50=np.linspace(0.9, 0.3, 7),
                              ap=np.linspace(0.6, 0.1, 7), map50=0.6, map=0.35, mp=0.7, mr=0.6)
        fake = SimpleNamespace(names=dict(enumerate(CLASSES7)), metrics=SimpleNamespace(box=box),
                               confusion_matrix=SimpleNamespace(matrix=np.ones((8, 8))))
        summary = evaluate.summarize_validator(fake)
        self.assertEqual(list(summary["per_class"]), CLASSES7)
        self.assertEqual(summary["per_class"]["OTHER"], {"ap50": 0.3, "ap50_95": 0.1})
        self.assertEqual(summary["confusion_matrix"]["labels"], [*CLASSES7, "background"])
        result = {**summary, "speed": {"ms_per_img": 1.0},
                  "bootstrap": {"map50_ci95": [0.5, 0.7], "map50_95_ci95": [0.3, 0.4]}}
        markdown = evaluate.comparison_markdown({
            "split": "val", "images": 1, "boxes": 1, "device": "CPU test", "val_metrics": {}, "pairwise": [],
            "settings": {"bootstrap_resamples": 1, "seed": 0, "imgsz": 640}, "models": {"A": result}})
        self.assertIn("| OTHER | 0.300 |", markdown)
        self.assertIn("latency at imgsz 640", markdown)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "per_class.png"
            evaluate.plot_per_class({"A": result, "B": result}, "val", out)
            evaluate.plot_confusion({"A": result}, "val", Path(tmp) / "confusion.png")
            self.assertEqual(out.read_bytes()[:4], b"\x89PNG")
            self.assertTrue((Path(tmp) / "confusion.png").is_file())


class TestBootstrap(unittest.TestCase):
    def setUp(self):
        self.flats = {"A": evaluate.stack_stats(fake_stats(seed=3)), "B": evaluate.stack_stats(fake_stats(seed=4))}

    def test_reproducible_with_a_seed(self):
        a = evaluate.bootstrap_maps(self.flats, 30, seed=7)
        b = evaluate.bootstrap_maps(self.flats, 30, seed=7)
        c = evaluate.bootstrap_maps(self.flats, 30, seed=8)
        self.assertEqual(a["A"].shape, (30, 2))
        for label in self.flats:
            np.testing.assert_array_equal(a[label], b[label])
        self.assertFalse(np.array_equal(a["A"], c["A"]))

    def test_identical_models_differ_by_exactly_zero(self):
        flat = self.flats["A"]
        flats = {"A": flat, "B": {k: v.copy() for k, v in flat.items()}}
        samples = evaluate.bootstrap_maps(flats, 50, seed=0)
        observed = {label: dict(zip(("map50", "map50_95"), evaluate.map_from_stats(f))) for label, f in flats.items()}
        (pair,) = evaluate.pairwise_differences(samples, observed)
        self.assertEqual(pair["pair"], "B-A")
        for key in ("map50", "map50_95"):
            self.assertEqual(pair[key]["ci95"], [0.0, 0.0])
            self.assertEqual(pair[key]["p_gt_0"], 0.0)
            self.assertEqual(pair[key]["observed_diff"], 0.0)
            self.assertFalse(pair[key]["significant"])
        self.assertTrue(pair["verdict"].startswith("Not significant"))

    def test_clearly_better_model_is_significant(self):
        flats = {"A": self.flats["A"], "C": evaluate.stack_stats(fake_stats(seed=3, all_correct=True))}
        samples = evaluate.bootstrap_maps(flats, 50, seed=0)
        observed = {label: dict(zip(("map50", "map50_95"), evaluate.map_from_stats(f))) for label, f in flats.items()}
        (pair,) = evaluate.pairwise_differences(samples, observed)
        self.assertGreater(pair["map50_95"]["ci95"][0], 0)
        self.assertEqual(pair["map50_95"]["p_gt_0"], 1.0)
        self.assertIn("PSO is significantly better on mAP@50-95", pair["verdict"])

    def test_pairs_for_three_models(self):
        flats = {**self.flats, "C": evaluate.stack_stats(fake_stats(seed=5))}
        samples = evaluate.bootstrap_maps(flats, 5, seed=0)
        observed = {label: dict(zip(("map50", "map50_95"), evaluate.map_from_stats(f))) for label, f in flats.items()}
        pairs = evaluate.pairwise_differences(samples, observed)
        self.assertEqual([p["pair"] for p in pairs], ["B-A", "C-A", "C-B"])

    def test_models_on_different_images_are_refused(self):
        flats = {"A": self.flats["A"], "B": evaluate.stack_stats(fake_stats(n_images=39))}
        with self.assertRaises(ValueError):
            evaluate.bootstrap_maps(flats, 5, seed=0)


class TestModelsArgument(unittest.TestCase):
    def parse_error(self, argv):
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
            evaluate.parse_args(argv)
        return err.getvalue()

    def test_default_labels(self):
        args = evaluate.parse_args([])
        self.assertEqual(list(args.models), ["A", "B", "C"])
        self.assertEqual(args.models["B"], evaluate.REPO / "results/runs/arm_b/weights/best.pt")
        self.assertEqual(args.split, "test")  # the final run's default; checks always pass --split val
        self.assertEqual([evaluate.display_name(x) for x in "ABC"], ["Defaults", "Random search", "PSO"])

    def test_custom_labels(self):
        args = evaluate.parse_args(["--models", "X=runs/x.pt", "A=a.pt", "--split", "val"])
        self.assertEqual(args.models, {"X": Path("runs/x.pt"), "A": Path("a.pt")})
        self.assertEqual(evaluate.display_name("X"), "X")

    def test_bad_format_is_an_error(self):
        self.assertIn("expected LABEL=path", self.parse_error(["--models", "results/best.pt"]))
        self.assertIn("expected LABEL=path", self.parse_error(["--models", "=best.pt"]))
        self.assertIn("more than once", self.parse_error(["--models", "A=a.pt", "A=b.pt"]))

    def test_missing_file_message(self):
        msg = evaluate.missing_model_message("B", evaluate.REPO / "results/runs/arm_b/weights/best.pt")
        for text in ("phase5a_train_arm_b.ipynb", "arm_b_best.pt", "results/runs/arm_b/weights/best.pt", "rename"):
            self.assertIn(text, msg)
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SystemExit) as cm:
            evaluate.require_model_files({"C": Path(tmp) / "best.pt"})
        self.assertIn("phase5b_train_arm_c.ipynb", str(cm.exception.code))


class TestCheckpointCheck(unittest.TestCase):
    def info(self, **changes):
        train_args = {"epochs": 100, "imgsz": 416, "max_det": 321, **DEFAULTS}
        names = changes.pop("names", CLASSES)
        train_args.update(changes)
        return {"names": names, "train_args": train_args}

    def test_match(self):
        self.assertEqual(evaluate.check_checkpoint(self.info(), CLASSES, dict(DEFAULTS)), [])
        nearly = self.info(lr0=DEFAULTS["lr0"] * (1 + 1e-7))  # within the relative tolerance of 1e-6
        self.assertEqual(evaluate.check_checkpoint(nearly, CLASSES, dict(DEFAULTS)), [])

    def test_hyperparameter_mismatch(self):
        problems = evaluate.check_checkpoint(self.info(lr0=0.00702571), CLASSES, dict(DEFAULTS))
        self.assertEqual(len(problems), 1)
        self.assertIn("lr0", problems[0])

    def test_wrong_class_names_and_schedule(self):
        coco = self.info(names=[f"class{i}" for i in range(80)], epochs=3)
        problems = evaluate.check_checkpoint(coco, CLASSES, None)
        self.assertTrue(any("80 classes" in p for p in problems))
        self.assertTrue(any("epochs=3" in p for p in problems))
        swapped = self.info(names=[CLASSES[1], CLASSES[0], *CLASSES[2:]])
        self.assertEqual(len(evaluate.check_checkpoint(swapped, CLASSES, None)), 1)

    def test_other_labels_skip_the_config_comparison(self):
        self.assertIsNone(evaluate.expected_params("X"))
        self.assertEqual(evaluate.check_checkpoint(self.info(lr0=0.5), CLASSES, None), [])

    def test_expected_configs_of_the_arms(self):
        for label, file in (("A", "baseline.json"), ("B", "best_random.json"), ("C", "best_pso.json")):
            params = json.loads((evaluate.REPO / "results" / file).read_text())["params"]
            self.assertEqual(evaluate.expected_params(label), {k: float(params[k]) for k in NAMES})

    def test_non_default_epochs_and_imgsz(self):
        """A Gen 2 checkpoint (60 epochs at 640) passes against a 60/640 config and fails the Gen 1 one."""
        gen2 = self.info(epochs=60, imgsz=640)
        self.assertEqual(evaluate.check_checkpoint(gen2, CLASSES, dict(DEFAULTS), epochs=60, imgsz=640), [])
        problems = evaluate.check_checkpoint(gen2, CLASSES, dict(DEFAULTS))  # Gen 1 defaults: 100 and 416
        self.assertEqual(len(problems), 2)
        self.assertTrue(any("imgsz=640, expected 416" in p for p in problems))
        self.assertEqual(evaluate.check_checkpoint(gen2, CLASSES, None, epochs=None, imgsz=None), [])

    def test_run_checks_with_an_arms_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "arm_a.json"
            cfg.write_text(json.dumps({"params": dict(DEFAULTS), "epochs": 60, "imgsz": 640}))
            arms = {"A": {"name": "Defaults", "model": "a.pt", "config": str(cfg), "val_json": str(cfg),
                          "epochs_csv": None, "epochs": None, "imgsz": None}}
            self.assertEqual(evaluate.expected_schedule("A", arms), (60, 640))  # read from the arm's JSON
            self.assertEqual(evaluate.eval_imgsz(["A"], arms), 640)
            models = {"A": Path("a.pt")}
            evaluate.run_checks(models, {"A": self.info(epochs=60, imgsz=640, names=CLASSES7)}, CLASSES7, arms)
            with self.assertRaises(SystemExit) as cm:
                evaluate.run_checks(models, {"A": self.info(names=CLASSES7)}, CLASSES7, arms)
            self.assertIn("epochs=100, expected 60", str(cm.exception.code))
            self.assertIn("imgsz=416, expected 640", str(cm.exception.code))
            with self.assertRaises(SystemExit) as cm:  # a 6-class checkpoint against a 7-class --data yaml
                evaluate.run_checks(models, {"A": self.info(epochs=60, imgsz=640)}, CLASSES7, arms)
            self.assertIn("the --data yaml has 7", str(cm.exception.code))
            unknown = {"A": {**arms["A"], "val_json": str(Path(tmp) / "not_trained_yet.json")}}
            with self.assertRaises(SystemExit) as cm:
                evaluate.run_checks(models, {"A": self.info(epochs=60, imgsz=640)}, CLASSES, unknown)
            self.assertIn("expected epochs is unknown", str(cm.exception.code))

    def test_schedule_of_other_labels_and_eval_imgsz(self):
        self.assertEqual(evaluate.expected_schedule("X"), (100, 416))  # Gen 1: what every arm shares
        self.assertEqual(evaluate.eval_imgsz(["A", "B", "C", "X"]), 416)
        mixed = {"A": {**evaluate.GEN1_ARMS["A"], "imgsz": 640}, "B": evaluate.GEN1_ARMS["B"]}
        self.assertEqual(evaluate.expected_schedule("X", mixed), (100, None))
        with self.assertRaises(SystemExit):
            evaluate.eval_imgsz(["A", "B"], mixed)


class TestArmsConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, obj):
        path = self.dir / name
        path.write_text(json.dumps(obj) if name.endswith(".json") else yaml.safe_dump(obj))
        return path

    def parse_error(self, argv):
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
            evaluate.parse_args(argv)
        return err.getvalue()

    def test_default_is_the_gen1_table(self):
        args = evaluate.parse_args([])
        self.assertIs(args.arms_table, evaluate.GEN1_ARMS)
        gen1 = {"A": ("results/baseline.json", "results/baseline.json", "results/baseline_epochs.csv"),
                "B": ("results/best_random.json", "results/arm_b.json", "results/arm_b_epochs.csv"),
                "C": ("results/best_pso.json", "results/arm_c.json", "results/arm_c_epochs.csv")}
        for label, (config, val_json, csv) in gen1.items():
            arm = evaluate.GEN1_ARMS[label]
            self.assertEqual((arm["config"], arm["val_json"], arm["epochs_csv"]), (config, val_json, csv))
            self.assertEqual(evaluate.expected_schedule(label), (100, 416))
        # the same table written to a file (yaml or json) reads back unchanged
        for name in ("gen1.yaml", "gen1.json"):
            self.assertEqual(evaluate.load_arms(self.write(name, {"arms": evaluate.GEN1_ARMS})), evaluate.GEN1_ARMS)

    def test_gen2_template(self):
        template = evaluate.REPO / "configs" / "gen2" / "arms.yaml"
        arms = evaluate.load_arms(template)
        self.assertEqual(list(arms), ["A", "B", "C"])
        self.assertEqual([arms[x]["name"] for x in "ABC"], ["Defaults", "Random search", "PSO"])
        for x, cfg in zip("abc", ("arm_a", "best_random", "best_pso")):
            arm = arms[x.upper()]
            self.assertEqual(arm["model"], f"results/gen2/runs/arm_{x}/weights/best.pt")
            self.assertEqual(arm["config"], f"results/gen2/{cfg}.json")
            self.assertEqual(arm["val_json"], f"results/gen2/arm_{x}.json")
            self.assertEqual(arm["epochs_csv"], f"results/gen2/arm_{x}_epochs.csv")
            self.assertIsNone(arm["epochs"])  # read from the arm's JSON after training
            self.assertIsNone(arm["imgsz"])
        args = evaluate.parse_args(["--arms", str(template), "--split", "val"])
        self.assertEqual(args.models["C"], evaluate.REPO / "results/gen2/runs/arm_c/weights/best.pt")
        self.assertEqual(evaluate.display_name("B", args.arms_table), "Random search")

    def test_minimal_arm_and_custom_names(self):
        arms = evaluate.load_arms(self.write("a.yaml", {"arms": {"A": {"model": "a.pt", "imgsz": 640},
                                                                 "G2": {"name": "PSO gen 2", "model": "c.pt"}}}))
        self.assertEqual(arms["A"]["name"], "A")
        self.assertIsNone(arms["A"]["config"])
        self.assertIsNone(evaluate.expected_params("A", arms))  # no config: no hyperparameter comparison
        self.assertEqual(evaluate.arm_setting(arms["A"], "imgsz"), 640)
        self.assertIsNone(evaluate.arm_setting(arms["A"], "epochs"))
        self.assertEqual(evaluate.arm_title("G2", arms), "G2: PSO gen 2")
        self.assertIn("check the model path in the arms table", evaluate.missing_model_message("A", "a.pt", arms))

    def test_bad_arms_files(self):
        cases = [({"A": {"model": "a.pt"}}, "top-level 'arms:'"),
                 ({"arms": {"A": {"name": "x"}}}, "needs at least a 'model'"),
                 ({"arms": {"A": {"model": "a.pt", "imgsize": 640}}}, "unknown key"),
                 ({"arms": {"A": {"model": "a.pt", "epochs": "sixty"}}}, "whole number")]
        for k, (obj, message) in enumerate(cases):
            self.assertIn(message, self.parse_error(["--arms", str(self.write(f"bad{k}.yaml", obj))]))
        self.assertIn("--arms", self.parse_error(["--arms", str(self.dir / "missing.yaml")]))


class TestTag(unittest.TestCase):
    def test_tag_from_the_data_yaml(self):
        self.assertEqual(evaluate.output_tag("gen2/data_test_real_world.yaml"), "real_world")
        self.assertEqual(evaluate.output_tag("gen2/data_test_studio.yaml"), "studio")
        self.assertEqual(evaluate.output_tag("configs/data.yaml"), "")
        self.assertEqual(evaluate.output_tag("gen2/data_test_studio.yaml", "mine"), "mine")  # --tag wins
        self.assertEqual(evaluate.output_tag("gen2/data_test_studio.yaml", ""), "")  # --tag "" turns it off

    def test_output_names(self):
        self.assertEqual(evaluate.output_prefix("test", ""), "test")  # Gen 1: test_results.json etc.
        self.assertEqual(evaluate.output_prefix("test", "real_world"), "test_real_world")
        self.assertEqual(evaluate.parse_args([]).tag, "")
        args = evaluate.parse_args(["--data", "x/data_test_real_world.yaml", "--split", "val"])
        self.assertEqual(evaluate.output_prefix(args.split, args.tag), "val_real_world")
        self.assertEqual(evaluate.parse_args(["--tag", "studio", "--split", "val"]).tag, "studio")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            evaluate.parse_args(["--tag", "../escape"])


class TestReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_training_curves_with_one_arm_missing(self):
        csv = self.dir / "a_epochs.csv"
        # Ultralytics' older results.csv files pad the header names with spaces
        csv.write_text("                  epoch,   metrics/mAP50(B),  metrics/mAP50-95(B)\n"
                       "1,0.20,0.10\n2,0.35,0.20\n3,0.31,0.19\n")
        out = self.dir / "plots" / "training_curves.png"
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            drawn = evaluate.plot_training_curves({"A": csv, "B": self.dir / "missing.csv"}, out)
        self.assertEqual(drawn, ["A"])
        self.assertEqual(out.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertIn("missing.csv not found", stdout.getvalue())

    def test_training_curves_with_no_file(self):
        out = self.dir / "none.png"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(evaluate.plot_training_curves({"A": self.dir / "nope.csv"}, out), [])
        self.assertFalse(out.exists())

    def fake_report(self, labels):
        models, samples = {}, {}
        for k, label in enumerate(labels):
            flat = evaluate.stack_stats(fake_stats(seed=10 + k))
            m50, m = evaluate.map_from_stats(flat)
            samples[label] = evaluate.bootstrap_maps({label: flat}, 20, seed=0)[label]
            models[label] = {"metrics": {"map50": m50, "map50_95": m, "precision": 0.7, "recall": 0.6},
                             "per_class": {c: {"ap50": 0.5, "ap50_95": 0.3} for c in CLASSES},
                             "speed": {"ms_per_img": 31.4},
                             "bootstrap": {"map50_ci95": evaluate.percentile_ci(samples[label][:, 0]),
                                           "map50_95_ci95": evaluate.percentile_ci(samples[label][:, 1])}}
        pairwise = evaluate.pairwise_differences(samples, {k: v["metrics"] for k, v in models.items()}) \
            if len(labels) > 1 else []
        return {"split": "val", "images": 40, "boxes": 99, "device": "CPU test", "models": models,
                "val_metrics": {"A": {"map50": 0.657}}, "pairwise": pairwise,
                "settings": {"bootstrap_resamples": 20, "seed": 0}}

    def test_markdown_for_one_and_three_models(self):
        one = evaluate.comparison_markdown(self.fake_report(["A"]))
        self.assertIn("| A: Defaults | 0.657 |", one)
        self.assertIn("need at least 2 models", one)
        three = evaluate.comparison_markdown(self.fake_report(["A", "B", "C"]))
        self.assertIn("| B: Random search | - |", three)
        self.assertIn("PSO − Random search (C−B)", three)
        self.assertIn("| PLASTIC | 0.500 | 0.500 | 0.500 |", three)
        self.assertTrue(three.startswith("## Final comparison on the val split\n"))  # no tag: Gen 1 title

    def test_markdown_with_a_tag_and_an_arms_table(self):
        report = {**self.fake_report(["A", "B"]), "tag": "real_world"}
        arms = {"A": {"name": "Gen 2 defaults"}, "B": {"name": "Gen 2 random"}}
        markdown = evaluate.comparison_markdown(report, arms)
        self.assertTrue(markdown.startswith("## Final comparison on the val split (real_world)\n"))
        self.assertIn("| A: Gen 2 defaults |", markdown)
        self.assertIn("Gen 2 random − Gen 2 defaults (B−A)", markdown)


if __name__ == "__main__":
    unittest.main()
