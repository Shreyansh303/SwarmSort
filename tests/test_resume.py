"""Tests for train_final.py --resume (training.resume_run).

The fast tests need no training. test_slow_stall_then_resume really trains YOLOv8n on CPU (20 images, imgsz 64,
3 epochs, about 1-2 minutes; needs the local Gen 1 dataset): a patched trainer freezes after epoch 1, the watchdog kills it and retries with
--resume, and the run is finished from epoch 2. Skip it with SWARMSORT_SKIP_SLOW=1.
"""
import contextlib
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import training  # noqa: E402
from search_space import DEFAULTS  # noqa: E402
from watchdog_run import run_watched  # noqa: E402


class FakeYOLO:
    calls = []

    def __init__(self, model):
        self.model = model

    def train(self, **args):
        FakeYOLO.calls.append(args)
        self.trainer = SimpleNamespace(save_dir=Path(args["project"]) / args["name"],
                                       optimizer=SimpleNamespace(param_groups=[{"initial_lr": args["lr0"]}]))
        return SimpleNamespace(box=SimpleNamespace(map50=0.5, map=0.3, mp=0.6, mr=0.4))


class TestResumeHelpers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_run_args_equal_what_train_run_passes(self):
        """resume_run checks a checkpoint against run_args, so run_args must be exactly train_run's arguments."""
        import ultralytics

        params = {**DEFAULTS, "lr0": 0.0234, "mosaic": 0.61}
        cases = [(params, "d.yaml", 100, 640, "arm_c", self.dir / "runs", True, "0", True),
                 (DEFAULTS, "p.yaml", 12, 416, "pso_e12_r1_p1", "rel/runs", False, None, False)]
        for p, data, epochs, imgsz, name, project, val, device, plots in cases:
            FakeYOLO.calls.clear()
            with mock.patch.object(ultralytics, "YOLO", FakeYOLO):
                training.train_run(p, data, epochs, imgsz, name, project, val=val, device=device, plots=plots)
            self.assertEqual(FakeYOLO.calls[0],
                             training.run_args(p, data, epochs, imgsz, name, project, val, device=device, plots=plots))

    def test_mismatches(self):
        wanted = training.run_args(DEFAULTS, self.dir / "d.yaml", 100, 640, "arm_c", self.dir, True, device="0")
        saved = {k: v for k, v in wanted.items()}
        saved.update(device="cpu", project="/elsewhere", lr0=0.01 + 1e-15, data=str(self.dir / "x" / ".." / "d.yaml"))
        self.assertEqual(training.resume_mismatches(saved, wanted), [])
        saved.update(epochs=50, lr0=0.02, optimizer="auto")
        del saved["mosaic"]
        diff = "\n".join(training.resume_mismatches(saved, wanted))
        for k in ("epochs", "lr0", "optimizer", "mosaic: missing"):
            self.assertIn(k, diff)

    def test_trim_and_seconds(self):
        f = self.dir / "results.csv"
        f.write_text("epoch,time,metrics/mAP50(B)\n1,50,0.1\n2,100,0.2\n3,150,0.3\n4,40,0.35\n5,90,0.4\n6,95,0.41\n")
        with contextlib.redirect_stdout(io.StringIO()):
            rows = training.trim_epochs_csv(f, 5)
        self.assertEqual([r["epoch"] for r in rows], ["1", "2", "3", "4", "5"])
        self.assertEqual(len(f.read_text().splitlines()), 6)
        self.assertEqual(training.csv_training_seconds(rows), 150 + 90)  # two sessions
        self.assertEqual(training.trim_epochs_csv(self.dir / "missing.csv", 3), [])


FREEZER = r'''
"""train_final.py whose trainer freezes (sleeps) after saving epoch 1, unless the marker file exists."""
import os, runpy, sys, time
MARKER = sys.argv.pop(1)
from ultralytics.engine import trainer as T
save_model = T.BaseTrainer.save_model


def freezing_save_model(self):
    save_model(self)
    if self.epoch == 0 and not os.path.exists(MARKER):
        open(MARKER, "w").close()
        print("simulated freeze after epoch 1", flush=True)
        time.sleep(3600)


T.BaseTrainer.save_model = freezing_save_model
sys.argv = [sys.argv[1]] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(sys.argv[0]))  # as when the script is run directly
runpy.run_path(sys.argv[0], run_name="__main__")
'''


TRAIN_LIST = REPO / "configs" / "lists" / "train.txt"


@unittest.skipIf(os.environ.get("SWARMSORT_SKIP_SLOW") == "1", "slow CPU training test skipped (SWARMSORT_SKIP_SLOW=1)")
@unittest.skipUnless(TRAIN_LIST.exists() and (REPO / "yolov8n.pt").exists(),
                     "needs the local Gen 1 dataset (configs/lists/train.txt, see README Phase 1) and yolov8n.pt")
class TestResumeTraining(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make_data(self):
        """20 Gen 1 train images (and labels) copied into a scratch dataset; val = the first 10."""
        lines = [l.strip() for l in TRAIN_LIST.read_text().splitlines() if l.strip()]
        for sub in ("images", "labels"):
            (self.dir / "data" / sub).mkdir(parents=True)
        picked = []
        for line in lines[::max(1, len(lines) // 20)][:20]:
            img = Path(line)
            shutil.copy(img, self.dir / "data" / "images" / img.name)
            shutil.copy(img.parent.parent / "labels" / (img.stem + ".txt"), self.dir / "data" / "labels")
            picked.append((self.dir / "data" / "images" / img.name).as_posix())
        (self.dir / "train.txt").write_text("\n".join(picked) + "\n")
        (self.dir / "val.txt").write_text("\n".join(picked[:10]) + "\n")
        names = (REPO / "configs" / "data.yaml").read_text().split("names:")[1]
        data = self.dir / "data.yaml"
        data.write_text(f"path: {(self.dir / 'data').as_posix()}\ntrain: {(self.dir / 'train.txt').as_posix()}\n"
                        f"val: {(self.dir / 'val.txt').as_posix()}\nnc: 6\nnames:{names}")
        return data

    def test_slow_stall_then_resume(self):
        data = self.make_data()
        params = {**DEFAULTS, "lr0": 0.02, "mosaic": 0.5}
        (self.dir / "cfg.json").write_text(json.dumps({"params": params}))
        (self.dir / "freezer.py").write_text(textwrap.dedent(FREEZER))
        out_json, project = self.dir / "arm_t.json", self.dir / "runs"
        args = [str(REPO / "src" / "train_final.py"), "--config", str(self.dir / "cfg.json"), "--data", str(data),
                "--epochs", 3, "--imgsz", 64, "--name", "arm_t", "--out", str(out_json), "--project", str(project),
                "--device", "cpu", "--model", str(REPO / "yolov8n.pt"), "--resume"]
        cmd = [sys.executable, str(self.dir / "freezer.py"), str(self.dir / "marker"), *args]
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        log = io.StringIO()
        try:
            run_watched(cmd, watch=[project / "arm_t"], stall_minutes=0.75, max_retries=1, env=env, cwd=self.dir,
                        out=log)
        finally:  # the whole log, for inspection
            text = log.getvalue()
            (Path(tempfile.gettempdir()) / "swarmsort_resume_test.log").write_text(text, encoding="utf-8")

        self.assertIn("does not exist, so this is a normal run", text)  # first attempt: no last.pt yet
        self.assertIn("simulated freeze after epoch 1", text)
        self.assertIn("STALL", text)
        self.assertIn("Resuming", text)
        res = json.loads(out_json.read_text())
        self.assertTrue(res["resumed"])
        self.assertEqual(res["resumed_from_epoch"], 1)
        self.assertEqual(res["epochs"], 3)
        self.assertEqual(res["params"], params)
        self.assertEqual(res["train_args"]["lr0"], 0.02)
        self.assertEqual(res["train_args"]["epochs"], 3)
        self.assertEqual(res["optimizer"], "SGD")
        self.assertAlmostEqual(res["initial_lr"], 0.02)
        with open(project / "arm_t" / "results.csv", newline="") as f:
            epochs = [int(float(r["epoch"])) for r in csv.DictReader(f)]
        self.assertEqual(epochs, [1, 2, 3])
        print(f"\nresumed JSON: resumed={res['resumed']} resumed_from_epoch={res['resumed_from_epoch']} "
              f"epochs={res['epochs']} lr0={res['train_args']['lr0']} mosaic={res['train_args']['mosaic']} "
              f"map50={res['map50']} seconds={res['seconds']} resume_seconds={res['resume_seconds']}")

        # other arguments are refused before anything is trained
        bad = [sys.executable, *[str(a) for a in args]]
        bad[bad.index("--epochs") + 1] = "4"
        p = subprocess.run(bad, env=env, cwd=self.dir, capture_output=True, text=True, encoding="utf-8")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("Cannot resume", p.stdout + p.stderr)
        self.assertIn("epochs: checkpoint 3, wanted 4", p.stdout + p.stderr)

        # a finished run (stripped last.pt): --resume only validates best.pt again
        first = res
        p = subprocess.run([sys.executable, *[str(a) for a in args]], env=env, cwd=self.dir, capture_output=True,
                           text=True, encoding="utf-8")
        self.assertEqual(p.returncode, 0, p.stdout[-2000:] + p.stderr[-2000:])
        res = json.loads(out_json.read_text())
        self.assertEqual(res["resumed_from_epoch"], 3)
        self.assertIn("metrics_from", res)
        self.assertAlmostEqual(res["map50"], first["map50"], delta=0.02)
        print(f"finished-run resume: map50 {res['map50']} (resumed training reported {first['map50']})")


if __name__ == "__main__":
    unittest.main()
