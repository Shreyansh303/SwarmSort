import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import plot_search  # noqa: E402
import search  # noqa: E402
from search_space import HIGH, LOW, encode  # noqa: E402


class Interrupted(Exception):
    pass


def fake_fitness(stop_after=None):
    """A deterministic score of the hyperparameters in place of a proxy training (no Ultralytics)."""
    calls = []

    def fitness(params, name, args):
        if stop_after is not None and len(calls) >= stop_after:
            raise Interrupted(name)
        calls.append(name)
        u = (encode(params) - LOW) / (HIGH - LOW)
        score = round(float(np.exp(-np.sum((u - 0.3) ** 2))), 5)
        return {"map50": score, "map50_95": score / 2, "seconds": 1.0, "device": "fake", "ultralytics": "fake",
                "run_dir": f"runs/{name}", "finished_at": "2026-01-01T00:00:00+00:00"}
    return fitness


class TestPlotSearch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.check = self.dir / "proxy_check.json"
        self.check.write_text(json.dumps({"runs": {"proxy_e12_s416_defaults": {"map50": 0.47648}}}))

    def tearDown(self):
        self.tmp.cleanup()

    def search(self, method, stop_after=None):
        """Write a log in search.py's own format by running search.py with a fake fitness."""
        argv = ["--method", method, "--log", str(self.dir / f"search_{method}.json"),
                "--best", str(self.dir / f"best_{method}.json"), "--project", str(self.dir / "runs")]
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                search.main(argv, fitness=fake_fitness(stop_after))
            except Interrupted:
                pass
        return json.loads((self.dir / f"search_{method}.json").read_text())

    def plot(self, out="conv.png"):
        argv = ["--pso", str(self.dir / "search_pso.json"), "--random", str(self.dir / "search_random.json"),
                "--proxy-check", str(self.check), "--out", str(self.dir / out)]
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            summaries, reference = plot_search.main(argv)
        return summaries, reference, stdout.getvalue()

    def assert_png(self, name):
        path = self.dir / name
        self.assertTrue(path.exists())
        self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_both_arms(self):
        logs = {m: self.search(m) for m in ("pso", "random")}
        summaries, reference, text = self.plot()
        self.assert_png("conv.png")
        self.assertEqual(reference, 0.47648)
        for method, log in logs.items():
            s = summaries[method]
            self.assertEqual(s["evaluations"], 32)
            self.assertEqual(s["best_map50"], log["summary"]["best_map50"])
            self.assertEqual(s["best_eval"], log["summary"]["best_index"] + 1)
            self.assertEqual(s["best_params"], log["summary"]["best_params"])
            self.assertEqual(s["best_so_far"], log["summary"]["best_so_far"])
            self.assertIn(f"{s['best_map50']:.4f}", text)

    def test_one_arm_only(self):
        log = self.search("random")
        summaries, _, text = self.plot("random_only.png")
        self.assert_png("random_only.png")
        self.assertEqual(list(summaries), ["random"])
        self.assertEqual(summaries["random"]["best_eval"], log["summary"]["best_index"] + 1)
        self.assertFalse([line for line in text.splitlines() if line.startswith("pso ")])

    def test_unfinished_log(self):
        log = self.search("pso", stop_after=11)  # an interrupted search has no "summary" yet
        self.assertNotIn("summary", log)
        summaries, _, _ = self.plot("partial.png")
        self.assert_png("partial.png")
        scores = [e["map50"] for e in log["evaluations"]]
        self.assertEqual(summaries["pso"]["evaluations"], 11)
        self.assertEqual(summaries["pso"]["best_map50"], max(scores))
        self.assertEqual(summaries["pso"]["best_eval"], scores.index(max(scores)) + 1)

    def test_no_log_is_an_error(self):
        with self.assertRaises(SystemExit) as cm:
            self.plot()
        self.assertIn("No search log found", str(cm.exception.code))


if __name__ == "__main__":
    unittest.main()
