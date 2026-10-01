import contextlib
import inspect
import io
import json
import sys
import tempfile
import warnings
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import search  # noqa: E402
import training  # noqa: E402
from search_space import HIGH, LOW, encode  # noqa: E402


class Interrupted(Exception):
    pass


class FakeFitness:
    """Stands in for a proxy training: a smooth deterministic score of the hyperparameters, no Ultralytics."""

    def __init__(self, fail_after=None, nan_if=lambda name: False):
        self.names, self.fail_after, self.nan_if = [], fail_after, nan_if

    def __call__(self, params, name, args):
        if self.fail_after is not None and len(self.names) >= self.fail_after:
            raise Interrupted(f"session died before {name}")
        self.names.append(name)
        u = (encode(params) - LOW) / (HIGH - LOW)  # 0..1 per dimension
        score = float("nan") if self.nan_if(name) else float(np.exp(-np.sum((u - 0.3) ** 2)))
        return {"map50": round(score, 5), "map50_95": round(score / 2, 5), "seconds": 1.0, "device": "fake",
                "ultralytics": "fake", "run_dir": f"runs/{name}", "finished_at": "2026-01-01T00:00:00+00:00"}


def run(argv, fitness):
    with contextlib.redirect_stdout(io.StringIO()):
        return search.main(argv, fitness=fitness)


class TestSearch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def argv(self, method, tag="a", *extra):
        return ["--method", method, "--log", str(self.dir / f"{tag}_{method}.json"),
                "--best", str(self.dir / f"{tag}_best_{method}.json"), "--project", str(self.dir / "runs"), *extra]

    def test_budget_is_32_for_both_methods(self):
        for method in ("pso", "random"):
            fake = FakeFitness()
            log = run(self.argv(method), fake)
            self.assertEqual(len(fake.names), 32)
            self.assertEqual(len(set(fake.names)), 32)
            self.assertEqual([e["index"] for e in log["evaluations"]], list(range(32)))
            self.assertTrue(log["complete"])

    def test_pso_rounds_and_random_points(self):
        log = run(self.argv("pso"), FakeFitness())
        self.assertEqual([e["round"] for e in log["evaluations"]], [r for r in (1, 2, 3, 4) for _ in range(8)])
        self.assertEqual(len(log["summary"]["gbest_per_round"]), 4)
        log = run(self.argv("random"), FakeFitness())
        expected = np.round(search.sample(32, seed=123), 6)
        np.testing.assert_allclose([e["vector"] for e in log["evaluations"]], expected, atol=1e-9)

    def test_resume_repeats_nothing_and_matches_uninterrupted_run(self):
        for method in ("pso", "random"):
            for k in (0, 5, 8, 31):
                full = run(self.argv(method, "full"), FakeFitness())
                first = FakeFitness(fail_after=k)
                with self.assertRaises(Interrupted):
                    run(self.argv(method, f"cut{k}"), first)
                log_path = self.dir / f"cut{k}_{method}.json"
                self.assertEqual(len(json.loads(log_path.read_text())["evaluations"]), k)
                second = FakeFitness()
                resumed = run(self.argv(method, f"cut{k}"), second)
                self.assertEqual(len(first.names) + len(second.names), 32, f"{method} k={k}")
                self.assertFalse(set(first.names) & set(second.names), "an evaluation was repeated")
                self.assertEqual(resumed, full)
                self.assertEqual(json.loads(log_path.read_text()),
                                 json.loads((self.dir / f"full_{method}.json").read_text()))
                again = FakeFitness()  # a finished log replays without any training
                run(self.argv(method, f"cut{k}"), again)
                self.assertEqual(again.names, [])

    def test_mismatched_settings_refused(self):
        with self.assertRaises(Interrupted):
            run(self.argv("pso"), FakeFitness(fail_after=3))
        for extra, key in ((["--seed", "7"], "seed"), (["--proxy-epochs", "20"], "proxy_epochs"),
                           (["--n-particles", "6"], "n_particles"), (["--imgsz", "640"], "imgsz"),
                           (["--data", "other.yaml"], "data")):
            fake = FakeFitness()
            with self.assertRaises(SystemExit) as cm:
                run(self.argv("pso", "a", *extra), fake)
            self.assertIn(key, str(cm.exception.code))
            self.assertEqual(fake.names, [])
        # a random-search log is not resumed as PSO
        run(self.argv("random", "r"), FakeFitness())
        with self.assertRaises(SystemExit) as cm:
            run(["--method", "pso", "--log", str(self.dir / "r_random.json"), "--best", str(self.dir / "x.json")],
                FakeFitness())
        self.assertIn("method", str(cm.exception.code))
        # nor a log made with other search-space bounds
        log_path = self.dir / "a_pso.json"
        log = json.loads(log_path.read_text())
        log["settings"]["search_space"]["high"][0] = 0.0
        log_path.write_text(json.dumps(log))
        with self.assertRaises(SystemExit) as cm:
            run(self.argv("pso"), FakeFitness())
        self.assertIn("search_space", str(cm.exception.code))

    def test_tampered_position_refused(self):
        with self.assertRaises(Interrupted):
            run(self.argv("pso"), FakeFitness(fail_after=4))
        log_path = self.dir / "a_pso.json"
        log = json.loads(log_path.read_text())
        log["evaluations"][2]["vector"][0] += 0.01
        log_path.write_text(json.dumps(log))
        with self.assertRaises(SystemExit) as cm:
            run(self.argv("pso"), FakeFitness())
        self.assertIn("evaluation 2", str(cm.exception.code))

    def test_best_file_matches_max_and_loads_in_train_final(self):
        import train_final

        for method in ("pso", "random"):
            log = run(self.argv(method), FakeFitness())
            best_path = self.dir / f"a_best_{method}.json"
            best = json.loads(best_path.read_text())
            top = max(e["map50"] for e in log["evaluations"])
            self.assertEqual(best["map50"], top)
            entry = log["evaluations"][best["index"]]
            self.assertEqual(entry["map50"], top)
            self.assertEqual(train_final.load_params(str(best_path)), entry["params"])

    def assert_no_nan_token(self, *paths):
        for path in paths:
            text = Path(path).read_text()
            for token in ("NaN", "Infinity"):
                self.assertNotIn(token, text, f"{path} is not valid JSON")

    def test_nan_at_first_evaluation_never_best(self):
        for method in ("pso", "random"):
            fake = FakeFitness(nan_if=lambda name: name.endswith(("_r1_p1", "_s01")))
            log = run(self.argv(method, "nan"), fake)
            first = log["evaluations"][0]
            self.assertTrue(first["failed"])
            self.assertEqual(first["map50"], 0.0)
            self.assertIsNone(first["map50_95"])
            self.assertNotEqual(log["summary"]["best_index"], 0)
            self.assertGreater(log["summary"]["best_map50"], 0.0)
            self.assert_no_nan_token(self.dir / f"nan_{method}.json", self.dir / f"nan_best_{method}.json")

    def test_all_nan_pso_round_resumes(self):
        nan_round1 = lambda name: "_r1_" in name  # noqa: E731
        full = run(self.argv("pso", "full"), FakeFitness(nan_if=nan_round1))
        self.assertTrue(all(e.get("failed") for e in full["evaluations"][:8]))
        self.assertFalse(any(e.get("failed") for e in full["evaluations"][8:]))
        first = FakeFitness(fail_after=11, nan_if=nan_round1)
        with self.assertRaises(Interrupted):
            run(self.argv("pso", "cut"), first)
        second = FakeFitness(nan_if=nan_round1)
        resumed = run(self.argv("pso", "cut"), second)
        self.assertEqual(len(first.names) + len(second.names), 32)
        self.assertEqual(resumed, full)
        self.assert_no_nan_token(self.dir / "cut_pso.json", self.dir / "cut_best_pso.json")

    def test_all_failed_writes_no_best(self):
        with self.assertRaises(SystemExit) as cm:
            run(self.argv("random", "dead"), FakeFitness(nan_if=lambda name: True))
        self.assertIn("failed", str(cm.exception.code))
        self.assertFalse((self.dir / "dead_best_random.json").exists())
        self.assert_no_nan_token(self.dir / "dead_random.json")

    def test_smoke_defaults(self):
        args = search.parse_args(["--method", "pso", "--smoke"])
        self.assertEqual((args.n_particles, args.n_iters, args.proxy_epochs), (2, 2, 2))
        self.assertTrue(Path(args.log).name == "smoke_search_pso.json")
        self.assertTrue(Path(args.best).name == "smoke_best_pso.json")
        self.assertEqual(Path(args.project).name, "runs_smoke")
        args = search.parse_args(["--method", "random", "--smoke"])
        self.assertEqual((args.n_samples, args.seed), (3, 123))

    def test_training_call_matches_proxy_check(self):
        """search.proxy_fitness must call train_run exactly as proxy_check.py calls its proxy runs."""
        import proxy_check

        def bound(call):
            b = inspect.signature(training.train_run).bind(*call.args, **call.kwargs)
            b.apply_defaults()
            return {k: v for k, v in b.arguments.items() if k not in ("params", "name")}

        fake_result = {"map50": 0.5, "map50_95": 0.3, "precision": 0.5, "recall": 0.5, "seconds": 1.0,
                       "run_dir": "x", "optimizer": "SGD", "initial_lr": 0.01, "train_args": {}}
        for extra in ([], ["--smoke"], ["--proxy-epochs", "20"]):
            with mock.patch.object(proxy_check, "train_run", return_value=fake_result) as pc, \
                    mock.patch.object(proxy_check, "device_name", return_value="fake"), \
                    mock.patch.object(sys, "argv", ["proxy_check.py", "--log", str(self.dir / "pc.json"), *extra]), \
                    contextlib.redirect_stdout(io.StringIO()):
                warnings.simplefilter("ignore")  # Spearman of constant fake scores is undefined; irrelevant here
                proxy_check.main()
            (self.dir / "pc.json").unlink()
            proxy_calls = [bound(c) for c in pc.call_args_list if "proxy_e" in c.args[4]]
            with mock.patch.object(search, "train_run", return_value=fake_result) as sc, \
                    mock.patch.object(search, "device_name", return_value="fake"):
                search.proxy_fitness({}, "n", search.parse_args(["--method", "pso", *extra]))
            self.assertEqual(bound(sc.call_args), proxy_calls[0], f"args {extra}")
            self.assertFalse(bound(sc.call_args)["val"])


if __name__ == "__main__":
    unittest.main()
