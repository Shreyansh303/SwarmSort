import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pso import PSO  # noqa: E402
from pso_benchmark import minimize, rastrigin, sphere  # noqa: E402


class TestPSO(unittest.TestCase):
    def test_sphere_converges(self):
        best, _ = minimize(sphere, seed=0)
        self.assertLess(best[-1], 1e-3)

    def test_rastrigin_monotone_and_improves(self):
        best, _ = minimize(rastrigin, seed=0)
        self.assertTrue(np.all(np.diff(best) <= 0), "best-so-far got worse")
        self.assertLess(best[-1], 0.3 * best[0])

    def test_bounds_and_velocity_clamp(self):
        low, high = np.array([-1.0, 0.0, -5.0]), np.array([1.0, 0.5, 5.0])
        pso = PSO(low, high, n_particles=10, n_iters=30, seed=3)
        v_max = 0.2 * (high - low)
        self.assertTrue(np.all(np.abs(pso.v) <= 0.1 * (high - low)))
        for _ in range(pso.n_iters):
            x = pso.ask()
            self.assertTrue(np.all((x >= low) & (x <= high)), "position outside the bounds")
            self.assertTrue(np.all(np.abs(pso.v) <= v_max + 1e-12), "velocity above the clamp")
            pso.tell(-np.sum((x - high) ** 2, axis=1))  # optimum on a corner: pushes particles into the walls
        self.assertTrue(np.all((pso.x >= low) & (pso.x <= high)))

    def test_same_seed_same_history(self):
        _, a = minimize(rastrigin, seed=7, n_iters=10)
        _, b = minimize(rastrigin, seed=7, n_iters=10)
        _, c = minimize(rastrigin, seed=8, n_iters=10)
        for key in a.history:
            np.testing.assert_array_equal(np.array(a.history[key]), np.array(b.history[key]))
        self.assertFalse(np.array_equal(a.history["positions"][0], c.history["positions"][0]))

    def test_budget_is_exact(self):
        calls = []

        def f(x):
            calls.append(x)
            return sphere(x)

        _, pso = minimize(f, seed=1, n_particles=8, n_iters=4)
        self.assertEqual(len(calls), 8 * 4)
        self.assertEqual(len(pso.history["fitness"]), 4)
        with self.assertRaises(RuntimeError):
            pso.ask()

    def test_non_finite_fitness(self):
        pso = PSO([0.0, -1.0], [1.0, 1.0], n_particles=4, n_iters=3, seed=0)
        start = pso.ask()
        pso.tell([np.nan, np.nan, np.inf, -np.inf])  # a whole round of failed evaluations
        self.assertEqual(pso.gbest_f, -np.inf)
        np.testing.assert_array_equal(pso.gbest_x, start[0])
        self.assertTrue(np.all(np.isfinite(pso.ask())))
        pso.tell([0.1, np.nan, 0.3, 0.2])
        self.assertEqual(pso.gbest_f, 0.3)
        pso.tell([np.nan] * 4)
        self.assertEqual(pso.gbest_f, 0.3)

    def test_inertia_schedule(self):
        pso = PSO([0.0], [1.0], n_particles=2, n_iters=4, seed=0)
        np.testing.assert_allclose(pso.weights, [0.9, 0.65, 0.4])  # 3 update steps after the initial round


if __name__ == "__main__":
    unittest.main()
