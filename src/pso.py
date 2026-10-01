"""Particle Swarm Optimization (standard inertia-weight PSO), written out by hand. Maximizes the fitness.

Every particle i has a position x_i (one candidate solution), a velocity v_i and a memory of the best position it
has visited (pbest_i); the swarm remembers the best position any particle has visited (gbest). One update step:
    v <- w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)     (inertia + pull to own best + pull to swarm best)
    x <- x + v
r1, r2 ~ U(0, 1) are drawn fresh for every particle and dimension. The inertia weight w falls linearly from 0.9 to
0.4 over the update steps (explore first, then refine). Velocities are clamped to +-20% of each dimension's range;
positions are clamped to the bounds, and a clamped dimension's velocity is set to zero (the particle stops at the
wall instead of pushing further out).

Usage (ask/tell): the caller evaluates the positions, the optimizer only does the bookkeeping.
    pso = PSO(low, high, n_particles=8, n_iters=4, seed=42)
    for _ in range(pso.n_iters):
        positions = pso.ask()
        pso.tell([fitness(x) for x in positions])
n_iters counts rounds of evaluations: round 1 is the random initial swarm, followed by n_iters - 1 update steps,
so the budget is exactly n_particles x n_iters evaluations.
"""
import numpy as np


class PSO:
    def __init__(self, low, high, n_particles, n_iters, seed,
                 w_start=0.9, w_end=0.4, c1=1.5, c2=1.5, v_max_frac=0.2, v_init_frac=0.1):
        self.low, self.high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
        span = self.high - self.low
        self.n_iters, self.c1, self.c2 = n_iters, c1, c2
        self.v_max = v_max_frac * span                  # velocity limit per dimension
        self.rng = np.random.default_rng(seed)          # the only source of randomness -> reproducible runs
        shape = (n_particles, len(self.low))
        self.x = self.rng.uniform(self.low, self.high, size=shape)
        self.v = self.rng.uniform(-v_init_frac * span, v_init_frac * span, size=shape)
        self.pbest_x, self.pbest_f = self.x.copy(), np.full(n_particles, -np.inf)
        self.gbest_x, self.gbest_f = self.x[0].copy(), -np.inf  # placeholder until the first finite fitness
        self.weights = np.linspace(w_start, w_end, max(n_iters - 1, 1))  # w for update step 1, 2, ...
        self.round = 0                                  # rounds evaluated so far
        self.history = {"positions": [], "fitness": [], "gbest_f": [], "gbest_x": []}

    def ask(self):
        """Positions to evaluate in this round (one row per particle)."""
        if self.round >= self.n_iters:
            raise RuntimeError("PSO budget used up: all n_iters rounds have been evaluated")
        return self.x.copy()

    def tell(self, fitness):
        """Take this round's fitness values, update pbest/gbest, then move the swarm (except after the last round)."""
        f = np.asarray(fitness, dtype=float)
        if f.shape != (len(self.x),):
            raise ValueError(f"expected {len(self.x)} fitness values, got shape {f.shape}")
        f = np.where(np.isfinite(f), f, -np.inf)        # a failed evaluation (NaN/inf) counts as the worst
        better = f > self.pbest_f                       # particles that beat their own best
        self.pbest_x[better], self.pbest_f[better] = self.x[better], f[better]
        i = int(np.argmax(self.pbest_f))
        if self.pbest_f[i] > self.gbest_f:              # new swarm best
            self.gbest_x, self.gbest_f = self.pbest_x[i].copy(), float(self.pbest_f[i])
        for key, value in (("positions", self.x.copy()), ("fitness", f), ("gbest_f", self.gbest_f),
                           ("gbest_x", self.gbest_x.copy())):
            self.history[key].append(value)
        self.round += 1
        if self.round < self.n_iters:                   # no move after the final round: it would never be evaluated
            self._move(self.weights[self.round - 1])

    def _move(self, w):
        r1, r2 = self.rng.random(self.x.shape), self.rng.random(self.x.shape)
        self.v = w * self.v + self.c1 * r1 * (self.pbest_x - self.x) + self.c2 * r2 * (self.gbest_x - self.x)
        self.v = np.clip(self.v, -self.v_max, self.v_max)
        self.x = self.x + self.v
        outside = (self.x < self.low) | (self.x > self.high)
        self.x = np.clip(self.x, self.low, self.high)
        self.v[outside] = 0.0
