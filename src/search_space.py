"""The 6-D hyperparameter search space shared by random search and PSO.

A point in the space is a numpy vector in "search units": log-scaled dimensions hold the base-10 exponent
(a particle at -3.2 means lr0 = 10**-3.2), linear dimensions hold the value itself. decode() turns a vector
into keyword arguments for Ultralytics' model.train(). No Ultralytics import, so PSO code can reuse it freely.
"""
import numpy as np
from scipy.stats import qmc

# name: (low, high, log10-scaled?)
SPACE = {
    "lr0": (1e-4, 1e-2, True),
    "lrf": (0.01, 0.2, False),
    "momentum": (0.60, 0.98, False),
    "weight_decay": (1e-5, 1e-3, True),
    "mosaic": (0.0, 1.0, False),
    "scale": (0.2, 0.8, False),
}
NAMES = list(SPACE)
IS_LOG = np.array([log for _, _, log in SPACE.values()])
LOW = np.array([np.log10(lo) if log else lo for lo, _, log in SPACE.values()])
HIGH = np.array([np.log10(hi) if log else hi for _, hi, log in SPACE.values()])

# Ultralytics 8.4 defaults for the searched hyperparameters (Arm A)
DEFAULTS = {"lr0": 0.01, "lrf": 0.01, "momentum": 0.937, "weight_decay": 0.0005, "mosaic": 1.0, "scale": 0.5}


def decode(vector):
    """Search-unit vector -> {hyperparameter: value}. Out-of-bounds values are clipped to the bounds."""
    v = np.clip(np.asarray(vector, dtype=float), LOW, HIGH)
    values = np.where(IS_LOG, 10.0 ** v, v)
    # 6 significant digits keep the logs readable and make 10**log10(x) round-trip exactly
    return {name: float(f"{x:.6g}") for name, x in zip(NAMES, values)}


def encode(params):
    """{hyperparameter: value} -> search-unit vector (inverse of decode)."""
    values = np.array([params[name] for name in NAMES], dtype=float)
    return np.where(IS_LOG, np.log10(values), values)


def sample(n, seed):
    """n points drawn uniformly inside the bounds (in search units), reproducible for a given seed."""
    return np.random.default_rng(seed).uniform(LOW, HIGH, size=(n, len(NAMES)))


def latin_hypercube(n, seed):
    """n points by Latin-hypercube sampling: each dimension's range is cut into n equal strata and every stratum
    holds exactly one point, so even a handful of points spans every hyperparameter's whole range."""
    # random-cd reorders points within their strata to lower discrepancy, which removes chance correlations
    # between dimensions (without it, seed 42 gave lr0 and mosaic a rank correlation of 0.79)
    lhs = qmc.LatinHypercube(d=len(NAMES), optimization="random-cd", seed=seed)
    return LOW + lhs.random(n) * (HIGH - LOW)
