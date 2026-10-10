## Final comparison on the test split (real_world)

468 images, 779 boxes. 95% CIs: percentile, 1000 paired bootstrap resamples of the images (seed 0). P and R at Ultralytics' max-F1 confidence. ms/img: median batch-1 latency at imgsz 640 on Tesla T4.

| Arm | val mAP@50 (training) | test mAP@50 [95% CI] | test mAP@50-95 [95% CI] | P | R | ms/img |
|---|---|---|---|---|---|---|
| A: Defaults | 0.454 | 0.310 [0.278, 0.440] | 0.254 [0.224, 0.371] | 0.459 | 0.390 | 8.2 |
| B: Random search | 0.473 | 0.298 [0.271, 0.407] | 0.238 [0.213, 0.330] | 0.366 | 0.330 | 8.8 |
| C: PSO | 0.466 | 0.310 [0.279, 0.415] | 0.248 [0.220, 0.336] | 0.556 | 0.334 | 8.0 |

### Pairwise differences (paired bootstrap)

| Pair | Δ mAP@50 [95% CI] | P(Δ>0) | Δ mAP@50-95 [95% CI] | P(Δ>0) | Verdict |
|---|---|---|---|---|---|
| Random search − Defaults (B−A) | -0.012 [-0.118, +0.053] | 0.48 | -0.016 [-0.111, +0.040] | 0.42 | Not significant: both 95% CIs contain 0, so this split cannot tell Random search and Defaults apart. |
| PSO − Defaults (C−A) | +0.001 [-0.111, +0.065] | 0.59 | -0.006 [-0.107, +0.049] | 0.51 | Not significant: both 95% CIs contain 0, so this split cannot tell PSO and Defaults apart. |
| PSO − Random search (C−B) | +0.013 [-0.015, +0.036] | 0.76 | +0.010 [-0.015, +0.031] | 0.74 | Not significant: both 95% CIs contain 0, so this split cannot tell PSO and Random search apart. |

Δ is the difference of the two arms' mAPs on the whole split (as in the main table); the 95% CI and P(Δ>0) come from the paired bootstrap (P(Δ>0): share of resamples in which the first arm of the pair scored higher). A difference is significant only if its CI excludes 0.

### Per-class AP@50 (test split)

| Class | A: Defaults | B: Random search | C: PSO |
|---|---|---|---|
| BIODEGRADABLE | 0.249 | 0.020 | 0.003 |
| CARDBOARD | 0.503 | 0.563 | 0.618 |
| GLASS | 0.251 | 0.296 | 0.245 |
| METAL | 0.405 | 0.411 | 0.451 |
| PAPER | 0.239 | 0.274 | 0.330 |
| PLASTIC | 0.470 | 0.466 | 0.464 |
| OTHER | 0.051 | 0.053 | 0.062 |
