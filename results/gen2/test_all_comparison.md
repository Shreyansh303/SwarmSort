## Final comparison on the test split (all)

1514 images, 8192 boxes. 95% CIs: percentile, 1000 paired bootstrap resamples of the images (seed 0). P and R at Ultralytics' max-F1 confidence. ms/img: median batch-1 latency at imgsz 640 on Tesla T4.

| Arm | val mAP@50 (training) | test mAP@50 [95% CI] | test mAP@50-95 [95% CI] | P | R | ms/img |
|---|---|---|---|---|---|---|
| A: Defaults | 0.454 | 0.479 [0.453, 0.513] | 0.329 [0.309, 0.358] | 0.549 | 0.445 | 8.1 |
| B: Random search | 0.473 | 0.497 [0.471, 0.531] | 0.342 [0.322, 0.372] | 0.563 | 0.470 | 8.7 |
| C: PSO | 0.466 | 0.500 [0.474, 0.534] | 0.343 [0.323, 0.374] | 0.582 | 0.470 | 8.1 |

### Pairwise differences (paired bootstrap)

| Pair | Δ mAP@50 [95% CI] | P(Δ>0) | Δ mAP@50-95 [95% CI] | P(Δ>0) | Verdict |
|---|---|---|---|---|---|
| Random search − Defaults (B−A) | +0.017 [+0.006, +0.030] | 1.00 | +0.013 [+0.004, +0.022] | 1.00 | Random search is significantly better on mAP@50; Random search is significantly better on mAP@50-95. |
| PSO − Defaults (C−A) | +0.020 [+0.011, +0.032] | 1.00 | +0.014 [+0.006, +0.022] | 1.00 | PSO is significantly better on mAP@50; PSO is significantly better on mAP@50-95. |
| PSO − Random search (C−B) | +0.003 [-0.008, +0.015] | 0.70 | +0.001 [-0.007, +0.009] | 0.61 | Not significant: both 95% CIs contain 0, so this split cannot tell PSO and Random search apart. |

Δ is the difference of the two arms' mAPs on the whole split (as in the main table); the 95% CI and P(Δ>0) come from the paired bootstrap (P(Δ>0): share of resamples in which the first arm of the pair scored higher). A difference is significant only if its CI excludes 0.

### Per-class AP@50 (test split)

| Class | A: Defaults | B: Random search | C: PSO |
|---|---|---|---|
| BIODEGRADABLE | 0.612 | 0.628 | 0.619 |
| CARDBOARD | 0.540 | 0.568 | 0.572 |
| GLASS | 0.605 | 0.631 | 0.638 |
| METAL | 0.614 | 0.625 | 0.644 |
| PAPER | 0.425 | 0.447 | 0.457 |
| PLASTIC | 0.510 | 0.527 | 0.505 |
| OTHER | 0.050 | 0.051 | 0.063 |
