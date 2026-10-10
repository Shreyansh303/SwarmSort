## Final comparison on the test split (studio)

1046 images, 7413 boxes. 95% CIs: percentile, 1000 paired bootstrap resamples of the images (seed 0). P and R at Ultralytics' max-F1 confidence. ms/img: median batch-1 latency at imgsz 640 on Tesla T4.

| Arm | val mAP@50 (training) | test mAP@50 [95% CI] | test mAP@50-95 [95% CI] | P | R | ms/img |
|---|---|---|---|---|---|---|
| A: Defaults | 0.454 | 0.571 [0.529, 0.619] | 0.386 [0.356, 0.425] | 0.660 | 0.514 | 8.4 |
| B: Random search | 0.473 | 0.595 [0.557, 0.639] | 0.404 [0.374, 0.442] | 0.694 | 0.515 | 8.5 |
| C: PSO | 0.466 | 0.590 [0.550, 0.635] | 0.398 [0.367, 0.439] | 0.677 | 0.533 | 8.2 |

### Pairwise differences (paired bootstrap)

| Pair | Δ mAP@50 [95% CI] | P(Δ>0) | Δ mAP@50-95 [95% CI] | P(Δ>0) | Verdict |
|---|---|---|---|---|---|
| Random search − Defaults (B−A) | +0.024 [+0.009, +0.038] | 1.00 | +0.019 [+0.008, +0.029] | 1.00 | Random search is significantly better on mAP@50; Random search is significantly better on mAP@50-95. |
| PSO − Defaults (C−A) | +0.019 [+0.005, +0.033] | 1.00 | +0.013 [+0.004, +0.023] | 1.00 | PSO is significantly better on mAP@50; PSO is significantly better on mAP@50-95. |
| PSO − Random search (C−B) | -0.005 [-0.018, +0.009] | 0.24 | -0.006 [-0.015, +0.003] | 0.12 | Not significant: both 95% CIs contain 0, so this split cannot tell PSO and Random search apart. |

Δ is the difference of the two arms' mAPs on the whole split (as in the main table); the 95% CI and P(Δ>0) come from the paired bootstrap (P(Δ>0): share of resamples in which the first arm of the pair scored higher). A difference is significant only if its CI excludes 0.

### Per-class AP@50 (test split)

| Class | A: Defaults | B: Random search | C: PSO |
|---|---|---|---|
| BIODEGRADABLE | 0.614 | 0.628 | 0.619 |
| CARDBOARD | 0.550 | 0.570 | 0.566 |
| GLASS | 0.630 | 0.658 | 0.668 |
| METAL | 0.651 | 0.663 | 0.681 |
| PAPER | 0.442 | 0.462 | 0.466 |
| PLASTIC | 0.540 | 0.588 | 0.539 |
| OTHER | - | - | - |
