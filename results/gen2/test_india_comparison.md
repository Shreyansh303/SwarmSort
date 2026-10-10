## Final comparison on the test split (india)

700 images, 4502 boxes. 95% CIs: percentile, 1000 paired bootstrap resamples of the images (seed 0). P and R at Ultralytics' max-F1 confidence. ms/img: median batch-1 latency at imgsz 640 on Tesla T4.

| Arm | val mAP@50 (training) | test mAP@50 [95% CI] | test mAP@50-95 [95% CI] | P | R | ms/img |
|---|---|---|---|---|---|---|
| A: Defaults | 0.454 | 0.027 [0.024, 0.031] | 0.015 [0.013, 0.018] | 0.075 | 0.046 | 8.0 |
| B: Random search | 0.473 | 0.023 [0.021, 0.027] | 0.013 [0.012, 0.015] | 0.070 | 0.043 | 8.0 |
| C: PSO | 0.466 | 0.027 [0.024, 0.031] | 0.015 [0.013, 0.017] | 0.073 | 0.047 | 8.6 |

### Pairwise differences (paired bootstrap)

| Pair | Δ mAP@50 [95% CI] | P(Δ>0) | Δ mAP@50-95 [95% CI] | P(Δ>0) | Verdict |
|---|---|---|---|---|---|
| Random search − Defaults (B−A) | -0.003 [-0.006, -0.001] | 0.00 | -0.002 [-0.003, -0.001] | 0.00 | Defaults is significantly better on mAP@50; Defaults is significantly better on mAP@50-95. |
| PSO − Defaults (C−A) | +0.000 [-0.003, +0.003] | 0.47 | -0.000 [-0.002, +0.001] | 0.33 | Not significant: both 95% CIs contain 0, so this split cannot tell PSO and Defaults apart. |
| PSO − Random search (C−B) | +0.003 [+0.001, +0.006] | 1.00 | +0.002 [+0.000, +0.003] | 1.00 | PSO is significantly better on mAP@50; PSO is significantly better on mAP@50-95. |

Δ is the difference of the two arms' mAPs on the whole split (as in the main table); the 95% CI and P(Δ>0) come from the paired bootstrap (P(Δ>0): share of resamples in which the first arm of the pair scored higher). A difference is significant only if its CI excludes 0.

### Per-class AP@50 (test split)

| Class | A: Defaults | B: Random search | C: PSO |
|---|---|---|---|
| BIODEGRADABLE | - | - | - |
| CARDBOARD | 0.001 | 0.001 | 0.001 |
| GLASS | - | - | - |
| METAL | 0.000 | 0.000 | 0.000 |
| PAPER | 0.035 | 0.028 | 0.031 |
| PLASTIC | 0.097 | 0.089 | 0.101 |
| OTHER | 0.000 | 0.000 | 0.002 |
