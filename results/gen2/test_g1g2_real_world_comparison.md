## Final comparison on the test split (g1g2_real_world)

468 images, 647 boxes. 95% CIs: percentile, 1000 paired bootstrap resamples of the images (seed 0). P and R at Ultralytics' max-F1 confidence. ms/img: median batch-1 latency on Tesla T4. Each model is validated and timed at its own training image size, i.e. as deployed: G1A 416, G1B 416, G1C 416, G2A 640, G2B 640, G2C 640. The val column is each arm's own training validation set, so it can differ between arms. Predictions are restricted to classes [0, 1, 2, 3, 4, 5] (Ultralytics' classes filter in the NMS).

| Arm | val mAP@50 (training) | test mAP@50 [95% CI] | test mAP@50-95 [95% CI] | P | R | ms/img |
|---|---|---|---|---|---|---|
| G1A: Gen 1 Defaults | 0.657 | 0.047 [0.038, 0.080] | 0.036 [0.029, 0.063] | 0.071 | 0.291 | 8.8 |
| G1B: Gen 1 Random search | 0.643 | 0.038 [0.031, 0.069] | 0.029 [0.024, 0.053] | 0.104 | 0.229 | 8.5 |
| G1C: Gen 1 PSO | 0.657 | 0.096 [0.041, 0.214] | 0.069 [0.031, 0.153] | 0.105 | 0.256 | 8.4 |
| G2A: Gen 2 Defaults | 0.454 | 0.353 [0.315, 0.504] | 0.290 [0.258, 0.428] | 0.508 | 0.452 | 8.6 |
| G2B: Gen 2 Random search | 0.473 | 0.338 [0.304, 0.473] | 0.272 [0.243, 0.389] | 0.412 | 0.370 | 8.2 |
| G2C: Gen 2 PSO | 0.466 | 0.352 [0.315, 0.485] | 0.282 [0.251, 0.392] | 0.623 | 0.374 | 8.2 |

### Pairwise differences (paired bootstrap)

| Pair | Δ mAP@50 [95% CI] | P(Δ>0) | Δ mAP@50-95 [95% CI] | P(Δ>0) | Verdict |
|---|---|---|---|---|---|
| Gen 1 Random search − Gen 1 Defaults (G1B−G1A) | -0.010 [-0.025, +0.005] | 0.11 | -0.007 [-0.021, +0.004] | 0.12 | Not significant: both 95% CIs contain 0, so this split cannot tell Gen 1 Random search and Gen 1 Defaults apart. |
| Gen 1 PSO − Gen 1 Defaults (G1C−G1A) | +0.048 [-0.020, +0.163] | 0.71 | +0.033 [-0.017, +0.114] | 0.69 | Not significant: both 95% CIs contain 0, so this split cannot tell Gen 1 PSO and Gen 1 Defaults apart. |
| Gen 2 Defaults − Gen 1 Defaults (G2A−G1A) | +0.305 [+0.266, +0.453] | 1.00 | +0.254 [+0.220, +0.385] | 1.00 | Gen 2 Defaults is significantly better on mAP@50; Gen 2 Defaults is significantly better on mAP@50-95. |
| Gen 2 Random search − Gen 1 Defaults (G2B−G1A) | +0.291 [+0.257, +0.407] | 1.00 | +0.236 [+0.207, +0.338] | 1.00 | Gen 2 Random search is significantly better on mAP@50; Gen 2 Random search is significantly better on mAP@50-95. |
| Gen 2 PSO − Gen 1 Defaults (G2C−G1A) | +0.305 [+0.267, +0.419] | 1.00 | +0.246 [+0.216, +0.342] | 1.00 | Gen 2 PSO is significantly better on mAP@50; Gen 2 PSO is significantly better on mAP@50-95. |
| Gen 1 PSO − Gen 1 Random search (G1C−G1B) | +0.058 [-0.005, +0.171] | 0.92 | +0.039 [-0.005, +0.119] | 0.89 | Not significant: both 95% CIs contain 0, so this split cannot tell Gen 1 PSO and Gen 1 Random search apart. |
| Gen 2 Defaults − Gen 1 Random search (G2A−G1B) | +0.315 [+0.277, +0.461] | 1.00 | +0.261 [+0.227, +0.391] | 1.00 | Gen 2 Defaults is significantly better on mAP@50; Gen 2 Defaults is significantly better on mAP@50-95. |
| Gen 2 Random search − Gen 1 Random search (G2B−G1B) | +0.301 [+0.265, +0.419] | 1.00 | +0.243 [+0.213, +0.348] | 1.00 | Gen 2 Random search is significantly better on mAP@50; Gen 2 Random search is significantly better on mAP@50-95. |
| Gen 2 PSO − Gen 1 Random search (G2C−G1B) | +0.314 [+0.276, +0.431] | 1.00 | +0.253 [+0.223, +0.352] | 1.00 | Gen 2 PSO is significantly better on mAP@50; Gen 2 PSO is significantly better on mAP@50-95. |
| Gen 2 Defaults − Gen 1 PSO (G2A−G1C) | +0.257 [+0.160, +0.397] | 1.00 | +0.221 [+0.156, +0.343] | 1.00 | Gen 2 Defaults is significantly better on mAP@50; Gen 2 Defaults is significantly better on mAP@50-95. |
| Gen 2 Random search − Gen 1 PSO (G2B−G1C) | +0.243 [+0.134, +0.418] | 1.00 | +0.203 [+0.125, +0.346] | 1.00 | Gen 2 Random search is significantly better on mAP@50; Gen 2 Random search is significantly better on mAP@50-95. |
| Gen 2 PSO − Gen 1 PSO (G2C−G1C) | +0.256 [+0.142, +0.424] | 1.00 | +0.213 [+0.131, +0.350] | 1.00 | Gen 2 PSO is significantly better on mAP@50; Gen 2 PSO is significantly better on mAP@50-95. |
| Gen 2 Random search − Gen 2 Defaults (G2B−G2A) | -0.014 [-0.137, +0.063] | 0.48 | -0.018 [-0.130, +0.049] | 0.42 | Not significant: both 95% CIs contain 0, so this split cannot tell Gen 2 Random search and Gen 2 Defaults apart. |
| Gen 2 PSO − Gen 2 Defaults (G2C−G2A) | -0.001 [-0.131, +0.075] | 0.56 | -0.008 [-0.125, +0.058] | 0.50 | Not significant: both 95% CIs contain 0, so this split cannot tell Gen 2 PSO and Gen 2 Defaults apart. |
| Gen 2 PSO − Gen 2 Random search (G2C−G2B) | +0.013 [-0.019, +0.042] | 0.73 | +0.010 [-0.019, +0.035] | 0.70 | Not significant: both 95% CIs contain 0, so this split cannot tell Gen 2 PSO and Gen 2 Random search apart. |

Δ is the difference of the two arms' mAPs on the whole split (as in the main table); the 95% CI and P(Δ>0) come from the paired bootstrap (P(Δ>0): share of resamples in which the first arm of the pair scored higher). A difference is significant only if its CI excludes 0.

### Per-class AP@50 (test split)

| Class | G1A: Gen 1 Defaults | G1B: Gen 1 Random search | G1C: Gen 1 PSO | G2A: Gen 2 Defaults | G2B: Gen 2 Random search | G2C: Gen 2 PSO |
|---|---|---|---|---|---|---|
| BIODEGRADABLE | 0.003 | 0.009 | 0.332 | 0.249 | 0.020 | 0.003 |
| CARDBOARD | 0.020 | 0.045 | 0.056 | 0.503 | 0.563 | 0.618 |
| GLASS | 0.033 | 0.028 | 0.010 | 0.251 | 0.296 | 0.247 |
| METAL | 0.081 | 0.052 | 0.062 | 0.405 | 0.411 | 0.450 |
| PAPER | 0.029 | 0.008 | 0.012 | 0.239 | 0.274 | 0.330 |
| PLASTIC | 0.117 | 0.084 | 0.103 | 0.469 | 0.466 | 0.464 |
