## Final comparison on the test split (g1g2_studio)

1046 images, 7413 boxes. 95% CIs: percentile, 1000 paired bootstrap resamples of the images (seed 0). P and R at Ultralytics' max-F1 confidence. ms/img: median batch-1 latency on Tesla T4. Each model is validated and timed at its own training image size, i.e. as deployed: G1A 416, G1B 416, G1C 416, G2A 640, G2B 640, G2C 640. The val column is each arm's own training validation set, so it can differ between arms. Predictions are restricted to classes [0, 1, 2, 3, 4, 5] (Ultralytics' classes filter in the NMS).

| Arm | val mAP@50 (training) | test mAP@50 [95% CI] | test mAP@50-95 [95% CI] | P | R | ms/img |
|---|---|---|---|---|---|---|
| G1A: Gen 1 Defaults | 0.657 | 0.676 [0.641, 0.719] | 0.474 [0.442, 0.514] | 0.750 | 0.578 | 7.6 |
| G1B: Gen 1 Random search | 0.643 | 0.672 [0.636, 0.716] | 0.475 [0.443, 0.516] | 0.759 | 0.579 | 7.5 |
| G1C: Gen 1 PSO | 0.657 | 0.669 [0.628, 0.716] | 0.471 [0.438, 0.513] | 0.753 | 0.584 | 7.5 |
| G2A: Gen 2 Defaults | 0.454 | 0.571 [0.529, 0.619] | 0.386 [0.356, 0.425] | 0.660 | 0.514 | 8.7 |
| G2B: Gen 2 Random search | 0.473 | 0.595 [0.557, 0.639] | 0.404 [0.374, 0.442] | 0.694 | 0.515 | 8.7 |
| G2C: Gen 2 PSO | 0.466 | 0.590 [0.550, 0.635] | 0.398 [0.367, 0.439] | 0.677 | 0.533 | 8.5 |

### Pairwise differences (paired bootstrap)

| Pair | Δ mAP@50 [95% CI] | P(Δ>0) | Δ mAP@50-95 [95% CI] | P(Δ>0) | Verdict |
|---|---|---|---|---|---|
| Gen 1 Random search − Gen 1 Defaults (G1B−G1A) | -0.003 [-0.015, +0.010] | 0.32 | +0.001 [-0.008, +0.010] | 0.57 | Not significant: both 95% CIs contain 0, so this split cannot tell Gen 1 Random search and Gen 1 Defaults apart. |
| Gen 1 PSO − Gen 1 Defaults (G1C−G1A) | -0.007 [-0.021, +0.007] | 0.17 | -0.003 [-0.013, +0.007] | 0.32 | Not significant: both 95% CIs contain 0, so this split cannot tell Gen 1 PSO and Gen 1 Defaults apart. |
| Gen 2 Defaults − Gen 1 Defaults (G2A−G1A) | -0.105 [-0.125, -0.084] | 0.00 | -0.088 [-0.103, -0.075] | 0.00 | Gen 1 Defaults is significantly better on mAP@50; Gen 1 Defaults is significantly better on mAP@50-95. |
| Gen 2 Random search − Gen 1 Defaults (G2B−G1A) | -0.081 [-0.100, -0.062] | 0.00 | -0.070 [-0.084, -0.056] | 0.00 | Gen 1 Defaults is significantly better on mAP@50; Gen 1 Defaults is significantly better on mAP@50-95. |
| Gen 2 PSO − Gen 1 Defaults (G2C−G1A) | -0.086 [-0.105, -0.066] | 0.00 | -0.075 [-0.090, -0.062] | 0.00 | Gen 1 Defaults is significantly better on mAP@50; Gen 1 Defaults is significantly better on mAP@50-95. |
| Gen 1 PSO − Gen 1 Random search (G1C−G1B) | -0.004 [-0.016, +0.008] | 0.28 | -0.003 [-0.011, +0.004] | 0.22 | Not significant: both 95% CIs contain 0, so this split cannot tell Gen 1 PSO and Gen 1 Random search apart. |
| Gen 2 Defaults − Gen 1 Random search (G2A−G1B) | -0.101 [-0.124, -0.080] | 0.00 | -0.089 [-0.104, -0.076] | 0.00 | Gen 1 Random search is significantly better on mAP@50; Gen 1 Random search is significantly better on mAP@50-95. |
| Gen 2 Random search − Gen 1 Random search (G2B−G1B) | -0.078 [-0.100, -0.056] | 0.00 | -0.070 [-0.084, -0.056] | 0.00 | Gen 1 Random search is significantly better on mAP@50; Gen 1 Random search is significantly better on mAP@50-95. |
| Gen 2 PSO − Gen 1 Random search (G2C−G1B) | -0.083 [-0.104, -0.061] | 0.00 | -0.076 [-0.089, -0.062] | 0.00 | Gen 1 Random search is significantly better on mAP@50; Gen 1 Random search is significantly better on mAP@50-95. |
| Gen 2 Defaults − Gen 1 PSO (G2A−G1C) | -0.098 [-0.120, -0.074] | 0.00 | -0.086 [-0.101, -0.071] | 0.00 | Gen 1 PSO is significantly better on mAP@50; Gen 1 PSO is significantly better on mAP@50-95. |
| Gen 2 Random search − Gen 1 PSO (G2B−G1C) | -0.074 [-0.095, -0.050] | 0.00 | -0.067 [-0.082, -0.052] | 0.00 | Gen 1 PSO is significantly better on mAP@50; Gen 1 PSO is significantly better on mAP@50-95. |
| Gen 2 PSO − Gen 1 PSO (G2C−G1C) | -0.079 [-0.100, -0.057] | 0.00 | -0.073 [-0.088, -0.058] | 0.00 | Gen 1 PSO is significantly better on mAP@50; Gen 1 PSO is significantly better on mAP@50-95. |
| Gen 2 Random search − Gen 2 Defaults (G2B−G2A) | +0.024 [+0.009, +0.038] | 1.00 | +0.019 [+0.008, +0.029] | 1.00 | Gen 2 Random search is significantly better on mAP@50; Gen 2 Random search is significantly better on mAP@50-95. |
| Gen 2 PSO − Gen 2 Defaults (G2C−G2A) | +0.019 [+0.005, +0.033] | 1.00 | +0.013 [+0.004, +0.023] | 1.00 | Gen 2 PSO is significantly better on mAP@50; Gen 2 PSO is significantly better on mAP@50-95. |
| Gen 2 PSO − Gen 2 Random search (G2C−G2B) | -0.005 [-0.018, +0.009] | 0.25 | -0.006 [-0.015, +0.003] | 0.12 | Not significant: both 95% CIs contain 0, so this split cannot tell Gen 2 PSO and Gen 2 Random search apart. |

Δ is the difference of the two arms' mAPs on the whole split (as in the main table); the 95% CI and P(Δ>0) come from the paired bootstrap (P(Δ>0): share of resamples in which the first arm of the pair scored higher). A difference is significant only if its CI excludes 0.

### Per-class AP@50 (test split)

| Class | G1A: Gen 1 Defaults | G1B: Gen 1 Random search | G1C: Gen 1 PSO | G2A: Gen 2 Defaults | G2B: Gen 2 Random search | G2C: Gen 2 PSO |
|---|---|---|---|---|---|---|
| BIODEGRADABLE | 0.632 | 0.631 | 0.645 | 0.614 | 0.628 | 0.619 |
| CARDBOARD | 0.641 | 0.631 | 0.625 | 0.550 | 0.570 | 0.566 |
| GLASS | 0.747 | 0.756 | 0.763 | 0.631 | 0.658 | 0.668 |
| METAL | 0.751 | 0.735 | 0.754 | 0.651 | 0.663 | 0.681 |
| PAPER | 0.583 | 0.591 | 0.592 | 0.442 | 0.462 | 0.466 |
| PLASTIC | 0.701 | 0.690 | 0.634 | 0.540 | 0.588 | 0.539 |
