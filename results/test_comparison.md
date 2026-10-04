## Final comparison on the test split

1046 images, 7415 boxes. 95% CIs: percentile, 1000 paired bootstrap resamples of the images (seed 0). P and R at Ultralytics' max-F1 confidence. ms/img: median batch-1 latency at imgsz 416 on CPU Intel Core Ultra 5 125H.

| Arm | val mAP@50 (training) | test mAP@50 [95% CI] | test mAP@50-95 [95% CI] | P | R | ms/img |
|---|---|---|---|---|---|---|
| A: Defaults | 0.657 | 0.676 [0.639, 0.722] | 0.474 [0.441, 0.518] | 0.747 | 0.581 | 31.4 |
| B: Random search | 0.643 | 0.672 [0.634, 0.717] | 0.475 [0.441, 0.519] | 0.746 | 0.587 | 38.3 |
| C: PSO | 0.657 | 0.668 [0.627, 0.717] | 0.471 [0.437, 0.516] | 0.747 | 0.586 | 31.7 |

### Pairwise differences (paired bootstrap)

| Pair | Δ mAP@50 [95% CI] | P(Δ>0) | Δ mAP@50-95 [95% CI] | P(Δ>0) | Verdict |
|---|---|---|---|---|---|
| Random search − Defaults (B−A) | -0.004 [-0.016, +0.009] | 0.30 | +0.001 [-0.008, +0.010] | 0.59 | Not significant: both 95% CIs contain 0, so this split cannot tell Random search and Defaults apart. |
| PSO − Defaults (C−A) | -0.008 [-0.021, +0.007] | 0.16 | -0.003 [-0.012, +0.007] | 0.34 | Not significant: both 95% CIs contain 0, so this split cannot tell PSO and Defaults apart. |
| PSO − Random search (C−B) | -0.004 [-0.015, +0.007] | 0.24 | -0.003 [-0.011, +0.004] | 0.19 | Not significant: both 95% CIs contain 0, so this split cannot tell PSO and Random search apart. |

Δ is the difference of the two arms' mAPs on the whole split (as in the main table); the 95% CI and P(Δ>0) come from the paired bootstrap (P(Δ>0): share of resamples in which the first arm of the pair scored higher). A difference is significant only if its CI excludes 0.

### Per-class AP@50 (test split)

| Class | A: Defaults | B: Random search | C: PSO |
|---|---|---|---|
| BIODEGRADABLE | 0.632 | 0.631 | 0.643 |
| CARDBOARD | 0.641 | 0.628 | 0.623 |
| GLASS | 0.747 | 0.757 | 0.765 |
| METAL | 0.753 | 0.736 | 0.754 |
| PAPER | 0.583 | 0.590 | 0.588 |
| PLASTIC | 0.699 | 0.693 | 0.638 |
