## Test mAP@50 [95% CI] per domain

| Arm | all (1514 images) | studio (1046 images) | real_world (468 images) | india (700 images) |
|---|---|---|---|---|
| A: Defaults | 0.479 [0.453, 0.513] | 0.571 [0.529, 0.619] | 0.310 [0.278, 0.440] | 0.027 [0.024, 0.031] |
| B: Random search | 0.497 [0.471, 0.531] | 0.595 [0.557, 0.639] | 0.298 [0.271, 0.407] | 0.023 [0.021, 0.027] |
| C: PSO | 0.500 [0.474, 0.534] | 0.590 [0.550, 0.635] | 0.310 [0.279, 0.415] | 0.027 [0.024, 0.031] |

Pairwise verdicts (paired bootstrap, 95%):
- all, B vs A: Random search is significantly better on mAP@50; Random search is significantly better on mAP@50-95.
- all, C vs A: PSO is significantly better on mAP@50; PSO is significantly better on mAP@50-95.
- all, C vs B: Not significant: both 95% CIs contain 0, so this split cannot tell PSO and Random search apart.
- studio, B vs A: Random search is significantly better on mAP@50; Random search is significantly better on mAP@50-95.
- studio, C vs A: PSO is significantly better on mAP@50; PSO is significantly better on mAP@50-95.
- studio, C vs B: Not significant: both 95% CIs contain 0, so this split cannot tell PSO and Random search apart.
- real_world, B vs A: Not significant: both 95% CIs contain 0, so this split cannot tell Random search and Defaults apart.
- real_world, C vs A: Not significant: both 95% CIs contain 0, so this split cannot tell PSO and Defaults apart.
- real_world, C vs B: Not significant: both 95% CIs contain 0, so this split cannot tell PSO and Random search apart.
- india, B vs A: Defaults is significantly better on mAP@50; Defaults is significantly better on mAP@50-95.
- india, C vs A: Not significant: both 95% CIs contain 0, so this split cannot tell PSO and Defaults apart.
- india, C vs B: PSO is significantly better on mAP@50; PSO is significantly better on mAP@50-95.

india: DWSD-only test set built in g2_6 (build_dataset.py --only dwsd), 700 images; MD5 cross-check against the 9647 main-build images: 0 identical files
