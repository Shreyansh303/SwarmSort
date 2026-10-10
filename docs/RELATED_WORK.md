# Related work

Papers collected for the SwarmSort research paper, which will compare prior work with this project and show which gaps it fills. The status column says whether a paper's content has been checked against its full text. Until a paper is verified, only cite its title, authors and venue, not specific claims.

Status: **verified** = full text read and the notes checked · **unverified** = notes come from the title, venue or common knowledge only.

## Papers found by the student (2026-10-10)

| # | Paper | Link | Task | Optimiser / method | Relevance to SwarmSort | Status |
|---|---|---|---|---|---|---|
| 1 | Majchrowska, S. et al. (2022). *Deep learning-based waste detection in natural and urban environments.* Waste Management, 138, 274–284. | https://www.sciencedirect.com/science/article/pii/S0956053X21006474 | Waste **detection** in natural and urban scenes | Two-stage: EfficientDet-D2 detector + EfficientNet-B2 classifier; hyperparameters chosen by hand | **Base paper.** SwarmSort uses a single-stage detector, a PSO hyperparameter search, an equal-budget comparison with random search and defaults, and real-world test sets | partly verified (read for the Gen 1 report) |
| 2 | Aguerchi, K., Jabrane, Y., Habba, M., & El Hassani, A. H. (2024). *A CNN Hyperparameters Optimization Based on Particle Swarm Optimization for Mammography Breast Cancer Classification.* Journal of Imaging, 10(2), 30. | https://doi.org/10.3390/jimaging10020030 | Medical image **classification** | PSO tunes CNN hyperparameters | **Method precedent:** PSO-based CNN tuning, in another domain. SwarmSort applies it to waste *detection* and tests it against random search at the same budget | unverified |
| 3 | Kaya, V. (2023). *Classification of waste materials with a smart garbage system for sustainable development: a novel model.* Frontiers in Environmental Science, 11. | https://doi.org/10.3389/fenvs.2023.1228732 | Waste **classification** (smart bin) | Deep learning model | Same goal (automated sorting), framed as single-image classification rather than multi-item detection | unverified |
| 4 | Sayed, G. I., Abd Elfattah, M., Darwish, A., & Hassanien, A. E. (2024). *Intelligent and sustainable waste classification model based on multi-objective beluga whale optimization and deep learning.* Environmental Science and Pollution Research, 31, 31492–31510. | https://doi.org/10.1007/s11356-024-33233-w | Waste **classification** | Metaheuristic (multi-objective beluga whale optimization) + deep learning | **Closest competitor:** a swarm metaheuristic for waste AI, but for classification. It allows a "why PSO" comparison and the detection-vs-classification gap | unverified |
| 5 | *Smart Waste Classification System using Deep Learning for Automated Recycling.* IEEE (document 11046029); authors and year to be filled in. | https://ieeexplore.ieee.org/document/11046029 | Waste **classification** | Deep learning | A recent classification baseline for automated recycling | unverified |
| 6 | Thung, G., & Yang, M. (2016). *Classification of Trash for Recyclability Status.* Stanford CS229 project. | https://cs229.stanford.edu/proj2016/poster/ThungYang-ClassificationOfTrashForRecyclabilityStatus-poster.pdf | Waste **classification** (TrashNet: about 2,500 images, 6 classes: glass, paper, cardboard, plastic, metal, trash) | SVM vs CNN | The classic starting point of waste-image AI, and the original idea for this project | partly verified (widely known dataset) |

## Background references already cited in the Gen 1 report

| Paper | Used for |
|---|---|
| Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. *Proc. ICNN.* | Original PSO |
| Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer. | Inertia weight (w decays from 0.9 to 0.4) |
| Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *JMLR*, 13. | Random search as a strong baseline (Arm B) |
| McKay, M. D., Beckman, R. J., & Conover, W. J. (1979). *Technometrics.* | Latin hypercube sampling (proxy check) |
| Redmon, J. et al. (2016). You Only Look Once. *CVPR.* | Single-stage detection |
| Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8 (software). | Detector |
| Lin, T.-Y. et al. (2014). Microsoft COCO. *ECCV.* | mAP metric, pretrained weights |
| Spearman, C. (1904). | Rank correlation (proxy validity, ρ = 0.976) |
| Efron, B., & Tibshirani, R. (1993). *An Introduction to the Bootstrap.* | Paired-bootstrap confidence intervals |
| Ministry of Environment, Forest and Climate Change, India (2016). Solid Waste Management Rules. | Green/blue bin rules in the app |

## Generation 2 data and future-work references (not yet in the report)

| Paper | Used for | Status |
|---|---|---|
| Proença, P. F., & Simões, P. (2020). TACO: Trash Annotations in Context. arXiv:2003.06975. | Gen 2 real-world training data | unverified |
| Ali et al. (2025). DWSD, Dense Waste Segmentation Dataset. *Data in Brief* (PMC11848783). | Gen 2 India test set; its published class table doesn't match the masks (see EVOLUTION.md) | partly verified |
| Majchrowska et al. technical report, arXiv:2105.06808 ("Waste detection in Pomerania"). | Base-paper details, detect-waste dataset list | unverified |
| Clerc, M., & Kennedy, J. (2002). The particle swarm: explosion, stability, and convergence. | PSO constriction factor (future work) | unverified |
| Li, L. et al. (2018). Hyperband. *JMLR.* | Multi-fidelity search (future work) | unverified |
| Akyon, F. C. et al. (2022). Slicing Aided Hyper Inference (SAHI). | Small-object sliced inference (future work) | unverified |

## Gaps SwarmSort addresses (draft, to refine once papers are verified)

1. **Detection, not classification.** Papers 3–6 label one class per image. SwarmSort boxes and classifies every item in a photo, which is what real sorting needs.
2. **Metaheuristic tuning for waste detection.** Paper 2 tunes CNNs with PSO in medical imaging, and paper 4 uses a swarm metaheuristic for waste classification. SwarmSort brings PSO tuning to waste object detection.
3. **A fair, tested comparison.** SwarmSort compares PSO with random search and the defaults at an *equal budget*, on a test split used once, with paired-bootstrap confidence intervals. Check whether papers 2 and 4 include a random-search baseline or a significance test.
4. **Real-world scenes.** Gen 2 trains on and evaluates real street, park and beach photos (TACO, HITL), plus a separate India test set (DWSD), and reports studio vs real-world accuracy separately. The base paper also targets natural scenes; compare its setup.
5. **Honest negative results.** In Gen 1 the arms tied. Reporting this, with the proxy-gap lesson, is itself a contribution that few tuning papers show.

## To do before writing the paper

- [ ] Read papers 2–5 in full: task, dataset, metrics, results, and whether they include baselines or significance tests. Update each status to verified.
- [ ] Fill in the authors and year of paper 5.
- [ ] Build a comparison table (paper × task × dataset × optimiser × baseline × statistical test × real-world evaluation).
- [ ] Add the Gen 2 final results (g2_6) to the gap analysis.
