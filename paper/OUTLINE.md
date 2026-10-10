# SwarmSort research paper: outline, gap analysis and flow

Status: draft outline for review (updated 2026-10-11). Target: journal article in LaTeX; the journal is not chosen yet. Gen 2 test results are filled in (from `results/gen2/test_*`). Only the Gen 1 vs Gen 2 comparison is still running, and it is marked **[PENDING G1 vs G2]**.

---

## 1. Title

**Chosen (2026-10-11):** *SwarmSort: A Fair Evaluation of Particle Swarm Hyperparameter Optimization for YOLOv8 Waste Detection Across Studio, Real-World, and Indian Data* (American spelling, as IEEE uses). Author: Shreyansh Jain. Format: IEEE journal style (IEEEtran, two columns), LaTeX, 10–15 pages.

Options that were considered:

1. *Does swarm tuning pay off? An equal-budget comparison of particle swarm optimisation, random search and default hyperparameters for lightweight waste detection, from studio to real-world and Indian scenes*
2. *SwarmSort: a fair evaluation of particle swarm hyperparameter optimisation for YOLOv8 waste detection across studio, real-world and Indian data*
3. *Particle swarm optimisation versus random search for training a lightweight waste detector: an equal-budget study with real-world and Indian test sets*

## 2. The story in one paragraph

Waste-sorting AI is mostly studied as single-object image classification on clean studio photos. Where swarm or other nature-inspired optimisers are used to tune the networks, they are applied to classifiers, compared with other metaheuristics or with nothing, and reported as wins from single runs. We ask a simpler and stricter question: if you give PSO, random search and the default settings the same budget, does PSO actually train a better multi-object waste detector, and does the answer hold when the photos move from the studio to real streets, beaches and Indian neighbourhoods? We build a lightweight YOLOv8n detector and a hand-written PSO, check that the cheap proxy used for the search ranks settings correctly (ρ = 0.976), and test once on held-out data with paired-bootstrap confidence intervals. On studio data alone (Gen 1) the three arms tie (test mAP@50 0.676 / 0.672 / 0.668; every 95% CI of the differences contains 0). With real-world training data (Gen 2), both tuned arms beat the defaults on the mixed test set (+0.017 for random search, +0.020 for PSO, both significant), but PSO and random search cannot be told apart. The gains vanish on real-world photos (0.298–0.310, no significant differences), and all three models fail on Indian street photos (0.023–0.027). In short: tuning helps a little, the choice of optimiser does not matter, and the data matter most. The contribution is the protocol and this honest answer, and both address gaps in how tuning claims are tested in waste AI.

## 3. Research gaps (evidence-based)

The gaps below come from 44 references, each confirmed to exist via Crossref or arXiv: 6 papers you found, 22 from a targeted search, and 16 dataset and method references. The search was targeted, not a systematic review, so every gap is worded as "in the literature we reviewed". Papers marked † were read only at abstract level and are cited for their existence or their stated scope only.

### G1. Task and scene gap: classification on studio photos dominates
- **Evidence:**
  - Smart-bin and recycling papers classify one object per image on plain backgrounds: Yang & Thung 2016 (TrashNet), Kaya 2023, Sayed et al. 2024, Alshammari et al. 2025†, TrashBox (Kumsetty et al. 2022)†.
  - Fang et al. 2023 (review) finds that most vision work it surveys is whole-image classification.
  - Several "detection" papers still use near-single-object data: Melinte et al. 2020 boxes TrashNet photos; Kumar et al. 2021; UGV-NBWASTE averages about 1.1 objects per image.
- **Counter-evidence (we state it):** multi-object waste detection exists, mainly in CV and robotics venues: TACO, ZeroWaste†, UAVVaste, PlastOPol, TrashCan, FloW†, WaRP†, and the base paper (Majchrowska et al. 2022).
- **Final wording:** *"Waste recognition for sorting is dominated by single-label classification of one object on a plain background; multi-object detection in cluttered scenes is a smaller strand, and several detection studies still rely on single-object studio images."*
- **How SwarmSort addresses it:** multi-object detection (YOLOv8n) on cluttered studio and real-world scenes, with 7 material classes.

### G2. Optimisation gap: swarm tuning has not been applied to waste *detectors*
- **Evidence:**
  - Metaheuristic tuning in waste AI targets classifiers: Sayed et al. 2024 (beluga whale optimisation, InceptionV3) and Mao et al. 2021† (GA, DenseNet121).
  - PSO tuning of CNNs exists in other domains: Aguerchi et al. 2024 (mammography; only 3 architecture hyperparameters, 4 particles × 3 iterations).
  - Every waste-detection paper read in full used default, published or hand-set hyperparameters, or a tuning step it does not describe: Majchrowska, PlastOPol, UAVVaste, TrashCan, MJU-Waste, Fulton, Melinte, Kumar, Liao & Juang, Bartolo.
  - Fang et al. mention PSO/GA only for routing and prediction, not for vision models.
- **Final wording:** *"In the literature we reviewed, no waste detector had its training hyperparameters tuned by a swarm optimiser; detection studies used default or hand-set values, or a tuning step they do not describe."*
- **How SwarmSort addresses it:** a hand-written PSO tunes 6 training hyperparameters (lr0, lrf, momentum, weight decay, mosaic, scale) of a waste detector.

### G3. Evidence gap: tuning claims lack fair baselines and uncertainty estimates
- **Evidence:**
  - Sayed et al. compare against 4 other metaheuristics at an equal population and iteration count, with a Wilcoxon test on search fitness. They do not compare against random search, and their test metrics are single numbers without CIs; their abstract and table also report different accuracies.
  - Aguerchi et al. run no baselines and a single run.
  - None of the 10 waste-detection papers read in full reports repeated runs, confidence intervals or significance tests. Bartolo et al. state that statistics were not applicable; Kumar et al. report the best value on the split also used for testing.
- **Counter-evidence (we state it):** Sayed does compare optimisers at an equal budget and does test significance on the search fitness, so we do **not** claim that nobody compares optimisers.
- **Final wording:** *"Metaheuristic tuning in this area is compared against other metaheuristics or against nothing; we found no comparison with random search, which Bergstra & Bengio (2012) showed to be a strong baseline, at an equal evaluation budget, and test results are reported as single numbers without confidence intervals."*
- **How SwarmSort addresses it:** three arms (defaults, random search, PSO) at an equal budget of 32 proxy trainings; a proxy-validity check fixed before the run (ρ = 0.976); a test split used once; paired-bootstrap 95% CIs on every difference.

### G4. Generalisation gap: little studio-to-real testing, and no Indian detection benchmark
- **Evidence:**
  - TrashNet-based classifiers (Kaya, Sayed, Yang & Thung) never leave the studio. Lu & Chen 2022 (review) say many studies use simplified environments and artificially collected data, and Fang et al. note lab pictures differ from real waste.
  - The same YOLOv5x, trained and tested on each dataset separately, scores 84.9 AP50 on PlastOPol but 63.3 on TACO (Córdova et al. 2022). Accuracy depends heavily on the dataset, so results from one dataset say little about another.
  - Liao & Juang 2022 see detection degrade at new field sites, reported qualitatively.
  - India: DWSD (Ali et al. 2025, Kolkata) reports segmentation pixel accuracy only. Kumar et al. 2021 use mostly clean self-collected images and no separate test split. AquaVision† uses TACO-derived data, and SpotGarbage† (2016) does garbage-region segmentation. We found no Indian real-world waste test set with published detection baselines.
- **Counter-evidence (we state it):** cross-domain tests do exist (Bartolo et al. 2025, a workshop paper; Fulton et al. 2019, test footage shifted in time and place). South Asian detection data exist: UGV-NBWASTE (Bangladesh, no baselines) and pLitterStreet† (Sri Lanka and others, preprint).
- **Final wording:** *"Studio-to-real-world and cross-dataset evaluation is rare and recent, and we found no Indian real-world waste test set with published object-detection baselines."*
- **How SwarmSort addresses it:** Gen 2 trains on studio and real-world data (TACO, HITL), reports studio, real-world and India (DWSD, test-only) results separately, and gives the first detection numbers on DWSD that we know of: every arm scores only 0.023–0.027 mAP@50, which exposes a large domain gap. DWSD's mask-derived boxes merge touching items, so part of that gap is a labelling mismatch; the paper must say this.

### Observation O5 (framed as a contribution, not as a literature gap)
Every applied paper we read reports its method as an improvement. Sayed et al. call their optimiser best on all metrics even though one comparison is not significant. Clear null results on tuning were not found, which is evidence of absence only, so we do not claim a gap. **SwarmSort reports null and negative results** (the Gen 1 tie; PSO = random search in Gen 2; no gain on real-world photos; failure on Indian photos) **and explains them.**

### Table 1: literature comparison matrix (goes in Section 2.5)

Only papers read in full text appear here; abstract-only papers are cited in the text.
Abbreviations: Cls = classification, Det = detection, Seg = segmentation, RS = random search, n.r. = not reported.

| Study | Task | Scenes | How hyperparameters were set | Tuning baselines | Runs / CIs | Cross-domain test | India data |
|---|---|---|---|---|---|---|---|
| Yang & Thung 2016 (TrashNet) | Cls | Studio | Manual | — | 1 / no | No | No |
| Kaya 2023 | Cls | Studio (TrashNet) | Fixed | — | 1 / no | No | No |
| Sayed et al. 2024 | Cls | Studio (TrashNet) | Beluga whale opt. | 4 metaheuristics, equal pop./iter.; untuned reference; no RS | Runs n.r. + Wilcoxon on fitness / no test CIs | No | No |
| Aguerchi et al. 2024 | Cls (medical) | Clinical | PSO (architecture) | None | 1 / no | No | — |
| Majchrowska et al. 2022 (base) | Det (1 class) + Cls | Real-world, mixed | Manual | None | 1 / no | No | No |
| Hong et al. 2020 (TrashCan) | Det / Seg | Underwater | Fixed | None | 1 / no | No | No |
| Kraft et al. 2021 (UAVVaste) | Det | Aerial | Not described | None | 1 / no | No | No |
| Córdova et al. 2022 (PlastOPol) | Det (1 class) | Real-world | Published defaults | None | No SDs / no | Two datasets, each tested separately | No |
| Wang et al. 2020 (MJU-Waste) | Seg | Lab + TACO | Fixed (grid search for CRF only) | None | 1 / no | No | No |
| Fulton et al. 2019 | Det | Underwater | Community settings | None | 1 / no | Shifted footage | No |
| Melinte et al. 2020 | Det | Studio (TrashNet) | Manual + LR range test | None | 1 / no | Lab set only | No |
| Kumar et al. 2021 | Det | Mostly clean | Fixed config | None | 1 / test = val | No | Yes (authors) |
| Liao & Juang 2022 | Det | Aerial, coastal | "Tuned", method n.r. | None | No | Qualitative, new sites | No |
| Bartolo et al. 2025 | Det | Aerial | Manual | None | 1 / no | Yes | No |
| Ali et al. 2025 (DWSD) | Seg (dataset) | Real-world | n.r. | — | 1 / no | No | Yes |
| **SwarmSort (ours)** | **Det (7 classes)** | **Studio + real-world** | **PSO vs RS vs defaults** | **Equal budget, 32 proxy trainings each** | **1 training seed / paired-bootstrap 95% CIs** | **Yes: studio, real-world, India** | **Yes (DWSD test)** |

Note: our "1 training seed" is stated openly in the matrix and in Limitations.

---

## 4. Research questions

- **RQ1:** At an equal budget of 32 proxy trainings, does PSO find YOLOv8n training hyperparameters that beat random search and the default settings on a held-out test set? (G2, G3)
- **RQ2:** Does a short proxy training (12 epochs on 40% of the data) rank hyperparameter settings the same way longer training does? (G3)
- **RQ3:** Does the answer to RQ1 change when the data move from studio images to real-world and Indian scenes, and how large is the studio-to-real-world drop? (G1, G4)

## 5. Contributions (each mapped to a gap)

| # | Contribution | Gap |
|---|---|---|
| C1 | A lightweight multi-object waste detector (YOLOv8n, 7 material classes) trained on studio and real-world scenes | G1 |
| C2 | The first swarm-tuned waste detector we know of: a hand-written PSO over 6 training hyperparameters | G2 |
| C3 | An equal-budget, three-arm protocol (defaults, random search, PSO) with a pre-registered proxy-validity check, a test split used once, and paired-bootstrap CIs | G3 |
| C4 | A two-generation evaluation that reports studio, real-world and Indian (DWSD) results separately, including the first detection numbers on DWSD that we know of | G4 |
| C5 | An honest answer: PSO never beat random search at an equal budget, and tuning gains were small and did not reach real-world or Indian photos; with an explanation, plus open code and data lists | O5 |

---

## 6. Section-by-section outline (with content and sources)

Target length: 10–15 IEEE two-column pages, which is about 10,000–11,500 words plus about 8 figures and 9 tables, with about 45 references. Word budget: Abstract 250 · Introduction 1,100 · Related Work 2,600 · Methods 2,600 · Results 2,000 · Discussion 1,400 · Future Work 900 · Conclusion 350.

### Abstract (≈250 words)
Problem, gaps, method (three arms, equal budget, proxy check, bootstrap), datasets (studio, real-world, India), key numbers (Gen 1 tie; Gen 2: tuned arms +0.017/+0.020 over the defaults on the mixed test, PSO ≈ random, real-world 0.30–0.31 with no differences, India 0.02–0.03), conclusion.
**Keywords:** waste detection; particle swarm optimisation; hyperparameter optimisation; YOLOv8; random search; real-world generalisation; India.

### 1. Introduction (≈900 words)
1. The waste problem and why sorting needs automation. Cite the Indian context with the SWM Rules 2016 (bin colours: green, white and black, per Rule 15(h)) and one of the reviews (Abdallah 2020†, Fang 2023).
2. Real sorting needs *detection* (many items per photo), not just classification. Cite TrashNet-style work vs the base paper.
3. Deep detectors depend on training hyperparameters, and metaheuristics are popular for tuning them (Sayed, Aguerchi). But how good is the evidence? This is G2 and G3 in two sentences.
4. Real-world gap: many studies use simplified, lab-like data (Lu & Chen; Fang), and accuracy varies widely between datasets (PlastOPol: the same YOLOv5x scores 84.9 vs 63.3 AP50 on two litter datasets). This is G4.
5. The research questions RQ1–RQ3 and contributions C1–C5 as a bulleted list.
6. Paper organisation.

### 2. Related work and research gaps (≈2,000 words: the heart of what the teacher asked for)
- **2.1 Review method (short):** the sources searched (Crossref, arXiv, Europe PMC, publisher pages), the keywords, the inclusion rule (peer-reviewed, 2016–2026, waste vision or PSO/HPO for CNNs), and how many were screened and included. We state that this is a targeted review, not a systematic one.
- **2.2 Vision-based waste recognition: from classification to detection.**
  - Classification: TrashNet (Yang & Thung), Kaya, Sayed, Alshammari†, TrashBox†.
  - Reviews: Abdallah†, Lu & Chen†, Fang.
  - Detection datasets: TACO†, ZeroWaste†, UAVVaste, PlastOPol, TrashCan, MJU-Waste, FloW†, WaRP†.
  - Detectors: the base paper (two-stage EfficientDet-D2 + EfficientNet-B2), Fulton, Melinte, Kumar, Liao & Juang, Bartolo, Sánchez-Ferrer†.
- **2.3 Waste data from India and South Asia:** SpotGarbage†, Kumar, AquaVision†, TrashBox†, DWSD, UGV-NBWASTE, pLitterStreet†.
- **2.4 Hyperparameter optimisation and swarm intelligence.**
  - PSO: Kennedy & Eberhart 1995, Shi & Eberhart 1998 (inertia weight), Clerc & Kennedy 2002.
  - Random search as a strong baseline: Bergstra & Bengio 2012.
  - Multi-fidelity and early stopping: Hyperband (Li et al. 2018).
  - Metaheuristic tuning of CNNs: Aguerchi, Sayed, Mao†.
- **2.5 Comparative analysis and research gaps:** Table 1 (the matrix above), then the gap statements G1–G4 and observation O5, then Table 2 (gap → contribution, section 5 above).

### 3. Materials and methods (≈2,200 words)
- **3.1 Experimental design.** The three arms; equal budget by evaluation count. Figure 1: pipeline (update `docs/images/pipeline.png` for Gen 2).
- **3.2 Datasets** (Table 3).
  - Gen 1: Kaggle garbage-detection data, 10,464 images, 6 classes, 416 px. The provided split was rejected (split by class) and replaced with a stratified 70/20/10 re-split. Duplicate control (dHash + correlation): 47 groups, 0 cross-split leakage.
  - Gen 2: TACO 1,500 + HITL 3,201 + Gen 1 subsample, 7 classes (OTHER added), 640 px; train/val/test = 6,292 / 1,841 / 1,514 images.
  - DWSD (India) is test-only. Its class mapping was inferred from the masks, and boxes come from connected regions.
- **3.3 Detector:** YOLOv8n, COCO-pretrained, Ultralytics 8.4.169; why a nano model (edge deployment).
- **3.4 Search space** (Table 4): 6 hyperparameters, ranges, log scale for lr0 and weight_decay.
- **3.5 PSO:**
  - Equations for the velocity and position updates.
  - Settings: 8 particles × 4 rounds; w from 0.9 to 0.4 (cite the correct source, still being confirmed); c1 = c2 = 1.5; velocity clamp ±20% of range; boundary handling by clipping.
  - Sanity check on Sphere and Rastrigin (`results/plots/pso_benchmarks.png`).
- **3.6 Random search and defaults:** 32 log-uniform samples (seed 123); Ultralytics defaults.
- **3.7 Proxy fitness and its validation:** 12 epochs on a stratified 40% subset, fitness = val mAP@50. The check uses 7 LHS settings plus the defaults, comparing proxy runs with 40-epoch full-data runs; the pass threshold ρ ≥ 0.6 was fixed in advance.
- **3.8 Full training:** 100 epochs, SGD, batch 16, warmup 3%, close_mosaic 10%, seed 42; Tesla T4.
- **3.9 Evaluation protocol:** test split used once; mAP@50 and mAP@50-95; consistency check; paired image bootstrap (1,000 resamples) with 95% CIs on differences; Gen 2 results per domain.

### 4. Results (≈1,800 words)
- **4.1 Proxy validity (RQ2):** ρ = 0.976, n = 8, one adjacent swap. Figure: proxy vs medium scatter (new; easy to make). Cross-generation rank agreement on repeated settings: ρ = 0.991 (random) and 0.892 (PSO).
- **4.2 Search behaviour:**
  - Gen 1: random search found its best (0.4995) at evaluation 2 and never improved. PSO's round means rose from 0.39 to 0.48 while the spread shrank (SD 0.12 to 0.01), with a final best of 0.4953. The defaults scored 0.4765 on the same proxy.
  - Gen 2: random 0.3340 vs PSO 0.3290; PSO never improved after round 1.
  - Figure: search convergence (Gen 1 exists; Gen 2 to make).
- **4.3 Gen 1 test results (RQ1, studio):** Table 6 with A 0.676 / B 0.672 / C 0.668 and pairwise CIs that all contain 0; per-class AP (PSO is worse on PLASTIC, 0.638 vs 0.699); latency.
- **4.4 Gen 2 results (RQ1, mixed data):** Table 7 (source: `results/gen2/test_{all,studio,real_world,india}_comparison.md`, `test_domains.md`; test split used once, Kaggle T4).
  - Val (one run each): A 0.4535 / B 0.4734 / C 0.4662.
  - Test mAP@50 [95% CI], all (1,514 images): A 0.479 [0.453, 0.513], B 0.497 [0.471, 0.531], C 0.500 [0.474, 0.534].
  - Pairwise (all): B−A +0.017 [+0.006, +0.030] and C−A +0.020 [+0.011, +0.032], both significant; C−B +0.003 [−0.008, +0.015], not significant. The mAP@50-95 verdicts agree.
  - Studio (1,046): A 0.571, B 0.595, C 0.590. B−A +0.024 and C−A +0.019 are significant; C−B is not.
  - Per class (all): PSO is best on 5 of 7 classes, random search on 2 (BIODEGRADABLE, PLASTIC), and the defaults on none. OTHER is very weak for every arm (AP@50 0.050–0.063).
  - Latency on a T4 at 640 px: 8.1–8.7 ms per image.
- **4.5 Domain gap (RQ3):** Figure: per-domain bars, plus sample detections from each domain (Gen 2 plots still to make; there is no `results/gen2/plots/` yet).
  - Mean across arms, studio → real-world → India: about 0.59 → 0.31 → 0.03 mAP@50.
  - Real-world (468 images, 779 boxes): A 0.310, B 0.298, C 0.310; no pair is significant. The intervals are wide and lopsided (e.g. A [0.278, 0.440]) because BIODEGRADABLE has a single real-world test box; say so.
  - India (DWSD, 700 images, 4,502 boxes, test-only): A 0.027, B 0.023, C 0.027 (precision 0.070–0.075, recall 0.043–0.047). The defaults and PSO are significantly above random search, but the gaps are a few thousandths on scores near zero, so they carry no practical meaning. Reasons: no Indian training images, augmented-looking photos, mask-derived boxes that merge touching items, and an inferred class mapping.
  - **Gen 1 vs Gen 2 on the same test images** (6 shared classes; OTHER removed): does adding real-world training data help on real-world photos? The Project session is running this on Kaggle now (`src/compare_generations.py`), and its numbers go here. [PENDING G1 vs G2]
- **4.6 Compute cost:** search about 2 GPU-hours per method per generation; full trainings 1.2–2.1 h each; about 22 GPU-hours recorded in total.

### 5. Discussion (≈1,200 words)
- **5.1 Why PSO never beat random search, and why tuning helped only in Gen 2:**
  - Gen 1: the default settings already sat near a flat optimum (they ranked high on the proxy).
  - Gen 2: the new data (real-world photos, 640 px, 7 classes) probably moved the optimum away from the defaults, so any reasonable search helped (about +0.02). This is a hypothesis: Gen 2 has no defaults-on-proxy score to test it.
  - The budget is small (32 evaluations in 6 dimensions).
  - Proxy noise is about 0.02 mAP, larger than the gaps between top settings.
  - Random search is strong in low effective dimensions (Bergstra & Bengio).
  - PSO's swarm converged (consistency) without reaching better optima.
- **5.2 What this means for metaheuristic-tuning claims in waste AI:** G3 restated as a recommendation: always include defaults and random search at an equal budget, and report CIs.
- **5.3 Real-world and Indian deployment:**
  - Scores halve from studio to real-world photos and collapse on Indian streets, whatever the tuning. A domain with no training data fails almost completely.
  - For smart bins and municipal use in India, local training data matter more than the optimiser. This links straight to future-work item 6.2.
  - A test set must be labelled like the training data: DWSD's merged boxes punish correct per-item detections, so its score measures the labelling difference as well as the model.
- **5.4 Limitations and threats to validity:**
  - One training seed per arm and one search seed per method (the CIs cover test sampling only).
  - The proxy check was n = 8, Gen 1 only, against a 40-epoch reference.
  - Proxy fitness uses last-epoch weights, while full training uses best.pt.
  - DWSD labels were inferred from masks, with boxes from connected regions.
  - Gen 2 Arm C was resumed after a session freeze.
  - No external baseline (the base paper's pipeline was not rerun).

### 6. Future work: possible improvements (≈800 words)
Every improvement is tied to a limitation (5.4) or a measured result, so the section reads as "what we found → what to try next", not as a wish list. It draws on `docs/FUTURE_WORK.md` (item numbers in brackets) plus gaps found while preparing the paper (marked *new*). It ends with Table 9, a prioritised roadmap.

- **6.1 Strengthen the optimisation study (the soft-computing core)**
  - *Several training seeds per arm* (e.g. 3 each), so the CIs cover training noise as well as test sampling. Addresses: one seed per arm. [4.1]
  - *Several search seeds per method*, to see whether PSO's result is repeatable. *new*
  - *Multi-fidelity search*: successive halving or Hyperband (Li et al. 2018). Addresses: the 12-epoch proxy ranked settings well overall, but the small gaps between the top settings disappeared after 100 epochs. [4.2]
  - *PSO variants*: constriction factor (Clerc & Kennedy 2002), ring topology, adaptive inertia, and more particles and rounds. Addresses: global-best PSO with 32 evaluations may converge early (in Gen 2, PSO never improved after round 1). [4.3]
  - *Other optimisers at the same budget*: genetic algorithm, differential evolution, and Bayesian optimisation such as TPE/Optuna. This also answers "why PSO and not beluga whale optimisation?" (Sayed et al. 2024). [4.4]
  - *A wider search space*: colour augmentation, translate, loss gains, image size. Addresses: the 6 tuned settings left little room to beat the defaults on studio data. [4.5]
  - *Domain-aware fitness*: score candidates on real-world validation images. Addresses: in Gen 2, tuning beat the defaults on the studio and mixed test sets, but not on real-world photos. [4.6]
  - *Repeat the proxy-validity check* for Gen 2, against a 100-epoch reference. *new*
  - *Stronger statistics*: p-values with a multiple-comparison correction (e.g. Holm), CIs for per-class AP, and stored bootstrap resamples. *new*
- **6.2 Data and the Indian domain**
  - *Indian training data*: train on part of DWSD and keep a held-out part for testing, with per-item boxes instead of mask regions. Addresses: every arm scores only 0.023–0.027 test mAP@50 on DWSD. This is the main next step. [2.6]
  - *Fix the weak OTHER class* (test AP@50 0.050–0.063 for every arm): add examples, split it into clearer classes, or merge it. [2.7]
  - *Clutter augmentations*: copy-paste, smaller object scales, blur and lighting changes. Addresses: small, overlapping, partly hidden items. [2.4]
  - *Classes that follow the SWM Rules 2016 streams* (bio-degradable, non-bio-degradable, domestic hazardous), and separating transparent plastic from glass. [2.5]
- **6.3 Model**
  - *Larger or newer detectors* (YOLOv8s/m, YOLO11) at the same training settings; the trade-off is speed on edge devices. [3.1]
  - *Sliced inference* (SAHI, Akyon et al. 2022) for many small items in wide photos. [3.3]
  - *Confidence calibration* (e.g. temperature scaling), so that a score of 0.5 means roughly 50% correct. [3.4]
- **6.4 Combining the three models (ensemble)** [1.1, 1.2]
  - *Why:* the three arms were trained with different hyperparameters and make different mistakes. On real photos they often disagree, and no single arm wins on every Gen 2 test set (PSO is highest on all, random search on studio, and defaults and PSO tie on real-world and India). Combining them could beat each one alone, and it reuses models we already have, so no extra training is needed.
  - *How (three options, compared on the same test sets):*
    1. **Box fusion:** pool the boxes of A, B and C and merge overlapping ones with weighted box fusion (Solovyev et al. 2021; to verify) or NMS.
    2. **Majority voting:** keep a box only if at least 2 of the 3 models find it (IoU ≥ 0.5). Expected to raise precision, and boxes found by only one model can be flagged as "uncertain" in the app.
    3. **Weighted voting:** weight each model by its own validation mAP@50, never by test scores.
  - *Soft-computing extension:* the search already produces a *population* of good settings (the final PSO swarm and the top random samples). Training the top-k of them and ensembling those uses the swarm as an ensemble generator, not just a single-answer optimiser.
  - *Evaluation:* paired bootstrap against the best single arm on all, studio, real-world and India; report the cost. Inference takes about 3× as long: roughly 100 ms per image on the Gen 1 laptop CPU, from 31–38 ms per model.
- **6.5 Deployment**
  - *Per-class confidence thresholds*, since many wrong boxes on real photos sit just above the default threshold. [1.3]
  - *Edge export* (ONNX/TFLite) and a latency test on a Raspberry Pi or phone, the same class of hardware Alshammari et al. 2025 use for a smart-bin classifier. [5.3]
  - *Live video* from a camera, since real sorting is continuous. [5.2]

**Table 9: future-work roadmap** (order by expected value for effort)

| Priority | Improvement | Addresses | Effort |
|---|---|---|---|
| 1 | Indian training data, per-item boxes | DWSD failure (0.023–0.027 mAP@50) | Large |
| 2 | Ensemble of A + B + C (fusion or voting) | Arms make different mistakes; no arm wins everywhere | Small (no training) |
| 3 | 3 training seeds per arm, plus several search seeds | Tuning effect vs training noise | Medium |
| 4 | Domain-aware fitness (real-world validation) | Tuning gains did not carry to real-world photos | Medium |
| 5 | Multi-fidelity search (Hyperband) | Proxy gaps vanish at 100 epochs | Medium |
| 6 | GA / DE / Bayesian optimisation at the same budget | "Why PSO?" | Medium |
| 7 | Fix the OTHER class | AP@50 0.05–0.06 | Medium |
| 8 | Larger detector; SAHI | Small items, look-alike materials | Medium |
| 9 | Edge export and latency on a Raspberry Pi | Real smart-bin hardware | Medium |
| 10 | Swarm as an ensemble generator (top-k settings) | Uses the whole PSO population | Medium |

### 7. Conclusion (≈300 words)
- Answers to RQ1–RQ3 in one or two sentences each.
- The main message: tuning claims need equal-budget baselines and confidence intervals, and real-world (especially Indian) data matter more than the optimiser.
- One sentence pointing to Section 6.

### Back matter (depends on the journal)
Data availability (dataset links and licences: CC BY 4.0 / CC0), code availability (GitHub), funding, competing interests, author contributions, acknowledgements.

---

## 7. Flow (how each section hands over to the next)

1. **Intro:** sorting needs detection → tuning is popular → but the evidence is weak → we ask RQ1–RQ3.
2. **Related work:** prove the weakness with the matrix (Table 1) → four gaps.
3. **Methods:** each design choice closes a gap: equal budget closes G3, detection closes G1, PSO on a detector closes G2, and per-domain tests close G4.
4. **Results:** answer the RQs in order: RQ2 first (the proxy is valid, so the search is trustworthy), then RQ1, then RQ3.
5. **Discussion:** explain the results (PSO = random; small gains that do not transfer) → turn them into a recommendation for the field → be honest about the limits.
6. **Future work:** each limitation becomes a concrete improvement, prioritised in a roadmap (Table 9).
7. **Conclusion:** short answers and the main message.

## 8. Figures and tables plan

| # | Item | Status |
|---|---|---|
| Fig. 1 | Experimental pipeline (three arms, two generations) | Exists for Gen 1 (`docs/images/pipeline.png`); needs a Gen 2 update |
| Fig. 2 | PSO flowchart and real trajectories (`docs/images/pso_flowchart.png`, `pso_real.png`) | Exists |
| Fig. 3 | Proxy vs medium scatter (ρ = 0.976) | To make (from `results/proxy_check.json`) |
| Fig. 4 | Search convergence, Gen 1 and Gen 2 | Gen 1 exists; Gen 2 to make |
| Fig. 5 | Per-class / per-domain test AP | Gen 1 exists; Gen 2 data ready, plot to make |
| Fig. 6 | Sample detections: studio, real-world, India | Gen 1 exists; Gen 2 to make (needs the Gen 2 weights) |
| Table 1 | Literature comparison matrix | Drafted above |
| Table 2 | Gap → contribution | Drafted above |
| Table 3 | Datasets | Data ready |
| Table 4 | Search space and PSO settings | Data ready |
| Table 5 | Proxy check (8 settings) | Data ready |
| Table 6 | Gen 1 test results and pairwise CIs | Data ready |
| Table 7 | Gen 2 test results per domain | Data ready |
| Table 7b | Gen 1 vs Gen 2 on the same test images | [PENDING G1 vs G2] |
| Table 8 | Compute cost | Data ready |
| Table 9 | Future-work roadmap | Drafted in Section 6 |

## 9. Open items before writing

1. **Journal** (sets the template, word limit and citation style).
2. **Title** (section 1).
3. **Authors and affiliation** (names, department, college; whether the teacher is a co-author).
4. **Gen 1 vs Gen 2 comparison** (running on Kaggle in the Project session): needed for 4.5 and Table 7b. The Gen 2 test results are already filled in.
5. **Citation checks still open:** the source of the 0.9 → 0.4 inertia schedule (probably Shi & Eberhart 1999, CEC); confirm Mao et al. 2021 from its full text before citing more than its title.
6. **Wording fix carried into the paper:** bin colours are green, white and black (SWM Rules 2016, Rule 15(h)), not green and blue.
7. **Future-work citations to look up** (not yet verified; a short targeted check, no full search): TPE (Bergstra et al. 2011) or Optuna (Akiba et al. 2019); one GA and one DE reference for hyperparameter tuning; copy-paste augmentation (Ghiasi et al. 2021); temperature scaling (Guo et al. 2017); weighted box fusion (Solovyev et al. 2021); the Ultralytics YOLO11 software citation.
