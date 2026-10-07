# How SwarmSort evolves

SwarmSort is built in generations. Each generation has one clear goal and ends with a result, a list of problems and a reason to start the next one. This file keeps that history in one place, so anyone (a classmate, an examiner, or us later) can see what was tried, what went wrong and why the project changed direction.

The full write-up of the finished work is in the [README](../README.md). Ideas that are not yet scheduled are in [FUTURE_WORK.md](FUTURE_WORK.md).

## Generations at a glance

| Gen | Dates | Data | Model | Method | Headline result | Status |
|---|---|---|---|---|---|---|
| 1 | 2026-09-30 to 2026-10-05 | Kaggle garbage-detection: 10,464 images, 6 classes, mostly single objects on clean backgrounds | YOLOv8n, imgsz 416 | Three arms at equal budget: A defaults, B random search, C PSO | Test mAP@50 A 0.676, B 0.672, C 0.668: a statistical tie | Finished |
| 2 | from 2026-10-07 | Part of the Gen 1 data plus real-world litter photos (TACO, HITL Recycling); DWSD (Kolkata streets) as a held-out India test set; 7 classes (OTHER added) | YOLOv8n, imgsz 640 | The same three arms, rerun on the merged data | None yet | In progress |

## How to update this document

- Add a new section for each new generation. Use the same headings as the generations below.
- Never rewrite history. If an earlier statement turns out to be wrong, add a dated note under it instead of deleting it.
- Record every decision with its date in the generation's decisions log.
- Take every number from a file in `results/` (or the new generation's own results files), never from memory.
- Update the summary table when a generation finishes or changes status.

---

## Generation 1: does PSO beat the defaults? (finished)

**Dates:** 2026-09-30 to 2026-10-05.

### Goal

Find out whether a hand-written Particle Swarm Optimizer (PSO) can choose better training hyperparameters for a YOLOv8n waste detector than the library defaults or plain random search, under a fair, equal budget.

### Data

- One dataset: Kaggle [Garbage Detection – 6 Waste Categories](https://www.kaggle.com/datasets/viswaprakash1990/garbage-detection) (viswaprakash1990), CC BY 4.0.
- 10,464 images of 416×416 px with YOLO boxes in 6 classes: BIODEGRADABLE, CARDBOARD, GLASS, METAL, PAPER, PLASTIC.
- Most images show one or a few objects on a clean, plain background.
- Our own split, 70/20/10, stratified by class and duplicate-aware (seed 42): 7,324 train, 2,094 val, 1,046 test images. Stored in `configs/split.csv`. See README: [Dataset and the re-split discovery](../README.md#1-dataset-and-the-re-split-discovery).

### Method

- **Model:** YOLOv8n, COCO-pretrained, imgsz 416.
- **Six searched hyperparameters:** lr0, lrf, momentum, weight_decay, mosaic, scale.
- **Three arms, same budget:**
  - A: Ultralytics defaults, no search.
  - B: random search, 32 proxy trainings (seed 123).
  - C: PSO, 8 particles × 4 rounds = 32 proxy trainings (seed 42).
- **Proxy fitness:** 12 epochs on a stratified 40% subset of train, scored by val mAP@50. Before any search we checked that the proxy ranks configurations like a 40-epoch full-data run: Spearman ρ = 0.976 on 8 configurations, against a pass mark of 0.6 fixed in advance. See README: [Proxy fitness and its validation](../README.md#5-proxy-fitness-and-its-validation).
- **Full training:** each arm's chosen settings trained once for 100 epochs with the same recipe (explicit SGD, seed 42, deterministic). See README: [Full training recipe](../README.md#6-full-training-recipe).
- **Test:** the test split was used once, at the very end. A paired bootstrap (1,000 resamples of the 1,046 test images) gave 95% confidence intervals for each model and each difference.
- **Delivered:** a Streamlit demo app with green/blue bin advice ([README: Demo app](../README.md#demo-app)) and a Word report (`report/SwarmSort_Report.docx`).

### Results

Sources: `results/search_*.json`, `results/best_*.json`, `results/baseline.json`, `results/arm_b.json`, `results/arm_c.json`, `results/test_comparison.md`, `results/test_results.json`.

**Search (proxy, val mAP@50).** Both searches beat the defaults on the proxy. The PSO swarm converged: its mean fitness per round rose from 0.393 to 0.482. Random search found its best point on its 2nd sample, and 28 of its 32 samples scored below the defaults.

| | A: Defaults | B: Random search | C: PSO |
|---|---|---|---|
| Proxy val mAP@50 | 0.4765 | 0.4995 | 0.4953 |
| Search time (Tesla T4) | none | 2.12 h | 1.91 h |

**Full training (val split, 100 epochs).**

| Arm | val mAP@50 | val mAP@50-95 | Training time (Tesla T4) |
|---|---|---|---|
| A: Defaults | 0.657 | 0.462 | 1.18 h |
| B: Random search | 0.643 | 0.458 | 1.15 h |
| C: PSO | 0.657 | 0.463 | 1.45 h |

**Test split (1,046 images, 7,415 boxes, used once).**

| Arm | test mAP@50 [95% CI] | test mAP@50-95 [95% CI] | ms/img (laptop CPU) |
|---|---|---|---|
| A: Defaults | 0.676 [0.639, 0.722] | 0.474 [0.441, 0.518] | 31.4 |
| B: Random search | 0.672 [0.634, 0.717] | 0.475 [0.441, 0.519] | 38.3 |
| C: PSO | 0.668 [0.627, 0.717] | 0.471 [0.437, 0.516] | 31.7 |

All pairwise differences in test mAP@50 (B−A −0.004, C−A −0.008, C−B −0.004) have 95% intervals that contain 0. **Verdict: a statistical tie.** The defaults were already close to the best this model and dataset allow. Full tables: [README: Results](../README.md#results) and `results/test_comparison.md`.

### Problems and fixes

| Problem | Symptom | Cause | Fix |
|---|---|---|---|
| Provided split was by class | The dataset's own valid folder had only 33 PAPER boxes, and its test folder had 0 GLASS boxes | The author's train/valid/test folders were not a random split | Pooled all images and re-split 70/20/10, stratified by class (seed 42); split saved in `configs/split.csv` |
| Near-duplicate leakage | Copies of the same photo could end up in both train and test, which inflates test scores | The dataset contains near-identical copies of some photos | Duplicate rules: same bytes, same source image, dHash within 10 bits, or dHash within 40 bits plus thumbnail correlation of at least 0.90. 138 near-duplicate pairs in 47 groups were kept in one split; 0.00% of val and test images have a copy elsewhere |
| Ultralytics `optimizer=auto` | Changing lr0 and momentum would have had no effect, so the search would be pointless | `auto` picks the optimizer and learning rate by itself and ignores lr0/momentum | Always pass `optimizer="SGD"` explicitly |
| Ultralytics `fraction` | A "40% subset" would not be a fair sample | `fraction` keeps the first N images in sorted order, not a random sample | Build our own stratified, seeded 40% subset and pass it as a separate data file |
| Hidden random draw | The first training in a process gave different results from later or resumed runs with the same seed | Ultralytics imports one module lazily during the first training, after seeding, and that import draws from Python's `random` | Import that module before seeding, so every run is identical |
| Kaggle mount depth | Dataset discovery failed on Kaggle | Kaggle mounts datasets several folders deeper than expected (`/kaggle/input/datasets/<owner>/<slug>/<folder>/...`) | The dataset finder searches up to 7 folder levels |
| `.pt` files arrive as `.zip` | Trained models downloaded from Kaggle had a `.zip` name | A PyTorch checkpoint is itself a zip archive, so the browser renames it | Rename the file back to `best.pt`; do not unzip it |
| The proxy gap | Random search's winner had the best proxy score (0.4995) but a lower full-training val mAP@50 than the defaults (0.643 against 0.657) | A short run on 40% of the data rewards fast learners; the top candidates were within about 0.02 of each other on the proxy, and that edge faded over 100 epochs | Reported as it is. Future fixes: several seeds per arm and multi-fidelity search ([FUTURE_WORK](FUTURE_WORK.md) items 4.1, 4.2) |
| **The domain gap** (found in the live demo) | On a real street photo with scattered litter, the models labelled crumpled polythene and cans as GLASS at 0.35–0.5 confidence, and the three models disagreed with each other, although all score about 0.67 test mAP@50 | Training images are mostly isolated objects on clean backgrounds. In a wide street scene, each item shrinks to about 10–20 px at imgsz 416 | Not fixable within Gen 1. This is what started Gen 2 |

The domain-gap row is an observation from using the demo, not a measured result. There is no labelled real-world test set in Gen 1, so real-world accuracy was never measured.

### Lessons

- A fair comparison (equal budget, one fixed recipe, test used once, paired bootstrap) is what makes a negative result believable.
- Check the dataset before trusting it: the provided split and the duplicates would both have given wrong numbers.
- Read the library's defaults carefully. Two Ultralytics settings and one hidden random draw would have silently broken the experiment.
- A proxy can rank very different settings well (ρ = 0.976) and still be unable to separate the top few.
- A good test score only describes images like the test set. Accuracy on a clean dataset says little about cluttered real scenes.
- Hyperparameter tuning cannot fix missing data. When the model has never seen litter in a real scene, the data has to change first.

### What triggered Generation 2

The domain gap. The goal of SwarmSort is to help sort real waste, but the Gen 1 models only work on photos that look like the training set. Tuning had already been shown to give no measurable gain on that data, so the next step is better data, not more tuning.

---

## Generation 2: waste in real places (in progress)

**Started:** 2026-10-07.

### Goal

Detect and classify waste in photos of normal environments (streets, parks, homes), not only isolated objects on clean backgrounds, and report studio-style and real-world accuracy separately.

### Data (chosen 2026-10-07)

The earlier plan listed `Garbage_dataset_PlusYaml` as a candidate. It was dropped on 2026-10-07 (see the decisions log).

| Source | Kaggle input | Format | Licence | Role | Domain |
|---|---|---|---|---|---|
| Gen 1 garbage-detection | `viswaprakash1990/garbage-detection` | YOLO | CC BY 4.0 | Train/val/test. Gen 1 split kept; a class-stratified, seeded sample of about 3,000 of 7,324 train and 900 of 2,094 val images; all 1,046 Gen 1 test images stay test | `studio` |
| TACO, official set (1,500 images, 4,784 boxes) | `kneroma/tacotrashdataset` | COCO | CC BY 4.0 | Train/val/test (70/20/10, duplicate-aware) | `real_world` |
| HITL Recycling (about 3,200 images, 1 box each) | `humansintheloop/recycling-dataset` | Supervisely JSON, class from the Material tag | CC0 | Train/val/test (70/20/10, duplicate-aware) | `real_world` |
| DWSD, Dense Waste Segmentation (784 images, Kolkata) | private upload of Mendeley `gr99ny6b8p` | Labelme polygons | CC BY 4.0 | **Test only**: all 784 images form the held-out "India real-world" test set | `india` |

- Classes (7, in index order): BIODEGRADABLE, CARDBOARD, GLASS, METAL, PAPER, PLASTIC, OTHER. Gen 1's six keep their indices; OTHER is a narrow catch-all (cigarettes, unidentified litter, mixed bin bags, textile and rubber, hazardous items, multi-layer blister packs).
- Every source's classes are mapped in `configs/gen2/sources.yaml`. TACO: all 60 categories map to a class, none is dropped. HITL: Object-tag overrides first (for example cigarette butt → OTHER, food waste → BIODEGRADABLE), then the Material tag; objects without a Material tag (about 8%) become OTHER. DWSD: its 14 classes (cloth → OTHER).
- Per-domain test sets: `data_test_studio.yaml`, `data_test_real_world.yaml`, `data_test_india.yaml`.
- Checked on TACO's real `annotations.json`: after mapping, BIODEGRADABLE 8, CARDBOARD 251, GLASS 254, METAL 544, PAPER 246, PLASTIC 2,221 and OTHER 1,260 boxes (4,784 in total), as the research predicted.

### Method (planned)

- A **multi-source dataset builder** (`src/build_dataset.py`):
  - readers for YOLO, COCO, Supervisely and Labelme sources, and a class-mapping config per source;
  - per-source options: force a source into one split (DWSD → test), subsample a pinned source (Gen 1), and drop-and-mask (remove a box and paint its region grey, available but not needed with OTHER);
  - a relocate mode, so later Kaggle notebooks can use the finished build when it is attached read-only;
  - a stratified, duplicate-aware split across all sources (the Gen 1 duplicate rules extended to the merged data);
  - Gen 1 test images stay in test, so the old test set is never trained on;
  - per-domain test lists, so studio (Gen 1 style) and real-world accuracy are reported separately.
- Rerun all three arms (A defaults, B random search, C PSO) with YOLOv8n on the merged data, on Kaggle GPUs, with the same fair-comparison rules as Gen 1.

### Plan

1. Survey and choose the extra datasets; check licences.
2. Write the class-mapping config for each source.
3. Build the multi-source builder and the merged split; check duplicates and leakage again.
4. Settle the open decisions below.
5. Run arms A, B and C on Kaggle.
6. Evaluate once on the test split, per domain, with the paired bootstrap.
7. Update the demo app and this document.

### Decisions log

| Date | Decision |
|---|---|
| 2026-10-07 | Gen 2 goal: detect and classify waste in normal environments (streets, parks, homes). |
| 2026-10-07 | TACO selected as the first real-world dataset. `Garbage_dataset_PlusYaml` is a candidate; other datasets are under research. |
| 2026-10-07 | Data is merged through a multi-source builder: class-mapping config, stratified duplicate-aware split, Gen 1 test images stay test, per-domain test lists. |
| 2026-10-07 | Model stays YOLOv8n, and all three arms (defaults, random search, PSO) are rerun on the merged data, on Kaggle. |
| 2026-10-07 | Step 0 (labelling our own phone photos as a real-world test set) is skipped. Held-out real-world dataset images serve as the real-world test set instead. |
| 2026-10-07 | Final sources: TACO official (`kneroma/tacotrashdataset`) and HITL Recycling (`humansintheloop/recycling-dataset`) for train/val/test, plus the Gen 1 dataset. |
| 2026-10-07 | DWSD (Mendeley gr99ny6b8p, Kolkata streets and parks) is used only as a held-out "India real-world" test set: all 784 images go to test. It is uploaded to Kaggle as a private dataset. |
| 2026-10-07 | `Garbage_dataset_PlusYaml` is dropped: mostly studio and web images, Roboflow augmentation copies, train/test leakage and label errors. |
| 2026-10-07 | An OTHER class is added: 7 classes, BIODEGRADABLE, CARDBOARD, GLASS, METAL, PAPER, PLASTIC, OTHER. Gen 1's six keep their indices. Gen 1 vs Gen 2 comparisons use the six shared classes. |
| 2026-10-07 | Training image size is 640. The builder stores images with a long side of at most 640 px (the training size), so data loading stays fast; a later generation that trains larger rebuilds the data. |
| 2026-10-07 | Gen 1 is subsampled to about 3,000 train and 900 val images (stratified by class, seeded); all 1,046 Gen 1 test images stay test, and Gen 1 split assignments still come from `configs/split.csv`. |
| 2026-10-07 | Domains: Gen 1 → `studio`; TACO and HITL → `real_world`; DWSD → `india`. Each has its own test list and yaml. |
| 2026-10-07 | Licences recorded per source: TACO CC BY 4.0, HITL CC0, DWSD CC BY 4.0, Gen 1 CC BY 4.0. |

**To be decided** (the OTHER class and the image size were settled on 2026-10-07, see above):

- Whether to re-run the proxy-validity check on the merged data.
- A stronger model (for example YOLOv8s) later, as Gen 3 ([FUTURE_WORK](FUTURE_WORK.md) item 3.1).

### Results

None yet.

### Problems and fixes

| Problem | Symptom | Cause | Fix |
|---|---|---|---|
| (none recorded yet) | | | |

### Status

In progress. Dataset selection and the multi-source builder come first.
