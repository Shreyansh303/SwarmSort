# Future work

Project history by generation, with dated decisions: [EVOLUTION.md](EVOLUTION.md).

A running list of improvements to SwarmSort, grouped by area and roughly ordered by value for effort. Each item says what problem it addresses, so the list can double as the "future work" discussion in the report and viva.

**Why this list exists.** The three models reach about 0.67 test mAP@50 on images that look like the training data (mostly one or a few items on clean backgrounds). On cluttered real-world photos, such as a street with scattered litter, they mislabel small items (crumpled polythene and cans shown as GLASS at 0.35–0.50 confidence), and the three models disagree with each other. Most items below target this domain gap.

Effort: **S** = hours, no GPU · **M** = a day, some GPU · **L** = several days or a new data pipeline.

| Status | Meaning |
|---|---|
| idea | not started |
| planned | agreed, not started |
| in progress | being worked on |
| skipped | dropped by a dated decision |
| done | finished (with commit or result) |

## 1. Quick wins (app and evaluation)

| # | Improvement | Problem it addresses | Effort | Status |
|---|---|---|---|---|
| 1.1 | **Ensemble of A + B + C** in the app: keep a box only if at least 2 of the 3 models agree (or merge boxes with weighted box fusion) | The models make different individual mistakes; agreement filters out uncertain guesses | S | idea |
| 1.2 | **Show model disagreement** as an "uncertain" flag on boxes only one model finds | Users can't tell confident detections from guesses | S | idea |
| 1.3 | **Raise the default confidence** (e.g. 0.35 → 0.5) or use per-class thresholds | Many wrong boxes on real photos sit just above 0.35 | S | idea |
| 1.4 | **Real-world test set**: 50–100 own phone photos, labelled (Label Studio or Roboflow), kept as a second, out-of-distribution test split | We only measure accuracy on dataset-style images; real-world accuracy is unknown | M | skipped (decision 2026-10-07; held-out real-world dataset images used instead) |

## 2. Data

| # | Improvement | Problem it addresses | Effort | Status |
|---|---|---|---|---|
| 2.1 | **Add TACO** (Trash Annotations in Context: litter photographed in real environments, about 60 categories), mapped to our 6 classes and mixed into training | Training images are mostly isolated objects on clean backgrounds; TACO has real, cluttered scenes | L | in progress (Gen 2) |
| 2.2 | **Add `Garbage_dataset_PlusYaml`** (Kaggle `engrbasit62`, about 12.7k images, 7 classes, already downloaded in the project folder), mapped to our classes | More variety per class; it also has an electronics class (see 2.5) | M | in progress (Gen 2) |
| 2.3 | Survey other detection datasets (e.g. ZeroWaste for conveyor-belt sorting, UAVVaste for aerial litter) and check their licences before use | Different viewpoints and settings | M | in progress (Gen 2) |
| 2.4 | **Augmentations for clutter**: copy-paste, smaller object scales, harder mosaic, blur and lighting changes | Small, overlapping, partly hidden items | M | idea |
| 2.5 | **More classes**: transparent plastic vs glass, e-waste and hazardous items (red/black bins under SWM Rules 2016) | GLASS vs PLASTIC confusion; bins the app can't recommend today | L | idea |

## 3. Model

| # | Improvement | Problem it addresses | Effort | Status |
|---|---|---|---|---|
| 3.1 | **Stronger model**: YOLOv8s/m or YOLO11n/s (supported by the same Ultralytics library), compared at equal training settings | YOLOv8n is the smallest model; a larger one should separate look-alike materials better (trade-off: slower on CPU) | M | idea |
| 3.2 | **Larger input size** (640 instead of 416) for real photos | Phone photos are shrunk to 416 px, so small items become 10–20 pixel blobs | M | idea |
| 3.3 | **Sliced inference (SAHI)**: run the detector on overlapping tiles of a large photo and merge the results | Many small items in one wide photo | S–M | idea |
| 3.4 | **Calibrate confidences** (e.g. temperature scaling) so 0.5 means roughly 50% correct | Scores are hard to interpret | M | idea |

## 4. Hyperparameter optimisation (the soft-computing part)

| # | Improvement | Problem it addresses | Effort | Status |
|---|---|---|---|---|
| 4.1 | **Several seeds per arm** (e.g. 3 full trainings each) | One training per arm can't separate the effect of tuning from training noise | M | idea |
| 4.2 | **Multi-fidelity search**: successive halving or Hyperband (start many configs cheaply, promote the best to longer runs) | The 12-epoch proxy ranked configs well overall, but gaps between top configs vanished after 100 epochs | M | idea |
| 4.3 | **PSO variants**: ring topology, constriction factor, adaptive inertia; more particles and rounds | Standard global-best PSO with 32 evaluations may converge early | M | idea |
| 4.4 | **Compare with other optimisers** at the same budget: genetic algorithm, differential evolution, Bayesian optimisation (e.g. Optuna's TPE) | Shows where PSO stands among search methods | M | idea |
| 4.5 | **Tune more dimensions**: colour augmentation (hsv_h/s/v), translate, loss gains (box, cls, dfl), image size | The 6 tuned settings left little room to beat the defaults | M | idea |
| 4.6 | **Tune for the target domain**: use the real-world set (1.4) as the fitness validation set | The search optimised for dataset-style images, not real use | M | idea |

## 5. Deployment

| # | Improvement | Problem it addresses | Effort | Status |
|---|---|---|---|---|
| 5.1 | **Host the demo online** (e.g. Streamlit Community Cloud; the weights are already in `app/models/`) | The demo only runs on a laptop | S | idea |
| 5.2 | **Live video** from a webcam or phone camera | Real sorting is continuous, not photo by photo | M | idea |
| 5.3 | **Edge export** (ONNX or TFLite) and a test on a Raspberry Pi or phone | Real bins need cheap, offline hardware | M | idea |

## Suggested order

1. **1.1 ensemble** and **1.3 confidence**: immediate demo improvement, no training.
2. **1.4 real-world test set**: without it, no later change can be measured on real photos.
3. **2.1 TACO** + **2.2 PlusYaml** + **3.1 stronger model**: the main fix for the domain gap (one Kaggle training run per variant).
4. **4.x**: research extensions to the PSO study.
