# SwarmSort

Real-time waste detection for segregation using a YOLOv8n detector whose training hyperparameters are tuned by Particle Swarm Optimization (PSO). Soft Computing course project (Unit V: swarm intelligence).

The experiment compares three arms at equal search budget: default hyperparameters, random search, and PSO.

## Dataset

[Garbage Detection – 6 Waste Categories](https://www.kaggle.com/datasets/viswaprakash1990/garbage-detection) (CC BY 4.0): 10,464 images with YOLO bounding boxes in 6 classes (biodegradable, cardboard, glass, metal, paper, plastic).

The dataset's own train/valid/test folders are split **by class**: valid has no paper or plastic images, and test has no glass, biodegradable or cardboard images. This project uses a stratified 70/20/10 re-split instead (seed 42), stored in [configs/split.csv](configs/split.csv). Near-duplicate images are kept in the same split, so no copy of an image leaks across splits. Two images count as copies if they have the same bytes, come from the same Roboflow source image, have 256-bit dHashes at most 10 bits apart, or have dHashes at most 40 bits apart and 64x64 grayscale thumbnails with a Pearson correlation of at least 0.90 (this catches shifted, watermarked and re-coloured copies). The looser rule also groups some look-alike bottles on white backgrounds, which only keeps them in the same split. The dataset files are never modified.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

Download the dataset from Kaggle and extract it into the project folder (it is git-ignored).

## Phase 1: dataset verification

```bash
.venv/Scripts/python src/verify_dataset.py --root "GARBAGE CLASSIFICATION"
```

This reuses `configs/split.csv`, runs the quality checks, and writes:

- `configs/data.yaml` and `configs/lists/`: the Ultralytics dataset config for this machine
- `results/dataset_report.json`: split statistics and check results
- `results/plots/label_check/`: label-visualization grids and near-duplicate pairs

On Kaggle, point `--root` at `/kaggle/input` and the dataset folder is found automatically. The split file is read from and written to the `--configs` directory (default `configs/`). If it is missing there, the default run (`--split auto`) stops with an error instead of falling back to the dataset's own folders; use `--split provided` to check those folders on purpose. `--split new` regenerates the file.

`configs/split.csv` is the authoritative split. Near-duplicate detection depends on the Pillow version and its resize filter, so regenerating the split on another machine (such as Kaggle) can group images differently and produce a different split. Always reuse the committed file (the default); do not run `--split new` there.

## Phase 2: baseline and proxy-validity check

Every training uses the images' native size (imgsz 416), explicit SGD, seed 42, deterministic mode, batch 16 and pretrained `yolov8n.pt`. Mosaic closing (`close_mosaic`, 10%) and warmup (`warmup_epochs`, 3%) scale with the number of epochs, so short and long runs follow the same schedule. Two Ultralytics defaults are avoided on purpose: `optimizer=auto` ignores `lr0` and `momentum`, and `fraction` keeps the first N sorted images instead of a random sample.

```bash
.venv/Scripts/python src/train_final.py   # Arm A: defaults, 100 epochs -> results/baseline.json
.venv/Scripts/python src/training.py      # stratified 40% train subset -> configs/data_proxy.yaml
.venv/Scripts/python src/proxy_check.py   # -> results/proxy_check.json
```

The proxy check tests whether a cheap training ranks hyperparameter settings the same way a longer one does. Seven Latin-hypercube configurations plus the defaults are each trained twice: as a proxy (12 epochs on the 40% subset) and as a medium run (40 epochs on the full train split). The proxy is accepted if the Spearman correlation of their validation mAP@50 is at least 0.6. With 8 points this is a practical check, not a significance test, so the p-value is reported too. If it fails, rerun with `--proxy-epochs 20`; the medium runs are reused.

The log is saved after every finished run, and a restart skips finished runs, so a dropped session loses at most one run. `--smoke` runs a tiny version of the pipeline on CPU and writes to `results/smoke_*.json` instead of the real results.

GPU runs use the Kaggle notebooks in [notebooks/](notebooks/): `phase2a_baseline.ipynb` (about 1 hour) and `phase2b_proxy_check.ipynb` (about 3 hours). Use the same accelerator type for every phase.

## Phase 3: PSO and search

`src/pso.py` is a hand-written particle swarm optimizer (standard inertia-weight PSO, about 70 lines, numpy only). Each particle is one hyperparameter vector in search units (see `src/search_space.py`). It remembers the best position it has visited (pbest), and the swarm remembers the best position any particle has visited (gbest). Each update step is:

```
v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
x = x + v
```

`r1` and `r2` are uniform random numbers in [0, 1], drawn for every particle and dimension. `c1 = c2 = 1.5`, and the inertia `w` falls linearly from 0.9 to 0.4 over the update steps. Velocities are limited to ±20% of each dimension's range. Positions are clipped to the bounds, and a clipped dimension's velocity is set to zero. Initial positions are uniform in the bounds and initial velocities are uniform in ±10% of the range. A seeded generator makes every run reproducible. The optimizer uses an ask/tell interface: `ask()` returns the positions to evaluate and `tell(fitnesses)` updates pbest/gbest and moves the swarm. With 8 particles and 4 rounds (the random initial swarm plus 3 update steps), the budget is 32 evaluations, the same as random search.

```bash
.venv/Scripts/python src/pso_benchmark.py                   # Sphere and Rastrigin -> results/plots/pso_benchmarks.png
.venv/Scripts/python -m unittest discover -s tests -v       # PSO and search tests (no training)
.venv/Scripts/python src/search.py --method pso --smoke     # tiny CPU pipeline check -> results/smoke_search_pso.json
```

The benchmark runs the PSO on 6-D Sphere and Rastrigin (20 particles × 50 rounds, 10 seeds) and plots the best value found so far.

`src/search.py` runs one search arm. `--method pso` (seed 42) runs 8 particles × 4 rounds, and `--method random` evaluates 32 uniform samples (seed 123). The fitness of a configuration is the val mAP@50 of one proxy training with exactly the proxy-check settings (40% subset, 12 epochs, imgsz 416). If the proxy check escalated, pass `--proxy-epochs 20`. Every evaluation goes to `results/search_<method>.json`, and the best one to `results/best_<method>.json`, which `src/train_final.py --config` accepts. The log is saved after every evaluation. A restart replays the algorithm from its seed, reuses the logged results and continues with the first missing evaluation. A log made with other settings is refused.

Phase 4 runs both searches on a Kaggle GPU.
