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
