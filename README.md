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
