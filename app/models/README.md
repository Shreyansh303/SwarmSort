# Demo app weights

Byte copies of the fully trained YOLOv8n models, committed so a fresh clone can run `app/app.py`. The app picks the generation in its sidebar and runs each model at the image size stored in its checkpoint (`train_args['imgsz']`).

## Generation 1 (studio data, 6 classes): `app/models/`

100 epochs, imgsz 416. Test mAP@50 is from `results/test_results.json`.

- `arm_a.pt`: Arm A (Ultralytics defaults), copy of `results/runs/baseline/weights/best.pt`, sha256 `072a560597f6`, test mAP@50 0.676
- `arm_b.pt`: Arm B (random-search winner), copy of `results/runs/arm_b/weights/best.pt`, sha256 `9efcd33c79b0`, test mAP@50 0.672
- `arm_c.pt`: Arm C (PSO winner), copy of `results/runs/arm_c/weights/best.pt`, sha256 `8d10d238c27b`, test mAP@50 0.668

## Generation 2 (studio + real-world data, 7 classes with OTHER): `app/models/gen2/`

100 epochs, imgsz 640, trained on Kaggle. Each file is a byte copy of the Kaggle output `gen2_arm_{a,b,c}_best.pt` (saved there as `results/gen2/runs/arm_{a,b,c}/weights/best.pt`). The sha256 matches the `sha256_12` recorded for that arm in `results/gen2/test_all_results.json`; test mAP@50 is on the 'all' test set from the same file.

- `arm_a.pt`: Arm A (Ultralytics defaults), from `notebooks/gen2/g2_1_arm_a.ipynb`, sha256 `e885e035f1e2`, test mAP@50 0.479
- `arm_b.pt`: Arm B (random-search winner), from `notebooks/gen2/g2_4_arm_b.ipynb`, sha256 `55d2355a84a4`, test mAP@50 0.497
- `arm_c.pt`: Arm C (PSO winner), from `notebooks/gen2/g2_5_arm_c.ipynb`, sha256 `1f2c6169597c`, test mAP@50 0.500

To check the copies: `sha256sum app/models/gen2/*.pt | cut -c1-12` (`tests/test_app.py` also compares them with the results files).
