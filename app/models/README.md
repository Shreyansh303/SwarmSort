# Demo app weights

Byte copies of the three fully trained YOLOv8n models (100 epochs, imgsz 416), committed so a fresh clone can run `app/app.py`. Test mAP@50 is from `results/test_results.json`.

- `arm_a.pt`: Arm A (Ultralytics defaults), copy of `results/runs/baseline/weights/best.pt`, sha256 `072a560597f6`, test mAP@50 0.676
- `arm_b.pt`: Arm B (random-search winner), copy of `results/runs/arm_b/weights/best.pt`, sha256 `9efcd33c79b0`, test mAP@50 0.672
- `arm_c.pt`: Arm C (PSO winner), copy of `results/runs/arm_c/weights/best.pt`, sha256 `8d10d238c27b`, test mAP@50 0.668
