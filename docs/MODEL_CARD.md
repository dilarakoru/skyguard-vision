# Local model card

- Original local checkpoint: `yolov11n_best.pt`.
- Application path: `models/fire-smoke.pt` (not tracked).
- SHA-256: `4071eda4c380e04be1e2000fded0ca96d7d8371587c7680ce4bd177ea4b9f904`.
- Size: 5,447,507 bytes.
- Observed labels: `fire`, `smoke`.
- Runtime validated: Ultralytics 8.3.15, CPU.
- Intended use: image-inspection research prototype.
- Verified behavior: one-image inference, not a benchmark.

## Training-run linkage

The checkpoint hash and byte size exactly match the historical Drive artifact `fire_detection_yolo11/weights/best.pt`.

The saved `args.yaml` for that run records:

- starting model: `yolo11n.pt`;
- dataset path: `/content/fire-and-smoke-detection-2/data.yaml`;
- notebook acquisition record: Roboflow workspace `middle-east-tech-university`, project `fire-and-smoke-detection-hiwia`, version 2;
- 50 epochs;
- batch size 16;
- image size 640;
- AMP enabled;
- seed 0 with deterministic mode enabled;
- validation enabled.

The same notebook log shows this run executing on **NVIDIA A100-SXM4-40GB (40,507 MiB)** with PyTorch 2.6.0 + CUDA 12.4. This establishes the checkpoint-to-run and run-to-compute links. It does **not** by itself establish the complete upstream dataset manifest, image-level attribution or redistribution rights, because that historical dataset is not republished in this repository.

## Limitations

Missed fires, false positives, domain shift and duplicate/overlapping boxes require separate evaluation. The current repository does not claim that the historical training validation metrics are a benchmark for field performance.

Ultralytics code retains its applicable upstream license. The project does not assign a new license to pretrained weights or datasets. A future public model release should include the authorized dataset sources, exact split manifest, training environment and held-out evaluation.

Other historical GPU training and model-development experiments are documented separately in [TRAINING_HISTORY.md](TRAINING_HISTORY.md); their metrics must not be attributed to this checkpoint unless the checkpoint/run linkage is explicitly established.
