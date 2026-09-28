# Local model card

- Original local checkpoint: `yolov11n_best.pt`.
- Portfolio path: `models/fire-smoke.pt` (not tracked).
- SHA-256: `4071eda4c380e04be1e2000fded0ca96d7d8371587c7680ce4bd177ea4b9f904`.
- Size: 5447507 bytes.
- Observed labels: fire, smoke.
- Runtime validated: Ultralytics 8.3.15, CPU.
- Intended use: image-inspection research prototype.
- Training provenance: existing user-supplied research artifact; full data lineage and redistribution rights not independently verified.
- Verified behavior: one-image inference, not a benchmark.
- Important limitation: missed fires, false positives, domain shift and duplicate/overlapping boxes require separate evaluation.

Ultralytics code retains its applicable upstream license. The project does not assign a new license to pretrained weights or datasets. A future model release should document dataset sources, splits, training configuration and evaluation before publishing the weights.
