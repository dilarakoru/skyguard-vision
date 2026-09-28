# Research utilities

These scripts preserve the useful engineering patterns from earlier SkyGuard fire/smoke experiments without publishing datasets, model weights, credentials or notebook-specific paths.

## GPU training

`train_gpu.py` is a configurable version of the CUDA training setup used in historical YOLO11/YOLO12 experiments.

Example:

```bash
python research/train_gpu.py \
  --data /path/to/data.yaml \
  --weights /path/to/yolo11l.pt \
  --device 0 \
  --epochs 100 \
  --batch 8 \
  --cache
```

The defaults reflect the project’s historical experimentation pattern: 640 px images, SGD, warmup, early stopping and geometric/color augmentation. The script requires local weights and a local dataset YAML; it does not download private data or include API keys.

## Dataset validation

`dataset_validation.py` performs pre-training checks that were also part of the historical notebooks:

- corrupted image detection;
- missing image/label pairs;
- orphan label files;
- invalid YOLO rows and out-of-range normalized coordinates.

Example:

```bash
python research/dataset_validation.py /path/to/dataset \
  --output runs/dataset_validation.csv
```

Expected dataset layout:

```text
dataset/
  train/images
  train/labels
  valid/images
  valid/labels
  test/images
  test/labels
```

These utilities complement the lightweight public inference application. Historical compute environments and recorded experiment metrics are documented in [../docs/TRAINING_HISTORY.md](../docs/TRAINING_HISTORY.md).
