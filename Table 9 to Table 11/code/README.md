# Table 9 to Table 11 Code

This directory contains the ASD-only interpretability evaluation used for Tables
9 through 11.

## Files

- `ASD evaluation code.py`: the active implementation. It evaluates LayerCAM,
  Grad-CAM++, and EigenCAM, computes centroid drift distance for each image, and
  reports per-group averages.

## Configuration

Set `MODEL_PATH`, `ORIG_IMG_DIR`, and each entry in `ADV_GROUPS` before running.
The current configuration uses a six-class DenseNet-121 model.

## Run

```powershell
python "ASD evaluation code.py"
```

Install `pytorch-grad-cam` before execution.

