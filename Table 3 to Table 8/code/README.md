# Table 3 to Table 8 Code

This directory contains the combined interpretability evaluation used for Tables
3 through 8.

## File

`evaluation code based on three interpretability methods.py` evaluates LayerCAM,
Grad-CAM++, and EigenCAM on original and adversarial images. It computes:

- IoU using the 80th percentile as the threshold.
- SSIM between explanation maps.
- ASD, the centroid drift distance between explanation maps.

The script prints per-image metrics and group-level averages for the configured
adversarial groups.

## Configuration

Set `MODEL_PATH`, `ORIG_IMG_DIR`, and each entry in `ADV_GROUPS` before running.
The model must match the six-class classification setup used by the saved
weights and class-index file.

## Run

```powershell
python "evaluation code based on three interpretability methods.py"
```

Install `pytorch-grad-cam` before execution.

