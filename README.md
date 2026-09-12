# CAIE Supplementary Materials

This repository contains the source code and image data used by the supplementary
materials for the CAIE paper. The code covers model training, adversarial example
generation, attack success-rate evaluation, interpretability evaluation, and the
figures and tables reported in the paper.

## Contents

| Directory | Description |
| --- | --- |
| `Figure 2/code` | Training scripts for DenseNet-121, ResNet-50, and ViT-B/16. |
| `Figure 3/code` | Training and adversarial attack scripts for FGSM, MI-FGSM, and PGD. |
| `Figure 4 to Figure 6/code` | Scripts that generate the ASD comparison charts. |
| `Figure 7 to Figure 9/code` | LayerCAM, Grad-CAM++, and EigenCAM visualization scripts. |
| `Table 1 and Table 2/code` | Attack generation and attack success-rate evaluation scripts. |
| `Table 3 to Table 8/code` | IoU, SSIM, and ASD evaluation for three explanation methods. |
| `Table 9 to Table 11/code` | ASD-only evaluation for three explanation methods. |
| `data` | MTSD and NEU-DET image datasets separated into training and test sets. |

Each `code` directory contains its own `README.md` with file-level details.

## Data

The `data` directory contains two six-class industrial defect datasets:

- `data/MTSD`: `Blowhole`, `Break`, `Crack`, `Fray`, `Free`, and `Uneven`.
- `data/NEU-DET`: `crazing`, `inclusion`, `patches`, `pitted_surface`,
  `rolled-in_scale`, and `scratches`.

Both datasets use `train` and `test` subdirectories.

## Environment

The scripts target Python 3 and use the following main packages:

- PyTorch and torchvision
- NumPy
- Pillow
- Matplotlib
- tqdm
- pytorch-grad-cam

Install the packages required by the scripts you plan to run. A CUDA-capable GPU
is optional, but training and batch attack generation will be much faster with
one.

## Running the Code

The original experimental scripts contain machine-specific absolute paths. The
Chinese path components have been translated to English in this edition, so the
paths must be updated to match the local locations of the datasets, model
weights, and output directories before execution.

For most experiments:

1. Update the paths near the top of the script.
2. Run the script from its own directory so imported local modules resolve
   correctly.
3. Generate adversarial examples before running the attack success-rate or
   interpretability evaluation scripts.

Example:

```powershell
cd "Figure 2\code"
python "densenet121 training code.py"
```

## Notes

- The scripts preserve the experimental settings used to produce the paper
  results. Some comments and helper interfaces come from the original code.
- All Python source files have been converted to English.
- `Table 9 to Table 11/code/ASD evaluation code using three explanation
  methods.py` is a zero-byte placeholder. The active implementation is
  `Table 9 to Table 11/code/ASD evaluation code.py`.

