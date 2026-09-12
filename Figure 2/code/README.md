# Figure 2 Code

This directory contains the training scripts used for the Figure 2 experiments.

## Files

- `densenet121 training code.py`: trains a six-class DenseNet-121 model and saves
  the best checkpoints, log file, and loss/accuracy curves.
- `resnet50 training code.py`: trains a six-class ResNet-50 model with the same
  logging and checkpoint workflow.
- `vitb16 training code.py`: trains a six-class ViT-B/16 model and saves the
  training history and best model.

## Configuration

Each script defines its own dataset paths, pretrained weight paths, checkpoint
directory, batch size, number of epochs, and learning rate. Update these values
before running.

## Run

```powershell
python "densenet121 training code.py"
python "resnet50 training code.py"
python "vitb16 training code.py"
```

The scripts expect the selected model weights and image data to exist at the
configured absolute paths.

