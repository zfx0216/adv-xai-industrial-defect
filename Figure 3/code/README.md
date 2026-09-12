# Figure 3 Code

This directory contains the model-training scripts and the main adversarial
attack implementations used for Figure 3.

## Training

- `densenet121 training code.py`
- `resnet50 training code.py`
- `vitb16 training code.py`

These scripts use the same training, logging, and checkpoint workflow as the
corresponding scripts in `Figure 2/code`.

## Attack Implementations

The attack code is organized by attack method and then by target mode:

- `FGSM/targeted`: FGSM generation for targeted examples.
- `FGSM/untargeted`: FGSM generation for untargeted examples.
- `MI-FGSM/targeted`: MI-FGSM generation for targeted examples.
- `MI-FGSM/untargeted`: MI-FGSM generation for untargeted examples.
- `PGD/targeted`: PGD generation for targeted examples.
- `PGD/untargeted`: PGD generation for untargeted examples.

Each attack directory contains:

- A `*attack Li.py` entry script with the experiment-specific paths and settings.
- An `attack.py` utility module used by the attack implementation.
- `MIFGSM.py` or `PGD.py`, which contains the attack algorithm.

Run the experiment-specific entry script after updating its model, input, and
output paths.

## Run

```powershell
python "FGSM\untargeted\FGSM untargeted attack Li.py"
python "MI-FGSM\untargeted\MI FGSM untargeted attack Li.py"
python "PGD\untargeted\PGD untargeted attack Li.py"
```

Use the scripts in the `targeted` directories for targeted attacks.

