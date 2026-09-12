# Table 1 and Table 2 Code

This directory contains the adversarial attack implementations and the scripts
used to evaluate attack success rates for Tables 1 and 2.

## Attack Success-Rate Evaluation

- `evaluate the attack success rate of attacks against the densenet121 model.py`
- `evaluate the attack success rate of attacks against the resnet50 model.py`
- `evaluate the attack success rate of attacks against the vitb16 model.py`

These scripts compare predictions on original and adversarial image folders,
load the corresponding class-index file and model weights, and print the attack
success rate.

## Attack Generation

The `FGSM`, `MI-FGSM`, and `PGD` directories contain targeted and untargeted
example generators. Each method includes an experiment-specific entry script
plus the supporting `attack.py`, `MIFGSM.py`, or `PGD.py` module.

## Run

Generate the adversarial images first, then update the paths in an evaluation
script and run it:

```powershell
python "evaluate the attack success rate of attacks against the densenet121 model.py"
```

The image filename for the FGSM scripts preserves the original experiment name;
check the configured output path if the filename and target mode appear
inconsistent.

