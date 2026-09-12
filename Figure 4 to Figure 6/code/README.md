# Figure 4 to Figure 6 Code

This directory contains the plotting scripts used to generate the ASD comparison
charts for three explanation methods.

## Files

- `comparison diagram of ASD explained by layercam.py`
- `comparison diagram of ASD explained by gradcam++.py`
- `comparison diagram of ASD explained by eigencam.py`

Each script stores the data in a nested dictionary indexed by attack type,
training type, and model, then draws a 2-by-3 comparison figure.

## Output

Each script writes a PDF to the `ASD_comparison_charts` directory configured at
the end of the file. Update the output path before running.

## Run

```powershell
python "comparison diagram of ASD explained by layercam.py"
python "comparison diagram of ASD explained by gradcam++.py"
python "comparison diagram of ASD explained by eigencam.py"
```

