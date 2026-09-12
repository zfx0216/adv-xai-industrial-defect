# Figure 7 to Figure 9 Code

This directory contains the explainability scripts used for the qualitative
visualizations in Figures 7 to 9.

## Subdirectories

- `layer-CAM`: LayerCAM visualization.
- `grad-CAM++`: Grad-CAM++ visualization.
- `eign-CAM`: EigenCAM visualization. The directory name preserves the original
  experiment naming.

Each subdirectory contains:

- `Generate evaluation images.py`: prepares or exports the images used for visual
  evaluation.
- `saliency map.py`: loads a trained DenseNet-121 model, computes the explanation
  map, overlays it on the input image, and saves the result.

## Configuration

Update the class-index path, model-weight path, input image directory, and output
directory at the top of each script.

## Run

Run the generation script first when preparing a new evaluation set, then run
the saliency script:

```powershell
cd "layer-CAM"
python "Generate evaluation images.py"
python "saliency map.py"
```

The same workflow applies to the `grad-CAM++` and `eign-CAM` directories.

