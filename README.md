# Weighted Integrated Gradients

A Python implementation of Weighted Integrated Gradients for interpreting deep neural network predictions on medical imaging data, specifically applied to dementia classification using the OASIS dataset.

## Overview

This project extends the traditional Integrated Gradients method by introducing various weighting functions to emphasize different parts of the integration path between a baseline and the input. The implementation is demonstrated on a CNN model trained to classify MRI brain images for dementia detection.

## Features

- **Multiple Weighting Functions**: Includes 9 different weighting functions for customizing the integration path
- **OASIS Dataset Support**: Pre-configured for dementia classification (4 classes: Non Demented, Very Mild Dementia, Mild Dementia, Moderate Dementia)
- **Advanced Visualization**: Comprehensive visualization tools with morphological cleanup, outlining, and overlay capabilities
- **Random Baseline Integration**: Robust attribution computation using multiple random baselines

## Weighting Functions

The following weighting functions are implemented:

1. **Square Root** (`sqrt_weighting_function`): Emphasizes early-to-mid integration steps
2. **Reciprocal** (`reciprocal_weighting_function`): Heavily weights early steps near the baseline
3. **Linear Late** (`linear_late_weighting_function`): Linearly increases weight toward the input
4. **Linear Early** (`linear_early_weighting_function`): Linearly decreases weight toward the input
5. **Quadratic Early** (`quadratic_early_weighting_function`): Quadratically emphasizes baseline region
6. **Quadratic Late** (`quadratic_late_weighting_function`): Quadratically emphasizes input region
7. **Logarithmic** (`logarithmic_weighting_function`): Sharp rise near the input
8. **Exponential** (`exponential_weighting_function`): Exponential focus near the input
9. **Beta(2,2)** (`beta22_weighting_function`): Emphasizes the middle of the integration path

## Project Structure

```
Weighted-Integrated-Gradients/
├── WeightedIntegratedGradients/     # Core weighted IG implementation
│   ├── __init__.py
│   └── weighted_integrated_gradients.py
├── OasisModel/                      # CNN model for dementia classification
│   ├── oasis_model_utils.py
│   ├── best_oasis_model.pth
│   └── final_oasis_model.pth
├── VisualizationLibrary/           # Attribution visualization tools
│   ├── __init__.py
│   └── visualization_lib.py
├── OasisImages/                    # Sample MRI images
│   ├── Mild Dementia/
│   ├── Moderate Dementia/
│   ├── Non Demented/
│   └── Very mild Dementia/
├── output/                         # Generated visualization outputs
├── ig_oasis.ipynb                 # Main demonstration notebook
└── pyproject.toml                 # Project dependencies
```

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. To install:

```bash
# Clone the repository
git clone https://github.com/malfalasi1017/Weighted-Integrated-Gradients.git
cd Weighted-Integrated-Gradients

# Install dependencies using uv
uv sync
```

### Dependencies

- Python ≥ 3.8
- PyTorch ≥ 2.5.1
- torchvision ≥ 0.20.1
- NumPy ≥ 1.21.0
- Matplotlib ≥ 3.5.0
- Pillow ≥ 8.0.0
- SciPy ≥ 1.7.0
- Jupyter ≥ 1.0.0

## Usage

### Basic Example

```python
import sys
sys.path.append('./OasisModel')
sys.path.append('./WeightedIntegratedGradients')
sys.path.append('./VisualizationLibrary')

from oasis_model_utils import load_oasis_model, load_image
from weighted_integrated_gradients import (
    random_baseline_weighted_integrated_gradients,
    exponential_weighting_function
)
from visualization_lib import Visualize, show_pil_image, pil_image

# Load model
model, label_names, device = load_oasis_model('./OasisModel/best_oasis_model.pth')

# Load and preprocess image
img = load_image('path/to/mri_image.jpg')
# ... preprocessing steps ...

# Generate weighted integrated gradients
attributions = random_baseline_weighted_integrated_gradients(
    img,
    target_label_index,
    predictions_and_gradients,
    steps=50,
    num_random_trials=10,
    weighting_function=exponential_weighting_function
)

# Visualize results
visualization = Visualize(
    attributions, img,
    clip_above_percentile=95,
    clip_below_percentile=60,
    morphological_cleanup=True,
    outlines=True,
    overlay=True
)
show_pil_image(pil_image(visualization))
```

### Jupyter Notebook

Run the main demonstration notebook:

```bash
jupyter notebook ig_oasis.ipynb
```

The notebook includes:
- Model loading and setup
- Comparison of different weighting functions
- Visualization with various post-processing options
- Batch processing examples

## Key Components

### Weighted Integrated Gradients

The core algorithm computes attributions using:

```python
def weighted_integrated_gradients(inp, target_label_index, predictions_and_gradients, 
                                baseline, weighting_function=None, steps=50):
    # Generate interpolated inputs
    scaled_inputs = [baseline + (float(i)/steps)*(inp-baseline) for i in range(0, steps+1)]
    
    # Get predictions and gradients
    predictions, grads = predictions_and_gradients(scaled_inputs, target_label_index)
    
    # Apply weighting function if provided
    if weighting_function is not None:
        weights = np.array([weighting_function(k/steps) for k in range(1, steps+1)])
        # Normalize weights and apply
        weights = weights / np.sum(weights)
        # ... weighted averaging ...
    
    return weighted_integrated_gradients, predictions
```

### Visualization Features

- **Clipping**: Threshold attributions by percentile
- **Morphological Cleanup**: Remove noise using morphological operations
- **Outlines**: Extract and highlight important regions
- **Overlays**: Combine attributions with original image
- **Distribution Plots**: Analyze attribution distributions

## Applications

This implementation is particularly useful for:

- **Medical Image Analysis**: Understanding CNN decisions on medical images
- **Model Interpretability**: Comparing different attribution methods
- **Research**: Investigating the effects of different weighting schemes
- **Debugging**: Identifying model biases or artifacts

## Medical Use Case: Dementia Classification

The included OASIS model classifies MRI brain images into four categories:
- Non Demented
- Very Mild Dementia  
- Mild Dementia
- Moderate Dementia

The weighted integrated gradients help identify which brain regions the model focuses on for each prediction, providing insights into the decision-making process.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bug reports and feature requests.

## Acknowledgments

- Based on the original Integrated Gradients method by Sundararajan et al.
- OASIS dataset for providing the medical imaging data
- PyTorch community for the deep learning framework
