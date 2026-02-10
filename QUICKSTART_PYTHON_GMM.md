# Quick Start Guide - Python GMM Segmentation

This guide will get you up and running with the Python GMM segmentation in 5 minutes.

## Installation

1. **Install dependencies:**
   ```bash
   pip install numpy SimpleITK
   ```

2. **Get the script:**
   The script is in this repository: `gmm_dicom_segmentation.py`

## Basic Usage

### Command Line

```bash
# Most basic usage - segment with 3 clusters
python gmm_dicom_segmentation.py /path/to/dicom/directory -o output.nii

# Specify number of clusters
python gmm_dicom_segmentation.py /path/to/dicom/directory -o output.nii --clusters 5

# Mark specific clusters as foreground
python gmm_dicom_segmentation.py /path/to/dicom/directory -o output.nii \
    --clusters 4 --foreground 1 2 3
```

### Python Code

```python
from gmm_dicom_segmentation import GMMDicomSegmentation

# Create segmenter
segmenter = GMMDicomSegmentation(n_clusters=3, random_seed=42)

# Run complete pipeline
speed_map = segmenter.run('/path/to/dicom/directory')

# Save results
segmenter.save_results(speed_map, 'output.nii')

# Print cluster information
segmenter.print_cluster_info()
```

## What You Get

The script outputs a **speed/probability map** where:
- **Positive values** = Higher probability of foreground tissue
- **Negative values** = Higher probability of background tissue
- **Range**: -32767 to +32767

## Creating Binary Segmentation

```python
import SimpleITK as sitk
import numpy as np

# Load the speed map
speed_image = sitk.ReadImage('output.nii')
speed_array = sitk.GetArrayFromImage(speed_image)

# Threshold at 0 to get binary mask
binary_mask = (speed_array > 0).astype(np.uint8)

# Save binary mask
mask_image = sitk.GetImageFromArray(binary_mask)
sitk.WriteImage(mask_image, 'binary_segmentation.nii')
```

## Testing the Installation

Run the test suite to verify everything works:

```bash
python test_gmm_segmentation.py
```

You should see:
```
============================================================
GMM DICOM Segmentation - Test Suite
============================================================

Testing Gaussian class...
  ✓ Gaussian test passed

... (more tests)

============================================================
All tests passed! ✓
============================================================
```

## Running Examples

Try the included examples:

```bash
python examples_gmm_segmentation.py
```

This will demonstrate 5 different usage patterns with synthetic data.

## Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--clusters` | Number of tissue types | 3 |
| `--samples` | Voxels sampled for training | 10000 |
| `--em-iterations` | EM optimization iterations | 10 |
| `--foreground` | Which clusters are foreground | all |
| `--seed` | Random seed (reproducibility) | random |

## Typical Workflows

### Brain MRI Segmentation (3 tissues)
```bash
python gmm_dicom_segmentation.py /data/brain_t1 -o brain_seg.nii \
    --clusters 3 --samples 15000
```

### CT Scan with Background Removal
```bash
# First run to see cluster statistics
python gmm_dicom_segmentation.py /data/ct_scan -o ct_temp.nii --clusters 4

# Then rerun with specific foreground clusters (e.g., 1, 2, 3)
python gmm_dicom_segmentation.py /data/ct_scan -o ct_seg.nii \
    --clusters 4 --foreground 1 2 3
```

### High-Quality Segmentation
```bash
python gmm_dicom_segmentation.py /data/mri -o high_quality.nii \
    --clusters 3 \
    --samples 20000 \
    --em-iterations 20 \
    --seed 42
```

## Troubleshooting

**Problem**: "No module named 'numpy'" or "No module named 'SimpleITK'"  
**Solution**: `pip install numpy SimpleITK`

**Problem**: "No DICOM files found"  
**Solution**: Ensure the directory contains DICOM files (*.dcm or no extension)

**Problem**: Results vary between runs  
**Solution**: Use `--seed 42` for reproducibility

**Problem**: "Too few valid samples"  
**Solution**: Image may have many NaN/Inf values. Try reducing `--clusters`

## Next Steps

1. **Read the full documentation**: `GMM_PYTHON_README.md`
2. **Understand the algorithm**: `CLUSTER_SEGMENTATION_WORKFLOW.md`
3. **See implementation details**: `PYTHON_IMPLEMENTATION_SUMMARY.md`
4. **View the flowchart**: `CLUSTER_SEGMENTATION_DIAGRAM.txt`

## Getting Help

Run with `--help` to see all options:
```bash
python gmm_dicom_segmentation.py --help
```

Enable verbose logging:
```bash
python gmm_dicom_segmentation.py /path/to/dicom -o output.nii --verbose
```

## Performance Tips

- **Small images** (< 1M voxels): Use default settings
- **Large images** (> 10M voxels): Consider reducing `--samples` to 5000
- **Multiple runs**: Set `--seed` for consistency
- **Faster processing**: Reduce `--em-iterations` to 5

## Integration with ITK-SNAP

After generating the speed map with Python:

1. Open ITK-SNAP
2. Load your DICOM as main image
3. Snake → Load Preprocessing Image → Select your output.nii
4. (Optional) Continue with active contour evolution in ITK-SNAP

---

**That's it!** You're ready to use automatic GMM segmentation on your DICOM images.
