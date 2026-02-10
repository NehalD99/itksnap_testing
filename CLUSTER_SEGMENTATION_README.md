# Cluster-Based Segmentation Documentation

## Quick Navigation

This repository contains ITK-SNAP, a medical image segmentation tool that implements cluster-based segmentation from 3D DICOM images using Gaussian Mixture Models (GMM).

### Documentation Files

1. **[CLUSTER_SEGMENTATION_WORKFLOW.md](./CLUSTER_SEGMENTATION_WORKFLOW.md)** - Comprehensive written documentation
   - Detailed explanation of all components
   - Mathematical formulas and algorithms
   - Code architecture and design patterns
   - Usage examples and integration points

2. **[CLUSTER_SEGMENTATION_DIAGRAM.txt](./CLUSTER_SEGMENTATION_DIAGRAM.txt)** - Visual flowchart and diagrams
   - Complete workflow from DICOM to segmentation
   - Phase-by-phase breakdown with details
   - Data structure visualizations
   - Performance optimization notes

## Quick Summary

### How Cluster-Based Segmentation Works

**Input:** 3D DICOM image series  
**Output:** Segmented regions (probability map or binary mask)

**Process:**
1. **Load DICOM** → Parse and sort DICOM files into 3D volume
2. **Sample Data** → Extract ~10,000 random voxels from image
3. **Initialize Clusters** → Use K-means++ to find initial cluster centers
4. **Refine with EM** → Iteratively optimize Gaussian Mixture Model parameters
5. **Classify Image** → Apply trained GMM to all voxels in parallel
6. **Generate Output** → Produce probability/speed map for segmentation

### Key Components

- **GMM Algorithm**: `Logic/Preprocessing/GMM/`
  - `UnsupervisedClustering` - Main orchestrator
  - `GaussianMixtureModel` - Model representation
  - `EMGaussianMixtures` - EM optimization
  - `KMeansPlusPlus` - Smart initialization

- **Image Processing**: `Logic/Preprocessing/`
  - `GMMClassifyImageFilter` - Applies GMM to full image

- **DICOM Support**: `Logic/ImageWrapper/`
  - `GuidedNativeImageIO` - DICOM loading
  - `MultiFrameDicomSeriesSorter` - Series organization

- **User Interface**: `GUI/Model/`
  - `SnakeWizardModel` - Interactive segmentation workflow

### Technical Highlights

- **Multi-component Support**: Handles multi-modal imaging (T1, T2, FLAIR, etc.)
- **Efficient Sampling**: Trains on subset of data for speed
- **Multi-threaded**: Parallel image classification
- **Numerically Stable**: Log-space arithmetic prevents underflow
- **Interactive Refinement**: User can adjust clusters in real-time

### Mathematical Foundation

**Gaussian Mixture Model:**
```
p(x) = Σ w_k · N(x | μ_k, Σ_k)
```

**EM Algorithm:**
- **E-step**: Compute posterior probabilities (which cluster likely generated each sample)
- **M-step**: Update cluster parameters (mean, covariance, weight) based on posteriors
- **Iterate** until convergence

**Final Classification:**
```
speed(x) = Σ P(k|x) · foreground_factor_k
```
where `foreground_factor` is +1 for foreground clusters, -1 for background

## For Developers

### Building and Running

Follow the standard ITK-SNAP build instructions in [README.md](./README.md)

### Extending the Code

To modify clustering behavior:
1. Adjust parameters in `UnsupervisedClustering` (number of clusters, samples)
2. Modify EM convergence criteria in `EMGaussianMixtures`
3. Customize initialization strategy in `KMeansPlusPlus`
4. Add preprocessing steps in `GMMClassifyImageFilter`

### Testing

The clustering code can be tested with:
- Any 3D DICOM series
- Multi-component images (RGB, multi-sequence MRI)
- Various cluster counts (typically 3-10)

## Citation

If you use ITK-SNAP's cluster-based segmentation in your research, please cite:

> Paul A. Yushkevich, Joseph Piven, Heather Cody Hazlett, Rachel Gimpel Smith,
> Sean Ho, James C. Gee, and Guido Gerig. User-guided 3D active contour
> segmentation of anatomical structures: Significantly improved efficiency and
> reliability. Neuroimage. 2006 Jul 1; 31(3):1116-28.

## Support

For questions about cluster-based segmentation:
- Check the [ITK-SNAP documentation](http://www.itksnap.org/pmwiki/pmwiki.php?n=Documentation.HomePage)
- Post on [ITK-SNAP Users' List](http://www.itksnap.org/pmwiki/pmwiki.php?n=MailingLists)
- Review the comprehensive documentation files in this repository

## License

ITK-SNAP is distributed under the GNU General Public License v3.0. See [COPYING](./COPYING) for details.
