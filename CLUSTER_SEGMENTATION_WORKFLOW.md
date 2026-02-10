# Cluster-Based Segmentation from 3D DICOM Images in ITK-SNAP

## Overview

ITK-SNAP implements cluster-based segmentation from 3D DICOM images using a **Gaussian Mixture Model (GMM)** approach combined with **Expectation-Maximization (EM)** algorithm for image segmentation. This document explains the complete workflow from loading DICOM files to generating segmented regions.

## Architecture Components

### 1. DICOM Loading (`Logic/ImageWrapper/GuidedNativeImageIO`)

The DICOM loading mechanism supports multiple DICOM formats:
- **FORMAT_DICOM_DIR**: Directory containing multiple DICOM files (3D volumes)
- **FORMAT_DICOM_DIR_4DCTA**: 4D CTA DICOM series
- **FORMAT_DICOM_FILE**: Single DICOM file
- **FORMAT_ECHO_CARTESIAN_DICOM**: Echocardiography Cartesian DICOM

**Key Classes:**
- `GuidedNativeImageIO`: Main class for loading DICOM files
- `MultiFrameDicomSeriesSorter`: Sorts DICOM series for 4D data
- `ExtendedGDCMSerieHelper`: Helper for GDCM (Grassroots DICOM) library

**Workflow:**
1. Parse DICOM directory to identify series
2. Group files by Series ID
3. Sort files by Image Position Patient (IPP)
4. Load pixel data into ITK image structures

### 2. Clustering Core (`Logic/Preprocessing/GMM/`)

The clustering implementation uses Gaussian Mixture Models with the following components:

#### 2.1 UnsupervisedClustering (`UnsupervisedClustering.h/cxx`)

Main orchestrator for the clustering process.

**Key Methods:**
- `SetDataSource(SNAPImageData*)`: Sets the image data to be clustered
- `SetNumberOfClusters(int)`: Configures number of clusters (default: 3)
- `SetNumberOfSamples(int)`: Sets number of samples for training (default: 10,000 or total voxels if less)
- `InitializeClusters()`: Initializes clustering using K-means++
- `Iterate()`: Performs one EM iteration to refine clusters

**Sampling Strategy:**
```cpp
// Random sampling from the image
- Samples up to 10,000 voxels (or all if image has fewer)
- Collects samples from main image and all overlay layers
- Stores "center samples" from central 60% of image for relevance sorting
- Skips non-finite values (NaN, Inf)
```

#### 2.2 KMeansPlusPlus (`KMeansPlusPlus.h/cxx`)

Initialization algorithm for cluster centers.

**Purpose:**
- Provides smart initialization for EM algorithm
- Uses k-means++ algorithm for better initial cluster centers
- Faster convergence compared to random initialization

**Algorithm:**
1. Randomly select first cluster center
2. For each subsequent center:
   - Calculate distance of each point to nearest existing center
   - Select new center with probability proportional to squared distance
3. Iterate until k centers are selected

#### 2.3 EMGaussianMixtures (`EMGaussianMixtures.h/cxx`)

Implements the Expectation-Maximization algorithm.

**EM Algorithm Steps:**

**E-Step (Expectation):**
- For each sample, compute posterior probability of belonging to each cluster
- Uses current Gaussian parameters (mean, covariance, weight)

**M-Step (Maximization):**
- Update Gaussian parameters based on weighted samples
- Recalculate means, covariances, and mixture weights

**Iterations:**
- Default: 10 iterations
- Can be called multiple times via `UpdateOnce()`

#### 2.4 GaussianMixtureModel (`GaussianMixtureModel.h/cxx`)

Data structure representing the mixture model.

**Components:**
- **Gaussians**: Vector of multivariate Gaussian distributions
- **Weights**: Mixture weights (sum to 1.0)
- **Foreground State**: Boolean flags indicating if cluster represents foreground/background

**Key Operations:**
- `EvaluateLogPDF(index, x)`: Evaluate log probability density at point x for cluster index
- `GetMean(index)`: Get mean vector for cluster
- `GetCovariance(index)`: Get covariance matrix for cluster
- `SetForeground(index)` / `SetBackground(index)`: Mark cluster as foreground/background

#### 2.5 Gaussian (`Gaussian.h/cxx`)

Represents a single multivariate Gaussian distribution.

**Mathematical Model:**
```
PDF(x) = (1/((2π)^(d/2) * |Σ|^(1/2))) * exp(-0.5 * (x-μ)ᵀΣ⁻¹(x-μ))

where:
- x: input vector (d dimensions)
- μ: mean vector
- Σ: covariance matrix
- d: dimensionality
```

### 3. Image Classification (`Logic/Preprocessing/GMMClassifyImageFilter`)

Applies the trained GMM to classify entire image volumes.

**Template Class:**
```cpp
template <class TInputImage, class TInputVectorImage, class TOutputImage>
class GMMClassifyImageFilter
```

**Key Methods:**
- `AddScalarImage(image)`: Add scalar component to classification
- `AddVectorImage(image)`: Add vector (multi-component) image
- `SetMixtureModel(model)`: Set trained GMM
- `DynamicThreadedGenerateData()`: Multi-threaded classification

**Classification Process:**
1. For each voxel in the image:
   - Extract feature vector (from all input images/components)
   - Evaluate log PDF for each Gaussian cluster
   - Compute posterior probability for each cluster
   - Calculate weighted probability difference (foreground - background)
   - Store result as speed/probability map

**Output:**
- Probability map (speed image) with values in range [-32767, 32767]
- Positive values indicate foreground probability
- Negative values indicate background probability

### 4. Integration Layer (`Logic/Framework/IRISApplication`)

Manages the clustering engine at the application level.

**Key Components:**
- `m_ClusteringEngine`: Instance of UnsupervisedClustering
- `m_GMMPreviewWrapper`: Wrapper for preview of GMM results
- `m_LastUsedMixtureModel`: Cache of last trained model

**Workflow Integration:**
```cpp
// Initialize clustering
m_ClusteringEngine = UnsupervisedClustering::New();
m_ClusteringEngine->SetDataSource(m_SNAPImageData);
m_ClusteringEngine->InitializeClusters();

// Refine with EM iterations
m_ClusteringEngine->Iterate();

// Apply to full image
GMMClassifyImageFilter->SetMixtureModel(m_ClusteringEngine->GetMixtureModel());
```

### 5. GUI Integration (`GUI/Model/SnakeWizardModel`)

User interface model for interactive segmentation workflow.

**User-Adjustable Parameters:**
- **Number of Clusters**: How many tissue types to identify (default: 3)
- **Number of Samples**: Training samples for GMM (default: 10,000)
- **Foreground Cluster**: Which cluster(s) represent target tissue
- **Cluster Weights**: Manual adjustment of cluster importance
- **Cluster Means**: Fine-tuning of cluster centers

**Interactive Features:**
- Real-time preview of clustering results
- Manual cluster refinement
- Iterative EM optimization
- Visualization of cluster distributions

## Complete Workflow: DICOM to Segmentation

### Step 1: Load 3D DICOM Data

```
DICOM Directory
    ↓ (GuidedNativeImageIO)
Series Identification
    ↓ (MultiFrameDicomSeriesSorter)
Sorted DICOM Files
    ↓ (GDCM Reader)
3D Image Volume (ITK Image)
```

### Step 2: Sample Image Data

```
3D Image Volume
    ↓ (UnsupervisedClustering::SampleDataSource)
Random Voxel Sampling
    ↓
Feature Matrix (N samples × M components)
    - N ≤ 10,000 samples
    - M = number of image components (intensity channels)
```

### Step 3: Initialize Clusters

```
Feature Matrix
    ↓ (KMeansPlusPlus)
Initial Cluster Centers
    ↓
Initial Gaussian Parameters
    - Means
    - Covariances
    - Weights (uniform)
```

### Step 4: Refine Clusters (EM Iterations)

```
Current GMM Parameters
    ↓ (EMGaussianMixtures::UpdateOnce)
E-Step: Compute Posteriors
    ↓
M-Step: Update Parameters
    ↓
Refined GMM Parameters
    ↓ (Repeat 10× or until convergence)
Final GMM Model
```

### Step 5: Classify Full Image

```
Final GMM Model + Full 3D Image
    ↓ (GMMClassifyImageFilter)
For Each Voxel:
    - Extract features
    - Compute cluster posteriors
    - Weight by foreground/background
    ↓
Probability/Speed Map
    - Range: [-32767, 32767]
    - Positive = Foreground probability
    - Negative = Background probability
```

### Step 6: Segmentation Refinement

```
Probability Map
    ↓ (Optional: Snake/Level Set Evolution)
Final Segmentation
    ↓ (Thresholding)
Binary Segmentation Mask
```

## Mathematical Details

### Gaussian Mixture Model

The probability density function of a GMM with K components:

```
p(x) = Σᵢ₌₁ᴷ wᵢ · 𝒩(x | μᵢ, Σᵢ)

where:
- wᵢ: weight of component i (Σwᵢ = 1)
- 𝒩(x | μᵢ, Σᵢ): Gaussian PDF with mean μᵢ and covariance Σᵢ
```

### Posterior Probability Computation

For a voxel with feature vector x:

```
P(cluster k | x) = (wₖ · 𝒩(x | μₖ, Σₖ)) / (Σⱼ₌₁ᴷ wⱼ · 𝒩(x | μⱼ, Σⱼ))
```

Implementation uses log-space for numerical stability:

```cpp
log_pdf[k] = log(𝒩(x | μₖ, Σₖ))
log_w[k] = log(wₖ)

posterior[k] = ComputePosterior(log_pdf, w, log_w, k)
```

### Final Speed Value

```
speed(x) = Σₖ P(k|x) · fₖ · 0x7fff

where:
- fₖ = +1 if cluster k is foreground
- fₖ = -1 if cluster k is background
- 0x7fff = 32767 (normalization factor)
```

## Multi-Component Support

The system supports multi-component images:

**Examples:**
1. **Single T1 MRI**: 1 component
2. **RGB Image**: 3 components  
3. **Multi-sequence MRI**: T1 + T2 + FLAIR = multiple components
4. **Main + Overlays**: Main image components + overlay image components

Each voxel's feature vector concatenates all components across all loaded images.

## Cluster Relevance Sorting

After initialization, clusters are sorted by "relevance":

**Purpose:**
- Present most relevant clusters first to user
- "Relevant" = high posterior probability in central image region

**Algorithm:**
1. Collect samples from central 60% of image (up to 400 samples)
2. For each cluster k, compute:
   ```
   relevance[k] = -Σ P(k | x_center_samples)
   ```
3. Sort clusters by relevance (ascending, so most relevant first)
4. Reorder GMM parameters accordingly

This ensures that tissue types in the region of interest appear first in the cluster list.

## Performance Optimizations

1. **Sampling**: Only 10,000 voxels used for training (vs. millions in full image)
2. **Multi-threading**: GMMClassifyImageFilter uses ITK's multi-threading
3. **Log-space arithmetic**: Prevents numerical underflow in probability calculations
4. **Smart initialization**: K-means++ provides better starting point than random

## Key Files Reference

| Component | Files |
|-----------|-------|
| DICOM Loading | `Logic/ImageWrapper/GuidedNativeImageIO.{h,cxx}` |
|  | `Common/MultiFrameDicomSeriesSorter.h` |
|  | `Common/ExtendedGDCMSerieHelper.cxx` |
| Clustering Core | `Logic/Preprocessing/GMM/UnsupervisedClustering.{h,cxx}` |
|  | `Logic/Preprocessing/GMM/GaussianMixtureModel.{h,cxx}` |
|  | `Logic/Preprocessing/GMM/EMGaussianMixtures.{h,cxx}` |
|  | `Logic/Preprocessing/GMM/KMeansPlusPlus.{h,cxx}` |
|  | `Logic/Preprocessing/GMM/Gaussian.{h,cxx}` |
| Image Classification | `Logic/Preprocessing/GMMClassifyImageFilter.{h,txx}` |
| Application Integration | `Logic/Framework/IRISApplication.{h,cxx}` |
| GUI/Model | `GUI/Model/SnakeWizardModel.{h,cxx}` |
| GUI/View | `GUI/Qt/Windows/SpeedImageDialog.{h,cxx}` |
|  | `GUI/Renderer/GMMRenderer.cxx` |

## Usage Example (Conceptual)

```cpp
// 1. Load DICOM series
GuidedNativeImageIO::Pointer reader = GuidedNativeImageIO::New();
reader->SetFileFormat(FORMAT_DICOM_DIR);
reader->SetFileName("/path/to/dicom/series");
Image3D::Pointer image = reader->ReadImage();

// 2. Create image data source
SNAPImageData::Pointer imageData = SNAPImageData::New();
imageData->SetMainImage(image);

// 3. Initialize clustering
UnsupervisedClustering::Pointer clustering = UnsupervisedClustering::New();
clustering->SetDataSource(imageData);
clustering->SetNumberOfClusters(3);  // e.g., background, gray matter, white matter
clustering->InitializeClusters();

// 4. Refine with EM iterations
for(int i = 0; i < 10; i++)
  clustering->Iterate();

// 5. Classify full image
GMMClassifyImageFilter::Pointer classifier = GMMClassifyImageFilter::New();
classifier->AddScalarImage(image);
classifier->SetMixtureModel(clustering->GetMixtureModel());
classifier->Update();

// 6. Get probability map
SpeedImage::Pointer speedMap = classifier->GetOutput();

// 7. Threshold to get segmentation
// (speedMap > 0) = foreground regions
```

## Advanced Features

### Foreground/Background Assignment

Users can interactively mark clusters as foreground or background:
- **Foreground clusters** contribute positively to speed map
- **Background clusters** contribute negatively
- Allows complex multi-class segmentation

### Manual Parameter Tuning

Advanced users can adjust:
- Cluster means (shift cluster centers)
- Cluster covariances (change cluster shape)
- Cluster weights (change relative importance)

Changes are reflected immediately in the speed map preview.

### Multi-Modal Segmentation

The system can segment based on multiple image modalities simultaneously:
- Combine T1, T2, FLAIR MRI sequences
- Each voxel represented by vector [T1_intensity, T2_intensity, FLAIR_intensity]
- GMM clusters in multi-dimensional feature space

## Conclusion

ITK-SNAP's cluster-based segmentation provides a powerful, semi-automatic approach to 3D medical image segmentation:

1. **Automatic tissue classification** via Gaussian Mixture Models
2. **Efficient sampling** for fast training on large 3D volumes  
3. **Interactive refinement** through user-adjustable parameters
4. **Multi-modal support** for complex segmentation tasks
5. **Seamless DICOM integration** for clinical workflows

The combination of unsupervised learning (GMM-EM) with user guidance (foreground/background assignment, parameter tuning) creates a flexible framework suitable for a wide range of medical imaging applications.
