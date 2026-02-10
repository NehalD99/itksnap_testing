#!/usr/bin/env python3
"""
GMM-based Automatic Tissue Classification from DICOM Images

This script implements the same cluster-based segmentation algorithm as ITK-SNAP,
but in pure Python without user interaction or snake evolution. It performs automatic
tissue classification using Gaussian Mixture Models (GMM) with Expectation-Maximization (EM).

The algorithm follows these steps:
1. Load DICOM series
2. Random voxel sampling (~10,000 samples)
3. Initialize clusters with K-means++
4. Refine clusters with EM algorithm
5. Classify full image using trained GMM
6. Output probability/speed map

Author: Based on ITK-SNAP's C++ implementation
License: GPL v3.0
"""

import numpy as np
import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Gaussian:
    """
    Represents a single multivariate Gaussian distribution.
    
    Equivalent to ITK-SNAP's Gaussian class.
    """
    
    def __init__(self, dim: int):
        """
        Initialize a Gaussian distribution.
        
        Args:
            dim: Dimensionality of the Gaussian
        """
        self.dim = dim
        self.mean = np.zeros(dim)
        self.covariance = np.eye(dim)
        self.covariance_inv = np.eye(dim)
        self.covariance_det = 1.0
        self.normalization_factor = 1.0
        
    def set_parameters(self, mean: np.ndarray, covariance: np.ndarray):
        """
        Set Gaussian parameters and precompute derived quantities.
        
        Args:
            mean: Mean vector
            covariance: Covariance matrix
        """
        self.mean = mean.copy()
        self.covariance = covariance.copy()
        
        # Add small regularization to prevent singular covariance
        reg = 1e-6 * np.eye(self.dim)
        cov_reg = self.covariance + reg
        
        # Precompute inverse and determinant
        try:
            self.covariance_inv = np.linalg.inv(cov_reg)
            self.covariance_det = np.linalg.det(cov_reg)
        except np.linalg.LinAlgError:
            logger.warning("Singular covariance matrix, using regularized version")
            self.covariance_inv = np.linalg.inv(cov_reg + 0.01 * np.eye(self.dim))
            self.covariance_det = np.linalg.det(cov_reg + 0.01 * np.eye(self.dim))
        
        # Normalization factor: 1 / ((2π)^(d/2) * |Σ|^(1/2))
        self.normalization_factor = 1.0 / (
            np.power(2.0 * np.pi, self.dim / 2.0) * 
            np.sqrt(max(self.covariance_det, 1e-300))
        )
    
    def evaluate_log_pdf(self, x: np.ndarray) -> float:
        """
        Evaluate log probability density function at point x.
        
        Args:
            x: Input vector
            
        Returns:
            Log probability density
        """
        # Compute (x - μ)
        diff = x - self.mean
        
        # Compute -0.5 * (x-μ)ᵀ Σ⁻¹ (x-μ)
        mahalanobis = -0.5 * np.dot(diff, np.dot(self.covariance_inv, diff))
        
        # Add log normalization factor
        log_norm = np.log(max(self.normalization_factor, 1e-300))
        
        return log_norm + mahalanobis


class GaussianMixtureModel:
    """
    Represents a Gaussian Mixture Model.
    
    Equivalent to ITK-SNAP's GaussianMixtureModel class.
    """
    
    def __init__(self, n_components: int, n_features: int):
        """
        Initialize GMM.
        
        Args:
            n_components: Number of Gaussian components (clusters)
            n_features: Dimensionality of each Gaussian
        """
        self.n_components = n_components
        self.n_features = n_features
        self.gaussians = [Gaussian(n_features) for _ in range(n_components)]
        self.weights = np.ones(n_components) / n_components
        self.foreground_state = np.ones(n_components, dtype=bool)  # All foreground by default
        
    def set_gaussian(self, index: int, mean: np.ndarray, covariance: np.ndarray):
        """Set parameters for a specific Gaussian component."""
        self.gaussians[index].set_parameters(mean, covariance)
        
    def set_weight(self, index: int, weight: float):
        """Set weight for a specific component."""
        self.weights[index] = weight
        
    def get_mean(self, index: int) -> np.ndarray:
        """Get mean of a specific component."""
        return self.gaussians[index].mean
        
    def get_covariance(self, index: int) -> np.ndarray:
        """Get covariance of a specific component."""
        return self.gaussians[index].covariance
        
    def get_weight(self, index: int) -> float:
        """Get weight of a specific component."""
        return self.weights[index]
        
    def evaluate_log_pdf(self, index: int, x: np.ndarray) -> float:
        """Evaluate log PDF of a specific component at point x."""
        return self.gaussians[index].evaluate_log_pdf(x)
        
    def set_foreground(self, index: int):
        """Mark a component as foreground."""
        self.foreground_state[index] = True
        
    def set_background(self, index: int):
        """Mark a component as background."""
        self.foreground_state[index] = False
        
    def is_foreground(self, index: int) -> bool:
        """Check if a component is foreground."""
        return self.foreground_state[index]


class KMeansPlusPlus:
    """
    K-means++ initialization algorithm.
    
    Equivalent to ITK-SNAP's KMeansPlusPlus class.
    """
    
    def __init__(self, data: np.ndarray, n_clusters: int):
        """
        Initialize K-means++.
        
        Args:
            data: Data matrix (n_samples x n_features)
            n_clusters: Number of clusters
        """
        self.data = data
        self.n_samples, self.n_features = data.shape
        self.n_clusters = n_clusters
        self.centroids = []
        self.labels = np.zeros(self.n_samples, dtype=int)
        
    def initialize(self) -> GaussianMixtureModel:
        """
        Run K-means++ initialization.
        
        Returns:
            Initialized GaussianMixtureModel
        """
        logger.info(f"Initializing {self.n_clusters} clusters with K-means++")
        
        # Step 1: Choose first centroid randomly
        first_idx = np.random.randint(0, self.n_samples)
        self.centroids = [self.data[first_idx].copy()]
        
        # Step 2: Choose remaining centroids
        for k in range(1, self.n_clusters):
            # Compute distance squared to nearest centroid for each point
            distances_sq = np.zeros(self.n_samples)
            for i in range(self.n_samples):
                min_dist_sq = float('inf')
                for centroid in self.centroids:
                    dist_sq = np.sum((self.data[i] - centroid) ** 2)
                    min_dist_sq = min(min_dist_sq, dist_sq)
                distances_sq[i] = min_dist_sq
            
            # Choose next centroid with probability proportional to distance squared
            probabilities = distances_sq / (np.sum(distances_sq) + 1e-10)
            next_idx = np.random.choice(self.n_samples, p=probabilities)
            self.centroids.append(self.data[next_idx].copy())
            
            logger.info(f"  Selected centroid {k+1}/{self.n_clusters}")
        
        # Assign points to nearest centroids
        for i in range(self.n_samples):
            min_dist = float('inf')
            for k, centroid in enumerate(self.centroids):
                dist = np.sum((self.data[i] - centroid) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    self.labels[i] = k
        
        # Create GMM from K-means++ results
        gmm = GaussianMixtureModel(self.n_clusters, self.n_features)
        
        for k in range(self.n_clusters):
            # Get points assigned to this cluster
            cluster_points = self.data[self.labels == k]
            
            if len(cluster_points) > 0:
                # Compute mean and covariance
                mean = np.mean(cluster_points, axis=0)
                covariance = np.cov(cluster_points.T)
                
                # Handle single point or identical points
                if cluster_points.shape[0] == 1 or np.all(covariance == 0):
                    covariance = 0.01 * np.eye(self.n_features)
                elif len(covariance.shape) == 0:  # Scalar covariance (1D case)
                    covariance = np.array([[max(covariance, 0.01)]])
                    
                weight = len(cluster_points) / self.n_samples
            else:
                # Empty cluster - use centroid with small covariance
                mean = self.centroids[k]
                covariance = 0.01 * np.eye(self.n_features)
                weight = 1.0 / self.n_clusters
            
            gmm.set_gaussian(k, mean, covariance)
            gmm.set_weight(k, weight)
        
        logger.info("K-means++ initialization complete")
        return gmm


class EMGaussianMixtures:
    """
    Expectation-Maximization algorithm for Gaussian Mixture Models.
    
    Equivalent to ITK-SNAP's EMGaussianMixtures class.
    """
    
    def __init__(self, data: np.ndarray, gmm: GaussianMixtureModel, max_iterations: int = 10):
        """
        Initialize EM algorithm.
        
        Args:
            data: Data matrix (n_samples x n_features)
            gmm: Initial Gaussian Mixture Model
            max_iterations: Maximum number of EM iterations
        """
        self.data = data
        self.n_samples, self.n_features = data.shape
        self.gmm = gmm
        self.max_iterations = max_iterations
        self.n_components = gmm.n_components
        
        # Allocate arrays for posteriors
        self.posteriors = np.zeros((self.n_samples, self.n_components))
        self.log_pdf = np.zeros((self.n_samples, self.n_components))
        
    def compute_posterior(self, log_pdf: np.ndarray, weights: np.ndarray, 
                         log_weights: np.ndarray, j: int) -> float:
        """
        Compute posterior probability for component j using log-space arithmetic.
        
        This is the numerically stable way to compute:
        P(j|x) = w_j * N(x|μ_j,Σ_j) / Σ_k w_k * N(x|μ_k,Σ_k)
        
        Args:
            log_pdf: Log PDFs for all components
            weights: Component weights
            log_weights: Log of component weights
            j: Component index
            
        Returns:
            Posterior probability
        """
        # Find maximum log value for numerical stability (log-sum-exp trick)
        max_log = np.max(log_pdf + log_weights)
        
        # Compute denominator in log space
        log_sum = max_log + np.log(np.sum(weights * np.exp(log_pdf - max_log)))
        
        # Compute posterior
        log_posterior = log_weights[j] + log_pdf[j] - log_sum
        
        return np.exp(log_posterior)
    
    def e_step(self):
        """
        E-step: Compute posterior probabilities (responsibilities).
        """
        # Compute log PDFs for all samples and all components
        for k in range(self.n_components):
            for i in range(self.n_samples):
                self.log_pdf[i, k] = self.gmm.evaluate_log_pdf(k, self.data[i])
        
        # Compute posteriors using log-space arithmetic
        log_weights = np.log(self.gmm.weights + 1e-300)
        
        for i in range(self.n_samples):
            for k in range(self.n_components):
                self.posteriors[i, k] = self.compute_posterior(
                    self.log_pdf[i], 
                    self.gmm.weights, 
                    log_weights, 
                    k
                )
    
    def m_step(self):
        """
        M-step: Update Gaussian parameters based on posteriors.
        """
        for k in range(self.n_components):
            # Sum of responsibilities for this component
            n_k = np.sum(self.posteriors[:, k])
            
            if n_k < 1e-10:
                # Component has no support - keep current parameters
                logger.warning(f"Component {k} has no support, keeping current parameters")
                continue
            
            # Update weight
            weight = n_k / self.n_samples
            
            # Update mean
            mean = np.sum(self.posteriors[:, k:k+1] * self.data, axis=0) / n_k
            
            # Update covariance
            diff = self.data - mean
            covariance = np.dot((self.posteriors[:, k:k+1] * diff).T, diff) / n_k
            
            # Add regularization to prevent singular covariance
            covariance += 1e-6 * np.eye(self.n_features)
            
            # Update GMM
            self.gmm.set_weight(k, weight)
            self.gmm.set_gaussian(k, mean, covariance)
    
    def compute_log_likelihood(self) -> float:
        """
        Compute log-likelihood of the data given current GMM.
        
        Returns:
            Log-likelihood value
        """
        log_likelihood = 0.0
        
        for i in range(self.n_samples):
            # Compute log of mixture density
            log_probs = np.array([
                np.log(self.gmm.weights[k] + 1e-300) + self.gmm.evaluate_log_pdf(k, self.data[i])
                for k in range(self.n_components)
            ])
            
            # Log-sum-exp for numerical stability
            max_log = np.max(log_probs)
            log_likelihood += max_log + np.log(np.sum(np.exp(log_probs - max_log)))
        
        return log_likelihood
    
    def fit(self) -> GaussianMixtureModel:
        """
        Run EM algorithm to convergence or max iterations.
        
        Returns:
            Trained GaussianMixtureModel
        """
        logger.info(f"Starting EM algorithm (max {self.max_iterations} iterations)")
        
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iterations):
            # E-step
            self.e_step()
            
            # M-step
            self.m_step()
            
            # Compute log-likelihood
            log_likelihood = self.compute_log_likelihood()
            
            logger.info(f"  Iteration {iteration+1}/{self.max_iterations}, "
                       f"Log-likelihood: {log_likelihood:.2f}")
            
            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < 1e-4:
                logger.info(f"Converged after {iteration+1} iterations")
                break
            
            prev_log_likelihood = log_likelihood
        
        logger.info("EM algorithm complete")
        return self.gmm


class GMMDicomSegmentation:
    """
    Main class for GMM-based DICOM segmentation.
    
    Equivalent to ITK-SNAP's UnsupervisedClustering + GMMClassifyImageFilter.
    """
    
    def __init__(self, n_clusters: int = 3, n_samples: int = 10000, 
                 em_iterations: int = 10, random_seed: Optional[int] = None):
        """
        Initialize segmentation pipeline.
        
        Args:
            n_clusters: Number of tissue clusters
            n_samples: Number of voxels to sample for training
            em_iterations: Number of EM iterations
            random_seed: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.n_samples = n_samples
        self.em_iterations = em_iterations
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.image_data = None
        self.sampled_data = None
        self.gmm = None
        
    def load_dicom_series(self, dicom_path: str) -> np.ndarray:
        """
        Load DICOM series from directory or file.
        
        Args:
            dicom_path: Path to DICOM directory or file
            
        Returns:
            3D image array (z, y, x) or 4D for multi-component
        """
        try:
            import SimpleITK as sitk
        except ImportError:
            logger.error("SimpleITK is required for DICOM loading. Install with: pip install SimpleITK")
            sys.exit(1)
        
        logger.info(f"Loading DICOM from: {dicom_path}")
        
        if os.path.isdir(dicom_path):
            # Load DICOM series from directory
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
            
            if len(dicom_names) == 0:
                raise ValueError(f"No DICOM files found in {dicom_path}")
            
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            logger.info(f"Loaded {len(dicom_names)} DICOM slices")
        else:
            # Load single DICOM file
            image = sitk.ReadImage(dicom_path)
            logger.info("Loaded single DICOM file")
        
        # Convert to numpy array
        image_array = sitk.GetArrayFromImage(image)
        
        logger.info(f"Image shape: {image_array.shape}, dtype: {image_array.dtype}")
        logger.info(f"Image range: [{image_array.min()}, {image_array.max()}]")
        
        self.image_data = image_array
        return image_array
    
    def sample_voxels(self) -> np.ndarray:
        """
        Randomly sample voxels from the image for training.
        
        Equivalent to UnsupervisedClustering::SampleDataSource()
        
        Returns:
            Sampled data matrix (n_samples x n_features)
        """
        if self.image_data is None:
            raise ValueError("No image data loaded. Call load_dicom_series() first.")
        
        logger.info("Sampling voxels for training")
        
        # Get image shape
        if len(self.image_data.shape) == 3:
            # Single component image
            z, y, x = self.image_data.shape
            n_components = 1
            flat_data = self.image_data.reshape(-1, 1)
        elif len(self.image_data.shape) == 4:
            # Multi-component image
            z, y, x, n_components = self.image_data.shape
            flat_data = self.image_data.reshape(-1, n_components)
        else:
            raise ValueError(f"Unsupported image shape: {self.image_data.shape}")
        
        total_voxels = z * y * x
        n_samples = min(self.n_samples, total_voxels)
        
        logger.info(f"Total voxels: {total_voxels}, sampling: {n_samples}")
        
        # Random sampling without replacement
        sample_indices = np.random.choice(total_voxels, size=n_samples, replace=False)
        sampled_data = flat_data[sample_indices]
        
        # Filter out non-finite values
        finite_mask = np.all(np.isfinite(sampled_data), axis=1)
        sampled_data = sampled_data[finite_mask]
        
        logger.info(f"Samples after filtering non-finite values: {len(sampled_data)}")
        
        if len(sampled_data) < self.n_clusters:
            raise ValueError(f"Too few valid samples ({len(sampled_data)}) for {self.n_clusters} clusters")
        
        self.sampled_data = sampled_data
        return sampled_data
    
    def initialize_clusters(self):
        """
        Initialize clusters using K-means++.
        
        Equivalent to UnsupervisedClustering::InitializeClusters()
        """
        if self.sampled_data is None:
            raise ValueError("No sampled data. Call sample_voxels() first.")
        
        # Run K-means++ initialization
        kmeans = KMeansPlusPlus(self.sampled_data, self.n_clusters)
        self.gmm = kmeans.initialize()
    
    def refine_clusters(self):
        """
        Refine clusters using EM algorithm.
        
        Equivalent to UnsupervisedClustering::Iterate() called multiple times
        """
        if self.gmm is None:
            raise ValueError("No GMM initialized. Call initialize_clusters() first.")
        
        # Run EM algorithm
        em = EMGaussianMixtures(self.sampled_data, self.gmm, self.em_iterations)
        self.gmm = em.fit()
    
    def classify_image(self, foreground_clusters: Optional[List[int]] = None) -> np.ndarray:
        """
        Classify all voxels in the image using the trained GMM.
        
        Equivalent to GMMClassifyImageFilter::DynamicThreadedGenerateData()
        
        Args:
            foreground_clusters: List of cluster indices to mark as foreground.
                                If None, all clusters are considered foreground.
        
        Returns:
            Probability/speed map with same spatial shape as input
        """
        if self.gmm is None:
            raise ValueError("No GMM trained. Call initialize_clusters() and refine_clusters() first.")
        
        if self.image_data is None:
            raise ValueError("No image data loaded.")
        
        logger.info("Classifying full image")
        
        # Set foreground/background state
        if foreground_clusters is not None:
            for k in range(self.n_clusters):
                if k in foreground_clusters:
                    self.gmm.set_foreground(k)
                else:
                    self.gmm.set_background(k)
        
        # Get image dimensions
        if len(self.image_data.shape) == 3:
            z, y, x = self.image_data.shape
            n_components = 1
            flat_data = self.image_data.reshape(-1, 1)
            output_shape = (z, y, x)
        else:
            z, y, x, n_components = self.image_data.shape
            flat_data = self.image_data.reshape(-1, n_components)
            output_shape = (z, y, x)
        
        total_voxels = z * y * x
        speed_map = np.zeros(total_voxels)
        
        # Precompute log weights
        log_weights = np.log(self.gmm.weights + 1e-300)
        
        # Foreground factors (+1 for foreground, -1 for background)
        fg_factors = np.array([1.0 if self.gmm.is_foreground(k) else -1.0 
                              for k in range(self.n_clusters)])
        
        # Process in batches for efficiency
        batch_size = 10000
        n_batches = (total_voxels + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_voxels)
            batch_data = flat_data[start_idx:end_idx]
            
            # Compute log PDFs for all components
            log_pdfs = np.zeros((len(batch_data), self.n_clusters))
            for k in range(self.n_clusters):
                for i, x in enumerate(batch_data):
                    log_pdfs[i, k] = self.gmm.evaluate_log_pdf(k, x)
            
            # Compute posteriors and weighted probability difference
            for i in range(len(batch_data)):
                pdiff = 0.0
                for k in range(self.n_clusters):
                    # Compute posterior using log-space arithmetic
                    max_log = np.max(log_pdfs[i] + log_weights)
                    log_sum = max_log + np.log(
                        np.sum(self.gmm.weights * np.exp(log_pdfs[i] - max_log))
                    )
                    posterior = np.exp(log_weights[k] + log_pdfs[i, k] - log_sum)
                    
                    # Accumulate weighted by foreground/background
                    pdiff += posterior * fg_factors[k]
                
                # Scale to [-32767, 32767] as in ITK-SNAP
                speed_map[start_idx + i] = pdiff * 32767.0
            
            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                logger.info(f"  Processed {end_idx}/{total_voxels} voxels "
                           f"({100*end_idx//total_voxels}%)")
        
        # Reshape to original spatial dimensions
        speed_map = speed_map.reshape(output_shape)
        
        logger.info("Classification complete")
        return speed_map
    
    def run(self, dicom_path: str, foreground_clusters: Optional[List[int]] = None) -> np.ndarray:
        """
        Run complete segmentation pipeline.
        
        Args:
            dicom_path: Path to DICOM directory or file
            foreground_clusters: List of cluster indices to mark as foreground
        
        Returns:
            Probability/speed map
        """
        # Step 1: Load DICOM
        self.load_dicom_series(dicom_path)
        
        # Step 2: Sample voxels
        self.sample_voxels()
        
        # Step 3: Initialize clusters with K-means++
        self.initialize_clusters()
        
        # Step 4: Refine with EM
        self.refine_clusters()
        
        # Step 5: Classify full image
        speed_map = self.classify_image(foreground_clusters)
        
        return speed_map
    
    def save_results(self, speed_map: np.ndarray, output_path: str):
        """
        Save segmentation results.
        
        Args:
            speed_map: Probability/speed map to save
            output_path: Output file path
        """
        try:
            import SimpleITK as sitk
        except ImportError:
            logger.error("SimpleITK is required for saving. Install with: pip install SimpleITK")
            return
        
        # Convert to SimpleITK image
        output_image = sitk.GetImageFromArray(speed_map)
        
        # Save
        sitk.WriteImage(output_image, output_path)
        logger.info(f"Saved speed map to: {output_path}")
    
    def print_cluster_info(self):
        """Print information about the learned clusters."""
        if self.gmm is None:
            logger.warning("No GMM trained yet")
            return
        
        logger.info("\n" + "="*60)
        logger.info("Cluster Information")
        logger.info("="*60)
        
        for k in range(self.n_clusters):
            mean = self.gmm.get_mean(k)
            weight = self.gmm.get_weight(k)
            fg_status = "Foreground" if self.gmm.is_foreground(k) else "Background"
            
            logger.info(f"\nCluster {k}:")
            logger.info(f"  Weight: {weight:.4f}")
            logger.info(f"  Mean: {mean}")
            logger.info(f"  Status: {fg_status}")
        
        logger.info("="*60 + "\n")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="GMM-based automatic tissue classification from DICOM images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Segment with 3 clusters (default)
  python gmm_dicom_segmentation.py /path/to/dicom/series -o output.nii
  
  # Segment with 5 clusters, mark clusters 0 and 1 as foreground
  python gmm_dicom_segmentation.py /path/to/dicom/series -o output.nii \\
      --clusters 5 --foreground 0 1
  
  # Use 20000 samples and 15 EM iterations
  python gmm_dicom_segmentation.py /path/to/dicom/series -o output.nii \\
      --samples 20000 --em-iterations 15
        """
    )
    
    parser.add_argument('dicom_path', help='Path to DICOM directory or file')
    parser.add_argument('-o', '--output', required=True, help='Output file path')
    parser.add_argument('--clusters', type=int, default=3,
                       help='Number of tissue clusters (default: 3)')
    parser.add_argument('--samples', type=int, default=10000,
                       help='Number of voxels to sample for training (default: 10000)')
    parser.add_argument('--em-iterations', type=int, default=10,
                       help='Number of EM iterations (default: 10)')
    parser.add_argument('--foreground', type=int, nargs='+',
                       help='Cluster indices to mark as foreground (default: all)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Validate inputs
    if not os.path.exists(args.dicom_path):
        logger.error(f"DICOM path does not exist: {args.dicom_path}")
        sys.exit(1)
    
    if args.clusters < 2:
        logger.error("Number of clusters must be at least 2")
        sys.exit(1)
    
    if args.foreground is not None:
        for fg_idx in args.foreground:
            if fg_idx < 0 or fg_idx >= args.clusters:
                logger.error(f"Invalid foreground cluster index: {fg_idx}")
                sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run segmentation
    try:
        segmenter = GMMDicomSegmentation(
            n_clusters=args.clusters,
            n_samples=args.samples,
            em_iterations=args.em_iterations,
            random_seed=args.seed
        )
        
        speed_map = segmenter.run(args.dicom_path, args.foreground)
        
        # Print cluster information
        segmenter.print_cluster_info()
        
        # Save results
        segmenter.save_results(speed_map, args.output)
        
        logger.info(f"\nSegmentation complete!")
        logger.info(f"Speed map range: [{speed_map.min():.2f}, {speed_map.max():.2f}]")
        logger.info(f"Output saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Error during segmentation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
