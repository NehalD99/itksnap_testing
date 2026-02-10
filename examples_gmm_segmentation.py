#!/usr/bin/env python3
"""
Example usage of GMM DICOM segmentation.

This script demonstrates practical usage patterns for the GMM segmentation tool.
"""

import numpy as np
import sys
import os
from pathlib import Path

# Ensure the gmm_dicom_segmentation module is available
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gmm_dicom_segmentation import GMMDicomSegmentation

def example1_basic_usage():
    """
    Example 1: Basic usage with synthetic data.
    
    This demonstrates the simplest way to use the segmentation pipeline.
    """
    print("\n" + "="*70)
    print("Example 1: Basic Usage with Synthetic Data")
    print("="*70)
    
    # Create synthetic 3D image with 3 tissue types
    print("\n1. Creating synthetic brain-like MRI data...")
    np.random.seed(42)
    
    # Dimensions
    z, y, x = 30, 30, 30
    image = np.zeros((z, y, x))
    
    # CSF (low intensity)
    csf_mask = (np.random.rand(z, y, x) > 0.7)
    image[csf_mask] = 50 + np.random.randn(np.sum(csf_mask)) * 10
    
    # Gray matter (medium intensity)
    gm_mask = (~csf_mask) & (np.random.rand(z, y, x) > 0.5)
    image[gm_mask] = 100 + np.random.randn(np.sum(gm_mask)) * 10
    
    # White matter (high intensity)
    wm_mask = (~csf_mask) & (~gm_mask)
    image[wm_mask] = 150 + np.random.randn(np.sum(wm_mask)) * 10
    
    print(f"   Created {z}x{y}x{x} volume")
    print(f"   Intensity range: [{image.min():.1f}, {image.max():.1f}]")
    
    # Create segmenter
    print("\n2. Creating GMM segmenter with 3 clusters...")
    segmenter = GMMDicomSegmentation(
        n_clusters=3,
        n_samples=2000,
        em_iterations=10,
        random_seed=42
    )
    
    # Set image data directly (normally you'd use load_dicom_series)
    segmenter.image_data = image
    
    # Run pipeline
    print("\n3. Running segmentation pipeline...")
    print("   a) Sampling voxels...")
    segmenter.sample_voxels()
    
    print("   b) Initializing clusters with K-means++...")
    segmenter.initialize_clusters()
    
    print("   c) Refining clusters with EM...")
    segmenter.refine_clusters()
    
    print("   d) Classifying full image...")
    speed_map = segmenter.classify_image()
    
    # Display results
    print("\n4. Results:")
    segmenter.print_cluster_info()
    print(f"   Speed map range: [{speed_map.min():.2f}, {speed_map.max():.2f}]")
    
    # Create binary segmentation
    binary_seg = (speed_map > 0).astype(np.uint8)
    print(f"   Foreground voxels: {np.sum(binary_seg)} / {binary_seg.size}")
    
    return segmenter, speed_map


def example2_advanced_usage():
    """
    Example 2: Advanced usage with foreground/background selection.
    
    This shows how to mark specific clusters as foreground or background.
    """
    print("\n" + "="*70)
    print("Example 2: Advanced Usage with Foreground/Background Selection")
    print("="*70)
    
    # Create synthetic data with 4 tissue types
    print("\n1. Creating synthetic CT-like data with 4 tissue types...")
    np.random.seed(123)
    
    z, y, x = 40, 40, 40
    image = np.zeros((z, y, x))
    
    # Air/background (very low)
    image[:10, :, :] = -1000 + np.random.randn(10, y, x) * 50
    
    # Soft tissue
    image[10:20, :, :] = 50 + np.random.randn(10, y, x) * 20
    
    # Muscle
    image[20:30, :, :] = 100 + np.random.randn(10, y, x) * 20
    
    # Bone
    image[30:, :, :] = 400 + np.random.randn(10, y, x) * 50
    
    print(f"   Created {z}x{y}x{x} volume with 4 tissue types")
    
    # Create and run segmenter
    print("\n2. Running segmentation with 4 clusters...")
    segmenter = GMMDicomSegmentation(
        n_clusters=4,
        n_samples=3000,
        em_iterations=10,
        random_seed=123
    )
    
    segmenter.image_data = image
    segmenter.sample_voxels()
    segmenter.initialize_clusters()
    segmenter.refine_clusters()
    
    # First classification with all foreground
    print("\n3. Initial classification (all clusters as foreground)...")
    speed_map_all_fg = segmenter.classify_image()
    
    # Print cluster info to decide which are foreground
    segmenter.print_cluster_info()
    
    # Manually select foreground clusters based on means
    # Let's say we want clusters with mean > 50 to be foreground
    foreground_clusters = []
    for k in range(segmenter.n_clusters):
        mean = segmenter.gmm.get_mean(k)[0]
        if mean > 50:
            foreground_clusters.append(k)
    
    print(f"\n4. Selected foreground clusters: {foreground_clusters}")
    
    # Reclassify with foreground/background
    print("\n5. Re-classifying with foreground/background selection...")
    speed_map_selective = segmenter.classify_image(foreground_clusters)
    
    print(f"   All foreground - Speed range: [{speed_map_all_fg.min():.2f}, {speed_map_all_fg.max():.2f}]")
    print(f"   Selective - Speed range: [{speed_map_selective.min():.2f}, {speed_map_selective.max():.2f}]")
    
    return segmenter, speed_map_selective


def example3_parameter_comparison():
    """
    Example 3: Compare different parameter settings.
    
    This shows how parameters affect the results.
    """
    print("\n" + "="*70)
    print("Example 3: Parameter Comparison")
    print("="*70)
    
    # Create synthetic data
    print("\n1. Creating synthetic data...")
    np.random.seed(456)
    z, y, x = 25, 25, 25
    image = np.zeros((z, y, x))
    
    # Two tissue types
    image[:, :15, :] = 80 + np.random.randn(z, 15, x) * 15
    image[:, 15:, :] = 120 + np.random.randn(z, 10, x) * 15
    
    # Test different numbers of clusters
    print("\n2. Testing different numbers of clusters...")
    
    for n_clusters in [2, 3, 4]:
        print(f"\n   Testing {n_clusters} clusters...")
        segmenter = GMMDicomSegmentation(
            n_clusters=n_clusters,
            n_samples=1500,
            em_iterations=5,
            random_seed=456
        )
        
        segmenter.image_data = image
        segmenter.sample_voxels()
        segmenter.initialize_clusters()
        segmenter.refine_clusters()
        
        # Print summary
        print(f"      Learned cluster means:")
        for k in range(n_clusters):
            mean = segmenter.gmm.get_mean(k)[0]
            weight = segmenter.gmm.get_weight(k)
            print(f"        Cluster {k}: mean={mean:.1f}, weight={weight:.3f}")
    
    # Test different numbers of samples
    print("\n3. Testing different numbers of samples (3 clusters)...")
    
    for n_samples in [500, 2000, 5000]:
        print(f"\n   Testing {n_samples} samples...")
        segmenter = GMMDicomSegmentation(
            n_clusters=3,
            n_samples=n_samples,
            em_iterations=5,
            random_seed=456
        )
        
        segmenter.image_data = image
        segmenter.sample_voxels()
        
        print(f"      Actually sampled: {len(segmenter.sampled_data)} voxels")


def example4_reproducibility():
    """
    Example 4: Demonstrate reproducibility with random seed.
    """
    print("\n" + "="*70)
    print("Example 4: Reproducibility with Random Seed")
    print("="*70)
    
    # Create synthetic data
    np.random.seed(789)
    z, y, x = 20, 20, 20
    image = np.random.randn(z, y, x) * 50 + 100
    
    print("\n1. Running segmentation twice with same seed...")
    
    results = []
    for run in [1, 2]:
        print(f"\n   Run {run}:")
        segmenter = GMMDicomSegmentation(
            n_clusters=3,
            n_samples=1000,
            em_iterations=5,
            random_seed=42  # Same seed
        )
        
        segmenter.image_data = image
        speed_map = segmenter.run(None)
        
        # Store cluster means
        means = [segmenter.gmm.get_mean(k) for k in range(3)]
        results.append(means)
        
        print(f"      Cluster means: {[m[0] for m in means]}")
    
    # Check if results are identical
    print("\n2. Checking reproducibility...")
    for k in range(3):
        diff = abs(results[0][k][0] - results[1][k][0])
        print(f"   Cluster {k} mean difference: {diff:.10f}")
        if diff < 1e-6:
            print(f"      ✓ Identical")
        else:
            print(f"      ✗ Different")


def example5_save_load():
    """
    Example 5: Save and load results (requires SimpleITK).
    """
    print("\n" + "="*70)
    print("Example 5: Save and Load Results")
    print("="*70)
    
    try:
        import SimpleITK as sitk
        
        # Create synthetic data
        print("\n1. Creating synthetic data...")
        np.random.seed(999)
        z, y, x = 15, 15, 15
        image = np.random.randn(z, y, x) * 30 + 100
        
        # Run segmentation
        print("\n2. Running segmentation...")
        segmenter = GMMDicomSegmentation(
            n_clusters=3,
            n_samples=800,
            em_iterations=5,
            random_seed=999
        )
        
        segmenter.image_data = image
        segmenter.sample_voxels()
        segmenter.initialize_clusters()
        segmenter.refine_clusters()
        speed_map = segmenter.classify_image()
        
        # Save results
        print("\n3. Saving results...")
        temp_dir = "/tmp/gmm_example"
        os.makedirs(temp_dir, exist_ok=True)
        
        speed_path = os.path.join(temp_dir, "speed_map.nii")
        segmenter.save_results(speed_map, speed_path)
        
        # Create binary segmentation
        binary_seg = (speed_map > 0).astype(np.uint8)
        binary_image = sitk.GetImageFromArray(binary_seg)
        binary_path = os.path.join(temp_dir, "binary_seg.nii")
        sitk.WriteImage(binary_image, binary_path)
        print(f"   Saved binary segmentation to: {binary_path}")
        
        # Load and verify
        print("\n4. Loading and verifying...")
        loaded_speed = sitk.ReadImage(speed_path)
        loaded_array = sitk.GetArrayFromImage(loaded_speed)
        
        print(f"   Original shape: {speed_map.shape}")
        print(f"   Loaded shape: {loaded_array.shape}")
        print(f"   Arrays match: {np.allclose(speed_map, loaded_array)}")
        
    except ImportError:
        print("\nSimpleITK not available. Install with: pip install SimpleITK")
        print("This example requires SimpleITK for saving/loading.")


def main():
    """Run all examples."""
    print("\n" + "#"*70)
    print("# GMM DICOM Segmentation - Practical Examples")
    print("#"*70)
    
    try:
        # Run examples
        example1_basic_usage()
        example2_advanced_usage()
        example3_parameter_comparison()
        example4_reproducibility()
        example5_save_load()
        
        print("\n" + "#"*70)
        print("# All examples completed successfully!")
        print("#"*70 + "\n")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
