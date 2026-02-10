#!/usr/bin/env python3
"""
Test script for GMM DICOM segmentation.

This script demonstrates and tests the GMM segmentation pipeline with synthetic data.
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gmm_dicom_segmentation import (
    Gaussian, GaussianMixtureModel, KMeansPlusPlus, 
    EMGaussianMixtures, GMMDicomSegmentation
)

def test_gaussian():
    """Test Gaussian class."""
    print("Testing Gaussian class...")
    
    # Create 2D Gaussian
    g = Gaussian(2)
    mean = np.array([1.0, 2.0])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    g.set_parameters(mean, cov)
    
    # Evaluate at mean (should be high probability)
    log_pdf = g.evaluate_log_pdf(mean)
    print(f"  Log PDF at mean: {log_pdf:.4f}")
    
    # Evaluate at distant point (should be low probability)
    distant = np.array([10.0, 10.0])
    log_pdf_distant = g.evaluate_log_pdf(distant)
    print(f"  Log PDF at distant point: {log_pdf_distant:.4f}")
    
    assert log_pdf > log_pdf_distant, "PDF at mean should be higher than at distant point"
    print("  ✓ Gaussian test passed\n")


def test_kmeans_plusplus():
    """Test K-means++ initialization."""
    print("Testing K-means++ initialization...")
    
    # Create synthetic data with 3 clusters
    np.random.seed(42)
    cluster1 = np.random.randn(100, 2) + np.array([0, 0])
    cluster2 = np.random.randn(100, 2) + np.array([5, 5])
    cluster3 = np.random.randn(100, 2) + np.array([10, 0])
    data = np.vstack([cluster1, cluster2, cluster3])
    
    # Run K-means++
    kmeans = KMeansPlusPlus(data, 3)
    gmm = kmeans.initialize()
    
    print(f"  Initialized {gmm.n_components} clusters")
    print(f"  Feature dimension: {gmm.n_features}")
    
    # Check that weights sum to 1
    total_weight = sum(gmm.weights)
    print(f"  Total weight: {total_weight:.6f}")
    assert abs(total_weight - 1.0) < 1e-6, "Weights should sum to 1"
    
    # Print cluster means
    for k in range(gmm.n_components):
        mean = gmm.get_mean(k)
        print(f"  Cluster {k} mean: {mean}")
    
    print("  ✓ K-means++ test passed\n")


def test_em_algorithm():
    """Test EM algorithm."""
    print("Testing EM algorithm...")
    
    # Create synthetic data
    np.random.seed(42)
    cluster1 = np.random.randn(100, 2) * 0.5 + np.array([0, 0])
    cluster2 = np.random.randn(100, 2) * 0.5 + np.array([5, 5])
    data = np.vstack([cluster1, cluster2])
    
    # Initialize with K-means++
    kmeans = KMeansPlusPlus(data, 2)
    gmm_init = kmeans.initialize()
    
    print(f"  Initial log-likelihood...")
    
    # Run EM
    em = EMGaussianMixtures(data, gmm_init, max_iterations=10)
    gmm_final = em.fit()
    
    # Check that we got valid parameters
    for k in range(gmm_final.n_components):
        mean = gmm_final.get_mean(k)
        weight = gmm_final.get_weight(k)
        print(f"  Cluster {k}: mean={mean}, weight={weight:.4f}")
        assert np.all(np.isfinite(mean)), "Mean should be finite"
        assert 0 <= weight <= 1, "Weight should be in [0, 1]"
    
    print("  ✓ EM algorithm test passed\n")


def test_full_pipeline():
    """Test full segmentation pipeline with synthetic 3D data."""
    print("Testing full segmentation pipeline...")
    
    # Create synthetic 3D image (small for speed)
    np.random.seed(42)
    z, y, x = 20, 20, 20
    
    # Create image with 3 regions
    image = np.zeros((z, y, x))
    image[:, :10, :] = 50 + np.random.randn(z, 10, x) * 5  # Region 1
    image[:, 10:, :10] = 100 + np.random.randn(z, 10, 10) * 5  # Region 2
    image[:, 10:, 10:] = 150 + np.random.randn(z, 10, 10) * 5  # Region 3
    
    print(f"  Created synthetic image: shape={image.shape}")
    print(f"  Image range: [{image.min():.2f}, {image.max():.2f}]")
    
    # Test with both single-threaded and multi-threaded
    for n_jobs in [1, -1]:
        job_desc = "single-threaded" if n_jobs == 1 else "multi-threaded"
        print(f"\n  Testing {job_desc} (n_jobs={n_jobs})...")
        
        # Create segmenter
        segmenter = GMMDicomSegmentation(n_clusters=3, n_samples=1000, 
                                        em_iterations=5, random_seed=42, n_jobs=n_jobs)
        
        # Set image data directly
        segmenter.image_data = image
        
        # Run pipeline steps
        print(f"    Step 1: Sampling voxels...")
        segmenter.sample_voxels()
        print(f"      Sampled {len(segmenter.sampled_data)} voxels")
        
        print(f"    Step 2: Initializing clusters...")
        segmenter.initialize_clusters()
        
        print(f"    Step 3: Refining with EM...")
        segmenter.refine_clusters()
        
        print(f"    Step 4: Classifying image...")
        speed_map = segmenter.classify_image()
        
        print(f"    Speed map shape: {speed_map.shape}")
        print(f"    Speed map range: [{speed_map.min():.2f}, {speed_map.max():.2f}]")
        
        assert speed_map.shape == image.shape, "Speed map should have same shape as input"
        assert np.all(np.isfinite(speed_map)), "Speed map should be finite"
    
    # Print cluster info
    segmenter.print_cluster_info()
    
    print("  ✓ Full pipeline test passed\n")




def test_multithreading():
    """Test multithreading performance."""
    print("Testing multithreading performance...")
    
    # Create a larger synthetic image
    np.random.seed(789)
    z, y, x = 30, 30, 30
    image = np.random.randn(z, y, x) * 50 + 100
    
    print(f"  Created image: {z}x{y}x{x} = {z*y*x} voxels")
    
    import time
    
    # Test single-threaded
    print("\n  Running single-threaded (n_jobs=1)...")
    segmenter_st = GMMDicomSegmentation(
        n_clusters=3, n_samples=1000, em_iterations=3, random_seed=789, n_jobs=1
    )
    segmenter_st.image_data = image
    segmenter_st.sample_voxels()
    segmenter_st.initialize_clusters()
    segmenter_st.refine_clusters()
    
    start_st = time.time()
    speed_map_st = segmenter_st.classify_image()
    time_st = time.time() - start_st
    print(f"    Time: {time_st:.3f} seconds")
    
    # Test multi-threaded
    print("\n  Running multi-threaded (n_jobs=-1)...")
    segmenter_mt = GMMDicomSegmentation(
        n_clusters=3, n_samples=1000, em_iterations=3, random_seed=789, n_jobs=-1
    )
    segmenter_mt.image_data = image
    segmenter_mt.sample_voxels()
    segmenter_mt.initialize_clusters()
    segmenter_mt.refine_clusters()
    
    start_mt = time.time()
    speed_map_mt = segmenter_mt.classify_image()
    time_mt = time.time() - start_mt
    print(f"    Time: {time_mt:.3f} seconds")
    
    # Check results are identical (with same random seed)
    print(f"\n  Verifying results match...")
    assert np.allclose(speed_map_st, speed_map_mt, rtol=1e-10), "Results should be identical"
    print(f"    ✓ Results are identical")
    
    # Report speedup
    speedup = time_st / time_mt if time_mt > 0 else 1.0
    print(f"\n  Performance:")
    print(f"    Single-threaded: {time_st:.3f}s")
    print(f"    Multi-threaded:  {time_mt:.3f}s")
    print(f"    Speedup: {speedup:.2f}x")
    
    print("  ✓ Multithreading test passed\n")


def test_with_simpleITK():
    """Test DICOM loading with SimpleITK (if available)."""
    try:
        import SimpleITK as sitk
        print("Testing SimpleITK integration...")
        
        # Create a simple synthetic image
        np.random.seed(42)
        array = np.random.randn(10, 10, 10).astype(np.float32) * 100 + 500
        
        # Convert to SimpleITK and save as temp file
        image = sitk.GetImageFromArray(array)
        temp_file = "/tmp/test_image.nii"
        sitk.WriteImage(image, temp_file)
        print(f"  Saved test image to {temp_file}")
        
        # Load it back
        loaded = sitk.ReadImage(temp_file)
        loaded_array = sitk.GetArrayFromImage(loaded)
        
        print(f"  Loaded array shape: {loaded_array.shape}")
        assert np.allclose(array, loaded_array), "Loaded array should match original"
        
        # Clean up
        os.remove(temp_file)
        
        print("  ✓ SimpleITK test passed\n")
        
    except ImportError:
        print("  SimpleITK not available, skipping test\n")


def main():
    """Run all tests."""
    print("="*60)
    print("GMM DICOM Segmentation - Test Suite")
    print("="*60 + "\n")
    
    try:
        test_gaussian()
        test_kmeans_plusplus()
        test_em_algorithm()
        test_full_pipeline()
        test_multithreading()
        test_with_simpleITK()
        
        print("="*60)
        print("All tests passed! ✓")
        print("="*60)
        return 0
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"Test failed: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
