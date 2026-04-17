#!/usr/bin/env python3
"""
Simple test script to verify point cloud cropping functionality
"""

import numpy as np
import open3d as o3d

def create_test_point_cloud():
    """Create a simple test point cloud with points at various depths"""
    # Create points with different Z coordinates (depths)
    points = np.array([
        [0.0, 0.0, 0.3],   # Close point - should be kept
        [0.5, 0.5, 0.6],   # Medium distance - should be kept
        [1.0, 1.0, 0.8],   # At threshold - should be kept
        [1.5, 1.5, 0.85],  # Just over threshold - should be removed
        [2.0, 2.0, 1.0],   # Far point - should be removed
        [0.0, 1.0, 0.4],   # Another close point - should be kept
        [1.0, 0.0, 0.9],   # Over threshold - should be removed
    ])

    # Create corresponding colors (RGB)
    colors = np.array([
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue
        [1.0, 1.0, 0.0],  # Yellow
        [1.0, 0.0, 1.0],  # Magenta
        [0.0, 1.0, 1.0],  # Cyan
        [0.5, 0.5, 0.5],  # Gray
    ])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def crop_point_cloud_by_depth(pcd, max_depth=0.8):
    """
    Crop point cloud by removing points with Z-coordinate (depth) greater than max_depth.
    This is the same function as in aff_exec_modified_crop.py
    """
    pcd_pos = np.asarray(pcd.points)
    pcd_colors = np.asarray(pcd.colors)

    # Filter points with depth (Z-coordinate) less than max_depth
    depth_mask = pcd_pos[:, 2] <= max_depth

    cropped_pcd = o3d.geometry.PointCloud()
    cropped_pcd.points = o3d.utility.Vector3dVector(pcd_pos[depth_mask])
    cropped_pcd.colors = o3d.utility.Vector3dVector(pcd_colors[depth_mask])

    return cropped_pcd

def test_crop_functionality():
    """Test the point cloud cropping functionality"""
    print("Testing point cloud cropping functionality...")

    # Create test point cloud
    test_pcd = create_test_point_cloud()
    original_points = len(test_pcd.points)
    print(f"\nOriginal point cloud: {original_points} points")
    print("Original points:")
    for i, point in enumerate(np.asarray(test_pcd.points)):
        print(f"  Point {i}: [{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}]")

    # Test cropping with max_depth = 0.8 meters
    cropped_pcd = crop_point_cloud_by_depth(test_pcd, max_depth=0.8)
    cropped_points = len(cropped_pcd.points)

    print(f"\nAfter cropping (max_depth = 0.8m): {cropped_points} points")
    print(f"Removed {original_points - cropped_points} points with depth > 0.8m")
    print("Remaining points:")
    for i, point in enumerate(np.asarray(cropped_pcd.points)):
        print(f"  Point {i}: [{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}]")

    # Verify the cropping worked correctly
    expected_kept_points = 4  # Points at indices 0, 1, 2, 5 should be kept
    if cropped_points == expected_kept_points:
        print(f"\n✓ Test PASSED: Correctly kept {expected_kept_points} points")
    else:
        print(f"\n✗ Test FAILED: Expected {expected_kept_points} points, but got {cropped_points}")

    # Check that all remaining points have Z <= 0.8
    remaining_points = np.asarray(cropped_pcd.points)
    all_within_threshold = np.all(remaining_points[:, 2] <= 0.8)

    if all_within_threshold:
        print("✓ All remaining points are within depth threshold (Z <= 0.8m)")
    else:
        print("✗ Some remaining points exceed depth threshold!")

    return cropped_points == expected_kept_points and all_within_threshold

if __name__ == "__main__":
    print("=" * 60)
    print("Point Cloud Cropping Functionality Test")
    print("=" * 60)

    success = test_crop_functionality()

    print("\n" + "=" * 60)
    if success:
        print("All tests PASSED! ✓")
        print("The cropping functionality is working correctly.")
    else:
        print("Some tests FAILED! ✗")
    print("=" * 60)