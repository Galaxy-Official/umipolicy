#!/usr/bin/env python3
"""
Example usage of the multi-camera ROS2 RealSense system.

This example shows how to:
1. Set up a multi-camera manager for three perspectives (external, left_wrist, right_wrist)
2. Get processed data in the format similar to the DataPostProcessingManager reference
3. Use single camera compatibility mode with BaseCamera interface
"""

import time
import numpy as np
from typing import Dict

from realsense_ros2 import (
    create_multi_camera_manager, 
    create_realsense_ros2_camera,
    MultiCameraROS2Manager
)
from base_camera import BaseCamera


def example_multi_camera_usage():
    """Example of using the multi-camera ROS2 manager."""
    print("=== Multi-Camera ROS2 Example ===")
    
    # Configure your camera serial numbers here
    camera_config = {
        "external": "036422060422",      # External/base camera
        "left_wrist": "036422061234",    # Left wrist camera  
        "right_wrist": "036422065678"    # Right wrist camera
    }
    
    # Create multi-camera manager
    manager = create_multi_camera_manager(
        camera_config=camera_config,
        image_resize_shape=(640, 480),  # Resize all images to this size
        debug=True
    )
    
    print("Multi-camera manager created. Starting data collection...")
    
    try:
        for i in range(5):
            # Get all camera data in obs_dict format (similar to reference code)
            obs_dict = manager.get_all_camera_data()
            
            print(f"\n--- Frame {i+1} ---")
            
            # Process external camera
            if 'external_img' in obs_dict:
                print(f"External camera: {obs_dict['external_img'].shape}")
                if 'external_depth' in obs_dict:
                    print(f"External depth: {obs_dict['external_depth'].shape}")
            
            # Process left wrist camera
            if 'left_wrist_img' in obs_dict:
                print(f"Left wrist camera: {obs_dict['left_wrist_img'].shape}")
                if 'left_wrist_depth' in obs_dict:
                    print(f"Left wrist depth: {obs_dict['left_wrist_depth'].shape}")
            
            # Process right wrist camera  
            if 'right_wrist_img' in obs_dict:
                print(f"Right wrist camera: {obs_dict['right_wrist_img'].shape}")
                if 'right_wrist_depth' in obs_dict:
                    print(f"Right wrist depth: {obs_dict['right_wrist_depth'].shape}")
            
            # You can also get individual camera data
            external_color, external_depth = manager.get_camera_data("external")
            if external_color is not None:
                print(f"Individual external camera access: {external_color.shape}")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping data collection...")
    
    print("Multi-camera example completed!")


def example_single_camera_compatibility():
    """Example showing backward compatibility with BaseCamera interface."""
    print("\n=== Single Camera Compatibility Example ===")
    
    try:
        # Create single camera using BaseCamera factory
        camera = BaseCamera.create_camera("realsense_ros2", "036422060422")
        
        print("Single camera created via BaseCamera interface")
        
        for i in range(3):
            # Use familiar BaseCamera interface
            color_img, depth_img = camera.get_data(return_type="numpy")
            
            if color_img is not None:
                print(f"Frame {i+1} - Color: {color_img.shape}")
                if depth_img is not None:
                    print(f"Frame {i+1} - Depth: {depth_img.shape}")
            
            # Test other methods
            intrinsics = camera.getIntrinsics()
            print(f"Camera intrinsics shape: {intrinsics.shape}")
            
            time.sleep(1)
            
    except Exception as e:
        print(f"Single camera example error: {e}")
    
    print("Single camera compatibility example completed!")


def example_data_processing_integration():
    """Example showing integration with data processing similar to reference code."""
    print("\n=== Data Processing Integration Example ===")
    
    camera_config = {
        "external": "036422060422",
        "left_wrist": "036422061234", 
        "right_wrist": "036422065678"
    }
    
    manager = create_multi_camera_manager(
        camera_config=camera_config,
        image_resize_shape=(224, 224),  # Smaller size for processing
        debug=False
    )
    
    # Simulate data processing workflow
    obs_dict = manager.get_all_camera_data()
    
    # Add timestamp (similar to reference code)
    obs_dict['timestamp'] = np.array([time.time()])
    
    # Process images (example operations)
    for key, image in obs_dict.items():
        if key.endswith('_img') and isinstance(image, np.ndarray):
            # Example: normalize images
            normalized_img = image.astype(np.float32) / 255.0
            obs_dict[f'{key}_normalized'] = normalized_img
            print(f"Processed {key}: {image.shape} -> normalized")
    
    print(f"Total observation keys: {list(obs_dict.keys())}")
    print("Data processing integration example completed!")


if __name__ == "__main__":
    print("ROS2 Multi-Camera RealSense Usage Examples")
    print("==========================================")
    print("Make sure your ROS2 camera publishers are running before testing!")
    print("Update camera serial numbers in the examples to match your setup.\n")
    
    try:
        # Run examples
        example_multi_camera_usage()
        example_single_camera_compatibility() 
        example_data_processing_integration()
        
    except ImportError:
        print("ROS2 dependencies not available. Please install:")
        print("- rclpy")
        print("- sensor_msgs") 
        print("- cv_bridge")
        
    except Exception as e:
        print(f"Example error: {e}")
        print("Make sure your camera publishers are running and serial numbers are correct.")
    
    print("\nAll examples completed!")