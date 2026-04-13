import logging
import threading
import time
from typing import Optional, Tuple, Union, Dict

import cv2
import numpy as np
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image, CompressedImage
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("ROS2 dependencies not available. RealSenseROS2 will not work.")

from PIL import Image as PILImage

from .base_camera import BaseCamera

logger = logging.getLogger(__name__)


class MultiCameraROS2Manager:
    """
    Multi-camera ROS 2 manager that handles three camera perspectives:
    external, left_wrist, right_wrist
    """
    
    def __init__(self, camera_config: Dict[str, str], image_resize_shape: Tuple[int, int] = None, debug: bool = False):
        """
        Initialize multi-camera ROS 2 client.
        
        Args:
            camera_config: Dict mapping perspective names to camera serial numbers
                          e.g., {"external": "036422060422", "left_wrist": "036422061234", "right_wrist": "036422065678"}
            image_resize_shape: Target size for image resizing (width, height)
            debug: Enable debug mode
        """
        if not ROS2_AVAILABLE:
            raise ImportError("ROS2 dependencies not available")
            
        self.camera_config = camera_config
        self.resize_shape = image_resize_shape
        self.debug = debug
        
        # Initialize ROS 2 if not already initialized
        if not rclpy.ok():
            rclpy.init()
        
        # Create ROS 2 node
        self.node = Node('multi_realsense_client')
        
        # Data storage for each camera perspective
        self.camera_data = {
            perspective: {
                'color_image': None,
                'depth_image': None,
                'timestamp': None
            } for perspective in camera_config.keys()
        }
        self.data_lock = threading.Lock()
        
        # CV bridge for image conversion
        self.cv_bridge = CvBridge()
        
        # QoS profile for reliable image streaming
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Create subscribers for each camera
        self.subscribers = {}
        for perspective, serial_number in camera_config.items():
            camera_name = self._get_camera_name_from_serial(serial_number)
            
            # Color image subscriber
            color_topic = f'/{camera_name}/color/image_raw'
            self.subscribers[f'{perspective}_color'] = self.node.create_subscription(
                Image,
                color_topic,
                lambda msg, p=perspective: self._color_callback(msg, p),
                qos_profile
            )
            
            # Depth image subscriber
            depth_topic = f'/{camera_name}/depth/points'
            self.subscribers[f'{perspective}_depth'] = self.node.create_subscription(
                CompressedImage,
                depth_topic,
                lambda msg, p=perspective: self._depth_callback(msg, p),
                qos_profile
            )
            
            logger.info(f"Subscribed to {perspective} camera topics: {color_topic}, {depth_topic}")
        
        # Start spinning in a separate thread
        self.spin_thread = threading.Thread(target=self._spin_node, daemon=True)
        self.spin_thread.start()
        
        # Wait for initial data
        self._wait_for_initial_data()
        
        logger.info("Multi-camera ROS2 manager initialized")
    
    def _get_camera_name_from_serial(self, serial_number: str) -> str:
        """Map camera serial number to camera name."""
        serial_to_name = {
            "036422060422": "camera_base",
            "036422061234": "camera_wrist", 
            "036422065678": "camera_head",
            # Add more mappings as needed
        }
        return serial_to_name.get(serial_number, f"camera_{serial_number}")
    
    def _color_callback(self, msg: Image, perspective: str):
        """Callback for color image messages."""
        try:
            # Convert ROS Image to OpenCV format (BGR)
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            with self.data_lock:
                self.camera_data[perspective]['color_image'] = cv_image
                self.camera_data[perspective]['timestamp'] = time.time()
        except Exception as e:
            logger.error(f"Error processing {perspective} color image: {e}")
    
    def _depth_callback(self, msg: CompressedImage, perspective: str):
        """Callback for compressed depth image messages."""
        try:
            # Decode compressed depth image
            np_arr = np.frombuffer(msg.data, np.uint8)
            depth_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            
            with self.data_lock:
                self.camera_data[perspective]['depth_image'] = depth_image
        except Exception as e:
            logger.error(f"Error processing {perspective} depth image: {e}")
    
    def _spin_node(self):
        """Spin the ROS 2 node in a separate thread."""
        try:
            rclpy.spin(self.node)
        except Exception as e:
            logger.error(f"Error spinning ROS 2 node: {e}")
    
    def _wait_for_initial_data(self, timeout: float = 10.0):
        """Wait for initial data from all cameras."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.data_lock:
                all_cameras_ready = all(
                    data['color_image'] is not None 
                    for data in self.camera_data.values()
                )
                if all_cameras_ready:
                    logger.info("Received initial data from all cameras")
                    return
            
            time.sleep(0.1)
        
        logger.warning("Timeout waiting for initial data from some cameras")
    
    @staticmethod
    def resize_image_by_size(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize image to specified size."""
        if size is None:
            return image
        return cv2.resize(image, size)
    
    def get_all_camera_data(self) -> Dict[str, np.ndarray]:
        """
        Get processed data from all cameras in format similar to reference code.
        
        Returns:
            Dictionary with processed camera data
        """
        obs_dict = {}
        
        with self.data_lock:
            # Process each camera perspective
            for perspective, data in self.camera_data.items():
                if data['color_image'] is not None:
                    # Convert BGR to RGB
                    color_image = cv2.cvtColor(data['color_image'], cv2.COLOR_BGR2RGB)
                    # Resize if needed
                    color_image = self.resize_image_by_size(color_image, self.resize_shape)
                    
                    # Add to observation dict using naming convention from reference
                    if perspective == 'external':
                        obs_dict['external_img'] = color_image
                        if data['depth_image'] is not None:
                            obs_dict['external_depth'] = data['depth_image']
                    elif perspective == 'left_wrist':
                        obs_dict['left_wrist_img'] = color_image
                        if data['depth_image'] is not None:
                            obs_dict['left_wrist_depth'] = data['depth_image']
                    elif perspective == 'right_wrist':
                        obs_dict['right_wrist_img'] = color_image
                        if data['depth_image'] is not None:
                            obs_dict['right_wrist_depth'] = data['depth_image']
                    
                    if self.debug:
                        logger.debug(f"Processed {perspective} image with shape: {color_image.shape}")
        
        return obs_dict
    
    def get_camera_data(self, perspective: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get data from a specific camera perspective.
        
        Args:
            perspective: Camera perspective name
            
        Returns:
            Tuple of (color_image, depth_image)
        """
        with self.data_lock:
            if perspective not in self.camera_data:
                logger.error(f"Unknown camera perspective: {perspective}")
                return None, None
            
            data = self.camera_data[perspective]
            if data['color_image'] is None:
                return None, None
            
            # Convert BGR to RGB and resize
            color_image = cv2.cvtColor(data['color_image'], cv2.COLOR_BGR2RGB)
            color_image = self.resize_image_by_size(color_image, self.resize_shape)
            
            depth_image = data['depth_image']
            
            return color_image, depth_image


@BaseCamera.register_camera("realsense_ros2")
class RealSenseROS2(BaseCamera):
    """
    ROS 2 client-based RealSense camera interface - single camera version
    for backward compatibility with BaseCamera interface.
    """
    
    def __init__(self, camera_sn: str):
        """Initialize single ROS 2 RealSense camera client."""
        if not ROS2_AVAILABLE:
            raise ImportError("ROS2 dependencies not available")
            
        # Use multi-camera manager with single camera
        self.camera_manager = MultiCameraROS2Manager({"main": camera_sn})
        self.camera_sn = camera_sn
        
        # Camera intrinsics (default values)
        self.mtx = np.array([
            [615.0, 0, 320.0],
            [0, 615.0, 240.0],
            [0, 0, 1.0]
        ])
        
        logger.info(f"RealSense ROS2 client initialized for camera: {camera_sn}")
    
    def get_data(self, return_type: str = "PIL") -> Tuple[Union[PILImage.Image, np.ndarray], Optional[np.ndarray]]:
        """Get the latest color and depth images."""
        color_image, depth_image = self.camera_manager.get_camera_data("main")
        
        if color_image is None:
            logger.warning("No color image available")
            return None, None
        
        if return_type == "PIL":
            return PILImage.fromarray(color_image), depth_image
        elif return_type == "numpy":
            return color_image, depth_image
        else:
            raise ValueError(f"Unsupported return_type: {return_type}")
    
    def inpaint(self, img: np.ndarray, missing_value: int = 0) -> np.ndarray:
        """Inpaint missing values in depth image."""
        # cv2 inpainting doesn't handle the border properly
        img_padded = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        mask = (img_padded == missing_value).astype(np.uint8)
        
        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        scale = np.abs(img_padded).max()
        if scale > 0:
            img_normalized = img_padded.astype(np.float32) / scale
            img_inpainted = cv2.inpaint(img_normalized, mask, 1, cv2.INPAINT_NS)
            # Back to original size and value range.
            img_result = img_inpainted[1:-1, 1:-1] * scale
            return img_result
        return img
    
    def getXYZRGB(self, color: np.ndarray, depth: np.ndarray, robot_pose: np.ndarray, 
                  camera_pose: np.ndarray, cam_intrinsics: np.ndarray, inpaint: bool = True) -> np.ndarray:
        """Convert color and depth images to XYZRGB point cloud."""
        try:
            import open3d as o3d
        except ImportError:
            logger.error("open3d not available for point cloud generation")
            return np.array([])

        if inpaint:
            depth = self.inpaint(depth)
            
        color_image = o3d.geometry.Image(color)
        depth_image = o3d.geometry.Image(depth)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image, depth_image, convert_rgb_to_intensity=False
        )

        fx, fy, cx, cy = (
            cam_intrinsics[0, 0], cam_intrinsics[1, 1],
            cam_intrinsics[0, 2], cam_intrinsics[1, 2]
        )
        width, height = color.shape[1], color.shape[0]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        world_pose = np.dot(robot_pose, camera_pose)
        pcd.transform(world_pose)

        xyz = np.asarray(pcd.points)
        rgb = np.asarray(pcd.colors)
        return np.hstack((xyz, rgb))
    
    def getleft(self, obj1: np.ndarray) -> np.ndarray:
        """Filter point cloud to left region."""
        index = np.bitwise_and(obj1[:, 0] < 1.2, obj1[:, 0] > 0.2)
        index = np.bitwise_and(obj1[:, 1] < 0.5, index)
        index = np.bitwise_and(obj1[:, 1] > -0.5, index)
        index = np.bitwise_and(obj1[:, 2] > 0.35, index)
        index = np.bitwise_and(obj1[:, 2] < 0.7, index)
        return obj1[index]
    
    def getIntrinsics(self) -> np.ndarray:
        """Get camera intrinsics matrix."""
        return self.mtx
    
    def __del__(self):
        """Cleanup ROS 2 resources."""
        try:
            if hasattr(self, 'camera_manager') and hasattr(self.camera_manager, 'node'):
                self.camera_manager.node.destroy_node()
        except Exception as e:
            logger.debug(f"Cleanup warning: {e}")


# Convenience functions
def create_realsense_ros2_camera(camera_sn: str) -> RealSenseROS2:
    """Create a ROS 2 RealSense camera client."""
    return RealSenseROS2(camera_sn)


def create_multi_camera_manager(camera_config: Dict[str, str], 
                               image_resize_shape: Tuple[int, int] = None, 
                               debug: bool = False) -> MultiCameraROS2Manager:
    """
    Create a multi-camera ROS 2 manager for three perspectives.
    
    Args:
        camera_config: Dict mapping perspective names to camera serial numbers
        image_resize_shape: Target size for image resizing (width, height)  
        debug: Enable debug mode
        
    Returns:
        MultiCameraROS2Manager instance
        
    Example:
        camera_config = {
            "external": "036422060422",
            "left_wrist": "036422061234", 
            "right_wrist": "036422065678"
        }
        manager = create_multi_camera_manager(camera_config, (640, 480), debug=True)
        obs_dict = manager.get_all_camera_data()
    """
    return MultiCameraROS2Manager(camera_config, image_resize_shape, debug)


if __name__ == "__main__":
    # Test the multi-camera ROS2 setup
    print("Testing multi-camera ROS2 setup...")
    
    # Example camera configuration - update with your serial numbers
    camera_config = {
        "external": "036422060422",
        "left_wrist": "036422061234", 
        "right_wrist": "036422065678"
    }
    
    try:
        # Test multi-camera manager
        print("Creating multi-camera manager...")
        manager = create_multi_camera_manager(
            camera_config, 
            image_resize_shape=(640, 480), 
            debug=True
        )
        
        print("Starting data collection loop...")
        for i in range(10):
            obs_dict = manager.get_all_camera_data()
            
            if 'external_img' in obs_dict:
                print(f"External image shape: {obs_dict['external_img'].shape}")
            if 'left_wrist_img' in obs_dict:
                print(f"Left wrist image shape: {obs_dict['left_wrist_img'].shape}")
            if 'right_wrist_img' in obs_dict:
                print(f"Right wrist image shape: {obs_dict['right_wrist_img'].shape}")
                
            time.sleep(1)
            
        print("Multi-camera test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Make sure your ROS 2 camera publishers are running and camera serial numbers are correct.")
    
    print("Test finished.")