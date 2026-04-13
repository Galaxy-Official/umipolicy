from abc import ABC, abstractmethod
import json

class BaseCamera(ABC):
    camera_registry = {}

    @abstractmethod
    def __init__(self, camera_id: str):
        pass

    @abstractmethod
    def get_data(self):
        pass

    @classmethod
    def register_camera(cls, camera_name):
        def wapper(subclass):
            cls.camera_registry[camera_name] = subclass
            return subclass

        return wapper

    @classmethod
    def create_camera(cls, camera_name: str, camera_sn: str, camera_v4l_path:str, fps:int, resolution):
        if camera_name not in cls.camera_registry:
            raise ValueError(
                f"Camera with key '{camera_name}' is not registered."
            )
        task_class = cls.camera_registry[camera_name]
        if camera_name == "realsense":
            return task_class(camera_sn)
        else:
            return task_class(camera_sn, camera_v4l_path, fps, resolution)

    @classmethod
    def create_cameras_from_config(cls, config_path: str = "perception/config/camera/camera.json"):
        with open(config_path, "r") as f:
            config = json.load(f)
        
        camera_dict = {}
        for view, camera_info in config.items():
            if camera_info["type"] == "realsense":
                camera = cls.create_camera(
                    camera_info["type"],
                    camera_info["sn"],
                )
            elif camera_info["type"] == "mvs_cam":
                from perception.cameras.mvs_cam import MVSCamera
                camera = MVSCamera({
                    "serial": camera_info.get("serial"),
                    "exposure": camera_info.get("exposure")
                })
            else:
                camera = cls.create_camera(
                    camera_info["type"],
                    camera_info.get("camera_index"),
                    camera_info.get("camera_v4l_path"),
                    camera_info.get("fps"),
                    camera_info.get("resolution")
                )
                
            camera_dict[view] = camera
        return camera_dict

# TODO test
if __name__ == "__main__":
    camera_dict = BaseCamera.create_multi_cameras_from_config(["wrist", "base"])
    for key, camera in camera_dict.items():
        camera.get_data()