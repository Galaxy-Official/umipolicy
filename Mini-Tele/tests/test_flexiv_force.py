import time
from lerobot.scripts.control_robot import busy_wait

from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.utils import Robot

from lerobot.common.robot_devices.control_utils import control_loop
from lerobot.common.utils.utils import init_hydra_config



def test_flexiv_force(
    robot: Robot,
    ):
    record_time_s = 30
    fps = 30
    if not robot.is_connected:
        robot.connect()
    robot.set_robot_home_position()

    observations = []
    for _ in range(record_time_s * fps):
        start_time = time.perf_counter()
        observation = robot.capture_observation()

        observations.append(observation)
        dt_s = time.perf_counter() - start_time
        busy_wait(1 / fps - dt_s)

    return observations

if __name__ == "__main__":
    robot_path = "lerobot/configs/robot/koch_flexiv.yaml"
    robot_overrides = None
    robot_cfg = init_hydra_config(robot_path, robot_overrides)
    robot = make_robot(robot_cfg)
    observations = test_flexiv_force(robot)