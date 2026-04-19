import argparse
import time
import os
from pathlib import Path
from typing import List
from loguru import logger

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.control_utils_tactile import (
    control_loop,
    has_method,
    init_keyboard_listener,
    init_policy,
    log_control_info,
    record_episode,
    reset_environment,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
    stop_recording,
    warmup_record,
)
from lerobot.common.robot_devices.MiniTeleop_utils import LeadArmReader
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.utils.utils import init_hydra_config, none_or_int
from lerobot.utils.utils import log_say, init_logging

@safe_disconnect
def record(
    robot: Robot,
    root: Path,
    repo_id: str,
    single_task: str,
    fps: int | None = None,
    warmup_time_s: int | float = 2,
    episode_time_s: int | float = 10,
    reset_time_s: int | float = 5,
    num_episodes: int = 50,
    video: bool = True,
    run_compute_stats: bool = True,
    push_to_hub: bool = True,
    tags: list[str] | None = None,
    num_image_writer_processes: int = 0,
    num_image_writer_threads_per_camera: int = 4,
    display_cameras: bool = True,
    play_sounds: bool = True,
    resume: bool = False,
    local_files_only: bool = False,
) -> LeRobotDataset:
    listener = None
    events = None
    policy = None
    device = None
    use_amp = None

    if single_task:
        task = single_task
    else:
        raise NotImplementedError("Only single-task recording is supported for now")

    # Features are implicitly retrieved from old robot
    features = robot.features

    if resume:
        dataset = LeRobotDataset(
            repo_id,
            root=root,
            local_files_only=local_files_only,
        )
        dataset.start_image_writer(
            num_processes=num_image_writer_processes,
            num_threads=num_image_writer_threads_per_camera * len(robot.cameras),
        )
        # sanity_check_dataset_robot_compatibility(dataset, robot, fps, video)
    else:
        # Create empty dataset using the explicit features format mandated by 3.0
        # sanity_check_dataset_name(repo_id, policy)
        
        # LeRobot 3.0 API
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=root,
            robot_type=getattr(robot, 'robot_type', 'flexiv_handcap'),
            use_videos=video,
            image_writer_processes=num_image_writer_processes,
            image_writer_threads=num_image_writer_threads_per_camera * len(robot.cameras),
        )

    if not robot.is_connected:
        robot.connect()

    listener, events = init_keyboard_listener()

    enable_teleoperation = policy is None
    logger.info(f"Warmup record == {play_sounds}")
    
    warmup_record(
        robot,
        events,
        enable_teleoperation,
        warmup_time_s,
        display_cameras,
        fps
    )
    
    logger.info("Warmup Record End =====")
    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    recorded_episodes = 0
    while True:
        if recorded_episodes >= num_episodes:
            break

        robot.set_robot_home_position()

        logger.info(f"Recording episode {dataset.num_episodes} == {play_sounds}")
        logger.info(f"Per episodes continues for {episode_time_s} s.")
        
        # record_episode natively calls dataset.add_frame({**observation, **action})
        record_episode(
            dataset=dataset,
            robot=robot,
            events=events,
            episode_time_s=episode_time_s,
            display_cameras=display_cameras,
            policy=policy,
            device=device,
            use_amp=use_amp,
            fps=fps,
        )

        if not events["stop_recording"] and (
            (dataset.num_episodes < num_episodes - 1) or events["rerecord_episode"]
        ):
            logger.info(f"Reset the environment == {play_sounds}")
            reset_environment(robot, events, reset_time_s)

        if events["rerecord_episode"]:
            logger.info(f"Re-record episode == {play_sounds}")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        # LeRobot 3.0 explicitly requires save_episode
        dataset.save_episode(task)
        recorded_episodes += 1

        if events["stop_recording"]:
            break

    logger.info(f"Stop recording {play_sounds} blocking={True}")
    stop_recording(robot, listener, display_cameras)

    # In LeRobot 3.0, consolidate is strictly removed/deprecated!
    # dataset.consolidate(run_compute_stats)

    if push_to_hub:
        dataset.push_to_hub(tags=tags)

    logger.info(f"Exiting == {play_sounds}")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record a teleoperation session.")
    parser.add_argument(
        "--robot-path",
        type=str,
        default="lerobot/configs/robot/koch.yaml",
        help="Path to robot yaml file used to instantiate the robot using `make_robot` factory function.",
    )
    parser.add_argument(
        "--robot-overrides",
        type=str,
        default=None,
        help="Path to robot overrides json file used to instantiate the robot using `make_robot` factory function.",
    )
    parser.add_argument(
        "--fps", type=none_or_int, default=None, help="Frames per second (set to None to disable)"
    )
    parser.add_argument(
        "--single-task",
        type=str,
        default="Task description",
        help="A short but accurate description of the task.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/test",
    )
    parser.add_argument(
        "--local-files-only",
        type=int,
        default=0,
    )
    parser.add_argument("--teleop", type=str, default="koch", help="Leader arm type")
    parser.add_argument("--teleop_port", type=str, default="/dev/ttyUSB0", help="Leader arm port")
    parser.add_argument("--use_tactile", action="store_true", help="Enable tactile cameras")
    parser.add_argument("--warmup-time-s", "--warmup_time_s", dest="warmup_time_s", type=int, default=10)
    parser.add_argument("--episode-time-s", "--episode_time_s", dest="episode_time_s", type=int, default=60)
    parser.add_argument("--reset-time-s", "--reset_time_s", dest="reset_time_s", type=int, default=60)
    parser.add_argument("--num-episodes", "--num_episodes", dest="num_episodes", type=int, default=50)
    parser.add_argument("--run-compute-stats", type=int, default=1)
    parser.add_argument("--push-to-hub", type=int, default=1)
    parser.add_argument("--tags", type=str, nargs="*")
    parser.add_argument("--num-image-writer-processes", type=int, default=0)
    parser.add_argument("--num-image-writer-threads-per-camera", type=int, default=4)
    parser.add_argument("--resume", type=int, default=0)
    args = parser.parse_args()
    
    init_logging()
    logger.add("logs/Teleop-Record-{time:YYYY-MM-DD-hh-mm}.log", rotation="1 month", level="INFO")
    
    robot_path = args.robot_path
    robot_overrides = args.robot_overrides
    kwargs = vars(args)
    del kwargs["robot_path"]
    del kwargs["robot_overrides"]
    del kwargs["teleop"]
    del kwargs["teleop_port"]
    del kwargs["use_tactile"]

    robot_cfg = init_hydra_config(robot_path, robot_overrides)
    robot = make_robot(robot_cfg)
    record(robot, **kwargs)

