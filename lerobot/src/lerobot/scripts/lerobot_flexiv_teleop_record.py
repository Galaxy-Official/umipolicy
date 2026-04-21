import argparse
import time
import os
import torch
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
    features = robot.features.copy()
    features["observation.state"] = {
        "dtype": "float32", "shape": (10,), "names": ["x", "y", "z", "qx", "qy", "qz", "gripper", "none1", "none2", "none3"]
    }
    features["action"] = {
        "dtype": "float32", "shape": (10,), "names": ["x", "y", "z", "qx", "qy", "qz", "gripper", "none1", "none2", "none3"]
    }

    # ---------------------------------------------------------
    # Zero-Latency Monkey-patch 10D eef pose representation
    # ---------------------------------------------------------
    if hasattr(robot, "flexiv"):
        import math
        import lib_py.flexivrdk as flexivrdk
        
        def fast_new_get_state(*args):
            # One single rapid pull directly from RDK
            rs = flexivrdk.RobotStates()
            robot.flexiv.getRobotStates(rs)
            
            pose = rs.tcpPose
            qw, qx, qy, qz = pose[3], pose[4], pose[5], pose[6]
            
            # Pure native Python math for Quat -> RotVec (1000x faster than Scipy)
            norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
            if norm < 1e-12: norm = 1.0
            qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
            
            angle = 2 * math.acos(max(min(qw, 1.0), -1.0))
            sin_half = math.sqrt(max(1.0 - qw * qw, 0.0))
            if sin_half < 1e-6:
                rx, ry, rz = 0.0, 0.0, 0.0
            else:
                rx = qx / sin_half * angle
                ry = qy / sin_half * angle
                rz = qz / sin_half * angle
                
            robot_states_gripper = flexivrdk.GripperStates()
            robot.gripper.getGripperStates(robot_states_gripper)
            gripper = float(robot_states_gripper.width)
            
            eef_10d = [pose[0], pose[1], pose[2], rx, ry, rz, gripper, 0.0, 0.0, 0.0]
            return torch.tensor(eef_10d, dtype=torch.float32)
            
        # Instance method shadow ensures capture_observation leverages this natively without double-querying!
        robot.get_state = fast_new_get_state
        
        old_teleop = robot.teleop_step
        def new_teleop(*args, **kwargs):
            res = old_teleop(*args, **kwargs)
            if res is not None:
                obs, act = res
                # obs["observation.state"] is natively 10D thanks to fast_new_get_state
                target_action = obs["observation.state"].tolist() 
                try:
                    if hasattr(robot, "teleop") and robot.teleop is not None:
                        t_eef = robot.teleop.get_lead_arm_eef()
                        if t_eef is not None and len(t_eef) > 0:
                            t = t_eef[0]
                            gripper = target_action[6]
                            target_action = [t[0], t[1], t[2], t[3], t[4], t[5], gripper, 0.0, 0.0, 0.0]
                except Exception:
                    pass
                act["action"] = torch.tensor(target_action, dtype=torch.float32)
            return res
        robot.teleop_step = new_teleop
    # ---------------------------------------------------------

    if root is not None:
        root = Path(root) / repo_id

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

    from lerobot.common.robot_devices.control_utils_tactile import is_headless
    import cv2
    from pynput import keyboard

    if not robot.is_connected:
        robot.connect()

    state = {
        "is_recording": False, 
        "save_episode": False, 
        "discard_episode": False, 
        "exit_app": False
    }
    
    timers = {"last_space": 0.0}

    def on_press(key):
        try:
            if key == keyboard.Key.space:
                now = time.time()
                if now - timers["last_space"] < 0.4:
                    state["save_episode"] = True
                    timers["last_space"] = 0.0 
                else:
                    state["is_recording"] = not state["is_recording"]
                    if state["is_recording"]:
                        logger.opt(colors=True).info("<green>==> Recording state: [RECORDING]</green>")
                    else:
                        logger.opt(colors=True).warning("<red>==> Recording state: [PAUSED]</red>")
                    timers["last_space"] = now
            elif key == keyboard.Key.left:
                logger.warning("==> Discarding current episode!")
                state["discard_episode"] = True
            elif key == keyboard.Key.esc:
                logger.warning("==> Exiting application!")
                state["exit_app"] = True
        except Exception:
            pass

    listener = None
    if not is_headless():
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
    else:
        logger.warning("Headless environment detected! Keyboard bindings will not work!")

    logger.info("Initializing Home Position (Only running ONCE)...")
    robot.set_robot_home_position()
    
    recorded_episodes = 0
    logger.info("-------------------- Controls --------------------")
    logger.info(" [SPACE] x1: Start / Pause recording (Toggle)")
    logger.info(" [SPACE] x2: Save current recording & Add Episode +1")
    logger.info(" [LEFT]    : Discard currently unsaved recording")
    logger.info(" [ESC]     : Exit complete program and process dataset")
    logger.info("--------------------------------------------------")
    logger.opt(colors=True).warning("<red>=> Current State: [PAUSED] (You can move arm freely. Press SPACE to start saving frames)</red>")

    last_render_time = time.perf_counter()
    visual_queue = None
    visual_process = None
    if display_cameras and not is_headless():
        import multiprocessing as mp
        from lerobot.scripts.camera_viewer import viewer_process
        mp.set_start_method('spawn', force=True)
        visual_queue = mp.Queue(maxsize=1)
        visual_process = mp.Process(target=viewer_process, args=(visual_queue,))
        visual_process.daemon = True
        visual_process.start()

    while not state["exit_app"] and recorded_episodes < num_episodes:
        start_loop_t = time.perf_counter()

        observation, action = robot.teleop_step(record_data=True)

        if state["is_recording"]:
            frame = {**observation, **action}
            if task is not None:
                frame["task"] = task
            dataset.add_frame(frame)

        if display_cameras and not is_headless() and visual_queue is not None:
            now_t = time.perf_counter()
            if now_t - last_render_time >= 0.033: # max 30 FPS UI refresh
                image_keys = [k for k in observation if "image" in k]
                render_dict = {}
                for k in image_keys:
                    render_dict[k] = cv2.cvtColor(observation[k].numpy(), cv2.COLOR_RGB2BGR)
                try:
                    if not visual_queue.full():
                        visual_queue.put_nowait(render_dict)
                except Exception:
                    pass
                last_render_time = now_t

        if state["discard_episode"]:
            dataset.clear_episode_buffer()
            state["is_recording"] = False
            state["discard_episode"] = False
            logger.opt(colors=True).warning("<red>=> Episode discarded. Returned to [PAUSED]</red>")

        if state["save_episode"]:
            try:
                dataset.save_episode(task)
                recorded_episodes += 1
                logger.success(f"=== Episode {recorded_episodes} Saved Successfully! ===")
            except Exception as e:
                logger.error(f"Failed to save episode: {e}. Is the buffer empty?")
            
            state["is_recording"] = False
            state["save_episode"] = False
            logger.opt(colors=True).warning("<red>=> Current State: [PAUSED] (Move arms freely to next target)</red>")

        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

    # ---------------------------------------------------------
    # Auto-save buffer if ESC was pressed while still recording
    # ---------------------------------------------------------
    try:
        if state["is_recording"]:
            dataset.save_episode(task)
            recorded_episodes += 1
            logger.success(f"=== Auto-saved final partial episode {recorded_episodes} on exit! ===")
    except Exception as e:
        pass

    logger.info("Disconnecting and cleaning up...")
    robot.disconnect()
    if listener:
        listener.stop()
        
    if visual_process is not None:
        try:
            visual_queue.put_nowait("STOP")
            visual_process.join(timeout=1.0)
        except Exception:
            pass

    # In LeRobot 3.0, consolidate is strictly removed/deprecated!
    # dataset.consolidate(run_compute_stats)

    if push_to_hub:
        dataset.push_to_hub(tags=tags)

    # ---------------------------------
    # Added for lingbot-va compatibility
    # ---------------------------------
    logger.info("Injecting action_config for lingbot-va compatibility...")
    import json
    episodes_file = dataset.root / "meta" / "episodes.jsonl"
    if episodes_file.exists():
        new_lines = []
        with open(episodes_file, 'r') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                length = data.get("length", 0)
                if "tasks" not in data or not data["tasks"]:
                    data["tasks"] = [task if task else "perform the manipulation task"]
                
                data["action_config"] = [
                    {
                        "start_frame": 0,
                        "end_frame": length,
                        "action_text": data["tasks"][0],
                    }
                ]
                new_lines.append(json.dumps(data) + "\n")
        
        with open(episodes_file, 'w') as f:
            f.writelines(new_lines)
        logger.info(f"Successfully updated {episodes_file} to lingbot-va format.")

    logger.info(f"Exiting == {play_sounds}")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record a teleoperation session.")
    parser.add_argument(
        "--robot-path",
        type=str,
        default="configs/robot/koch_flexiv.yaml",
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

