#!/usr/bin/env python

import argparse
import logging
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.transform as st
import torch
from tqdm import tqdm, trange

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.pose_utils import rot6d_to_mat
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.utils import init_logging

def get_figure_canvas(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    # Convert RGBA to BGR for cv2
    img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    return img

def main():
    parser = argparse.ArgumentParser(description="Offline evaluation and visualization")
    parser.add_argument("--policy-path", type=str, required=True, help="Path to pre-trained policy")
    parser.add_argument("--repo-id", type=str, required=True, help="Dataset repo id")
    parser.add_argument("--root", type=str, default=None, help="Dataset root path")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of random episodes to evaluate")
    parser.add_argument("--output-video", type=str, default="offline_eval_eps", help="Output video root directory (will append run_name/checkpoint)")
    parser.add_argument("--fps", type=int, default=10, help="FPS of output video")
    parser.add_argument("--video-backend", type=str, default="pyav", help="Video backend to use for decoding videos")
    
    args = parser.parse_args()
    init_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Parse policy path to extract run_name and checkpoint for output directory
    policy_path_obj = Path(args.policy_path)
    run_name = "unknown_run"
    checkpoint_name = "unknown_ckpt"

    parts = policy_path_obj.parts
    if len(parts) >= 3:
        if parts[-2] == "checkpoints":
            run_name = parts[-3]
            checkpoint_name = parts[-1]
        elif len(parts) >= 4 and parts[-3] == "checkpoints":
            run_name = parts[-4]
            checkpoint_name = parts[-2]
        else:
            run_name = parts[-2]
            checkpoint_name = parts[-1]

    output_dir = Path(args.output_video) / run_name / checkpoint_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Videos will be saved to {output_dir}")

    # Load policy
    logging.info(f"Loading policy from {args.policy_path}")
    policy_config = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cls = get_policy_class(policy_config.type)
    policy = policy_cls.from_pretrained(args.policy_path, config=policy_config)
    policy.eval()
    policy.to(device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=args.policy_path,
        dataset_stats=None,
    )

    # Load full dataset metadata to pick episodes
    logging.info(f"Loading dataset {args.repo_id}")
    dataset = LeRobotDataset(args.repo_id, root=args.root, video_backend=args.video_backend)
    
    # Randomly select episodes
    total_episodes = dataset.meta.total_episodes
    if args.num_episodes > total_episodes:
        logging.warning(f"Requested {args.num_episodes} episodes, but only {total_episodes} available. Evaluating all.")
        args.num_episodes = total_episodes
        
    selected_episodes = random.sample(range(total_episodes), args.num_episodes)
    logging.info(f"Selected {args.num_episodes} random episodes: {selected_episodes}")

    obs_keys = policy.config.input_features.keys()
    
    fig = plt.figure(figsize=(16, 10))
    # 设置大图：3D, 以及三个小图 X-Y, Y-Z, X-Z, 和旋转向量、夹爪宽度
    ax_3d = fig.add_subplot(2, 3, 1, projection='3d')
    ax_xy = fig.add_subplot(2, 3, 2)
    ax_yz = fig.add_subplot(2, 3, 3)
    ax_xz = fig.add_subplot(2, 3, 4)
    ax_rot = fig.add_subplot(2, 3, 5)
    ax_grip = fig.add_subplot(2, 3, 6)

    for ep_idx in selected_episodes:
        # Re-initialize state per episode
        policy.reset()
        if preprocessor: preprocessor.reset()
        if postprocessor: postprocessor.reset()
        
        gt_traj = []
        pr_traj = []
        gt_gripper = []
        pr_gripper = []
        video_writer = None
        
        ep_start = dataset.meta.episodes[ep_idx]["dataset_from_index"]
        ep_end = dataset.meta.episodes[ep_idx]["dataset_to_index"]
        ep_len = ep_end - ep_start
        
        output_name = str(output_dir / f"eval_viz_{ep_idx:04d}.mp4")
        logging.info(f"Starting inference on episode {ep_idx} (frames {ep_start}-{ep_end}). Saving to {output_name}")
        
        pbar = tqdm(total=ep_len, desc=f"Ep {ep_idx}")
        
        for i in range(ep_start, ep_end):
            item = dataset[i]
            
            # Prepare observation
            obs_batch = {}
            for k in obs_keys:
                if k in item:
                    obs_batch[k] = item[k].unsqueeze(0).to(device)
                    
            # Apply Normalization
            obs_batch = preprocessor(obs_batch)
                    
            # Inference
            with torch.no_grad():
                action_pred_norm = policy.select_action(obs_batch)
                
            # Unnormalize Output
            action_pred_unnorm = postprocessor({"action": action_pred_norm})["action"]
                
            action_pred = action_pred_unnorm.squeeze(0).cpu().numpy()
            action_gt = item["action"].numpy()
            
            # 解析 Ground Truth Action (原始数据格式： [x, y, z, rx, ry, rz, gripper, pad, pad, pad])
            gt_pos = action_gt[:3]
            gt_rotvec = action_gt[3:6]
            gt_grip = action_gt[6] if len(action_gt) > 6 else 0.0
            gt_6d = np.concatenate([gt_pos, gt_rotvec])
            
            # 解析 Predicted Action (模型实际输出也是与 GT 相同的格式： [x, y, z, rx, ry, rz, gripper, pad, pad, pad])
            pr_pos = action_pred[:3]
            pr_rotvec = action_pred[3:6]
            pr_grip = action_pred[6] if len(action_pred) > 6 else 0.0
            
            pr_6d = np.concatenate([pr_pos, pr_rotvec])
            
            gt_traj.append(gt_6d)
            pr_traj.append(pr_6d)
            gt_gripper.append(gt_grip)
            pr_gripper.append(pr_grip)
            
            # --- Plotting --- #
            # We re-plot everything every N frames or just update, to be robust we clear and re-plot
            if (i - ep_start) % 2 == 0 or i == ep_end - 1: # small optimization
                ax_3d.clear()
                ax_xy.clear()
                ax_yz.clear()
                ax_xz.clear()
                ax_rot.clear()
                ax_grip.clear()
                
                gt_arr = np.array(gt_traj)
                pr_arr = np.array(pr_traj)
                gt_g_arr = np.array(gt_gripper)
                pr_g_arr = np.array(pr_gripper)
                time_steps = np.arange(len(gt_arr))
                
                # 3D: Z down, Y left, X inward. matplotlib default: X right, Y in, Z up
                ax_3d.plot(gt_arr[:, 0], gt_arr[:, 1], gt_arr[:, 2], label="GT", color='g', linewidth=2)
                ax_3d.plot(pr_arr[:, 0], pr_arr[:, 1], pr_arr[:, 2], label="Pred", color='r', linewidth=2)
                ax_3d.scatter(gt_arr[-1, 0], gt_arr[-1, 1], gt_arr[-1, 2], color='g', s=50)
                ax_3d.scatter(pr_arr[-1, 0], pr_arr[-1, 1], pr_arr[-1, 2], color='r', s=50)
                
                ax_3d.set_xlabel('X (inward)')
                ax_3d.set_ylabel('Y (left)')
                ax_3d.set_zlabel('Z (down)')
                # Invert Z to make it point downwards
                ax_3d.invert_zaxis()
                # Invert Y to make it point left
                ax_3d.invert_yaxis()
                ax_3d.legend()
                ax_3d.set_title(f'3D Trajectory (Ep {ep_idx})')
                
                # X-Y
                ax_xy.plot(gt_arr[:, 0], gt_arr[:, 1], color='g')
                ax_xy.plot(pr_arr[:, 0], pr_arr[:, 1], color='r')
                ax_xy.invert_yaxis()
                ax_xy.set_title('X-Y View')
                ax_xy.set_xlabel('X'); ax_xy.set_ylabel('Y (left)')
                
                # Y-Z
                ax_yz.plot(gt_arr[:, 1], gt_arr[:, 2], color='g')
                ax_yz.plot(pr_arr[:, 1], pr_arr[:, 2], color='r')
                ax_yz.invert_xaxis()
                ax_yz.invert_yaxis()
                ax_yz.set_title('Y-Z View')
                ax_yz.set_xlabel('Y (left)'); ax_yz.set_ylabel('Z (down)')
                
                # X-Z
                ax_xz.plot(gt_arr[:, 0], gt_arr[:, 2], color='g')
                ax_xz.plot(pr_arr[:, 0], pr_arr[:, 2], color='r')
                ax_xz.invert_yaxis()
                ax_xz.set_title('X-Z View')
                ax_xz.set_xlabel('X'); ax_xz.set_ylabel('Z (down)')
                
                # Rotvec
                ax_rot.plot(time_steps, gt_arr[:, 3], 'g--', label='GT Rx')
                ax_rot.plot(time_steps, gt_arr[:, 4], 'g-.', label='GT Ry')
                ax_rot.plot(time_steps, gt_arr[:, 5], 'g:', label='GT Rz')
                ax_rot.plot(time_steps, pr_arr[:, 3], 'r--', label='Pr Rx')
                ax_rot.plot(time_steps, pr_arr[:, 4], 'r-.', label='Pr Ry')
                ax_rot.plot(time_steps, pr_arr[:, 5], 'r:', label='Pr Rz')
                ax_rot.set_title('Rotation Vector (Rx, Ry, Rz)')
                ax_rot.set_xlabel('Steps')
                ax_rot.legend(fontsize='small')

                # Gripper
                ax_grip.plot(time_steps, gt_g_arr, 'g', label='GT Gripper')
                ax_grip.plot(time_steps, pr_g_arr, 'r', label='Pred Gripper')
                ax_grip.set_title('Gripper Width')
                ax_grip.set_xlabel('Steps')
                ax_grip.legend(fontsize='small')
                
                plot_img = get_figure_canvas(fig)
            
            # --- Observation Images --- #
            img_keys = [k for k in obs_keys if k.startswith("observation.images")]
            imgs = []
            for k in img_keys:
                img_tensor = item[k]
                # [C, H, W] to [H, W, C]
                img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                # RGB to BGR for cv2
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                imgs.append(img_np)
                
            if len(imgs) > 0:
                obs_strip = np.concatenate(imgs, axis=1)
            else:
                obs_strip = np.zeros((300, 300, 3), dtype=np.uint8)
                
            # Composite frame
            # We need to resize obs_strip to match the width of plot_img
            target_width = plot_img.shape[1]
            scale = target_width / obs_strip.shape[1]
            target_height = int(obs_strip.shape[0] * scale)
            obs_resized = cv2.resize(obs_strip, (target_width, target_height))
            
            composite = np.vstack([obs_resized, plot_img])
            
            if video_writer is None:
                h, w, _ = composite.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_name, fourcc, args.fps, (w, h))
                
            video_writer.write(composite)
            pbar.update(1)

        pbar.close()
        
        if video_writer is not None:
            video_writer.release()
            
        logging.info(f"Finished episode {ep_idx}.")

if __name__ == "__main__":
    main()
