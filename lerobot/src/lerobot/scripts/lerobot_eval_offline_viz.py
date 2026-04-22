#!/usr/bin/env python

import argparse
import logging
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm, trange

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.utils.utils import init_logging

def get_figure_canvas(fig):
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # matplotlib uses RGB, cv2 uses BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def main():
    parser = argparse.ArgumentParser(description="Offline evaluation and visualization")
    parser.add_argument("--policy-path", type=str, required=True, help="Path to pre-trained policy")
    parser.add_argument("--repo-id", type=str, required=True, help="Dataset repo id")
    parser.add_argument("--root", type=str, default=None, help="Dataset root path")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of random episodes to evaluate")
    parser.add_argument("--output-video", type=str, default="offline_eval_eps", help="Output video file prefix")
    parser.add_argument("--fps", type=int, default=10, help="FPS of output video")
    
    args = parser.parse_args()
    init_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load policy
    logging.info(f"Loading policy from {args.policy_path}")
    policy_config = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cls = get_policy_class(policy_config.type)
    policy = policy_cls.from_pretrained(args.policy_path, config=policy_config)
    policy.eval()
    policy.to(device)

    # Load full dataset metadata to pick episodes
    logging.info(f"Loading dataset {args.repo_id}")
    dataset = LeRobotDataset(args.repo_id, root=args.root)
    
    # Randomly select episodes
    total_episodes = dataset.meta.total_episodes
    if args.num_episodes > total_episodes:
        logging.warning(f"Requested {args.num_episodes} episodes, but only {total_episodes} available. Evaluating all.")
        args.num_episodes = total_episodes
        
    selected_episodes = random.sample(range(total_episodes), args.num_episodes)
    logging.info(f"Selected {args.num_episodes} random episodes: {selected_episodes}")

    obs_keys = policy.config.input_features.keys()
    
    fig = plt.figure(figsize=(12, 10))
    # 设置大图：3D, 以及三个小图 X-Y, Y-Z, X-Z
    ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_yz = fig.add_subplot(2, 2, 3)
    ax_xz = fig.add_subplot(2, 2, 4)

    for ep_idx in selected_episodes:
        # Re-initialize state per episode
        policy.reset()
        gt_traj = []
        pr_traj = []
        video_writer = None
        
        ep_start = dataset.episode_data_index["from"][ep_idx].item()
        ep_end = dataset.episode_data_index["to"][ep_idx].item()
        ep_len = ep_end - ep_start
        
        output_name = f"{args.output_video}_{ep_idx:04d}.mp4"
        logging.info(f"Starting inference on episode {ep_idx} (frames {ep_start}-{ep_end}). Saving to {output_name}")
        
        pbar = tqdm(total=ep_len, desc=f"Ep {ep_idx}")
        
        for i in range(ep_start, ep_end):
            item = dataset[i]
            
            # Prepare observation
            obs_batch = {}
            for k in obs_keys:
                if k in item:
                    obs_batch[k] = item[k].unsqueeze(0).to(device)
                    
            # Inference
            with torch.no_grad():
                action_pred = policy.select_action(obs_batch)
                
            action_pred = action_pred.squeeze(0).cpu().numpy()
            action_gt = item["action"].numpy()
            
            # Assuming XYZ are the first 3 dims
            gt_x, gt_y, gt_z = action_gt[0], action_gt[1], action_gt[2]
            pr_x, pr_y, pr_z = action_pred[0], action_pred[1], action_pred[2]
            
            gt_traj.append([gt_x, gt_y, gt_z])
            pr_traj.append([pr_x, pr_y, pr_z])
            
            # --- Plotting --- #
            # We re-plot everything every N frames or just update, to be robust we clear and re-plot
            if (i - ep_start) % 2 == 0 or i == ep_end - 1: # small optimization
                ax_3d.clear()
                ax_xy.clear()
                ax_yz.clear()
                ax_xz.clear()
                
                gt_arr = np.array(gt_traj)
                pr_arr = np.array(pr_traj)
                
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
