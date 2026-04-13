from typing import Dict, Callable, Tuple, List
import numpy as np
import collections
from lerobot.scripts.umi_realworld.utils.cv2_util import get_image_transform

from lerobot.scripts.umi_realworld.utils.pose_repr_util import (
    compute_relative_pose, 
    convert_pose_mat_rep
)
from lerobot.umi.common.pose_util import (
    pose_to_mat, mat_to_pose, 
    certain_pose_type_to_mat, mat_to_certain_pose_type)


def get_real_obs_resolution(
        shape_meta: dict
        ) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            co,ho,wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res


def get_real_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            this_imgs_in = env_obs[key]
            t,hi,wi,ci = this_imgs_in.shape
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
        elif type == 'low_dim':
            this_data_in = env_obs[key]
            obs_dict_np[key] = this_data_in
    return obs_dict_np


def get_real_umi_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        obs_pose_repr: str='abs',
        pose_type: str="10d",
        tx_robot1_robot0: np.ndarray=None,
        episode_start_pose: List[np.ndarray]=None,
        use_first_as_relative_pose_base = False,
        joint = False
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    # process non-pose
    obs_shape_meta = shape_meta['obs']
    robot_prefix_map = collections.defaultdict(list)
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            this_imgs_in = env_obs[key]
            t,hi,wi,ci = this_imgs_in.shape
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
        elif type == 'low_dim' and ('eef' not in key):
            this_data_in = env_obs[key]
            obs_dict_np[key] = this_data_in
            # handle multi-robots
            ks = key.split('_')
            if ks[0].startswith('robot'):
                robot_prefix_map[ks[0]].append(key)
    if joint:
        return obs_dict_np

    # generate relative pose
    for robot_prefix in robot_prefix_map.keys():
        # convert pose to mat
        pose_mat = certain_pose_type_to_mat(np.concatenate([
            env_obs[robot_prefix + '_eef_pos'],
            env_obs[robot_prefix + '_eef_rot_axis_angle']
        ], axis=-1), pose_type=pose_type)

        # solve reltaive obs
        if use_first_as_relative_pose_base:
            obs_pose_mat = convert_pose_mat_rep(
                pose_mat, 
                base_pose_mat=pose_mat[0],
                pose_rep=obs_pose_repr,
                backward=False)
        else:
            obs_pose_mat = convert_pose_mat_rep(
                pose_mat, 
                base_pose_mat=pose_mat[-1],
                pose_rep=obs_pose_repr,
                backward=False)

        obs_pose = mat_to_certain_pose_type(obs_pose_mat, pose_type)
        obs_dict_np[robot_prefix + '_eef_pos'] = obs_pose[...,:3]
        obs_dict_np[robot_prefix + '_eef_rot_axis_angle'] = obs_pose[...,3:]
    
    # generate pose relative to other robot
    n_robots = len(robot_prefix_map)
    for robot_id in range(n_robots):
        # convert pose to mat
        assert f'robot{robot_id}' in robot_prefix_map
        tx_robota_tcpa = certain_pose_type_to_mat(np.concatenate([
            env_obs[f'robot{robot_id}_eef_pos'],
            env_obs[f'robot{robot_id}_eef_rot_axis_angle']
        ], axis=-1), pose_type=pose_type)
        for other_robot_id in range(n_robots):
            if robot_id == other_robot_id:
                continue
            tx_robotb_tcpb = certain_pose_type_to_mat(np.concatenate([
                env_obs[f'robot{other_robot_id}_eef_pos'],
                env_obs[f'robot{other_robot_id}_eef_rot_axis_angle']
            ], axis=-1), pose_type=pose_type)
            tx_robota_robotb = tx_robot1_robot0
            if robot_id == 0:
                tx_robota_robotb = np.linalg.inv(tx_robot1_robot0)
            tx_robota_tcpb = tx_robota_robotb @ tx_robotb_tcpb

            rel_obs_pose_mat = convert_pose_mat_rep(
                tx_robota_tcpa,
                base_pose_mat=tx_robota_tcpb[-1],
                pose_rep='relative',
                backward=False)
            rel_obs_pose = mat_to_certain_pose_type(rel_obs_pose_mat, pose_type)
            obs_dict_np[f'robot{robot_id}_eef_pos_wrt{other_robot_id}'] = rel_obs_pose[:,:3]
            obs_dict_np[f'robot{robot_id}_eef_rot_axis_angle_wrt{other_robot_id}'] = rel_obs_pose[:,3:]

    # generate relative pose with respect to episode start
    if episode_start_pose is not None:
        for robot_id in range(n_robots):        
            # convert pose to mat
            pose_mat = certain_pose_type_to_mat(np.concatenate([
                env_obs[f'robot{robot_id}_eef_pos'],
                env_obs[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1), pose_type=pose_type)
            
            # get start pose
            start_pose = episode_start_pose[robot_id]
            start_pose_mat = certain_pose_type_to_mat(start_pose, pose_type=pose_type)
            rel_obs_pose_mat = convert_pose_mat_rep(
                pose_mat,
                base_pose_mat=start_pose_mat,
                pose_rep='relative',
                backward=False)
            
            rel_obs_pose = mat_to_certain_pose_type(rel_obs_pose_mat, pose_type)
            # obs_dict_np[f'robot{robot_id}_eef_pos_wrt_start'] = rel_obs_pose[:,:3]
            obs_dict_np[f'robot{robot_id}_eef_rot_axis_angle_wrt_start'] = rel_obs_pose[:,3:]

    return obs_dict_np

def get_real_umi_action(
        action: np.ndarray,
        env_obs: Dict[str, np.ndarray], 
        action_pose_repr: str='abs',
        pose_type: str='10d'
    ):


    pose_dim = {
        "10d": 3+6,
        "quat": 3+4,
        "se3": 3+3,
        "rotvec": 3+3,
    }[pose_type]

    n_robots = int(action.shape[-1] // (pose_dim+1))
    env_action = list()
    # for robot_idx in range(n_robots):
        # convert pose to mat
    pose_mat = certain_pose_type_to_mat(env_obs["observation.state"][:, :-1].detach().cpu().numpy(), pose_type=pose_type)
    
    start = 0
    action_pose10d = action[..., start:start+pose_dim]
    action_grip = action[..., start+pose_dim:start+pose_dim+1]
    action_pose_mat = certain_pose_type_to_mat(action_pose10d, pose_type)
    
    # solve relative action
    action_mat = convert_pose_mat_rep(
        action_pose_mat, 
        base_pose_mat=pose_mat,
        pose_rep=action_pose_repr,
        backward=True)

    # convert action to pose
    action_pose = mat_to_pose(action_mat)
    env_action.append(action_pose)
    env_action.append(action_grip)

    env_action = np.concatenate(env_action, axis=-1)
    return env_action


def get_real_umi_inference_action(
        action: np.ndarray,
        env_obs: List[np.ndarray],
        action_pose_repr: str='abs',
        pose_type: str='10d'
    ):


    pose_dim = {
        "10d": 3+6,
        "quat": 3+4,
        "se3": 3+3,
        "rotvec": 3+3,
    }[pose_type]

    n_robots = int(action.shape[-1] // (pose_dim+1))
    env_action = list()
    # for robot_idx in range(n_robots):
        # convert pose to mat
    
    start = 0
    pose_mat = certain_pose_type_to_mat(env_obs[..., 0:-1], pose_type="rotvec")

    action_pose10d = action[..., start:start+pose_dim]
    action_grip = action[..., start+pose_dim:start+pose_dim+1]
    action_pose_mat = certain_pose_type_to_mat(action_pose10d, pose_type)
    
    # solve relative action
    action_mat = convert_pose_mat_rep(
        action_pose_mat, 
        base_pose_mat=pose_mat,
        pose_rep=action_pose_repr,
        backward=True)
    
    # convert action to pose
    action_pose = mat_to_pose(action_mat)
    env_action.append(action_pose)
    env_action.append(action_grip)

    env_action = np.concatenate(env_action, axis=-1)
    return env_action
