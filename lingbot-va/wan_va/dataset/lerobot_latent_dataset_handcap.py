# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import lerobot.datasets.lerobot_dataset_handcap as lerobot_dataset_handcap
def get_episode_data_index(meta_episodes, episodes=None):
    ep_data_idx = {"from": [], "to": []}
    meta_iterable = meta_episodes.values() if isinstance(meta_episodes, dict) else meta_episodes
    for item in meta_iterable:
        if "dataset_from_index" in item:
            ep_data_idx["from"].append(item["dataset_from_index"])
            ep_data_idx["to"].append(item["dataset_to_index"])
        elif "from" in item:
            ep_data_idx["from"].append(item["from"])
            ep_data_idx["to"].append(item["to"])
    return ep_data_idx
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
import numpy as np
from pathlib import Path
from collections.abc import Callable
import os
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
from lerobot.datasets.pose_utils import mat_to_certain_pose_type, pose_to_mat
from lerobot.utils.constants import HF_LEROBOT_HOME

def recursive_find_file(directory, filename='info.json'):
    result = []
    try:
        for root, dirs, files in os.walk(directory):
            if filename in files:
                full_path = os.path.join(root, filename)
                result.append(full_path)
    except PermissionError:
        print(f"Error: can not access {directory}")
    except Exception as e:
        print(f"Error: {e}")
    return result

def construct_lerobot(
    repo_id,
    config,
):
    return LatentLeRobotDatasetHandcap(
        repo_id=repo_id,
        config=config,
    )

def construct_lerobot_multi_processor(config, 
                                      num_init_worker=8,
                                      ):
    datasets_out_lst = []
    construct_func = partial(
        construct_lerobot,
        config=config,
    )
    repo_list = recursive_find_file(config.dataset_path, 'info.json')
    repo_list = [v.split('/meta/info.json')[0] for v in repo_list]
    with Pool(num_init_worker) as pool:
        datasets_out_lst = pool.map(construct_func, repo_list)
                
    return datasets_out_lst

def get_relative_pose(pose):
    if torch.is_tensor(pose):
        pose = pose.detach().cpu().numpy()
    
    rot = R.from_quat(pose[:, 3:7])
    first_rot = R.from_quat(np.tile(pose[:1, 3:7], (pose.shape[0], 1)))
    trans = pose[:, :3]
    relative_trans = trans - trans[0:1]

    relative_rot = first_rot.inv() * rot
    relative_quat = relative_rot.as_quat()

    relative_pose = np.concatenate([relative_trans, relative_quat], axis=1)
    return torch.from_numpy(relative_pose)

def process_to_relative_rot6d(obs_state_tensor, action_tensor):
    """
    Converts absolute rotvec 10D data to relative rot6d 10D data.

    Input:
        obs_state_tensor: Shape [10] or [N, 10].
        action_tensor: Shape [10] or [C, 10].
    Output:
        obs_state_new, action_new as torch.Tensors.
    """
    obs_state_tensor = torch.as_tensor(obs_state_tensor)
    action_tensor = torch.as_tensor(action_tensor)
    obs_is_1d = obs_state_tensor.ndim == 1
    act_is_1d = action_tensor.ndim == 1

    obs_state = obs_state_tensor.unsqueeze(0) if obs_is_1d else obs_state_tensor
    act_state = action_tensor.unsqueeze(0) if act_is_1d else action_tensor
    if obs_state.shape[-1] != 10 or act_state.shape[-1] != 10:
        raise ValueError(
            f"Expected 10D obs/action for relative rot6d conversion, got "
            f"obs={tuple(obs_state.shape)} action={tuple(act_state.shape)}"
        )

    obs_np = obs_state.detach().cpu().numpy()
    act_np = act_state.detach().cpu().numpy()

    n_obs = obs_np.shape[0]
    combined_np = np.concatenate([obs_np, act_np], axis=0)
    combined_pose_mat = pose_to_mat(combined_np[..., :-4])

    base_pose_mat = combined_pose_mat[n_obs - 1]
    combined_rel_mat = np.linalg.inv(base_pose_mat) @ combined_pose_mat
    combined_pose_10d = mat_to_certain_pose_type(combined_rel_mat, "10d")

    combined_gripper = combined_np[:, -4:-3]
    combined_new = np.concatenate([combined_pose_10d, combined_gripper], axis=-1)

    obs_new = combined_new[:n_obs]
    act_new = combined_new[n_obs:]
    if obs_is_1d:
        obs_new = obs_new[0]
    if act_is_1d:
        act_new = act_new[0]

    return torch.from_numpy(obs_new).float(), torch.from_numpy(act_new).float()

class MultiLatentLeRobotDatasetHandcap(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        num_init_worker=128,
    ):
        self._datasets = construct_lerobot_multi_processor(config, 
                                                           num_init_worker, 
                                                           )
        self.item_id_to_dataset_id, self.acc_dset_num = (
            self._get_item_id_to_dataset_id()
        )
        if len(self) == 0:
            raise ValueError(f"CRITICAL ERROR: The dataset loaded from '{config.dataset_path}' contains 0 valid episodes! "
                             "This typically means either the dataset_path does not exist/is empty, OR you have not run the latent extraction script "
                             "(run_extract_latents.sh) on this dataset yet. Lingbot-VA ONLY trains on extracted latents.")

    def __len__(
        self,
    ):
        return sum(len(v) for v in self._datasets)

    def _get_item_id_to_dataset_id(self):
        item_id_to_dataset_id = {}
        acc_dset_num = {}
        acc_nums = [0]
        id = 0
        for dset_id, dset in enumerate(self._datasets):
            acc_nums.append(acc_nums[-1] + len(dset))
            for _ in range(len(dset)):
                item_id_to_dataset_id[id] = dset_id
                id += 1
        for did in range(len(self._datasets)):
            acc_dset_num[did] = acc_nums[did]
        return item_id_to_dataset_id, acc_dset_num

    def __getitem__(self, idx) -> dict:
        assert idx < len(self)
        cur_dset = self._datasets[self.item_id_to_dataset_id[idx]]
        local_idx = idx - self.acc_dset_num[self.item_id_to_dataset_id[idx]]
        return cur_dset[local_idx]

class LatentLeRobotDatasetHandcap(LeRobotDataset):
    def __init__(
        self,
        repo_id,
        config=None,
    ):
        if getattr(config, 'use_handcap', False):
            self.dataset = lerobot_dataset_handcap.LeRobotDatasetHandcap(repo_id=repo_id, root=Path(repo_id))
            self.meta = self.dataset.meta
            self.repo_id = repo_id
            self.root = Path(repo_id)
            self.episodes = None
        else:
            super().__init__(repo_id, root=Path(repo_id))
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)
        
        self.latent_path = Path(repo_id) / 'latents'
        if not os.path.exists(config.empty_emb_path):
            fallback_path = Path(__file__).resolve().parent.parent.parent / 'empty_emb.pt'
            if fallback_path.exists():
                print(f"Loading empty_emb.pt from offline fallback: {fallback_path}")
                config.empty_emb_path = str(fallback_path)
            else:
                raise FileNotFoundError(f"Could not find empty_emb.pt locally or at {config.empty_emb_path}. Please make sure you push it.")
            
        self.empty_emb = torch.load(config.empty_emb_path, weights_only=False)
        self.config = config
        self.cfg_prob = config.cfg_prob
        self.used_video_keys = config.obs_cam_keys
        self.convert_action_to_relative_rot6d = getattr(config, 'convert_action_to_relative_rot6d', False)
        self.q01 = np.array(config.norm_stat['q01'], dtype='float')[None]
        self.q99 = np.array(config.norm_stat['q99'], dtype='float')[None]
        hf_columns = ['action']
        if self.convert_action_to_relative_rot6d:
            hf_columns.append('observation.state')
        self._hf_torch_view = self.hf_dataset.with_format(
                type='torch',
                columns=hf_columns,
                output_all_columns=False
            )
        self.parse_meta()

    @property
    def hf_dataset(self):
        if hasattr(self, 'dataset'):
            return self.dataset.hf_dataset
        return super().hf_dataset

    def parse_meta(self):
        out = []
        meta_iterable = self.meta.episodes.values() if isinstance(self.meta.episodes, dict) else self.meta.episodes
        for value in meta_iterable:
            episode_index = value["episode_index"]
            tasks = value.get("tasks", ["perform the manipulation task"])
            action_config = value.get("action_config", [
                {
                    "start_frame": 0,
                    "end_frame": value["length"],
                    "action_text": tasks[0] if isinstance(tasks, list) and len(tasks)>0 else tasks,
                }
            ])
            for acfg in action_config:
                cur_meta = {
                    "episode_index": episode_index,
                    "tasks": tasks,
                }
                cur_meta.update(acfg)

                check_statu = self._check_meta(
                    cur_meta["start_frame"],
                    cur_meta["end_frame"],
                    cur_meta["episode_index"],
                )

                if check_statu:
                    out.append(cur_meta)
        if len(out) == 0:
            print(f"[Latent Dataset Debug] WARNING: No valid episodes parsed! check_statu returned False for all episodes in {self.repo_id}.")
        self.new_metas = out

    def _check_meta(self, start_frame, end_frame, episode_index):
        if hasattr(self.meta, 'get_episode_chunk'):
            episode_chunk = self.meta.get_episode_chunk(episode_index)
        else:
            try:
                episode_chunk = self.meta.episodes[episode_index].get("meta/episodes/chunk_index", episode_index // 1000)
            except (AttributeError, KeyError, IndexError):
                episode_chunk = episode_index // 1000
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            if not os.path.exists(latent_file):
                print(f"[Latent Dataset Debug] Missing latent file: {latent_file}. Failing check for episode {episode_index}.")
                return False
        return True

    def _get_global_idx(self, episode_index: int, local_index: int):
        ep_start = self.episode_data_index["from"][episode_index]
        return local_index + ep_start

    def _get_range_hf_data(self, start_frame, end_frame):
        batch = self._hf_torch_view[start_frame:end_frame]
        return batch

    def _flatten_latent_dict(self, latent_dict):
        out = {}
        for key, value in latent_dict.items():
            for inner_key, inner_value in value.items():
                new_key = f"{key}.{inner_key}"
                out[new_key] = inner_value
        return out

    def _get_range_latent_data(self, start_frame, end_frame, episode_index):
        if hasattr(self.meta, 'get_episode_chunk'):
            episode_chunk = self.meta.get_episode_chunk(episode_index)
        else:
            try:
                episode_chunk = self.meta.episodes[episode_index].get("meta/episodes/chunk_index", episode_index // 1000)
            except (AttributeError, KeyError, IndexError):
                episode_chunk = episode_index // 1000
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        out = {}
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            assert os.path.exists(latent_file)
            latent_data = torch.load(latent_file, weights_only=False)
            out[key] = latent_data
        
        return self._flatten_latent_dict(out)
    
        
    def _cat_video_latents(self,
                           data_dict
                           ):
        latent_lst = []
        for key in self.used_video_keys:
            latent= data_dict[f"{key}.latent"]
            latent_num_frames = data_dict[f"{key}.latent_num_frames"]
            latent_height = data_dict[f"{key}.latent_height"]
            latent_width = data_dict[f"{key}.latent_width"]
            latent = rearrange(latent, 
                                 '(f h w) c -> f h w c', 
                                 f=latent_num_frames, 
                                 h=latent_height, 
                                 w=latent_width)
            latent_lst.append(latent)
        if getattr(self.config, 'use_handcap', False) and self.config.env_type == 'handcap':
            if getattr(self.config, 'use_tactile', False) and len(latent_lst) >= 3:
                # Wrist config with tactile: usually latent_lst[0]=wrist, [1]=left_tactile, [2]=right_tactile
                wrist_latent = latent_lst[0]  # [f, h, w_w, c]
                tactile_latents = torch.cat(latent_lst[1:], dim=2)  # [f, h, w_t, c]
                # PAD wrist_latent if its width is smaller than the concatenated tactile latents
                if wrist_latent.shape[2] < tactile_latents.shape[2]:
                    pad_w = tactile_latents.shape[2] - wrist_latent.shape[2]
                    pad_tensor = torch.zeros(*wrist_latent.shape[:2], pad_w, wrist_latent.shape[3], device=wrist_latent.device, dtype=wrist_latent.dtype)
                    wrist_latent = torch.cat([wrist_latent, pad_tensor], dim=2)
                elif tactile_latents.shape[2] < wrist_latent.shape[2]:
                    pad_w = wrist_latent.shape[2] - tactile_latents.shape[2]
                    pad_tensor = torch.zeros(*tactile_latents.shape[:2], pad_w, tactile_latents.shape[3], device=tactile_latents.device, dtype=tactile_latents.dtype)
                    tactile_latents = torch.cat([tactile_latents, pad_tensor], dim=2)
                    
                cat_latent = torch.cat([wrist_latent, tactile_latents], dim=1)
            else:
                cat_latent = torch.cat(latent_lst, dim=2)
        elif self.config.env_type == 'robotwin_tshape':
            wrist_latent = torch.cat(latent_lst[1:], dim=2)
            cat_latent = torch.cat([wrist_latent, latent_lst[0]], dim=1)
        else:
            cat_latent = torch.cat(latent_lst, dim=2)

        text_emb = data_dict[f"{self.used_video_keys[0]}.text_emb"]
        if torch.rand(1).item() < self.cfg_prob:
            text_emb = self.empty_emb

        out_dict = dict(
            latents = cat_latent,
            text_emb = text_emb,
        )
        return out_dict
    
    def _action_post_process(self, local_start_frame, local_end_frame, latent_frame_ids, action, obs_state=None):
        act_shift = int(latent_frame_ids[0] - local_start_frame)
        frame_stride = latent_frame_ids[1] - latent_frame_ids[0]
        action = action[act_shift:]
        obs_state_new = None
        if self.convert_action_to_relative_rot6d:
            if obs_state is None:
                raise ValueError("observation.state is required when convert_action_to_relative_rot6d=True")
            obs_state_new, action = process_to_relative_rot6d(obs_state, action)
        if self.config.env_type == 'robotwin_tshape': ## TODO support get_relative_pose for other dataset, currently only support robotwin 
            left_action = get_relative_pose(action[:, :7])
            right_action = get_relative_pose(action[:, 8:15])
            action = np.concatenate([left_action, action[:, 7:8], right_action, action[:, 15:16]], axis=1)
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        action = np.pad(action, pad_width=((frame_stride * 4, 0), (0, 0)), mode='constant', constant_values=0)

        latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
        required_action_num = latent_frame_num * frame_stride * 4

        action = action[:required_action_num]
        action_mask = np.ones_like(action, dtype='bool')
        assert action.shape[0] == required_action_num


        action_paded = np.pad(action, ((0, 0), (0, 1)), mode='constant', constant_values=0)
        action_mask_padded = np.pad(action_mask, ((0, 0), (0, 1)), mode='constant', constant_values=0)

        action_aligned = action_paded[:, self.config.inverse_used_action_channel_ids]
        action_mask_aligned = action_mask_padded[:, self.config.inverse_used_action_channel_ids]
        action_aligned = (action_aligned - self.q01) / (
                self.q99 - self.q01 + 1e-6) * 2. - 1.
        action_aligned = rearrange(action_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_mask_aligned = rearrange(action_mask_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_aligned *= action_mask_aligned
        return (
            torch.from_numpy(action_aligned).float(),
            torch.from_numpy(action_mask_aligned).bool(),
            obs_state_new,
        )

    def __getitem__(self, idx) -> dict:
        idx = idx % len(self.new_metas)
        cur_meta = self.new_metas[idx]
        episode_index = cur_meta["episode_index"]
        start_frame = cur_meta["start_frame"]
        end_frame = cur_meta["end_frame"]
        local_start_frame = start_frame
        local_end_frame = end_frame

        ori_data_dict = self._get_range_latent_data(start_frame, end_frame, episode_index)

        latent_frame_ids = ori_data_dict[f"{self.used_video_keys[0]}.frame_ids"]
        start_frame = self._get_global_idx(episode_index, start_frame)
        end_frame = self._get_global_idx(episode_index, end_frame)

        hf_data_frames = self._get_range_hf_data(start_frame, end_frame)
        ori_data_dict.update(hf_data_frames)
        out_dict = self._cat_video_latents(ori_data_dict)

        actions, actions_mask, obs_state = self._action_post_process(
            local_start_frame,
            local_end_frame,
            latent_frame_ids,
            ori_data_dict['action'],
            ori_data_dict.get('observation.state'),
        )
        out_dict['actions'] = actions
        out_dict['actions_mask'] = actions_mask
        if obs_state is not None:
            out_dict['observation.state'] = obs_state

        out_dict['latents'] = out_dict['latents'].permute(3, 0, 1, 2)
        return out_dict

    def __len__(self):
        return len(self.new_metas)

if __name__ == '__main__':
    from wan_va.configs import VA_CONFIGS
    from tqdm import tqdm
    dset = MultiLatentLeRobotDataset(
        VA_CONFIGS['demo_train']
    )
    for key, value in dset[0].items():
        if isinstance(value, torch.Tensor):
            print(f'{key}: {value.shape} tensor')
        elif isinstance(value, np.ndarray):
            print(f'{key}: {value.shape} np')
        else:
            print(f'{key}: {value}')
    print(len(dset))
    dloader = DataLoader(
            dset,
            batch_size=1,
            shuffle=True,
            num_workers=32,
        )
    max_l = 0
    action_list = []
    for data in tqdm(dloader):
        _, _, F, H, W = data['latents'].shape
        max_l = max(max_l, F*H*W)
        action_list.append(data['actions'].flatten(2).permute(0, 2, 1).flatten(0, 1))
    action_all = torch.cat(action_list, dim=0)
    print(max_l)
    print(action_all.shape, action_all.mean(dim=0), action_all.min(dim=0)[0], action_all.max(dim=0)[0])
    
