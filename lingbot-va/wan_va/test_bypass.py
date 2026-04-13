import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# HUGE HACK: Mock huggingface_hub to prevent it from pinging the network!
import huggingface_hub.hf_api
orig_dataset_info = huggingface_hub.hf_api.HfApi.dataset_info
def mock_dataset_info(self, repo_id, *args, **kwargs):
    print("MOCK INTERCEPT: trying to fetch", repo_id)
    raise huggingface_hub.errors.RepositoryNotFoundError("Mock offline")

huggingface_hub.hf_api.HfApi.dataset_info = mock_dataset_info

from lerobot.datasets.lerobot_dataset import LeRobotDataset
try:
    ds = LeRobotDataset(repo_id="handcap_lingbot", root="./Data")
    print("LeRobotDataset init success!", len(ds))
except Exception as e:
    import traceback
    traceback.print_exc()
