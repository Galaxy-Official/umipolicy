import sys, os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))
from modules.utils import load_vae
vae = load_vae("./ckpt/lingbot-va-base/vae", torch.bfloat16, "cpu")
print(torch.tensor(vae.config.latents_mean).shape)
