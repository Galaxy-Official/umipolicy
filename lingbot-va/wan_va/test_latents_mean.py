import sys, os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))
from modules.utils import load_vae, WanVAEStreamingWrapper
vae = load_vae("./ckpt/lingbot-va-base/vae", torch.bfloat16, "cpu")
streaming_vae = WanVAEStreamingWrapper(vae)
vc = torch.randn(1, 3, 2, 256, 320, dtype=torch.bfloat16)
try:
    streaming_vae.encode_chunk(vc)
    print("Success 2!")
except Exception as e:
    print(f"Error 2: {e}")

vc3 = torch.randn(1, 3, 3, 256, 320, dtype=torch.bfloat16)
try:
    streaming_vae.encode_chunk(vc3)
    print("Success 3!")
except Exception as e:
    print(f"Error 3: {e}")

