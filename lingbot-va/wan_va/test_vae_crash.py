import sys, os
import torch
import traceback
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))
from modules.utils import load_vae, WanVAEStreamingWrapper

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = load_vae("./ckpt/lingbot-va-base/vae", torch.bfloat16, device)
    streaming_vae = WanVAEStreamingWrapper(vae)

    F_test = 120
    vc = torch.randn(1, 3, F_test, 256, 320, dtype=torch.bfloat16).to(device)
    streaming_vae.clear_cache()
    out = streaming_vae.encode_chunk(vc)
    print("SUCCESS shape:", out.shape)
except Exception as e:
    traceback.print_exc()
