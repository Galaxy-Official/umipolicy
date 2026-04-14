import sys, os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))
from modules.utils import load_vae, WanVAEStreamingWrapper

device = "cuda" if torch.cuda.is_available() else "cpu"
vae = load_vae("./ckpt/lingbot-va-base/vae", torch.bfloat16, device)
streaming_vae = WanVAEStreamingWrapper(vae)

for F_test in [2, 4, 16, 17]:
    vc = torch.randn(1, 3, F_test, 256, 320, dtype=torch.bfloat16).to(device)
    streaming_vae.clear_cache()
    try:
        out = streaming_vae.encode_chunk(vc)
        print(f"F={F_test} success, output shape: {out.shape}")
    except Exception as e:
        print(f"F={F_test} FAILED: {str(e)[:100]}")
