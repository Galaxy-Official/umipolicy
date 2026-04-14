import torch
import os

path = '/Users/macbookpro/Desktop/workspace/umipolicy/Data/handcap_lingbot/latents/chunk-000/'
for root, _, files in os.walk(path):
    for f in files[:5]: # just check a few
        if f.endswith('.pth'):
            try:
                data = torch.load(os.path.join(root, f), map_location='cpu')
                if torch.isnan(data).any():
                    print(f"NaN found in {f}")
            except:
                pass
print("Done checking.")
