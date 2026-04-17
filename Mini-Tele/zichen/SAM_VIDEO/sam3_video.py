from ultralytics.models.sam import SAM3VideoSemanticPredictor

# Initialize semantic video predictor
overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=640, model="/home/yushun/Workspace/Mini-Tele/zichen/ckpt/sam3.pt", half=True, save=True)
predictor = SAM3VideoSemanticPredictor(overrides=overrides)

# Track concepts using text prompts
results = predictor(source="/home/yushun/Workspace/Mini-Tele/zichen/xiaoqian_copy/SAM_VIDEO/episode_000001.mp4", text=["pick the apple from the pot"], stream=True)

# Process results
for r in results:
    r.show()  # Display frame with tracked objects