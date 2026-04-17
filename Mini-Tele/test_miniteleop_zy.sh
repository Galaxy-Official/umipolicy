export LEROBOT_HOME=$HOME/DATA/OCL4Rob/VQA-TEST/$(date +"%m-%d-%H-%M-%S") # set your save path
python lerobot/scripts/MiniTeleop_record.py \
  --robot-path lerobot/configs/robot/koch_flexiv.yaml \
  --warmup-time-s 5 \
  --episode-time-s 120 \
  --reset-time-s 20 \
  --num-episodes 60 \
  --push-to-hub  0 \
  --single-task "Pick up the block and place it in the pot."  \
  --fps 30 