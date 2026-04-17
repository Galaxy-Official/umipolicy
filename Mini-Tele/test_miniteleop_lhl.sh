export LEROBOT_HOME=/Disk3/OCL4Rob-DATA/OCL4Rob/twist/$(date +"%m-%d-%H-%M-%S") # set your save path
python lerobot/scripts/MiniTeleop_record.py \
  --robot-path lerobot/configs/robot/koch_flexiv.yaml \
  --warmup-time-s 5 \
  --episode-time-s 150 \
  --reset-time-s 5 \
  --num-episodes 30 \
  --push-to-hub  0 \
  --single-task "pick_up" \
  --fps 30