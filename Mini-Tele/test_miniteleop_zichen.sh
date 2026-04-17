export LEROBOT_HOME=/Disk2/zichen/lerobot_data/PourApplePan/$(date +"%m-%d-%H-%M-%S") # set your save path
python lerobot/scripts/MiniTeleop_record_withcam.py \
  --robot-path lerobot/configs/robot/koch_flexiv_zzc.yaml \
  --warmup-time-s 5 \
  --episode-time-s 1200 \
  --reset-time-s 10 \
  --num-episodes 5 \
  --push-to-hub  0 \
  --single-task "Pour out the apple from a pan." \
  --fps 30

