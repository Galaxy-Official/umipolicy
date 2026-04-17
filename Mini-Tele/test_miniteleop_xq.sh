export LEROBOT_HOME=data_arti/$(date +"%m-%d-%H-%M") # set your save path
python lerobot/scripts/MiniTeleop_record.py \
  --robot-path lerobot/configs/robot/koch_flexiv.yaml \
  --warmup-time-s 5 \
  --episode-time-s 180 \
  --reset-time-s 30 \
  --num-episodes 15 \
  --push-to-hub  0 \
  --single-task "box."  \
  --fps 30 