# conda activate lerobot
# sudo chmod 777 /dev/ttyUSB*
TASK_NAME=fry-test
export LEROBOT_HOME=~/$TASK_NAME/$(date +"%m-%d-%H-%M-%S") # set your save path
python lerobot/scripts/MiniTeleop_record.py \
  --robot-path lerobot/configs/robot/koch_flexiv.yaml \
  --warmup-time-s 5 \
  --episode-time-s 120 \
  --reset-time-s 10 \
  --num-episodes 25 \
  --push-to-hub  0 \
  --single-task "Fry." \
  --fps 30
