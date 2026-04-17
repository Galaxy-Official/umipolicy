# export LEROBOT_HOME=/Disk3/OCL4Rob-DATA/OCL4Rob/test/$(date +"%m-%d-%H-%M-%S") # set your save path
export LEROBOT_HOME=/Disk2/SnakeRob-DATA/test/$(date +"%m-%d-%H-%M-%S") 
cd /home/yushun/Workspace/Mini-Tele && python lerobot/scripts/MiniTeleop_record.py \
  --robot-path lerobot/configs/robot/koch_flexiv.yaml \
  --warmup-time-s 5 \
  --episode-time-s 120 \
  --reset-time-s 20 \
  --num-episodes 2 \
  --push-to-hub  0 \
  --single-task "backside image"  \
  --fps 30