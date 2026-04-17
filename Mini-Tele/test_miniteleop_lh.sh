export LEROBOT_HOME=/Disk2/lihongTeleDATA/collected/test_move_50_3/$(date +"%m-%d-%H-%M-%S") # set your save path
# /Disk2/lihongTeleDATA/collected/peg_in_hole/09-18-17-48-20/lerobot/test
#export LEROBOT_HOME=/home/yushun/Workspace/Mini-Tele/datasets/test/test/$(date +"%m-%d-%H-%M-%S") # set your save path
python lerobot/scripts/MiniTeleop_record_tactile.py \
  --robot-path lerobot/configs/robot/koch_flexiv_tactile.yaml \
  --warmup-time-s 5 \
  --episode-time-s 50 \
  --reset-time-s 10 \
  --num-episodes 50 \
  --push-to-hub  0 \
  --single-task "testmove."  \
  --fps 24
  