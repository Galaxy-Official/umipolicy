#!/bin/bash
# 这是一个便捷启动脚本，你可以随时修改里面的参数

bash start_handcap_remote_inference.sh \
  --policy-config pi0_simple_sorting \
  --policy-dir ~/umipolicy/openpi/ckpt/99999 \
  --robot-ip 192.168.1.100 \
  --prompt "simple sorting task" \
  --no-use-tactile
