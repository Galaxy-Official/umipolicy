# Remote Teleoperation Package

新增的远程遥操作实现已经集中到这个目录：

```text
lerobot/remote_teleop/
```

目录说明：

- `common.py`
  - 共享协议数据结构
  - 本地主臂读取
  - leader 到 follower 的映射
- `operator.py`
  - 操作者侧 client
- `server.py`
  - 机器人侧 server

外部启动方式保持不变：

```bash
python lerobot/scripts/remote_robot_server.py
python lerobot/scripts/remote_operator.py
```

完整用法见仓库根目录的 `Remote-Teleop-README.md`。
