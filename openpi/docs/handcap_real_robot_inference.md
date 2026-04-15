# Handcap 真机远程推理

本文档说明如何在同一台机器上启动 OpenPI 远程策略服务端和 Flexiv 真机推理客户端，并通过一键脚本完成 handcap 真机实验。

## 环境激活

服务端和客户端都使用同一个 conda 环境，两个 terminal 的激活步骤一致：

```bash
source /home/rhos/miniconda3/bin/activate
conda activate umi310nojax
```

如果你使用一键脚本，则不需要手动再做这两步，脚本会自动执行。

## 前置条件

启动前应确认以下条件成立：

- 当前仓库路径为 `/home/lihong/workspace/umipolicy/openpi`
- `uv` 可用
- `python`、`python3` 可用
- `flexivrdk` 可导入
- `websockets`、`tyro`、`cv2`、`scipy` 可导入
- handcap 相机配置文件存在：
  - `/home/lihong/workspace/umipolicy/lerobot/src/perception/configs/camera/handcap_camera.json`
- 机器人网络可达

## 启动前检查

先进入 `openpi` 根目录：

```bash
cd /home/lihong/workspace/umipolicy/openpi
```

检查环境和依赖：

```bash
source /home/rhos/miniconda3/bin/activate
conda activate umi310nojax

uv --version
python -c "import flexivrdk, tyro, websockets, cv2, scipy"
```

检查相机配置文件：

```bash
test -f /home/lihong/workspace/umipolicy/lerobot/src/perception/configs/camera/handcap_camera.json && echo OK
```

检查 checkpoint 路径：

```bash
test -d /path/to/checkpoint && echo OK
```

## 一键启动脚本

脚本位置：

```bash
/home/lihong/workspace/umipolicy/openpi/start_handcap_remote_inference.sh
```

### 最小启动示例

```bash
cd /home/lihong/workspace/umipolicy/openpi

./start_handcap_remote_inference.sh \
  --policy-config pi0_simple_sorting_tactile \
  --policy-dir /path/to/checkpoint \
  --robot-ip 192.168.2.100 \
  --prompt "pick the block and place it into the pot"
```

### tactile 模型示例

```bash
cd /home/lihong/workspace/umipolicy/openpi

./start_handcap_remote_inference.sh \
  --policy-config pi05_simple_sorting_tactile \
  --policy-dir /path/to/checkpoint \
  --robot-ip 192.168.2.100 \
  --prompt "pick the block and place it into the pot" \
  --task-name pi05_simple_sorting_tactile_eval \
  --ctrl-freq 30 \
  --obs-horizon 2 \
  --use-tactile
```

### 非 tactile 模型示例

```bash
cd /home/lihong/workspace/umipolicy/openpi

./start_handcap_remote_inference.sh \
  --policy-config pi0_simple_sorting \
  --policy-dir /path/to/checkpoint \
  --robot-ip 192.168.2.100 \
  --prompt "pick the block and place it into the pot" \
  --no-use-tactile
```

### dry-run 安全联通测试

第一次联调建议先做 `dry-run`，确认服务端、相机、推理链路和动作维度都正常，再做真实下发：

```bash
cd /home/lihong/workspace/umipolicy/openpi

./start_handcap_remote_inference.sh \
  --policy-config pi0_simple_sorting_tactile \
  --policy-dir /path/to/checkpoint \
  --robot-ip 192.168.2.100 \
  --prompt "pick the block and place it into the pot" \
  --dry-run
```

## 主要参数

必填参数：

- `--policy-config`
- `--policy-dir`
- `--robot-ip`
- `--prompt`

常用可选参数：

- `--server-port`
- `--task-name`
- `--ctrl-freq`
- `--obs-horizon`
- `--camera-config-path`
- `--use-tactile`
- `--no-use-tactile`
- `--action-latency`
- `--init-qpos`
- `--record-root`
- `--server-default-prompt`
- `--startup-timeout`
- `--log-dir`
- `--dry-run`

## 运行产物

### 日志目录

脚本会在下面目录创建日志：

```bash
openpi/logs/handcap_remote_inference/<timestamp>/
```

其中至少包含：

- `server.log`
- `client.log`
- `pids.env`
- `run_command.txt`

### 真机录制输出

客户端会继续按原脚本逻辑，把视频和状态落到：

```bash
openpi/realworld_replay_recording/<task_name>/<timestamp>/
```

通常会看到：

- `view1_wrist.mp4`
- `view2_tactile_left.mp4`
- `view3_tactile_right.mp4`
- `states.parquet`
- `states.json`

## 停止方式

脚本启动后会以前台方式管理两个后台进程。

- 在当前终端按 `Ctrl+C`，脚本会同时停止服务端和客户端
- 如果需要手动清理，可以查看日志目录中的 `pids.env`

## 常见问题

### 1. `flexivrdk` 无法导入

说明当前环境没有正确安装 Flexiv Python 绑定。需要先把 `flexivrdk` 装进 `umi310nojax`，否则客户端无法连接机器人。

### 2. `uv` 不可用

服务端通过 `uv run scripts/serve_policy.py` 启动。如果 `uv` 不在当前环境里，需要先安装或切到正确环境。

### 3. websocket 连接失败

优先检查：

- 服务端是否启动成功
- `server.log` 是否报错
- `--server-port` 是否被占用

### 4. checkpoint 与 `policy-config` 不匹配

服务端虽然能启动，但推理结果可能报维度错误或行为异常。应确保 checkpoint 与 handcap 配置名一致，例如：

- `pi0_simple_sorting`
- `pi0_simple_sorting_tactile`
- `pi05_simple_sorting`
- `pi05_simple_sorting_tactile`

### 5. 相机 `serial` 或 `camera_index` 不对

如果腕部或触觉图像采集失败，优先检查：

- `/home/lihong/workspace/umipolicy/lerobot/src/perception/configs/camera/handcap_camera.json`

需要确认：

- `wrist.serial`
- `left_tactile.camera_index`
- `right_tactile.camera_index`

## 备注

本方案不要求修改现有 Python 代码默认参数。推荐把实验相关配置全部通过一键脚本命令行传入，只在硬件变化时修改相机 JSON 配置文件。
