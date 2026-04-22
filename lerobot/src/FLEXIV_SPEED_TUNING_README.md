# Flexiv 机械臂速度调节指南 (Flexiv Speed Tuning Guide)

在 LeRobot 3.0 + Flexiv RDK 1.0+ 架构中，机械臂的速度主要分为三个不同的场景进行控制。如果您觉得机械臂运动过快或过慢，请根据当前执行的任务类型，修改对应的代码文件。

---

## 1. 数据录制 / 遥操跟随速度 (Teleop Record)

当您执行 `teleop.sh` 或者直接运行 `lerobot_flexiv_teleop_record.py` 时，机械臂受控于底层的驱动封装类。

**修改位置：**
文件：`lerobot/common/robot_devices/robots/flexiv.py` 
大概在 **第 70 行** (`__init__` 函数内部)：

```python
self.MAX_VEL = [0.3] * self.DOF  # 最大关节速度 (单位: 弧度/秒)
self.MAX_ACC = [0.3] * self.DOF  # 最大关节加速度 (单位: 弧度/秒^2)
```

- **调快**：将 `0.3` 改为 `0.5` 到 `1.0` 之间（1.0 约等于每秒转动 57 度，非常快，请注意安全）。
- **调慢**：将 `0.3` 改为 `0.1` 到 `0.2` 之间（适合初次测试或需要极高精度的精细操作）。

---

## 2. 模型推理执行速度 (Inference)

当您使用已经训练好的模型进行闭环推理（例如执行 `run_block_stack.sh` 或运行 `lerobot_flexiv.py`）时，速度限制是独立配置的。

**修改位置：**
文件：`lerobot/scripts/lerobot_flexiv.py`

在里面全局搜索 `SendJointPosition`，主要有两个地方（一个是 `reset()` 函数中，一个是 `exec_actions()` 函数中）：

```python
# 倒数第二个数组是 MAX_VEL，最后一个数组是 MAX_ACC
self.robot.SendJointPosition(result[1], [0]*7, [0]*7, [0.3]*7, [0.3]*7)
```

*提示：*
如果您发现模型推理时动作太猛烈，请把最后的两个 `[0.3]*7` 继续调小（例如 `[0.1]*7`）。推理时如果把加速度（最后一位）调得过小，可能会导致跟随延迟。

---

## 3. 回零速度 (Home)

在我们最新的重构中，执行回零时调用的是 `self.flexiv.ExecutePlan("PLAN-Home")`。
由于这是调用了机械臂内部的全局计划，**它的速度不再受 Python 代码控制**。

**修改方法：**
1. 打开 Flexiv Elements (网页控制台/示教器)。
2. 在项目列表中找到名字叫 `PLAN-Home` 的计划。
3. 点击进去，选中里面的关节运动节点，在右侧面板直接调节它的**全局速度百分比**和**加速度百分比**。
4. 保存计划即可永久生效。
