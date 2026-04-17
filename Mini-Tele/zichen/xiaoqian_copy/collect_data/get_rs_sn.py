# Get RealSense camera serial number

import pyrealsense2 as rs

# 创建 context
context = rs.context()

# 获取所有连接的设备
devices = context.devices

# 遍历设备，打印序列号
for device in devices:
    print(f"Device: {device.get_info(rs.camera_info.name)}")
    print(f"Serial Number: {device.get_info(rs.camera_info.serial_number)}")