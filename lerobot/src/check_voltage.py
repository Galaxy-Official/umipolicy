import dynamixel_sdk as dxl

port_handler = dxl.PortHandler("/dev/ttyUSB0")
packet_handler = dxl.PacketHandler(2.0)
port_handler.openPort()
port_handler.setBaudRate(1000000)

# 读取 1。模型号 2。当前实际电压
model_number, comm1, _ = packet_handler.read2ByteTxRx(port_handler, 1, 0)
voltage, comm2, _ = packet_handler.read2ByteTxRx(port_handler, 1, 144)

if comm1 == dxl.COMM_SUCCESS:
    print(f"✅ 1号舵机底层连接畅通！")
    print(f"👉 舵机实际型号代码 Model Number: {model_number}")
    print(f"👉 舵机自身检测到的当前电压为: {voltage / 10.0} V")
    
    if voltage < 9.0 and model_number not in [1190]:
        print("🚨 电压过低！12V舵机只接收到了微弱电压。请检查：主电源变压器灯是否亮起？电源线是否有暗断？扩展板保险丝是否烧断？")
    elif voltage < 4.0:
        print("🚨 电压过低！5V舵机只接收到了USB漏电的微弱电压。扩展板或者主供电绝对没起作用！")
    else:
        print("💡 电压正常范围内，这说明硬件报错可能是记忆锁定导致的。")
else:
    print("通讯异常！")

port_handler.closePort()
