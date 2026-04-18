import dynamixel_sdk as dxl

port_handler = dxl.PortHandler("/dev/ttyUSB0")
packet_handler = dxl.PacketHandler(2.0)
port_handler.openPort()
port_handler.setBaudRate(1000000)

# 读取控制表地址 70 (Hardware Error Status)
error_code, comm, err = packet_handler.read1ByteTxRx(port_handler, 1, 70)

if comm == dxl.COMM_SUCCESS:
    print(f"1号舵机当前硬件错误状态码: {error_code}")
    print("解析：")
    if error_code & 1: print(" - Input Voltage Error (输入电压过低或过高，主板电源没开、线太细、没接紧)")
    if error_code & 4: print(" - Overheating Error (过热锁死)")
    if error_code & 8: print(" - Motor Encoder Error (编码器损坏)")
    if error_code & 16: print(" - Electrical Shock Error (电击/内部短路保护)")
    if error_code & 32: print(" - Overload Error (过载卡死，被物理卡住或者负载过重)")
    if error_code == 0: print(" - 正常，无错误！(如果是0说明报错问题可能潜伏在其他ID或者有其他隐情)")
else:
    print(f"通讯读取失败，错误码：{packet_handler.getTxRxResult(comm)}")

port_handler.closePort()
