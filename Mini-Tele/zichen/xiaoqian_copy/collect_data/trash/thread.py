import threading
import time
from camera import CameraD400
from robot import Flexiv
from utilities.data_utils import transform_pc, transform_dir
from utilities.vis_utils import visualize

# 定义一个后台函数，用于周期性执行任务（这里示例为每 1/30 秒记录一次日志）
def background_task():
    while True:
        # 这里是你的主要逻辑，比如写日志、检查状态等
        print("记录一次日志...")

        # 等待 1/30 秒再继续执行
        time.sleep(1/30)

def main():
    # 创建线程，指定为“后台运行”（daemon=True），这样主程序结束时，后台线程也会自动结束
    thread = threading.Thread(target=background_task, daemon=True)
    thread.start()

    # 主程序的其他逻辑
    # 此处以一个示例循环作为替代
    for i in range(10):
        print(f"主程序执行第 {i+1} 步...")
        time.sleep(0.5)

    print("主程序结束。")

if __name__ == "__main__":
    main()
