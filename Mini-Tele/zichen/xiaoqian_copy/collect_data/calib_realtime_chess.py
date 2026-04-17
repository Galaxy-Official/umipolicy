#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pyrealsense2 as rs

def realtime_chessboard(camera_id=None, width=640, height=480, fps=30, ch_size=(8,11)):
    # 初始化 RealSense 彩色流
    pipeline = rs.pipeline()
    config = rs.config()
    if camera_id:
        config.enable_device(camera_id)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    print("实时棋盘角点检测，按 q 退出")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            if not color_frame:
                continue
            img = np.asanyarray(color_frame.get_data())

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(
                gray, ch_size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK
            )

            vis = img.copy()
            if ret:
                # 亚像素精细化
                corners2 = cv2.cornerSubPix(
                    gray, corners, (11,11), (-1,-1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
                )
                cv2.drawChessboardCorners(vis, ch_size, corners2, True)
                status = f"Detected {ch_size} | corners={len(corners2)}"
                color = (0, 255, 0)
            else:
                status = "Not detected. "
                color = (0, 0, 255)

            cv2.putText(vis, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("RealSense Chessboard", vis)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        pipeline.stop()

if __name__ == "__main__":
    # 修改为你的相机 SN；None 表示自动选择
    realtime_chessboard(camera_id="135122074278", width=640, height=480, fps=30, ch_size=(7,10))