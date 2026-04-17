#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pyrealsense2 as rs
import argparse

def collect_chessboard_images(camera_id=None, width=640, height=480, fps=30, n_images=20):
    pipeline = rs.pipeline()
    config = rs.config()
    if camera_id:
        config.enable_device(camera_id)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    images = []
    print("按空格采集，按 q 退出")
    while len(images) < n_images:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        img = np.asanyarray(color_frame.get_data())
        cv2.imshow("collect", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            images.append(img.copy())
            print(f"已采集 {len(images)}/{n_images}")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    pipeline.stop()
    return images

def calibrate_intrinsics(images, ch_size=(7,10), square_size=0.015):
    # 准备棋盘三维点坐标（Z=0）
    objp = np.zeros((ch_size[0]*ch_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:ch_size[0], 0:ch_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints, imgpoints = [], []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, ch_size,
                                                 flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            objpoints.append(objp)
            imgpoints.append(corners2)

    assert len(objpoints) >= 2, "有效棋盘图太少，请采集更多角度的图片"

    h, w = images[0].shape[:2]
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
    mean_error = 0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        error = cv2.norm(imgpoints[i], proj, cv2.NORM_L2)/len(proj)
        mean_error += error
    mean_error /= len(objpoints)
    return K, D, mean_error, (w, h)

def scale_K(K, src_size, dst_size):
    # 根据分辨率缩放内参
    (w0, h0), (w1, h1) = src_size, dst_size
    sx, sy = w1/w0, h1/h0
    K_scaled = K.copy()
    K_scaled[0,0] *= sx
    K_scaled[1,1] *= sy
    K_scaled[0,2] *= sx
    K_scaled[1,2] *= sy
    return K_scaled

def realtime_chessboard(camera_id=None, width=640, height=480, fps=30, ch_size=(9,11), save_dir="chess_capture"):
    """
    Real-time chessboard corner detection using Intel RealSense color stream.
    - Shows detection overlay.
    - Press 'q' to quit.
    - Press SPACE to save current frame (and corners if detected) into save_dir.
    """
    import os, time, json
    os.makedirs(save_dir, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    if camera_id:
        config.enable_device(camera_id)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    print("Real-time chessboard detection started. Press SPACE to save, 'q' to quit.")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            if not color_frame:
                continue
            img = np.asanyarray(color_frame.get_data())

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_eq = clahe.apply(gray)
            gray_prep = cv2.GaussianBlur(gray_eq, (3, 3), 0)

            vis = img.copy()

            ret = False
            corners = None
            if hasattr(cv2, 'findChessboardCornersSB'):
                try:
                    ret, corners = cv2.findChessboardCornersSB(
                        gray_prep, ch_size,
                        flags=(cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
                    )
                except Exception:
                    ret = False
                    corners = None

            if not ret:
                ret, corners = cv2.findChessboardCorners(
                    gray_prep, ch_size,
                    flags=(cv2.CALIB_CB_ADAPTIVE_THRESH |
                           cv2.CALIB_CB_NORMALIZE_IMAGE |
                           cv2.CALIB_CB_FILTER_QUADS)
                )

            corners2 = None
            if ret and corners is not None:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
                corners2 = cv2.cornerSubPix(gray_prep, corners, (15, 15), (-1, -1), criteria)
                cv2.drawChessboardCorners(vis, ch_size, corners2, True)
                status = f"Detected {ch_size} | corners={len(corners2)}"
                color = (0, 255, 0)
            else:
                status = "Not detected."
                color = (0, 0, 255)

            cv2.putText(vis, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("RealSense Chessboard", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                # Save current frame and corners (if any)
                ts = time.strftime('%Y%m%d_%H%M%S')
                base = f"{ts}_w{width}h{height}_c{ch_size[0]}x{ch_size[1]}"
                img_path = os.path.join(save_dir, base + ".png")
                vis_path = os.path.join(save_dir, base + "_vis.png")
                meta_path = os.path.join(save_dir, base + ".json")
                cv2.imwrite(img_path, img)
                cv2.imwrite(vis_path, vis)
                meta = {
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "ch_size": [ch_size[0], ch_size[1]],
                    "detected": bool(ret),
                    "corners": corners2.reshape(-1, 2).tolist() if corners2 is not None else None
                }
                with open(meta_path, 'w') as f:
                    json.dump(meta, f, indent=2)
                print(f"Saved: {img_path} | vis: {vis_path}")
    finally:
        cv2.destroyAllWindows()
        pipeline.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["collect", "calibrate", "realtime"], default="realtime",
                        help="工作模式：collect（采集棋盘图）、calibrate（标定）、realtime（实时检测）")
    parser.add_argument("--sn", type=str, default=None,
                        help="相机序列号，默认使用第一个相机")
    parser.add_argument("--width", type=int, default=640,
                        help="视频流宽度")
    parser.add_argument("--height", type=int, default=480,
                        help="视频流高度")
    parser.add_argument("--fps", type=int, default=30,
                        help="帧率")
    args = parser.parse_args()

    ch_size = (7, 10)  # 棋盘格内角点数
    # args.sn = "233522075695"  # left side camera
    # args.sn = "135122074278"  # top camera
    args.sn = "332322072673"  # right side camera
    if args.mode == "collect":
        collect_chessboard_images(camera_id=args.sn, width=args.width, height=args.height, fps=args.fps)
    elif args.mode == "calibrate":
        # 读取采集的图片文件夹
        import os
        from glob import glob
        images = []
        folder = f"chess_capture/{args.sn}"
        if os.path.exists(folder):
            image_files = glob(os.path.join(folder, "*.png"))
            images = [cv2.imread(f) for f in image_files]
        # 标定并打印结果
        K, D, err, src_size = calibrate_intrinsics(images, ch_size=ch_size)
        print("\n== OpenCV Calibration Results ==")
        print("Intrinsic K:\n", K)
        print("Distortion D:", D.ravel())
        print("Reprojection error:", err)
        print("Calibration resolution:", src_size)
        # 保存到文件
        np.savez(f"chess_capture/{args.sn}/intrinsics_opencv.npz", K=K, D=D, src_w=src_size[0], src_h=src_size[1])

        # 读取 RealSense 自带内参进行对比
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            if args.sn:
                config.enable_device(args.sn)
            config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
            profile = pipeline.start(config)
            color_stream = profile.get_stream(rs.stream.color)
            vs_profile = rs.video_stream_profile(color_stream)
            intr = vs_profile.get_intrinsics()
            pipeline.stop()
            # 构造 RealSense 相机矩阵与畸变
            K_rs = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)
            # RealSense 畸变参数顺序: [k1, k2, p1, p2, k3]
            D_rs = np.array(intr.coeffs, dtype=np.float64).reshape(1, -1)

            print("\n== RealSense Reported Intrinsics ==")
            print("Intrinsic K (fx, fy, cx, cy):", intr.fx, intr.fy, intr.ppx, intr.ppy)
            print("Distortion model:", intr.model)
            print("Distortion coeffs:", intr.coeffs)
            print("Resolution:", (intr.width, intr.height))

            # 对齐分辨率差异（如采集图片与当前流不同分辨率）
            if (src_size[0], src_size[1]) != (intr.width, intr.height):
                K_rs_scaled = scale_K(K_rs, (intr.width, intr.height), (src_size[0], src_size[1]))
            else:
                K_rs_scaled = K_rs

            # 打印对比
            print("\n== Comparison (OpenCV vs RealSense scaled to calibration resolution) ==")
            print("K_opencv=\n", K)
            print("K_rs_scaled=\n", K_rs_scaled)
            print("Delta K=\n", K - K_rs_scaled)
            # 简单百分比差异
            def pct(a, b):
                return 0 if b == 0 else (a - b) / b * 100.0
            print("fx diff %: {:.2f}%".format(pct(K[0,0], K_rs_scaled[0,0])))
            print("fy diff %: {:.2f}%".format(pct(K[1,1], K_rs_scaled[1,1])))
            print("cx diff %: {:.2f}%".format(pct(K[0,2], K_rs_scaled[0,2])))
            print("cy diff %: {:.2f}%".format(pct(K[1,2], K_rs_scaled[1,2])))

            # 保存对比结果
            np.savez(f"chess_capture/{args.sn}/intrinsics_rs.npz",
                     K_opencv=K, D_opencv=D, err=err, src_w=src_size[0], src_h=src_size[1],
                     K_rs=K_rs, D_rs=D_rs, K_rs_scaled=K_rs_scaled)
        except Exception as e:
            print("Failed to read RealSense intrinsics for comparison:", e)
    else:  # realtime
        cv2.namedWindow("RealSense Chessboard", cv2.WINDOW_NORMAL)
        realtime_chessboard(camera_id=args.sn, width=args.width, height=args.height, fps=args.fps, ch_size=ch_size, save_dir=f"chess_capture/{args.sn}")