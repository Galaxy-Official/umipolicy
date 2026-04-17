
import cv2
import numpy as np
import argparse


def guided_filter(I, p, radius, eps):
    kernel = (2*radius + 1, 2*radius + 1)
    mean_I = cv2.boxFilter(I, cv2.CV_64F, kernel)
    mean_p = cv2.boxFilter(p, cv2.CV_64F, kernel)
    corr_I = cv2.boxFilter(I*I, cv2.CV_64F, kernel)
    corr_Ip = cv2.boxFilter(I*p, cv2.CV_64F, kernel)

    var_I = corr_I - mean_I*mean_I
    cov_Ip = corr_Ip - mean_I*mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, kernel)
    mean_b = cv2.boxFilter(b, cv2.CV_64F, kernel)

    q = mean_a * I + mean_b
    return q


def bilateral_guided(frame, args):
    tmp = cv2.bilateralFilter(frame, args.d, args.sigma_color, args.sigma_space)
    I = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
    output = np.zeros_like(frame, dtype=np.uint8)
    for c in range(3):
        p = tmp[:, :, c].astype(np.float64)
        q = guided_filter(I, p, args.guided_radius, args.guided_eps)
        output[:, :, c] = np.clip(q, 0, 255)
    return output


def freq_filter(frame, args):
    # 对每个通道做频域低通或高通滤波
    h, w = frame.shape[:2]
    crow, ccol = h//2, w//2
    mask = np.zeros((h, w), np.uint8)
    if args.freq_method == 'lowpass':
        cv2.circle(mask, (ccol, crow), args.freq_cutoff, 1, -1)
    else:
        mask[:] = 1
        cv2.circle(mask, (ccol, crow), args.freq_cutoff, 0, -1)
    output = np.zeros_like(frame)
    for c in range(3):
        f = np.fft.fft2(frame[:, :, c])
        fshift = np.fft.fftshift(f)
        fshift_filtered = fshift * mask
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        output[:, :, c] = np.clip(np.abs(img_back), 0, 255)
    return output.astype(np.uint8)


def process_frame(frame, args):
    # 1. 预处理：帧差与中心裁剪已在外部实现
    result = frame.copy()

    # 2. 频域滤波可选
    if args.freq_method:
        result = freq_filter(result, args)

    # 3. 空间域双边+Guided
    if args.use_bilateral_guided:
        result = bilateral_guided(result, args)

    return result


def process_video(input_path, output_path, args):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {input_path}")

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise IOError("Cannot read first frame")
    h, w = first_frame.shape[:2]
    roi_x, roi_y = w//6, h//6
    roi_w, roi_h = w*2//3, h*2//3
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, fourcc, fps, (roi_w, roi_h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx > 0:
            frame = cv2.addWeighted(frame, 1.0, first_frame, -0.2, 0)
        cropped = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        processed = process_frame(cropped, args)
        out.write(processed)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='视频去噪：频域 + 双边+Guided')
    parser.add_argument('input', help='输入MP4文件路径')
    parser.add_argument('output', help='输出MP4文件路径')
    parser.add_argument('--freq-method', choices=['lowpass','highpass'],
                        help='启用频域滤波：lowpass 或 highpass')
    parser.add_argument('--freq-cutoff', type=int, default=30,
                        help='频域滤波截止半径 (像素)')
    parser.add_argument('--d', type=int, default=5, help='双边滤波直径')
    parser.add_argument('--sigma-color', type=float, default=75.0)
    parser.add_argument('--sigma-space', type=float, default=75.0)
    parser.add_argument('--guided-radius', type=int, default=8)
    parser.add_argument('--guided-eps', type=float, default=1e-2)
    parser.add_argument('--use-bilateral-guided', action='store_true',
                        help='启用双边 + Guided')
    args = parser.parse_args()
    process_video(args.input, args.output, args)
