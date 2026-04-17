import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def generate_frequency_spectrum(image_path, output_path=None, log_scale=True):
    """
    读取JPG图像并生成频率-强度频谱图
    
    参数:
    image_path: 输入JPG图像路径
    output_path: 频谱图保存路径(可选)
    log_scale: 是否使用对数尺度增强显示效果(默认True)
    """
    try:
        # 读取图像并转换为灰度
        img = Image.open(image_path).convert('L')
        img_array = np.array(img).astype(np.float32)
        
        # 执行二维傅里叶变换
        fft = np.fft.fft2(img_array)
        fft_shift = np.fft.fftshift(fft)  # 将低频移到中心
        
        # 计算幅度谱
        magnitude_spectrum = np.abs(fft_shift)
        
        # 计算径向平均频谱（一维）
        h, w = magnitude_spectrum.shape
        center_x, center_y = w // 2, h // 2
        
        # 创建距离中心点的网格
        y, x = np.indices((h, w))
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        r = r.astype(np.int32)
        
        # 计算每个半径的平均幅度
        tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / np.maximum(nr, 1)  # 避免除以零
        
        # 计算频率值（以像素为单位）
        max_r = min(center_x, center_y)
        frequencies = np.arange(0, max_r)
        radial_profile = radial_profile[:max_r]
        
        # 应用对数变换
        if log_scale:
            radial_profile = np.log1p(radial_profile)
        
        # 创建频率-强度图
        plt.figure(figsize=(12, 6))
        
        # 绘制频率-强度曲线
        plt.plot(frequencies, radial_profile, 'b-', linewidth=1.5)
        plt.title('图像频率-强度频谱')
        plt.xlabel('频率 (空间频率)')
        plt.ylabel('强度 (对数尺度)' if log_scale else '强度')
        plt.grid(True, alpha=0.3)
        
        # 添加关键点标记
        peak_freq = frequencies[np.argmax(radial_profile)]
        plt.axvline(x=peak_freq, color='r', linestyle='--', alpha=0.7, 
                    label=f'峰值频率: {peak_freq:.1f}')
        
        # 添加DC分量标记
        plt.axvline(x=0, color='g', linestyle='-', alpha=0.5, label='DC分量')
        
        # 添加高频区域标记
        high_freq = frequencies[-int(len(frequencies)*0.1)]
        plt.axvline(x=high_freq, color='m', linestyle='--', alpha=0.7, 
                    label=f'高频区域: >{high_freq:.1f}')
        
        plt.legend()
        
        # 保存或显示结果
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            print(f"频谱图已保存至: {output_path}")
        else:
            plt.show()
        
        return frequencies, radial_profile
    
    except Exception as e:
        print(f"处理图像时出错: {e}")
        return None, None

# 使用示例
if __name__ == "__main__":
    input_image = "/home/yushun/Workspace/Mini-Tele/lerobot/scripts/test/000.jpg"  # 替换为你的输入图像路径
    output_spectrum = "/home/yushun/Workspace/Mini-Tele/lerobot/scripts/test/f_ori.png"  # 频谱图输出路径
    
    freqs, profile = generate_frequency_spectrum(input_image, output_spectrum)
    
    # 输出关键频率点（可用于滤波）
    if freqs is not None and profile is not None:
        peak_idx = np.argmax(profile)
        print(f"峰值频率: {freqs[peak_idx]:.2f}")
        
        # 找到强度下降到峰值50%的频率点
        half_power = profile[peak_idx] * 0.5
        half_power_idx = np.where(profile < half_power)[0]
        if len(half_power_idx) > 0 and half_power_idx[0] > peak_idx:
            print(f"半功率频率: {freqs[half_power_idx[0]]:.2f}")