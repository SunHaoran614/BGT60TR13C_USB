import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import radar_func as rf
import json
import radar_settings  # 导入雷达设置模块

# 雷达参数 - 从设置模块中获取
def load_radar_config():
    """加载雷达配置参数"""
    try:
        # 直接从雷达设置模块中获取参数
        return radar_settings.get_radar_params()
    except Exception as e:
        print(f"警告: 无法从雷达设置模块加载参数: {e}")
        # 如果遇到问题，将使用radar_settings.py中的默认参数
        return radar_settings.RADAR_PARAMS

def load_radar_data(file_path):
    """加载雷达数据文件"""
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        sys.exit(1)
    
    print(f"加载雷达数据: {file_path}")
    data = np.load(file_path)
    print(f"数据形状: {data.shape}")
    return data

def ensure_dir_exists(dir_path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"创建目录: {dir_path}")
        # 检查目录是否已成功创建
        if os.path.exists(dir_path):
            print(f"目录 {dir_path} 创建成功")
        else:
            print(f"错误: 无法创建目录 {dir_path}")

def test_range_fft(radar_data, output_dir, radar_params):
    """测试距离FFT函数"""
    print("\n测试距离FFT...")
    # 提取单一帧进行测试
    test_frame = 0
    
    # 应用不同的窗函数
    windows = ['hanning', 'hamming', 'blackman', 'rectangular']
    plt.figure(figsize=(12, 10))
    
    # 计算距离轴（米）
    range_resolution = radar_params['range_resolution']
    samples = radar_params['num_samples']
    max_range = radar_params['max_range']  # 使用固定的1.7m
    
    # 计算距离轴 - 使用固定的1.7m作为最大不模糊距离
    half_len = samples // 2
    range_axis = np.linspace(0, max_range, half_len)
    
    print(f"距离轴: 最小={range_axis[0]:.3f}m, 最大={range_axis[-1]:.3f}m")
    
    for i, window in enumerate(windows):
        range_fft_data = rf.range_fft(radar_data[test_frame:test_frame+1], window=window)
        
        # 计算平均功率谱（跨天线和chirp）
        range_profile = np.mean(np.abs(range_fft_data[0]), axis=(0, 1))
        
        # 只显示一半的FFT（由于对称性）
        half_len = len(range_profile) // 2
        
        plt.subplot(2, 2, i+1)
        plt.plot(range_axis, 20 * np.log10(range_profile[:half_len] + 1e-10))
        plt.title(f'Range FFT ({window} window)')
        plt.xlabel('Range (m)')
        plt.ylabel('Power (dB)')
        plt.grid(True)
        
        # 设置X轴范围
        plt.xlim(0, max_range)
        
        # 添加垂直线标记距离分辨率
        plt.axvline(x=range_resolution, color='r', linestyle='--', 
                  label=f'Range Resolution: {range_resolution:.3f} m')
        
        # 添加文本注释显示最大距离值
        plt.text(0.98, 0.95, f'Max Range: {max_range:.2f} m',
                transform=plt.gca().transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if i == 0:  # 只在第一个子图显示图例
            plt.legend()
    
    plt.tight_layout()
    try:
        output_file = os.path.join(output_dir, 'range_fft_test.png')
        plt.savefig(output_file)
        print(f"距离FFT测试完成，结果已保存到'{output_file}'")
        # 检查文件是否已成功保存
        if os.path.exists(output_file):
            print(f"文件 {output_file} 保存成功")
        else:
            print(f"错误: 文件 {output_file} 未能保存")
    except Exception as e:
        print(f"保存图像时出错: {e}")
    
    return range_fft_data

def test_doppler_fft(range_fft_data, output_dir, radar_params):
    """测试多普勒FFT函数"""
    print("\n测试多普勒FFT...")
    
    # 应用多普勒FFT
    doppler_fft_data = rf.doppler_fft(range_fft_data)
    
    # 提取单一天线的距离-多普勒图
    rd_map = doppler_fft_data[0, 0]
    
    # 使用固定的最大距离值
    max_range = radar_params['max_range']  # 1.7m
    
    # 计算距离轴
    num_range_bins = rd_map.shape[1]
    range_axis = np.linspace(0, max_range, num_range_bins)
    
    # 计算速度轴（米/秒）
    num_doppler_bins = rd_map.shape[0]
    velocity_axis = np.linspace(-radar_params['max_velocity'], 
                               radar_params['max_velocity'], 
                               num_doppler_bins)
    
    # 显示结果
    plt.figure(figsize=(10, 8))
    rd_map_db = 20 * np.log10(np.abs(rd_map) + 1e-10)
    
    # 重新排列多普勒轴，使零速度在中间
    rd_map_shifted = np.fft.fftshift(rd_map_db, axes=0)
    
    plt.imshow(rd_map_shifted, aspect='auto', cmap='jet',
               extent=[0, max_range,  # 使用固定的1.7m
                      -radar_params['max_velocity'], 
                      radar_params['max_velocity']])
    plt.colorbar(label='Power (dB)')
    plt.title('Range-Doppler Map')
    plt.xlabel('Range (m)')
    plt.ylabel('Velocity (m/s)')
    
    # 添加水平线标记速度分辨率
    plt.axhline(y=radar_params['velocity_resolution'], color='r', linestyle='--')
    plt.axhline(y=-radar_params['velocity_resolution'], color='r', linestyle='--',
               label=f'Velocity Resolution: ±{radar_params["velocity_resolution"]:.3f} m/s')
    
    # 添加垂直线标记距离分辨率
    plt.axvline(x=radar_params['range_resolution'], color='g', linestyle='--',
               label=f'Range Resolution: {radar_params["range_resolution"]:.3f} m')
    
    # 添加文本注释显示最大距离
    plt.text(0.98, 0.95, f'Max Range: {max_range:.2f} m',
            transform=plt.gca().transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.legend(loc='upper right')
    
    try:
        output_file = os.path.join(output_dir, 'doppler_fft_test.png')
        plt.savefig(output_file)
        print(f"多普勒FFT测试完成，结果已保存到'{output_file}'")
    except Exception as e:
        print(f"保存图像时出错: {e}")
    
    return doppler_fft_data

def test_mti_filter(radar_data, output_dir):
    """测试MTI滤波器"""
    print("\n测试MTI滤波器...")
    
    # 选择一部分数据进行测试（避免处理时间过长）
    test_frames = min(100, radar_data.shape[0])
    test_data = radar_data[:test_frames]
    
    # 应用不同阶数的MTI滤波器
    mti_orders = [1, 2, 3]
    
    plt.figure(figsize=(15, 5*len(mti_orders)))
    
    # 选择一个固定的天线、chirp和采样点进行可视化
    antenna = 0
    chirp = 0
    sample = 64  # 中间采样点
    
    # 绘制原始数据
    plt.subplot(len(mti_orders)+1, 1, 1)
    plt.plot(np.abs(test_data[:, antenna, chirp, sample]))
    plt.title('Original Data')
    plt.xlabel('Frame')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # 测试不同阶数的MTI
    for i, order in enumerate(mti_orders):
        # 应用MTI滤波
        mti_data = rf.mti_filter(test_data, filter_order=order)
        
        # 绘制结果
        plt.subplot(len(mti_orders)+1, 1, i+2)
        plt.plot(np.abs(mti_data[:, antenna, chirp, sample]))
        plt.title(f'{order}-Order MTI Filter Result')
        plt.xlabel('Frame')
        plt.ylabel('Amplitude')
        plt.grid(True)
    
    plt.tight_layout()
    try:
        output_file = os.path.join(output_dir, 'mti_filter_test.png')
        plt.savefig(output_file)
        print(f"MTI滤波器测试完成，结果已保存到'{output_file}'")
    except Exception as e:
        print(f"保存图像时出错: {e}")
    
    return rf.mti_filter(test_data, filter_order=2)  # 返回2阶MTI结果用于后续测试

def test_cfar_detector(doppler_fft_data, output_dir, radar_params):
    """测试CFAR探测器"""
    print("\n测试CFAR探测器...")
    
    # 获取单一帧单一天线的距离-多普勒图
    rd_map = np.abs(doppler_fft_data[0, 0])
    
    # 使用固定的最大距离
    max_range = radar_params['max_range']  # 1.7m
    
    # 计算距离轴
    num_range_bins = rd_map.shape[1]
    range_axis = np.linspace(0, max_range, num_range_bins)
    
    # 计算速度轴（米/秒）
    num_doppler_bins = rd_map.shape[0]
    
    # 测试不同的CFAR方法
    cfar_methods = ['CA', 'OS', 'GO']
    
    plt.figure(figsize=(15, 5*len(cfar_methods)))
    
    # 原始距离-多普勒图
    plt.subplot(len(cfar_methods)+1, 1, 1)
    rd_map_db = 20 * np.log10(rd_map + 1e-10)
    
    # 重新排列多普勒轴，使零速度在中间
    rd_map_shifted = np.fft.fftshift(rd_map_db, axes=0)
    
    plt.imshow(rd_map_shifted, aspect='auto', cmap='jet',
               extent=[0, max_range, 
                      -radar_params['max_velocity'], 
                      radar_params['max_velocity']])
    plt.colorbar(label='Power (dB)')
    plt.title('Original Range-Doppler Map')
    plt.xlabel('Range (m)')
    plt.ylabel('Velocity (m/s)')
    
    # 显示最大距离信息
    plt.text(0.98, 0.95, f'Max Range: {max_range:.2f} m',
            transform=plt.gca().transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 应用不同的CFAR方法
    for i, method in enumerate(cfar_methods):
        detections = rf.cfar_detector(rd_map, guard_cells=2, reference_cells=4, pfa=1e-3, method=method)
        
        # 标记检测结果
        plt.subplot(len(cfar_methods)+1, 1, i+2)
        
        plt.imshow(rd_map_shifted, aspect='auto', cmap='jet',
                  extent=[0, max_range, 
                         -radar_params['max_velocity'], 
                         radar_params['max_velocity']])
        plt.colorbar(label='Power (dB)')
        
        # 在检测位置上叠加标记
        y_indices, x_indices = np.where(detections)
        
        # 转换为实际距离和速度
        velocities = []
        ranges = []
        for y, x in zip(y_indices, x_indices):
            # 转换多普勒索引为速度，考虑零速度在中间
            if y < num_doppler_bins // 2:
                velocity = (y + num_doppler_bins // 2) / num_doppler_bins * (2 * radar_params['max_velocity']) - radar_params['max_velocity']
            else:
                velocity = (y - num_doppler_bins // 2) / num_doppler_bins * (2 * radar_params['max_velocity']) - radar_params['max_velocity']
            
            # 转换距离索引为实际距离 - 使用固定的最大不模糊距离
            range_val = x / num_range_bins * max_range
            
            # 所有检测到的目标都应该在最大距离范围内
            velocities.append(velocity)
            ranges.append(range_val)
        
        plt.scatter(ranges, velocities, color='r', s=10, marker='x')
        
        plt.title(f'CFAR Detection Result ({method})')
        plt.xlabel('Range (m)')
        plt.ylabel('Velocity (m/s)')
    
    plt.tight_layout()
    try:
        output_file = os.path.join(output_dir, 'cfar_test.png')
        plt.savefig(output_file)
        print(f"CFAR探测器测试完成，结果已保存到'{output_file}'")
    except Exception as e:
        print(f"保存图像时出错: {e}")

def test_edacm_detector(doppler_fft_data, output_dir, radar_params):
    """测试EDACM探测器"""
    print("\n测试EDACM探测器...")
    
    # 获取单一帧单一天线的距离-多普勒图
    rd_map = np.abs(doppler_fft_data[0, 0])
    
    # 使用固定的最大距离
    max_range = radar_params['max_range']  # 1.7m
    
    # 计算距离轴
    num_range_bins = rd_map.shape[1]
    range_axis = np.linspace(0, max_range, num_range_bins)
    
    # 计算速度轴（米/秒）
    num_doppler_bins = rd_map.shape[0]
    
    # 测试不同的EDACM参数
    thresholds = [10, 13, 16]
    
    plt.figure(figsize=(15, 5*len(thresholds)))
    
    # 原始距离-多普勒图
    plt.subplot(len(thresholds)+1, 1, 1)
    rd_map_db = 20 * np.log10(rd_map + 1e-10)
    
    # 重新排列多普勒轴，使零速度在中间
    rd_map_shifted = np.fft.fftshift(rd_map_db, axes=0)
    
    plt.imshow(rd_map_shifted, aspect='auto', cmap='jet',
               extent=[0, max_range, 
                      -radar_params['max_velocity'], 
                      radar_params['max_velocity']])
    plt.colorbar(label='Power (dB)')
    plt.title('Original Range-Doppler Map')
    plt.xlabel('Range (m)')
    plt.ylabel('Velocity (m/s)')
    
    # 显示最大距离信息
    plt.text(0.98, 0.95, f'Max Range: {max_range:.2f} m',
            transform=plt.gca().transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 应用不同阈值的EDACM
    for i, threshold in enumerate(thresholds):
        detections = rf.edacm_detector(rd_map, window_size=16, threshold_snr=threshold, guard_cells=4)
        
        # 标记检测结果
        plt.subplot(len(thresholds)+1, 1, i+2)
        
        plt.imshow(rd_map_shifted, aspect='auto', cmap='jet',
                  extent=[0, max_range, 
                         -radar_params['max_velocity'], 
                         radar_params['max_velocity']])
        plt.colorbar(label='Power (dB)')
        
        # 在检测位置上叠加标记
        y_indices, x_indices = np.where(detections)
        
        # 转换为实际距离和速度
        velocities = []
        ranges = []
        for y, x in zip(y_indices, x_indices):
            # 转换多普勒索引为速度，考虑零速度在中间
            if y < num_doppler_bins // 2:
                velocity = (y + num_doppler_bins // 2) / num_doppler_bins * (2 * radar_params['max_velocity']) - radar_params['max_velocity']
            else:
                velocity = (y - num_doppler_bins // 2) / num_doppler_bins * (2 * radar_params['max_velocity']) - radar_params['max_velocity']
            
            # 转换距离索引为实际距离 - 使用固定的最大距离
            range_val = x / num_range_bins * max_range
            
            # 所有检测到的目标都应该在显示范围内
            velocities.append(velocity)
            ranges.append(range_val)
        
        plt.scatter(ranges, velocities, color='r', s=10, marker='x')
        
        plt.title(f'EDACM Detection Result (Threshold={threshold} dB)')
        plt.xlabel('Range (m)')
        plt.ylabel('Velocity (m/s)')
    
    plt.tight_layout()
    try:
        output_file = os.path.join(output_dir, 'edacm_test.png')
        plt.savefig(output_file)
        print(f"EDACM探测器测试完成，结果已保存到'{output_file}'")
    except Exception as e:
        print(f"保存图像时出错: {e}")

def test_visualization(doppler_fft_data, output_dir, radar_params):
    """测试可视化函数"""
    print("\n测试距离-多普勒图可视化...")
    
    # 获取单一帧单一天线的距离-多普勒图
    rd_map = np.abs(doppler_fft_data[0, 0])
    
    # 使用固定的最大距离
    max_range = radar_params['max_range']  # 1.7m
    
    # 计算距离轴
    num_range_bins = rd_map.shape[1]
    range_axis = np.linspace(0, max_range, num_range_bins)
    
    # 使用不同的动态范围
    dynamic_ranges = [40, 60, 80]
    
    plt.figure(figsize=(15, 5*len(dynamic_ranges)))
    
    for i, dr in enumerate(dynamic_ranges):
        plt.subplot(len(dynamic_ranges), 1, i+1)
        
        # 转换为功率并归一化
        power_db = 20 * np.log10(np.abs(rd_map) / np.max(np.abs(rd_map)) + 1e-10)
        
        # 限制动态范围
        power_db = np.maximum(power_db, -dr)
        
        # 重新排列多普勒轴，使零速度在中间
        power_db_shifted = np.fft.fftshift(power_db, axes=0)
        
        plt.imshow(power_db_shifted, aspect='auto', cmap='jet',
                  extent=[0, max_range, 
                         -radar_params['max_velocity'], 
                         radar_params['max_velocity']])
        plt.colorbar(label='Power (dB)')
        plt.title(f'Range-Doppler Map (Dynamic Range={dr} dB)')
        plt.xlabel('Range (m)')
        plt.ylabel('Velocity (m/s)')
        
        # 显示最大距离信息
        plt.text(0.98, 0.95, f'Max Range: {max_range:.2f} m',
                transform=plt.gca().transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    try:
        output_file = os.path.join(output_dir, 'visualization_test.png')
        plt.savefig(output_file)
        print(f"可视化测试完成，结果已保存到'{output_file}'")
    except Exception as e:
        print(f"保存图像时出错: {e}")

def main():
    """主函数"""
    # 创建输出目录
    output_dir = "test_results"
    ensure_dir_exists(output_dir)
    
    # 加载雷达参数
    radar_params = load_radar_config()
    print("\n雷达参数:")
    for key, value in radar_params.items():
        print(f"  {key}: {value}")
    
    # 指定雷达数据文件路径
    data_file = "Dataset/Radar_Data/Participant1/0.8m/radar_raw_data.npy"
    
    # 加载数据
    radar_data = load_radar_data(data_file)
    
    # 执行各种测试
    try:
        # 使用绝对路径
        output_dir_abs = os.path.abspath(output_dir)
        print(f"使用绝对路径输出目录: {output_dir_abs}")
        
        range_fft_data = test_range_fft(radar_data, output_dir_abs, radar_params)
        doppler_fft_data = test_doppler_fft(range_fft_data, output_dir_abs, radar_params)
        mti_data = test_mti_filter(radar_data, output_dir_abs)
        test_cfar_detector(doppler_fft_data, output_dir_abs, radar_params)
        test_edacm_detector(doppler_fft_data, output_dir_abs, radar_params)
        test_visualization(doppler_fft_data, output_dir_abs, radar_params)
        
        print(f"\n所有测试完成！结果已保存到 {output_dir_abs} 目录")
        
        # 列出生成的文件
        print("\n生成的文件:")
        if os.path.exists(output_dir_abs):
            files = os.listdir(output_dir_abs)
            for file in files:
                file_path = os.path.join(output_dir_abs, file)
                print(f"  {file} ({os.path.getsize(file_path)} bytes)")
        else:
            print(f"  警告: 目录 {output_dir_abs} 不存在")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")

if __name__ == "__main__":
    main() 