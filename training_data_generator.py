"""
    生成深度学习的训练数据
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import traceback
from radar_func import range_fft, mti_filter as apply_mti_filter, extract_phase_edacm

# 导入雷达设置
try:
    import radar_settings
    radar_params = radar_settings.get_radar_params()
except ImportError:
    print("警告：无法导入radar_settings，将使用默认参数")
    radar_params = {
        'fft_size': 128,
        'window_type': 'blackman',
        'range_resolution': 0.027,
        'wavelength': 0.00494,
        'distance_resolution': 0.027
    }

# 定义数据集路径
DATASET_ROOT = 'Dataset'
RADAR_DATA_DIR = os.path.join(DATASET_ROOT, 'Radar_Data')
HR_REF_DATA_DIR = os.path.join(DATASET_ROOT, 'HR_Ref_Data')
OUTPUT_DIR = 'training_data'

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 参数设置
WINDOW_SIZE_SECONDS = 10  # 10秒窗口
STEP_SIZE_SECONDS = 1     # 1秒步长
FRAME_RATE = 30           # 更正：帧率为30Hz

# 从radar_settings获取参数
FFT_SIZE = radar_params.get('fft_size', 128)
WINDOW_TYPE = radar_params.get('window_type', 'blackman')
RANGE_RESOLUTION = radar_params.get('range_resolution', 0.027)
WAVELENGTH = radar_params.get('wavelength', 0.00494)
DISTANCE_RESOLUTION = radar_params.get('distance_resolution', 0.027)

# 窗口大小（采样点数）
WINDOW_SIZE = int(WINDOW_SIZE_SECONDS * FRAME_RATE)
# 步进大小（采样点数）
STEP_SIZE = int(STEP_SIZE_SECONDS * FRAME_RATE)

# 处理单个参与者的数据
def process_participant_data(parti=1):
    """处理单个参与者在不同距离下的数据"""
    # 定义距离目录名称和对应的距离值（厘米）
    distances = {'0.4m': 40, '0.8m': 80, '1.2m': 120}
    
    # 存储所有特征和标签
    all_features = []
    all_hr_values = []
    all_time_points = []  # 保存窗口中心时间点
    all_distances = []    # 保存距离信息
    
    # 遍历所有距离
    for distance_dir, distance_cm in distances.items():
        # 构建雷达数据文件路径
        radar_file = os.path.join(RADAR_DATA_DIR, f'Participant{parti}', distance_dir, 'radar_raw_data.npy')
        # 构建参考心率数据文件路径
        ref_file = os.path.join(HR_REF_DATA_DIR, f'Participant{parti}', distance_dir, 'HR_ref.csv')
        
        if not os.path.exists(radar_file):
            print(f"未找到雷达数据文件: {radar_file}")
            continue
            
        if not os.path.exists(ref_file):
            print(f"未找到参考心率数据文件: {ref_file}")
            continue
            
        print(f"处理参与者 {parti} 在距离 {distance_dir} 的数据...")
        
        # 读取雷达数据
        try:
            radar_data = np.load(radar_file)
            print(f"雷达数据形状: {radar_data.shape}")
        except Exception as e:
            print(f"加载雷达数据失败: {e}")
            continue
        
        # 读取参考心率数据
        try:
            hr_df = pd.read_csv(ref_file)
            
            # 尝试找到包含心率数据的列
            hr_column = None
            possible_columns = ['HR (bpm)']
            
            for col in possible_columns:
                if col in hr_df.columns:
                    hr_column = col
                    break
            
            
            hr_values = hr_df[hr_column].values
            print(f"参考心率数据形状: {hr_values.shape}")
            
            # 确保心率数据与雷达数据长度匹配（简单方法：截断或填充）
            if len(hr_values) > radar_data.shape[0]:
                hr_values = hr_values[:radar_data.shape[0]]
                print(f"截断心率数据以匹配雷达帧数: {len(hr_values)}")
            elif len(hr_values) < radar_data.shape[0]:
                # 将最后一个心率值重复以匹配雷达帧数
                padding = np.full(radar_data.shape[0] - len(hr_values), hr_values[-1])
                hr_values = np.concatenate([hr_values, padding])
                print(f"填充心率数据以匹配雷达帧数: {len(hr_values)}")
        except Exception as e:
            print(f"加载参考心率数据失败: {e}")
            continue
        
        # 处理雷达数据
        try:
            # 执行距离FFT
            range_profile = range_fft(radar_data)
            print(f"距离谱形状: {range_profile.shape}")
            
            # 应用MTI滤波
            mti_filtered = apply_mti_filter(range_profile)
            print(f"MTI滤波后形状: {mti_filtered.shape}")
            
            # 相位提取
            phase_data = extract_phase_edacm(mti_filtered, DISTANCE_RESOLUTION, WAVELENGTH)
            print(f"相位数据形状: {phase_data.shape}")
            
            # 使用滑动窗口切片数据
            num_frames = len(phase_data)
            
            # 修改循环，使窗口从0秒开始，每次向后移动1秒
            # 但标签使用窗口中心点的值
            for start_idx in range(0, num_frames - WINDOW_SIZE + 1, STEP_SIZE):
                end_idx = start_idx + WINDOW_SIZE
                
                # 提取当前窗口的相位数据
                window_phase = phase_data[start_idx:end_idx]
                
                # 检查数据是否完整
                if len(window_phase) != WINDOW_SIZE:
                    continue
                
                # 计算窗口中心点索引
                center_idx = start_idx + WINDOW_SIZE // 2
                
                # 直接使用相位数据作为特征
                features = window_phase
                
                # 使用窗口中心点的心率作为标签
                center_hr = hr_values[center_idx]
                
                # 将特征、心率和其他信息添加到列表中
                all_features.append(features)
                all_hr_values.append(center_hr)
                all_time_points.append(center_idx / FRAME_RATE)  # 转换为秒
                all_distances.append(distance_cm)
                
        except Exception as e:
            print(f"处理雷达数据时出错: {e}")
            traceback.print_exc()
            continue
    
    # 将所有特征和标签转换为NumPy数组
    if all_features and all_hr_values:
        all_features = np.array(all_features)
        all_hr_values = np.array(all_hr_values)
        all_time_points = np.array(all_time_points)
        all_distances = np.array(all_distances)
        
        print(f"生成了 {len(all_features)} 条特征记录")
        return all_features, all_hr_values, all_time_points, all_distances
    else:
        print("未能生成任何特征记录")
        return None, None, None, None

def save_training_data():
    """保存训练数据到CSV文件"""
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 处理参与者数据
    features, hr_values, time_points, distances = process_participant_data(parti=1)
    
    if features is not None and hr_values is not None:
        # 创建特征名称列表
        feature_names = [f'feature_{i+1}' for i in range(features.shape[1])]
        
        # 创建DataFrame
        data_dict = {name: features[:, i] for i, name in enumerate(feature_names)}
        data_dict['heart_rate'] = hr_values
        data_dict['time_point'] = time_points  # 添加时间点信息
        data_dict['window_start'] = time_points - WINDOW_SIZE_SECONDS/2  # 窗口开始时间
        data_dict['window_end'] = time_points + WINDOW_SIZE_SECONDS/2    # 窗口结束时间
        data_dict['distance_cm'] = distances    # 添加距离信息
        
        # 创建训练数据DataFrame
        train_df = pd.DataFrame(data_dict)
        
        # 保存到CSV
        output_path = os.path.join(OUTPUT_DIR, 'training_data.csv')
        train_df.to_csv(output_path, index=False)
        print(f"训练数据已保存到: {output_path}")
        
        # 显示数据前几行
        print("训练数据预览:")
        print(train_df.head())
        
        # 打印每个距离的数据数量
        for dist in np.unique(distances):
            count = np.sum(distances == dist)
            print(f"距离 {dist}cm 的数据片段数量: {count}")
    else:
        print("警告: 未生成训练数据")

if __name__ == "__main__":
    save_training_data() 