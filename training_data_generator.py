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

# 导入信号分解模块
try:
    import signal_decomposition as sd
    signal_decomp_available = True
except ImportError:
    print("警告: 信号分解模块未安装，将仅使用原始相位数据")
    signal_decomp_available = False

# 导入雷达设置
try:
    import radar_settings
    radar_params = radar_settings.get_radar_params()
except ImportError:
    print("警告：无法导入radar_settings，将使用默认参数")


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

# 信号分解参数设置
# 可用的分解方法:
# - 'EMD': 经验模态分解，将信号分解为多个内在模态函数(IMF)
# - 'EEMD': 集合经验模态分解，解决EMD的模态混叠问题，但计算量更大
# - 'DWT': 离散小波变换，使用小波基对信号进行多尺度分析
# - 'VMD': 变分模态分解，一种非递归的自适应分解方法
VMD_MODES = 3    # VMD分解模态数
DWT_LEVEL = 5    # DWT分解级别
DWT_WAVELET = 'db4'  # DWT小波类型
SAVE_SEPARATE_FILES = True  # 是否为每种分解方法保存单独文件

def apply_signal_decomposition(phase_data, methods=None):
    """
    对相位数据应用不同的信号分解算法，使用signal_decomposition库
    
    参数:
        phase_data: 形状为(samples, 300)的相位数据数组，每行是一个10秒窗口的相位数据
        methods: 要应用的分解方法列表，如果为None则使用VMD方法
    
    返回:
        decomp_results: 包含各种分解结果的字典
    """
    if not signal_decomp_available:
        print("信号分解库未安装，跳过分解步骤")
        return {}
    
    # 如果methods为None，默认使用VMD
    if methods is None:
        methods = ['VMD']
    
    # 如果methods为空列表，直接返回空结果
    if len(methods) == 0:
        print("未指定任何分解方法，返回空结果")
        return {}
    
    # 初始化结果字典
    decomp_results = {}
    n_samples = phase_data.shape[0]
    
    # 遍历每种分解方法
    for method in methods:
        print(f"应用 {method} 分解算法...")
        
        if method == 'EMD':
            # 存储结果 - 为每个样本分配3个模态的空间(大多数生命体征信号可以用前3个IMF表示)
            imfs = np.zeros((n_samples, 3, WINDOW_SIZE))
            
            # 对每个样本应用EMD
            for i, sample in enumerate(phase_data):
                if i % 100 == 0:
                    print(f"  处理EMD样本 {i}/{n_samples}")
                try:
                    # 使用signal_decomposition库的EMD函数
                    all_imfs = sd.apply_emd(sample)
                    # 保存前3个IMF(或更少如果不足3个)
                    for j in range(min(3, len(all_imfs))):
                        imfs[i, j, :] = all_imfs[j]
                except Exception as e:
                    print(f"  样本{i}的EMD分解失败: {e}")
            
            # 保存结果
            decomp_results['EMD'] = imfs
        
        if method == 'EEMD':
            # 存储结果 - 为每个样本分配3个模态的空间
            eimfs = np.zeros((n_samples, 3, WINDOW_SIZE))
            
            # 对每个样本应用EEMD
            for i, sample in enumerate(phase_data):
                if i % 100 == 0:
                    print(f"  处理EEMD样本 {i}/{n_samples}")
                try:
                    # 使用signal_decomposition库的EEMD函数
                    all_eimfs = sd.apply_eemd(sample, noise_width=0.05, ensemble_size=100)
                    # 保存前3个EIMF(或更少如果不足3个)
                    for j in range(min(3, len(all_eimfs))):
                        eimfs[i, j, :] = all_eimfs[j]
                except Exception as e:
                    print(f"  样本{i}的EEMD分解失败: {e}")
            
            # 保存结果
            decomp_results['EEMD'] = eimfs
        
        if method == 'DWT':
            # 存储结果 - 每个样本的各级系数
            dwt_coeffs = np.zeros((n_samples, DWT_LEVEL + 1, WINDOW_SIZE))
            
            # 对每个样本应用DWT
            for i, sample in enumerate(phase_data):
                if i % 100 == 0:
                    print(f"  处理DWT样本 {i}/{n_samples}")
                try:
                    # 使用signal_decomposition库的DWT函数
                    coeffs = sd.apply_dwt(sample, wavelet=DWT_WAVELET, level=DWT_LEVEL)
                    # 保存各级系数
                    for j in range(len(coeffs)):
                        # 调整系数长度以匹配窗口大小
                        coeff = coeffs[j]
                        if len(coeff) < WINDOW_SIZE:
                            padded = np.zeros(WINDOW_SIZE)
                            padded[:len(coeff)] = coeff
                            dwt_coeffs[i, j, :] = padded
                        else:
                            dwt_coeffs[i, j, :WINDOW_SIZE] = coeff[:WINDOW_SIZE]
                except Exception as e:
                    print(f"  样本{i}的DWT分解失败: {e}")
            
            # 保存结果
            decomp_results['DWT'] = dwt_coeffs
        
        if method == 'VMD':
            # 存储结果 - 为每个样本分配K个模态的空间
            vmd_modes = np.zeros((n_samples, VMD_MODES, WINDOW_SIZE))
            
            # 对每个样本应用VMD
            for i, sample in enumerate(phase_data):
                if i % 100 == 0:
                    print(f"  处理VMD样本 {i}/{n_samples}")
                try:
                    # 使用signal_decomposition库的VMD函数
                    u, _, _ = sd.apply_vmd(sample, alpha=2000, K=VMD_MODES)
                    
                    # 保存所有模态
                    for j in range(VMD_MODES):
                        vmd_modes[i, j, :] = u[j, :]
                except Exception as e:
                    print(f"  样本{i}的VMD分解失败: {e}")
            
            # 保存结果
            decomp_results['VMD'] = vmd_modes
    
    return decomp_results

# 处理单个参与者的数据
def process_participant_data(parti=1):
    """处理单个参与者在不同距离下的数据"""
    # 定义距离目录名称和对应的距离值（厘米）
    distances = {'0.3m': 30, '0.6m': 60, '0.9m': 90, '1.2m': 120}
    
    # 存储所有特征和标签
    all_features = []
    all_hr_values = []
    all_rr_values = []  # 存储呼吸率值
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
        
        # 读取参考心率和呼吸率数据
        try:
            hr_df = pd.read_csv(ref_file)
            
            # 尝试找到包含心率数据的列
            hr_column = None
            possible_hr_columns = ['HR (bpm)']
            
            for col in possible_hr_columns:
                if col in hr_df.columns:
                    hr_column = col
                    break
            
            if hr_column is None:
                print(f"警告：无法找到心率列，请检查CSV文件格式")
                continue
                
            hr_values = hr_df[hr_column].values
            print(f"参考心率数据形状: {hr_values.shape}")
            
            # 尝试找到包含呼吸率数据的列
            rr_column = None
            possible_rr_columns = ['RR (bpm)']
            
            for col in possible_rr_columns:
                if col in hr_df.columns:
                    rr_column = col
                    break
            
            if rr_column is None:
                print(f"警告：无法找到呼吸率列，将使用默认值0")
                rr_values = np.zeros_like(hr_values)
            else:
                rr_values = hr_df[rr_column].values
                print(f"参考呼吸率数据形状: {rr_values.shape}")
            
            # 确保心率和呼吸率数据与雷达数据长度匹配（简单方法：截断或填充）
            if len(hr_values) > radar_data.shape[0]:
                hr_values = hr_values[:radar_data.shape[0]]
                rr_values = rr_values[:radar_data.shape[0]]
                print(f"截断心率和呼吸率数据以匹配雷达帧数: {len(hr_values)}")
            elif len(hr_values) < radar_data.shape[0]:
                # 将最后一个值重复以匹配雷达帧数
                hr_padding = np.full(radar_data.shape[0] - len(hr_values), hr_values[-1])
                hr_values = np.concatenate([hr_values, hr_padding])
                
                rr_padding = np.full(radar_data.shape[0] - len(rr_values), rr_values[-1])
                rr_values = np.concatenate([rr_values, rr_padding])
                
                print(f"填充心率和呼吸率数据以匹配雷达帧数: {len(hr_values)}")
        except Exception as e:
            print(f"加载参考数据失败: {e}")
            continue
        
        # 处理雷达数据
        try:
            # 执行距离FFT
            range_profile = range_fft(radar_data)
            print(f"距离谱形状: {range_profile.shape}")
            
            # 应用MTI滤波
            mti_filtered = apply_mti_filter(range_profile)
            print(f"MTI滤波后形状: {mti_filtered.shape}")
            
            # 使用滑动窗口切片数据
            num_frames = mti_filtered.shape[0]
            window_count = 0  # 计数已处理的窗口数
            seen_bins = set()  # 用于记录已经见过的range bin
            
            # 修改循环，使窗口从0秒开始，每次向后移动1秒
            # 但标签使用窗口中心点的值
            for start_idx in range(0, num_frames - WINDOW_SIZE + 1, STEP_SIZE):
                end_idx = start_idx + WINDOW_SIZE
                
                # 提取当前窗口的数据用于计算range bin
                window_data = mti_filtered[start_idx:end_idx]
                
                # 将4D数据转换为2D数据 - 选择第一根天线和第一个chirp
                window_data_2d = window_data[:, 0, 0, :]
                
                # 为当前窗口提取相位和目标bin
                # 为了确保方法的一致性，我们总是计算range bin，但只在必要时打印
                window_phase, target_bin = extract_phase_edacm(window_data_2d, DISTANCE_RESOLUTION, WAVELENGTH, False)
                
                # 如果这是一个新的range bin，打印信息
                if target_bin not in seen_bins:
                    target_distance = target_bin * DISTANCE_RESOLUTION
                    print(f"新目标：距离bin = {target_bin}，距离 = {target_distance:.2f}米")
                    seen_bins.add(target_bin)
                
                # 更新窗口计数
                window_count += 1
                
                # 检查数据是否完整
                if len(window_phase) != WINDOW_SIZE:
                    continue
                
                # 计算窗口中心点索引
                center_idx = start_idx + WINDOW_SIZE // 2
                
                # 直接使用相位数据作为特征
                features = window_phase
                
                # 使用窗口中心点的心率和呼吸率作为标签
                center_hr = hr_values[center_idx]
                center_rr = rr_values[center_idx]
                
                # 将特征、心率、呼吸率和其他信息添加到列表中
                all_features.append(features)
                all_hr_values.append(center_hr)
                all_rr_values.append(center_rr)
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
        all_rr_values = np.array(all_rr_values)
        all_time_points = np.array(all_time_points)
        all_distances = np.array(all_distances)
        
        print(f"生成了 {len(all_features)} 条特征记录")
        return all_features, all_hr_values, all_rr_values, all_time_points, all_distances
    else:
        print("未能生成任何特征记录")
        return None, None, None, None, None

def process_long_duration_data():
    """处理长时间（2小时）数据集"""
    print("处理Long_duration数据集...")
    
    # 存储所有特征和标签
    all_features = []
    all_hr_values = []
    all_rr_values = []
    all_time_points = []
    all_distances = []
    
    # 默认情况下使用60厘米作为距离值（可以根据实际情况调整）
    distance_cm = 60
    
    # 构建雷达数据文件路径
    radar_file = os.path.join(RADAR_DATA_DIR, 'Long_duration', 'radar_raw_data.npy')
    # 构建参考数据文件路径
    ref_file = os.path.join(HR_REF_DATA_DIR, 'Long_duration', 'HR_ref.csv')
    
    if not os.path.exists(radar_file):
        print(f"未找到长时间雷达数据文件: {radar_file}")
        return None, None, None, None, None
        
    if not os.path.exists(ref_file):
        print(f"未找到长时间参考数据文件: {ref_file}")
        return None, None, None, None, None
    
    # 读取参考心率和呼吸率数据
    try:
        hr_df = pd.read_csv(ref_file)
        
        # 尝试找到包含心率数据的列
        hr_column = None
        possible_hr_columns = ['HR (bpm)']
        
        for col in possible_hr_columns:
            if col in hr_df.columns:
                hr_column = col
                break
        
        if hr_column is None:
            print(f"警告：无法找到心率列，请检查CSV文件格式")
            return None, None, None, None, None
            
        hr_values = hr_df[hr_column].values
        print(f"长时间参考心率数据形状: {hr_values.shape}")
        
        # 尝试找到包含呼吸率数据的列
        rr_column = None
        possible_rr_columns = ['RR (bpm)']
        
        for col in possible_rr_columns:
            if col in hr_df.columns:
                rr_column = col
                break
        
        if rr_column is None:
            print(f"警告：无法找到呼吸率列，将使用默认值0")
            rr_values = np.zeros_like(hr_values)
        else:
            rr_values = hr_df[rr_column].values
            print(f"长时间参考呼吸率数据形状: {rr_values.shape}")
    except Exception as e:
        print(f"加载长时间参考数据失败: {e}")
        return None, None, None, None, None
    
    # 获取雷达数据的形状信息，不加载全部数据
    try:
        # 使用np.load的mmap_mode参数，避免将整个数组加载到内存
        radar_data_info = np.load(radar_file, mmap_mode='r')
        total_frames = radar_data_info.shape[0]
        print(f"长时间雷达数据形状: {radar_data_info.shape}")
        
        # 确保心率和呼吸率数据长度匹配雷达数据长度
        if len(hr_values) < total_frames:
            # 将最后一个值重复以匹配雷达帧数
            hr_padding = np.full(total_frames - len(hr_values), hr_values[-1])
            hr_values = np.concatenate([hr_values, hr_padding])
            
            rr_padding = np.full(total_frames - len(rr_values), rr_values[-1])
            rr_values = np.concatenate([rr_values, rr_padding])
            
            print(f"填充心率和呼吸率数据以匹配雷达帧数: {len(hr_values)}")
        elif len(hr_values) > total_frames:
            hr_values = hr_values[:total_frames]
            rr_values = rr_values[:total_frames]
            print(f"截断心率和呼吸率数据以匹配雷达帧数: {len(hr_values)}")
    except Exception as e:
        print(f"获取长时间雷达数据形状失败: {e}")
        return None, None, None, None, None
    
    try:
        # 分批处理数据，每批次处理18000帧（约10分钟，与Participant1数据量相当）
        batch_size = 18000
        
        for batch_start in range(0, total_frames, batch_size):
            batch_end = min(batch_start + batch_size, total_frames)
            batch_frames = batch_end - batch_start
            
            print(f"处理批次 {batch_start//batch_size + 1}/{(total_frames + batch_size - 1)//batch_size}，帧范围: {batch_start}-{batch_end-1}")
            
            # 只加载当前批次的数据
            batch_data = np.load(radar_file, mmap_mode='r')[batch_start:batch_end]
            
            # 执行距离FFT
            range_profile = range_fft(batch_data)
            print(f"批次距离谱形状: {range_profile.shape}")
            
            # 应用MTI滤波
            mti_filtered = apply_mti_filter(range_profile)
            print(f"批次MTI滤波后形状: {mti_filtered.shape}")
            
            # 使用滑动窗口切片数据
            num_frames = len(mti_filtered)
            window_count = 0  # 计数已处理的窗口数
            seen_bins = set()  # 用于记录已经见过的range bin
            
            # 确保窗口不会跨越不同批次的边界
            max_start_idx = num_frames - WINDOW_SIZE
            if max_start_idx < 0:
                print(f"批次帧数 {num_frames} 小于窗口大小 {WINDOW_SIZE}，跳过此批次")
                continue
                
            for start_idx in range(0, max_start_idx + 1, STEP_SIZE):
                end_idx = start_idx + WINDOW_SIZE
                
                # 提取当前窗口的数据用于计算range bin
                window_data = mti_filtered[start_idx:end_idx]
                
                # 将4D数据转换为2D数据 - 选择第一根天线和第一个chirp
                window_data_2d = window_data[:, 0, 0, :]
                
                # 为当前窗口提取相位和目标bin
                # 为了确保方法的一致性，我们总是计算range bin，但只在必要时打印
                window_phase, target_bin = extract_phase_edacm(window_data_2d, DISTANCE_RESOLUTION, WAVELENGTH, False)
                
                # 如果这是一个新的range bin，打印信息
                if target_bin not in seen_bins:
                    target_distance = target_bin * DISTANCE_RESOLUTION
                    print(f"新目标：距离bin = {target_bin}，距离 = {target_distance:.2f}米")
                    seen_bins.add(target_bin)
                
                # 更新窗口计数
                window_count += 1
                
                # 检查数据是否完整
                if len(window_phase) != WINDOW_SIZE:
                    continue
                
                # 计算全局窗口中心点索引
                global_center_idx = batch_start + start_idx + WINDOW_SIZE // 2
                
                # 直接使用相位数据作为特征
                features = window_phase
                
                # 使用窗口中心点的心率和呼吸率作为标签
                center_hr = hr_values[global_center_idx]
                center_rr = rr_values[global_center_idx]
                
                # 将特征、心率、呼吸率和其他信息添加到列表中
                all_features.append(features)
                all_hr_values.append(center_hr)
                all_rr_values.append(center_rr)
                all_time_points.append(global_center_idx / FRAME_RATE)  # 转换为秒
                all_distances.append(distance_cm)
            
            # 释放内存
            del batch_data, range_profile, mti_filtered
            import gc
            gc.collect()
            
    except Exception as e:
        print(f"处理长时间雷达数据时出错: {e}")
        traceback.print_exc()
    
    # 将所有特征和标签转换为NumPy数组
    if all_features and all_hr_values:
        all_features = np.array(all_features)
        all_hr_values = np.array(all_hr_values)
        all_rr_values = np.array(all_rr_values)
        all_time_points = np.array(all_time_points)
        all_distances = np.array(all_distances)
        
        print(f"从长时间数据中生成了 {len(all_features)} 条特征记录")
        return all_features, all_hr_values, all_rr_values, all_time_points, all_distances
    else:
        print("未能从长时间数据生成任何特征记录")
        return None, None, None, None, None

def save_training_data(apply_decomposition=True, decomp_methods=None, skip_original_generation=True):
    """
    保存训练数据为NPY格式（适合深度学习）
    
    参数:
        apply_decomposition: 是否应用信号分解算法
        decomp_methods: 要应用的信号分解方法列表，默认为None(使用所有可用方法)
        skip_original_generation: 如果为True，则跳过原始数据生成，直接从现有的training_data_original.npz加载数据
    """
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    original_data_path = os.path.join(OUTPUT_DIR, 'training_data_original.npz')
    
    # 存储数据变量的初始化
    features = None
    hr_values = None
    rr_values = None
    time_points = None
    distances = None
    data_sources = None
    metadata = None
    
    # 完全根据参数控制是否生成原始数据
    if skip_original_generation:
        try:
            print(f"跳过原始数据生成，尝试从 {original_data_path} 加载数据...")
            loaded_data = np.load(original_data_path)
            features = loaded_data['features']
            hr_values = loaded_data['hr_values']
            rr_values = loaded_data['rr_values']
            time_points = loaded_data['time_points']
            distances = loaded_data['distances']
            data_sources = loaded_data['data_sources']
            
            # 创建元数据数组供后续分解使用
            metadata = np.column_stack((
                hr_values, 
                rr_values, 
                time_points, 
                time_points - WINDOW_SIZE_SECONDS/2,  # window_start
                time_points + WINDOW_SIZE_SECONDS/2,  # window_end
                distances
            ))
            
            print(f"成功加载原始训练数据，特征形状: {features.shape}")
        except Exception as e:
            print(f"错误：无法加载原始数据文件 {original_data_path}")
            print(f"具体错误: {str(e)}")
            print("将重新生成原始数据")
            skip_original_generation = False  # 如果加载失败，则转为生成数据
    
    # 如果需要生成原始数据
    if not skip_original_generation:
        # 存储所有特征和标签
        all_features = []
        all_hr_values = []
        all_rr_values = []
        all_time_points = []
        all_distances = []
        all_data_sources = []  # 添加数据源标识
        
        # 处理Participant1数据
        print("处理Participant1数据...")
        p1_features, p1_hr_values, p1_rr_values, p1_time_points, p1_distances = process_participant_data(parti=1)
        
        if p1_features is not None:
            all_features.append(p1_features)
            all_hr_values.append(p1_hr_values)
            all_rr_values.append(p1_rr_values)
            all_time_points.append(p1_time_points)
            all_distances.append(p1_distances)
            # 添加数据源标识
            all_data_sources.append(np.full(len(p1_features), 'Participant1'))
        
        # 处理Long_duration数据
        print("处理Long_duration数据...")
        ld_features, ld_hr_values, ld_rr_values, ld_time_points, ld_distances = process_long_duration_data()
        
        if ld_features is not None:
            all_features.append(ld_features)
            all_hr_values.append(ld_hr_values)
            all_rr_values.append(ld_rr_values)
            all_time_points.append(ld_time_points)
            all_distances.append(ld_distances)
            # 添加数据源标识
            all_data_sources.append(np.full(len(ld_features), 'Long_duration'))
        
        # 合并所有数据
        if all_features:
            features = np.concatenate(all_features)
            hr_values = np.concatenate(all_hr_values)
            rr_values = np.concatenate(all_rr_values)
            time_points = np.concatenate(all_time_points)
            distances = np.concatenate(all_distances)
            data_sources = np.concatenate(all_data_sources)
            
            print(f"总共生成了 {len(features)} 条原始特征记录")
            
            # 创建元数据数组
            metadata = np.column_stack((
                hr_values, 
                rr_values, 
                time_points, 
                time_points - WINDOW_SIZE_SECONDS/2,  # window_start
                time_points + WINDOW_SIZE_SECONDS/2,  # window_end
                distances
            ))
            
            # 保存原始相位特征数据为.npz格式
            np.savez(
                original_data_path, 
                features=features, 
                metadata=metadata,
                hr_values=hr_values,
                rr_values=rr_values,
                time_points=time_points,
                distances=distances,
                data_sources=data_sources
            )
            print(f"原始训练数据已保存到: {original_data_path} (NPZ格式)")
            
            # 显示数据样例信息
            print(f"特征形状: {features.shape}")
            print(f"标签形状: {hr_values.shape}")
        else:
            print("警告: 未生成任何训练数据")
            return
    
    # 检查是否成功获取了数据
    if features is None or len(features) == 0:
        print("错误：无法获取有效数据，无法继续进行信号分解")
        return
    
    # 应用信号分解算法
    if apply_decomposition and signal_decomp_available:
        # 检查分解方法列表是否为空
        if decomp_methods is not None and len(decomp_methods) > 0:
            print("应用信号分解算法...")
            print(f"使用的分解方法: {decomp_methods}")
            
            # 应用信号分解
            decomp_results = apply_signal_decomposition(features, decomp_methods)
            
            if not decomp_results:
                print("警告：信号分解未生成任何结果")
                return
            
            # 如果选择分别保存不同的分解结果
            if SAVE_SEPARATE_FILES:
                for method, decomp_data in decomp_results.items():
                    print(f"处理 {method} 分解结果...")
                    
                    # 直接保存为.npz文件，保持多维结构
                    method_output_path = os.path.join(OUTPUT_DIR, f'training_data_{method}.npz')
                    np.savez(
                        method_output_path,
                        decomp_data=decomp_data,
                        metadata=metadata,
                        hr_values=hr_values,
                        rr_values=rr_values,
                        time_points=time_points,
                        distances=distances,
                        data_sources=data_sources
                    )
                    print(f"{method} 训练数据已保存到: {method_output_path} (NPZ格式)")
            
            else:
                # 将所有分解结果合并到一个文件中
                print("合并所有分解结果...")
                
                # 保存为.npz文件
                all_results_output_path = os.path.join(OUTPUT_DIR, 'training_data_all_features.npz')
                save_dict = {
                    'original_features': features,
                    'metadata': metadata,
                    'hr_values': hr_values,
                    'rr_values': rr_values,
                    'time_points': time_points,
                    'distances': distances
                }
                # 添加各种分解结果
                for method, decomp_data in decomp_results.items():
                    save_dict[f'{method}_data'] = decomp_data
                
                np.savez(all_results_output_path, **save_dict)
                print(f"包含所有分解特征的训练数据已保存到: {all_results_output_path} (NPZ格式)")
        else:
            print("跳过信号分解：未指定任何分解方法")
    elif not signal_decomp_available:
        print("警告：信号分解模块不可用，跳过信号分解步骤")
    elif not apply_decomposition:
        print("信息：根据参数设置，跳过信号分解步骤")
    
    # 打印每个距离的数据数量
    if distances is not None:
        for dist in np.unique(distances):
            count = np.sum(distances == dist)
            print(f"距离 {dist}cm 的数据片段数量: {count}")
    
    # 打印每个数据源的数据数量
    if data_sources is not None:
        for source in np.unique(data_sources):
            count = np.sum(data_sources == source)
            print(f"数据源 {source} 的数据片段数量: {count}")

if __name__ == "__main__":
    """
    主函数 - 生成训练数据并应用信号分解
    
    可用的信号分解方法:
    - 'EMD':  经验模态分解
    - 'EEMD': 集合经验模态分解
    - 'DWT':  离散小波变换
    - 'VMD':  变分模态分解
    """
    # 直接指定要使用的信号分解方法
    methods_to_use = ['EEMD']  # 只使用VMD方法
    # methods_to_use = ['EMD', 'EEMD', 'DWT', 'VMD']  # 使用所有方法
    # methods_to_use = []  # 不使用任何信号分解方法
    
    save_training_data(
        apply_decomposition=True,            # 是否应用信号分解
        decomp_methods=methods_to_use,       # 指定要使用的信号分解方法
        skip_original_generation=True        # 如果原始数据文件存在则直接使用
    ) 