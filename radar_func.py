import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def range_fft(data, window='hanning', zero_padding_factor=1):
    """
    对雷达数据执行距离FFT（在采样点维度）
    
    参数:
        data: 形状为(frames, antennas, chirps, samples)的雷达原始数据
        window: 窗函数类型，默认为hanning
        zero_padding_factor: 零填充因子，默认为1（不进行零填充）
    
    返回:
        距离FFT结果，形状为(frames, antennas, chirps, samples*zero_padding_factor)
    """
    frames, antennas, chirps, samples = data.shape
    
    # 应用窗函数
    if window.lower() == 'hanning':
        win = np.hanning(samples)
    elif window.lower() == 'hamming':
        win = np.hamming(samples)
    elif window.lower() == 'blackman':
        win = np.blackman(samples)
    elif window.lower() == 'rectangular' or window.lower() == 'none':
        win = np.ones(samples)
    else:
        raise ValueError(f"不支持的窗函数类型: {window}")
    
    # 准备输出数组
    fft_size = samples * zero_padding_factor
    range_fft_data = np.zeros((frames, antennas, chirps, fft_size), dtype=complex)
    
    # 应用窗函数并执行FFT
    for f in range(frames):
        for a in range(antennas):
            for c in range(chirps):
                windowed_data = data[f, a, c, :] * win
                range_fft_data[f, a, c, :] = np.fft.fft(windowed_data, n=fft_size)
    
    return range_fft_data



def doppler_fft(range_fft_data, window='hanning', zero_padding_factor=1):
    """
    对距离FFT结果执行多普勒FFT（在chirp维度）
    
    参数:
        range_fft_data: 距离FFT结果，形状为(frames, antennas, chirps, range_bins)
        window: 窗函数类型，默认为hanning
        zero_padding_factor: 零填充因子，默认为1（不进行零填充）
    
    返回:
        距离-多普勒FFT结果，形状为(frames, antennas, chirps*zero_padding_factor, range_bins)
    """
    frames, antennas, chirps, range_bins = range_fft_data.shape
    
    # 应用窗函数
    if window.lower() == 'hanning':
        win = np.hanning(chirps)
    elif window.lower() == 'hamming':
        win = np.hamming(chirps)
    elif window.lower() == 'blackman':
        win = np.blackman(chirps)
    elif window.lower() == 'rectangular' or window.lower() == 'none':
        win = np.ones(chirps)
    else:
        raise ValueError(f"不支持的窗函数类型: {window}")
    
    # 准备输出数组
    fft_size = chirps * zero_padding_factor
    doppler_fft_data = np.zeros((frames, antennas, fft_size, range_bins), dtype=complex)
    
    # 转置数据以便在chirp维度上执行FFT
    for f in range(frames):
        for a in range(antennas):
            for r in range(range_bins):
                windowed_data = range_fft_data[f, a, :, r] * win
                doppler_fft_data[f, a, :, r] = np.fft.fft(windowed_data, n=fft_size)
    
    return doppler_fft_data

def mti_filter(data, filter_order=2):
    """
    应用移动目标指示(MTI)滤波器，去除静态杂波
    使用均值相消法实现MTI
    
    参数:
        data: 形状为(frames, antennas, chirps, samples)的雷达原始数据
        filter_order: 保留参数（为兼容接口），在均值相消实现中不起作用
    
    返回:
        MTI处理后的数据，形状与输入相同
    """
    # 获取数据维度
    frames, antennas, chirps, samples = data.shape
    
    # 使用与原始数据相同类型的空数组初始化结果
    mti_data = np.zeros_like(data)
    
    # 分别处理每个天线和chirp，避免一次性分配大量内存
    for a in range(antennas):
        for c in range(chirps):
            for s in range(samples):
                # 提取当前处理的1D时间序列
                time_series = data[:, a, c, s]
                # 计算该时间序列的均值
                mean_val = np.mean(time_series)
                # 减去均值并存储结果
                mti_data[:, a, c, s] = time_series - mean_val
    
    return mti_data

def cfar_detector(rd_matrix, guard_cells=2, reference_cells=4, pfa=1e-4, method='ca'):
    """
    恒虚警率(CFAR)检测器，用于信号探测
    
    参数:
        rd_matrix: 距离-多普勒矩阵 (2D数组)
        guard_cells: 保护单元数量
        reference_cells: 参考单元数量
        pfa: 虚警概率
        method: CFAR方法 ('ca'=均匀平均, 'os'=有序统计量, 'go'=最大值)
    
    返回:
        二值掩码指示目标位置
    """
    rows, cols = rd_matrix.shape
    # 功率谱
    power = np.abs(rd_matrix) ** 2
    
    # 计算阈值因子
    if method.lower() == 'ca':
        # Cell Averaging CFAR
        num_ref_cells = reference_cells * 4  # 四边参考单元总数
        threshold_factor = num_ref_cells * (pfa ** (-1/num_ref_cells) - 1)
    elif method.lower() == 'os':
        # Ordered Statistic CFAR
        threshold_factor = 1 / pfa
    elif method.lower() == 'go':
        # Greatest Of CFAR
        threshold_factor = (pfa ** (-1/(2*reference_cells)) - 1)
    else:
        raise ValueError(f"不支持的CFAR方法: {method}")
    
    # 初始化结果矩阵
    detections = np.zeros((rows, cols), dtype=bool)
    
    # 窗口大小
    window_size = guard_cells + reference_cells
    
    # 应用CFAR
    for i in range(window_size, rows - window_size):
        for j in range(window_size, cols - window_size):
            # 当前单元值
            cell = power[i, j]
            
            # 定义保护区域和参考区域
            top = power[i-window_size:i-guard_cells, j]
            bottom = power[i+guard_cells+1:i+window_size+1, j]
            left = power[i, j-window_size:j-guard_cells]
            right = power[i, j+guard_cells+1:j+window_size+1]
            
            # 合并参考单元
            reference = np.concatenate((top, bottom, left, right))
            
            # 计算阈值
            if method.lower() == 'ca':
                threshold = np.mean(reference) * threshold_factor
            elif method.lower() == 'os':
                # 使用排序后的第k个值
                k = int(len(reference) * 0.75)  # 通常使用75%位置的值
                threshold = np.sort(reference)[k] * threshold_factor
            elif method.lower() == 'go':
                # 左右、上下取最大值
                mean1 = max(np.mean(top), np.mean(bottom))
                mean2 = max(np.mean(left), np.mean(right))
                threshold = max(mean1, mean2) * threshold_factor
            
            # 检测
            if cell > threshold:
                detections[i, j] = True
    
    return detections

def edacm_detector(data, window_size=16, threshold_snr=13, guard_cells=4):
    """
    能量探测自适应恒虚警率(EDACM)探测器，用于高灵敏度目标检测
    
    参数:
        data: 雷达数据，可以是原始数据或FFT后的结果
        window_size: 噪声估计的窗口大小
        threshold_snr: 检测阈值(dB)
        guard_cells: 保护单元数量
    
    返回:
        二值掩码指示目标位置
    """
    if len(data.shape) > 2:
        # 如果是多维数据，转换为2D进行处理
        # 假设我们处理一帧多普勒-距离图
        if len(data.shape) == 4:  # (frames, antennas, doppler, range)
            # 使用第一帧第一天线的数据
            rd_matrix = np.abs(data[0, 0, :, :])
        elif len(data.shape) == 3:  # (antennas, doppler, range)
            # 使用第一天线的数据
            rd_matrix = np.abs(data[0, :, :])
        else:
            rd_matrix = np.abs(data)
    else:
        rd_matrix = np.abs(data)
    
    rows, cols = rd_matrix.shape
    
    # 转换为功率
    power = rd_matrix ** 2
    
    # 初始化结果矩阵
    detections = np.zeros((rows, cols), dtype=bool)
    
    # 将阈值从dB转换为线性尺度
    linear_threshold = 10 ** (threshold_snr / 10)
    
    # 自适应阈值计算
    for i in range(rows):
        for j in range(cols):
            # 定义窗口边界
            r_start = max(0, i - window_size)
            r_end = min(rows, i + window_size + 1)
            c_start = max(0, j - window_size)
            c_end = min(cols, j + window_size + 1)
            
            # 排除保护单元定义窗口
            window = power[r_start:r_end, c_start:c_end].copy()
            
            # 去除保护单元区域
            guard_r_start = max(0, i - guard_cells - r_start)
            guard_r_end = min(window.shape[0], i + guard_cells + 1 - r_start)
            guard_c_start = max(0, j - guard_cells - c_start)
            guard_c_end = min(window.shape[1], j + guard_cells + 1 - c_start)
            
            # 将保护单元设为NaN
            window[guard_r_start:guard_r_end, guard_c_start:guard_c_end] = np.nan
            
            # 计算噪声水平(排除NaN值)
            noise_level = np.nanmedian(window)
            
            # 检测
            if power[i, j] > noise_level * linear_threshold:
                detections[i, j] = True
    
    return detections

def calculate_range_resolution(bandwidth, c=3e8):
    """
    计算距离分辨率
    
    参数:
        bandwidth: 频率带宽(Hz)
        c: 光速(m/s)，默认为3e8
    
    返回:
        距离分辨率(m)
    """
    return c / (2 * bandwidth)

def calculate_velocity_resolution(wavelength, total_time):
    """
    计算速度分辨率
    
    参数:
        wavelength: 信号波长(m)
        total_time: 测量总时间(s)
    
    返回:
        速度分辨率(m/s)
    """
    return wavelength / (2 * total_time)

def calculate_snr(signal, noise_level=None):
    """
    计算信噪比
    
    参数:
        signal: 信号数据
        noise_level: 已知噪声水平，如不提供则估计
    
    返回:
        SNR(dB)
    """
    signal_power = np.mean(np.abs(signal) ** 2)
    
    if noise_level is None:
        # 估计噪声水平
        noise_power = np.var(signal)
    else:
        noise_power = noise_level
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def visualize_range_doppler(rd_matrix, dynamic_range=60, title='Range-Doppler Map'):
    """
    可视化距离-多普勒图
    
    参数:
        rd_matrix: 距离-多普勒矩阵
        dynamic_range: 动态范围(dB)
        title: 图表标题
    """
    # 转换为功率并归一化
    power_db = 20 * np.log10(np.abs(rd_matrix) / np.max(np.abs(rd_matrix)) + 1e-10)
    
    # 限制动态范围
    power_db = np.maximum(power_db, -dynamic_range)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(power_db, aspect='auto', cmap='jet')
    plt.colorbar(label='Power (dB)')
    plt.title(title)
    plt.xlabel('Range (Bins)')
    plt.ylabel('Doppler (Bins)')
    plt.tight_layout()
    
    return plt.gcf()  # 返回图形对象

def extract_phase(range_fft_data, target_range_bin):
    """
    从距离FFT结果中提取特定距离的相位信息
    
    参数:
        range_fft_data: 距离FFT结果，形状为(frames, antennas, chirps, range_bins)
        target_range_bin: 目标距离bin索引
    
    返回:
        相位信息，形状为(frames, antennas, chirps)
    """
    frames, antennas, chirps, _ = range_fft_data.shape
    
    # 提取特定距离bin的数据
    bin_data = range_fft_data[:, :, :, target_range_bin]
    
    # 提取相位信息
    phase = np.angle(bin_data)
    
    return phase

def unwrap_phase(phase):
    """
    展开相位，消除相位跳变
    
    参数:
        phase: 相位数据，任意维度
    
    返回:
        展开后的相位数据，与输入维度相同
    """
    return np.unwrap(phase)

def phase_to_displacement(phase, wavelength):
    """
    将相位变化转换为位移
    
    参数:
        phase: 相位数据(弧度)
        wavelength: 雷达波长(米)
    
    返回:
        位移数据(米)
    """
    # 相位到位移的转换，考虑到往返路径(2π相位变化对应半个波长的位移)
    return phase * wavelength / (4 * np.pi)

def get_respiratory_heartrate(displacement, fs, filter_resp=(0.1, 0.5), filter_hr=(0.8, 2.0)):
    """
    从位移数据中分离提取呼吸率和心率信号
    
    参数:
        displacement: 位移时间序列数据
        fs: 采样率(Hz)
        filter_resp: 呼吸频率范围(Hz)，典型值(0.1-0.5Hz)对应6-30次/分钟
        filter_hr: 心率频率范围(Hz)，典型值(0.8-2.0Hz)对应48-120次/分钟
    
    返回:
        resp_signal: 呼吸信号
        hr_signal: 心率信号
    """
    from scipy.signal import butter, filtfilt
    
    # 设计呼吸带通滤波器
    b_resp, a_resp = butter(2, [filter_resp[0]/(fs/2), filter_resp[1]/(fs/2)], btype='bandpass')
    
    # 设计心率带通滤波器
    b_hr, a_hr = butter(2, [filter_hr[0]/(fs/2), filter_hr[1]/(fs/2)], btype='bandpass')
    
    # 应用滤波器
    resp_signal = filtfilt(b_resp, a_resp, displacement)
    hr_signal = filtfilt(b_hr, a_hr, displacement)
    
    return resp_signal, hr_signal

def find_target_range_bin(range_fft_data, distance, range_resolution):
    """
    根据实际距离找到最接近的距离bin索引
    
    参数:
        range_fft_data: 距离FFT结果
        distance: 目标实际距离(米)
        range_resolution: 距离分辨率(米/bin)
    
    返回:
        最接近目标距离的bin索引
    """
    # 计算理论上的bin索引
    bin_index = int(distance / range_resolution)
    
    # 确保索引在有效范围内
    bin_index = min(bin_index, range_fft_data.shape[3] // 2 - 1)
    bin_index = max(bin_index, 0)
    
    return bin_index

def generate_vitals_features(radar_data, distance, fs, window_size=600):
    """
    从雷达数据中生成生命体征特征
    
    参数:
        radar_data: 雷达原始数据，形状为(frames, antennas, chirps, samples)
        distance: 目标距离(米)
        fs: 帧率(Hz)
        window_size: 特征提取窗口大小(帧数)
    
    返回:
        features: 生命体征特征字典
    """
    import radar_settings
    import scipy.signal as signal
    
    # 获取雷达参数
    params = radar_settings.get_radar_params()
    range_resolution = params['range_resolution']
    wavelength = params['wavelength']
    
    # 执行距离FFT
    range_fft_result = range_fft(radar_data, window='hanning')
    
    # 找到目标距离bin
    target_bin = find_target_range_bin(range_fft_result, distance, range_resolution)
    
    # 提取相位信息
    phase_data = extract_phase(range_fft_result, target_bin)
    
    # 通常使用第一根天线，并对所有chirps平均
    phase = np.mean(phase_data[:, 0, :], axis=1)
    
    # 展开相位
    unwrapped_phase = unwrap_phase(phase)
    
    # 转换为位移
    displacement = phase_to_displacement(unwrapped_phase, wavelength)
    
    # 去趋势
    detrended = signal.detrend(displacement)
    
    # 分离呼吸和心率信号
    resp_signal, hr_signal = get_respiratory_heartrate(detrended, fs)
    
    # 特征提取
    features = {}
    
    # 使用滑动窗口进行分段
    step_size = window_size // 4  # 25%重叠
    segments = []
    
    for i in range(0, len(detrended) - window_size + 1, step_size):
        segment = {
            'displacement': detrended[i:i+window_size],
            'resp_signal': resp_signal[i:i+window_size],
            'hr_signal': hr_signal[i:i+window_size]
        }
        
        # 计算呼吸频谱
        resp_fft = np.abs(np.fft.rfft(segment['resp_signal']))
        resp_freq = np.fft.rfftfreq(window_size, 1/fs)
        resp_peak_idx = np.argmax(resp_fft)
        resp_rate = resp_freq[resp_peak_idx] * 60  # 转换为每分钟
        
        # 计算心率频谱
        hr_fft = np.abs(np.fft.rfft(segment['hr_signal']))
        hr_freq = np.fft.rfftfreq(window_size, 1/fs)
        hr_peak_idx = np.argmax(hr_fft)
        heart_rate = hr_freq[hr_peak_idx] * 60  # 转换为每分钟
        
        # 存储特征
        segment['resp_rate'] = resp_rate
        segment['heart_rate'] = heart_rate
        
        segments.append(segment)
    
    features['segments'] = segments
    
    return features

def extract_phase_edacm(radar_data, range_resolution, wavelength, verbose=False):
    """
    使用能量检测自适应CFAR方法提取目标相位。
    
    参数:
    radar_data: 2D numpy数组，形状为(num_frames, num_range_bins)，雷达距离谱数据
    range_resolution: float，距离分辨率（米/bin）
    wavelength: float，雷达波长（米）
    verbose: bool，是否打印目标信息，默认为False
    
    返回:
    phase_values: 1D numpy数组，目标的相位值
    target_bin: int，目标的距离bin索引
    """
    num_frames, num_range_bins = radar_data.shape
    
    # 设置最小和最大距离范围（米）
    min_range = 0.2  # 最小为0.2米
    max_range = 2.0  # 最大为2.0米
    
    # 转换为bin索引（确保不使用bin 0，它可能代表DC分量）
    min_bin = max(1, int(min_range / range_resolution))
    max_bin = min(num_range_bins - 1, int(max_range / range_resolution))
    
    # 计算距离范围内的平均功率
    if max_bin > min_bin:
        # 对整个时间序列计算每个距离bin的平均功率
        bin_power = np.mean(np.abs(radar_data[:, min_bin:max_bin+1])**2, axis=0)
        # 找到功率最强的bin作为目标bin
        target_bin = min_bin + np.argmax(bin_power)
    else:
        # 如果范围无效，则使用默认方法（整个距离范围内的最强回波）
        bin_power = np.mean(np.abs(radar_data)**2, axis=0)
        target_bin = np.argmax(bin_power)
    
    # 提取目标bin的时间序列
    target_signal = radar_data[:, target_bin]
    
    # 提取相位信息
    phase_values = np.angle(target_signal)
    
    # 只有当verbose为True时才打印
    if verbose:
        target_distance = target_bin * range_resolution
        print(f"检测到目标：距离bin = {target_bin}，距离 = {target_distance:.2f}米")
    
    return phase_values, target_bin 