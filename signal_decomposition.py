"""
信号分解算法库
包含多种常用的信号分解算法，用于生物信号分析、雷达信号处理等领域
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings

# =========== 基础工具函数 ===========

def check_signal(x):
    """检查输入信号是否为一维numpy数组"""
    if not isinstance(x, np.ndarray):
        try:
            x = np.array(x)
        except:
            raise ValueError("输入信号必须可转换为numpy数组")
    
    if x.ndim > 1:
        if x.shape[1] == 1:  # 如果是列向量
            x = x.flatten()
        else:
            raise ValueError("输入信号必须为一维数组")
    
    return x

def plot_decomposition(original, components, labels=None, figsize=(12, 8)):
    """绘制分解结果图"""
    n_components = len(components)
    
    plt.figure(figsize=figsize)
    # 绘制原始信号
    plt.subplot(n_components + 1, 1, 1)
    plt.plot(original)
    plt.title('原始信号')
    plt.grid(True)
    
    # 绘制分解后的各个分量
    for i, comp in enumerate(components):
        plt.subplot(n_components + 1, 1, i + 2)
        plt.plot(comp)
        if labels and i < len(labels):
            plt.title(labels[i])
        else:
            plt.title(f'分量 {i+1}')
        plt.grid(True)
    
    plt.tight_layout()
    return plt.gcf()


# =========== EMD 相关算法 ===========

def apply_emd(signal, max_imf=None):
    """
    应用经验模态分解(EMD)算法
    
    参数:
        signal: 一维信号数组
        max_imf: 最大IMF数量，None表示不限制
    
    返回:
        imfs: 包含所有IMF的数组，形状为(n_imfs, signal_length)
    
    示例:
        >>> imfs = apply_emd(signal)
        >>> plt.figure(figsize=(10, 6))
        >>> for i, imf in enumerate(imfs):
        >>>     plt.subplot(len(imfs), 1, i+1)
        >>>     plt.plot(imf)
        >>>     plt.title(f"IMF {i+1}")
    
    依赖:
        需要安装PyEMD库: pip install EMD-signal
    """
    try:
        from PyEMD import EMD
    except ImportError:
        raise ImportError("需要安装PyEMD库: pip install EMD-signal")
    
    signal = check_signal(signal)
    
    # 初始化EMD对象
    emd = EMD()
    
    # 执行分解
    imfs = emd(signal, max_imf=max_imf)
    
    return imfs

def apply_eemd(signal, noise_width=0.05, ensemble_size=100, max_imf=None):
    """
    应用集合经验模态分解(EEMD)算法，通过添加白噪声和多次平均提高EMD的稳定性
    
    参数:
        signal: 一维信号数组
        noise_width: 添加的噪声振幅相对于信号标准差的比例，默认0.05
        ensemble_size: 集合大小(运行次数)，默认100
        max_imf: 最大IMF数量，None表示不限制
    
    返回:
        imfs: 包含所有IMF的数组，形状为(n_imfs, signal_length)
    
    示例:
        >>> imfs = apply_eemd(signal, noise_width=0.1, ensemble_size=50)
        >>> plot_decomposition(signal, imfs, [f"EEMD IMF {i+1}" for i in range(len(imfs))])
    
    依赖:
        需要安装PyEMD库: pip install EMD-signal
    """
    try:
        from PyEMD import EEMD
    except ImportError:
        raise ImportError("需要安装PyEMD库: pip install EMD-signal")
    
    signal = check_signal(signal)
    
    # 初始化EEMD对象
    eemd = EEMD()
    eemd.noise_seed(12345)  # 设置随机种子以确保结果可重复
    
    # 设置参数
    eemd.noise_width = noise_width
    eemd.trials = ensemble_size
    
    # 执行分解
    imfs = eemd(signal, max_imf=max_imf)
    
    return imfs

def apply_ceemdan(signal, noise_width=0.05, ensemble_size=100, max_imf=None):
    """
    应用完全集合经验模态分解与自适应噪声(CEEMDAN)算法
    
    参数:
        signal: 一维信号数组
        noise_width: 添加的噪声振幅相对于信号标准差的比例，默认0.05
        ensemble_size: 集合大小(运行次数)，默认100
        max_imf: 最大IMF数量，None表示不限制
    
    返回:
        imfs: 包含所有IMF的数组，形状为(n_imfs, signal_length)
    
    示例:
        >>> imfs = apply_ceemdan(signal)
        >>> plot_decomposition(signal, imfs, [f"CEEMDAN IMF {i+1}" for i in range(len(imfs))])
    
    依赖:
        需要安装PyEMD库: pip install EMD-signal
    """
    try:
        from PyEMD import CEEMDAN
    except ImportError:
        raise ImportError("需要安装PyEMD库: pip install EMD-signal")
    
    signal = check_signal(signal)
    
    # 初始化CEEMDAN对象
    ceemdan = CEEMDAN()
    ceemdan.noise_seed(12345)  # 设置随机种子以确保结果可重复
    
    # 设置参数
    ceemdan.noise_width = noise_width
    ceemdan.trials = ensemble_size
    
    # 执行分解
    imfs = ceemdan(signal, max_imf=max_imf)
    
    return imfs


# =========== 小波相关算法 ===========

def apply_dwt(signal, wavelet='db4', level=None, mode='symmetric'):
    """
    应用离散小波变换(DWT)
    
    参数:
        signal: 一维信号数组
        wavelet: 小波类型，默认'db4'。可选类型包括: 'haar', 'db1-20', 'sym2-20', 'coif1-5'等
        level: 分解级别，默认为None(自动确定最大级别)
        mode: 边界处理模式，默认'symmetric'
    
    返回:
        coeffs: 小波系数列表[cA_n, cD_n, cD_n-1, ..., cD_1]，其中cA_n为最低频率近似系数，cD_i为细节系数
    
    示例:
        >>> coeffs = apply_dwt(signal, wavelet='db4', level=5)
        >>> # 重构信号
        >>> reconstructed = pywt.waverec(coeffs, 'db4')
    
    依赖:
        需要安装pywavelets库: pip install PyWavelets
    """
    try:
        import pywt
    except ImportError:
        raise ImportError("需要安装pywavelets库: pip install PyWavelets")
    
    signal = check_signal(signal)
    
    # 自动确定分解级别
    if level is None:
        level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
    
    # 执行小波分解
    coeffs = pywt.wavedec(signal, wavelet, mode=mode, level=level)
    
    return coeffs

def apply_swt(signal, wavelet='db4', level=None):
    """
    应用平稳小波变换(SWT)，解决了DWT平移不变性的问题
    
    参数:
        signal: 一维信号数组
        wavelet: 小波类型，默认'db4'
        level: 分解级别，默认为None(自动确定)
    
    返回:
        coeffs: 平稳小波系数列表，包含[cA_1, cD_1, cA_2, cD_2, ..., cA_n, cD_n]
    
    示例:
        >>> coeffs = apply_swt(signal, wavelet='db4', level=3)
        >>> # 提取每一级的近似和细节系数
        >>> cA3, cD3, cA2, cD2, cA1, cD1 = coeffs
    
    依赖:
        需要安装pywavelets库: pip install PyWavelets
    """
    try:
        import pywt
    except ImportError:
        raise ImportError("需要安装pywavelets库: pip install PyWavelets")
    
    signal = check_signal(signal)
    
    # 确保信号长度为2的幂次方，SWT要求
    padded_length = len(signal)
    if padded_length & (padded_length - 1) != 0:  # 如果不是2的幂次方
        next_power_of_2 = 2 ** (padded_length.bit_length())
        pad_width = next_power_of_2 - padded_length
        signal = np.pad(signal, (0, pad_width), mode='constant')
    
    # 自动确定分解级别
    if level is None:
        level = min(5, pywt.swt_max_level(len(signal)))
    
    # 执行平稳小波分解
    coeffs = pywt.swt(signal, wavelet, level=level)
    
    # 展平系数结构
    flattened_coeffs = []
    for ca, cd in coeffs:
        flattened_coeffs.extend([ca, cd])
    
    return flattened_coeffs

def apply_cwt(signal, scales=None, wavelet='morl', sampling_period=1.0, plot=False):
    """
    应用连续小波变换(CWT)
    
    参数:
        signal: 一维信号数组
        scales: 尺度参数，默认为None(自动生成)
        wavelet: 小波类型，默认'morl'(Morlet小波)
        sampling_period: 采样周期，默认为1.0
        plot: 是否绘制时频图，默认为False
    
    返回:
        coef: 小波系数数组，形状为(len(scales), len(signal))
        frequencies: 对应各尺度的频率
    
    示例:
        >>> coef, freqs = apply_cwt(signal, scales=np.arange(1, 128), wavelet='morl', plot=True)
    
    依赖:
        需要安装pywavelets库: pip install PyWavelets
    """
    try:
        import pywt
    except ImportError:
        raise ImportError("需要安装pywavelets库: pip install PyWavelets")
    
    signal = check_signal(signal)
    
    # 自动生成尺度数组
    if scales is None:
        width = min(len(signal) // 2, 256)
        scales = np.arange(1, width)
    
    # 执行连续小波变换
    coef, freqs = pywt.cwt(signal, scales, wavelet, sampling_period)
    
    # 绘制时频图
    if plot:
        plt.figure(figsize=(10, 8))
        
        # 绘制原始信号
        plt.subplot(211)
        plt.plot(np.arange(len(signal)) * sampling_period, signal)
        plt.title('原始信号')
        plt.grid(True)
        
        # 绘制小波变换结果
        plt.subplot(212)
        plt.imshow(np.abs(coef), aspect='auto', cmap='jet', 
                   extent=[0, len(signal) * sampling_period, freqs[-1], freqs[0]])
        plt.colorbar(label='幅度')
        plt.title('连续小波变换')
        plt.ylabel('频率(Hz)')
        plt.xlabel('时间(s)')
        plt.tight_layout()
        plt.show()
    
    return coef, freqs


# =========== VMD 算法 ===========

def apply_vmd(signal, alpha=2000, tau=0, K=3, DC=0, init=1, tol=1e-7, plot=False):
    """
    应用变分模态分解(VMD)
    
    参数:
        signal: 一维信号数组
        alpha: 带宽约束的惩罚参数，默认2000
        tau: 噪声容忍度，默认0
        K: 模态数量，默认3
        DC: 包含直流分量，默认0(不包含)
        init: 初始化方法，默认1(在频域中均匀分布)
        tol: 收敛容忍度，默认1e-7
        plot: 是否绘制分解结果，默认False
    
    返回:
        u: 分解的模态函数，形状为(K, len(signal))
        u_hat: 分解的频域模态函数
        omega: 各模态的中心频率
    
    示例:
        >>> u, u_hat, omega = apply_vmd(signal, K=5, plot=True)
    
    依赖:
        需要安装vmdpy库: pip install vmdpy
    """
    try:
        from vmdpy import VMD
    except ImportError:
        raise ImportError("需要安装vmdpy库: pip install vmdpy")
    
    signal = check_signal(signal)
    
    # 执行VMD分解
    u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)
    
    # 绘制分解结果
    if plot:
        t = np.arange(len(signal))
        
        plt.figure(figsize=(10, 6))
        
        # 绘制原始信号
        plt.subplot(K+1, 1, 1)
        plt.plot(t, signal)
        plt.title('原始信号')
        plt.grid(True)
        
        # 绘制各个模态
        for i in range(K):
            plt.subplot(K+1, 1, i+2)
            plt.plot(t, u[i, :])
            plt.title(f'模态 {i+1}, 中心频率: {omega[i, -1]:.2f}')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return u, u_hat, omega


# =========== 傅里叶相关算法 ===========

def apply_fft(signal, fs=1.0, plot=False):
    """
    应用快速傅里叶变换(FFT)
    
    参数:
        signal: 一维信号数组
        fs: 采样率，默认1.0
        plot: 是否绘制频谱图，默认False
    
    返回:
        freqs: 频率数组
        magnitude: 幅度谱
        phase: 相位谱
    
    示例:
        >>> freqs, magnitude, phase = apply_fft(signal, fs=100, plot=True)
    """
    signal = check_signal(signal)
    n = len(signal)
    
    # 计算FFT
    fft_result = np.fft.fft(signal)
    
    # 计算频率数组
    freqs = np.fft.fftfreq(n, 1/fs)
    
    # 计算幅度谱和相位谱
    magnitude = np.abs(fft_result) / n  # 归一化
    magnitude[1:-1] *= 2  # 由于频谱的对称性，非零频幅度翻倍
    phase = np.angle(fft_result)
    
    # 重新排序以便绘图
    idx = np.argsort(freqs)
    freqs = freqs[idx]
    magnitude = magnitude[idx]
    phase = phase[idx]
    
    # 绘制频谱图
    if plot:
        plt.figure(figsize=(12, 8))
        
        # 绘制原始信号
        plt.subplot(311)
        t = np.arange(n) / fs
        plt.plot(t, signal)
        plt.title('原始信号')
        plt.xlabel('时间 (s)')
        plt.grid(True)
        
        # 绘制幅度谱
        plt.subplot(312)
        plt.plot(freqs, magnitude)
        plt.title('幅度谱')
        plt.xlabel('频率 (Hz)')
        plt.ylabel('幅度')
        plt.grid(True)
        
        # 绘制相位谱
        plt.subplot(313)
        plt.plot(freqs, phase)
        plt.title('相位谱')
        plt.xlabel('频率 (Hz)')
        plt.ylabel('相位 (rad)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return freqs, magnitude, phase

def apply_stft(signal, fs=1.0, window='hann', nperseg=256, noverlap=None, plot=False):
    """
    应用短时傅里叶变换(STFT)
    
    参数:
        signal: 一维信号数组
        fs: 采样率，默认1.0
        window: 窗函数类型，默认'hann'
        nperseg: 每个分段的长度，默认256
        noverlap: 段之间的重叠点数，默认为None(nperseg//2)
        plot: 是否绘制频谱图，默认False
    
    返回:
        f: 频率数组
        t: 时间数组
        Zxx: STFT系数矩阵
    
    示例:
        >>> f, t, Zxx = apply_stft(signal, fs=1000, nperseg=512, plot=True)
    """
    signal = check_signal(signal)
    
    # 执行STFT
    f, t, Zxx = signal.stft(signal, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    
    # 绘制时频图
    if plot:
        plt.figure(figsize=(10, 8))
        
        # 绘制原始信号
        plt.subplot(211)
        time = np.arange(len(signal)) / fs
        plt.plot(time, signal)
        plt.title('原始信号')
        plt.xlabel('时间 (s)')
        plt.grid(True)
        
        # 绘制STFT频谱图
        plt.subplot(212)
        plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
        plt.title('STFT频谱图')
        plt.ylabel('频率 (Hz)')
        plt.xlabel('时间 (s)')
        plt.colorbar(label='幅度')
        
        plt.tight_layout()
        plt.show()
    
    return f, t, Zxx


# =========== PCA/ICA 算法 ===========

def apply_pca(data, n_components=None, whiten=False, plot=False):
    """
    应用主成分分析(PCA)
    
    参数:
        data: 数据矩阵，形状为(样本数, 特征数)
        n_components: 主成分数量，默认None表示保留所有成分
        whiten: 是否进行白化处理，默认False
        plot: 是否绘制主成分占比图，默认False
    
    返回:
        components: 主成分，形状为(n_components, 特征数)
        transformed_data: 转换后的数据，形状为(样本数, n_components)
        explained_variance_ratio: 各主成分解释的方差占比
    
    示例:
        >>> components, transformed_data, explained_var = apply_pca(data, n_components=2, plot=True)
    
    依赖:
        需要安装scikit-learn库: pip install scikit-learn
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        raise ImportError("需要安装scikit-learn库: pip install scikit-learn")
    
    if not isinstance(data, np.ndarray):
        try:
            data = np.array(data)
        except:
            raise ValueError("输入数据必须可转换为numpy数组")
    
    # 初始化PCA
    pca = PCA(n_components=n_components, whiten=whiten)
    
    # 执行PCA
    transformed_data = pca.fit_transform(data)
    
    # 绘制方差解释比例
    if plot:
        plt.figure(figsize=(10, 6))
        
        # 绘制方差解释比
        plt.subplot(211)
        plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
        plt.xlabel('主成分')
        plt.ylabel('解释方差比例')
        plt.title('各主成分解释方差比例')
        plt.grid(True)
        
        # 绘制累积方差解释比
        plt.subplot(212)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        plt.plot(range(1, len(cumulative) + 1), cumulative, 'o-')
        plt.xlabel('主成分数量')
        plt.ylabel('累积解释方差比例')
        plt.title('累积解释方差比例')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return pca.components_, transformed_data, pca.explained_variance_ratio_

def apply_ica(data, n_components=None, random_state=None, max_iter=200):
    """
    应用独立成分分析(ICA)，用于盲源分离
    
    参数:
        data: 数据矩阵，形状为(样本数, 特征数)
        n_components: 独立成分数量，默认None表示等于特征数
        random_state: 随机状态，默认None
        max_iter: 最大迭代次数，默认200
    
    返回:
        S: 分离的源信号
        A: 混合矩阵
        W: 分离矩阵
    
    示例:
        >>> S, A, W = apply_ica(mixed_signals, n_components=3)
        >>> # 绘制源信号
        >>> for i, sig in enumerate(S.T):
        >>>     plt.subplot(3, 1, i+1)
        >>>     plt.plot(sig)
        >>>     plt.title(f'源信号 {i+1}')
    
    依赖:
        需要安装scikit-learn库: pip install scikit-learn
    """
    try:
        from sklearn.decomposition import FastICA
    except ImportError:
        raise ImportError("需要安装scikit-learn库: pip install scikit-learn")
    
    if not isinstance(data, np.ndarray):
        try:
            data = np.array(data)
        except:
            raise ValueError("输入数据必须可转换为numpy数组")
    
    # 初始化ICA
    ica = FastICA(n_components=n_components, random_state=random_state, max_iter=max_iter)
    
    # 执行ICA
    S = ica.fit_transform(data)  # 估计源信号
    A = ica.mixing_  # 混合矩阵
    W = ica.components_  # 分离矩阵
    
    return S, A, W


# =========== SSA 算法 ===========

def apply_ssa(signal, window_length=None, n_components=None, plot=False):
    """
    应用奇异谱分析(SSA)
    
    参数:
        signal: 一维信号数组
        window_length: 窗口长度，默认为None(信号长度//2)
        n_components: 要提取的成分数量，默认为None(全部)
        plot: 是否绘制分解结果，默认False
    
    返回:
        components: 分解的奇异分量列表
        reconstructed_matrix: 重构矩阵
        singular_values: 奇异值
    
    示例:
        >>> comps, rec_matrix, s_values = apply_ssa(signal, window_length=50, n_components=5, plot=True)
    """
    signal = check_signal(signal)
    N = len(signal)
    
    # 设置窗口长度
    if window_length is None:
        window_length = N // 2
    if window_length >= N:
        window_length = N // 2
    
    # 构建轨迹矩阵
    K = N - window_length + 1
    trajectory_matrix = np.zeros((window_length, K))
    for i in range(K):
        trajectory_matrix[:, i] = signal[i:i+window_length]
    
    # 奇异值分解
    U, Sigma, Vt = np.linalg.svd(trajectory_matrix, full_matrices=False)
    
    # 确定成分数量
    if n_components is None:
        n_components = len(Sigma)
    
    # 重构各成分
    components = []
    reconstructed_matrix = np.zeros_like(trajectory_matrix)
    
    for i in range(min(n_components, len(Sigma))):
        component = U[:, i:i+1] @ np.diag(Sigma[i:i+1]) @ Vt[i:i+1, :]
        components.append(component)
        reconstructed_matrix += component
    
    # 对角线平均重构信号
    reconstructed_signals = []
    
    for component in components:
        reconst_signal = np.zeros(N)
        count = np.zeros(N)
        
        # 对角线平均
        for i in range(window_length):
            for j in range(K):
                idx = i + j
                if idx < N:
                    reconst_signal[idx] += component[i, j]
                    count[idx] += 1
        
        # 归一化
        reconst_signal /= count
        reconstructed_signals.append(reconst_signal)
    
    # 绘制结果
    if plot:
        plt.figure(figsize=(12, 8))
        
        # 绘制原始信号
        plt.subplot(n_components + 2, 1, 1)
        plt.plot(signal)
        plt.title('原始信号')
        plt.grid(True)
        
        # 绘制奇异值
        plt.subplot(n_components + 2, 1, 2)
        plt.bar(range(1, len(Sigma) + 1), Sigma)
        plt.xlabel('成分')
        plt.ylabel('奇异值')
        plt.title('奇异值分布')
        plt.grid(True)
        
        # 绘制各个成分
        for i, rec_signal in enumerate(reconstructed_signals[:n_components]):
            plt.subplot(n_components + 2, 1, i + 3)
            plt.plot(rec_signal)
            plt.title(f'成分 {i+1}')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return reconstructed_signals, reconstructed_matrix, Sigma


# =========== HHT (EMD + Hilbert变换) ===========

def apply_hht(signal, fs=1.0, plot=False):
    """
    应用希尔伯特-黄变换(HHT)，结合EMD和希尔伯特变换进行时频分析
    
    参数:
        signal: 一维信号数组
        fs: 采样率，默认1.0
        plot: 是否绘制瞬时频率图，默认False
    
    返回:
        imfs: EMD分解的IMF
        inst_freqs: 各IMF的瞬时频率
        inst_amps: 各IMF的瞬时幅度
    
    示例:
        >>> imfs, inst_freqs, inst_amps = apply_hht(signal, fs=100, plot=True)
    
    依赖:
        需要安装PyEMD库: pip install EMD-signal
    """
    try:
        from PyEMD import EMD
        from scipy.signal import hilbert
    except ImportError:
        raise ImportError("需要安装PyEMD和scipy库: pip install EMD-signal scipy")
    
    signal = check_signal(signal)
    N = len(signal)
    time = np.arange(N) / fs
    
    # 应用EMD
    emd = EMD()
    imfs = emd(signal)
    
    # 针对每个IMF进行希尔伯特变换
    inst_amps = []
    inst_freqs = []
    
    for imf in imfs:
        # 计算解析信号和瞬时幅度
        analytic_signal = hilbert(imf)
        amplitude_envelope = np.abs(analytic_signal)
        inst_amps.append(amplitude_envelope)
        
        # 计算瞬时相位和频率
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs
        # 添加一个值使长度与原信号一致
        instantaneous_frequency = np.append(instantaneous_frequency, instantaneous_frequency[-1])
        inst_freqs.append(instantaneous_frequency)
    
    # 绘制瞬时频率和幅度
    if plot:
        plt.figure(figsize=(14, 10))
        
        # 绘制原始信号
        plt.subplot(311)
        plt.plot(time, signal)
        plt.title('原始信号')
        plt.xlabel('时间 (s)')
        plt.grid(True)
        
        # 绘制IMF
        plt.subplot(312)
        for i, imf in enumerate(imfs):
            plt.plot(time, imf + i*2, label=f'IMF {i+1}')
        plt.title('IMF组件')
        plt.xlabel('时间 (s)')
        plt.grid(True)
        plt.legend(loc='upper right')
        
        # 绘制瞬时频率
        plt.subplot(313)
        for i, freq in enumerate(inst_freqs):
            plt.plot(time, freq, label=f'IMF {i+1}')
        plt.title('瞬时频率')
        plt.xlabel('时间 (s)')
        plt.ylabel('频率 (Hz)')
        plt.grid(True)
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    return imfs, inst_freqs, inst_amps


# =========== MP (匹配追踪) ===========

def apply_mp(signal, dictionary=None, n_atoms=10, n_iterations=100):
    """
    应用匹配追踪(MP)算法，使用原子字典分解信号
    
    参数:
        signal: 一维信号数组
        dictionary: 原子字典，默认为None(使用DCT基)
        n_atoms: 要提取的原子数量，默认10
        n_iterations: 最大迭代次数，默认100
    
    返回:
        reconstructed: 重构信号
        residual: 残差信号
        coefficients: 系数列表
        atoms: 选择的原子列表
    
    示例:
        >>> rec, res, coeffs, atoms = apply_mp(signal, n_atoms=20)
    
    注意:
        此实现使用简化的匹配追踪，完整实现可能需要专业库如PyMPTK
    """
    signal = check_signal(signal)
    N = len(signal)
    
    # 如果没有指定字典，使用DCT基
    if dictionary is None:
        # 创建DCT基
        dictionary = np.zeros((N, N))
        for k in range(N):
            if k == 0:
                norm_factor = np.sqrt(1/N)
            else:
                norm_factor = np.sqrt(2/N)
            dictionary[:, k] = norm_factor * np.cos(np.pi * k * (np.arange(N) + 0.5) / N)
    
    # 初始化
    residual = signal.copy()
    reconstructed = np.zeros_like(signal)
    coefficients = []
    atoms = []
    
    # MP算法主循环
    for _ in range(min(n_iterations, n_atoms)):
        # 计算残差与字典原子的内积
        correlations = np.abs(dictionary.T @ residual)
        
        # 找到最匹配的原子
        best_atom_idx = np.argmax(correlations)
        best_atom = dictionary[:, best_atom_idx]
        
        # 计算投影系数
        coefficient = np.dot(residual, best_atom)
        
        # 更新残差
        residual = residual - coefficient * best_atom
        
        # 更新重构信号
        reconstructed += coefficient * best_atom
        
        # 保存结果
        coefficients.append(coefficient)
        atoms.append(best_atom)
        
        # 检查收敛条件
        if np.linalg.norm(residual) < 1e-6:
            break
    
    return reconstructed, residual, coefficients, atoms


# =========== NMF (非负矩阵分解) ===========

def apply_nmf(data, n_components=2, max_iter=200, random_state=None, plot=False):
    """
    应用非负矩阵分解(NMF)
    
    参数:
        data: 非负数据矩阵，形状为(样本数, 特征数)
        n_components: 组件数量，默认2
        max_iter: 最大迭代次数，默认200
        random_state: 随机种子，默认None
        plot: 是否绘制分解结果，默认False
    
    返回:
        W: 权重矩阵，形状为(样本数, n_components)
        H: 特征矩阵，形状为(n_components, 特征数)
        reconstruction_err: 重构误差
    
    示例:
        >>> W, H, err = apply_nmf(data, n_components=3, plot=True)
    
    依赖:
        需要安装scikit-learn库: pip install scikit-learn
    """
    try:
        from sklearn.decomposition import NMF
    except ImportError:
        raise ImportError("需要安装scikit-learn库: pip install scikit-learn")
    
    if not isinstance(data, np.ndarray):
        try:
            data = np.array(data)
        except:
            raise ValueError("输入数据必须可转换为numpy数组")
    
    # 检查数据是否为非负
    if np.any(data < 0):
        warnings.warn("输入数据包含负值，将被替换为0")
        data = np.maximum(data, 0)
    
    # 初始化NMF
    nmf = NMF(n_components=n_components, max_iter=max_iter, random_state=random_state)
    
    # 执行NMF
    W = nmf.fit_transform(data)
    H = nmf.components_
    
    # 计算重构
    reconstruction = W @ H
    reconstruction_err = np.linalg.norm(data - reconstruction)
    
    # 绘制结果
    if plot and data.ndim == 2:
        plt.figure(figsize=(15, 5))
        
        # 绘制原始数据
        plt.subplot(131)
        plt.imshow(data, aspect='auto', cmap='viridis')
        plt.title('原始数据')
        plt.colorbar()
        
        # 绘制重构数据
        plt.subplot(132)
        plt.imshow(reconstruction, aspect='auto', cmap='viridis')
        plt.title(f'重构数据 (误差: {reconstruction_err:.4f})')
        plt.colorbar()
        
        # 绘制组件
        plt.subplot(133)
        for i in range(n_components):
            plt.plot(H[i], label=f'组件 {i+1}')
        plt.title('NMF组件')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return W, H, reconstruction_err


# =========== LMD (局部均值分解) ===========

def apply_lmd(signal, max_pf=10, max_iterations=10):
    """
    应用局部均值分解(LMD)
    
    参数:
        signal: 一维信号数组
        max_pf: 最大PF(乘积函数)数量，默认10
        max_iterations: 每个PF的最大迭代次数，默认10
    
    返回:
        pfs: 分解的PF组件列表
        residual: 残差信号
    
    示例:
        >>> pfs, residual = apply_lmd(signal, max_pf=5)
        >>> # 绘制PF组件
        >>> for i, pf in enumerate(pfs):
        >>>     plt.subplot(len(pfs), 1, i+1)
        >>>     plt.plot(pf)
        >>>     plt.title(f'PF {i+1}')
    
    注意:
        LMD实现较为复杂，此为简化版本，完整实现可能需要更多工程优化
    """
    signal = check_signal(signal)
    N = len(signal)
    
    # 初始化
    pfs = []
    residual = signal.copy()
    
    for _ in range(max_pf):
        if np.all(np.abs(residual) < 1e-10):
            break
        
        h = residual.copy()
        
        # 初始化PF提取
        a = np.ones_like(h)  # 包络
        prev_h = np.inf * np.ones_like(h)
        
        # 迭代多次以获取纯FM信号
        for _ in range(max_iterations):
            # 计算局部极大值和极小值点
            max_peaks, _ = signal.find_peaks(h)
            min_peaks, _ = signal.find_peaks(-h)
            
            if len(max_peaks) < 2 or len(min_peaks) < 2:
                break
            
            # 插值计算上下包络
            max_env = np.zeros_like(h)
            min_env = np.zeros_like(h)
            
            for i in range(N):
                if i in max_peaks:
                    max_env[i] = h[i]
                if i in min_peaks:
                    min_env[i] = h[i]
            
            # 平滑包络（简单移动平均）
            window_size = 5
            max_env = np.convolve(max_env, np.ones(window_size)/window_size, mode='same')
            min_env = np.convolve(min_env, np.ones(window_size)/window_size, mode='same')
            
            # 计算局部均值和包络
            m = (max_env + min_env) / 2
            a_new = (max_env - min_env) / 2
            
            # 更新h
            h_new = (h - m) / a_new
            a = a * a_new
            
            # 检查收敛性
            if np.linalg.norm(h - h_new) < 1e-6:
                break
            
            h = h_new
            
            # 防止无限循环
            if np.allclose(h, prev_h, rtol=1e-6):
                break
            prev_h = h.copy()
        
        # 计算PF组件
        pf = a * h
        pfs.append(pf)
        
        # 更新残差
        residual = residual - pf
    
    return pfs, residual


# =========== 应用示例 ===========

def example_vital_signs_decomposition(fs=100):
    """
    生物体征信号分解示例
    
    参数:
        fs: 采样频率，默认100Hz
    """
    # 生成模拟的生物体征信号（呼吸+心跳+噪声）
    t = np.arange(0, 20, 1/fs)
    
    # 呼吸信号：约0.2-0.3Hz
    respiratory = 0.8 * np.sin(2 * np.pi * 0.25 * t)
    
    # 心跳信号：约1-1.5Hz
    heartbeat = 0.3 * np.sin(2 * np.pi * 1.2 * t)
    
    # 添加高频噪声和漂移
    noise = 0.1 * np.random.randn(len(t))
    drift = 0.2 * np.sin(2 * np.pi * 0.05 * t)
    
    # 组合信号
    combined = respiratory + heartbeat + noise + drift
    
    print("=== 生物体征信号分解示例 ===")
    print(f"信号长度: {len(combined)}，采样率: {fs}Hz，持续时间: {len(combined)/fs}秒")
    
    # 绘制原始信号
    plt.figure(figsize=(12, 4))
    plt.plot(t, combined)
    plt.title("模拟的生物体征信号")
    plt.xlabel("时间 (秒)")
    plt.grid(True)
    plt.show()
    
    # 分别使用不同方法分解
    print("\n1. 使用EMD分解信号...")
    imfs = apply_emd(combined, plot=True)
    
    print("\n2. 使用EEMD分解信号...")
    eemd_imfs = apply_eemd(combined, noise_width=0.05, ensemble_size=10, plot=True)
    
    print("\n3. 使用VMD分解信号...")
    vmd_modes, _, _ = apply_vmd(combined, K=4, alpha=2000, plot=True)
    
    print("\n4. 使用小波变换分析信号...")
    coeffs = apply_dwt(combined, wavelet='db4', level=4, plot=True)
    
    print("\n5. 使用HHT分析信号的瞬时频率...")
    _, inst_freqs, inst_amps = apply_hht(combined, fs=fs, plot=True)
    
    return combined, imfs, vmd_modes


def example_radar_signal_decomposition(fs=1000):
    """
    雷达信号分解示例
    
    参数:
        fs: 采样频率，默认1000Hz
    """
    # 生成模拟的雷达信号，包含多个目标的回波和噪声
    t = np.arange(0, 5, 1/fs)
    
    # 目标1：静止目标
    target1 = 0.8 * np.ones_like(t)
    
    # 目标2：移动目标，产生多普勒频移
    doppler_freq = 30  # Hz
    target2 = 0.6 * np.sin(2 * np.pi * doppler_freq * t)
    
    # 目标3：周期性移动目标（例如呼吸）
    resp_freq = 0.3  # Hz
    target3 = 0.4 * np.sin(2 * np.pi * resp_freq * t)
    
    # 杂波和噪声
    clutter = 0.3 * np.sin(2 * np.pi * 0.1 * t)  # 慢变杂波
    noise = 0.2 * np.random.randn(len(t))  # 高斯噪声
    
    # 组合信号
    radar_signal = target1 + target2 + target3 + clutter + noise
    
    print("=== 雷达信号分解示例 ===")
    print(f"信号长度: {len(radar_signal)}，采样率: {fs}Hz，持续时间: {len(radar_signal)/fs}秒")
    
    # 绘制原始信号
    plt.figure(figsize=(12, 4))
    plt.plot(t, radar_signal)
    plt.title("模拟的雷达信号")
    plt.xlabel("时间 (秒)")
    plt.grid(True)
    plt.show()
    
    # 频域分析
    print("\n1. 使用FFT分析雷达信号频谱...")
    _, magnitude, frequency = apply_fft(radar_signal, fs=fs, plot=True)
    
    # 时频分析
    print("\n2. 使用STFT分析雷达信号的时频特性...")
    _, _, _, _ = apply_stft(radar_signal, fs=fs, nperseg=256, noverlap=128, plot=True)
    
    # 使用VMD分解
    print("\n3. 使用VMD分解雷达信号的不同组件...")
    vmd_modes, _, _ = apply_vmd(radar_signal, K=4, alpha=2000, plot=True)
    
    # 使用小波变换
    print("\n4. 使用连续小波变换分析雷达信号...")
    apply_cwt(radar_signal, scales=np.arange(1, 128), fs=fs, plot=True)
    
    return radar_signal, vmd_modes


if __name__ == "__main__":
    print("信号分解算法库示例")
    print("=" * 50)
    
    # 生物体征信号分解示例
    #example_vital_signs_decomposition()
    
    # 雷达信号分解示例
    #example_radar_signal_decomposition()
    
    # 简单测试示例
    print("\n基本测试:")
    
    # 生成测试信号：包含不同频率的正弦波
    fs = 1000  # 采样率
    t = np.arange(0, 1, 1/fs)
    
    # 信号组成: 10Hz + 50Hz + 100Hz + 噪声
    test_signal = np.sin(2*np.pi*10*t) + 0.5*np.sin(2*np.pi*50*t) + 0.3*np.sin(2*np.pi*100*t) + 0.1*np.random.randn(len(t))
    
    # 绘制测试信号
    plt.figure(figsize=(10, 4))
    plt.plot(t, test_signal)
    plt.title("测试信号")
    plt.xlabel("时间 (秒)")
    plt.grid(True)
    plt.show()
    
    # 使用VMD分解
    print("\nVMD分解测试信号...")
    modes, _, _ = apply_vmd(test_signal, K=3, plot=True)
    
    # 使用EMD分解
    print("\nEMD分解测试信号...")
    imfs = apply_emd(test_signal, plot=True)
    
    print("\n完成所有示例") 