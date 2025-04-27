"""
雷达相位数据心率预测模型推理脚本

该脚本展示如何使用训练好的模型对新的雷达相位数据进行心率预测。
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import time
import argparse

# 导入信号分解模块
try:
    import signal_decomposition as sd
    signal_decomp_available = True
except ImportError:
    print("警告: 信号分解模块未安装，将仅使用原始相位数据")
    signal_decomp_available = False

# 参数设置
WINDOW_SIZE_SECONDS = 10  # 10秒窗口
FRAME_RATE = 30           # 帧率为30Hz
WINDOW_SIZE = int(WINDOW_SIZE_SECONDS * FRAME_RATE)  # 窗口大小（采样点数）
VMD_MODES = 3    # VMD分解模态数


def apply_eemd(phase_data):
    """
    对相位数据应用EEMD分解
    
    参数:
        phase_data: 形状为(window_size,)的相位数据
        
    返回:
        eemd_data: 形状为(modes, window_size)的EEMD分解结果
    """
    if not signal_decomp_available:
        raise ImportError("信号分解模块未安装，无法执行EEMD分解")
    
    # 初始化结果数组
    eemd_data = np.zeros((3, len(phase_data)))
    
    try:
        # 应用EEMD分解
        all_eimfs = sd.apply_eemd(phase_data, noise_width=0.05, ensemble_size=100)
        
        # 保存前3个IMF（或更少如果不足3个）
        for j in range(min(3, len(all_eimfs))):
            eemd_data[j, :] = all_eimfs[j]
    except Exception as e:
        print(f"EEMD分解失败: {e}")
        # 如果分解失败，返回零数组
        pass
    
    return eemd_data


def apply_vmd(phase_data):
    """
    对相位数据应用VMD分解
    
    参数:
        phase_data: 形状为(window_size,)的相位数据
        
    返回:
        vmd_data: 形状为(modes, window_size)的VMD分解结果
    """
    if not signal_decomp_available:
        raise ImportError("信号分解模块未安装，无法执行VMD分解")
    
    # 初始化结果数组
    vmd_data = np.zeros((VMD_MODES, len(phase_data)))
    
    try:
        # 应用VMD分解
        u, _, _ = sd.apply_vmd(phase_data, alpha=2000, K=VMD_MODES)
        
        # 保存所有模态
        for j in range(VMD_MODES):
            vmd_data[j, :] = u[j, :]
    except Exception as e:
        print(f"VMD分解失败: {e}")
        # 如果分解失败，返回零数组
        pass
    
    return vmd_data


def preprocess_data(phase_data, decomp_method='EEMD'):
    """
    预处理相位数据，应用信号分解
    
    参数:
        phase_data: 形状为(window_size,)的相位数据数组
        decomp_method: 要应用的分解方法，可选: 'EEMD', 'VMD'
        
    返回:
        decomp_data: 分解后的数据
    """
    # 确保输入数据长度正确
    if len(phase_data) != WINDOW_SIZE:
        raise ValueError(f"输入数据长度应为{WINDOW_SIZE}，实际为{len(phase_data)}")
    
    # 应用信号分解
    if decomp_method == 'EEMD':
        decomp_data = apply_eemd(phase_data)
    elif decomp_method == 'VMD':
        decomp_data = apply_vmd(phase_data)
    else:
        raise ValueError(f"不支持的分解方法: {decomp_method}")
    
    # 调整为模型期望的输入形状
    decomp_data = decomp_data.reshape(1, decomp_data.shape[0], decomp_data.shape[1])
    
    return decomp_data


def predict_heart_rate(model, phase_data, decomp_method='EEMD'):
    """
    使用模型预测心率
    
    参数:
        model: 训练好的模型
        phase_data: 形状为(window_size,)的相位数据数组
        decomp_method: 要应用的分解方法
        
    返回:
        predicted_hr: 预测的心率值
    """
    # 预处理数据
    decomp_data = preprocess_data(phase_data, decomp_method)
    
    # 使用模型进行预测
    start_time = time.time()
    predicted_hr = model.predict(decomp_data, verbose=0)[0][0]
    end_time = time.time()
    
    # 计算推理时间
    inference_time = end_time - start_time
    
    return predicted_hr, inference_time


def load_test_data(data_file):
    """
    加载测试数据
    
    参数:
        data_file: 包含相位数据的.npy文件路径
        
    返回:
        phase_data: 相位数据数组
    """
    try:
        # 加载相位数据
        if data_file.endswith('.npy'):
            data = np.load(data_file)
        else:
            raise ValueError("不支持的文件格式，请使用.npy文件")
        
        # 如果数据是多维的，提取相位数据
        if len(data.shape) > 1:
            print(f"输入数据形状: {data.shape}")
            # 假设相位数据在最后一维
            phase_data = data.flatten()[:WINDOW_SIZE]
        else:
            phase_data = data[:WINDOW_SIZE]
        
        # 确保数据长度正确
        if len(phase_data) < WINDOW_SIZE:
            raise ValueError(f"数据长度不足，需要{WINDOW_SIZE}个样本，实际只有{len(phase_data)}个")
        
        return phase_data
    except Exception as e:
        print(f"加载测试数据失败: {e}")
        return None


def run_real_time_simulation(model, data_file, decomp_method='EEMD', time_step=1.0):
    """
    模拟实时心率预测
    
    参数:
        model: 训练好的模型
        data_file: 包含相位数据的.npy文件路径
        decomp_method: 信号分解方法
        time_step: 模拟实时更新的时间步长（秒）
    """
    # 加载全部数据
    full_data = np.load(data_file)
    
    # 确保数据足够长
    if len(full_data) < WINDOW_SIZE + 100:  # 至少需要额外100帧来模拟实时流
        print(f"警告: 数据长度较短，可能无法有效模拟实时流")
    
    # 设置图形显示
    plt.figure(figsize=(12, 6))
    plt.ion()  # 打开交互模式
    
    # 初始化结果列表
    time_points = []
    heart_rates = []
    inference_times = []
    
    # 计算步长（帧数）
    step_frames = int(time_step * FRAME_RATE)
    
    # 模拟实时处理
    for i in range(0, len(full_data) - WINDOW_SIZE, step_frames):
        # 提取当前窗口的数据
        current_window = full_data[i:i+WINDOW_SIZE]
        
        # 进行预测
        hr, inf_time = predict_heart_rate(model, current_window, decomp_method)
        
        # 添加结果
        time_point = i / FRAME_RATE  # 转换为秒
        time_points.append(time_point)
        heart_rates.append(hr)
        inference_times.append(inf_time)
        
        # 显示结果
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(time_points, heart_rates, 'b-')
        plt.xlabel('时间 (秒)')
        plt.ylabel('心率 (BPM)')
        plt.title(f'实时心率监测 (使用 {decomp_method} 分解)')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(current_window, 'g-')
        plt.xlabel('帧')
        plt.ylabel('相位值')
        plt.title(f'当前窗口相位数据 (最新心率: {hr:.1f} BPM, 推理时间: {inf_time*1000:.1f} ms)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.pause(0.01)
        
        # 模拟实时处理速度
        time.sleep(time_step * 0.1)  # 加速模拟，实际时间的10%
    
    plt.ioff()
    
    # 最终结果
    plt.figure(figsize=(12, 8))
    
    # 心率预测
    plt.subplot(2, 1, 1)
    plt.plot(time_points, heart_rates, 'b-')
    plt.xlabel('时间 (秒)')
    plt.ylabel('心率 (BPM)')
    plt.title(f'心率监测结果 (使用 {decomp_method} 分解)')
    plt.grid(True)
    
    # 推理时间
    plt.subplot(2, 1, 2)
    plt.plot(time_points, [t*1000 for t in inference_times], 'r-')
    plt.xlabel('时间 (秒)')
    plt.ylabel('推理时间 (毫秒)')
    plt.title(f'模型推理时间 (平均: {np.mean(inference_times)*1000:.2f} ms)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'real_time_simulation_{decomp_method}.png')
    plt.show()
    
    # 输出统计信息
    print("\n模拟实时心率监测统计:")
    print(f"总样本数: {len(heart_rates)}")
    print(f"平均心率: {np.mean(heart_rates):.2f} BPM")
    print(f"心率范围: {np.min(heart_rates):.2f} - {np.max(heart_rates):.2f} BPM")
    print(f"平均推理时间: {np.mean(inference_times)*1000:.2f} ms")
    
    return time_points, heart_rates, inference_times


def main():
    parser = argparse.ArgumentParser(description='雷达相位数据心率预测')
    parser.add_argument('--model', type=str, required=True, help='训练好的模型路径')
    parser.add_argument('--data', type=str, required=True, help='测试数据文件路径')
    parser.add_argument('--decomp', type=str, default='EEMD', choices=['EEMD', 'VMD'], 
                        help='信号分解方法 (默认: EEMD)')
    parser.add_argument('--simulate', action='store_true', help='是否模拟实时心率监测')
    parser.add_argument('--step', type=float, default=1.0, help='实时模拟的时间步长 (秒)')
    
    args = parser.parse_args()
    
    # 加载模型
    try:
        model = load_model(args.model)
        print(f"模型已加载: {args.model}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 检查信号分解模块
    if not signal_decomp_available:
        print("错误: 信号分解模块未安装，无法进行信号分解")
        return
    
    # 检查数据文件
    if not os.path.exists(args.data):
        print(f"错误: 找不到数据文件 {args.data}")
        return
    
    # 运行模式
    if args.simulate:
        # 模拟实时心率监测
        print(f"\n开始模拟实时心率监测 (使用 {args.decomp} 分解)...")
        run_real_time_simulation(model, args.data, args.decomp, args.step)
    else:
        # 单次预测
        phase_data = load_test_data(args.data)
        if phase_data is not None:
            predicted_hr, inference_time = predict_heart_rate(model, phase_data, args.decomp)
            print(f"\n预测结果:")
            print(f"心率: {predicted_hr:.2f} BPM")
            print(f"推理时间: {inference_time*1000:.2f} ms")


if __name__ == "__main__":
    main() 