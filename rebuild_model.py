"""
重新构建模型脚本

该脚本直接从 .h5 文件加载权重并重建模型，避免加载自定义函数的问题
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import h5py
import argparse

# 从radar_dl_models.py导入所有模型构建函数
from radar_dl_models import (
    create_cnn_gru_model,
    create_tcn_model,
    create_multi_input_model,
    create_mobilenet1d_model,
    create_lstm_model,
    create_bidirectional_lstm_model,
    create_cnn_lstm_model,
    create_attention_lstm_model,
    create_multiscale_lstm_model,
    create_transformer_model,
    create_resnet1d_model,
    create_tcan_model,
    create_inception_time_model,
    create_wavenet_model,
    create_nbeats_model,
    create_tft_model,
    create_omniscale_cnn_model,
    create_deep_state_space_model,
    create_lstnet_model
)

def get_model_architecture(model_name):
    """
    根据模型名称选择相应的构建函数
    
    参数:
        model_name: 模型名称
        
    返回:
        模型构建函数
    """
    model_builders = {
        'CNN_GRU': create_cnn_gru_model,
        'TCN': create_tcn_model,
        'Multi_Input': create_multi_input_model,
        'MobileNet1D': create_mobilenet1d_model,
        'LSTM': create_lstm_model,
        'BiLSTM': create_bidirectional_lstm_model,
        'CNN_LSTM': create_cnn_lstm_model,
        'AttentionLSTM': create_attention_lstm_model,
        'MultiscaleLSTM': create_multiscale_lstm_model,
        'Transformer': create_transformer_model,
        'ResNet1D': create_resnet1d_model,
        'TCAN': create_tcan_model,
        'InceptionTime': create_inception_time_model,
        'WaveNet': create_wavenet_model,
        'N-Beats': create_nbeats_model,
        'TFT': create_tft_model,
        'OmniScaleCNN': create_omniscale_cnn_model,
        'DeepStateSpace': create_deep_state_space_model,
        'LSTMNet': create_lstnet_model
    }
    
    return model_builders.get(model_name)

def extract_weights_from_h5(h5_file):
    """
    从.h5文件中提取权重
    
    参数:
        h5_file: .h5文件路径
        
    返回:
        权重字典 {层名称: 权重}
    """
    try:
        # 打开.h5文件
        with h5py.File(h5_file, 'r') as f:
            # 获取模型组
            model_weights = f['model_weights']
            
            # 提取所有层的权重
            weights_dict = {}
            for layer_name in model_weights.keys():
                if layer_name != 'keras_version' and layer_name != 'backend':
                    # 获取层权重组
                    layer_weights = model_weights[layer_name]
                    
                    # 检查是否有weights属性
                    if 'bias:0' in layer_weights or 'kernel:0' in layer_weights:
                        weights = []
                        
                        # 提取权重
                        for weight_name in ['kernel:0', 'bias:0']:
                            if weight_name in layer_weights:
                                weight_value = np.array(layer_weights[weight_name])
                                weights.append(weight_value)
                        
                        weights_dict[layer_name] = weights
            
            return weights_dict
    except Exception as e:
        print(f"提取权重失败: {e}")
        return None

def rebuild_model(model_path, output_path=None):
    """
    重建模型并加载权重
    
    参数:
        model_path: 原始模型路径(.h5文件)
        output_path: 新模型保存路径(可选)
        
    返回:
        重建的模型
    """
    # 解析模型名称
    model_file = os.path.basename(model_path)
    model_name_parts = model_file.split('_')
    
    if len(model_name_parts) < 2:
        print("无法解析模型名称，格式应为 '{模型名称}_{分解方法}_best.h5'")
        return None
    
    # 获取模型名称部分(可能包含下划线)
    decomp_method = model_name_parts[-2]  # 假设倒数第二部分是分解方法
    model_type = "_".join(model_name_parts[:-2])  # 模型名称可能包含多个部分
    
    print(f"解析的模型类型: {model_type}")
    print(f"解析的分解方法: {decomp_method}")
    
    # 获取模型构建函数
    model_builder = get_model_architecture(model_type)
    
    if model_builder is None:
        print(f"未找到模型 {model_type} 的构建函数")
        return None
    
    # 为模型构建创建默认输入形状
    input_shape = (300, 3)  # 假设形状是(时间步数，特征数)
    
    # 创建新模型
    print(f"使用默认参数重建 {model_type} 模型...")
    model = model_builder(input_shape)
    
    # 提取原始模型的权重
    print("从.h5文件提取权重...")
    weights_dict = extract_weights_from_h5(model_path)
    
    if weights_dict is None:
        print("无法提取权重，重建失败")
        return None
    
    # 设置权重
    print("将权重设置到新模型...")
    success = False
    
    try:
        # 方法1：逐层设置权重
        for layer in model.layers:
            if layer.name in weights_dict:
                layer_weights = weights_dict[layer.name]
                layer.set_weights(layer_weights)
                print(f"成功设置层 '{layer.name}' 的权重")
                success = True
        
        if success:
            print("模型权重已成功设置")
        else:
            print("警告: 没有设置任何层的权重")
    except Exception as e:
        print(f"设置权重失败: {e}")
        return None
    
    # 保存重建的模型(如果提供了输出路径)
    if output_path:
        print(f"保存重建的模型到 {output_path}...")
        try:
            model.save(output_path)
            print("模型保存成功")
        except Exception as e:
            print(f"保存模型失败: {e}")
    
    return model

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='重建模型并加载预训练权重')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='原始模型路径(.h5文件)')
    parser.add_argument('--output_path', type=str, default=None,
                       help='新模型保存路径(可选)')
    
    args = parser.parse_args()
    
    # 重建模型
    model = rebuild_model(args.model_path, args.output_path)
    
    if model is not None:
        print("模型重建成功")
        model.summary()
    else:
        print("模型重建失败")

if __name__ == "__main__":
    main() 