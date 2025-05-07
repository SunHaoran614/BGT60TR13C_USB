"""
雷达相位数据心率预测模型评估工具

该脚本是一个多功能评估工具，提供三种运行模式：
1. 单模型评估 (single): 详细评估单个模型性能
2. 批量评估 (batch): 批量评估多个模型并生成比较报告
3. 快速验证 (quick): 快速验证单个模型的基本性能

使用方法:
    python radar_model_evaluator.py --mode single --model_path trained_models/CNN_GRU_EEMD_best.h5
    python radar_model_evaluator.py --mode batch --models CNN_GRU BiLSTM
    python radar_model_evaluator.py --mode quick --model_path trained_models/CNN_GRU_EEMD_best.h5

特殊模型加载注意事项:
1. N-Beats模型问题：
   - 原错误信息: "arg 4 (defaults) must be None or tuple"
   - 原因: Lambda层中的Python函数序列化问题，尤其是默认参数的处理
   - 解决方法: 通过load_nbeats_model()函数重建模型并加载权重，然后保存为.keras格式

2. TCAN模型问题：
   - 原错误信息: "We could not automatically infer the shape of the Lambda's output"
   - 原因: Lambda层没有指定output_shape参数
   - 解决方法: 通过load_tcan_model()函数创建一个带有正确output_shape参数的新模型，加载权重
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import argparse
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Lambda, Subtract, Add

# 导入模型库函数
from radar_dl_models import (
    load_data,
    evaluate_model,
    create_tcan_model,
    create_nbeats_model,
    create_tcn_model
)

# 将N-Beats模型加载器的导入移到专用函数中
load_nbeats_model_external = None  # 全局变量，用于存储外部加载器

def get_nbeats_loader():
    """动态获取N-Beats模型加载器，只在需要时导入"""
    global load_nbeats_model_external
    
    if load_nbeats_model_external is None:
        try:
            from nbeats_model_loader import load_nbeats_model as external_loader
            load_nbeats_model_external = external_loader
            print("已导入外部N-Beats模型加载器")
        except ImportError:
            load_nbeats_model_external = None
    
    return load_nbeats_model_external

# 为LSTMNet模型定义的跳跃连接函数
def skip_connection_fn(x):
    """
    创建用于跳跃LSTM的输入序列
    
    参数:
        x: 输入张量，形状为 (batch_size, timesteps, features)
        
    返回:
        跳跃输入张量
    """
    # 获取原始形状
    batch_size = tf.shape(x)[0]
    timesteps = tf.shape(x)[1]
    features = tf.shape(x)[2]
    
    # 计算跳跃序列长度（固定为10，与LSTMNet模型中的默认值一致）
    skip_period = 10
    skip_seq_len = timesteps // skip_period
    
    # 创建索引
    indices = tf.range(skip_seq_len * skip_period, delta=skip_period)
    
    # 收集跳过的时间步
    skipped_steps = tf.gather(x, indices, axis=1)
    
    return skipped_steps

# 为N-Beats模型定义自定义层函数
def _nbeats_create_block(x, units, theta_dim, block_type='generic', block_id=0):
    """N-Beats模型中创建单个块的函数"""
    from tensorflow.keras.layers import Dense
    # 全连接层堆栈
    for i in range(4 - 1):  # 使用默认的basis_function_layers=4
        x = Dense(units, activation='relu', name=f'{block_type}_dense_{block_id}_{i}')(x)
    
    # 基函数参数（theta）
    theta = Dense(theta_dim, activation='linear', name=f'{block_type}_theta_{block_id}')(x)
    
    # 根据块类型构造回归结果和前向结果
    backcast = Dense(300, activation='linear', name=f'backcast_{block_type}_{block_id}')(theta)  # 假设时间步长为300
    forecast = Dense(1, activation='linear', name=f'forecast_{block_type}_{block_id}')(theta)    # 假设输出为1
    
    return backcast, forecast

# 为N-Beats模型定义自定义Lambda层函数
def create_zeros_like_timesteps(x, timesteps):
    """创建时间步的零张量"""
    return tf.zeros_like(x[:, 0:1, 0:1])[:, 0, 0:1] * 0.0 + tf.zeros((1, timesteps))

def create_zeros_for_forecast(x, output_units):
    """创建预测的零张量"""
    return tf.zeros_like(x[:, 0:1, 0:1])[:, 0, 0:1] * 0.0 + tf.zeros((1, output_units))

def extract_first_channel(x):
    """提取第一个通道"""
    return x[:, :, 0]

# 创建健壮的自定义对象映射，适应不同版本的TensorFlow
custom_objects = {
    'skip_connection_fn': skip_connection_fn,
    '_create_block': _nbeats_create_block,
    'create_zeros_like_timesteps': create_zeros_like_timesteps,
    'create_zeros_for_forecast': create_zeros_for_forecast,
    'extract_first_channel': extract_first_channel
}

# 尝试添加不同版本TensorFlow中可能存在的损失函数
try:
    # 尝试2.x版本风格
    custom_objects['mse'] = tf.keras.losses.MeanSquaredError()
    custom_objects['mae'] = tf.keras.losses.MeanAbsoluteError()
    custom_objects['mean_squared_error'] = tf.keras.losses.MeanSquaredError()
    custom_objects['mean_absolute_error'] = tf.keras.losses.MeanAbsoluteError()
    print("使用TensorFlow 2.x风格的损失函数")
except:
    pass

try:
    # 尝试函数式API
    from tensorflow.keras.losses import mse, mae
    custom_objects['mse'] = mse
    custom_objects['mae'] = mae
    custom_objects['mean_squared_error'] = mse
    custom_objects['mean_absolute_error'] = mae
    print("使用TensorFlow函数式损失函数")
except:
    pass

# 添加备用函数定义以防上述尝试都失败
if 'mse' not in custom_objects:
    def mse_func(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))
        
    def mae_func(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred))
        
    custom_objects['mse'] = mse_func
    custom_objects['mae'] = mae_func
    custom_objects['mean_squared_error'] = mse_func
    custom_objects['mean_absolute_error'] = mae_func
    print("使用手动定义的损失函数")

# 确保结果可重现
np.random.seed(42)
tf.random.set_seed(42)

# 默认配置参数
DEFAULT_DATA_PATH = 'training_data'
DEFAULT_MODELS_DIR = 'trained_models'
DEFAULT_RESULTS_DIR = 'results'
DEFAULT_DECOMP_METHOD = 'EEMD'

def load_test_data(data_path, decomp_method, use_original=False, test_size=0.2, random_state=42):
    """加载测试数据"""
    print(f"加载 {decomp_method} 分解测试数据...")
    
    if use_original:
        # 使用分解数据和原始数据
        (_, decomp_test,
         _, orig_test,
         _, y_test) = load_data(
            data_path, 
            decomp_method=decomp_method,
            use_original=True, 
            test_size=test_size,
            random_state=random_state
        )
        
        # 调整分解数据形状以符合Conv1D输入要求
        if len(decomp_test.shape) == 3:
            decomp_test = decomp_test.transpose(0, 2, 1)
        
        # 如果原始数据是2D的，调整为3D
        if len(orig_test.shape) == 2:
            orig_test = orig_test.reshape(orig_test.shape[0], orig_test.shape[1], 1)
        
        return decomp_test, orig_test, y_test
    else:
        # 只使用分解数据
        _, decomp_test, _, y_test = load_data(
            data_path, 
            decomp_method=decomp_method,
            use_original=False, 
            test_size=test_size,
            random_state=random_state
        )
        
        # 调整分解数据形状以符合Conv1D输入要求
        if len(decomp_test.shape) == 3:
            decomp_test = decomp_test.transpose(0, 2, 1)
        
        return decomp_test, None, y_test

def load_nbeats_model(model_path, input_shape=(300, 3)):
    """
    N-Beats模型加载适配器 - 通过重建模型并加载权重来解决Lambda层序列化问题
    
    参数:
        model_path: 模型文件路径
        input_shape: 输入数据形状，默认为(300, 3)
        
    返回:
        重建的N-Beats模型
    """
    print(f"尝试加载N-Beats模型: {model_path}")
    
    # 首先尝试使用外部加载器
    external_loader = get_nbeats_loader()
    if external_loader:
        try:
            model = external_loader(model_path)
            if model:
                print("✓ 使用外部加载器成功加载N-Beats模型!")
                return model
        except Exception as e:
            print(f"✗ 外部加载器加载失败: {e}")
    else:
        print("未找到外部N-Beats模型加载器，尝试使用内置加载逻辑...")
    
    # 如果外部加载器不可用或失败，使用内置逻辑
    try:
        # 创建新的N-Beats模型实例
        model = create_nbeats_model(input_shape=input_shape)
        
        # 尝试只加载权重，而不是整个模型
        try:
            model.load_weights(model_path)
            print("✓ 成功加载N-Beats模型权重!")
            
            # 保存为更兼容的.keras格式
            new_path = model_path.replace('.h5', '_rebuilt.keras')
            model.save(new_path, save_format='keras')
            print(f"✓ 重建的模型已保存到: {new_path}")
            
            return model
        except Exception as e:
            print(f"✗ 无法加载N-Beats模型权重: {e}")
            return None
    except Exception as e:
        print(f"✗ 无法创建N-Beats模型: {e}")
        return None

def load_tcn_model(model_path, input_shape=(300, 3)):
    """
    TCN模型加载适配器 - 通过重建模型并加载权重来解决Lambda层output_shape问题
    
    参数:
        model_path: 模型文件路径
        input_shape: 输入数据形状，默认为(300, 3)
        
    返回:
        重建的TCN模型
    """
    print(f"重建TCN模型并加载权重: {model_path}")
    
    try:
        # 创建新的TCN模型实例
        model = create_tcn_model(input_shape=input_shape)
        
        # 尝试只加载权重，而不是整个模型
        try:
            model.load_weights(model_path)
            print("✓ 成功加载TCN模型权重!")
            
            # 保存为更兼容的.keras格式
            new_path = model_path.replace('.h5', '_rebuilt.keras')
            model.save(new_path, save_format='keras')
            print(f"✓ 重建的模型已保存到: {new_path}")
            
            return model
        except Exception as e:
            print(f"✗ 无法加载TCN模型权重: {e}")
            return None
    except Exception as e:
        print(f"✗ 无法创建TCN模型: {e}")
        return None

def load_deepstatespace_model(model_path, input_shape=(300, 3)):
    """
    DeepStateSpace模型加载适配器 - 解决Lambda层序列化问题
    
    参数:
        model_path: 模型文件路径
        input_shape: 输入数据形状，默认为(300, 3)
        
    返回:
        重建的DeepStateSpace模型
    """
    print(f"尝试加载DeepStateSpace模型: {model_path}")
    
    try:
        # 允许不安全反序列化
        tf.keras.config.enable_unsafe_deserialization()
        
        # 加载模型
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # 重新编译模型
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        print("✓ 成功加载DeepStateSpace模型!")
        return model
    except Exception as e:
        print(f"✗ 无法加载DeepStateSpace模型: {e}")
        
        # 如果加载失败，尝试使用tf.saved_model.load
        try:
            print("尝试使用tf.saved_model.load加载...")
            model_dir = os.path.dirname(model_path)
            model_name_base = os.path.basename(model_path).replace('.keras', '')
            saved_model_path = os.path.join(model_dir, model_name_base)
            
            if os.path.exists(saved_model_path):
                model = tf.saved_model.load(saved_model_path)
                print(f"✓ 成功使用SavedModel加载!")
                return model
            else:
                print(f"✗ SavedModel目录不存在: {saved_model_path}")
                return None
        except Exception as e2:
            print(f"✗ 使用SavedModel加载失败: {e2}")
            return None

def load_tft_model(model_path, input_shape=(300, 3)):
    """
    TFT模型加载适配器 - 通过重建模型并加载权重来解决Lambda层output_shape问题
    
    参数:
        model_path: 模型文件路径
        input_shape: 输入数据形状，默认为(300, 3)
        
    返回:
        重建的TFT模型
    """
    from radar_dl_models import create_tft_model
    
    print(f"重建TFT模型并加载权重: {model_path}")
    
    try:
        # 创建新的TFT模型实例
        model = create_tft_model(input_shape=input_shape)
        
        # 尝试只加载权重，而不是整个模型
        try:
            model.load_weights(model_path)
            print("✓ 成功加载TFT模型权重!")
            
            # 保存为更兼容的.keras格式
            new_path = model_path.replace('.h5', '_rebuilt.keras')
            model.save(new_path, save_format='keras')
            print(f"✓ 重建的模型已保存到: {new_path}")
            
            return model
        except Exception as e:
            print(f"✗ 无法加载TFT模型权重: {e}")
            return None
    except Exception as e:
        print(f"✗ 无法创建TFT模型: {e}")
        return None

def load_transformer_model(model_path, input_shape=(300, 3)):
    """
    Transformer模型加载适配器 - 通过重建模型并加载权重来解决Lambda层position_encoding_fn问题
    
    参数:
        model_path: 模型文件路径
        input_shape: 输入数据形状，默认为(300, 3)
        
    返回:
        重建的Transformer模型
    """
    from radar_dl_models import create_transformer_model
    
    print(f"重建Transformer模型并加载权重: {model_path}")
    
    try:
        # 创建新的Transformer模型实例
        model = create_transformer_model(input_shape=input_shape)
        
        # 尝试只加载权重，而不是整个模型
        try:
            model.load_weights(model_path)
            print("✓ 成功加载Transformer模型权重!")
            
            # 保存为更兼容的.keras格式
            new_path = model_path.replace('.h5', '_rebuilt.keras')
            model.save(new_path, save_format='keras')
            print(f"✓ 重建的模型已保存到: {new_path}")
            
            return model
        except Exception as e:
            print(f"加载权重失败: {e}")
            raise
    except Exception as e:
        print(f"创建Transformer模型失败: {e}")
        raise

def evaluate_single_model(model_path, data_path, decomp_method, results_dir, 
                         use_original=False, plot=True, measure_time=True,
                         model_name=None, test_size=0.2, random_state=42):
    """评估单个模型并生成详细报告"""
    # 确保结果目录存在
    os.makedirs(results_dir, exist_ok=True)
    
    # 如果没有提供模型名称，从文件名提取
    if model_name is None:
        model_name = os.path.basename(model_path).replace('.h5', '').replace('.keras', '')
    
    print(f"评估模型: {model_name}")
    
    # 从模型名称中检测应该使用的分解方法
    if 'VMD' in model_name and decomp_method != 'VMD':
        print(f"⚠️ 警告: 模型名称中包含VMD，但指定的分解方法是{decomp_method}。建议使用VMD分解方法。")
        print(f"正在将分解方法从{decomp_method}切换到VMD...")
        decomp_method = 'VMD'
    elif 'EEMD' in model_name and decomp_method != 'EEMD':
        print(f"⚠️ 警告: 模型名称中包含EEMD，但指定的分解方法是{decomp_method}。建议使用EEMD分解方法。")
        print(f"正在将分解方法从{decomp_method}切换到EEMD...")
        decomp_method = 'EEMD'
    elif 'EMD' in model_name and decomp_method != 'EMD' and 'EEMD' not in model_name:
        print(f"⚠️ 警告: 模型名称中包含EMD，但指定的分解方法是{decomp_method}。建议使用EMD分解方法。")
        print(f"正在将分解方法从{decomp_method}切换到EMD...")
        decomp_method = 'EMD'
    elif 'DWT' in model_name and decomp_method != 'DWT':
        print(f"⚠️ 警告: 模型名称中包含DWT，但指定的分解方法是{decomp_method}。建议使用DWT分解方法。")
        print(f"正在将分解方法从{decomp_method}切换到DWT...")
        decomp_method = 'DWT'
        
    print(f"使用分解方法: {decomp_method}")
    
    # 加载测试数据
    print(f"加载测试数据...")
    try:
        if use_original:
            decomp_test, orig_test, y_test = load_test_data(
                data_path, decomp_method, True, test_size, random_state
            )
            X_test = [decomp_test, orig_test]  # 多输入模型
        else:
            decomp_test, _, y_test = load_test_data(
                data_path, decomp_method, False, test_size, random_state
            )
            X_test = decomp_test
            
        print(f"测试集形状: {decomp_test.shape if isinstance(X_test, np.ndarray) else [x.shape for x in X_test]}")
        print(f"测试标签形状: {y_test.shape}")
        
        # 获取输入形状，用于可能的模型重建
        if isinstance(X_test, np.ndarray):
            input_shape = X_test.shape[1:]
        else:
            input_shape = X_test[0].shape[1:]
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None
    
    # 加载模型
    model = None
    
    # 检查是否为特殊模型并尝试使用对应的适配器
    if 'TCAN' in model_path:
        print("检测到TCAN模型，尝试使用专用适配器...")
        model = load_tcan_model(model_path, input_shape)
    elif 'TCN' in model_path:
        print("检测到TCN模型，尝试使用专用适配器...")
        model = load_tcn_model(model_path, input_shape)
    elif 'N-Beats' in model_path or 'N_Beats' in model_path or 'NBeats' in model_path or 'N-beats' in model_path:
        print("检测到N-Beats模型，使用专用适配器...")
        model = load_nbeats_model(model_path, input_shape)
    elif 'TFT' in model_path:
        print("检测到TFT模型，尝试使用专用适配器...")
        model = load_tft_model(model_path, input_shape)
    elif 'Transformer' in model_path:
        print("检测到Transformer模型，尝试使用专用适配器...")
        model = load_transformer_model(model_path, input_shape)
    elif 'DeepStateSpace' in model_path:
        print("检测到DeepStateSpace模型，尝试使用专用适配器...")
        model = load_deepstatespace_model(model_path, input_shape)
    
    # 如果特殊适配器失败或不是特殊模型，尝试标准加载方法
    if model is None:
        try:
            # 尝试方法1：使用自定义对象映射加载
            print("尝试使用自定义对象加载模型...")
            model = load_model(model_path, custom_objects=custom_objects)
            print("模型加载成功！")
        except Exception as e:
            print(f"方法1加载失败: {e}")
            
            try:
                # 尝试方法2：不使用编译配置加载
                print("尝试方法2：不使用编译配置加载模型...")
                model = tf.keras.models.load_model(
                    model_path,
                    compile=False
                )
                # 手动编译模型
                model.compile(
                    optimizer='adam',
                    loss='mse',
                    metrics=['mae']
                )
                print(f"成功加载模型（方法2）")
            except Exception as e2:
                print(f"方法2加载失败: {e2}")
                
                try:
                    # 尝试方法3：使用tf.saved_model.load
                    print("尝试方法3：使用tf.saved_model.load加载...")
                    model_dir = os.path.dirname(model_path)
                    model_name_base = os.path.basename(model_path).replace('.h5', '')
                    saved_model_path = os.path.join(model_dir, model_name_base)
                    
                    if os.path.exists(saved_model_path):
                        model = tf.saved_model.load(saved_model_path)
                        print(f"成功加载模型（方法3）")
                    else:
                        raise FileNotFoundError(f"SavedModel目录不存在: {saved_model_path}")
                except Exception as e3:
                    print(f"方法3加载失败: {e3}")
                    print("所有加载方法都失败，无法评估模型")
                    return None
    
    # 评估模型
    print("开始评估模型...")
    metrics = evaluate_model(
        model, X_test, y_test, 
        plot_results=plot, 
        model_name=model_name,
        measure_inference_time=measure_time
    )
    
    # 打印评估结果
    print("\n===== 评估结果 =====")
    print(f"模型: {model_name}")
    print(f"MAE: {metrics['mae']:.4f} BPM")
    print(f"RMSE: {metrics['rmse']:.4f} BPM")
    print(f"R²: {metrics['r2']:.4f}")
    print(f"±3 BPM准确率: {metrics['accuracy_3bpm']:.2f}%")
    
    if 'inference_time_ms' in metrics:
        print(f"单样本推理时间: {metrics['inference_time_ms']:.2f} 毫秒")
        print(f"每秒处理样本数: {metrics['samples_per_second']:.2f}")
    
    # 保存评估结果到文件
    result_file = os.path.join(results_dir, f'{model_name}_evaluation.txt')
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"模型: {model_name}\n")
        f.write(f"分解方法: {decomp_method}\n")
        f.write(f"MAE: {metrics['mae']:.4f} BPM\n")
        f.write(f"RMSE: {metrics['rmse']:.4f} BPM\n")
        f.write(f"R²: {metrics['r2']:.4f}\n")
        f.write(f"MAPE: {metrics['mape']:.2f}%\n")
        f.write(f"最大误差: {metrics['max_error']:.4f} BPM\n")
        f.write(f"中位误差: {metrics['median_error']:.4f} BPM\n")
        f.write(f"误差标准差: {metrics['error_std']:.4f}\n")
        f.write(f"95%置信区间: [{metrics['ci_95_lower']:.2f}, {metrics['ci_95_upper']:.2f}] BPM\n")
        f.write(f"±3 BPM准确率: {metrics['accuracy_3bpm']:.2f}%\n")
        
        # 不同心率区间的性能
        if not np.isnan(metrics.get('mae_low_hr', np.nan)):
            f.write(f"\n不同心率区间性能:\n")
            f.write(f"低心率(<70 BPM) MAE: {metrics['mae_low_hr']:.4f} BPM\n")
            f.write(f"正常心率(70-90 BPM) MAE: {metrics['mae_normal_hr']:.4f} BPM\n")
            f.write(f"高心率(>90 BPM) MAE: {metrics['mae_high_hr']:.4f} BPM\n")
        
        # 推理性能
        if 'inference_time_ms' in metrics:
            f.write(f"\n推理性能:\n")
            f.write(f"单样本推理时间: {metrics['inference_time_ms']:.2f} 毫秒\n")
            f.write(f"批量推理时间(整个测试集): {metrics['batch_inference_time']:.4f} 秒\n")
            f.write(f"每秒处理样本数: {metrics['samples_per_second']:.2f}\n")
    
    print(f"\n评估结果已保存到: {result_file}")
    return metrics

def evaluate_batch_models(models_dir, data_path, decomp_method, results_dir, 
                         models_to_evaluate=None, use_original=False,
                         plot_individual=True, test_size=0.2, random_state=42):
    """批量评估多个模型并生成比较报告"""
    # 确保目录存在
    os.makedirs(results_dir, exist_ok=True)
    
    # 加载测试数据
    if use_original:
        decomp_test, orig_test, y_test = load_test_data(
            data_path, decomp_method, use_original=True, 
            test_size=test_size, random_state=random_state
        )
    else:
        decomp_test, _, y_test = load_test_data(
            data_path, decomp_method, use_original=False,
            test_size=test_size, random_state=random_state
        )
    
    # 存储评估结果
    results = {}
    
    # 获取所有模型文件
    model_files = [f for f in os.listdir(models_dir) if f.endswith('_best.h5')]
    
    # 如果指定了要评估的模型，则筛选
    if models_to_evaluate:
        model_files = [f for f in model_files if any(model in f for model in models_to_evaluate)]
    
    # 检查是否有模型可评估
    if not model_files:
        print(f"在 {models_dir} 目录中未找到符合条件的模型文件。")
        return None
    
    print(f"\n找到 {len(model_files)} 个模型进行评估:")
    for i, model_file in enumerate(model_files):
        print(f"  {i+1}. {model_file}")
    
    # 遍历每个模型进行评估
    for model_file in model_files:
        # 提取模型名称
        model_name = model_file.replace(f'_{decomp_method}_best.h5', '')
        print(f"\n评估模型: {model_name}")
        print("-" * 30)
        
        # 加载模型
        model_path = os.path.join(models_dir, model_file)
        model = None
        
        # 尝试不同的加载方法
        try:
            # 方法1：使用自定义对象映射
            print("尝试方法1：使用自定义对象映射加载模型...")
            model = load_model(model_path, custom_objects=custom_objects)
            print(f"成功加载模型（方法1）")
        except Exception as e:
            print(f"方法1加载失败: {e}")
            
            try:
                # 方法2：不使用编译配置加载
                print("尝试方法2：不使用编译配置加载模型...")
                model = tf.keras.models.load_model(
                    model_path,
                    compile=False
                )
                # 手动编译模型
                model.compile(
                    optimizer='adam',
                    loss='mse',
                    metrics=['mae']
                )
                print(f"成功加载模型（方法2）")
            except Exception as e2:
                print(f"方法2加载失败: {e2}")
                
                try:
                    # 方法3：使用tf.saved_model.load
                    print("尝试方法3：使用tf.saved_model.load加载...")
                    model_dir = os.path.dirname(model_path)
                    model_name_base = os.path.basename(model_path).replace('.h5', '')
                    saved_model_path = os.path.join(model_dir, model_name_base)
                    
                    if os.path.exists(saved_model_path):
                        model = tf.saved_model.load(saved_model_path)
                        print(f"成功加载模型（方法3）")
                    else:
                        raise FileNotFoundError(f"SavedModel目录不存在: {saved_model_path}")
                except Exception as e3:
                    print(f"方法3加载失败: {e3}")
                    print(f"所有加载方法都失败，跳过评估模型: {model_name}")
                    continue
        
        # 根据模型类型选择输入数据
        if model_name == 'Multi_Input' and use_original:
            X_test = [decomp_test, orig_test]
        else:
            X_test = decomp_test
        
        # 评估模型
        metrics = evaluate_model(
            model, X_test, y_test,
            plot_results=plot_individual,
            model_name=f"{model_name} with {decomp_method}"
        )
        
        # 保存结果
        results[model_name] = metrics
    
    # 创建性能比较表格
    if not results:
        print("没有成功评估任何模型。")
        return None
    
    model_names = list(results.keys())
    
    # 创建完整比较表格
    comparison_data = {
        '模型': model_names,
        'MAE (BPM)': [results[model]['mae'] for model in model_names],
        'RMSE (BPM)': [results[model]['rmse'] for model in model_names],
        'R²': [results[model]['r2'] for model in model_names],
        'MAPE (%)': [results[model]['mape'] for model in model_names],
        '最大误差 (BPM)': [results[model]['max_error'] for model in model_names],
        '中位误差 (BPM)': [results[model]['median_error'] for model in model_names],
        '±3BPM准确率 (%)': [results[model]['accuracy_3bpm'] for model in model_names]
    }
    
    # 添加推理时间相关指标（如果存在）
    if 'inference_time_ms' in results[model_names[0]]:
        comparison_data['单样本推理时间 (ms)'] = [results[model]['inference_time_ms'] for model in model_names]
        comparison_data['样本/秒'] = [results[model]['samples_per_second'] for model in model_names]
    
    results_df = pd.DataFrame(comparison_data)
    
    # 按MAE排序
    results_df = results_df.sort_values('MAE (BPM)')
    
    # 获取时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    # 保存结果到CSV
    csv_path = os.path.join(results_dir, f'model_comparison_{decomp_method}_{timestamp}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n评估结果已保存到: {csv_path}")
    
    # 打印结果表格
    print("\n模型性能比较:")
    print("=" * 60)
    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.width', 1000)  # 加宽显示
    print(results_df.to_string(index=False))
    print("=" * 60)
    
    # 绘制性能比较图
    plt.figure(figsize=(14, 10))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # MAE比较
    plt.subplot(2, 2, 1)
    bars = plt.bar(model_names, results_df['MAE (BPM)'].values, color='skyblue')
    plt.ylabel('MAE (BPM)')
    plt.title('不同模型的平均绝对误差 (MAE)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom')
    
    # RMSE比较
    plt.subplot(2, 2, 2)
    bars = plt.bar(model_names, results_df['RMSE (BPM)'].values, color='lightgreen')
    plt.ylabel('RMSE (BPM)')
    plt.title('不同模型的均方根误差 (RMSE)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom')
    
    # R²比较
    plt.subplot(2, 2, 3)
    bars = plt.bar(model_names, results_df['R²'].values, color='salmon')
    plt.ylabel('R²')
    plt.title('不同模型的决定系数 (R²)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')
    
    # ±3BPM准确率比较
    plt.subplot(2, 2, 4)
    bars = plt.bar(model_names, results_df['±3BPM准确率 (%)'].values, color='plum')
    plt.ylabel('准确率 (%)')
    plt.title('不同模型的±3BPM准确率')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存图表
    chart_path = os.path.join(results_dir, f'model_comparison_{decomp_method}_{timestamp}.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"比较图表已保存到: {chart_path}")
    
    # 如果有推理时间指标，生成性能图表
    if 'inference_time_ms' in results[model_names[0]]:
        plt.figure(figsize=(12, 6))
        
        # 精度vs速度散点图
        plt.subplot(1, 2, 1)
        plt.scatter(
            [results[model]['inference_time_ms'] for model in model_names],
            [results[model]['mae'] for model in model_names],
            alpha=0.7, s=100
        )
        
        # 添加模型名称标签
        for i, model in enumerate(model_names):
            plt.annotate(
                model, 
                (results[model]['inference_time_ms'], results[model]['mae']),
                xytext=(5, 5), textcoords='offset points'
            )
        
        plt.xlabel('推理时间 (毫秒/样本)')
        plt.ylabel('MAE (BPM)')
        plt.title('模型精度与速度权衡')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 推理时间柱状图
        plt.subplot(1, 2, 2)
        inference_times = [results[model]['inference_time_ms'] for model in model_names]
        bars = plt.bar(model_names, inference_times, color='lightblue')
        plt.ylabel('时间 (毫秒)')
        plt.title('单样本推理时间')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f'{height:.2f}ms', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存性能图表
        perf_chart_path = os.path.join(results_dir, f'model_performance_{decomp_method}_{timestamp}.png')
        plt.savefig(perf_chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"性能图表已保存到: {perf_chart_path}")
    
    return results_df

def quick_evaluate_model(model_path, data_path, decomp_method, use_original=False):
    """快速评估模型，仅返回基本性能指标，不保存图表或详细结果"""
    # 提取模型名称
    model_name = os.path.basename(model_path).replace('.h5', '')
    print(f"快速评估模型: {model_name}")
    
    # 加载模型
    model = None
    
    # 尝试不同的加载方法
    try:
        # 方法1：使用自定义对象映射
        print("尝试方法1：使用自定义对象映射加载模型...")
        model = load_model(model_path, custom_objects=custom_objects)
        print(f"成功加载模型（方法1）")
    except Exception as e:
        print(f"方法1加载失败: {e}")
        
        try:
            # 方法2：不使用编译配置加载
            print("尝试方法2：不使用编译配置加载模型...")
            model = tf.keras.models.load_model(
                model_path,
                compile=False
            )
            # 手动编译模型
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            print(f"成功加载模型（方法2）")
        except Exception as e2:
            print(f"方法2加载失败: {e2}")
            
            try:
                # 方法3：使用tf.saved_model.load
                print("尝试方法3：使用tf.saved_model.load加载...")
                model_dir = os.path.dirname(model_path)
                model_name_base = os.path.basename(model_path).replace('.h5', '')
                saved_model_path = os.path.join(model_dir, model_name_base)
                
                if os.path.exists(saved_model_path):
                    model = tf.saved_model.load(saved_model_path)
                    print(f"成功加载模型（方法3）")
                else:
                    raise FileNotFoundError(f"SavedModel目录不存在: {saved_model_path}")
            except Exception as e3:
                print(f"方法3加载失败: {e3}")
                print("所有加载方法都失败，无法评估模型")
                return None
    
    # 加载少量测试数据
    try:
        if use_original:
            decomp_test, orig_test, y_test = load_test_data(
                data_path, decomp_method, True, test_size=0.1
            )
            X_test = [decomp_test, orig_test]
        else:
            decomp_test, _, y_test = load_test_data(
                data_path, decomp_method, False, test_size=0.1
            )
            X_test = decomp_test
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None
    
    # 开始计时
    start_time = time.time()
    
    # 模型预测
    y_pred = model.predict(X_test, verbose=0)
    
    # 结束计时
    end_time = time.time()
    inference_time = end_time - start_time
    
    # 计算基础指标
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # 计算±3 BPM准确率
    abs_errors = np.abs(y_pred.flatten() - y_test)
    accuracy_3bpm = np.mean(abs_errors <= 3.0) * 100
    
    # 打印结果
    print("\n===== 快速评估结果 =====")
    print(f"MAE: {mae:.4f} BPM")
    print(f"RMSE: {rmse:.4f} BPM")
    print(f"R²: {r2:.4f}")
    print(f"±3 BPM准确率: {accuracy_3bpm:.2f}%")
    print(f"测试集样本数: {len(y_test)}")
    print(f"总推理时间: {inference_time:.4f}秒")
    print(f"平均每样本: {(inference_time*1000)/len(y_test):.2f}毫秒")
    
    results = {
        'model_name': model_name,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'accuracy_3bpm': accuracy_3bpm,
        'inference_time': inference_time,
        'samples': len(y_test),
        'ms_per_sample': (inference_time*1000)/len(y_test)
    }
    
    return results

# 为TCAN模型定义自定义加载适配器
def load_tcan_model(model_path, input_shape=(300, 3)):
    """
    TCAN模型加载适配器 - 根据保存的权重重建模型
    
    参数:
        model_path: 模型文件路径
        input_shape: 输入数据形状，默认为(300, 3)
        
    返回:
        重建的TCAN模型
    """
    print(f"使用TCAN适配器重建模型: {model_path}")
    
    # 创建新的TCAN模型
    try:
        # 使用默认参数创建模型
        model = create_tcan_model(input_shape=input_shape)
        
        # 尝试从h5文件加载权重
        try:
            # 首先尝试只加载权重而不是整个模型
            model.load_weights(model_path)
            print("✓ 成功加载TCAN模型权重!")
            
            # 保存新模型到更兼容的格式
            new_path = model_path.replace('.h5', '_rebuilt.keras')
            model.save(new_path)
            print(f"✓ 重建的模型已保存到: {new_path}")
            
            return model
        except Exception as e:
            print(f"✗ 无法加载权重: {e}")
            return None
    except Exception as e:
        print(f"✗ 无法创建TCAN模型: {e}")
        return None

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='雷达心率预测模型评估工具')
    
    # 基本参数
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['single', 'batch', 'quick'],
                       help='评估模式: single(单模型详细评估), batch(批量评估和比较), quick(快速验证)')
    
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH,
                       help=f'数据路径 (默认: {DEFAULT_DATA_PATH})')
    
    parser.add_argument('--decomp_method', type=str, default=DEFAULT_DECOMP_METHOD,
                       choices=['EEMD', 'EMD', 'VMD', 'DWT'],
                       help=f'分解方法 (默认: {DEFAULT_DECOMP_METHOD})')
    
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_DIR,
                       help=f'结果保存目录 (默认: {DEFAULT_RESULTS_DIR})')
    
    parser.add_argument('--use_original', action='store_true',
                       help='是否使用原始数据 (用于Multi_Input模型)')
    
    # 单模型评估参数
    parser.add_argument('--model_path', type=str, 
                       help='模型文件路径 (用于single和quick模式)')
    
    parser.add_argument('--plot', action='store_true',
                       help='生成评估图表 (用于single模式)')
    
    parser.add_argument('--measure_time', action='store_true',
                       help='测量推理时间 (用于single模式)')
    
    parser.add_argument('--model_name', type=str, default=None,
                       help='自定义模型名称 (用于single模式)')
    
    # 批量评估参数
    parser.add_argument('--models_dir', type=str, default=DEFAULT_MODELS_DIR,
                       help=f'模型目录 (用于batch模式，默认: {DEFAULT_MODELS_DIR})')
    
    parser.add_argument('--models', type=str, nargs='*',
                       help='要评估的模型名称列表 (用于batch模式，例如: CNN_GRU BiLSTM)')
    
    parser.add_argument('--plot_individual', action='store_true',
                       help='为每个模型生成单独的评估图表 (用于batch模式)')
    
    args = parser.parse_args()
    
    # 根据模式执行相应功能
    if args.mode == 'single':
        # 验证必要参数
        if args.model_path is None:
            parser.error("single模式需要 --model_path 参数")
        
        # 执行单模型评估
        evaluate_single_model(
            model_path=args.model_path,
            data_path=args.data_path,
            decomp_method=args.decomp_method,
            results_dir=args.results_dir,
            use_original=args.use_original,
            plot=args.plot,
            measure_time=args.measure_time,
            model_name=args.model_name
        )
    
    elif args.mode == 'batch':
        # 执行批量模型评估和比较
        evaluate_batch_models(
            models_dir=args.models_dir,
            data_path=args.data_path,
            decomp_method=args.decomp_method,
            results_dir=args.results_dir,
            models_to_evaluate=args.models,
            use_original=args.use_original,
            plot_individual=args.plot_individual
        )
    
    elif args.mode == 'quick':
        # 验证必要参数
        if args.model_path is None:
            parser.error("quick模式需要 --model_path 参数")
        
        # 执行快速模型评估
        quick_evaluate_model(
            model_path=args.model_path,
            data_path=args.data_path,
            decomp_method=args.decomp_method,
            use_original=args.use_original
        )

if __name__ == "__main__":
    main() 