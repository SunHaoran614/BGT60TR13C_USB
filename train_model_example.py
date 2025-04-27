"""
雷达相位数据心率预测模型训练示例

该脚本展示如何使用EEMD分解数据训练并比较各种深度学习模型，
用于从雷达相位数据中预测心率。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 导入我们实现的模型库
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
    load_data,
    train_model,
    evaluate_model
)

# 设置随机种子以确保结果可复现
np.random.seed(42)
tf.random.set_seed(42)

# 配置参数
DATA_PATH = 'training_data'  # 训练数据路径
DECOMP_METHOD = 'EEMD'  # 使用的分解方法：'EEMD', 'EMD', 'VMD' 或 'DWT'
USE_ORIGINAL = True  # 是否同时使用原始数据
EPOCHS = 100  # 训练轮数
BATCH_SIZE = 32  # 批次大小
LEARNING_RATE = 1e-3  # 学习率
TEST_SIZE = 0.2  # 测试集比例
VAL_SIZE = 0.2  # 验证集比例
MODELS_DIR = 'trained_models'  # 保存模型的目录
LOGS_DIR = 'logs'  # 日志目录
RESULTS_DIR = 'results'  # 结果保存目录

# 确保目录存在
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def train_and_evaluate_models():
    """训练并评估不同的深度学习模型"""
    
    print("="*50)
    print(f"正在使用{DECOMP_METHOD}分解数据训练模型...")
    print("="*50)
    
    # 加载数据
    if USE_ORIGINAL:
        # 使用分解数据和原始数据
        (decomp_train, decomp_test,
         orig_train, orig_test,
         y_train, y_test) = load_data(
            DATA_PATH, decomp_method=DECOMP_METHOD,
            use_original=True, test_size=TEST_SIZE
        )
    else:
        # 只使用分解数据
        decomp_train, decomp_test, y_train, y_test = load_data(
            DATA_PATH, decomp_method=DECOMP_METHOD,
            use_original=False, test_size=TEST_SIZE
        )
    
    # 确定输入形状
    if len(decomp_train.shape) == 3:
        # 形状为 (samples, modes, timesteps)
        n_samples, n_modes, n_timesteps = decomp_train.shape
        decomp_shape = (n_modes, n_timesteps)
    else:
        # 展平的形状 (samples, features)
        decomp_shape = (decomp_train.shape[1],)
    
    if USE_ORIGINAL:
        if len(orig_train.shape) == 2:
            # 形状为 (samples, timesteps)
            orig_shape = (orig_train.shape[1], 1)
            # 调整原始数据形状以符合Conv1D输入要求
            orig_train = orig_train.reshape(orig_train.shape[0], orig_train.shape[1], 1)
            orig_test = orig_test.reshape(orig_test.shape[0], orig_test.shape[1], 1)
        else:
            orig_shape = orig_train.shape[1:]
    else:
        orig_shape = None
    
    # 保存训练结果
    results = {}
    
    # 定义要训练的模型
    models_to_train = {
        'CNN_GRU': create_cnn_gru_model(decomp_shape),
        'TCN': create_tcn_model(decomp_shape),
        'MobileNet1D': create_mobilenet1d_model(decomp_shape, alpha=0.5),
        'LSTM': create_lstm_model(decomp_shape),
        'BiLSTM': create_bidirectional_lstm_model(decomp_shape),
        'CNN_LSTM': create_cnn_lstm_model(decomp_shape),
        'AttentionLSTM': create_attention_lstm_model(decomp_shape),
        'MultiscaleLSTM': create_multiscale_lstm_model(decomp_shape)
    }
    
    # 如果同时使用原始数据，添加多输入模型
    if USE_ORIGINAL:
        models_to_train['Multi_Input'] = create_multi_input_model(
            decomp_shape, orig_shape, DECOMP_METHOD
        )
    
    # 训练和评估每个模型
    for model_name, model in models_to_train.items():
        print(f"\n训练模型: {model_name}")
        print("-"*30)
        
        # 训练模型
        if model_name == 'Multi_Input' and USE_ORIGINAL:
            # 多输入模型需要分解数据和原始数据
            history = train_model(
                model, [decomp_train, orig_train], y_train,
                batch_size=BATCH_SIZE, epochs=EPOCHS,
                model_name=f"{model_name}_{DECOMP_METHOD}",
                tensorboard_dir=LOGS_DIR,
                checkpoint_dir=MODELS_DIR
            )
            
            # 评估模型
            metrics = evaluate_model(
                model, [decomp_test, orig_test], y_test,
                plot_results=True,
                model_name=f"{model_name} with {DECOMP_METHOD}"
            )
        else:
            # 单输入模型只需要分解数据
            history = train_model(
                model, decomp_train, y_train,
                batch_size=BATCH_SIZE, epochs=EPOCHS,
                model_name=f"{model_name}_{DECOMP_METHOD}",
                tensorboard_dir=LOGS_DIR,
                checkpoint_dir=MODELS_DIR
            )
            
            # 评估模型
            metrics = evaluate_model(
                model, decomp_test, y_test,
                plot_results=True,
                model_name=f"{model_name} with {DECOMP_METHOD}"
            )
        
        # 保存结果
        results[model_name] = {
            'history': history.history,
            'metrics': metrics
        }
        
        # 保存训练曲线
        plt.figure(figsize=(12, 5))
        
        # 训练损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='训练损失')
        plt.plot(history.history['val_loss'], label='验证损失')
        plt.xlabel('轮次')
        plt.ylabel('损失值')
        plt.title(f'{model_name} 训练和验证损失')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 训练MAE曲线
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='训练MAE')
        plt.plot(history.history['val_mae'], label='验证MAE')
        plt.xlabel('轮次')
        plt.ylabel('MAE (BPM)')
        plt.title(f'{model_name} 训练和验证MAE')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'{model_name}_{DECOMP_METHOD}_training_curves.png'))
        plt.close()
    
    # 比较所有模型性能
    compare_models(results)
    
    return results


def compare_models(results):
    """比较不同模型的性能"""
    # 提取性能指标
    model_names = list(results.keys())
    mae_values = [results[model]['metrics']['mae'] for model in model_names]
    rmse_values = [results[model]['metrics']['rmse'] for model in model_names]
    r2_values = [results[model]['metrics']['r2'] for model in model_names]
    mape_values = [results[model]['metrics']['mape'] for model in model_names]
    
    # 创建结果表格
    results_df = pd.DataFrame({
        '模型': model_names,
        'MAE (BPM)': mae_values,
        'RMSE (BPM)': rmse_values,
        'R²': r2_values,
        'MAPE (%)': mape_values
    })
    
    # 按MAE排序
    results_df = results_df.sort_values('MAE (BPM)')
    
    # 保存结果到CSV
    results_df.to_csv(os.path.join(RESULTS_DIR, f'model_comparison_{DECOMP_METHOD}.csv'), index=False)
    
    # 打印结果表格
    print("\n模型性能比较:")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60)
    
    # 绘制模型比较图
    plt.figure(figsize=(14, 10))
    
    # MAE比较
    plt.subplot(2, 2, 1)
    bars = plt.bar(model_names, mae_values, color='skyblue')
    plt.ylabel('MAE (BPM)')
    plt.title('不同模型的平均绝对误差 (MAE)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    # 在每个柱状图上添加数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom')
    
    # RMSE比较
    plt.subplot(2, 2, 2)
    bars = plt.bar(model_names, rmse_values, color='lightgreen')
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
    bars = plt.bar(model_names, r2_values, color='salmon')
    plt.ylabel('R²')
    plt.title('不同模型的决定系数 (R²)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')
    
    # MAPE比较
    plt.subplot(2, 2, 4)
    bars = plt.bar(model_names, mape_values, color='plum')
    plt.ylabel('MAPE (%)')
    plt.title('不同模型的平均绝对百分比误差 (MAPE)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'model_comparison_{DECOMP_METHOD}.png'))
    plt.close()


if __name__ == "__main__":
    # 训练并评估模型
    results = train_and_evaluate_models() 