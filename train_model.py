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
import tensorflow.keras as K
import matplotlib

# 添加健壮的matplotlib字体处理
try:
    # 检查字体是否存在
    from matplotlib.font_manager import FontProperties
    font_exists = False
    for font in ['SimHei', 'Microsoft YaHei', 'SimSun']:
        try:
            FontProperties(fname=font)
            font_exists = True
            break
        except:
            continue
    
    # 如果找不到中文字体，使用系统默认字体
    if not font_exists:
        print("警告: 未找到中文字体，将使用系统默认字体")
        matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    else:
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']
    
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.family'] = 'sans-serif'
except Exception as e:
    print(f"设置matplotlib字体时出错: {e}")
    print("将使用默认字体，某些中文可能无法正确显示")

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
    create_transformer_model,
    create_resnet1d_model,
    create_tcan_model,
    create_inception_time_model,
    create_wavenet_model,
    create_nbeats_model,
    create_tft_model,
    create_omniscale_cnn_model,
    create_deep_state_space_model,
    create_lstnet_model,
    load_data,
    train_model,
    evaluate_model
)

# 配置参数
DATA_PATH = 'training_data'  # 训练数据路径
DECOMP_METHOD = 'VMD'  # 使用的分解方法：'EEMD', 'EMD', 'VMD' 或 'DWT'
USE_ORIGINAL = False  # 是否同时使用原始数据
EPOCHS = 2000  # 训练轮数
BATCH_SIZE = 32  # 批次大小
LEARNING_RATE = 1e-3  # 学习率
TEST_SIZE = 0.2  # 测试集比例
VAL_SIZE = 0.2  # 验证集比例
RANDOM_SEED = 42  # 随机种子，控制训练集划分的随机性
MODELS_DIR = 'trained_models'  # 保存模型的目录
LOGS_DIR = 'logs'  # 日志目录
RESULTS_DIR = 'results'  # 结果保存目录

# 设置随机种子以确保结果可复现
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

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
            use_original=True, test_size=TEST_SIZE,
            random_state=RANDOM_SEED
        )
    else:
        # 只使用分解数据
        decomp_train, decomp_test, y_train, y_test = load_data(
            DATA_PATH, decomp_method=DECOMP_METHOD,
            use_original=False, test_size=TEST_SIZE,
            random_state=RANDOM_SEED
        )
    
    # 确定输入形状
    if len(decomp_train.shape) == 3:
        # 形状为 (samples, modes, timesteps)
        n_samples, n_modes, n_timesteps = decomp_train.shape
        # 交换维度顺序，将时间维度放在前面，特征维度放在后面
        decomp_shape = (n_timesteps, n_modes)
        # 调整分解数据形状以符合Conv1D输入要求: (samples, timesteps, features)
        decomp_train = decomp_train.transpose(0, 2, 1)
        decomp_test = decomp_test.transpose(0, 2, 1)
        print(f"调整后的分解数据形状: decomp_train.shape = {decomp_train.shape}, decomp_test.shape = {decomp_test.shape}")
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
    
    # 定义要训练的模型
    models_to_train = {
        # 'CNN_GRU': create_cnn_gru_model(decomp_shape),
        # 'TCN': create_tcn_model(decomp_shape),
        # 'MobileNet1D': create_mobilenet1d_model(decomp_shape, alpha=0.5),
        # 'LSTM': create_lstm_model(decomp_shape),
        # 'BiLSTM': create_bidirectional_lstm_model(decomp_shape),
        # 'CNN_LSTM': create_cnn_lstm_model(decomp_shape),
        # 'AttentionLSTM': create_attention_lstm_model(decomp_shape),
        # 'MultiscaleLSTM': create_multiscale_lstm_model(decomp_shape),
        'Transformer': create_transformer_model(decomp_shape),
        'ResNet1D': create_resnet1d_model(decomp_shape),
        'TCAN': create_tcan_model(decomp_shape),
        'InceptionTime': create_inception_time_model(decomp_shape),
        'WaveNet': create_wavenet_model(decomp_shape),
        'N-Beats': create_nbeats_model(decomp_shape),
        'TFT': create_tft_model(decomp_shape),
        # 'OmniScaleCNN': create_omniscale_cnn_model(decomp_shape), # 内存占用太大，暂时不训练
        'DeepStateSpace': create_deep_state_space_model(decomp_shape),
        'LSTMNet': create_lstnet_model(decomp_shape)
    }
    
    # 如果同时使用原始数据，添加多输入模型
    if USE_ORIGINAL:
        models_to_train['Multi_Input'] = create_multi_input_model(
            decomp_shape, orig_shape, DECOMP_METHOD
        )
    
    # 训练模型
    models_trained = {}
    results = {}
    
    for model_name, model in models_to_train.items():
        print(f"\n开始训练 {model_name} 模型...")
        
        # 训练模型
        history, training_details = train_model(
            model, decomp_train, y_train,
            batch_size=32, epochs=EPOCHS,
            model_name=f"{model_name}_{DECOMP_METHOD}",
            tensorboard_dir=LOGS_DIR,
            checkpoint_dir=MODELS_DIR
        )
        
        # 在测试集上评估模型
        print(f"评估 {model_name} 模型...")
        metrics = evaluate_model(
            model, decomp_test, y_test, 
            model_name=f"{model_name}_{DECOMP_METHOD}"
        )
        
        # 保存结果
        models_trained[model_name] = model
        results[model_name] = {
            'model': model,
            'history': history,
            'metrics': metrics,
            'training_details': training_details
        }
        
        # 清除会话以释放内存
        try:
            # 尝试方法1：使用K.backend
            K.backend.clear_session()
        except Exception as e:
            print(f"使用K.backend.clear_session()失败: {e}")
            try:
                # 尝试方法2：使用tf.keras.backend
                tf.keras.backend.clear_session()
            except Exception as e2:
                print(f"使用tf.keras.backend.clear_session()失败: {e2}")
                try:
                    # 尝试方法3：使用global tensorflow重置
                    tf.compat.v1.reset_default_graph()
                    print("使用tf.compat.v1.reset_default_graph()代替")
                except Exception as e3:
                    print(f"清理会话失败，但继续执行: {e3}")
    
    # 比较不同模型的性能
    compare_models(results)
    
    return results


def compare_models(results):
    """比较不同模型的性能"""
    # 提取性能指标和训练详情
    model_names = list(results.keys())
    
    # 创建结果表格 - 基本指标
    basic_metrics_df = pd.DataFrame({
        '模型': model_names,
        'MAE (BPM)': [results[model]['metrics']['mae'] for model in model_names],
        'RMSE (BPM)': [results[model]['metrics']['rmse'] for model in model_names],
        'R²': [results[model]['metrics']['r2'] for model in model_names],
        'MAPE (%)': [results[model]['metrics']['mape'] for model in model_names],
        '最大误差 (BPM)': [results[model]['metrics']['max_error'] for model in model_names],
        '中位误差 (BPM)': [results[model]['metrics']['median_error'] for model in model_names],
        '误差标准差': [results[model]['metrics']['error_std'] for model in model_names],
        '95%置信区间下限': [results[model]['metrics']['ci_95_lower'] for model in model_names],
        '95%置信区间上限': [results[model]['metrics']['ci_95_upper'] for model in model_names],
        '±3BPM准确率 (%)': [results[model]['metrics']['accuracy_3bpm'] for model in model_names]
    })
    
    # 添加心率区间性能
    if 'mae_low_hr' in results[model_names[0]]['metrics']:
        basic_metrics_df['低心率区间MAE'] = [results[model]['metrics'].get('mae_low_hr', np.nan) for model in model_names]
        basic_metrics_df['正常心率区间MAE'] = [results[model]['metrics'].get('mae_normal_hr', np.nan) for model in model_names]
        basic_metrics_df['高心率区间MAE'] = [results[model]['metrics'].get('mae_high_hr', np.nan) for model in model_names]
    
    # 添加推理性能指标
    if 'inference_time_ms' in results[model_names[0]]['metrics']:
        basic_metrics_df['单样本推理时间 (ms)'] = [results[model]['metrics']['inference_time_ms'] for model in model_names]
        basic_metrics_df['样本/秒'] = [results[model]['metrics']['samples_per_second'] for model in model_names]
    
    # 创建训练详情表格
    training_df = pd.DataFrame({
        '模型': model_names,
        '参数总量': [results[model]['training_details']['total_params'] for model in model_names],
        '可训练参数': [results[model]['training_details']['trainable_params'] for model in model_names],
        '层数': [results[model]['training_details']['n_layers'] for model in model_names],
        '训练时间 (秒)': [results[model]['training_details']['training_time_seconds'] for model in model_names],
        '平均每轮时间 (秒)': [results[model]['training_details']['avg_epoch_time_seconds'] for model in model_names],
        '实际训练轮次': [results[model]['training_details']['actual_epochs'] for model in model_names],
        '最终学习率': [results[model]['training_details']['final_learning_rate'] for model in model_names],
        '最佳验证损失': [results[model]['training_details']['best_val_loss'] for model in model_names],
        '模型大小 (MB)': [results[model]['training_details']['model_size_mb'] for model in model_names],
        '批次大小': [results[model]['training_details']['batch_size'] for model in model_names],
        '训练样本数': [results[model]['training_details']['train_samples'] for model in model_names],
        '输入形状': [results[model]['training_details']['input_shape'] for model in model_names]
    })
    
    # 合并两个表格
    all_metrics_df = pd.merge(basic_metrics_df, training_df, on='模型')
    
    # 按MAE排序
    basic_metrics_df = basic_metrics_df.sort_values('MAE (BPM)')
    all_metrics_df = all_metrics_df.sort_values('MAE (BPM)')
    
    # 添加时间戳到CSV文件名
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    # 保存带时间戳的结果到CSV
    basic_metrics_df.to_csv(os.path.join(RESULTS_DIR, f'model_comparison_{DECOMP_METHOD}_{timestamp}.csv'), index=False)
    all_metrics_df.to_csv(os.path.join(RESULTS_DIR, f'model_comparison_detailed_{DECOMP_METHOD}_{timestamp}.csv'), index=False)
    
    # 同时保存一个最新版本（可能会覆盖）
    basic_metrics_df.to_csv(os.path.join(RESULTS_DIR, f'model_comparison_{DECOMP_METHOD}_latest.csv'), index=False)
    all_metrics_df.to_csv(os.path.join(RESULTS_DIR, f'model_comparison_detailed_{DECOMP_METHOD}_latest.csv'), index=False)
    
    # 打印结果表格
    print("\n基本性能指标:")
    print("=" * 60)
    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.width', 1000)  # 加宽显示
    print(basic_metrics_df.to_string(index=False))
    print("=" * 60)
    
    # 使用特殊符号的安全表示法
    try:
        r2_text = "R\u00B2"  # 使用Unicode字符U+00B2表示²
    except:
        r2_text = "R²"  # 直接使用²字符，可能在某些环境不显示
    
    # 绘制模型比较图
    try:
        plt.figure(figsize=(14, 10))
        
        # MAE比较
        plt.subplot(2, 2, 1)
        bars = plt.bar(model_names, basic_metrics_df['MAE (BPM)'].values, color='skyblue')
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
        bars = plt.bar(model_names, basic_metrics_df['RMSE (BPM)'].values, color='lightgreen')
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
        bars = plt.bar(model_names, basic_metrics_df['R²'].values, color='salmon')
        plt.ylabel(r2_text)
        plt.title(f'不同模型的决定系数 ({r2_text})')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom')
        
        # ±3BPM准确率比较
        plt.subplot(2, 2, 4)
        bars = plt.bar(model_names, basic_metrics_df['±3BPM准确率 (%)'].values, color='plum')
        plt.ylabel('准确率 (%)')
        plt.title('不同模型的±3BPM准确率')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'model_comparison_{DECOMP_METHOD}_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"生成比较图表时出错: {e}")
        print("将跳过图表生成并继续执行")
    
    # 创建第二个图表 - 训练资源比较
    try:
        plt.figure(figsize=(14, 10))
        
        # 参数量比较
        plt.subplot(2, 2, 1)
        params_values = [results[model]['training_details']['total_params'] / 1000 for model in model_names]  # 转换为千参数
        bars = plt.bar(model_names, params_values, color='lightblue')
        plt.ylabel('参数量 (千)')
        plt.title('不同模型的参数量')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(params_values)*0.02,
                     f'{height:.1f}K', ha='center', va='bottom')
        
        # 模型大小比较
        plt.subplot(2, 2, 2)
        size_values = [results[model]['training_details']['model_size_mb'] for model in model_names]
        bars = plt.bar(model_names, size_values, color='lightgreen')
        plt.ylabel('大小 (MB)')
        plt.title('不同模型的文件大小')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(size_values)*0.02,
                     f'{height:.1f}MB', ha='center', va='bottom')
        
        # 训练时间比较
        plt.subplot(2, 2, 3)
        time_values = [results[model]['training_details']['training_time_seconds'] / 60 for model in model_names]  # 转换为分钟
        bars = plt.bar(model_names, time_values, color='salmon')
        plt.ylabel('时间 (分钟)')
        plt.title('不同模型的训练时间')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(time_values)*0.02,
                     f'{height:.1f}分钟', ha='center', va='bottom')
        
        # 推理时间比较
        if 'inference_time_ms' in results[model_names[0]]['metrics']:
            plt.subplot(2, 2, 4)
            inference_values = [results[model]['metrics']['inference_time_ms'] for model in model_names]
            bars = plt.bar(model_names, inference_values, color='plum')
            plt.ylabel('时间 (毫秒)')
            plt.title('不同模型的单样本推理时间')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.7, axis='y')
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + max(inference_values)*0.02,
                        f'{height:.2f}ms', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'model_resources_{DECOMP_METHOD}_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"生成资源比较图表时出错: {e}")
        print("将跳过图表生成并继续执行")


if __name__ == "__main__":
    # 训练并评估模型
    results = train_and_evaluate_models() 