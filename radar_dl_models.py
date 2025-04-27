"""
雷达信号深度学习模型库

该模块包含多种轻量级深度学习模型，专为雷达相位信号处理和心率预测设计。
模型包括:
- CNN+GRU混合模型
- 轻量级TCN (时序卷积网络)
- 多输入融合模型 (用于多种信号分解结果)
- 超轻量级MobileNet-1D模型
- LSTM及其变体 (双向LSTM, CNN+LSTM, Attention LSTM)

每个模型都经过优化，平衡准确性和计算效率，适合在资源受限环境下运行。
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, MaxPooling1D, GlobalAveragePooling1D,
    Dense, Dropout, LSTM, Bidirectional, GRU, Concatenate, Add, 
    Activation, Lambda, Flatten, TimeDistributed, Attention, LayerNormalization
)
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
import matplotlib.pyplot as plt


# ===================== 1. CNN + GRU 混合模型 =====================

def create_cnn_gru_model(input_shape, output_units=1, dropout_rate=0.2):
    """
    轻量级CNN+GRU混合模型，适合处理心率预测的相位时间序列数据
    
    参数:
        input_shape: 输入数据形状 (时间步长, 特征)
        output_units: 输出单元数量，默认为1 (心率预测)
        dropout_rate: Dropout比率，用于减轻过拟合
        
    返回:
        编译好的Keras模型
    """
    # 输入层
    inputs = Input(shape=input_shape)
    
    # CNN部分 - 提取空间特征
    x = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # GRU部分 - 捕获时序依赖
    gru = GRU(64, return_sequences=False)(x)
    
    # 全连接层
    x = Dense(32, activation='relu')(gru)
    x = Dropout(dropout_rate)(x)
    output = Dense(output_units, activation='linear')(x)
    
    # 构建模型
    model = Model(inputs=inputs, outputs=output)
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# ===================== 2. 轻量级TCN模型 =====================

def residual_block(x, dilation_rate, nb_filters, kernel_size, dropout_rate=0.2):
    """
    TCN的残差块，使用空洞卷积捕获长期时序依赖
    
    参数:
        x: 输入张量
        dilation_rate: 空洞率
        nb_filters: 滤波器数量
        kernel_size: 卷积核大小
        dropout_rate: Dropout比率
        
    返回:
        处理后的特征张量
    """
    # 添加填充以确保因果卷积
    padding_size = (kernel_size - 1) * dilation_rate
    x_padding = Lambda(lambda x: tf.pad(x, [[0, 0], [padding_size, 0], [0, 0]]))(x)
    
    # 两个因果卷积层
    conv1 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                   dilation_rate=dilation_rate, padding='valid')(x_padding)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Dropout(dropout_rate)(conv1)
    
    # 第二个卷积层
    x_padding2 = Lambda(lambda x: tf.pad(x, [[0, 0], [padding_size, 0], [0, 0]]))(conv1)
    conv2 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                   dilation_rate=dilation_rate, padding='valid')(x_padding2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Dropout(dropout_rate)(conv2)
    
    # 如果输入和输出维度不匹配，则使用1x1卷积调整
    if x.shape[-1] != nb_filters:
        x = Conv1D(filters=nb_filters, kernel_size=1, padding='same')(x)
    
    # 添加跳跃连接
    output = Add()([conv2, x])
    
    return output

def create_tcn_model(input_shape, output_units=1, nb_filters=64, kernel_size=3, 
                     dilations=[1, 2, 4, 8], dropout_rate=0.2):
    """
    创建轻量级TCN (时序卷积网络) 模型
    
    参数:
        input_shape: 输入数据形状
        output_units: 输出单元数量
        nb_filters: 每层的滤波器数量
        kernel_size: 卷积核大小
        dilations: 空洞率列表
        dropout_rate: Dropout比率
        
    返回:
        编译好的Keras模型
    """
    inputs = Input(shape=input_shape)
    
    x = inputs
    # 创建TCN的残差块堆栈
    for dilation in dilations:
        x = residual_block(x, dilation, nb_filters, kernel_size, dropout_rate)
    
    # 全局池化后接全连接层
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(output_units, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# ===================== 3. 多输入融合模型 =====================

def create_multi_input_model(decomp_shape, original_shape=None, decomp_type='EEMD', 
                             output_units=1, dropout_rate=0.3):
    """
    处理多种输入的融合模型，可同时处理原始信号和分解结果
    
    参数:
        decomp_shape: 分解数据的形状 (通常为 modes, time_steps)
        original_shape: 原始数据形状，如果为None则不使用原始数据
        decomp_type: 分解方法的名称 ('EEMD', 'EMD', 'VMD', 'DWT')
        output_units: 输出单元数量
        dropout_rate: Dropout比率
        
    返回:
        编译好的Keras模型
    """
    # 分解信号输入分支
    decomp_input = Input(shape=decomp_shape, name=f'{decomp_type}_input')
    
    # 处理分解信号 - 使用卷积提取特征
    x1 = Conv1D(32, 5, activation='relu', padding='same')(decomp_input)
    x1 = MaxPooling1D(2)(x1)
    x1 = Conv1D(64, 3, activation='relu', padding='same')(x1)
    x1 = GlobalAveragePooling1D()(x1)
    
    # 如果有原始信号输入
    if original_shape:
        original_input = Input(shape=original_shape, name='original_input')
        x2 = Conv1D(32, 5, activation='relu', padding='same')(original_input)
        x2 = MaxPooling1D(2)(x2)
        x2 = Conv1D(64, 3, activation='relu', padding='same')(x2)
        x2 = GlobalAveragePooling1D()(x2)
        
        # 合并两个分支
        x = Concatenate()([x1, x2])
        inputs = [decomp_input, original_input]
    else:
        x = x1
        inputs = decomp_input
    
    # 共享全连接层
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(output_units, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=output)
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# ===================== 4. 超轻量级MobileNet风格的1D模型 =====================

def create_mobilenet1d_model(input_shape, output_units=1, alpha=0.5, dropout_rate=0.2):
    """
    超轻量级1D-MobileNet模型，使用深度可分离卷积降低参数量
    
    参数:
        input_shape: 输入数据形状
        output_units: 输出单元数量
        alpha: 通道乘数，用于控制模型大小
        dropout_rate: Dropout比率
        
    返回:
        编译好的Keras模型
    """
    inputs = Input(shape=input_shape)
    
    # 初始卷积层
    x = Conv1D(int(32 * alpha), 3, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 构建深度可分离卷积块
    def depthwise_block(x, filters, strides=1):
        # 深度卷积 - 对每个通道单独卷积
        x = tf.keras.layers.DepthwiseConv1D(
            kernel_size=3, 
            strides=strides, 
            padding='same', 
            depth_multiplier=1
        )(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # 逐点卷积 - 1x1卷积整合通道信息
        x = Conv1D(int(filters * alpha), 1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    
    # 堆叠深度可分离卷积块
    x = depthwise_block(x, 64)
    x = depthwise_block(x, 128, strides=2)
    x = depthwise_block(x, 128)
    x = depthwise_block(x, 256, strides=2)
    
    # 全局池化和密集层
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(output_units, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# ===================== 5. 基础LSTM模型 =====================

def create_lstm_model(input_shape, output_units=1, lstm_units=64, dropout_rate=0.2):
    """
    基础LSTM模型，适合捕获时间序列中的长期依赖关系
    
    参数:
        input_shape: 输入数据形状
        output_units: 输出单元数量
        lstm_units: LSTM层的单元数量
        dropout_rate: Dropout比率
        
    返回:
        编译好的Keras模型
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(lstm_units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(lstm_units//2),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dense(output_units, activation='linear')
    ])
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# ===================== 6. 双向LSTM模型 =====================

def create_bidirectional_lstm_model(input_shape, output_units=1, lstm_units=64, dropout_rate=0.2):
    """
    双向LSTM模型，可以同时从前向和后向捕获时序信息
    
    参数:
        input_shape: 输入数据形状
        output_units: 输出单元数量
        lstm_units: LSTM层的单元数量
        dropout_rate: Dropout比率
        
    返回:
        编译好的Keras模型
    """
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(lstm_units, return_sequences=True)),
        Dropout(dropout_rate),
        Bidirectional(LSTM(lstm_units//2)),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dense(output_units, activation='linear')
    ])
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# ===================== 7. CNN + LSTM模型 =====================

def create_cnn_lstm_model(input_shape, output_units=1, lstm_units=64, dropout_rate=0.2):
    """
    CNN + LSTM混合模型，CNN提取局部特征，LSTM捕获时序依赖
    
    参数:
        input_shape: 输入数据形状
        output_units: 输出单元数量
        lstm_units: LSTM层的单元数量
        dropout_rate: Dropout比率
        
    返回:
        编译好的Keras模型
    """
    inputs = Input(shape=input_shape)
    
    # CNN部分 - 提取局部特征
    x = Conv1D(32, 5, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # LSTM部分 - 捕获时序依赖
    x = LSTM(lstm_units, return_sequences=False)(x)
    x = Dropout(dropout_rate)(x)
    
    # 全连接层
    x = Dense(32, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(output_units, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=output)
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# ===================== 8. Attention LSTM模型 =====================

def create_attention_lstm_model(input_shape, output_units=1, lstm_units=64, dropout_rate=0.2):
    """
    带注意力机制的LSTM模型，可以自动关注重要的时间步
    
    参数:
        input_shape: 输入数据形状
        output_units: 输出单元数量
        lstm_units: LSTM层的单元数量
        dropout_rate: Dropout比率
        
    返回:
        编译好的Keras模型
    """
    inputs = Input(shape=input_shape)
    
    # LSTM层，返回序列
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    lstm_out = Dropout(dropout_rate)(lstm_out)
    
    # 应用注意力机制
    attention_out = Attention()([lstm_out, lstm_out])
    
    # 全局池化提取特征
    x = GlobalAveragePooling1D()(attention_out)
    
    # 全连接层
    x = Dense(32, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(output_units, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=output)
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# ===================== 9. 多尺度LSTM模型 =====================

def create_multiscale_lstm_model(input_shape, output_units=1, dropout_rate=0.2):
    """
    多尺度LSTM模型，在不同时间尺度上捕获信息
    
    参数:
        input_shape: 输入数据形状
        output_units: 输出单元数量
        dropout_rate: Dropout比率
        
    返回:
        编译好的Keras模型
    """
    inputs = Input(shape=input_shape)
    
    # 不同尺度的LSTM分支
    # 分支1：捕获全局信息
    lstm1 = LSTM(64, return_sequences=False)(inputs)
    lstm1 = Dropout(dropout_rate)(lstm1)
    
    # 分支2：先降采样再处理，捕获中尺度信息
    pool1 = MaxPooling1D(pool_size=2)(inputs)
    lstm2 = LSTM(32, return_sequences=False)(pool1)
    lstm2 = Dropout(dropout_rate)(lstm2)
    
    # 分支3：更大程度降采样，捕获粗粒度信息
    pool2 = MaxPooling1D(pool_size=4)(inputs)
    lstm3 = LSTM(16, return_sequences=False)(pool2)
    lstm3 = Dropout(dropout_rate)(lstm3)
    
    # 合并不同尺度的特征
    merged = Concatenate()([lstm1, lstm2, lstm3])
    
    # 全连接层
    x = Dense(32, activation='relu')(merged)
    x = Dropout(dropout_rate)(x)
    output = Dense(output_units, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=output)
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# ===================== 训练和评估函数 =====================

def load_data(data_path, decomp_method='EEMD', use_original=False, test_size=0.2, random_state=42):
    """
    加载和预处理训练数据
    
    参数:
        data_path: 数据文件路径 (.npz格式)
        decomp_method: 使用的分解方法
        use_original: 是否同时使用原始数据
        test_size: 测试集比例
        random_state: 随机种子
        
    返回:
        训练和测试数据集
    """
    print(f"加载 {decomp_method} 分解数据...")
    
    # 加载分解数据
    data_file = os.path.join(data_path, f'training_data_{decomp_method}.npz')
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"未找到数据文件: {data_file}")
    
    data = np.load(data_file)
    
    # 根据分解方法获取相应的特征
    if decomp_method in ['EMD', 'EEMD', 'VMD']:
        decomp_data = data['decomp_data']  # 形状应该是 (samples, modes, time_steps)
    elif decomp_method == 'DWT':
        decomp_data = data['decomp_data']  # DWT系数
    else:
        raise ValueError(f"不支持的分解方法: {decomp_method}")
    
    # 获取标签
    hr_values = data['hr_values']  # 心率标签
    
    # 加载原始数据（如果需要）
    if use_original:
        original_file = os.path.join(data_path, 'training_data_original.npz')
        if not os.path.exists(original_file):
            raise FileNotFoundError(f"未找到原始数据文件: {original_file}")
        
        original_data = np.load(original_file)
        original_features = original_data['features']
        
        # 划分训练集和测试集
        (decomp_train, decomp_test, 
         orig_train, orig_test, 
         y_train, y_test) = train_test_split(
            decomp_data, original_features, hr_values, 
            test_size=test_size, random_state=random_state
        )
        
        return (decomp_train, decomp_test, 
                orig_train, orig_test, 
                y_train, y_test)
    
    # 只使用分解数据
    else:
        # 划分训练集和测试集
        decomp_train, decomp_test, y_train, y_test = train_test_split(
            decomp_data, hr_values, test_size=test_size, random_state=random_state
        )
        
        return decomp_train, decomp_test, y_train, y_test


def train_model(model, X_train, y_train, X_val=None, y_val=None, 
                batch_size=32, epochs=100, model_name="model", 
                tensorboard_dir="logs", checkpoint_dir="models"):
    """
    训练模型并保存
    
    参数:
        model: 要训练的模型
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征 (可选)
        y_val: 验证标签 (可选)
        batch_size: 批次大小
        epochs: 训练轮数
        model_name: 模型名称
        tensorboard_dir: TensorBoard日志目录
        checkpoint_dir: 模型检查点目录
        
    返回:
        训练历史记录
    """
    # 创建必要的目录
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 设置回调函数
    callbacks = [
        EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f"{model_name}_best.h5"),
            monitor='val_loss' if X_val is not None else 'loss',
            save_best_only=True,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(tensorboard_dir, model_name),
            histogram_freq=1
        )
    ]
    
    # 训练模型
    if X_val is not None and y_val is not None:
        # 使用提供的验证集
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
    else:
        # 从训练集中划分验证集
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
    
    # 保存最终模型
    model.save(os.path.join(checkpoint_dir, f"{model_name}_final.h5"))
    
    return history


def evaluate_model(model, X_test, y_test, plot_results=True, model_name="model"):
    """
    评估模型性能并可选地可视化结果
    
    参数:
        model: 要评估的模型
        X_test: 测试特征
        y_test: 测试标签
        plot_results: 是否绘制预测vs真实值图表
        model_name: 模型名称（用于图表标题）
        
    返回:
        评估指标字典
    """
    # 在测试集上评估模型
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集MAE: {test_mae:.4f} BPM")
    
    # 预测测试集
    y_pred = model.predict(X_test)
    
    # 计算额外的指标
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # 转换为百分比
    
    print(f"RMSE: {rmse:.4f} BPM")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    # 可视化结果
    if plot_results:
        plt.figure(figsize=(10, 6))
        
        # 绘制散点图
        plt.scatter(y_test, y_pred, alpha=0.5)
        
        # 绘制理想预测线 (y=x)
        min_val = min(np.min(y_test), np.min(y_pred))
        max_val = max(np.max(y_test), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('实际心率 (BPM)', fontsize=12)
        plt.ylabel('预测心率 (BPM)', fontsize=12)
        plt.title(f'{model_name} 模型心率预测结果\nMAE: {test_mae:.2f} BPM, R²: {r2:.3f}', fontsize=14)
        
        # 添加误差分布直方图作为子图
        errors = y_pred.flatten() - y_test
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=20, alpha=0.7, color='skyblue')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('预测误差 (BPM)', fontsize=12)
        plt.ylabel('频率', fontsize=12)
        plt.title(f'{model_name} 预测误差分布', fontsize=14)
        
        plt.tight_layout()
        plt.show()
    
    # 返回评估指标
    metrics = {
        'mae': test_mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }
    
    return metrics


# ===================== 示例使用代码 =====================

def example_usage():
    """示例使用代码，展示如何使用上述模型和函数"""
    
    # 加载数据 (EEMD分解)
    data_path = 'training_data'  # 数据目录
    decomp_train, decomp_test, y_train, y_test = load_data(
        data_path, decomp_method='EEMD', use_original=False
    )
    
    print(f"分解数据形状: {decomp_train.shape}")
    print(f"训练标签形状: {y_train.shape}")
    
    # 获取输入形状
    n_samples, n_modes, n_timesteps = decomp_train.shape
    input_shape = (n_modes, n_timesteps)
    
    # 创建CNN+GRU模型
    model = create_cnn_gru_model(input_shape)
    print(model.summary())
    
    # 训练模型
    history = train_model(
        model, decomp_train, y_train,
        batch_size=32, epochs=50,
        model_name="CNN_GRU_EEMD"
    )
    
    # 评估模型
    metrics = evaluate_model(model, decomp_test, y_test, model_name="CNN+GRU with EEMD")
    
    return model, history, metrics


if __name__ == "__main__":
    # 运行示例代码
    example_usage() 