"""
雷达信号深度学习模型库

该模块包含多种轻量级深度学习模型，专为雷达相位信号处理和心率预测设计。
模型包括:
- CNN+GRU混合模型
- 轻量级TCN (时序卷积网络)
- 多输入融合模型 (用于多种信号分解结果)
- 超轻量级MobileNet-1D模型
- LSTM及其变体 (双向LSTM, CNN+LSTM, Attention LSTM)
- Transformer模型
- ResNet1D模型
- TCAN模型
- InceptionTime模型
- WaveNet模型
- N-BEATS模型
- TFT (Temporal Fusion Transformer) 模型
- OmniScaleCNN模型
- Deep State Space Model
- LSTNet (Long- and Short-term Time-series Network) 模型

每个模型都经过优化，平衡准确性和计算效率，适合在资源受限环境下运行。
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, MaxPooling1D, GlobalAveragePooling1D,
    Dense, Dropout, LSTM, Bidirectional, GRU, Concatenate, Add, 
    Activation, Lambda, Flatten, TimeDistributed, Attention, LayerNormalization,
    Subtract, Reshape, Multiply, RepeatVector, Dot, Permute
)
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
import matplotlib.pyplot as plt
import matplotlib


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
    x = MaxPooling1D(pool_size=1)(x)
    
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=1)(x)
    
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


# ===================== 10. Transformer模型 =====================

def create_transformer_model(input_shape, output_units=1, num_heads=4, 
                            dropout_rate=0.2, ff_dim=64, num_transformer_blocks=2):
    """
    基于Transformer架构的心率预测模型，利用自注意力机制捕获时序数据中的长距离依赖
    
    参数:
        input_shape: 输入数据形状 (时间步长, 特征)
        output_units: 输出单元数量，默认为1 (心率预测)
        num_heads: 多头注意力的头数
        dropout_rate: Dropout比率
        ff_dim: 前馈网络的维度
        num_transformer_blocks: Transformer块的数量
        
    返回:
        编译好的Keras模型
    """
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout_rate=0.2):
        # 多头自注意力
        attention_output = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout_rate
        )(inputs, inputs)
        attention_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        
        # 前馈网络
        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(inputs.shape[-1])
        ])
        ffn_output = ffn(attention_output)
        ffn_output = tf.keras.layers.Dropout(dropout_rate)(ffn_output)
        
        # 第二个残差连接和归一化
        sequence_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)
        return sequence_output
    
    inputs = Input(shape=input_shape)
    
    # 修复位置编码实现
    timesteps, features = input_shape
    
    # 创建适合时间步长的位置编码 - 使用Lambda层包装TensorFlow操作
    def position_encoding_fn(x):
        batch_size = tf.shape(x)[0]
        position_indices = tf.range(timesteps)
        position_encodings = tf.cast(position_indices, dtype=tf.float32)
        position_encodings = tf.expand_dims(position_encodings, axis=-1)  # [timesteps, 1]
        
        # 广播位置编码到每个特征维度
        position_encodings = tf.tile(position_encodings, [1, features])  # [timesteps, features]
        position_encodings = position_encodings / tf.cast(timesteps, tf.float32)  # 归一化位置编码
        
        # 扩展为批次维度并复制到批次大小
        position_encodings = tf.expand_dims(position_encodings, axis=0)  # [1, timesteps, features]
        position_encodings = tf.tile(position_encodings, [batch_size, 1, 1])  # [batch_size, timesteps, features]
        
        return position_encodings
    
    # 使用Lambda层生成位置编码
    position_embedding = Lambda(position_encoding_fn)(inputs)
    
    # 将位置编码添加到输入
    x = Add()([inputs, position_embedding])
    
    # Transformer编码器堆栈
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size=input_shape[1]//num_heads if input_shape[1]>=num_heads else 1, 
                               num_heads=min(num_heads, input_shape[1]), ff_dim=ff_dim, dropout_rate=dropout_rate)
    
    # 全局平均池化
    x = GlobalAveragePooling1D()(x)
    
    # 全连接层
    x = Dense(32, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(output_units, activation="linear")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"]
    )
    
    return model


# ===================== 11. ResNet1D模型 =====================

def create_resnet1d_model(input_shape, output_units=1, filters=64, kernel_size=3, 
                         n_block_layers=3, n_blocks=4, dropout_rate=0.2):
    """
    一维ResNet模型，使用残差连接解决深度网络中的梯度消失问题
    
    参数:
        input_shape: 输入数据形状
        output_units: 输出单元数量
        filters: 基础卷积滤波器数量
        kernel_size: 卷积核大小
        n_block_layers: 每个块中的卷积层数量
        n_blocks: 残差块的数量
        dropout_rate: Dropout比率
        
    返回:
        编译好的Keras模型
    """
    def residual_block(x, filters, kernel_size, downsample=False):
        """残差块"""
        y = Conv1D(filters, kernel_size=kernel_size, strides=1 if not downsample else 2, 
                  padding='same')(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        
        for i in range(n_block_layers-1):
            y = Conv1D(filters, kernel_size=kernel_size, padding='same')(y)
            y = BatchNormalization()(y)
            if i < n_block_layers-2:  # 最后一层卷积后不使用激活函数
                y = Activation('relu')(y)
        
        # 如果需要下采样或通道数不匹配，则调整跳跃连接
        if downsample or x.shape[-1] != filters:
            x = Conv1D(filters, kernel_size=1, strides=1 if not downsample else 2, 
                      padding='same')(x)
            x = BatchNormalization()(x)
        
        # 跳跃连接
        out = Add()([x, y])
        out = Activation('relu')(out)
        return out
    
    inputs = Input(shape=input_shape)
    
    # 第一层卷积
    x = Conv1D(filters, kernel_size=7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
    
    # 构建残差块
    for i in range(n_blocks):
        # 除第一个块外，其余块第一层进行下采样
        x = residual_block(x, filters * (2**min(i, 2)), kernel_size, 
                          downsample=(i > 0))
    
    # 全局池化
    x = GlobalAveragePooling1D()(x)
    
    # 全连接层
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


# ===================== 12. TCAN模型 =====================

def create_tcan_model(input_shape, output_units=1, nb_filters=64, kernel_size=3, 
                     dilations=[1, 2, 4, 8, 16], attention_units=32, dropout_rate=0.2):
    """
    时序卷积注意力网络(TCAN)模型，结合TCN的长时序建模能力和注意力机制的选择性关注
    
    参数:
        input_shape: 输入数据形状
        output_units: 输出单元数量
        nb_filters: 卷积滤波器数量
        kernel_size: 卷积核大小
        dilations: 空洞率列表
        attention_units: 注意力层的单元数
        dropout_rate: Dropout比率
        
    返回:
        编译好的Keras模型
    """
    # TCN部分 - 基于前面定义的residual_block函数
    def tcn_block(x, dilation_rate, nb_filters, kernel_size, dropout_rate=0.2):
        """TCN残差块，添加注意力机制"""
        padding_size = (kernel_size - 1) * dilation_rate
        # 添加output_shape参数
        x_padding = Lambda(
            lambda x: tf.pad(x, [[0, 0], [padding_size, 0], [0, 0]]),
            output_shape=lambda input_shape: (input_shape[0], input_shape[1] + padding_size, input_shape[2])
        )(x)
        
        # 两个因果卷积层
        conv1 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                      dilation_rate=dilation_rate, padding='valid')(x_padding)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv1 = Dropout(dropout_rate)(conv1)
        
        # 第二个卷积层
        # 添加output_shape参数
        x_padding2 = Lambda(
            lambda x: tf.pad(x, [[0, 0], [padding_size, 0], [0, 0]]),
            output_shape=lambda input_shape: (input_shape[0], input_shape[1] + padding_size, input_shape[2])
        )(conv1)
        conv2 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                      dilation_rate=dilation_rate, padding='valid')(x_padding2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        conv2 = Dropout(dropout_rate)(conv2)
        
        # 如果输入和输出维度不匹配，则使用1x1卷积调整
        if x.shape[-1] != nb_filters:
            x = Conv1D(filters=nb_filters, kernel_size=1, padding='same')(x)
        
        # 添加跳跃连接
        out = Add()([conv2, x])
        return out
    
    # 自注意力机制 - 兼容版本实现
    def self_attention_block(x, units):
        """自注意力机制块，通过学习加权关注序列的不同部分
           优化版本：解决张量类型兼容性问题
        """
        # 查询、键、值变换
        query = Dense(units, name='attention_query')(x)
        key = Dense(units, name='attention_key')(x)
        value = Dense(units, name='attention_value')(x)
        
        # 计算注意力分数 - 使用独立的Keras层
        # 避免直接使用tf.matmul，改用Dot层
        # 修复：将axes=(-1, -1)更改为axes=(2, 1)，正确匹配query和转置后key的维度
        score = Dot(axes=(2, 1))([query, Permute((2, 1))(key)])
        
        # 缩放因子 - 使用Python原生数值，而不是TensorFlow张量
        # 计算 1/sqrt(d_k) 的值作为常量
        scale_factor = 1.0 / (units ** 0.5)
        # 添加output_shape参数
        score = Lambda(
            lambda x: x * scale_factor,
            output_shape=lambda input_shape: input_shape
        )(score)
        
        # 应用softmax获取注意力权重
        attention_weights = Activation('softmax')(score)
        
        # 加权求和
        # 修复：使用正确的axes参数，确保维度匹配
        context = Dot(axes=(2, 1))([attention_weights, value])
        
        # 残差连接 - 如果维度不同，先进行投影
        if int(context.shape[-1]) != int(x.shape[-1]):
            context = Dense(int(x.shape[-1]), name='attention_projection')(context)
        
        # 使用Add合并同类型张量
        out = Add()([x, context])
        
        return out
    
    inputs = Input(shape=input_shape)
    
    # 初始卷积层
    x = Conv1D(filters=nb_filters, kernel_size=1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # TCN层堆栈
    skip_connections = []
    for dilation in dilations:
        x = tcn_block(x, dilation, nb_filters, kernel_size, dropout_rate)
        skip_connections.append(x)
    
    # 合并跳跃连接
    if len(skip_connections) > 1:
        # 使用Lambda进行包装以确保类型兼容性
        x = Lambda(
            lambda tensors: tf.keras.layers.add(tensors),
            output_shape=lambda input_shapes: input_shapes[0]
        )(skip_connections)
    else:
        x = skip_connections[0]
    
    # 应用自注意力机制
    x = self_attention_block(x, attention_units)
    
    # 全局池化
    x = GlobalAveragePooling1D()(x)
    
    # 全连接层
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


# ===================== 13. InceptionTime模型 =====================

def create_inception_time_model(input_shape, output_units=1, nb_filters=32, 
                              use_residual=True, depth=6, kernel_sizes=[1, 3, 5, 8], dropout_rate=0.2):
    """
    InceptionTime模型，通过多尺度卷积和残差连接实现高效时间序列处理
    
    参数:
        input_shape: 输入数据形状
        output_units: 输出单元数量
        nb_filters: 各卷积分支的滤波器数量
        use_residual: 是否使用残差连接
        depth: Inception模块的数量
        kernel_sizes: 不同卷积核大小列表，用于多尺度特征提取
        dropout_rate: Dropout比率
        
    返回:
        编译好的Keras模型
    """
    def _inception_module(input_tensor, stride=1, activation='relu'):
        """Inception模块，包含多个不同卷积核大小的并行卷积分支"""
        # 带有不同内核大小的并行卷积层
        conv_list = []
        
        for kernel_size in kernel_sizes:
            conv_branch = Conv1D(filters=nb_filters, kernel_size=kernel_size, 
                                strides=stride, padding='same', 
                                activation=activation)(input_tensor)
            conv_branch = BatchNormalization()(conv_branch)
            conv_list.append(conv_branch)
        
        # 最大池化分支
        max_pool_branch = MaxPooling1D(pool_size=3, strides=stride, padding='same')(input_tensor)
        max_pool_branch = Conv1D(filters=nb_filters, kernel_size=1, 
                               padding='same', activation=activation)(max_pool_branch)
        max_pool_branch = BatchNormalization()(max_pool_branch)
        conv_list.append(max_pool_branch)
        
        # 合并所有分支
        x = Concatenate(axis=-1)(conv_list)
        x = Activation(activation)(x)
        return x
    
    def _shortcut_layer(input_tensor, out_tensor):
        """残差连接层"""
        shortcut = Conv1D(filters=out_tensor.shape[-1], kernel_size=1, 
                         padding='same')(input_tensor)
        shortcut = BatchNormalization()(shortcut)
        x = Add()([shortcut, out_tensor])
        x = Activation('relu')(x)
        return x
    
    # 构建网络
    inputs = Input(shape=input_shape)
    x = inputs
    
    # 初始卷积
    x = Conv1D(filters=nb_filters, kernel_size=kernel_sizes[-1], 
              padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    # 堆叠Inception模块
    for d in range(depth):
        # Inception模块
        x_inception = _inception_module(x)
        
        # 添加残差连接（每2个模块）
        if use_residual and d % 2 == 1:
            x = _shortcut_layer(x, x_inception)
        else:
            x = x_inception
    
    # 全局池化
    x = GlobalAveragePooling1D()(x)
    
    # 全连接层
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


# ===================== 14. WaveNet模型 =====================

def create_wavenet_model(input_shape, output_units=1, n_filters=32, 
                        n_layers=5, kernel_size=3, dropout_rate=0.2):
    """
    WaveNet模型，使用扩张因果卷积捕获长期时序依赖，适合心率等生理信号预测
    
    参数:
        input_shape: 输入数据形状
        output_units: 输出单元数量
        n_filters: 卷积滤波器数量
        n_layers: 扩张卷积层的数量
        kernel_size: 卷积核大小
        dropout_rate: Dropout比率
        
    返回:
        编译好的Keras模型
    """
    inputs = Input(shape=input_shape)
    
    # 初始卷积层
    x = Conv1D(filters=n_filters, kernel_size=1, padding='same')(inputs)
    
    # 残差输出列表
    skip_connections = []
    
    # 构建扩张卷积层堆栈
    for i in range(n_layers):
        # 扩张率随层数指数增长
        dilation_rate = 2 ** i
        
        # 门控扩张卷积（类似原始WaveNet架构）
        # 滤波器分为两部分：tanh和sigmoid，然后相乘
        tanh_out = Conv1D(
            filters=n_filters, 
            kernel_size=kernel_size,
            padding='causal',
            dilation_rate=dilation_rate,
            activation='tanh'
        )(x)
        
        sigmoid_out = Conv1D(
            filters=n_filters, 
            kernel_size=kernel_size,
            padding='causal',
            dilation_rate=dilation_rate,
            activation='sigmoid'
        )(x)
        
        # 门控机制 - 乘法
        z = tf.keras.layers.Multiply()([tanh_out, sigmoid_out])
        
        # 1x1卷积调整通道数
        z = Conv1D(filters=n_filters, kernel_size=1, padding='same')(z)
        
        # 残差连接
        x = tf.keras.layers.Add()([x, z])
        
        # 跳跃连接
        skip_connections.append(z)
    
    # 合并所有跳跃连接的输出
    if len(skip_connections) > 1:
        out = tf.keras.layers.Add()(skip_connections)
    else:
        out = skip_connections[0]
    
    # 激活
    out = Activation('relu')(out)
    
    # 1x1卷积叠层
    out = Conv1D(filters=n_filters, kernel_size=1, padding='same', activation='relu')(out)
    out = Dropout(dropout_rate)(out)
    out = Conv1D(filters=n_filters, kernel_size=1, padding='same', activation='relu')(out)
    
    # 全局池化
    out = GlobalAveragePooling1D()(out)
    
    # 全连接层
    out = Dense(32, activation='relu')(out)
    out = Dropout(dropout_rate)(out)
    outputs = Dense(output_units, activation='linear')(out)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# ===================== 15. N-BEATS模型 =====================

def create_nbeats_model(input_shape, output_units=1, 
                       theta_dim=16, basis_function_layers=4,
                       stacks=2, blocks_per_stack=3, dropout_rate=0.2):
    """
    N-BEATS模型，使用基函数展开神经网络架构，通过解释性架构分解信号成趋势和周期成分
    
    参数:
        input_shape: 输入数据形状
        output_units: 输出单元数量
        theta_dim: 基函数展开的参数维度
        basis_function_layers: 基函数网络的层数
        stacks: 堆栈数量（通常为2：一个趋势堆栈和一个周期堆栈）
        blocks_per_stack: 每个堆栈中块的数量
        dropout_rate: Dropout比率
        
    返回:
        编译好的Keras模型
    """
    def _create_block(x, units, theta_dim, block_type='generic', block_id=0):
        """创建单个N-BEATS块"""
        # 全连接层堆栈
        for i in range(basis_function_layers - 1):
            x = Dense(units, activation='relu', name=f'{block_type}_dense_{block_id}_{i}')(x)
        
        # 基函数参数（theta）
        theta = Dense(theta_dim, activation='linear', name=f'{block_type}_theta_{block_id}')(x)
        
        # 根据块类型构造回归结果和前向结果
        # 为简化起见，这里使用线性层，但可以根据需要实现更复杂的基函数
        backcast = Dense(input_shape[0], activation='linear', name=f'backcast_{block_type}_{block_id}')(theta)
        forecast = Dense(output_units, activation='linear', name=f'forecast_{block_type}_{block_id}')(theta)
        
        return backcast, forecast
    
    # 验证输入数据形状
    if len(input_shape) != 2:
        # 确保输入形状为(timesteps, features)
        raise ValueError("N-BEATS需要输入形状为(timesteps, features)的数据")
    
    timesteps, n_features = input_shape
    
    # 模型输入
    inputs = Input(shape=input_shape)
    
    # 展平输入以便传递给全连接层
    x = Flatten()(inputs)
    
    # 初始化总体回归和前向预测
    # 使用Keras兼容的方式创建零张量
    zeros_like_timesteps = Lambda(lambda x, ts=timesteps: tf.zeros_like(x[:, 0:1, 0:1])[:, 0, 0:1] * 0.0 + tf.zeros((1, ts)))(inputs)
    zeros_for_forecast = Lambda(lambda x, ou=output_units: tf.zeros_like(x[:, 0:1, 0:1])[:, 0, 0:1] * 0.0 + tf.zeros((1, ou)))(inputs)
    
    backcast = zeros_like_timesteps
    forecast = zeros_for_forecast
    
    # 构建N-BEATS堆栈和块
    block_counter = 0  # 添加全局块计数器
    for stack_id in range(stacks):
        for block_id in range(blocks_per_stack):
            # 块的隐藏单元数，这里简单地使用固定值
            units = 256
            
            # 决定块类型
            if stack_id == 0:
                block_type = 'trend'
            else:
                block_type = 'seasonality'
            
            # 使用残余输入 - 使用Lambda层处理Keras张量
            # 提取输入的第一个特征通道并使用Keras层处理
            input_first_channel = Lambda(lambda x: x[:, :, 0], name=f'extract_channel_{block_counter}')(inputs)
            # 计算残差
            residual_input = Subtract(name=f'residual_subtract_{block_counter}')([input_first_channel, backcast])
            # 重塑为(batch_size, timesteps, 1)
            residual_reshaped = Reshape((timesteps, 1), name=f'residual_reshape_{block_counter}')(residual_input)
            
            # 展平用于全连接层
            block_input = Flatten(name=f'block_flatten_{block_counter}')(residual_reshaped)
            
            # 创建块 - 传递全局块ID确保唯一命名
            block_backcast, block_forecast = _create_block(
                block_input, units, theta_dim, block_type, block_id=block_counter
            )
            
            # 累积回归和前向预测
            backcast = Add(name=f'backcast_add_{block_counter}')([backcast, block_backcast])
            forecast = Add(name=f'forecast_add_{block_counter}')([forecast, block_forecast])
            
            # 更新块计数器
            block_counter += 1
    
    # 构建模型
    model = Model(inputs=inputs, outputs=forecast)
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# ===================== 16. TFT (Temporal Fusion Transformer) 模型 =====================

def create_tft_model(input_shape, output_units=1, hidden_units=64, 
                    lstm_units=32, num_heads=4, dropout_rate=0.2):
    """
    时间融合Transformer (TFT) 模型，专为多变量时间序列预测设计的Transformer变体
    
    参数:
        input_shape: 输入数据形状 (timesteps, features)
        output_units: 输出单元数量
        hidden_units: 隐藏层单元数
        lstm_units: LSTM层单元数
        num_heads: 注意力头数量
        dropout_rate: Dropout比率
        
    返回:
        编译好的Keras模型
    """
    # 输入层
    inputs = Input(shape=input_shape)
    
    # 变量选择网络
    def create_variable_selection_network(x, hidden_units):
        # 对每个变量生成权重
        timesteps, features = x.shape[1:]
        
        # 生成每个时间步的变量权重
        # 首先通过时间分布式Dense层为每个时间步独立生成变量权重
        selection_weights = TimeDistributed(
            Dense(features, activation='softmax')
        )(x)
        
        # 创建加权的特征 - 每个特征乘以其对应的权重
        weighted_features = Multiply()([x, selection_weights])
        
        # 压缩特征维度 - 对特征维度进行加权求和
        selected_features = Lambda(lambda x: tf.reduce_sum(x, axis=-1, keepdims=True))(weighted_features)
        
        return selected_features
    
    # 应用变量选择网络进行特征提取
    processed_inputs = create_variable_selection_network(inputs, hidden_units)
    
    # 使用双向LSTM提取序列信息
    lstm_layer = Bidirectional(LSTM(lstm_units, return_sequences=True))(processed_inputs)
    
    # 静态协变量编码器 (简化实现)
    # 在实际应用中，应该区分静态特征和时间特征
    static_covariates = GlobalAveragePooling1D()(inputs)
    static_context = Dense(hidden_units, activation='relu')(static_covariates)
    
    # 扩展静态上下文以匹配序列长度
    static_context_tiled = RepeatVector(input_shape[0])(static_context)
    
    # 合并LSTM和静态上下文
    combined = Concatenate(axis=-1)([lstm_layer, static_context_tiled])
    
    # 应用自注意力机制
    def apply_multihead_attention(x, num_heads, key_dim):
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim//num_heads
        )(x, x)
        return LayerNormalization(epsilon=1e-6)(x + attention_output)
    
    attended = apply_multihead_attention(combined, num_heads, hidden_units)
    
    # 点式前馈网络
    def point_wise_feed_forward(x, hidden_units):
        # 获取输入的最后一个维度，确保输出与输入形状匹配
        input_dim = x.shape[-1]
        
        # 创建Sequential模型，确保最后的输出维度与输入相同
        ffn = tf.keras.Sequential([
            Dense(hidden_units * 2, activation='relu'),
            Dense(input_dim)  # 输出维度与输入相同，而不是固定的hidden_units
        ])
        
        # 应用前馈网络并添加残差连接
        ffn_output = ffn(x)
        return LayerNormalization(epsilon=1e-6)(x + ffn_output)
    
    outputs_transformer = point_wise_feed_forward(attended, hidden_units)
    
    # 应用门控跳跃网络 (Gated Skip Network)
    # 这是TFT的一个重要组件，允许模型直接连接输入和输出
    def gated_skip_connection(x, lstm_layer):
        gates = Dense(lstm_layer.shape[-1], activation='sigmoid')(x)
        gated_output = Multiply()([gates, lstm_layer])
        return gated_output
    
    skip_connection = gated_skip_connection(static_context_tiled, lstm_layer)
    
    # 形状可能不匹配，需要投影skip_connection到与outputs_transformer相同的维度
    skip_connection_dim = skip_connection.shape[-1]
    transformer_dim = outputs_transformer.shape[-1]
    
    # 如果维度不同，使用Dense层进行投影
    if skip_connection_dim != transformer_dim:
        skip_connection = Dense(transformer_dim, activation='linear', name='skip_projection')(skip_connection)
    
    outputs_with_skip = Add()([outputs_transformer, skip_connection])
    
    # 预测组件
    outputs_final = Dense(hidden_units, activation='relu')(outputs_with_skip)
    outputs_final = Dropout(dropout_rate)(outputs_final)
    
    # 使用最后一个时间步作为预测
    outputs_final = Lambda(lambda x: x[:, -1, :])(outputs_final)
    outputs = Dense(output_units, activation='linear')(outputs_final)
    
    # 构建模型
    model = Model(inputs=inputs, outputs=outputs)
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# ===================== 17. OmniScaleCNN模型 =====================

def create_omniscale_cnn_model(input_shape, output_units=1, channels=64, 
                              depth=3, kernel_range=range(1, 40), dropout_rate=0.2):
    """
    OmniScaleCNN模型，使用可学习参数自动选择最优感受野大小的多尺度CNN
    
    参数:
        input_shape: 输入数据形状
        output_units: 输出单元数量
        channels: 卷积通道数
        depth: OS模块的堆叠深度
        kernel_range: 考虑的卷积核大小范围
        dropout_rate: Dropout比率
        
    返回:
        编译好的Keras模型
    """
    def _create_os_block(x, channels, kernel_range):
        """创建一个OmniScale模块，具有连续多尺度感受野"""
        # 初始1x1卷积减少维度
        x_conv = Conv1D(channels, kernel_size=1, padding='same')(x)
        x_conv = BatchNormalization()(x_conv)
        x_conv = Activation('relu')(x_conv)
        
        # 多尺度卷积层
        # 注：原始论文使用可学习的尺度混合策略，这里简化为固定权重
        conv_outputs = []
        
        for k in kernel_range:
            # 使用不同大小的卷积核
            conv = Conv1D(channels, kernel_size=k, padding='same')(x_conv)
            conv = BatchNormalization()(conv)
            conv = Activation('relu')(conv)
            conv_outputs.append(conv)
        
        # 合并所有尺度的输出
        if len(conv_outputs) > 1:
            x_out = tf.keras.layers.Average()(conv_outputs)
        else:
            x_out = conv_outputs[0]
        
        # 残差连接
        # 如果输入和输出通道不匹配，使用1x1卷积调整
        if x.shape[-1] != channels:
            shortcut = Conv1D(channels, kernel_size=1, padding='same')(x)
            shortcut = BatchNormalization()(shortcut)
        else:
            shortcut = x
        
        out = Add()([x_out, shortcut])
        out = Activation('relu')(out)
        
        return out
    
    inputs = Input(shape=input_shape)
    
    # 初始卷积层
    x = Conv1D(channels, kernel_size=7, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 堆叠OmniScale块
    for i in range(depth):
        # 简化版本：每个层级使用相同数量的通道
        # 原始论文中通道数会随着层级增加而增加
        x = _create_os_block(x, channels, kernel_range)
        
        # 使用步长为2的最大池化来减少序列长度(每隔一层)
        if i < depth - 1 and i % 2 == 0:
            x = MaxPooling1D(pool_size=2)(x)
    
    # 全局池化
    x = GlobalAveragePooling1D()(x)
    
    # 全连接层
    x = Dense(channels, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(output_units, activation='linear')(x)
    
    # 构建模型
    model = Model(inputs=inputs, outputs=outputs)
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# ===================== 18. Deep State Space Model =====================

def create_deep_state_space_model(input_shape, output_units=1, state_dim=32, 
                                rnn_units=64, emission_layers=2, dropout_rate=0.2):
    """
    深度状态空间模型，将传统状态空间模型与深度学习结合，适合建模有潜在状态的动态系统
    
    参数:
        input_shape: 输入数据形状
        output_units: 输出单元数量
        state_dim: 潜在状态维度
        rnn_units: RNN层的单元数
        emission_layers: 发射网络的层数
        dropout_rate: Dropout比率
        
    返回:
        编译好的Keras模型
    """
    inputs = Input(shape=input_shape)
    
    # 转换网络 - 将观测转换为潜在状态
    def create_transformation_network(x, state_dim):
        """将输入观测转换为潜在状态表示"""
        # 特征提取
        x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        
        x = Conv1D(32, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        
        # 生成初始状态
        x = GRU(state_dim, return_sequences=True)(x)
        
        return x
    
    # 创建转换网络获取潜在状态序列
    states = create_transformation_network(inputs, state_dim)
    
    # 状态转移网络 - 建模状态的动态演化
    def state_transition_network(states):
        """建模状态的动态演化"""
        # 双向LSTM捕获状态转移的前向和后向依赖
        x = Bidirectional(LSTM(rnn_units, return_sequences=True))(states)
        x = Dropout(dropout_rate)(x)
        
        # 残差连接，确保梯度流动
        if states.shape[-1] != x.shape[-1]:
            states_transformed = Conv1D(x.shape[-1], kernel_size=1, padding='same')(states)
            states = states_transformed
            
        x = Add()([x, states])
        x = LayerNormalization()(x)
        
        return x
    
    # 应用状态转移网络
    dynamic_states = state_transition_network(states)
    
    # 发射网络 - 从状态生成观测（预测）
    def emission_network(states, layers, output_units):
        """从状态生成最终预测"""
        x = states
        
        # 多层处理
        for i in range(layers - 1):
            x = Dense(64, activation='relu')(x)
            x = Dropout(dropout_rate)(x)
            x = LayerNormalization()(x)
        
        # 取最后一个时间步
        x = Lambda(lambda x: x[:, -1, :])(x)
        
        # 生成最终输出
        predictions = Dense(output_units, activation='linear')(x)
        
        return predictions
    
    # 生成最终预测
    outputs = emission_network(dynamic_states, emission_layers, output_units)
    
    # 构建模型
    model = Model(inputs=inputs, outputs=outputs)
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# ===================== 19. LSTNet模型 =====================

def create_lstnet_model(input_shape, output_units=1, conv_filters=64, 
                       conv_kernel=3, lstm_units=64, recurrent_skip=10, 
                       skip_lstm_units=32, ar_window=3, dropout_rate=0.2):
    """
    LSTNet (Long- and Short-term Time-series Network) 模型，
    专为多变量时间序列预测设计，结合CNN、RNN和跳跃循环连接
    
    参数:
        input_shape: 输入数据形状 (timesteps, features)
        output_units: 输出单元数量
        conv_filters: 卷积滤波器数量
        conv_kernel: 卷积核大小
        lstm_units: LSTM层的单元数
        recurrent_skip: 跳跃连接的长度
        skip_lstm_units: 跳跃LSTM的单元数
        ar_window: 自回归窗口大小
        dropout_rate: Dropout比率
        
    返回:
        编译好的Keras模型
    """
    inputs = Input(shape=input_shape)
    
    # CNN层 - 提取短期模式
    conv = Conv1D(
        filters=conv_filters,
        kernel_size=conv_kernel,
        padding='same',
        activation='relu'
    )(inputs)
    conv = Dropout(dropout_rate)(conv)
    
    # LSTM层 - 捕获长期依赖关系
    lstm_out = LSTM(lstm_units, return_sequences=False)(conv)
    lstm_out = Dropout(dropout_rate)(lstm_out)
    
    # 跳跃循环连接 - 捕获超长期依赖
    if recurrent_skip > 0:
        # 重新整形以创建跳过窗口的序列
        skip_input = create_skip_connection_input(conv, recurrent_skip)
        
        # 使用LSTM处理跳跃窗口
        skip_lstm = LSTM(skip_lstm_units, return_sequences=False)(skip_input)
        skip_lstm = Dropout(dropout_rate)(skip_lstm)
        
        # 连接LSTM和跳跃LSTM输出
        lstm_out = Concatenate()([lstm_out, skip_lstm])
    
    # 自回归组件 - 使用全连接层处理原始特征
    if ar_window > 0:
        # 提取最后AR窗口的时间步
        ar_input = Lambda(
            lambda x: x[:, -ar_window:, :],
            output_shape=(ar_window, input_shape[1])
        )(inputs)
        
        # 展平用于线性层
        ar_input = Flatten()(ar_input)
        ar_output = Dense(output_units, activation='linear')(ar_input)
        
        # 综合预测 - 线性组合LSTM和AR结果
        lstm_output = Dense(output_units, activation='linear')(lstm_out)
        outputs = Add()([lstm_output, ar_output])
    else:
        # 没有AR组件时直接使用LSTM输出
        outputs = Dense(output_units, activation='linear')(lstm_out)
    
    # 构建模型
    model = Model(inputs=inputs, outputs=outputs)
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# 辅助函数：创建跳跃连接输入
def create_skip_connection_input(x, skip_period):
    """
    创建用于跳跃LSTM的输入序列
    
    参数:
        x: 输入张量，形状为 (batch_size, timesteps, features)
        skip_period: 跳跃周期
        
    返回:
        跳跃输入张量
    """
    def skip_connection_fn(x):
        # 获取原始形状
        batch_size = tf.shape(x)[0]
        timesteps = tf.shape(x)[1]
        features = tf.shape(x)[2]
        
        # 计算跳跃序列长度
        skip_seq_len = timesteps // skip_period
        
        # 创建索引
        indices = tf.range(skip_seq_len * skip_period, delta=skip_period)
        
        # 收集跳过的时间步
        skipped_steps = tf.gather(x, indices, axis=1)
        
        return skipped_steps
    
    # 使用Lambda层包装函数
    return Lambda(skip_connection_fn)(x)


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
        训练历史记录和训练详细信息字典
    """
    # 创建必要的目录
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 创建临时目录用于计算模型大小
    temp_model_dir = os.path.join(checkpoint_dir, "temp")
    os.makedirs(temp_model_dir, exist_ok=True)
    
    # 记录模型参数量
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable_params = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    # 获取模型层数
    n_layers = len(model.layers)
    
    # 设置回调函数
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=100,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=20,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f"{model_name}_best.keras"),
            monitor='val_loss' if X_val is not None else 'loss',
            save_best_only=True,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(tensorboard_dir, model_name),
            histogram_freq=1
        )
    ]
    
    # 记录开始时间
    import time
    start_time = time.time()
    
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
    
    # 记录训练结束时间和总训练时间
    end_time = time.time()
    training_time = end_time - start_time
    
    # 计算每个epoch的平均时间
    actual_epochs = len(history.history['loss'])
    avg_epoch_time = training_time / actual_epochs
    
    # 保存最终模型
    model.save(os.path.join(checkpoint_dir, f"{model_name}_final.keras"))
    
    # 获取最终学习率
    try:
        # 新版TensorFlow使用learning_rate属性
        if hasattr(model.optimizer, 'learning_rate'):
            final_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
        # 旧版TensorFlow使用lr属性
        elif hasattr(model.optimizer, 'lr'):
            final_lr = float(tf.keras.backend.get_value(model.optimizer.lr))
        # 备选方案
        elif hasattr(model.optimizer, '_decayed_lr'):
            final_lr = float(model.optimizer._decayed_lr(tf.float32).numpy())
        else:
            final_lr = 0.0  # 无法获取时设置默认值
    except:
        final_lr = 0.0  # 捕获任何可能的异常并提供默认值
    
    # 获取最佳验证损失
    if 'val_loss' in history.history:
        best_val_loss = min(history.history['val_loss'])
    else:
        best_val_loss = min(history.history['loss'])
    
    # 计算模型大小（MB）- 使用自定义目录避免权限问题
    try:
        temp_model_path = os.path.join(temp_model_dir, f"{model_name}_temp.keras")
        model.save(temp_model_path)
        if os.path.exists(temp_model_path):
            model_size_mb = os.path.getsize(temp_model_path) / (1024 * 1024)
            # 清理临时文件
            try:
                os.remove(temp_model_path)
            except:
                pass  # 忽略删除失败
        else:
            # 备用方法：基于参数量估计模型大小
            # 假设每个参数占用4字节(float32)
            model_size_mb = (total_params * 4) / (1024 * 1024)
    except Exception as e:
        print(f"估算模型大小时出错: {e}")
        # 备用方法：基于参数量估计模型大小
        model_size_mb = (total_params * 4) / (1024 * 1024)
    
    # 收集训练详细信息
    training_details = {
        'model_name': model_name,
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'non_trainable_params': int(non_trainable_params),
        'n_layers': n_layers,
        'training_time_seconds': training_time,
        'avg_epoch_time_seconds': avg_epoch_time,
        'actual_epochs': actual_epochs,
        'final_learning_rate': final_lr,
        'best_val_loss': best_val_loss,
        'batch_size': batch_size,
        'model_size_mb': model_size_mb,
        'train_samples': len(X_train) if isinstance(X_train, np.ndarray) else len(X_train[0]),
        'input_shape': str(model.input_shape)
    }
    
    # 清理临时目录
    try:
        import shutil
        shutil.rmtree(temp_model_dir, ignore_errors=True)
    except:
        pass  # 忽略清理失败
    
    return history, training_details


def evaluate_model(model, X_test, y_test, plot_results=True, model_name="model", measure_inference_time=True):
    """
    评估模型性能并可选地可视化结果
    
    参数:
        model: 要评估的模型
        X_test: 测试特征
        y_test: 测试标签
        plot_results: 是否绘制预测vs真实值图表
        model_name: 模型名称（用于图表标题）
        measure_inference_time: 是否测量推理时间
        
    返回:
        评估指标字典
    """
    # 设置中文字体显示 - 在评估开始前就设置好字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']  # 中文字体设置
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
    matplotlib.rcParams['font.family'] = 'sans-serif'  # 设置默认字体
    
    # 在测试集上评估模型
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集MAE: {test_mae:.4f} BPM")
    
    # 测量推理时间（如果需要）
    if measure_inference_time:
        import time
        # 进行一次预热预测，确保模型加载到GPU
        _ = model.predict(X_test[:1])
        
        # 测量单样本推理时间
        start_time = time.time()
        n_repeats = 100  # 重复100次以获得更稳定的时间
        for _ in range(n_repeats):
            _ = model.predict(X_test[:1], verbose=0)
        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000 / n_repeats
        
        # 测量批量推理时间（整个测试集）
        batch_start_time = time.time()
        _ = model.predict(X_test, verbose=0)
        batch_end_time = time.time()
        batch_inference_time = batch_end_time - batch_start_time
    
    # 预测测试集
    y_pred = model.predict(X_test)
    
    # 计算额外的指标
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
    import scipy.stats as stats
    
    # 基本指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # 转换为百分比
    
    # 额外指标
    errors = y_pred.flatten() - y_test
    abs_errors = np.abs(errors)
    max_error = np.max(abs_errors)
    median_error = np.median(abs_errors)
    
    # 计算95%置信区间
    ci_95_lower = np.percentile(errors, 2.5)
    ci_95_upper = np.percentile(errors, 97.5)
    
    # 计算标准偏差
    error_std = np.std(errors)
    
    # 计算分类准确率（如果预测在±3 BPM内算准确）
    tolerance = 3.0  # BPM
    accuracy_3bpm = np.mean(abs_errors <= tolerance) * 100  # 百分比
    
    # 计算不同心率区间的性能
    low_hr_mask = y_test < 70
    normal_hr_mask = (y_test >= 70) & (y_test < 90)
    high_hr_mask = y_test >= 90
    
    # 只有在每个区间有足够样本时才计算
    mae_low_hr = np.mean(abs_errors[low_hr_mask]) if np.sum(low_hr_mask) > 5 else np.nan
    mae_normal_hr = np.mean(abs_errors[normal_hr_mask]) if np.sum(normal_hr_mask) > 5 else np.nan
    mae_high_hr = np.mean(abs_errors[high_hr_mask]) if np.sum(high_hr_mask) > 5 else np.nan
    
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f} BPM")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"最大误差: {max_error:.4f} BPM")
    print(f"95% 置信区间: [{ci_95_lower:.2f}, {ci_95_upper:.2f}] BPM")
    print(f"±{tolerance} BPM准确率: {accuracy_3bpm:.2f}%")
    
    if measure_inference_time:
        print(f"单样本推理时间: {inference_time_ms:.2f} 毫秒")
        print(f"批量推理时间: {batch_inference_time:.4f} 秒 (测试集: {len(y_test)} 样本)")
    
    # 可视化结果
    if plot_results:
        # 确保结果目录存在
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        
        # 使用特殊符号的安全方式
        r2_text = f"R\u00B2: {r2:.3f}"  # 使用Unicode字符U+00B2表示²
        
        # 绘制散点图 - 预测值vs真实值
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        
        # 绘制理想预测线 (y=x)
        min_val = min(np.min(y_test), np.min(y_pred))
        max_val = max(np.max(y_test), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # 添加±3 BPM线
        plt.plot([min_val, max_val], [min_val + tolerance, max_val + tolerance], 'g--', alpha=0.5)
        plt.plot([min_val, max_val], [min_val - tolerance, max_val - tolerance], 'g--', alpha=0.5)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('实际心率 (BPM)', fontsize=12)
        plt.ylabel('预测心率 (BPM)', fontsize=12)
        plt.title(f'{model_name} 模型心率预测结果\nMAE: {test_mae:.2f} BPM, {r2_text}, ±3BPM准确率: {accuracy_3bpm:.1f}%', fontsize=14)
        plt.tight_layout()
        # 保存预测散点图
        plt.savefig(os.path.join(results_dir, f'{model_name.replace(" ", "_")}_scatter.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图表，防止显示
        
        # 添加误差分布直方图
        plt.figure(figsize=(10, 6))
        errors = y_pred.flatten() - y_test
        plt.hist(errors, bins=20, alpha=0.7, color='skyblue')
        plt.axvline(0, color='r', linestyle='--')
        plt.axvline(ci_95_lower, color='g', linestyle='--', alpha=0.7)
        plt.axvline(ci_95_upper, color='g', linestyle='--', alpha=0.7)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('预测误差 (BPM)', fontsize=12)
        plt.ylabel('频率', fontsize=12)
        plt.title(f'{model_name} 预测误差分布\n95%置信区间: [{ci_95_lower:.2f}, {ci_95_upper:.2f}] BPM', fontsize=14)
        plt.tight_layout()
        # 保存误差直方图
        plt.savefig(os.path.join(results_dir, f'{model_name.replace(" ", "_")}_error_hist.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图表，防止显示
    
    # 返回评估指标
    metrics = {
        'mae': test_mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'max_error': max_error,
        'median_error': median_error,
        'error_std': error_std,
        'ci_95_lower': ci_95_lower,
        'ci_95_upper': ci_95_upper,
        'accuracy_3bpm': accuracy_3bpm,
        'mae_low_hr': mae_low_hr,
        'mae_normal_hr': mae_normal_hr,
        'mae_high_hr': mae_high_hr
    }
    
    # 添加推理时间指标（如果测量了）
    if measure_inference_time:
        metrics['inference_time_ms'] = inference_time_ms
        metrics['batch_inference_time'] = batch_inference_time
        metrics['samples_per_second'] = len(y_test) / batch_inference_time
    
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
    history, training_details = train_model(
        model, decomp_train, y_train,
        batch_size=32, epochs=50,
        model_name="CNN_GRU_EEMD"
    )
    
    # 评估模型
    metrics = evaluate_model(model, decomp_test, y_test, model_name="CNN+GRU with EEMD")
    
    return model, history, metrics, training_details


if __name__ == "__main__":
    # 运行示例代码
    example_usage() 