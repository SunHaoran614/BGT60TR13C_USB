import sys
import os
import numpy as np
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QSlider, QFileDialog, 
                            QLabel, QSpinBox, QComboBox, QGroupBox, QCheckBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib as mpl
import radar_settings  # 导入雷达设置模块

# 设置matplotlib支持中文显示
# 尝试设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 验证中文字体是否可用 - 兼容新旧版本 matplotlib
    # 尝试使用新方法
    try:
        # 新版本 matplotlib
        import matplotlib.font_manager as fm
        if hasattr(fm, 'fontManager'):
            fm.fontManager.findfont('SimHei', rebuild_if_missing=True)
        elif hasattr(fm, '_rebuild'):
            # 旧版本 matplotlib
            mpl.font_manager._rebuild()
        else:
            # 防止找不到任何方法的情况
            pass
    except Exception as e:
        print(f"字体缓存重建时出错: {e}")
    
    # 检查字体是否可用
    zh_font_available = False
    fonts = [f.name for f in mpl.font_manager.fontManager.ttflist]
    for font in plt.rcParams['font.sans-serif']:
        if font in fonts:
            zh_font_available = True
            break
    
    if not zh_font_available:
        # 如果没有中文字体，则使用英文标签
        print("警告: 未找到支持中文的字体，将使用英文标签")
except Exception as e:
    print(f"设置中文字体时出错: {e}")

# 添加均值相消法MTI函数
def MTI_avg(fft_data: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    静态杂波滤除:MTI(Moving Targets Indication 动目标显示)均值相消法
    :param fft_data: 原始数据
    :param axis: axis 0 时计算每列的均值，进行均值相消
    :return: 返回降噪后的数据
    """
    fft_data_denoise = fft_data.copy()  # 深拷贝数组，否则传递的是指针
    return fft_data_denoise - np.mean(fft_data_denoise, axis=axis)

class RadarReplayApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("雷达数据Range-FFT回放")
        self.setMinimumSize(1000, 600)
        
        # 雷达数据
        self.radar_data = None
        self.current_frame = 0
        self.frame_count = 0
        self.play_speed = 30  # 默认帧率30fps
        self.is_playing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        
        # 雷达参数
        self.radar_params = self.load_default_radar_params()
        
        # 设置UI
        self.setup_ui()
    
    def load_default_radar_params(self):
        """加载默认雷达参数"""
        try:
            # 直接从雷达设置模块中获取参数
            return radar_settings.get_radar_params()
        except Exception as e:
            print(f"警告: 从雷达设置模块加载参数失败: {e}")
            # 使用RADAR_PARAMS中的默认值
            return radar_settings.RADAR_PARAMS
    
    def setup_ui(self):
        """设置用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # 创建绘图区域
        self.setup_plot_area(main_layout)
        
        # 创建控制区域
        self.setup_control_panel(main_layout)
        
        # 创建状态栏
        self.statusBar().showMessage("就绪。请加载雷达数据文件。")
    
    def setup_plot_area(self, parent_layout):
        """设置绘图区域"""
        # 创建matplotlib图形
        self.figure = Figure(figsize=(10, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        # 创建两个子图，一个用于原始数据，一个用于MTI数据
        self.ax1 = self.figure.add_subplot(211)  # 上半部分显示原始数据
        self.ax1.set_title('原始雷达回波Range-FFT')
        self.ax1.set_xlabel('距离 (m)')
        self.ax1.set_ylabel('功率 (dB)')
        self.ax1.grid(True)
        
        self.ax2 = self.figure.add_subplot(212)  # 下半部分显示MTI数据
        self.ax2.set_title('MTI处理后雷达回波Range-FFT')
        self.ax2.set_xlabel('距离 (m)')
        self.ax2.set_ylabel('功率 (dB)')
        self.ax2.grid(True)
        
        # 调整子图间距
        self.figure.tight_layout(pad=3.0)
        
        # 添加到布局
        parent_layout.addWidget(self.canvas, 1)
    
    def setup_control_panel(self, parent_layout):
        """设置控制面板"""
        control_layout = QHBoxLayout()
        
        # 文件控制组
        file_group = QGroupBox("文件控制")
        file_layout = QHBoxLayout(file_group)
        
        self.load_btn = QPushButton("加载数据")
        self.load_btn.clicked.connect(self.load_radar_data)
        file_layout.addWidget(self.load_btn)
        
        control_layout.addWidget(file_group)
        
        # 播放控制组
        playback_group = QGroupBox("播放控制")
        playback_layout = QHBoxLayout(playback_group)
        
        self.play_btn = QPushButton("播放")
        self.play_btn.clicked.connect(self.toggle_playback)
        self.play_btn.setEnabled(False)
        playback_layout.addWidget(self.play_btn)
        
        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self.reset_playback)
        self.reset_btn.setEnabled(False)
        playback_layout.addWidget(self.reset_btn)
        
        playback_layout.addWidget(QLabel("帧率:"))
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 120)
        self.fps_spinbox.setValue(30)
        self.fps_spinbox.valueChanged.connect(self.update_playback_speed)
        playback_layout.addWidget(self.fps_spinbox)
        
        control_layout.addWidget(playback_group)
        
        # 帧控制组
        frame_group = QGroupBox("帧控制")
        frame_layout = QHBoxLayout(frame_group)
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.frame_slider_changed)
        frame_layout.addWidget(self.frame_slider)
        
        self.frame_label = QLabel("0/0")
        frame_layout.addWidget(self.frame_label)
        
        control_layout.addWidget(frame_group)
        
        # 显示设置组
        display_group = QGroupBox("显示设置")
        display_layout = QHBoxLayout(display_group)
        
        display_layout.addWidget(QLabel("天线:"))
        self.antenna_combo = QComboBox()
        self.antenna_combo.addItem("所有天线")
        self.antenna_combo.currentIndexChanged.connect(self.update_plot)
        display_layout.addWidget(self.antenna_combo)
        
        display_layout.addWidget(QLabel("窗函数:"))
        self.window_combo = QComboBox()
        self.window_combo.addItems(["hanning", "hamming", "blackman", "rectangular"])
        self.window_combo.currentTextChanged.connect(self.update_plot)
        display_layout.addWidget(self.window_combo)
        
        # 添加MTI帧数设置
        display_layout.addWidget(QLabel("MTI帧数:"))
        self.mti_frames_spinbox = QSpinBox()
        self.mti_frames_spinbox.setRange(2, 50)  # 最少需要2帧，最多50帧
        self.mti_frames_spinbox.setValue(30)     # 默认使用30帧
        self.mti_frames_spinbox.valueChanged.connect(self.update_plot)
        display_layout.addWidget(self.mti_frames_spinbox)
        
        # 添加是否显示MTI复选框
        self.show_mti_checkbox = QCheckBox("显示MTI")
        self.show_mti_checkbox.setChecked(True)  # 默认显示MTI
        self.show_mti_checkbox.stateChanged.connect(self.toggle_mti_display)
        display_layout.addWidget(self.show_mti_checkbox)
        
        # 添加X轴范围设置
        display_layout.addWidget(QLabel("X轴范围(m):"))
        self.x_max_spinbox = QDoubleSpinBox()
        self.x_max_spinbox.setRange(0.1, 30.0)  # 范围从0.1米到30米
        self.x_max_spinbox.setValue(7.0)  # 默认值设置为7米
        self.x_max_spinbox.setSingleStep(0.5)  # 步进值为0.5米
        self.x_max_spinbox.setDecimals(1)  # 显示一位小数
        self.x_max_spinbox.valueChanged.connect(self.update_plot)
        display_layout.addWidget(self.x_max_spinbox)
        
        # 添加Y轴上限设置
        display_layout.addWidget(QLabel("Y轴上限:"))
        self.y_max_spinbox = QSpinBox()
        self.y_max_spinbox.setRange(10, 10000)  # 将最大值从1000增加到10000
        self.y_max_spinbox.setValue(500)  # 默认值调整为更合适的500
        self.y_max_spinbox.setSingleStep(100)  # 步进值增加到100方便调整
        self.y_max_spinbox.valueChanged.connect(self.update_plot)
        display_layout.addWidget(self.y_max_spinbox)
        
        control_layout.addWidget(display_group)
        
        parent_layout.addLayout(control_layout)
    
    def load_radar_data(self):
        """加载雷达数据文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择雷达数据文件", "", "NumPy Files (*.npy)")
        
        if file_path:
            try:
                self.statusBar().showMessage(f"正在加载文件: {file_path}...")
                self.radar_data = np.load(file_path)
                
                # 检查数据维度
                if len(self.radar_data.shape) != 4:
                    raise ValueError(f"数据维度不正确，应为4维: {self.radar_data.shape}")
                
                self.frame_count = self.radar_data.shape[0]
                self.current_frame = 0
                
                # 更新天线下拉框
                antenna_count = self.radar_data.shape[1]
                self.antenna_combo.clear()
                self.antenna_combo.addItem("所有天线")
                for i in range(antenna_count):
                    self.antenna_combo.addItem(f"天线 {i+1}")
                
                # 更新UI元素
                self.frame_slider.setRange(0, self.frame_count - 1)
                self.frame_slider.setValue(0)
                self.frame_slider.setEnabled(True)
                self.play_btn.setEnabled(True)
                self.reset_btn.setEnabled(True)
                
                self.update_frame_label()
                self.update_plot()
                
                # 打印数据信息和雷达参数
                print(f"Data shape: {self.radar_data.shape}")
                print(f"Max range: {self.radar_params['max_range']} m")
                
                self.statusBar().showMessage(f"已加载数据: {self.frame_count}帧, {antenna_count}个天线")
            except Exception as e:
                self.statusBar().showMessage(f"错误: 无法加载数据 - {str(e)}")
                print(f"错误: {e}")
    
    def toggle_playback(self):
        """切换播放/暂停状态"""
        if not self.radar_data is None:
            if self.is_playing:
                self.timer.stop()
                self.play_btn.setText("播放")
                self.is_playing = False
            else:
                # 如果已经到最后一帧，从头开始
                if self.current_frame >= self.frame_count - 1:
                    self.current_frame = 0
                    self.frame_slider.setValue(0)
                
                self.timer.start(1000 // self.play_speed)
                self.play_btn.setText("暂停")
                self.is_playing = True
    
    def reset_playback(self):
        """重置回放到第一帧"""
        if self.is_playing:
            self.timer.stop()
            self.play_btn.setText("播放")
            self.is_playing = False
        
        self.current_frame = 0
        self.frame_slider.setValue(0)
        self.update_plot()
    
    def update_playback_speed(self):
        """更新播放速度"""
        self.play_speed = self.fps_spinbox.value()
        if self.is_playing:
            self.timer.stop()
            self.timer.start(1000 // self.play_speed)
    
    def frame_slider_changed(self):
        """帧滑块值改变事件"""
        self.current_frame = self.frame_slider.value()
        self.update_frame_label()
        self.update_plot()
    
    def update_frame_label(self):
        """更新帧标签显示"""
        self.frame_label.setText(f"{self.current_frame + 1}/{self.frame_count}")
    
    def toggle_mti_display(self):
        """切换MTI显示状态"""
        if self.show_mti_checkbox.isChecked():
            # 显示MTI子图
            self.ax2.set_visible(True)
        else:
            # 隐藏MTI子图
            self.ax2.set_visible(False)
        
        # 调整图形布局
        self.figure.tight_layout(pad=3.0)
        self.canvas.draw()
        
        # 更新图形
        self.update_plot()
    
    def update_plot(self):
        """更新绘图"""
        if self.radar_data is None:
            return
        
        # 如果正在播放，推进到下一帧
        if self.is_playing and self.sender() == self.timer:
            self.current_frame += 1
            if self.current_frame >= self.frame_count:
                self.current_frame = 0
            
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.current_frame)
            self.frame_slider.blockSignals(False)
            self.update_frame_label()
        
        # 清除当前图形
        self.ax1.clear()
        self.ax2.clear()
        
        # 获取当前帧数据
        frame_data = self.radar_data[self.current_frame:self.current_frame+1]
        
        # 执行Range-FFT
        window_type = self.window_combo.currentText()
        range_fft_data = self.calculate_range_fft(frame_data, window=window_type)
        
        # 执行均值相消法MTI
        # 获取用户设置的帧数
        max_frames = self.mti_frames_spinbox.value()
        # 根据可用帧数决定MTI处理范围
        start_frame = max(0, self.current_frame - max_frames // 2)
        end_frame = min(self.frame_count, start_frame + max_frames)
        
        # 确保至少有2帧进行均值相消
        if end_frame - start_frame < 2:
            if start_frame > 0:
                start_frame -= 1
            elif end_frame < self.frame_count:
                end_frame += 1
        
        # 读取多帧数据计算均值相消
        mti_input_data = self.radar_data[start_frame:end_frame]
        
        # 如果只有一帧，无法执行MTI
        if len(mti_input_data) <= 1:
            self.statusBar().showMessage("警告: 数据帧数不足，无法执行MTI均值相消")
            mti_range_fft_data = range_fft_data.copy()
        else:
            # 执行Range-FFT
            mti_input_range_fft = self.calculate_range_fft(mti_input_data, window=window_type)
            
            # 使用均值相消法MTI
            mti_range_fft_data = MTI_avg(mti_input_range_fft, axis=0)
            
            # 如果当前帧在处理范围内，选择对应帧的MTI结果
            frame_index = self.current_frame - start_frame
            if 0 <= frame_index < len(mti_range_fft_data):
                mti_range_fft_data = mti_range_fft_data[frame_index:frame_index+1]
            else:
                # 否则使用最后一帧的结果
                mti_range_fft_data = mti_range_fft_data[-1:]
                
            self.statusBar().showMessage(f"使用均值相消法MTI，帧范围: {start_frame+1}-{end_frame}，总计{end_frame-start_frame}帧")
        
        # 获取选中的天线索引
        antenna_index = self.antenna_combo.currentIndex() - 1  # -1是因为第一项是"所有天线"
        
        # 计算功率谱
        if antenna_index < 0:
            # 如果选择"所有天线"，计算所有天线的平均值
            range_profile = np.mean(np.abs(range_fft_data[0]), axis=(0, 1))
            title_suffix = f'(所有天线)'
            
            # 也计算MTI后的功率谱
            if len(mti_range_fft_data) > 0:
                mti_range_profile = np.mean(np.abs(mti_range_fft_data[-1]), axis=(0, 1))
            else:
                mti_range_profile = range_profile.copy()  # 如果MTI没有结果，使用原始数据
        else:
            # 使用选定的天线，仍然需要在chirp维度上平均
            range_profile = np.mean(np.abs(range_fft_data[0, antenna_index]), axis=0)
            title_suffix = f'(天线 {antenna_index+1})'
            
            # 也计算MTI后的功率谱
            if len(mti_range_fft_data) > 0:
                mti_range_profile = np.mean(np.abs(mti_range_fft_data[-1, antenna_index]), axis=0)
            else:
                mti_range_profile = range_profile.copy()  # 如果MTI没有结果，使用原始数据
        
        # 只显示一半的FFT（由于对称性）
        half_len = len(range_profile) // 2
        
        # 计算距离轴
        range_axis = self.calculate_range_axis(half_len)
        
        # 使用幅度值绘制原始数据
        magnitude = range_profile[:half_len]
        self.ax1.plot(range_axis, magnitude)
        self.ax1.set_title(f'原始Range-FFT {title_suffix} (帧 {self.current_frame + 1}/{self.frame_count})')
        self.ax1.set_xlabel('距离 (m)')
        self.ax1.set_ylabel('幅度 (Magnitude)')
        self.ax1.grid(True)
        
        # 使用幅度值绘制MTI数据
        mti_magnitude = mti_range_profile[:half_len]
        self.ax2.plot(range_axis, mti_magnitude)
        self.ax2.set_title(f'MTI处理后Range-FFT {title_suffix} (均值相消法)')
        self.ax2.set_xlabel('距离 (m)')
        self.ax2.set_ylabel('幅度 (Magnitude)')
        self.ax2.grid(True)
        
        # 获取用户设置的Y轴上限和X轴范围
        y_max = self.y_max_spinbox.value()
        user_x_max = self.x_max_spinbox.value()
        
        # 计算实际可见距离范围（基于用户设置和实际计算出的最大距离）
        max_calculated_range = np.max(range_axis)  # 基于FFT点数和距离分辨率计算出的最大距离
        
        # 设置两个子图的x轴和y轴范围一致
        self.ax1.set_xlim(0, user_x_max)  # X轴范围为0到用户设置的最大距离
        self.ax1.set_ylim(0, y_max)  # Y轴使用用户设置的上限值
        
        self.ax2.set_xlim(0, user_x_max)  # X轴范围为0到用户设置的最大距离
        self.ax2.set_ylim(0, y_max)  # Y轴使用用户设置的上限值
        
        # 添加文本注释显示理论最大距离和实际计算距离
        max_range = self.radar_params['max_range']
        range_resolution = self.radar_params['range_resolution']
        self.ax1.text(0.98, 0.95, 
                 f'理论最大距离: {max_range:.2f} m\n'
                 f'计算最大距离: {max_calculated_range:.2f} m\n'
                 f'距离分辨率: {range_resolution:.3f} m',
                 transform=self.ax1.transAxes, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 根据复选框状态显示或隐藏MTI子图
        self.ax2.set_visible(self.show_mti_checkbox.isChecked())
        
        # 调整图形布局
        self.figure.tight_layout(pad=3.0)
        
        # 更新canvas
        self.canvas.draw()
    
    def calculate_range_fft(self, data, window='hanning'):
        """计算Range-FFT"""
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
            win = np.hanning(samples)
        
        # 准备输出数组
        range_fft_data = np.zeros((frames, antennas, chirps, samples), dtype=complex)
        
        # 应用窗函数并执行FFT
        for f in range(frames):
            for a in range(antennas):
                for c in range(chirps):
                    windowed_data = data[f, a, c, :] * win
                    range_fft_data[f, a, c, :] = np.fft.fft(windowed_data)
        
        return range_fft_data
    
    def calculate_range_axis(self, num_bins):
        """计算距离轴"""
        # 使用正确的雷达理论公式计算距离轴
        # 距离轴 = 频率索引 * 距离分辨率
        # 或者: 距离 = (频率索引 * 采样率) / (2 * FFT长度 * 调频斜率)
        
        # 获取雷达参数
        range_resolution = self.radar_params['range_resolution']  # 距离分辨率 = c/(2*带宽)
        samples = self.radar_params['num_samples']  # FFT大小
        
        # 计算实际距离轴（频率索引 * 距离分辨率）
        # 只使用前半部分FFT对应有效距离
        range_axis = np.arange(num_bins) * range_resolution
        
        return range_axis

def main():
    app = QApplication(sys.argv)
    window = RadarReplayApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 