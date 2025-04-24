import numpy as np
import os
import argparse
import datetime
import re
import json
from pathlib import Path
import radar_settings  # 导入雷达设置模块

def extract_timestamp(folder_name):
    """从文件夹名称中提取时间戳"""
    pattern = r'BGT60TR13C_record_(\d{8}-\d{6})_'
    match = re.search(pattern, folder_name)
    if match:
        timestamp_str = match.group(1)
        # 格式: 'YYYYMMDD-HHMMSS'
        # 返回naive datetime对象（不带时区信息）
        return datetime.datetime.strptime(timestamp_str, '%Y%m%d-%H%M%S')
    return None

def get_radar_npy_path(folder_path):
    """获取雷达数据文件的路径"""
    # 检查是否有RadarIfxAvian_00子文件夹
    radar_folder = os.path.join(folder_path, 'RadarIfxAvian_00')
    if os.path.isdir(radar_folder):
        radar_file = os.path.join(radar_folder, 'radar.npy')
        if os.path.isfile(radar_file):
            return radar_file
    
    # 如果没有找到特定结构，就搜索整个文件夹中的.npy文件
    npy_files = list(Path(folder_path).glob('**/*.npy'))
    if npy_files:
        return str(npy_files[0])
    
    return None

def load_radar_data(folder_path):
    """加载文件夹中的雷达数据"""
    radar_file = get_radar_npy_path(folder_path)
    if not radar_file:
        print(f"错误: 在{folder_path}中未找到radar.npy文件")
        return None
    
    print(f"加载数据文件: {radar_file}")
    return np.load(radar_file)

def load_config_file(folder_path):
    """加载配置文件，获取帧率信息"""
    # 首先尝试从radar_settings模块获取参数
    try:
        radar_params = radar_settings.get_radar_params()
        if 'frame_rate' in radar_params:
            print(f"从radar_settings模块加载帧率: {radar_params['frame_rate']} Hz")
            return radar_params['frame_rate']
    except Exception as e:
        print(f"从radar_settings模块加载参数时出错: {e}")
    
    # 如果无法从模块获取，尝试从配置文件获取
    # 尝试加载RadarIfxAvian_00下的config.json
    config_file = os.path.join(folder_path, 'RadarIfxAvian_00', 'config.json')
    if os.path.isfile(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                if 'device_config' in config and 'fmcw_single_shape' in config['device_config']:
                    frame_time = config['device_config']['fmcw_single_shape'].get('frame_repetition_time_s')
                    if frame_time:
                        frame_rate = 1.0 / frame_time
                        print(f"从配置文件加载帧率: {frame_rate} Hz")
                        return frame_rate
        except Exception as e:
            print(f"读取配置文件出错: {e}")
    
    # 尝试加载根目录下的meta.json
    meta_file = os.path.join(folder_path, 'meta.json')
    if os.path.isfile(meta_file):
        try:
            with open(meta_file, 'r') as f:
                meta = json.load(f)
                if 'frame_rate_hz' in meta:
                    frame_rate = meta['frame_rate_hz']
                    print(f"从meta.json加载帧率: {frame_rate} Hz")
                    return frame_rate
        except Exception as e:
            print(f"读取meta文件出错: {e}")
    
    # 如果无法找到确切的帧率，返回默认值
    print("无法从配置文件获取帧率，将使用默认帧率(30Hz)")
    return 30.0

def get_start_time_from_meta(folder_path):
    """从meta.json中获取数据采集开始时间"""
    meta_file = os.path.join(folder_path, 'meta.json')
    if os.path.isfile(meta_file):
        try:
            with open(meta_file, 'r') as f:
                meta = json.load(f)
                if 'date_captured' in meta:
                    # 格式通常为: "2025-04-20T00:52:35+0800"
                    date_str = meta['date_captured']
                    
                    # 移除时区信息，返回naive datetime对象以兼容其他时间处理
                    if '+' in date_str:
                        date_str = date_str.split('+')[0]
                    
                    return datetime.datetime.fromisoformat(date_str)
        except Exception as e:
            print(f"从meta.json读取开始时间出错: {e}")
    
    return None

def trim_radar_data(data, start_frame, end_frame):
    """裁剪雷达数据"""
    if start_frame < 0:
        start_frame = 0
    if end_frame > data.shape[0] or end_frame == -1:
        end_frame = data.shape[0]
    
    return data[start_frame:end_frame]

def calculate_frame_time(start_time, frame_rate, frame_index):
    """计算特定帧的时间"""
    frame_interval = 1.0 / frame_rate
    return start_time + datetime.timedelta(seconds=frame_index * frame_interval)

def trim_data_by_time(data, record_start_time, frame_rate, trim_start_time, trim_end_time=None, frame_count=None):
    """根据时间范围裁剪数据"""
    # 计算开始帧
    if trim_start_time < record_start_time:
        print(f"警告: 裁剪开始时间 {trim_start_time} 早于记录开始时间 {record_start_time}，将使用记录开始时间")
        start_frame = 0
    else:
        time_diff = (trim_start_time - record_start_time).total_seconds()
        start_frame = int(time_diff * frame_rate)
        print(f"时间差: {time_diff} 秒，换算为帧号: {start_frame}，({start_frame/frame_rate:.2f}秒)")
    
    # 计算结束帧
    if frame_count is not None:
        # 如果指定了帧数，则从开始帧算起裁剪指定数量的帧
        end_frame = start_frame + frame_count
        if end_frame > data.shape[0]:
            print(f"警告: 指定帧数超过了可用帧数，将裁剪到最后一帧 ({data.shape[0]})")
            end_frame = data.shape[0]
        else:
            print(f"从第 {start_frame} 帧开始，裁剪 {frame_count} 帧")
    elif trim_end_time is None:
        # 如果未指定结束时间也未指定帧数，则裁剪到最后
        end_frame = data.shape[0]
        print(f"未指定结束时间或帧数，将裁剪到最后一帧 ({end_frame})")
    else:
        # 根据结束时间计算结束帧
        time_diff = (trim_end_time - record_start_time).total_seconds()
        end_frame = int(time_diff * frame_rate)
        if end_frame > data.shape[0]:
            print(f"警告: 裁剪结束时间对应的帧 ({end_frame}) 超过了记录长度 ({data.shape[0]})，将使用最后一帧")
            end_frame = data.shape[0]
        else:
            print(f"结束时间对应的帧号: {end_frame}")
    
    # 确保开始帧不超过结束帧
    if start_frame >= end_frame:
        print(f"错误: 计算的开始帧 ({start_frame}) 大于等于结束帧 ({end_frame})，无法裁剪")
        return data[0:0]  # 返回空数组
    
    actual_frame_count = end_frame - start_frame
    print(f"裁剪帧范围: {start_frame} - {end_frame}，共{actual_frame_count}帧")
    
    return trim_radar_data(data, start_frame, end_frame)

def main():
    parser = argparse.ArgumentParser(description='裁剪雷达数据')
    parser.add_argument('--input_folder', type=str, required=True, help='输入文件夹路径')
    parser.add_argument('--output_file', type=str, required=True, help='输出文件路径')
    parser.add_argument('--frame_rate', type=float, help='帧率(Hz)，如不指定将从配置文件中读取')
    parser.add_argument('--start_frame', type=int, default=0, help='开始帧索引(基于0)')
    parser.add_argument('--end_frame', type=int, default=-1, help='结束帧索引(基于0)，-1表示直到最后')
    parser.add_argument('--time_based', action='store_true', help='是否基于时间裁剪')
    parser.add_argument('--start_time', type=str, help='裁剪开始时间(格式: YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end_time', type=str, help='裁剪结束时间(格式: YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--frame_count', type=int, help='从开始时间起要裁剪的帧数量')
    parser.add_argument('--use_folder_time', action='store_true', help='是否使用文件夹名或meta.json中的时间作为开始时间')
    parser.add_argument('--use_meta_time', action='store_true', help='优先使用meta.json中的时间（默认已启用）')
    
    args = parser.parse_args()
    
    # 加载数据
    radar_data = load_radar_data(args.input_folder)
    if radar_data is None:
        return
    
    print(f"原始数据形状: {radar_data.shape}")
    
    # 如果没有指定帧率，尝试从配置文件中获取
    if args.frame_rate is None:
        args.frame_rate = load_config_file(args.input_folder)
    
    print(f"使用帧率: {args.frame_rate} Hz")
    
    # 裁剪数据
    if args.time_based:
        # 获取记录开始时间
        record_start_time = None
        
        # 先尝试从meta.json获取数据实际开始记录的时间
        meta_start_time = get_start_time_from_meta(args.input_folder)
        if meta_start_time:
            record_start_time = meta_start_time
            print(f"从meta.json获取的开始时间: {record_start_time}")
        else:
            # 如果无法从meta.json获取，尝试从文件夹名称获取
            folder_name = os.path.basename(args.input_folder.rstrip('/\\'))
            folder_time = extract_timestamp(folder_name)
            if folder_time:
                record_start_time = folder_time
                print(f"从文件夹名提取的开始时间: {record_start_time}")
            elif args.start_time and args.use_folder_time:
                # 如果用户强制使用提供的时间作为记录开始时间
                record_start_time = datetime.datetime.strptime(args.start_time, '%Y-%m-%d %H:%M:%S')
                print(f"使用用户指定的时间作为记录开始时间: {record_start_time}")
            else:
                print("错误: 无法确定记录开始时间。请检查meta.json或文件夹名，或使用--use_folder_time并指定--start_time。")
                return
        
        # 解析裁剪时间范围
        trim_start_time = None
        if args.start_time and not args.use_folder_time:
            # 如果用户指定了裁剪开始时间且不作为记录开始时间
            trim_start_time = datetime.datetime.strptime(args.start_time, '%Y-%m-%d %H:%M:%S')
            print(f"使用用户指定的裁剪开始时间: {trim_start_time}")
        else:
            # 默认从数据开始时间开始裁剪
            trim_start_time = record_start_time
            print(f"从记录开始时间开始裁剪: {trim_start_time}")
        
        trim_end_time = None
        if args.end_time:
            trim_end_time = datetime.datetime.strptime(args.end_time, '%Y-%m-%d %H:%M:%S')
            print(f"指定的裁剪结束时间: {trim_end_time}")
        
        # 基于时间裁剪
        trimmed_data = trim_data_by_time(
            radar_data, 
            record_start_time, 
            args.frame_rate, 
            trim_start_time, 
            trim_end_time,
            args.frame_count
        )
        # 获取实际的起始帧索引，用于后续计算时间范围
        start_frame = 0
        if trim_start_time >= record_start_time:
            time_diff = (trim_start_time - record_start_time).total_seconds()
            start_frame = int(time_diff * args.frame_rate)
    else:
        # 基于帧索引裁剪
        if args.frame_count and args.end_frame == -1:
            # 如果指定了帧数且未指定结束帧，则裁剪指定数量的帧
            end_frame = args.start_frame + args.frame_count
        else:
            end_frame = args.end_frame
        
        trimmed_data = trim_radar_data(radar_data, args.start_frame, end_frame)
        actual_frame_count = trimmed_data.shape[0]
        print(f"裁剪帧范围: {args.start_frame} - {end_frame}，共{actual_frame_count}帧")
    
    print(f"裁剪后数据形状: {trimmed_data.shape}")
    
    # 保存裁剪后的数据
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    np.save(args.output_file, trimmed_data)
    print(f"裁剪后的数据已保存到: {args.output_file}")

    # 打印时间范围信息
    if args.time_based and record_start_time:
        # 裁剪后数据的第一帧实际上是原始数据的第start_frame帧
        # 因此需要从record_start_time加上start_frame对应的时间偏移
        first_frame_offset = start_frame  # 使用上面重新计算的start_frame值
        
        # 计算裁剪后数据的实际开始时间
        trimmed_start_time = calculate_frame_time(record_start_time, args.frame_rate, first_frame_offset)
        
        # 计算裁剪后数据的实际结束时间
        trimmed_end_time = calculate_frame_time(record_start_time, args.frame_rate, first_frame_offset + trimmed_data.shape[0] - 1)
        
        print(f"裁剪后数据的时间范围: {trimmed_start_time} 到 {trimmed_end_time}")
        print(f"总时长: {(trimmed_end_time - trimmed_start_time).total_seconds()} 秒")

if __name__ == "__main__":
    main() 