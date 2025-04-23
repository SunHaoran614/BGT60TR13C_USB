# 雷达数据裁剪工具使用说明

## 功能介绍

这个工具用于裁剪BGT60TR13C雷达的原始数据（.npy格式），支持三种裁剪方式：
1. 基于帧索引裁剪
2. 基于时间范围裁剪（指定开始时间和结束时间）
3. 基于特定时间和帧数量裁剪（指定开始时间和要裁剪的帧数）

## 安装依赖

```bash
pip install numpy
```

## 使用方法

### 1. 基于帧索引裁剪

直接指定开始帧和结束帧：

```bash
python radar_data_trimmer.py --input_folder Dataset/data/BGT60TR13C_record_20250420-005235_120cm --output_file trimmed_data.npy --start_frame 100 --end_frame 500
```

指定开始帧和要裁剪的帧数：

```bash
python radar_data_trimmer.py --input_folder Dataset/data/BGT60TR13C_record_20250420-005235_120cm --output_file trimmed_data.npy --start_frame 100 --frame_count 400
```

### 2. 基于时间裁剪（使用meta.json或文件夹名称中的时间作为记录开始时间）

```bash
python radar_data_trimmer.py --input_folder Dataset/data/BGT60TR13C_record_20250420-005235_120cm --output_file trimmed_data.npy --time_based --use_folder_time --end_time "2025-04-20 00:53:35"
```

### 3. 基于时间裁剪（手动指定记录开始时间和结束时间）

```bash
python radar_data_trimmer.py --input_folder Dataset/data/BGT60TR13C_record_20250420-005235_120cm --output_file trimmed_data.npy --time_based --start_time "2025-04-20 00:52:35" --end_time "2025-04-20 00:53:35"
```

### 4. 指定开始时间并裁剪固定帧数（新功能）

```bash
python radar_data_trimmer.py --input_folder Dataset/data/BGT60TR13C_record_20250420-005235_120cm --output_file trimmed_data.npy --time_based --start_time "2025-04-20 00:52:35" --frame_count 18000
```

## 参数说明

| 参数 | 说明 |
|------|------|
| `--input_folder` | 输入文件夹路径，包含.npy雷达数据文件 |
| `--output_file` | 输出文件路径，保存裁剪后的.npy数据 |
| `--frame_rate` | 帧率(Hz)，如不指定则自动从配置文件中读取 |
| `--start_frame` | 开始帧索引(基于0)，默认0 |
| `--end_frame` | 结束帧索引(基于0)，-1表示直到最后一帧 |
| `--frame_count` | 要裁剪的帧数量，可与start_frame或start_time配合使用 |
| `--time_based` | 是否基于时间裁剪，使用此参数开启基于时间的裁剪 |
| `--start_time` | 裁剪开始时间，格式: YYYY-MM-DD HH:MM:SS |
| `--end_time` | 裁剪结束时间，格式: YYYY-MM-DD HH:MM:SS |
| `--use_folder_time` | 是否使用meta.json或文件夹名中的时间作为记录开始时间 |
| `--use_meta_time` | 优先使用meta.json中的时间（默认已启用） |

## 自动功能

1. **自动检测雷达数据文件**：脚本会自动在输入文件夹及其子文件夹（如RadarIfxAvian_00）中查找radar.npy文件

2. **自动提取帧率**：如果未指定帧率，脚本会尝试从以下位置获取：
   - 首先尝试从RadarIfxAvian_00/config.json中提取frame_repetition_time_s并计算帧率
   - 如果失败，尝试从meta.json中提取frame_rate_hz
   - 都失败的情况下使用默认帧率30Hz

3. **自动提取开始时间**：使用--use_folder_time参数时：
   - 优先从meta.json的date_captured字段获取（ISO 8601格式如"2025-04-20T00:52:35+0800"）
   - 如果meta.json不存在或无法解析，则从文件夹名称中提取时间戳

## 注意事项

1. 当使用`--use_folder_time`参数时，工具会优先尝试从meta.json中获取"date_captured"字段作为记录开始时间
2. 如果meta.json不存在或无法读取，才会尝试从文件夹名称中提取时间戳，文件夹名格式必须为`BGT60TR13C_record_YYYYMMDD-HHMMSS_*`
3. 指定精确的帧率很重要，特别是进行长时间裁剪时，精度误差会累积
4. 对于我们的数据，配置文件中的精确帧率值为30.0Hz (帧重复时间为0.03333333507180214秒)
5. 如果裁剪结束时间超过了记录长度，将使用最后一帧
6. 使用`--frame_count`参数时会优先考虑帧数量而非结束时间

## 示例场景

### 截取某个特定时间段的数据（使用meta.json中的开始时间）
```bash
python radar_data_trimmer.py --input_folder Dataset/data/BGT60TR13C_record_20250420-005235_120cm --output_file trimmed_data.npy --time_based --use_folder_time --start_time "2025-04-20 00:52:45" --end_time "2025-04-20 00:53:15"
```

### 从特定时间开始裁剪精确的18000帧数据
```bash
python radar_data_trimmer.py --input_folder Dataset/data/BGT60TR13C_record_20250420-005235_120cm --output_file trimmed_data.npy --time_based --start_time "2025-04-20 00:52:45" --frame_count 18000
```

### meta.json中的时间格式
```json
{
    "date_captured": "2025-04-20T00:52:35+0800"
}
```
这个ISO 8601格式的时间会被正确解析，包括时区信息（如果存在）。这个时间通常与文件夹名称中的时间戳一致，但更为精确，因为它包含了时区信息。 