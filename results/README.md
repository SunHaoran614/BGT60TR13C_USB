# 模型评估结果聚合工具

这个文件夹包含了一个用于整合模型评估结果的工具`aggregate_results.py`。

## 功能

该脚本可以按照信号分解算法（EEMD、EMD、VMD或DWT）自动整合所有模型评估结果文件（.txt格式），并生成一个表格文件（CSV和Excel格式），其中：
- 表格的行是各个模型
- 表格的列是各种性能指标

提取的指标包括：
- MAE (BPM)
- RMSE (BPM)
- R²
- MAPE (%)
- 最大误差 (BPM)
- 中位误差 (BPM)
- 误差标准差
- 95%置信区间
- ±3 BPM准确率 (%)
- 不同心率区间性能
- 推理时间指标

## 使用方法

1. 确保你已经安装了所需的Python库：
   ```bash
   pip install pandas openpyxl
   ```

2. 运行脚本，指定要处理的分解方法：
   ```bash
   # 处理EEMD分解方法的所有评估结果
   python aggregate_results.py --decomp_method EEMD
   
   # 处理VMD分解方法的所有评估结果
   python aggregate_results.py --decomp_method VMD
   
   # 处理EMD分解方法的所有评估结果
   python aggregate_results.py --decomp_method EMD
   
   # 处理DWT分解方法的所有评估结果
   python aggregate_results.py --decomp_method DWT
   
   # 处理所有分解方法的评估结果
   python aggregate_results.py --decomp_method all
   ```

3. 脚本会自动：
   - 扫描当前目录下的所有评估结果文件（.txt文件）
   - 提取与指定分解方法相关的文件
   - 解析每个文件中的性能指标
   - 生成包含所有结果的表格
   - 保存为CSV和Excel格式（带有时间戳）

## 输出文件

- CSV格式: `model_results_{decomp_method}_{timestamp}.csv`
- Excel格式: `model_results_{decomp_method}_{timestamp}.xlsx`

例如：`model_results_EEMD_20250504_1030.csv`

## 注意事项

1. 确保评估结果文件的格式符合radar_model_evaluator.py生成的标准格式
2. 脚本会自动识别文件名或内容中包含分解方法名称的文件
3. 如果某些指标在文件中不存在，表格中对应的单元格将为空值 