"""
模型评估结果整合工具

该脚本用于整合results文件夹下按信号分解算法分类的所有.txt评估结果文件，
生成一个表格文件，其中行是模型名称，列是不同的性能指标。

使用方法:
    python aggregate_results.py --decomp_method EEMD
    python aggregate_results.py --decomp_method VMD
    python aggregate_results.py --decomp_method EMD
    python aggregate_results.py --decomp_method DWT
    python aggregate_results.py --decomp_method all  # 处理所有分解方法
"""

import os
import re
import pandas as pd
import argparse
import glob
from datetime import datetime

def parse_evaluation_file(file_path):
    """
    解析单个评估结果文件，提取模型名称和性能指标
    
    参数:
        file_path: 评估结果文件路径
        
    返回:
        包含模型名称和性能指标的字典
    """
    results = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取模型名称
        model_name_match = re.search(r'模型: (.+?)(?:_EEMD|_EMD|_VMD|_DWT|_rebuilt|$)', content)
        if model_name_match:
            model_name = model_name_match.group(1).strip()
            results['模型名称'] = model_name
        else:
            # 使用文件名作为备选
            base_name = os.path.basename(file_path)
            model_name = base_name.replace('_evaluation.txt', '')
            results['模型名称'] = model_name
        
        # 提取分解方法
        decomp_match = re.search(r'分解方法: (\w+)', content)
        if decomp_match:
            results['分解方法'] = decomp_match.group(1)
        
        # 提取常见指标
        metrics = {
            'MAE (BPM)': r'MAE: ([\d\.]+) BPM',
            'RMSE (BPM)': r'RMSE: ([\d\.]+) BPM',
            'R²': r'R²: ([-\d\.]+)',
            'MAPE (%)': r'MAPE: ([\d\.]+)%',
            '最大误差 (BPM)': r'最大误差: ([\d\.]+) BPM',
            '中位误差 (BPM)': r'中位误差: ([\d\.]+) BPM',
            '误差标准差': r'误差标准差: ([\d\.]+)',
            '95%置信区间下限 (BPM)': r'95%置信区间: \[([-\d\.]+),',
            '95%置信区间上限 (BPM)': r'95%置信区间: \[[-\d\.]+, ([-\d\.]+)\]',
            '±3 BPM准确率 (%)': r'±3 BPM准确率: ([\d\.]+)%'
        }
        
        for metric_name, pattern in metrics.items():
            match = re.search(pattern, content)
            if match:
                try:
                    results[metric_name] = float(match.group(1))
                except ValueError:
                    results[metric_name] = None
            else:
                results[metric_name] = None
        
        # 提取不同心率区间性能（如果有）
        hr_metrics = {
            '低心率 MAE (BPM)': r'低心率\(<70 BPM\) MAE: ([\d\.]+) BPM',
            '正常心率 MAE (BPM)': r'正常心率\(70-90 BPM\) MAE: ([\d\.]+) BPM',
            '高心率 MAE (BPM)': r'高心率\(>90 BPM\) MAE: ([\d\.]+) BPM'
        }
        
        for metric_name, pattern in hr_metrics.items():
            match = re.search(pattern, content)
            if match:
                try:
                    results[metric_name] = float(match.group(1))
                except ValueError:
                    results[metric_name] = None
            else:
                results[metric_name] = None
        
        # 提取推理性能（如果有）
        inference_metrics = {
            '单样本推理时间 (ms)': r'单样本推理时间: ([\d\.]+) 毫秒',
            '批量推理时间 (秒)': r'批量推理时间\(整个测试集\): ([\d\.]+) 秒',
            '每秒处理样本数': r'每秒处理样本数: ([\d\.]+)'
        }
        
        for metric_name, pattern in inference_metrics.items():
            match = re.search(pattern, content)
            if match:
                try:
                    results[metric_name] = float(match.group(1))
                except ValueError:
                    results[metric_name] = None
            else:
                results[metric_name] = None
        
        return results
    
    except Exception as e:
        print(f"处理文件时出错: {file_path}")
        print(f"错误信息: {e}")
        return {'模型名称': os.path.basename(file_path), '错误': str(e)}

def aggregate_results(decomp_method='all'):
    """
    按分解方法整合评估结果
    
    参数:
        decomp_method: 分解方法 (EEMD, EMD, VMD, DWT, 或 all)
        
    返回:
        包含所有模型结果的DataFrame
    """
    results_dir = '.'  # 当前目录（results文件夹）
    all_results = []
    
    # 确定要处理的分解方法
    if decomp_method.lower() == 'all':
        methods = ['EEMD', 'EMD', 'VMD', 'DWT']
    else:
        methods = [decomp_method.upper()]
    
    for method in methods:
        # 查找所有包含该分解方法的评估结果文件
        # 使用多种模式匹配不同的命名可能性
        patterns = [
            f"*_{method}_*.txt",  # 例如: CNN_GRU_EEMD_best_evaluation.txt
            f"*_{method}.txt",     # 例如: CNN_GRU_EEMD.txt
            f"*{method}*.txt"      # 任何包含该方法名称的文件
        ]
        
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(results_dir, pattern)))
        
        # 去重
        files = list(set(files))
        
        print(f"找到{len(files)}个{method}分解方法的评估文件")
        
        # 解析每个文件
        for file_path in files:
            result = parse_evaluation_file(file_path)
            if result and '分解方法' not in result:
                result['分解方法'] = method
            if result:
                all_results.append(result)
    
    # 创建DataFrame
    if all_results:
        df = pd.DataFrame(all_results)
        
        # 按照分解方法和性能（MAE）排序
        sort_columns = ['分解方法']
        if 'MAE (BPM)' in df.columns:
            sort_by = ['分解方法', 'MAE (BPM)']
        else:
            sort_by = ['分解方法', '模型名称']
        
        df = df.sort_values(by=sort_by)
        
        return df
    else:
        print("未找到任何评估结果文件")
        return None

def save_results(df, decomp_method='all'):
    """
    保存整合的结果到CSV和Excel文件
    
    参数:
        df: 包含结果的DataFrame
        decomp_method: 分解方法名称，用于文件命名
    """
    if df is None or df.empty:
        print("没有结果可保存")
        return
    
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # 保存CSV文件
    if decomp_method.lower() == 'all':
        csv_filename = f"model_results_all_{timestamp}.csv"
        excel_filename = f"model_results_all_{timestamp}.xlsx"
    else:
        csv_filename = f"model_results_{decomp_method}_{timestamp}.csv"
        excel_filename = f"model_results_{decomp_method}_{timestamp}.xlsx"
    
    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"结果已保存为CSV: {csv_filename}")
    
    # 尝试保存为Excel文件
    try:
        df.to_excel(excel_filename, index=False, engine='openpyxl')
        print(f"结果已保存为Excel: {excel_filename}")
    except Exception as e:
        print(f"保存Excel文件失败: {e}")
        print("请确保已安装openpyxl库: pip install openpyxl")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='整合模型评估结果')
    parser.add_argument('--decomp_method', type=str, default='all',
                       choices=['EEMD', 'EMD', 'VMD', 'DWT', 'all'],
                       help='分解方法 (默认: all)')
    
    args = parser.parse_args()
    
    # 整合结果
    print(f"整合{args.decomp_method}分解方法的评估结果...")
    df = aggregate_results(args.decomp_method)
    
    # 显示结果摘要
    if df is not None and not df.empty:
        print("\n找到的模型数量:")
        print(df['分解方法'].value_counts())
        
        print("\n结果预览:")
        pd.set_option('display.max_columns', None)  # 显示所有列
        pd.set_option('display.width', 1000)  # 加宽显示
        print(df.head())
        
        # 保存结果
        save_results(df, args.decomp_method)
    
    print("整合完成！")

if __name__ == "__main__":
    main() 