"""
工具函数模块
包含常用的辅助函数
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def timer(func):
    """装饰器：计算函数执行时间"""

    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        print(f"[{func.__name__}] 执行时间: {(end - start).total_seconds():.2f}秒")
        return result

    return wrapper


def check_data_quality(df):
    """检查数据质量"""
    print("=" * 50)
    print("数据质量检查报告")
    print("=" * 50)

    # 缺失值检查
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\n缺失值情况:")
        print(missing[missing > 0])

    # 数据类型检查
    print("\n数据类型:")
    print(df.dtypes)

    # 重复值检查
    print(f"\n重复记录数: {df.duplicated().sum()}")

    # 异常值检查（负值）
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            print(f"  {col}: {neg_count}个负值")

    return None


def describe_dataframe(df, name="DataFrame"):
    """详细描述数据框"""
    print(f"\n{'=' * 50}")
    print(f"{name} 描述性统计")
    print(f"{'=' * 50}")
    print(f"形状: {df.shape}")
    print(f"内存占用: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
    print(f"\n数值列统计:")
    print(df.describe())

    return None


def save_results(df, filepath, index=False):
    """保存结果到CSV"""
    df.to_csv(filepath, index=index)
    print(f"结果已保存至: {filepath}")


def calculate_percentile_rank(series):
    """计算百分位排名"""
    return series.rank(pct=True) * 100