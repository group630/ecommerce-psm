"""
数据预处理模块
加载并清洗Online Retail数据集
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import timer, check_data_quality, describe_dataframe


class DataPreprocessor:
    """数据预处理器"""

    def __init__(self, filepath):
        """
        初始化预处理器

        Parameters:
        -----------
        filepath : str
            数据文件路径
        """
        self.filepath = filepath
        self.raw_data = None
        #self.clean_data = None

    @timer
    def load_data(self):
        """加载原始数据"""
        print(f"正在加载数据: {self.filepath}")
        self.raw_data = pd.read_excel(self.filepath, engine='openpyxl')
        print(f"数据加载完成，形状: {self.raw_data.shape}")
        return self.raw_data

    @timer
    def clean_data(self):
        """数据清洗"""
        df = self.raw_data.copy()

        print("开始数据清洗...")
        initial_rows = len(df)

        # 1. 剔除取消订单（发票号以C开头）
        cancel_mask = df['InvoiceNo'].astype(str).str.startswith('C')
        df = df[~cancel_mask]
        print(f"  剔除取消订单: {cancel_mask.sum()}条")

        # 2. 剔除CustomerID缺失的记录
        before = len(df)
        df = df[df['CustomerID'].notna()]
        print(f"  剔除CustomerID缺失: {before - len(df)}条")

        # 3. 剔除数量为负或0的记录
        before = len(df)
        df = df[df['Quantity'] > 0]
        print(f"  剔除数量<=0: {before - len(df)}条")

        # 4. 剔除单价为负或0的记录
        before = len(df)
        df = df[df['UnitPrice'] > 0]
        print(f"  剔除单价<=0: {before - len(df)}条")

        # 5. 转换数据类型
        df['CustomerID'] = df['CustomerID'].astype(int)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

        # 6. 计算总金额
        df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

        # 7. 提取时间特征
        df['Year'] = df['InvoiceDate'].dt.year
        df['Month'] = df['InvoiceDate'].dt.month
        df['Day'] = df['InvoiceDate'].dt.day
        df['Hour'] = df['InvoiceDate'].dt.hour
        df['Weekday'] = df['InvoiceDate'].dt.weekday
        df['Week'] = df['InvoiceDate'].dt.isocalendar().week

        print(f"清洗完成，保留: {len(df)}条记录 (保留率: {len(df) / initial_rows * 100:.2f}%)")

        self.clean_data = df
        return df

    def construct_user_features(self):
        """构造用户特征"""
        df = self.clean_data

        # 用户聚合特征
        user_features = df.groupby('CustomerID').agg({
            'TotalAmount': ['sum', 'mean', 'std'],
            'InvoiceNo': 'nunique',  # 购买次数（订单数）
            'Quantity': ['sum', 'mean'],
            'StockCode': 'nunique',  # 购买品类数
            'UnitPrice': 'mean'
        }).reset_index()

        # 扁平化列名
        user_features.columns = [
            'CustomerID', 'TotalSpent', 'AvgOrderAmount', 'StdOrderAmount',
            'PurchaseCount', 'TotalQuantity', 'AvgQuantityPerOrder',
            'UniqueProducts', 'AvgUnitPrice'
        ]

        # 填充标准差缺失值
        user_features['StdOrderAmount'] = user_features['StdOrderAmount'].fillna(0)

        # 计算客单价
        user_features['AvgOrderValue'] = user_features['TotalSpent'] / user_features['PurchaseCount']

        # 计算活跃天数
        user_daily = df.groupby(['CustomerID', df['InvoiceDate'].dt.date]).size().reset_index()
        user_daily.columns = ['CustomerID', 'Date', 'Count']
        active_days = user_daily.groupby('CustomerID')['Date'].nunique().reset_index()
        active_days.columns = ['CustomerID', 'ActiveDays']
        user_features = user_features.merge(active_days, on='CustomerID', how='left')

        # 计算用户生命周期（从首次购买到末次购买的天数）
        first_last = df.groupby('CustomerID')['InvoiceDate'].agg(['min', 'max']).reset_index()
        first_last.columns = ['CustomerID', 'FirstPurchase', 'LastPurchase']
        first_last['LifecycleDays'] = (first_last['LastPurchase'] - first_last['FirstPurchase']).dt.days
        user_features = user_features.merge(first_last[['CustomerID', 'LifecycleDays']], on='CustomerID', how='left')

        # 计算购买频率（次/月）
        user_features['PurchaseFreq'] = user_features['PurchaseCount'] / (user_features['LifecycleDays'] / 30 + 0.01)

        # 计算退货率（需要退货数据）
        # 注：取消订单已被剔除，此处简化处理

        self.user_features = user_features
        return user_features

    def construct_product_features(self):
        """构造商品特征"""
        df = self.clean_data

        product_features = df.groupby('StockCode').agg({
            'UnitPrice': 'mean',
            'Quantity': ['sum', 'mean'],
            'InvoiceNo': 'nunique',  # 被购买次数
            'CustomerID': 'nunique'  # 购买用户数
        }).reset_index()

        product_features.columns = [
            'StockCode', 'AvgPrice', 'TotalSold', 'AvgQuantityPerOrder',
            'PurchaseCount', 'UniqueCustomers'
        ]

        # 商品描述
        product_desc = df.groupby('StockCode')['Description'].first().reset_index()
        product_features = product_features.merge(product_desc, on='StockCode', how='left')

        # 商品价格分位数
        product_features['PriceQuartile'] = pd.qcut(product_features['AvgPrice'], 4,
                                                    labels=['Q1(低价)', 'Q2', 'Q3', 'Q4(高价)'])

        self.product_features = product_features
        return product_features

    def run_pipeline(self):
        """运行完整的数据预处理流程"""
        self.load_data()
        self.clean_data()
        self.construct_user_features()
        self.construct_product_features()

        # 输出统计信息
        print("\n" + "=" * 50)
        print("数据预处理完成!")
        print("=" * 50)
        print(f"用户数: {self.user_features['CustomerID'].nunique()}")
        print(f"商品数: {self.product_features['StockCode'].nunique()}")
        print(f"交易记录数: {len(self.clean_data)}")
        print(f"总交易金额: {self.clean_data['TotalAmount'].sum():.2f} 英镑")

        return self.clean_data, self.user_features, self.product_features


# 主函数测试
if __name__ == "__main__":
    # 请将路径替换为实际数据文件路径
    preprocessor = DataPreprocessor("../data/Online Retail.xlsx")
    clean_data, user_features, product_features = preprocessor.run_pipeline()

    # 保存处理后的数据
    clean_data.to_csv("./results/tables/clean_retail_data.csv", index=False)
    user_features.to_csv("./results/tables/user_features.csv", index=False)
    product_features.to_csv("./results/tables/product_features.csv", index=False)

    print("\n处理后数据已保存至 ../data/ 目录")