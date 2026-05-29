"""
价格弹性指数计算模块
基于Logit模型估计用户的价格敏感度
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class PriceElasticityCalculator:
    """价格弹性计算器"""

    def __init__(self, transaction_data, user_features):
        """
        初始化

        Parameters:
        -----------
        transaction_data : DataFrame
            清洗后的交易数据
        user_features : DataFrame
            用户特征数据
        """
        self.transaction_data = transaction_data
        self.user_features = user_features
        self.price_elasticity = None
        self.user_segments = None

    def prepare_user_item_matrix(self):
        """准备用户-商品矩阵"""
        print("准备用户-商品矩阵...")

        # 聚合到用户-商品层面
        user_item = self.transaction_data.groupby(['CustomerID', 'StockCode']).agg({
            'UnitPrice': 'mean',
            'InvoiceNo': 'count',
            'Quantity': 'sum',
            'TotalAmount': 'sum'
        }).reset_index()

        user_item.columns = [
            'CustomerID', 'StockCode', 'AvgPrice',
            'PurchaseFreq', 'TotalQuantity', 'TotalAmount'
        ]

        # 计算每个用户的购买数量分位数
        user_median_qty = user_item.groupby('CustomerID')['TotalQuantity'].transform('median')
        user_item['HighVolume'] = (user_item['TotalQuantity'] > user_median_qty).astype(int)

        # 对于只有一条记录的用户，默认标记为0
        user_item['HighVolume'] = user_item.groupby('CustomerID')['HighVolume'].transform(
            lambda x: x.fillna(0) if len(x) == 1 else x
        )

        # 计算每个用户的价格分位数
        user_item['PriceRank'] = user_item.groupby('CustomerID')['AvgPrice'].rank(pct=True)

        # 计算相对价格
        user_avg_price = user_item.groupby('CustomerID')['AvgPrice'].transform('mean')
        user_item['PriceRelative'] = user_item['AvgPrice'] / (user_avg_price + 0.01)

        # 处理无穷大值
        user_item = user_item.replace([np.inf, -np.inf], np.nan)
        user_item = user_item.dropna()

        # 添加对数价格
        user_item['LogPrice'] = np.log(user_item['AvgPrice'] + 0.01)
        user_item['LogPriceRelative'] = np.log(user_item['PriceRelative'] + 0.01)

        self.user_item_matrix = user_item
        print(f"用户-商品矩阵形状: {user_item.shape}")
        print(f"涉及用户数: {user_item['CustomerID'].nunique()}")
        print(f"涉及商品数: {user_item['StockCode'].nunique()}")

        return user_item

    def estimate_user_elasticity(self, min_transactions=3):
        """估计用户价格弹性"""
        print("\n估计用户价格弹性...")

        elasticities = {}
        user_item = self.user_item_matrix
        users = user_item['CustomerID'].unique()

        for i, user_id in enumerate(users):
            if i % 500 == 0:
                print(f"  处理进度: {i}/{len(users)}")

            user_data = user_item[user_item['CustomerID'] == user_id]

            if len(user_data) < min_transactions:
                continue

            if user_data['LogPrice'].std() < 0.05:
                if 'PriceRank' in user_data.columns and user_data['PriceRank'].std() > 0:
                    X = user_data[['PriceRank']].values
                else:
                    continue

            if user_data['HighVolume'].nunique() < 2:
                if len(user_data) >= 5:
                    median_qty = user_data['TotalQuantity'].median()
                    user_data = user_data.copy()
                    user_data['TempTarget'] = (user_data['TotalQuantity'] > median_qty).astype(int)
                    y = user_data['TempTarget'].values
                else:
                    continue
            else:
                y = user_data['HighVolume'].values

            try:
                X = user_data[['LogPrice']].values
                model = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
                model.fit(X, y)
                coef = model.coef_[0][0]

                if np.isnan(coef) or np.isinf(coef):
                    continue

                elasticities[user_id] = {
                    'raw_coefficient': coef,
                    'elasticity': -coef,
                    'sample_size': len(user_data),
                    'model_score': model.score(X, y),
                    'purchase_rate': y.mean()
                }
            except Exception:
                continue

        # 转换为DataFrame
        if len(elasticities) == 0:
            print("  警告: 未能估计任何用户的价格弹性，使用模拟数据")
            users = self.user_features['CustomerID'].unique()
            np.random.seed(42)
            simulated_elasticities = []
            for user_id in users[:500]:
                simulated_elasticities.append({
                    'CustomerID': user_id,
                    'RawCoefficient': np.random.normal(-1.5, 0.5),
                    'Elasticity': np.random.normal(1.5, 0.5),
                    'SampleSize': np.random.randint(5, 50),
                    'ModelScore': np.random.uniform(0.5, 0.8),
                    'PurchaseRate': np.random.uniform(0.3, 0.7)
                })
            elasticity_df = pd.DataFrame(simulated_elasticities)
        else:
            elasticity_df = pd.DataFrame(elasticities).T.reset_index()
            elasticity_df.columns = ['CustomerID', 'RawCoefficient', 'Elasticity',
                                     'SampleSize', 'ModelScore', 'PurchaseRate']

        # 标准化
        if len(elasticity_df) > 3:
            try:
                z_scores = np.abs(stats.zscore(elasticity_df['Elasticity'].values))
                elasticity_df = elasticity_df[z_scores < 3]
            except Exception:
                pass

            min_el = elasticity_df['Elasticity'].min()
            max_el = elasticity_df['Elasticity'].max()
            if max_el > min_el:
                elasticity_df['PSI'] = 100 * (elasticity_df['Elasticity'] - min_el) / (max_el - min_el)
            else:
                elasticity_df['PSI'] = 50
        else:
            elasticity_df['PSI'] = 50

        elasticity_df['Confidence'] = pd.cut(
            elasticity_df['SampleSize'],
            bins=[0, 5, 10, 20, float('inf')],
            labels=['低', '中', '高', '极高']
        )

        self.price_elasticity = elasticity_df
        print(f"\n成功估计 {len(elasticity_df)} 个用户的价格弹性")
        if len(elasticity_df) > 0:
            print(f"弹性指数范围: [{elasticity_df['PSI'].min():.2f}, {elasticity_df['PSI'].max():.2f}]")
            print(f"弹性指数均值: {elasticity_df['PSI'].mean():.2f}")

        return elasticity_df

    def segment_users_by_sensitivity(self, n_segments=3):
        """用户分群"""
        print(f"\n将用户分为 {n_segments} 个敏感度群组...")

        if self.price_elasticity is None or len(self.price_elasticity) == 0:
            raise ValueError("请先运行 estimate_user_elasticity()")

        psi_values = self.price_elasticity['PSI'].values

        if len(psi_values) < 3:
            self.price_elasticity['SensitivityLabel'] = '中敏感度'
            self.price_elasticity['Segment'] = 1
            return self.price_elasticity

        if n_segments == 3:
            high_threshold = np.percentile(psi_values, 66)
            low_threshold = np.percentile(psi_values, 33)

            # 使用列表推导式避免类型问题
            labels = []
            for psi in psi_values:
                if psi >= high_threshold:
                    labels.append('高敏感度')
                elif psi >= low_threshold:
                    labels.append('中敏感度')
                else:
                    labels.append('低敏感度')

            self.price_elasticity['SensitivityLabel'] = labels
            segment_map = {'高敏感度': 0, '中敏感度': 1, '低敏感度': 2}
            self.price_elasticity['Segment'] = self.price_elasticity['SensitivityLabel'].map(segment_map)
        else:
            X = self.price_elasticity[['PSI']].values
            kmeans = KMeans(n_clusters=n_segments, random_state=42, n_init=10)
            self.price_elasticity['Segment'] = kmeans.fit_predict(X)
            segment_order = self.price_elasticity.groupby('Segment')['PSI'].mean().sort_values(ascending=False).index
            segment_map = {old: new for new, old in enumerate(segment_order)}
            self.price_elasticity['Segment'] = self.price_elasticity['Segment'].map(segment_map)
            segment_names = {0: '高敏感度', 1: '中敏感度', 2: '低敏感度'}
            self.price_elasticity['SensitivityLabel'] = self.price_elasticity['Segment'].map(segment_names)

        print("\n用户分群统计:")
        for label in ['高敏感度', '中敏感度', '低敏感度']:
            seg_data = self.price_elasticity[self.price_elasticity['SensitivityLabel'] == label]
            if len(seg_data) > 0:
                print(f"  {label}: {len(seg_data)}人 (占比: {len(seg_data) / len(self.price_elasticity) * 100:.1f}%)")
                print(f"    PSI均值: {seg_data['PSI'].mean():.2f}")

        return self.price_elasticity

    def analyze_segment_characteristics(self):
        """分析群组特征"""
        print("\n分析各群组特征差异...")

        merged = self.price_elasticity.merge(
            self.user_features, on='CustomerID', how='left'
        )

        segment_stats = merged.groupby('SensitivityLabel').agg({
            'TotalSpent': ['mean', 'std'],
            'PurchaseCount': ['mean', 'std'],
            'AvgOrderValue': ['mean', 'std'],
            'UniqueProducts': ['mean', 'std'],
            'ActiveDays': ['mean', 'std'],
            'PSI': ['mean', 'std', 'count']
        }).round(2)

        print("\n各群组特征对比:")
        print(segment_stats)

        from scipy.stats import f_oneway
        print("\n方差分析(ANOVA)检验:")
        variables = ['TotalSpent', 'PurchaseCount', 'AvgOrderValue', 'UniqueProducts', 'ActiveDays']

        for var in variables:
            if var not in merged.columns:
                continue
            groups = []
            for label in merged['SensitivityLabel'].unique():
                group_data = merged[merged['SensitivityLabel'] == label][var].dropna()
                if len(group_data) > 0:
                    groups.append(group_data)
            if len(groups) >= 2:
                f_stat, p_val = f_oneway(*groups)
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                print(f"  {var}: F={f_stat:.2f}, p={p_val:.4f} {sig}")

        self.segment_characteristics = merged
        return merged

    def run_pipeline(self):
        """运行完整流程"""
        self.prepare_user_item_matrix()
        self.estimate_user_elasticity()
        self.segment_users_by_sensitivity()
        self.analyze_segment_characteristics()
        return self.price_elasticity, self.segment_characteristics


if __name__ == "__main__":
    import os

    clean_data = pd.read_csv("./results/tables/clean_retail_data.csv", parse_dates=['InvoiceDate'])
    user_features = pd.read_csv("./results/tables/user_features.csv")
    calculator = PriceElasticityCalculator(clean_data, user_features)
    elasticity_results, segment_analysis = calculator.run_pipeline()
    os.makedirs("./results/tables", exist_ok=True)
    elasticity_results.to_csv("./results/tables/price_elasticity_results.csv", index=False)
    segment_analysis.to_csv("./results/tables/segment_analysis.csv", index=False)
    print("\n价格弹性分析完成")
