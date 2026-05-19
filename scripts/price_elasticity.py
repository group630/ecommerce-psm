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

        # 计算每个用户的价格中位数
        user_median_price = user_item.groupby('CustomerID')['AvgPrice'].median().to_dict()
        user_item['PriceRelative'] = user_item.apply(
            lambda x: x['AvgPrice'] / user_median_price.get(x['CustomerID'], 1), axis=1
        )

        # 购买频率二值化（高于同用户中位数的标记为高频率）
        user_median_freq = user_item.groupby('CustomerID')['PurchaseFreq'].transform('median')
        user_item['HighFreq'] = (user_item['PurchaseFreq'] > user_item.groupby('CustomerID')['PurchaseFreq'].transform(
            'median')).astype(int)

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

    def estimate_user_elasticity(self, min_transactions=5):
        """
        为每个用户估计价格弹性系数

        Parameters:
        -----------
        min_transactions : int
            最小交易次数阈值
        """
        print("\n估计用户价格弹性...")

        elasticities = {}
        model_details = {}

        user_item = self.user_item_matrix
        users = user_item['CustomerID'].unique()

        for i, user_id in enumerate(users):
            if i % 500 == 0:
                print(f"  处理进度: {i}/{len(users)}")

            user_data = user_item[user_item['CustomerID'] == user_id]

            # 检查样本量
            if len(user_data) < min_transactions:
                continue

            # 检查价格变异
            if user_data['LogPrice'].std() < 0.1:
                continue

            # 检查目标变量是否有足够的变化
            if user_data['HighFreq'].nunique() < 2:
                continue

            try:
                # 使用Logistic回归估计价格对购买频率的影响
                X = user_data[['LogPrice']].values
                y = user_data['HighFreq'].values

                model = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
                model.fit(X, y)

                # 弹性系数（负值越大表示价格越敏感）
                coef = model.coef_[0][0]

                # 计算预测概率
                y_pred_proba = model.predict_proba(X)[:, 1]

                # 存储结果
                elasticities[user_id] = {
                    'raw_coefficient': coef,
                    'elasticity': -coef,  # 取正便于解释
                    'sample_size': len(user_data),
                    'model_score': model.score(X, y),
                    'purchase_rate': user_data['HighFreq'].mean()
                }

            except Exception as e:
                continue

        # 转换为DataFrame
        elasticity_df = pd.DataFrame(elasticities).T.reset_index()
        elasticity_df.columns = ['CustomerID', 'RawCoefficient', 'Elasticity',
                                 'SampleSize', 'ModelScore', 'PurchaseRate']

        # 剔除异常值（超出3倍标准差）
        z_scores = np.abs(stats.zscore(elasticity_df['Elasticity']))
        elasticity_df = elasticity_df[z_scores < 3]

        # 标准化到0-100区间
        min_el = elasticity_df['Elasticity'].min()
        max_el = elasticity_df['Elasticity'].max()
        elasticity_df['PSI'] = 100 * (elasticity_df['Elasticity'] - min_el) / (max_el - min_el)

        # 添加置信度标签（基于样本量）
        elasticity_df['Confidence'] = pd.cut(
            elasticity_df['SampleSize'],
            bins=[0, 10, 20, 50, float('inf')],
            labels=['低', '中', '高', '极高']
        )

        self.price_elasticity = elasticity_df
        print(f"\n成功估计 {len(elasticity_df)} 个用户的价格弹性")
        print(f"弹性指数范围: [{elasticity_df['PSI'].min():.2f}, {elasticity_df['PSI'].max():.2f}]")
        print(f"弹性指数均值: {elasticity_df['PSI'].mean():.2f}")
        print(f"弹性指数标准差: {elasticity_df['PSI'].std():.2f}")

        return elasticity_df

    def segment_users_by_sensitivity(self, n_segments=3):
        """
        基于价格敏感度对用户进行分群

        Parameters:
        -----------
        n_segments : int
            分群数量
        """
        print(f"\n将用户分为 {n_segments} 个敏感度群组...")

        if self.price_elasticity is None:
            raise ValueError("请先运行 estimate_user_elasticity()")

        # 使用KMeans聚类
        X = self.price_elasticity[['PSI']].values

        kmeans = KMeans(n_clusters=n_segments, random_state=42, n_init=10)
        self.price_elasticity['Segment'] = kmeans.fit_predict(X)

        # 根据PSI均值排序（高敏感度 -> 低敏感度）
        segment_order = self.price_elasticity.groupby('Segment')['PSI'].mean().sort_values(ascending=False).index
        segment_map = {old: new for new, old in enumerate(segment_order)}
        self.price_elasticity['Segment'] = self.price_elasticity['Segment'].map(segment_map)

        # 添加敏感度标签
        segment_names = {
            0: '高敏感度',
            1: '中敏感度',
            2: '低敏感度'
        }
        self.price_elasticity['SensitivityLabel'] = self.price_elasticity['Segment'].map(segment_names)

        # 统计各群组
        print("\n用户分群统计:")
        for seg in sorted(self.price_elasticity['Segment'].unique()):
            seg_data = self.price_elasticity[self.price_elasticity['Segment'] == seg]
            print(
                f"  {segment_names[seg]}: {len(seg_data)}人 (占比: {len(seg_data) / len(self.price_elasticity) * 100:.1f}%)")
            print(f"    PSI均值: {seg_data['PSI'].mean():.2f}")
            print(f"    PSI范围: [{seg_data['PSI'].min():.2f}, {seg_data['PSI'].max():.2f}]")

        return self.price_elasticity

    def analyze_segment_characteristics(self):
        """分析不同敏感度群组的特征差异"""
        print("\n分析各群组特征差异...")

        # 合并用户特征
        merged = self.price_elasticity.merge(
            self.user_features, on='CustomerID', how='left'
        )

        # 计算各群组的统计量
        segment_stats = merged.groupby('SensitivityLabel').agg({
            'TotalSpent': ['mean', 'std'],
            'PurchaseCount': ['mean', 'std'],
            'AvgOrderValue': ['mean', 'std'],
            'UniqueProducts': ['mean', 'std'],
            'ActiveDays': ['mean', 'std'],
            'PurchaseFreq': ['mean', 'std'],
            'PSI': ['mean', 'std', 'count']
        }).round(2)

        print("\n各群组特征对比:")
        print(segment_stats)

        # ANOVA检验
        from scipy.stats import f_oneway

        print("\n方差分析(ANOVA)检验:")
        variables = ['TotalSpent', 'PurchaseCount', 'AvgOrderValue', 'UniqueProducts', 'ActiveDays']

        for var in variables:
            groups = [merged[merged['SensitivityLabel'] == label][var].dropna()
                      for label in merged['SensitivityLabel'].unique()]
            if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                f_stat, p_val = f_oneway(*groups)
                print(
                    f"  {var}: F={f_stat:.2f}, p={p_val:.4f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''}")

        self.segment_characteristics = merged
        return merged

    def run_pipeline(self):
        """运行完整的价格弹性分析流程"""
        self.prepare_user_item_matrix()
        self.estimate_user_elasticity()
        self.segment_users_by_sensitivity()
        self.analyze_segment_characteristics()

        return self.price_elasticity, self.segment_characteristics


# 主函数测试
if __name__ == "__main__":
    # 加载预处理数据
    clean_data = pd.read_csv("./results/tables/clean_retail_data.csv", parse_dates=['InvoiceDate'])
    user_features = pd.read_csv("./results/tables/user_features.csv")

    # 计算价格弹性
    calculator = PriceElasticityCalculator(clean_data, user_features)
    elasticity_results, segment_analysis = calculator.run_pipeline()

    # 保存结果
    elasticity_results.to_csv("./results/tables/price_elasticity_results.csv", index=False)
    segment_analysis.to_csv("./results/tables/segment_analysis.csv", index=False)

    print("\n价格弹性分析完成，结果已保存至 ./results/tables")