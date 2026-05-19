"""
A/B离线模拟与促销效果评估模块
使用倾向得分匹配(PSM)和双重差分法(DID)评估促销因果效应
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class ABSimulation:
    """A/B离线模拟器"""

    def __init__(self, transaction_data, user_features, elasticity_data):
        """
        初始化

        Parameters:
        -----------
        transaction_data : DataFrame
            交易数据
        user_features : DataFrame
            用户特征
        elasticity_data : DataFrame
            价格弹性数据
        """
        self.transaction_data = transaction_data
        self.user_features = user_features
        self.elasticity_data = elasticity_data

        # 识别促销期（2011年11月最后一周模拟促销）
        self.promotion_start = '2011-11-21'
        self.promotion_end = '2011-12-04'

        # 基准期（促销前4周）
        self.baseline_start = '2011-10-24'
        self.baseline_end = '2011-11-20'

    def identify_treatment_users(self):
        """识别处理组用户（在促销期购买了促销商品的用户）"""
        print("识别处理组用户...")

        # 标记促销期交易
        promo_mask = (self.transaction_data['InvoiceDate'] >= self.promotion_start) & \
                     (self.transaction_data['InvoiceDate'] <= self.promotion_end)

        promo_data = self.transaction_data[promo_mask]

        # 定义促销商品（这里选择销量前20%的商品作为促销商品）
        product_sales = self.transaction_data.groupby('StockCode')['TotalAmount'].sum()
        threshold = product_sales.quantile(0.8)
        promo_products = product_sales[product_sales >= threshold].index.tolist()

        # 处理组：购买了促销商品的用户
        treatment_users = promo_data[promo_data['StockCode'].isin(promo_products)]['CustomerID'].unique()

        # 控制组候选：在促销期购买但未买促销商品的用户
        control_candidates = promo_data[~promo_data['StockCode'].isin(promo_products)]['CustomerID'].unique()

        print(f"  处理组用户数: {len(treatment_users)}")
        print(f"  控制组候选数: {len(control_candidates)}")
        print(f"  促销商品数: {len(promo_products)}")

        self.treatment_users = treatment_users
        self.control_candidates = control_candidates
        self.promo_products = promo_products

        return treatment_users, control_candidates

    def calculate_user_metrics(self, user_list, start_date, end_date):
        """
        计算指定用户在特定时期内的指标

        Parameters:
        -----------
        user_list : array
            用户ID列表
        start_date : str
            开始日期
        end_date : str
            结束日期
        """
        mask = (self.transaction_data['InvoiceDate'] >= start_date) & \
               (self.transaction_data['InvoiceDate'] <= end_date)

        period_data = self.transaction_data[mask & self.transaction_data['CustomerID'].isin(user_list)]

        metrics = period_data.groupby('CustomerID').agg({
            'TotalAmount': 'sum',
            'InvoiceNo': 'nunique',
            'Quantity': 'sum',
            'TotalAmount': 'count'
        }).reset_index()

        metrics.columns = ['CustomerID', 'TotalAmount', 'OrderCount', 'TotalQuantity']

        # 添加缺失的用户（金额为0）
        all_users = pd.DataFrame({'CustomerID': user_list})
        metrics = all_users.merge(metrics, on='CustomerID', how='left').fillna(0)

        # 计算平均订单金额
        metrics['AvgOrderValue'] = metrics.apply(
            lambda x: x['TotalAmount'] / x['OrderCount'] if x['OrderCount'] > 0 else 0, axis=1
        )

        return metrics

    def propensity_score_matching(self, k=1, caliper=0.25):
        """
        倾向得分匹配

        Parameters:
        -----------
        k : int
            匹配数量（1:1匹配）
        caliper : float
            卡钳值
        """
        print("\n执行倾向得分匹配...")

        # 计算基准期指标
        baseline_treatment = self.calculate_user_metrics(
            self.treatment_users, self.baseline_start, self.baseline_end
        )
        baseline_control = self.calculate_user_metrics(
            self.control_candidates, self.baseline_start, self.baseline_end
        )

        # 标记处理状态
        baseline_treatment['Treatment'] = 1
        baseline_control['Treatment'] = 0

        # 合并数据
        match_data = pd.concat([baseline_treatment, baseline_control], ignore_index=True)

        # 选择匹配变量（log转换减少偏态）
        match_data['LogAmount'] = np.log(match_data['TotalAmount'] + 1)
        match_data['LogOrderCount'] = np.log(match_data['OrderCount'] + 1)
        match_data['LogAvgOrder'] = np.log(match_data['AvgOrderValue'] + 1)

        # 标准化
        scaler = StandardScaler()
        match_vars = ['LogAmount', 'LogOrderCount', 'LogAvgOrder']
        match_data_scaled = scaler.fit_transform(match_data[match_vars])

        # 分离处理组和对照组
        treatment_mask = match_data['Treatment'] == 1
        control_mask = match_data['Treatment'] == 0

        X_treatment = match_data_scaled[treatment_mask]
        X_control = match_data_scaled[control_mask]

        # 计算倾向得分（使用Logistic回归）
        lr = LogisticRegression()
        lr.fit(match_data_scaled, match_data['Treatment'])
        match_data['PropensityScore'] = lr.predict_proba(match_data_scaled)[:, 1]

        # KNN匹配
        nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        nn.fit(X_control)

        distances, indices = nn.kneighbors(X_treatment)

        # 获取匹配结果
        treatment_ids = match_data.loc[treatment_mask, 'CustomerID'].values
        control_ids_all = match_data.loc[control_mask, 'CustomerID'].values
        control_scores = match_data.loc[control_mask, 'PropensityScore'].values

        matched_pairs = []
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if dist[0] <= caliper:  # 卡钳筛选
                matched_control = control_ids_all[idx[0]]
                matched_pairs.append({
                    'treatment_id': treatment_ids[i],
                    'control_id': matched_control,
                    'distance': dist[0],
                    'propensity_diff': abs(
                        match_data.loc[treatment_mask, 'PropensityScore'].iloc[i] - control_scores[idx[0]])
                })

        matched_df = pd.DataFrame(matched_pairs)
        matched_treatment = matched_df['treatment_id'].values
        matched_control = matched_df['control_id'].values

        # 确保1:1匹配
        matched_treatment = np.unique(matched_treatment)
        matched_control = np.unique(matched_control)

        # 如果数量不均衡，截断到较小长度
        min_len = min(len(matched_treatment), len(matched_control))
        matched_treatment = matched_treatment[:min_len]
        matched_control = matched_control[:min_len]

        print(f"匹配完成: 处理组 {len(matched_treatment)}人, 对照组 {len(matched_control)}人")

        # 平衡性检验
        self.check_balance(match_data, matched_treatment, matched_control, match_vars)

        self.matched_treatment = matched_treatment
        self.matched_control = matched_control
        self.match_data = match_data

        return matched_treatment, matched_control

    def check_balance(self, data, treatment_ids, control_ids, variables):
        """平衡性检验"""
        print("\n平衡性检验:")

        treatment_data = data[data['CustomerID'].isin(treatment_ids)]
        control_data = data[data['CustomerID'].isin(control_ids)]

        for var in variables:
            t_mean = treatment_data[var].mean()
            c_mean = control_data[var].mean()
            std_diff = (t_mean - c_mean) / np.sqrt((treatment_data[var].var() + control_data[var].var()) / 2)

            # t检验
            from scipy.stats import ttest_ind
            t_stat, p_val = ttest_ind(treatment_data[var], control_data[var])

            print(f"  {var}: 标准化偏差={abs(std_diff) * 100:.2f}%, p={p_val:.4f}")

    def did_analysis(self):
        """
        双重差分法分析
        """
        print("\n执行双重差分分析...")

        # 计算各时期指标
        # 基准期
        baseline_treat = self.calculate_user_metrics(self.matched_treatment, self.baseline_start, self.baseline_end)
        baseline_control = self.calculate_user_metrics(self.matched_control, self.baseline_start, self.baseline_end)

        # 促销期
        promo_treat = self.calculate_user_metrics(self.matched_treatment, self.promotion_start, self.promotion_end)
        promo_control = self.calculate_user_metrics(self.matched_control, self.promotion_start, self.promotion_end)

        # 添加标识
        baseline_treat['Period'] = 0
        baseline_treat['Treatment'] = 1
        baseline_control['Period'] = 0
        baseline_control['Treatment'] = 0

        promo_treat['Period'] = 1
        promo_treat['Treatment'] = 1
        promo_control['Period'] = 1
        promo_control['Treatment'] = 0

        # 合并面板数据
        panel_data = pd.concat([
            baseline_treat, baseline_control,
            promo_treat, promo_control
        ], ignore_index=True)

        # 交互项
        panel_data['Treat_Period'] = panel_data['Treatment'] * panel_data['Period']

        # 结果变量变换（避免0值问题）
        panel_data['LogAmount'] = np.log(panel_data['TotalAmount'] + 1)
        panel_data['LogOrderCount'] = np.log(panel_data['OrderCount'] + 1)

        # DID回归
        # 模型1: 使用金额作为结果变量
        model1 = ols('LogAmount ~ Treatment + Period + Treat_Period', data=panel_data).fit()

        # 模型2: 使用订单数作为结果变量
        model2 = ols('LogOrderCount ~ Treatment + Period + Treat_Period', data=panel_data).fit()

        # 模型3: 加入更多控制变量
        model3 = ols('LogAmount ~ Treatment + Period + Treat_Period + LogOrderCount', data=panel_data).fit()

        print("\n模型1 (金额) 结果:")
        print(model1.summary())

        print("\n模型2 (订单数) 结果:")
        print(model2.summary())

        print("\n模型3 (加控制变量) 结果:")
        print(model3.summary())

        # 平行趋势检验
        self.parallel_trend_test()

        self.did_results = {
            'model1': model1,
            'model2': model2,
            'model3': model3,
            'panel_data': panel_data
        }

        return model1, model2, model3

    def parallel_trend_test(self):
        """平行趋势检验"""
        print("\n平行趋势检验...")

        # 促销前4周每周的指标
        weeks = []
        for i in range(4, 0, -1):
            week_start = pd.to_datetime(self.baseline_end) - pd.Timedelta(days=7 * i)
            week_end = week_start + pd.Timedelta(days=6)
            weeks.append((week_start.strftime('%Y-%m-%d'), week_end.strftime('%Y-%m-%d'), f'T-{i}'))

        results = []
        for week_start, week_end, week_label in weeks:
            treat_metrics = self.calculate_user_metrics(self.matched_treatment, week_start, week_end)
            control_metrics = self.calculate_user_metrics(self.matched_control, week_start, week_end)

            treat_avg = treat_metrics['TotalAmount'].mean()
            control_avg = control_metrics['TotalAmount'].mean()

            # t检验
            from scipy.stats import ttest_ind
            t_stat, p_val = ttest_ind(treat_metrics['TotalAmount'], control_metrics['TotalAmount'])

            results.append({
                'Week': week_label,
                'Treatment_Mean': treat_avg,
                'Control_Mean': control_avg,
                'Difference': treat_avg - control_avg,
                'P_Value': p_val,
                'Significant': p_val < 0.05
            })

        results_df = pd.DataFrame(results)
        print("\n平行趋势检验结果:")
        print(results_df.to_string())

        # 判断是否平行（所有p值>0.05）
        if results_df['Significant'].any():
            print("\n警告: 部分时期存在显著差异，平行趋势假设可能不成立")
        else:
            print("\n平行趋势假设成立")

        return results_df

    def heterogeneous_effect_analysis(self):
        """异质性分析：不同敏感度群体的响应差异"""
        print("\n异质性分析...")

        # 合并弹性数据
        treat_elasticity = self.elasticity_data[self.elasticity_data['CustomerID'].isin(self.matched_treatment)]

        # 按敏感度分组
        high_sensitive = treat_elasticity[treat_elasticity['SensitivityLabel'] == '高敏感度']['CustomerID'].tolist()
        mid_sensitive = treat_elasticity[treat_elasticity['SensitivityLabel'] == '中敏感度']['CustomerID'].tolist()
        low_sensitive = treat_elasticity[treat_elasticity['SensitivityLabel'] == '低敏感度']['CustomerID'].tolist()

        # 计算各组DID效应
        results = []

        for group_name, group_users in [('高敏感', high_sensitive), ('中敏感', mid_sensitive),
                                        ('低敏感', low_sensitive)]:
            if len(group_users) == 0:
                continue

            # 计算基准期和促销期金额
            baseline = self.calculate_user_metrics(group_users, self.baseline_start, self.baseline_end)
            promo = self.calculate_user_metrics(group_users, self.promotion_start, self.promotion_end)

            baseline_mean = baseline['TotalAmount'].mean()
            promo_mean = promo['TotalAmount'].mean()

            # 计算对照组相应指标
            control_baseline = self.calculate_user_metrics(self.matched_control, self.baseline_start, self.baseline_end)
            control_promo = self.calculate_user_metrics(self.matched_control, self.promotion_start, self.promotion_end)

            control_baseline_mean = control_baseline['TotalAmount'].mean()
            control_promo_mean = control_promo['TotalAmount'].mean()

            # 计算DID估计量
            did_effect = (promo_mean - baseline_mean) - (control_promo_mean - control_baseline_mean)
            relative_effect = did_effect / (baseline_mean + 1)

            results.append({
                'Group': group_name,
                'Sample_Size': len(group_users),
                'Baseline_Amount': baseline_mean,
                'Promo_Amount': promo_mean,
                'Control_Baseline': control_baseline_mean,
                'Control_Promo': control_promo_mean,
                'DID_Effect': did_effect,
                'Relative_Effect': relative_effect * 100
            })

        results_df = pd.DataFrame(results)
        print("\n异质性分析结果:")
        print(results_df.to_string())

        return results_df

    def discount_elasticity_analysis(self):
        """折扣力度弹性分析"""
        print("\n折扣力度分析...")

        # 模拟不同折扣力度的效果
        # 由于原数据没有折扣信息，我们基于价格分布进行模拟

        # 获取促销商品的原始价格
        promo_products_prices = self.transaction_data[
            self.transaction_data['StockCode'].isin(self.promo_products)
        ].groupby('StockCode')['UnitPrice'].mean()

        discount_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

        results = []
        for discount in discount_levels:
            # 模拟：折扣后的预期销量提升（基于价格弹性）
            # 这里使用平均弹性系数进行模拟
            avg_elasticity = self.elasticity_data['PSI'].mean() / 100  # 转换为弹性系数

            # 价格变化百分比
            price_change = -discount  # 价格下降

            # 根据弹性公式: 销量变化% = 弹性 * 价格变化%
            demand_change = avg_elasticity * price_change

            # ROI计算: (增量收入 - 折扣成本) / 折扣成本
            # 简化模拟
            incremental_sales = demand_change
            discount_cost = discount
            roi = (incremental_sales - discount_cost) / discount_cost if discount_cost > 0 else 0

            results.append({
                'Discount': f'{discount * 100:.0f}%',
                'Discount_Rate': discount,
                'Predicted_Demand_Change': demand_change * 100,
                'Estimated_ROI': roi
            })

        results_df = pd.DataFrame(results)
        print("\n折扣力度分析结果:")
        print(results_df.to_string())

        # 找出最优折扣区间
        max_roi_idx = results_df['Estimated_ROI'].idxmax()
        print(
            f"\n最优折扣区间: {results_df.loc[max_roi_idx, 'Discount']} (ROI={results_df.loc[max_roi_idx, 'Estimated_ROI']:.2f})")

        return results_df

    def run_pipeline(self):
        """运行完整的A/B模拟分析"""
        print("=" * 60)
        print("开始A/B离线模拟分析")
        print("=" * 60)

        self.identify_treatment_users()
        self.propensity_score_matching()
        self.did_analysis()
        self.heterogeneous_effect_analysis()
        self.discount_elasticity_analysis()

        return self.did_results


# 主函数测试
if __name__ == "__main__":
    # 加载数据
    clean_data = pd.read_csv("./results/tables/clean_retail_data.csv", parse_dates=['InvoiceDate'])
    user_features = pd.read_csv("./results/tables/user_features.csv")
    elasticity_data = pd.read_csv("./results/tables/price_elasticity_results.csv")

    # 运行A/B模拟
    simulator = ABSimulation(clean_data, user_features, elasticity_data)
    results = simulator.run_pipeline()

    print("\nA/B模拟分析完成!")