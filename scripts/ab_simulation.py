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
        """
        识别处理组用户（修正版）
        使用多种策略识别促销活动
        """
        print("识别处理组用户...")

        # 策略1: 尝试识别价格下降的商品
        transaction_copy = self.transaction_data.copy()
        transaction_copy['Week'] = transaction_copy['InvoiceDate'].dt.isocalendar().week

        # 计算各商品各周的平均价格
        week_price = transaction_copy.groupby(['StockCode', 'Week'])['UnitPrice'].mean().reset_index()

        # 计算价格环比变化
        week_price['PrevPrice'] = week_price.groupby('StockCode')['UnitPrice'].shift(1)
        week_price['PriceChange'] = (week_price['UnitPrice'] - week_price['PrevPrice']) / week_price['PrevPrice']

        # 促销周（47-48周对应11月底-12月初）
        promo_weeks = [47, 48]

        # 找出促销周内价格下降超过5%的商品
        price_drop_products = week_price[
            (week_price['Week'].isin(promo_weeks)) &
            (week_price['PriceChange'] < -0.05)
            ]['StockCode'].unique().tolist()

        if len(price_drop_products) >= 20:
            promo_products = price_drop_products
            print(f"  通过价格下降识别到 {len(promo_products)} 个促销商品")
        else:
            # 策略2: 选择促销周内销量激增的商品
            weekly_sales = transaction_copy.groupby(['StockCode', 'Week'])['Quantity'].sum().reset_index()
            weekly_sales['PrevSales'] = weekly_sales.groupby('StockCode')['Quantity'].shift(1)
            weekly_sales['SalesIncrease'] = (weekly_sales['Quantity'] - weekly_sales['PrevSales']) / weekly_sales[
                'PrevSales']

            surge_products = weekly_sales[
                (weekly_sales['Week'].isin(promo_weeks)) &
                (weekly_sales['SalesIncrease'] > 0.5)
                ]['StockCode'].unique().tolist()

            if len(surge_products) >= 20:
                promo_products = surge_products
                print(f"  通过销量激增识别到 {len(promo_products)} 个促销商品")
            else:
                # 策略3: 随机选择（模拟A/B测试）
                all_products = self.transaction_data['StockCode'].unique()
                np.random.seed(42)
                promo_products = np.random.choice(all_products, size=min(100, len(all_products)),
                                                  replace=False).tolist()
                print(f"  使用随机商品模拟促销: {len(promo_products)} 个商品")

        # 促销期交易数据
        promo_mask = (self.transaction_data['InvoiceDate'] >= self.promotion_start) & \
                     (self.transaction_data['InvoiceDate'] <= self.promotion_end)
        promo_data = self.transaction_data[promo_mask]

        # 处理组：在促销期购买了促销商品的用户
        treatment_users = promo_data[promo_data['StockCode'].isin(promo_products)]['CustomerID'].unique()

        # 控制组候选：在促销期有购买但未买促销商品的用户
        control_candidates = promo_data[~promo_data['StockCode'].isin(promo_products)]['CustomerID'].unique()

        # 确保处理组和对照组都有足够的样本
        if len(treatment_users) < 50:
            print(f"  警告: 处理组样本量不足({len(treatment_users)})，放宽条件")
            # 放宽条件：选择所有在促销期购买的用户作为处理组
            treatment_users = promo_data['CustomerID'].unique()
            # 选择基准期有购买但促销期未购买的用户作为对照组
            baseline_mask = (self.transaction_data['InvoiceDate'] >= self.baseline_start) & \
                            (self.transaction_data['InvoiceDate'] <= self.baseline_end)
            baseline_users = self.transaction_data[baseline_mask]['CustomerID'].unique()
            control_candidates = [u for u in baseline_users if u not in treatment_users]

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

        if len(period_data) == 0:
            # 返回空指标
            metrics = pd.DataFrame({'CustomerID': user_list})
            metrics['TotalAmount'] = 0
            metrics['OrderCount'] = 0
            metrics['TotalQuantity'] = 0
            metrics['AvgOrderValue'] = 0
            return metrics

        metrics = period_data.groupby('CustomerID').agg({
            'TotalAmount': 'sum',
            'InvoiceNo': 'nunique',
            'Quantity': 'sum'
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

    def propensity_score_matching(self, k=1, caliper=0.5):
        """
        倾向得分匹配（修正版）

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

        # 添加用户画像特征
        if 'PurchaseFreq' in self.user_features.columns:
            user_features_subset = self.user_features[['CustomerID', 'PurchaseFreq', 'ActiveDays', 'UniqueProducts']]
            match_data = match_data.merge(user_features_subset, on='CustomerID', how='left')
            match_data['LogPurchaseFreq'] = np.log(match_data['PurchaseFreq'] + 0.01)
            match_data['LogActiveDays'] = np.log(match_data['ActiveDays'] + 1)
            match_data['LogUniqueProducts'] = np.log(match_data['UniqueProducts'] + 1)

        # 匹配变量列表
        match_vars = ['LogAmount', 'LogOrderCount', 'LogAvgOrder']
        if 'LogPurchaseFreq' in match_data.columns:
            match_vars.extend(['LogPurchaseFreq', 'LogActiveDays', 'LogUniqueProducts'])

        # 处理缺失值
        match_data[match_vars] = match_data[match_vars].fillna(0)

        # 标准化
        scaler = StandardScaler()
        match_data_scaled = scaler.fit_transform(match_data[match_vars])

        # 计算倾向得分（使用Logistic回归）
        lr = LogisticRegression(max_iter=1000)
        lr.fit(match_data_scaled, match_data['Treatment'])
        match_data['PropensityScore'] = lr.predict_proba(match_data_scaled)[:, 1]

        # 分离处理组和对照组
        treatment_mask = match_data['Treatment'] == 1
        control_mask = match_data['Treatment'] == 0

        X_treatment = match_data_scaled[treatment_mask]
        X_control = match_data_scaled[control_mask]

        treatment_indices = match_data[treatment_mask].index.tolist()
        control_indices = match_data[control_mask].index.tolist()

        if len(X_treatment) == 0 or len(X_control) == 0:
            print("  错误: 处理组或对照组为空")
            return [], []

        # KNN匹配
        nn = NearestNeighbors(n_neighbors=min(k, len(X_control)), metric='euclidean')
        nn.fit(X_control)

        distances, indices = nn.kneighbors(X_treatment)

        # 获取匹配结果
        treatment_ids = match_data.loc[treatment_mask, 'CustomerID'].values
        control_ids_all = match_data.loc[control_mask, 'CustomerID'].values
        control_scores = match_data.loc[control_mask, 'PropensityScore'].values

        matched_pairs = []
        used_controls = set()

        for i, (dist, idx) in enumerate(zip(distances, indices)):
            # 检查距离是否在卡钳范围内
            if dist[0] <= caliper:
                matched_control = control_ids_all[idx[0]]
                # 确保每个对照组用户只被匹配一次
                if matched_control not in used_controls:
                    matched_pairs.append({
                        'treatment_id': treatment_ids[i],
                        'control_id': matched_control,
                        'distance': dist[0],
                        'propensity_diff': abs(
                            match_data.loc[treatment_mask, 'PropensityScore'].iloc[i] - control_scores[idx[0]])
                    })
                    used_controls.add(matched_control)

        matched_df = pd.DataFrame(matched_pairs)

        if len(matched_df) == 0:
            print("  警告: 没有找到匹配对，使用最近邻匹配")
            # 降级：不使用卡钳限制
            for i, (dist, idx) in enumerate(zip(distances, indices)):
                matched_control = control_ids_all[idx[0]]
                if matched_control not in used_controls:
                    matched_pairs.append({
                        'treatment_id': treatment_ids[i],
                        'control_id': matched_control,
                        'distance': dist[0],
                        'propensity_diff': abs(
                            match_data.loc[treatment_mask, 'PropensityScore'].iloc[i] - control_scores[idx[0]])
                    })
                    used_controls.add(matched_control)
            matched_df = pd.DataFrame(matched_pairs)

        matched_treatment = matched_df['treatment_id'].unique()
        matched_control = matched_df['control_id'].unique()

        # 确保1:1匹配
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
            if var not in treatment_data.columns or var not in control_data.columns:
                continue

            t_mean = treatment_data[var].mean()
            c_mean = control_data[var].mean()

            t_std = treatment_data[var].std()
            c_std = control_data[var].std()
            pooled_std = np.sqrt((t_std ** 2 + c_std ** 2) / 2)

            if pooled_std > 0:
                std_diff = (t_mean - c_mean) / pooled_std
            else:
                std_diff = 0

            # t检验
            from scipy.stats import ttest_ind
            t_stat, p_val = ttest_ind(treatment_data[var], control_data[var])

            status = "✓ 平衡" if abs(std_diff) < 0.2 else "⚠ 不平衡"
            print(f"  {var}: 标准化偏差={abs(std_diff) * 100:.2f}%, p={p_val:.4f} {status}")

    def did_analysis(self):
        """
        双重差分法分析（修正版）
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

        # 添加用户固定效应（可选）
        panel_data['CustomerID'] = panel_data['CustomerID'].astype('category')

        # DID回归
        # 模型1: 使用金额作为结果变量
        try:
            model1 = ols('LogAmount ~ Treatment + Period + Treat_Period', data=panel_data).fit()
        except Exception as e:
            print(f"  模型1拟合失败: {e}")
            model1 = None

        # 模型2: 使用订单数作为结果变量
        try:
            model2 = ols('LogOrderCount ~ Treatment + Period + Treat_Period', data=panel_data).fit()
        except Exception as e:
            print(f"  模型2拟合失败: {e}")
            model2 = None

        # 模型3: 加入更多控制变量
        try:
            model3 = ols('LogAmount ~ Treatment + Period + Treat_Period + LogOrderCount', data=panel_data).fit()
        except Exception as e:
            print(f"  模型3拟合失败: {e}")
            model3 = None

        if model1 is not None:
            print("\n模型1 (金额) 结果:")
            print(model1.summary())

            did_coef = model1.params.get('Treat_Period', 0)
            lift_pct = (np.exp(did_coef) - 1) * 100
            print(f"\n促销效应: 消费金额提升 {lift_pct:.1f}%")

        if model2 is not None:
            print("\n模型2 (订单数) 结果:")
            print(model2.summary())

        if model3 is not None:
            print("\n模型3 (加控制变量) 结果:")
            print(model3.summary())

        # 平行趋势检验
        parallel_results = self.parallel_trend_test()

        self.did_results = {
            'model1': model1,
            'model2': model2,
            'model3': model3,
            'panel_data': panel_data,
            'parallel_results': parallel_results
        }

        return model1, model2, model3

    def parallel_trend_test(self):
        """
        平行趋势检验（完全修复版）
        """
        print("\n平行趋势检验...")

        results = []

        # 确定可用的用户
        treatment_users = []
        control_users = []

        if hasattr(self, 'matched_treatment') and len(self.matched_treatment) > 0:
            treatment_users = self.matched_treatment
            control_users = self.matched_control
            print(f"  使用匹配样本: 处理组 {len(treatment_users)}人, 对照组 {len(control_users)}人")
        elif hasattr(self, 'treatment_users') and len(self.treatment_users) > 0:
            # 如果匹配样本不足，使用原始样本
            treatment_users = list(self.treatment_users)[:100] if len(self.treatment_users) > 100 else list(
                self.treatment_users)
            control_users = list(self.control_candidates)[:100] if len(self.control_candidates) > 100 else list(
                self.control_candidates)
            print(f"  使用原始样本: 处理组 {len(treatment_users)}人, 对照组 {len(control_users)}人")
        else:
            print("  错误: 没有可用的用户进行平行趋势检验")
            return pd.DataFrame()

        if len(treatment_users) == 0 or len(control_users) == 0:
            print("  错误: 处理组或对照组为空")
            return pd.DataFrame()

        # 计算基准期的均值（作为基期）
        try:
            baseline_treat_metrics = self.calculate_user_metrics(
                treatment_users, self.baseline_start, self.baseline_end
            )
            baseline_control_metrics = self.calculate_user_metrics(
                control_users, self.baseline_start, self.baseline_end
            )
        except Exception as e:
            print(f"  计算基准期指标失败: {e}")
            return pd.DataFrame()

        baseline_treat_mean = baseline_treat_metrics['TotalAmount'].mean()
        baseline_control_mean = baseline_control_metrics['TotalAmount'].mean()

        print(f"  基准期均值 - 处理组: {baseline_treat_mean:.2f}, 对照组: {baseline_control_mean:.2f}")

        # 检验促销前4周
        from scipy.stats import ttest_ind

        for i in range(4, 0, -1):
            # 计算每周的起止日期
            week_end = pd.to_datetime(self.baseline_start) - pd.Timedelta(days=1)
            week_start = week_end - pd.Timedelta(days=6)

            # 对于T-4, T-3, T-2, T-1，需要依次向前推
            week_end = week_end - pd.Timedelta(days=7 * (4 - i))
            week_start = week_start - pd.Timedelta(days=7 * (4 - i))

            # 确保时间范围有效
            min_date = self.transaction_data['InvoiceDate'].min()
            if week_start < min_date:
                print(f"  跳过 {week_label}: 超出数据范围")
                continue

            week_label = f'T-{i}'

            try:
                treat_metrics = self.calculate_user_metrics(
                    treatment_users,
                    week_start.strftime('%Y-%m-%d'),
                    week_end.strftime('%Y-%m-%d')
                )
                control_metrics = self.calculate_user_metrics(
                    control_users,
                    week_start.strftime('%Y-%m-%d'),
                    week_end.strftime('%Y-%m-%d')
                )
            except Exception as e:
                print(f"  计算第{i}周指标失败: {e}")
                continue

            # 使用均值
            treat_avg = treat_metrics['TotalAmount'].mean()
            control_avg = control_metrics['TotalAmount'].mean()

            # 计算变化率
            if baseline_treat_mean > 0:
                treat_change = ((treat_avg - baseline_treat_mean) / baseline_treat_mean) * 100
            else:
                treat_change = 0

            if baseline_control_mean > 0:
                control_change = ((control_avg - baseline_control_mean) / baseline_control_mean) * 100
            else:
                control_change = 0

            # t检验
            try:
                t_stat, p_val = ttest_ind(treat_metrics['TotalAmount'], control_metrics['TotalAmount'])
            except Exception as e:
                print(f"  t检验失败: {e}")
                p_val = 1.0

            results.append({
                'Week': week_label,
                'Treatment_Mean': round(treat_avg, 2),
                'Control_Mean': round(control_avg, 2),
                'Treatment_Change': round(treat_change, 2),
                'Control_Change': round(control_change, 2),
                'Difference': round(treat_avg - control_avg, 2),
                'P_Value': round(p_val, 4),
                'Significant': p_val < 0.05 if not np.isnan(p_val) else False
            })

        if len(results) == 0:
            print("  警告: 没有成功计算任何一周的数据")
            return pd.DataFrame()

        results_df = pd.DataFrame(results)

        print("\n平行趋势检验结果:")
        print(
            results_df[['Week', 'Treatment_Mean', 'Control_Mean', 'Difference', 'P_Value', 'Significant']].to_string())

        # 判断是否平行
        if results_df['Significant'].any():
            print("\n⚠ 警告: 部分时期存在显著差异，平行趋势假设可能不成立")
            print("  建议: 使用事件研究法或合成控制法作为稳健性检验")
        else:
            print("\n✓ 平行趋势假设成立")

        return results_df

    def heterogeneous_effect_analysis(self):
        """异质性分析：不同敏感度群体的响应差异（修正版）"""
        print("\n异质性分析...")

        results = []

        # 获取处理组的弹性数据
        treat_elasticity = self.elasticity_data[self.elasticity_data['CustomerID'].isin(self.matched_treatment)]

        if len(treat_elasticity) == 0:
            print("  警告: 处理组中没有弹性数据，使用全部弹性数据")
            treat_elasticity = self.elasticity_data

        # 按敏感度分组（基于分位数）
        psi_values = treat_elasticity['PSI'].values
        high_threshold = np.percentile(psi_values, 66) if len(psi_values) > 0 else 60
        low_threshold = np.percentile(psi_values, 33) if len(psi_values) > 0 else 40

        high_sensitive = treat_elasticity[treat_elasticity['PSI'] >= high_threshold]['CustomerID'].tolist()
        mid_sensitive = treat_elasticity[(treat_elasticity['PSI'] >= low_threshold) &
                                         (treat_elasticity['PSI'] < high_threshold)]['CustomerID'].tolist()
        low_sensitive = treat_elasticity[treat_elasticity['PSI'] < low_threshold]['CustomerID'].tolist()

        print(f"  高敏感组: {len(high_sensitive)}人 (PSI>={high_threshold:.1f})")
        print(f"  中敏感组: {len(mid_sensitive)}人")
        print(f"  低敏感组: {len(low_sensitive)}人 (PSI<{low_threshold:.1f})")

        # 计算对照组均值（用于DID计算）
        control_baseline = self.calculate_user_metrics(self.matched_control, self.baseline_start, self.baseline_end)
        control_promo = self.calculate_user_metrics(self.matched_control, self.promotion_start, self.promotion_end)
        control_baseline_mean = control_baseline['TotalAmount'].mean()
        control_promo_mean = control_promo['TotalAmount'].mean()
        control_change = control_promo_mean - control_baseline_mean

        for group_name, group_users in [('高敏感', high_sensitive), ('中敏感', mid_sensitive),
                                        ('低敏感', low_sensitive)]:
            if len(group_users) == 0:
                results.append({
                    'Group': group_name,
                    'Sample_Size': 0,
                    'Baseline_Amount': 0,
                    'Promo_Amount': 0,
                    'DID_Effect': 0,
                    'Relative_Effect': 0
                })
                continue

            # 计算基准期和促销期金额
            baseline = self.calculate_user_metrics(group_users, self.baseline_start, self.baseline_end)
            promo = self.calculate_user_metrics(group_users, self.promotion_start, self.promotion_end)

            baseline_mean = baseline['TotalAmount'].mean()
            promo_mean = promo['TotalAmount'].mean()

            # 组内变化
            group_change = promo_mean - baseline_mean

            # DID估计量 = (处理组变化) - (对照组变化)
            did_effect = group_change - control_change

            # 相对效应（百分比）
            if baseline_mean > 0:
                relative_effect = (did_effect / baseline_mean) * 100
            else:
                relative_effect = did_effect * 100 if did_effect > 0 else 0

            results.append({
                'Group': group_name,
                'Sample_Size': len(group_users),
                'Baseline_Amount': round(baseline_mean, 2),
                'Promo_Amount': round(promo_mean, 2),
                'Group_Change': round(group_change, 2),
                'Control_Change': round(control_change, 2),
                'DID_Effect': round(did_effect, 2),
                'Relative_Effect': round(relative_effect, 1)
            })

        results_df = pd.DataFrame(results)
        print("\n异质性分析结果:")
        print(results_df[['Group', 'Sample_Size', 'Baseline_Amount', 'Promo_Amount', 'DID_Effect',
                          'Relative_Effect']].to_string())

        # 计算响应倍数
        if len(results_df) >= 2:
            high_effect = results_df[results_df['Group'] == '高敏感']['Relative_Effect'].values
            low_effect = results_df[results_df['Group'] == '低敏感']['Relative_Effect'].values
            if len(high_effect) > 0 and len(low_effect) > 0 and low_effect[0] != 0:
                ratio = high_effect[0] / low_effect[0]
                print(f"\n高敏感组响应强度是低敏感组的 {ratio:.1f} 倍")

        return results_df

    def discount_elasticity_analysis(self):
        """
        折扣力度弹性分析（修正版）
        基于文献和行业数据的合理模拟
        """
        print("\n折扣力度分析...")

        # 折扣力度范围
        discount_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

        # 价格弹性系数（基于文献：电商商品弹性通常在-1.0到-2.5之间）
        # 参考: 章韬等(2025)的研究
        base_elasticity = -1.8

        results = []

        for discount in discount_levels:
            # 考虑边际递减效应
            if discount <= 0.15:
                elasticity = base_elasticity * 1.0  # 正常弹性
            elif discount <= 0.25:
                elasticity = base_elasticity * 0.8  # 弹性减弱
            else:
                elasticity = base_elasticity * 0.5  # 深度折扣弹性大幅减弱

            # 销量提升百分比 = |弹性| × 折扣率 × 100
            demand_lift = abs(elasticity) * discount * 100

            # 考虑促销溢出效应（吸引新用户）
            if discount <= 0.20:
                spillover = 1.0
            elif discount <= 0.30:
                spillover = 1.2
            else:
                spillover = 0.8  # 深度折扣可能损害品牌形象

            demand_lift = demand_lift * spillover

            # ROI计算
            # 假设初始销售额为100单位
            initial_sales = 100
            incremental_sales = initial_sales * (demand_lift / 100)

            # 折扣成本 = 折扣率 × (初始销售额 + 增量销售额) × 使用率
            usage_rate = min(0.9, 0.5 + discount)  # 折扣越大使用率越高
            discount_cost = discount * (initial_sales + incremental_sales) * usage_rate

            if discount_cost > 0:
                roi = (incremental_sales - discount_cost) / discount_cost
            else:
                roi = 0

            results.append({
                'Discount': f'{discount * 100:.0f}%',
                'Discount_Rate': discount,
                'Elasticity_Used': round(elasticity, 2),
                'Predicted_Demand_Change': round(demand_lift, 1),
                'Estimated_ROI': round(roi, 2)
            })

        results_df = pd.DataFrame(results)
        print("\n折扣力度分析结果:")
        print(results_df.to_string())

        # 找出最优折扣
        best_idx = results_df['Estimated_ROI'].idxmax()
        best_discount = results_df.loc[best_idx, 'Discount']
        best_roi = results_df.loc[best_idx, 'Estimated_ROI']
        print(f"\n✓ 最优折扣区间: {best_discount} (ROI={best_roi:.2f})")

        # 输出建议
        if best_roi < 1:
            print("  注意: ROI小于1，建议优化促销成本结构")
        else:
            print(f"  建议: 采用{best_discount}折扣可实现最佳投入产出比")

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
    import os

    # 加载数据
    clean_data = pd.read_csv("./results/tables/clean_retail_data.csv", parse_dates=['InvoiceDate'])
    user_features = pd.read_csv("./results/tables/user_features.csv")
    elasticity_data = pd.read_csv("./results/tables/price_elasticity_results.csv")

    # 运行A/B模拟
    simulator = ABSimulation(clean_data, user_features, elasticity_data)
    results = simulator.run_pipeline()

    print("\nA/B模拟分析完成!")
