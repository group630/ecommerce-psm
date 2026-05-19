"""
可视化分析模块
生成论文所需的图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置绘图风格
sns.set_style("whitegrid")
sns.set_palette("husl")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class Visualizer:
    """可视化器"""

    def __init__(self, output_dir="../results/figures/"):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)

    def plot_price_elasticity_distribution(self, elasticity_data):
        """图1: 价格弹性指数分布"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 直方图
        axes[0].hist(elasticity_data['PSI'], bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(elasticity_data['PSI'].mean(), color='red', linestyle='--',
                        label=f'均值: {elasticity_data["PSI"].mean():.1f}')
        axes[0].axvline(elasticity_data['PSI'].median(), color='green', linestyle='--',
                        label=f'中位数: {elasticity_data["PSI"].median():.1f}')
        axes[0].set_xlabel('价格弹性指数 (PSI)')
        axes[0].set_ylabel('用户数')
        axes[0].set_title('价格弹性指数分布')
        axes[0].legend()

        # 箱线图（按敏感度分组）
        segment_order = ['高敏感度', '中敏感度', '低敏感度']
        segment_data = [elasticity_data[elasticity_data['SensitivityLabel'] == seg]['PSI']
                        for seg in segment_order]

        bp = axes[1].boxplot(segment_data, labels=segment_order, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#ff9999', '#66b3ff', '#99ff99']):
            patch.set_facecolor(color)
        axes[1].set_ylabel('价格弹性指数 (PSI)')
        axes[1].set_title('不同敏感度群组的PSI分布')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}fig1_elasticity_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"图表已保存: fig1_elasticity_distribution.png")

    def plot_segment_characteristics(self, segment_analysis):
        """图2: 不同敏感度群组特征对比"""
        # 计算各群组均值
        segment_means = segment_analysis.groupby('SensitivityLabel').agg({
            'AvgOrderValue': 'mean',
            'PurchaseCount': 'mean',
            'UniqueProducts': 'mean',
            'ActiveDays': 'mean'
        }).reset_index()

        # 标准化以便比较
        for col in ['AvgOrderValue', 'PurchaseCount', 'UniqueProducts', 'ActiveDays']:
            segment_means[f'{col}_norm'] = segment_means[col] / segment_means[col].max()

        # 雷达图
        categories = ['平均客单价', '购买频次', '品类丰富度', '活跃天数']

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        colors = {'高敏感度': '#ff6b6b', '中敏感度': '#4ecdc4', '低敏感度': '#45b7d1'}

        for _, row in segment_means.iterrows():
            values = [row['AvgOrderValue_norm'], row['PurchaseCount_norm'],
                      row['UniqueProducts_norm'], row['ActiveDays_norm']]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=row['SensitivityLabel'],
                    color=colors[row['SensitivityLabel']])
            ax.fill(angles, values, alpha=0.1, color=colors[row['SensitivityLabel']])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('不同价格敏感度群组的特征雷达图', size=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}fig2_segment_radar.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"图表已保存: fig2_segment_radar.png")

        # 柱状图
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(categories))
        width = 0.25

        for i, (_, row) in enumerate(segment_means.iterrows()):
            values = [row['AvgOrderValue_norm'], row['PurchaseCount_norm'],
                      row['UniqueProducts_norm'], row['ActiveDays_norm']]
            offset = (i - 1) * width
            ax.bar(x + offset, values, width, label=row['SensitivityLabel'], color=colors[row['SensitivityLabel']])

        ax.set_xlabel('特征维度')
        ax.set_ylabel('标准化得分')
        ax.set_title('不同价格敏感度群组的特征对比')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}fig2_segment_bars.png", dpi=300, bbox_inches='tight')
        plt.show()

    def plot_did_results(self, did_results):
        """图3: DID分析结果"""
        panel_data = did_results['panel_data']

        # 计算各组各时期的均值
        summary = panel_data.groupby(['Treatment', 'Period']).agg({
            'TotalAmount': 'mean',
            'OrderCount': 'mean'
        }).reset_index()

        summary['Group'] = summary['Treatment'].map({1: '处理组', 0: '对照组'})
        summary['Period_Label'] = summary['Period'].map({0: '基准期', 1: '促销期'})

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 金额变化
        for group in ['处理组', '对照组']:
            data = summary[summary['Group'] == group]
            axes[0].plot(data['Period_Label'], data['TotalAmount'], 'o-', label=group, markersize=8, linewidth=2)
        axes[0].set_xlabel('时期')
        axes[0].set_ylabel('平均消费金额 (英镑)')
        axes[0].set_title('DID分析: 消费金额变化')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 订单数变化
        for group in ['处理组', '对照组']:
            data = summary[summary['Group'] == group]
            axes[1].plot(data['Period_Label'], data['OrderCount'], 'o-', label=group, markersize=8, linewidth=2)
        axes[1].set_xlabel('时期')
        axes[1].set_ylabel('平均订单数')
        axes[1].set_title('DID分析: 订单数变化')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}fig3_did_results.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"图表已保存: fig3_did_results.png")

    def plot_heterogeneous_effects(self, hetero_results):
        """图4: 异质性分析结果"""
        fig, ax = plt.subplots(figsize=(10, 6))

        groups = hetero_results['Group'].tolist()
        effects = hetero_results['Relative_Effect'].tolist()

        bars = ax.bar(groups, effects, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # 添加数值标签
        for bar, effect in zip(bars, effects):
            height = bar.get_height()
            ax.annotate(f'{effect:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

        ax.set_xlabel('用户群组')
        ax.set_ylabel('促销效应 (相对提升百分比)')
        ax.set_title('不同价格敏感度群组的促销响应差异')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}fig4_heterogeneous_effects.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"图表已保存: fig4_heterogeneous_effects.png")

    def plot_discount_analysis(self, discount_results):
        """图5: 折扣力度分析"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        discounts = discount_results['Discount_Rate'].values
        demand_change = discount_results['Predicted_Demand_Change'].values
        roi = discount_results['Estimated_ROI'].values

        # 需求变化
        axes[0].plot(discounts * 100, demand_change, 'o-', color='steelblue', linewidth=2, markersize=8)
        axes[0].set_xlabel('折扣力度 (%)')
        axes[0].set_ylabel('需求变化 (%)')
        axes[0].set_title('折扣力度与需求变化的关系')
        axes[0].grid(True, alpha=0.3)

        # ROI
        axes[1].plot(discounts * 100, roi, 'o-', color='coral', linewidth=2, markersize=8)
        axes[1].axhline(y=1, color='red', linestyle='--', label='ROI=1 (盈亏平衡)')
        axes[1].set_xlabel('折扣力度 (%)')
        axes[1].set_ylabel('估计ROI')
        axes[1].set_title('折扣力度与ROI的关系')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 标记最优区间
        optimal_idx = roi.argmax()
        axes[1].annotate(f'最优: {discounts[optimal_idx] * 100:.0f}%',
                         xy=(discounts[optimal_idx] * 100, roi[optimal_idx]),
                         xytext=(10, 10), textcoords='offset points',
                         arrowprops=dict(arrowstyle='->', color='red'))

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}fig5_discount_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"图表已保存: fig5_discount_analysis.png")

    def plot_parallel_trend(self, parallel_results):
        """图6: 平行趋势检验"""
        fig, ax = plt.subplots(figsize=(10, 6))

        weeks = parallel_results['Week'].tolist()
        treat_means = parallel_results['Treatment_Mean'].tolist()
        control_means = parallel_results['Control_Mean'].tolist()

        x = np.arange(len(weeks))
        width = 0.35

        ax.bar(x - width / 2, treat_means, width, label='处理组', color='steelblue')
        ax.bar(x + width / 2, control_means, width, label='对照组', color='lightcoral')

        ax.set_xlabel('促销前周数')
        ax.set_ylabel('平均消费金额 (英镑)')
        ax.set_title('平行趋势检验: 促销前各周处理组与对照组对比')
        ax.set_xticks(x)
        ax.set_xticklabels(weeks)
        ax.legend()

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}fig6_parallel_trend.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"图表已保存: fig6_parallel_trend.png")

    def plot_correlation_heatmap(self, segment_analysis):
        """图7: 特征相关性热力图"""
        # 选择数值列
        numeric_cols = ['TotalSpent', 'PurchaseCount', 'AvgOrderValue',
                        'UniqueProducts', 'ActiveDays', 'PurchaseFreq', 'PSI']

        corr_data = segment_analysis[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 8))

        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        sns.heatmap(corr_data, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, square=True, linewidths=0.5, ax=ax)

        ax.set_title('用户特征与价格弹性的相关性矩阵')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}fig7_correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"图表已保存: fig7_correlation_heatmap.png")

    def run_all_plots(self, elasticity_data, segment_analysis, did_results, hetero_results, discount_results,
                      parallel_results):
        """生成所有图表"""
        print("\n" + "=" * 50)
        print("生成可视化图表")
        print("=" * 50)

        self.plot_price_elasticity_distribution(elasticity_data)
        self.plot_segment_characteristics(segment_analysis)
        self.plot_did_results(did_results)
        self.plot_heterogeneous_effects(hetero_results)
        self.plot_discount_analysis(discount_results)
        self.plot_parallel_trend(parallel_results)
        self.plot_correlation_heatmap(segment_analysis)

        print(f"\n所有图表已保存至: {self.output_dir}")


# 主函数测试
if __name__ == "__main__":
    # 加载数据
    elasticity_data = pd.read_csv("../data/price_elasticity_results.csv")
    segment_analysis = pd.read_csv("../data/segment_analysis.csv")

    # 注意: DID结果和异质性结果需要先运行03_ab_simulation.py
    # 这里仅作为示例，实际使用时需要从ABSimulation对象获取

    visualizer = Visualizer()

    # 如果所有数据都存在，运行所有图表
    # visualizer.run_all_plots(elasticity_data, segment_analysis,
    #                          did_results, hetero_results, discount_results, parallel_results)