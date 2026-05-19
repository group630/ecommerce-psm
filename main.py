"""
电商价格敏感度与促销效果评估
主程序入口 - 运行完整分析流程
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加脚本路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/scripts')

from scripts.data_preprocessing import DataPreprocessor
from scripts.price_elasticity import PriceElasticityCalculator
from scripts.ab_simulation import ABSimulation
from scripts.visualization import Visualizer


def main():
    """主函数"""
    print("=" * 60)
    print("电商价格敏感度与促销效果评估系统")
    print("=" * 60)

    # 配置文件路径
    DATA_PATH = "data/Online Retail.xlsx"
    OUTPUT_DIR = "results"

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/tables", exist_ok=True)

    # ==================== 步骤1: 数据预处理 ====================
    print("\n" + "=" * 50)
    print("步骤1: 数据预处理")
    print("=" * 50)

    preprocessor = DataPreprocessor(DATA_PATH)
    clean_data, user_features, product_features = preprocessor.run_pipeline()

    # 保存预处理结果
    clean_data.to_csv(f"./results/tables/clean_retail_data.csv", index=False)
    user_features.to_csv(f"./results/tables/user_features.csv", index=False)
    product_features.to_csv(f"./results/tables/product_features.csv", index=False)

    # ==================== 步骤2: 价格弹性计算 ====================
    print("\n" + "=" * 50)
    print("步骤2: 价格弹性指数计算")
    print("=" * 50)

    calculator = PriceElasticityCalculator(clean_data, user_features)
    elasticity_results, segment_analysis = calculator.run_pipeline()

    # 保存弹性结果
    elasticity_results.to_csv(f"./results/tables/price_elasticity_results.csv", index=False)
    segment_analysis.to_csv(f"./results/tables/segment_analysis.csv", index=False)

    # ==================== 步骤3: A/B模拟分析 ====================
    print("\n" + "=" * 50)
    print("步骤3: A/B离线模拟分析")
    print("=" * 50)

    simulator = ABSimulation(clean_data, user_features, elasticity_results)
    did_results = simulator.run_pipeline()

    # 获取异质性分析结果和折扣分析结果
    hetero_results = simulator.heterogeneous_effect_analysis()
    discount_results = simulator.discount_elasticity_analysis()

    # 获取平行趋势检验结果（需要从simulator中获取）
    parallel_results = simulator.parallel_trend_test()

    # 保存分析结果
    hetero_results.to_csv(f"./results/tables/heterogeneous_effects.csv", index=False)
    discount_results.to_csv(f"./results/tables/discount_analysis.csv", index=False)
    parallel_results.to_csv(f"./results/tables/parallel_trend.csv", index=False)

    # ==================== 步骤4: 可视化 ====================
    print("\n" + "=" * 50)
    print("步骤4: 生成可视化图表")
    print("=" * 50)

    visualizer = Visualizer(output_dir=f"{OUTPUT_DIR}/figures/")
    visualizer.run_all_plots(
        elasticity_results,
        segment_analysis,
        did_results,
        hetero_results,
        discount_results,
        parallel_results
    )

    # ==================== 完成 ====================
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)
    print(f"结果已保存至: {OUTPUT_DIR}/")
    print(f"  - 数据文件: {OUTPUT_DIR}/*.csv")
    print(f"  - 图表文件: {OUTPUT_DIR}/figures/*.png")

    # 输出关键发现
    print("\n" + "=" * 60)
    print("关键发现总结")
    print("=" * 60)

    print(f"\n1. 价格弹性分布:")
    print(f"   - 均值: {elasticity_results['PSI'].mean():.2f}")
    print(f"   - 标准差: {elasticity_results['PSI'].std():.2f}")
    print(f"   - 高敏感用户占比: {(elasticity_results['SensitivityLabel'] == '高敏感度').mean() * 100:.1f}%")

    print(f"\n2. DID分析结果:")
    did_coef = did_results['model1'].params['Treat_Period']
    did_pval = did_results['model1'].pvalues['Treat_Period']
    print(f"   - DID系数: {did_coef:.4f} (p={did_pval:.4f})")
    print(f"   - 促销提升效应: {(np.exp(did_coef) - 1) * 100:.1f}%")

    print(f"\n3. 异质性分析:")
    if len(hetero_results) > 0:
        high_effect = hetero_results[hetero_results['Group'] == '高敏感']['Relative_Effect'].values[0]
        low_effect = hetero_results[hetero_results['Group'] == '低敏感']['Relative_Effect'].values[0]
        print(f"   - 高敏感组效应: {high_effect:.1f}%")
        print(f"   - 低敏感组效应: {low_effect:.1f}%")
        print(f"   - 响应倍数: {high_effect / low_effect:.1f}倍")

    print(f"\n4. 最优折扣区间:")
    best_idx = discount_results['Estimated_ROI'].idxmax()
    best_discount = discount_results.loc[best_idx, 'Discount']
    best_roi = discount_results.loc[best_idx, 'Estimated_ROI']
    print(f"   - 最优折扣: {best_discount}")
    print(f"   - 对应ROI: {best_roi:.2f}")


if __name__ == "__main__":
    main()