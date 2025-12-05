import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from tqdm import tqdm
import warnings
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
import torch
import torch.nn.functional as F

warnings.filterwarnings('ignore')

# 设置中文字体和图形样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

class CompleteCNNAnalysisPipeline:
    """
    完整的CNN模型性能分析流水线 - 严格复现《(Re-)Imag(in)ing Price Trends》论文指标
    数据范围：2000-2019年
    """
    
    def __init__(self, data_dir, output_dir="./cnn_complete_analysis_2000_2019"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.yearly_data = {}
        self.combined_data = None
        self.results = {
            'yearly_metrics': {'classification': [], 'portfolio': []},
            'pooled_metrics': {},
            'cross_year_summary': {}
        }
        self.years_range = range(2000, 2020)  # 2000-2019年
    
    def load_all_years_data(self):
        """加载2000-2019年所有数据"""
        print("加载2000-2019年数据...")
        loaded_years = []
        
        for year in tqdm(self.years_range, desc="加载年份数据"):
            # 匹配文件模式
            file_patterns = [
                f"*{year}*with_cnn.feather",
                f"*{year}*labels_w_delay_with_cnn.feather", 
                f"*{year}*cnn.feather"
            ]
            
            file_path = None
            for pattern in file_patterns:
                matching_files = list(self.data_dir.glob(pattern))
                if matching_files:
                    file_path = matching_files[0]
                    break
            
            if file_path and file_path.exists():
                try:
                    df = pd.read_feather(file_path)
                    # 确保包含必要列
                    required_cols = ['Date', 'cnn_prob_20d', 'Ret_20d']
                    if all(col in df.columns for col in required_cols):
                        df['Date'] = pd.to_datetime(df['Date'])
                        df['Year'] = df['Date'].dt.year
                        self.yearly_data[year] = df
                        loaded_years.append(year)
                        print(f"✓ 成功加载 {year} 年数据: {len(df):,} 样本")
                    else:
                        print(f"⚠ {year} 年数据缺少必要列")
                except Exception as e:
                    print(f"✗ 加载 {year} 年数据失败: {e}")
            else:
                print(f"⚠ 未找到 {year} 年数据文件")
        
        # 合并所有数据用于整体计算
        if self.yearly_data:
            self.combined_data = pd.concat(self.yearly_data.values(), ignore_index=True)
            print(f"\n数据加载摘要:")
            print(f"成功加载年份: {len(loaded_years)}年 ({min(loaded_years)}-{max(loaded_years)})")
            print(f"总样本数: {len(self.combined_data):,}")
            print(f"时间范围: {self.combined_data['Date'].min()} 到 {self.combined_data['Date'].max()}")
        else:
            print("未找到任何有效数据")
    
    def calculate_comprehensive_correlations(self, df):
        """
        计算完整的相关性指标（复现原文表2）
        包含Spearman和Pearson相关系数
        """
        df_clean = df.dropna(subset=['cnn_prob_20d', 'Ret_20d'])
        
        if len(df_clean) < 2:
            return {}
        
        # 1. 整体相关性
        spearman_overall = df_clean['cnn_prob_20d'].corr(df_clean['Ret_20d'], method='spearman')
        pearson_overall = df_clean['cnn_prob_20d'].corr(df_clean['Ret_20d'], method='pearson')
        
        # 2. 横截面相关性（按时间点）
        def cross_sectional_corr(group, method='spearman'):
            if len(group) < 2:
                return np.nan
            return group['cnn_prob_20d'].corr(group['Ret_20d'], method=method)
        
        # 计算每个时间点的横截面相关性
        spearman_cross = df_clean.groupby('Date').apply(cross_sectional_corr, method='spearman')
        pearson_cross = df_clean.groupby('Date').apply(cross_sectional_corr, method='pearson')
        
        # 3. 信息系数(Information Coefficient)分析
        ic_results = self.calculate_information_coefficient(df_clean)
        
        return {
            'spearman': {
                'overall': spearman_overall,
                'cross_sectional_mean': spearman_cross.mean(),
                'cross_sectional_std': spearman_cross.std(),
                'cross_sectional_ts': spearman_cross,
                'significant': abs(spearman_overall) > 0.05
            },
            'pearson': {
                'overall': pearson_overall,
                'cross_sectional_mean': pearson_cross.mean(),
                'cross_sectional_std': pearson_cross.std(),
                'cross_sectional_ts': pearson_cross,
                'significant': abs(pearson_overall) > 0.05
            },
            'information_coefficient': ic_results,
            'sample_size': len(df_clean)
        }
    
    def calculate_information_coefficient(self, df):
        """计算信息系数(IC)及相关统计量"""
        if len(df) < 2:
            return {}
        
        # 按时间点计算IC（使用Spearman）
        def period_ic(group):
            if len(group) < 2:
                return np.nan
            return group['cnn_prob_20d'].corr(group['Ret_20d'], method='spearman')
        
        ic_series = df.groupby('Date').apply(period_ic).dropna()
        
        if len(ic_series) == 0:
            return {}
        
        # IC统计量
        mean_ic = ic_series.mean()
        std_ic = ic_series.std()
        ic_ir = mean_ic / std_ic if std_ic > 0 else 0
        positive_ratio = (ic_series > 0).mean()
        
        # IC的t检验
        if len(ic_series) > 1 and std_ic > 0:
            t_stat, p_value = stats.ttest_1samp(ic_series, 0)
        else:
            t_stat, p_value = 0, 1.0
        
        return {
            'mean_ic': mean_ic,
            'std_ic': std_ic,
            'ic_ir': ic_ir,  # 信息比率
            'positive_ratio': positive_ratio,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_5pct': p_value < 0.05,
            'significant_1pct': p_value < 0.01,
            'ic_time_series': ic_series,
            'periods': len(ic_series)
        }
    
    def calculate_classification_metrics(self, df):
        """计算完整的分类指标"""
        df_clean = df.dropna(subset=['cnn_prob_20d', 'Ret_20d'])
        
        if len(df_clean) == 0:
            return {}
        
        # 方向预测
        df_clean = df_clean.copy()
        df_clean['pred_direction'] = (df_clean['cnn_prob_20d'] > 0.5).astype(int)
        df_clean['actual_direction'] = (df_clean['Ret_20d'] > 0).astype(int)
        
        # 混淆矩阵
        tp = ((df_clean['pred_direction'] == 1) & (df_clean['actual_direction'] == 1)).sum()
        fp = ((df_clean['pred_direction'] == 1) & (df_clean['actual_direction'] == 0)).sum()
        fn = ((df_clean['pred_direction'] == 0) & (df_clean['actual_direction'] == 1)).sum()
        tn = ((df_clean['pred_direction'] == 0) & (df_clean['actual_direction'] == 0)).sum()
        
        total = tp + fp + fn + tn
        if total == 0:
            return {}
        
        # 基础分类指标
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # AUC-ROC
        try:
            auc_roc = roc_auc_score(df_clean['actual_direction'], df_clean['cnn_prob_20d'])
        except:
            auc_roc = 0.5
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'total_samples': total,
            'positive_ratio': df_clean['actual_direction'].mean()  # 正样本比例
        }
    
    def calculate_portfolio_performance(self, df, n_deciles=10):
        """计算投资组合表现（复现原文表3）"""
        df_clean = df.dropna(subset=['cnn_prob_20d', 'Ret_20d'])
        
        if len(df_clean) < n_deciles * 5:  # 确保足够样本
            return None
        
        # 按预测概率分十分位
        df_clean = df_clean.copy()
        df_clean['decile'] = df_clean.groupby('Date')['cnn_prob_20d'].transform(
            lambda x: pd.qcut(x, n_deciles, labels=False, duplicates='drop')
        )
        
        # 等权重组合收益
        decile_returns_ew = df_clean.groupby(['Date', 'decile'])['Ret_20d'].mean().unstack()
        
        # 市值权重组合收益
        def value_weighted_return(group):
            if 'MarketCap' in group.columns:
                total_mcap = group['MarketCap'].sum()
                if total_mcap > 0:
                    return np.average(group['Ret_20d'], weights=group['MarketCap'])
            return group['Ret_20d'].mean()
        
        decile_returns_vw = df_clean.groupby(['Date', 'decile']).apply(value_weighted_return).unstack()
        
        # H-L组合（多空策略）
        if n_deciles-1 in decile_returns_ew.columns and 0 in decile_returns_ew.columns:
            hl_returns_ew = decile_returns_ew[n_deciles-1] - decile_returns_ew[0]
            hl_returns_vw = decile_returns_vw[n_deciles-1] - decile_returns_vw[0]
        else:
            # 如果分位不完整，使用最高最低分位
            available_deciles = [col for col in decile_returns_ew.columns if not pd.isna(col)]
            if len(available_deciles) >= 2:
                high_decile = max(available_deciles)
                low_decile = min(available_deciles)
                hl_returns_ew = decile_returns_ew[high_decile] - decile_returns_ew[low_decile]
                hl_returns_vw = decile_returns_vw[high_decile] - decile_returns_vw[low_decile]
            else:
                return None
        
        # 计算夏普比率和统计显著性
        sharpe_ew, sig_ew = self.calculate_annualized_sharpe(hl_returns_ew)
        sharpe_vw, sig_vw = self.calculate_annualized_sharpe(hl_returns_vw)
        
        return {
            'sharpe_ratio_ew': sharpe_ew,
            'sharpe_ratio_vw': sharpe_vw,
            'hl_return_mean_ew': hl_returns_ew.mean(),
            'hl_return_std_ew': hl_returns_ew.std(),
            'hl_cumulative_ew': (1 + hl_returns_ew).prod() - 1,
            'significance_ew': sig_ew,
            'significance_vw': sig_vw,
            'decile_returns_ew': decile_returns_ew,
            'decile_returns_vw': decile_returns_vw,
            'sample_size': len(df_clean),
            'periods': len(hl_returns_ew)
        }
    
    def calculate_annualized_sharpe(self, returns, periods_per_year=12):
        """计算年化夏普比率及统计显著性"""
        if len(returns) < 2 or returns.std() == 0:
            return 0, {'significant': False, 't_stat': 0, 'p_value': 1.0}
        
        annual_return = returns.mean() * periods_per_year
        annual_volatility = returns.std() * np.sqrt(periods_per_year)
        sharpe_ratio = annual_return / annual_volatility
        
        # 统计显著性检验
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        significant_5pct = p_value < 0.05
        significant_1pct = p_value < 0.01
        
        significance = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_5pct': significant_5pct,
            'significant_1pct': significant_1pct
        }
        
        return sharpe_ratio, significance
    
    def run_complete_analysis(self):
        """执行完整分析流程"""
        print("="*70)
        print("开始2000-2019年CNN模型性能完整分析")
        print("="*70)
        
        try:
            # 1. 加载数据
            self.load_all_years_data()
            
            if not self.yearly_data:
                print("未找到有效数据，分析终止")
                return
            
            # 2. 分年计算各项指标
            print("\n步骤1: 分年计算分类和组合指标...")
            self.calculate_yearly_metrics()
            
            # 3. 整体计算相关性指标
            print("\n步骤2: 整体计算相关性指标...")
            self.calculate_pooled_correlations()
            
            # 4. 整体计算分类指标
            print("\n步骤3: 整体计算分类指标...")
            self.calculate_pooled_classification()
            
            # 5. 计算跨年汇总统计
            print("\n步骤4: 计算跨年汇总...")
            self.calculate_cross_year_summary()
            
            # 6. 保存所有结果
            print("\n步骤5: 保存结果...")
            self.save_comprehensive_results()
            
            # 7. 生成可视化
            print("\n步骤6: 生成可视化...")
            self.create_comprehensive_visualizations()
            
            # 8. 打印报告
            print("\n步骤7: 生成分析报告...")
            self.print_detailed_report()
            
            print("\n" + "="*70)
            print("分析完成！")
            print("="*70)
            
        except Exception as e:
            print(f"分析过程出错: {e}")
            import traceback
            traceback.print_exc()
    
    def calculate_yearly_metrics(self):
        """计算分年指标"""
        yearly_classification = []
        yearly_portfolio = []
        
        for year, df in tqdm(self.yearly_data.items(), desc="分年计算"):
            # 分类指标
            class_metrics = self.calculate_classification_metrics(df)
            if class_metrics:
                class_metrics['year'] = year
                yearly_classification.append(class_metrics)
            
            # 投资组合指标
            port_metrics = self.calculate_portfolio_performance(df)
            if port_metrics:
                port_metrics['year'] = year
                yearly_portfolio.append(port_metrics)
        
        self.results['yearly_metrics']['classification'] = yearly_classification
        self.results['yearly_metrics']['portfolio'] = yearly_portfolio
    
    def calculate_pooled_correlations(self):
        """计算整体相关性指标"""
        if self.combined_data is not None:
            corr_metrics = self.calculate_comprehensive_correlations(self.combined_data)
            self.results['pooled_metrics']['correlations'] = corr_metrics
    
    def calculate_pooled_classification(self):
        """计算整体分类指标"""
        if self.combined_data is not None:
            class_metrics = self.calculate_classification_metrics(self.combined_data)
            self.results['pooled_metrics']['classification'] = class_metrics
    
    def calculate_cross_year_summary(self):
        """计算跨年汇总统计"""
        class_df = pd.DataFrame(self.results['yearly_metrics']['classification'])
        port_df = pd.DataFrame([m for m in self.results['yearly_metrics']['portfolio'] if m is not None])
        
        if class_df.empty:
            self.results['cross_year_summary'] = {}
            return
        
        summary = {
            # 样本统计
            'total_years': len(class_df),
            'total_samples': class_df['total_samples'].sum() if 'total_samples' in class_df.columns else 0,
            'mean_samples_per_year': class_df['total_samples'].mean() if 'total_samples' in class_df.columns else 0,
            
            # 分类指标汇总
            'mean_accuracy': class_df['accuracy'].mean(),
            'std_accuracy': class_df['accuracy'].std(),
            'mean_precision': class_df['precision'].mean() if 'precision' in class_df.columns else 0,
            'mean_recall': class_df['recall'].mean() if 'recall' in class_df.columns else 0,
            'mean_f1_score': class_df['f1_score'].mean() if 'f1_score' in class_df.columns else 0,
            'mean_auc_roc': class_df['auc_roc'].mean() if 'auc_roc' in class_df.columns else 0,
            
            # 时间趋势
            'accuracy_trend': self.calculate_trend(class_df, 'accuracy'),
            'f1_trend': self.calculate_trend(class_df, 'f1_score') if 'f1_score' in class_df.columns else 0,
        }
        
        # 投资组合指标汇总
        if not port_df.empty:
            summary.update({
                'mean_sharpe_ew': port_df['sharpe_ratio_ew'].mean(),
                'mean_sharpe_vw': port_df['sharpe_ratio_vw'].mean(),
                'std_sharpe_ew': port_df['sharpe_ratio_ew'].std(),
                'sharpe_trend': self.calculate_trend(port_df, 'sharpe_ratio_ew'),
                'significant_years_5pct': port_df['significance_ew'].apply(
                    lambda x: x.get('significant_5pct', False) if isinstance(x, dict) else False
                ).sum() if 'significance_ew' in port_df.columns else 0
            })
        
        self.results['cross_year_summary'] = summary
    
    def calculate_trend(self, df, column):
        """计算时间趋势斜率"""
        if column not in df.columns or len(df) < 2:
            return 0
        x = np.arange(len(df))
        y = df[column].values
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return 0
        slope = np.polyfit(x[mask], y[mask], 1)[0]
        return slope
    
    def save_comprehensive_results(self):
        """保存完整结果到文件"""
        print("保存分析结果...")
        
        try:
            # 1. 保存分年分类指标
            if self.results['yearly_metrics']['classification']:
                class_df = pd.DataFrame(self.results['yearly_metrics']['classification'])
                class_df.to_csv(self.output_dir / "yearly_classification_metrics.csv", 
                              index=False, float_format='%.6f')
                print("✓ 分年分类指标已保存")
            
            # 2. 保存分年组合指标
            valid_portfolio = [m for m in self.results['yearly_metrics']['portfolio'] if m is not None]
            if valid_portfolio:
                port_df = pd.DataFrame(valid_portfolio)
                port_df.to_csv(self.output_dir / "yearly_portfolio_metrics.csv", 
                              index=False, float_format='%.6f')
                print("✓ 分年组合指标已保存")
            
            # 3. 保存整体相关性指标
            if self.results['pooled_metrics'].get('correlations'):
                corr_data = {}
                for corr_type, metrics in self.results['pooled_metrics']['correlations'].items():
                    if isinstance(metrics, dict):
                        for key, value in metrics.items():
                            if not isinstance(value, (pd.Series, pd.DataFrame)) and not isinstance(value, dict):
                                corr_data[f'{corr_type}_{key}'] = value
                
                if corr_data:
                    pd.DataFrame([corr_data]).to_csv(self.output_dir / "correlation_metrics.csv", 
                                                    index=False, float_format='%.6f')
                    print("✓ 相关性指标已保存")
            
            # 4. 保存整体分类指标
            if self.results['pooled_metrics'].get('classification'):
                class_metrics = self.results['pooled_metrics']['classification']
                pd.DataFrame([class_metrics]).to_csv(self.output_dir / "pooled_classification_metrics.csv", 
                                                    index=False, float_format='%.6f')
                print("✓ 整体分类指标已保存")
            
            # 5. 保存跨年汇总
            if self.results['cross_year_summary']:
                summary_df = pd.DataFrame([self.results['cross_year_summary']])
                summary_df.to_csv(self.output_dir / "cross_year_summary.csv", 
                                index=False, float_format='%.6f')
                print("✓ 跨年汇总已保存")
            
            # 6. 保存IC时间序列
            ic_ts = self.results['pooled_metrics'].get('correlations', {}).get('information_coefficient', {}).get('ic_time_series')
            if ic_ts is not None:
                ic_ts.to_csv(self.output_dir / "information_coefficient_timeseries.csv", 
                           index=True, float_format='%.6f')
                print("✓ IC时间序列已保存")
            
            print(f"✓ 所有结果已保存到: {self.output_dir}")
            
        except Exception as e:
            print(f"保存结果时出错: {e}")
    
    def create_comprehensive_visualizations(self):
        """创建综合可视化图表"""
        print("生成可视化图表...")
        
        try:
            # 1. 年度准确率趋势图
            if self.results['yearly_metrics']['classification']:
                class_df = pd.DataFrame(self.results['yearly_metrics']['classification'])
                plt.figure(figsize=(12, 6))
                plt.plot(class_df['year'], class_df['accuracy'], 'o-', linewidth=2, markersize=6)
                plt.xlabel('年份')
                plt.ylabel('分类准确率')
                plt.title('CNN模型年度分类准确率趋势 (2000-2019)')
                plt.grid(True, alpha=0.3)
                plt.xticks(class_df['year'][::2])
                plt.savefig(self.output_dir / "yearly_accuracy_trend.png", dpi=300, bbox_inches='tight')
                plt.close()
                print("✓ 准确率趋势图已生成")
            
            # 2. 年度夏普比率趋势
            valid_portfolio = [m for m in self.results['yearly_metrics']['portfolio'] if m is not None]
            if valid_portfolio:
                port_df = pd.DataFrame(valid_portfolio)
                if not port_df.empty and 'sharpe_ratio_ew' in port_df.columns:
                    plt.figure(figsize=(12, 6))
                    plt.plot(port_df['year'], port_df['sharpe_ratio_ew'], 's-', linewidth=2, markersize=6, label='等权重')
                    if 'sharpe_ratio_vw' in port_df.columns:
                        plt.plot(port_df['year'], port_df['sharpe_ratio_vw'], '^-', linewidth=2, markersize=6, label='市值权重')
                    plt.xlabel('年份')
                    plt.ylabel('年化夏普比率')
                    plt.title('H-L组合年度夏普比率趋势 (2000-2019)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.savefig(self.output_dir / "yearly_sharpe_trend.png", dpi=300, bbox_inches='tight')
                    plt.close()
                    print("✓ 夏普比率趋势图已生成")
            
            # 3# 3. 信息系数(IC)时间序列图
            if 'information_coefficient' in self.results['pooled_metrics'].get('correlations', {}):
                ic_data = self.results['pooled_metrics']['correlations']['information_coefficient']
                if 'ic_time_series' in ic_data and ic_data['ic_time_series'] is not None:
                    ic_series = ic_data['ic_time_series']
                    plt.figure(figsize=(12, 6))
                    plt.plot(ic_series.index, ic_series.values, linewidth=1, alpha=0.7)
                    plt.axhline(y=ic_series.mean(), color='r', linestyle='--', 
                               label=f'均值: {ic_series.mean():.3f}')
                    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                    plt.xlabel('日期')
                    plt.ylabel('信息系数 (IC)')
                    plt.title('信息系数时间序列 (2000-2019)', fontsize=14)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(self.output_dir / "information_coefficient_timeseries.png", 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    print("✓ IC时间序列图已生成")

            # 4. 预测概率分布直方图
            if self.combined_data is not None and 'cnn_prob_20d' in self.combined_data.columns:
                probs = self.combined_data['cnn_prob_20d'].dropna()
                if len(probs) > 0:
                    plt.figure(figsize=(10, 6))
                    plt.hist(probs, bins=50, alpha=0.7, edgecolor='black', density=True)
                    plt.axvline(x=0.5, color='r', linestyle='--', label='决策边界 (0.5)')
                    plt.xlabel('CNN预测概率')
                    plt.ylabel('密度')
                    plt.title('CNN预测概率分布 (2000-2019全样本)', fontsize=14)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(self.output_dir / "prediction_probability_distribution.png", 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    print("✓ 预测概率分布图已生成")

            # 5. 分位数组合收益热力图
            if self.results['yearly_metrics']['portfolio']:
                # 提取所有年份的分位数收益数据
                all_decile_returns = []
                years = []
                for result in self.results['yearly_metrics']['portfolio']:
                    if result and 'decile_returns_ew' in result:
                        decile_returns = result['decile_returns_ew'].mean()  # 时间序列平均
                        all_decile_returns.append(decile_returns)
                        years.append(result['year'])
                
                if all_decile_returns and len(all_decile_returns) > 0:
                    returns_df = pd.DataFrame(all_decile_returns, index=years)
                    returns_df = returns_df.sort_index()
                    
                    plt.figure(figsize=(14, 10))
                    sns.heatmap(returns_df.T, annot=True, fmt=".3f", cmap="RdYlGn", 
                               cbar_kws={'label': '平均收益'}, center=0)
                    plt.xlabel('年份')
                    plt.ylabel('分位数')
                    plt.title('分位数组合年平均收益热力图 (等权重)', fontsize=14)
                    plt.tight_layout()
                    plt.savefig(self.output_dir / "decile_returns_heatmap.png", 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    print("✓ 分位数收益热力图已生成")

            # 6. H-L组合累积收益曲线图
            if self.results['yearly_metrics']['portfolio']:
                # 计算代表性年份的累积收益
                representative_years = [2000, 2005, 2010, 2015]  # 选择代表性年份
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.ravel()
                
                plotted_count = 0
                for i, year in enumerate(representative_years):
                    # 查找该年的组合数据
                    year_result = next((r for r in self.results['yearly_metrics']['portfolio'] 
                                      if r and r.get('year') == year), None)
                    
                    if year_result and 'decile_returns_ew' in year_result:
                        decile_returns = year_result['decile_returns_ew']
                        if len(decile_returns.columns) >= 2:
                            # 计算H-L组合收益
                            high_decile = decile_returns.columns[-1]
                            low_decile = decile_returns.columns[0]
                            hl_returns = decile_returns[high_decile] - decile_returns[low_decile]
                            cumulative_returns = (1 + hl_returns).cumprod()
                            
                            if i < len(axes):
                                axes[i].plot(cumulative_returns.index, cumulative_returns.values, 
                                           linewidth=2, color=f'C{i}')
                                axes[i].set_title(f'{year}年 H-L组合累积收益', fontsize=12)
                                axes[i].set_xlabel('日期')
                                axes[i].set_ylabel('累积收益')
                                axes[i].grid(True, alpha=0.3)
                                axes[i].tick_params(axis='x', rotation=45)
                                plotted_count += 1
                
                if plotted_count > 0:
                    # 移除空的子图
                    for j in range(plotted_count, 4):
                        if j < len(axes):
                            fig.delaxes(axes[j])
                    
                    plt.tight_layout()
                    plt.savefig(self.output_dir / "hl_strategy_cumulative_returns.png", 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    print("✓ H-L组合累积收益图已生成")

            # 7. 相关性热力图（不同指标间）
            correlation_data = {}
            if self.results['yearly_metrics']['classification']:
                class_df = pd.DataFrame(self.results['yearly_metrics']['classification'])
                # 提取数值型指标
                numeric_cols = class_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    correlation_matrix = class_df[numeric_cols].corr()
                    
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                               center=0, square=True)
                    plt.title('年度指标相关性热力图', fontsize=14)
                    plt.tight_layout()
                    plt.savefig(self.output_dir / "yearly_metrics_correlation_heatmap.png", 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    print("✓ 指标相关性热力图已生成")

            # 8. 年度样本量分布图
            if self.results['yearly_metrics']['classification']:
                class_df = pd.DataFrame(self.results['yearly_metrics']['classification'])
                if 'total_samples' in class_df.columns:
                    plt.figure(figsize=(12, 6))
                    plt.bar(class_df['year'], class_df['total_samples'], 
                           alpha=0.7, color='skyblue', edgecolor='black')
                    plt.xlabel('年份')
                    plt.ylabel('样本数量')
                    plt.title('年度样本量分布 (2000-2019)', fontsize=14)
                    plt.xticks(class_df['year'][::2])
                    plt.grid(True, alpha=0.3, axis='y')
                    plt.tight_layout()
                    plt.savefig(self.output_dir / "yearly_sample_distribution.png", 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    print("✓ 样本量分布图已生成")

            print("🎨 所有可视化图表生成完成！")

        except Exception as e:
            print(f"生成可视化图表时出错: {e}")
            import traceback
            traceback.print_exc()

    def print_detailed_report(self):
        """打印详细分析报告"""
        print("\n" + "="*80)
        print("CNN模型性能详细分析报告 (2000-2019)")
        print("="*80)
        
        # 基本统计信息
        summary = self.results['cross_year_summary']
        print(f"\n1. 样本统计概要:")
        print(f"   覆盖年份: {summary.get('total_years', 0)}年")
        print(f"   总样本数: {summary.get('total_samples', 0):,}")
        print(f"   年平均样本数: {summary.get('mean_samples_per_year', 0):,.0f}")
        
        # 分类性能结果
        class_df = pd.DataFrame(self.results['yearly_metrics']['classification'])
        if not class_df.empty:
            print(f"\n2. 分类性能指标 (2000-2019年平均±标准差):")
            print(f"   准确率: {class_df['accuracy'].mean():.3f} ± {class_df['accuracy'].std():.3f}")
            if 'precision' in class_df.columns:
                print(f"   精确率: {class_df['precision'].mean():.3f} ± {class_df['precision'].std():.3f}")
            if 'recall' in class_df.columns:
                print(f"   召回率: {class_df['recall'].mean():.3f} ± {class_df['recall'].std():.3f}")
            if 'f1_score' in class_df.columns:
                print(f"   F1分数: {class_df['f1_score'].mean():.3f} ± {class_df['f1_score'].std():.3f}")
            if 'auc_roc' in class_df.columns:
                print(f"   AUC-ROC: {class_df['auc_roc'].mean():.3f} ± {class_df['auc_roc'].std():.3f}")
        
        # 投资组合性能结果
        port_df = pd.DataFrame([m for m in self.results['yearly_metrics']['portfolio'] if m is not None])
        if not port_df.empty:
            print(f"\n3. 投资组合性能指标 (H-L策略):")
            print(f"   等权重夏普比率: {port_df['sharpe_ratio_ew'].mean():.3f} ± {port_df['sharpe_ratio_ew'].std():.3f}")
            if 'sharpe_ratio_vw' in port_df.columns:
                print(f"   市值权重夏普比率: {port_df['sharpe_ratio_vw'].mean():.3f} ± {port_df['sharpe_ratio_vw'].std():.3f}")
            print(f"   等权重H-L年均收益: {port_df.get('hl_return_mean_ew', pd.Series([0])).mean()*12:.3f}")
        
        # 相关性分析结果
        corr_metrics = self.results['pooled_metrics'].get('correlations', {})
        if corr_metrics:
            print(f"\n4. 相关性分析结果:")
            if 'spearman' in corr_metrics:
                spearman = corr_metrics['spearman']
                print(f"   斯皮尔曼相关系数: {spearman.get('overall', 0):.3f}")
                if spearman.get('significant', False):
                    print("     (统计显著)")
            
            if 'pearson' in corr_metrics:
                pearson = corr_metrics['pearson']
                print(f"   皮尔逊相关系数: {pearson.get('overall', 0):.3f}")
                if pearson.get('significant', False):
                    print("     (统计显著)")
            
            if 'information_coefficient' in corr_metrics:
                ic_info = corr_metrics['information_coefficient']
                print(f"   信息系数(IC)均值: {ic_info.get('mean_ic', 0):.3f}")
                print(f"   IC信息比率: {ic_info.get('ic_ir', 0):.3f}")
                if ic_info.get('significant_5pct', False):
                    print("     (5%水平显著)")
                elif ic_info.get('significant_1pct', False):
                    print("     (1%水平显著)")
        
        # 时间趋势分析
        print(f"\n5. 时间趋势分析:")
        print(f"   准确率年际趋势斜率: {summary.get('accuracy_trend', 0):.4f}")
        if 'sharpe_trend' in summary:
            print(f"   夏普比率年际趋势斜率: {summary['sharpe_trend']:.4f}")
        
        # 统计显著性
        if not port_df.empty and 'significance_ew' in port_df.columns:
            significant_years = port_df['significance_ew'].apply(
                lambda x: x.get('significant_5pct', False) if isinstance(x, dict) else False
            ).sum()
            total_years = len(port_df)
            print(f"\n6. 统计显著性分析:")
            print(f"   5%显著性水平显著年份: {significant_years}/{total_years} ({significant_years/total_years:.1%})")
        
        # 模型性能评估
        print(f"\n7. 模型性能总体评估:")
        accuracy_mean = summary.get('mean_accuracy', 0)
        if accuracy_mean > 0.55:
            print("   📈 分类性能: 优秀 (准确率 > 55%)")
        elif accuracy_mean > 0.52:
            print("   📊 分类性能: 良好 (准确率 > 52%)")
        else:
            print("   📉 分类性能: 需改进 (准确率 ≤ 52%)")
        
        sharpe_mean = summary.get('mean_sharpe_ew', 0)
        if sharpe_mean > 1.0:
            print("   💹 投资价值: 优秀 (夏普比率 > 1.0)")
        elif sharpe_mean > 0.5:
            print("   📈 投资价值: 良好 (夏普比率 > 0.5)")
        else:
            print("   📉 投资价值: 有限 (夏普比率 ≤ 0.5)")
        
        print("\n" + "="*80)

    def calculate_comprehensive_correlations(self, df):
        """
        计算完整的相关性指标（复现原文表2）
        包含Spearman和Pearson相关系数
        """
        df_clean = df.dropna(subset=['cnn_prob_20d', 'Ret_20d'])
        
        if len(df_clean) < 2:
            return {}
        
        # 1. 整体相关性
        spearman_overall = df_clean['cnn_prob_20d'].corr(df_clean['Ret_20d'], method='spearman')
        pearson_overall = df_clean['cnn_prob_20d'].corr(df_clean['Ret_20d'], method='pearson')
        
        # 2. 横截面相关性（按时间点）
        def cross_sectional_corr(group, method='spearman'):
            if len(group) < 2:
                return np.nan
            return group['cnn_prob_20d'].corr(group['Ret_20d'], method=method)
        
        # 计算每个时间点的横截面相关性
        spearman_cross = df_clean.groupby('Date').apply(cross_sectional_corr, method='spearman')
        pearson_cross = df_clean.groupby('Date').apply(cross_sectional_corr, method='pearson')
        
        # 3. 信息系数(Information Coefficient)分析
        ic_results = self.calculate_information_coefficient(df_clean)
        
        return {
            'spearman': {
                'overall': spearman_overall,
                'cross_sectional_mean': spearman_cross.mean(),
                'cross_sectional_std': spearman_cross.std(),
                'cross_sectional_ts': spearman_cross,
                'significant': abs(spearman_overall) > 0.05
            },
            'pearson': {
                'overall': pearson_overall,
                'cross_sectional_mean': pearson_cross.mean(),
                'cross_sectional_std': pearson_cross.std(),
                'cross_sectional_ts': pearson_cross,
                'significant': abs(pearson_overall) > 0.05
            },
            'information_coefficient': ic_results,
            'sample_size': len(df_clean)
        }

    def run_complete_analysis_pipeline(self):
        """执行完整的分析流水线"""
        try:
            # 1. 数据加载阶段
            print("="*80)
            print("阶段1: 数据加载与预处理")
            print("="*80)
            self.load_all_years_data()
            
            if not self.yearly_data:
                print("❌ 未找到有效数据，分析终止")
                return None
            
            # 2. 分年指标计算
            print("\n" + "="*80)
            print("阶段2: 分年指标计算")
            print("="*80)
            self.calculate_yearly_metrics()
            
            # 3. 整体指标计算
            print("\n" + "="*80)
            print("阶段3: 整体指标计算")
            print("="*80)
            self.calculate_pooled_metrics()
            
            # 4. 汇总统计计算
            print("\n" + "="*80)
            print("阶段4: 汇总统计分析")
            print("="*80)
            self.calculate_cross_year_summary()
            
            # 5. 结果保存
            print("\n" + "="*80)
            print("阶段5: 结果保存")
            print("="*80)
            self.save_comprehensive_results()
            
            # 6. 可视化生成
            print("\n" + "="*80)
            print("阶段6: 可视化生成")
            print("="*80)
            self.create_comprehensive_visualizations()
            
            # 7. 报告生成
            print("\n" + "="*80)
            print("阶段7: 分析报告生成")
            print("="*80)
            self.print_detailed_report()
            
            print("\n🎉 分析流程完成！")
            return self.results
            
        except Exception as e:
            print(f"❌ 分析过程出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def calculate_pooled_metrics(self):
        """计算整体指标"""
        if self.combined_data is not None:
            # 相关性指标
            corr_metrics = self.calculate_comprehensive_correlations(self.combined_data)
            self.results['pooled_metrics']['correlations'] = corr_metrics
            
            # 分类指标
            class_metrics = self.calculate_classification_metrics(self.combined_data)
            self.results['pooled_metrics']['classification'] = class_metrics

# 使用示例
def main():
    """主执行函数"""
    # 配置路径
    data_directory = "/workspace_ssd/wangjiang/monthly_60d_cnn_prob_baseline"  # 根据实际路径修改
    output_directory = "./cnn_complete_analysis_results_60d"
    
    # 创建分析实例
    analyzer = CompleteCNNAnalysisPipeline(data_directory, output_directory)
    
    # 执行完整分析
    results = analyzer.run_complete_analysis_pipeline()
    
    if results is not None:
        print("\n" + "="*80)
        print("分析成功完成！")
        print("="*80)
        
        # 生成简要总结
        summary = results['cross_year_summary']
        if summary:
            print(f"关键发现摘要:")
            print(f"• 平均分类准确率: {summary.get('mean_accuracy', 0):.3f}")
            print(f"• 平均夏普比率: {summary.get('mean_sharpe_ew', 0):.3f}")
            print(f"• 覆盖样本数: {summary.get('total_samples', 0):,}")
            print(f"• 时间范围: 2000-2019年 ({summary.get('total_years', 0)}年)")
        
        return results
    else:
        print("分析失败，请检查数据和配置")
        return None

if __name__ == "__main__":
    # 执行主分析
    main_results = main()