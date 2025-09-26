#!/usr/bin/env python3
"""
Tower Defense ELM Learning Efficiency Experiment - 詳細統計分析
学習効率実験の結果を詳細に分析し、可視化する
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from datetime import datetime

# 日本語フォント設定
plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['font.size'] = 10

def load_experiment_data():
    """実験データを読み込む"""
    try:
        with open('learning_results.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print("実験データファイルが見つかりません")
        return None

def analyze_elm_only_performance(elm_data):
    """ELMのみ条件の性能分析"""
    scores = [episode['score'] for episode in elm_data]
    rewards = [episode['reward'] for episode in elm_data]
    steps = [episode['steps'] for episode in elm_data]
    towers = [episode['towers'] for episode in elm_data]
    
    analysis = {
        'trial_count': len(elm_data),
        'score_stats': {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'median': np.median(scores)
        },
        'reward_stats': {
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'min': np.min(rewards),
            'max': np.max(rewards)
        },
        'steps_stats': {
            'mean': np.mean(steps),
            'std': np.std(steps),
            'min': np.min(steps),
            'max': np.max(steps)
        },
        'towers_stats': {
            'mean': np.mean(towers),
            'std': np.std(towers),
            'min': np.min(towers),
            'max': np.max(towers)
        },
        'learning_progress': {
            'first_10_avg': np.mean(scores[:10]),
            'last_10_avg': np.mean(scores[-10:]),
            'improvement': np.mean(scores[-10:]) - np.mean(scores[:10])
        }
    }
    
    return analysis

def analyze_elm_llm_performance(elm_llm_data):
    """ELM+LLM条件の性能分析"""
    scores = [episode['score'] for episode in elm_llm_data]
    rewards = [episode['reward'] for episode in elm_llm_data]
    steps = [episode['steps'] for episode in elm_llm_data]
    towers = [episode['towers'] for episode in elm_llm_data]
    
    analysis = {
        'trial_count': len(elm_llm_data),
        'score_stats': {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'median': np.median(scores)
        },
        'reward_stats': {
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'min': np.min(rewards),
            'max': np.max(rewards)
        },
        'steps_stats': {
            'mean': np.mean(steps),
            'std': np.std(steps),
            'min': np.min(steps),
            'max': np.max(steps)
        },
        'towers_stats': {
            'mean': np.mean(towers),
            'std': np.std(towers),
            'min': np.min(towers),
            'max': np.max(towers)
        },
        'learning_progress': {
            'first_10_avg': np.mean(scores[:10]),
            'last_10_avg': np.mean(scores[-10:]),
            'improvement': np.mean(scores[-10:]) - np.mean(scores[:10])
        }
    }
    
    return analysis

def statistical_comparison(elm_data, elm_llm_data):
    """統計的比較分析"""
    elm_scores = [episode['score'] for episode in elm_data]
    elm_llm_scores = [episode['score'] for episode in elm_llm_data]
    
    # t検定
    t_stat, t_pvalue = stats.ttest_ind(elm_scores, elm_llm_scores)
    
    # Mann-Whitney U検定
    u_stat, u_pvalue = stats.mannwhitneyu(elm_scores, elm_llm_scores, alternative='two-sided')
    
    # 効果量（Cohen's d）
    pooled_std = np.sqrt(((len(elm_scores) - 1) * np.var(elm_scores, ddof=1) + 
                         (len(elm_llm_scores) - 1) * np.var(elm_llm_scores, ddof=1)) / 
                        (len(elm_scores) + len(elm_llm_scores) - 2))
    cohens_d = (np.mean(elm_llm_scores) - np.mean(elm_scores)) / pooled_std
    
    # 信頼区間
    elm_ci = stats.t.interval(0.95, len(elm_scores)-1, 
                             loc=np.mean(elm_scores), 
                             scale=stats.sem(elm_scores))
    elm_llm_ci = stats.t.interval(0.95, len(elm_llm_scores)-1, 
                                 loc=np.mean(elm_llm_scores), 
                                 scale=stats.sem(elm_llm_scores))
    
    comparison = {
        't_test': {
            'statistic': t_stat,
            'p_value': t_pvalue,
            'significant': t_pvalue < 0.05
        },
        'mann_whitney': {
            'statistic': u_stat,
            'p_value': u_pvalue,
            'significant': u_pvalue < 0.05
        },
        'effect_size': {
            'cohens_d': cohens_d,
            'interpretation': interpret_cohens_d(cohens_d)
        },
        'confidence_intervals': {
            'elm_only': elm_ci,
            'elm_llm': elm_llm_ci
        },
        'descriptive': {
            'elm_mean': np.mean(elm_scores),
            'elm_llm_mean': np.mean(elm_llm_scores),
            'difference': np.mean(elm_llm_scores) - np.mean(elm_scores),
            'improvement_rate': ((np.mean(elm_llm_scores) - np.mean(elm_scores)) / 
                               max(np.mean(elm_scores), 1)) * 100
        }
    }
    
    return comparison

def interpret_cohens_d(d):
    """Cohen's dの解釈"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "効果なし"
    elif abs_d < 0.5:
        return "小さな効果"
    elif abs_d < 0.8:
        return "中程度の効果"
    else:
        return "大きな効果"

def create_visualizations(data):
    """可視化の作成"""
    elm_scores = [episode['score'] for episode in data['elm_only']]
    elm_llm_scores = [episode['score'] for episode in data['elm_with_llm']]
    
    # 図1: スコア分布の比較
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Tower Defense ELM Learning Efficiency Experiment - 結果分析', fontsize=16, fontweight='bold')
    
    # スコア分布
    axes[0, 0].hist(elm_scores, bins=20, alpha=0.7, label='ELMのみ', color='skyblue')
    axes[0, 0].hist(elm_llm_scores, bins=20, alpha=0.7, label='ELM+LLM', color='lightcoral')
    axes[0, 0].set_xlabel('スコア')
    axes[0, 0].set_ylabel('頻度')
    axes[0, 0].set_title('スコア分布の比較')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # ボックスプロット
    box_data = [elm_scores, elm_llm_scores]
    box_labels = ['ELMのみ', 'ELM+LLM']
    axes[0, 1].boxplot(box_data, labels=box_labels)
    axes[0, 1].set_ylabel('スコア')
    axes[0, 1].set_title('スコア分布のボックスプロット')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 学習曲線
    elm_episodes = range(1, len(elm_scores) + 1)
    elm_llm_episodes = range(1, len(elm_llm_scores) + 1)
    
    axes[1, 0].plot(elm_episodes, elm_scores, 'o-', alpha=0.7, label='ELMのみ', color='skyblue')
    axes[1, 0].plot(elm_llm_episodes, elm_llm_scores, 'o-', alpha=0.7, label='ELM+LLM', color='lightcoral')
    axes[1, 0].set_xlabel('エピソード')
    axes[1, 0].set_ylabel('スコア')
    axes[1, 0].set_title('学習曲線')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 移動平均
    window = 5
    if len(elm_scores) >= window:
        elm_ma = pd.Series(elm_scores).rolling(window=window).mean()
        axes[1, 1].plot(elm_episodes, elm_ma, '-', linewidth=2, label=f'ELMのみ (MA{window})', color='blue')
    
    if len(elm_llm_scores) >= window:
        elm_llm_ma = pd.Series(elm_llm_scores).rolling(window=window).mean()
        axes[1, 1].plot(elm_llm_episodes, elm_llm_ma, '-', linewidth=2, label=f'ELM+LLM (MA{window})', color='red')
    
    axes[1, 1].set_xlabel('エピソード')
    axes[1, 1].set_ylabel('スコア (移動平均)')
    axes[1, 1].set_title(f'学習曲線 (移動平均, ウィンドウ={window})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_efficiency_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 図2: 詳細分析
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Tower Defense ELM Learning Efficiency Experiment - 詳細分析', fontsize=16, fontweight='bold')
    
    # タワー配置数の比較
    elm_towers = [episode['towers'] for episode in data['elm_only']]
    elm_llm_towers = [episode['towers'] for episode in data['elm_with_llm']]
    
    axes[0, 0].hist(elm_towers, bins=20, alpha=0.7, label='ELMのみ', color='skyblue')
    axes[0, 0].hist(elm_llm_towers, bins=20, alpha=0.7, label='ELM+LLM', color='lightcoral')
    axes[0, 0].set_xlabel('タワー配置数')
    axes[0, 0].set_ylabel('頻度')
    axes[0, 0].set_title('タワー配置数の分布')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 報酬の比較
    elm_rewards = [episode['reward'] for episode in data['elm_only']]
    elm_llm_rewards = [episode['reward'] for episode in data['elm_with_llm']]
    
    axes[0, 1].scatter(elm_episodes, elm_rewards, alpha=0.7, label='ELMのみ', color='skyblue')
    axes[0, 1].scatter(elm_llm_episodes, elm_llm_rewards, alpha=0.7, label='ELM+LLM', color='lightcoral')
    axes[0, 1].set_xlabel('エピソード')
    axes[0, 1].set_ylabel('報酬')
    axes[0, 1].set_title('報酬の推移')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # ステップ数の比較
    elm_steps = [episode['steps'] for episode in data['elm_only']]
    elm_llm_steps = [episode['steps'] for episode in data['elm_with_llm']]
    
    axes[1, 0].hist(elm_steps, bins=20, alpha=0.7, label='ELMのみ', color='skyblue')
    axes[1, 0].hist(elm_llm_steps, bins=20, alpha=0.7, label='ELM+LLM', color='lightcoral')
    axes[1, 0].set_xlabel('ステップ数')
    axes[1, 0].set_ylabel('頻度')
    axes[1, 0].set_title('ステップ数の分布')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 効率性分析（スコア/ステップ）
    elm_efficiency = [s/st if st > 0 else 0 for s, st in zip(elm_scores, elm_steps)]
    elm_llm_efficiency = [s/st if st > 0 else 0 for s, st in zip(elm_llm_scores, elm_llm_steps)]
    
    axes[1, 1].scatter(elm_episodes, elm_efficiency, alpha=0.7, label='ELMのみ', color='skyblue')
    axes[1, 1].scatter(elm_llm_episodes, elm_llm_efficiency, alpha=0.7, label='ELM+LLM', color='lightcoral')
    axes[1, 1].set_xlabel('エピソード')
    axes[1, 1].set_ylabel('効率性 (スコア/ステップ)')
    axes[1, 1].set_title('学習効率性の推移')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_efficiency_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_report(elm_analysis, elm_llm_analysis, comparison):
    """分析レポートの生成"""
    report = f"""
# Tower Defense ELM Learning Efficiency Experiment - 統計分析レポート

## 実行日時
{datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}

## 実験データ概要

### ELMのみ条件
- 試行回数: {elm_analysis['trial_count']}回
- 平均スコア: {elm_analysis['score_stats']['mean']:.1f} ± {elm_analysis['score_stats']['std']:.1f}点
- スコア範囲: {elm_analysis['score_stats']['min']:.0f} - {elm_analysis['score_stats']['max']:.0f}点
- 平均タワー配置数: {elm_analysis['towers_stats']['mean']:.1f}個
- 学習改善: {elm_analysis['learning_progress']['improvement']:.1f}点

### ELM+LLM条件
- 試行回数: {elm_llm_analysis['trial_count']}回
- 平均スコア: {elm_llm_analysis['score_stats']['mean']:.1f} ± {elm_llm_analysis['score_stats']['std']:.1f}点
- スコア範囲: {elm_llm_analysis['score_stats']['min']:.0f} - {elm_llm_analysis['score_stats']['max']:.0f}点
- 平均タワー配置数: {elm_llm_analysis['towers_stats']['mean']:.1f}個
- 学習改善: {elm_llm_analysis['learning_progress']['improvement']:.1f}点

## 統計的比較

### 主要結果
- **平均スコア差**: {comparison['descriptive']['difference']:.1f}点
- **改善率**: {comparison['descriptive']['improvement_rate']:.1f}%
- **効果量 (Cohen's d)**: {comparison['effect_size']['cohens_d']:.3f} ({comparison['effect_size']['interpretation']})

### 統計的検定
- **t検定**: t = {comparison['t_test']['statistic']:.3f}, p = {comparison['t_test']['p_value']:.6f}
- **有意性**: {'有意' if comparison['t_test']['significant'] else '非有意'} (α = 0.05)
- **Mann-Whitney U検定**: U = {comparison['mann_whitney']['statistic']:.0f}, p = {comparison['mann_whitney']['p_value']:.6f}

### 信頼区間 (95%)
- **ELMのみ**: [{comparison['confidence_intervals']['elm_only'][0]:.1f}, {comparison['confidence_intervals']['elm_only'][1]:.1f}]
- **ELM+LLM**: [{comparison['confidence_intervals']['elm_llm'][0]:.1f}, {comparison['confidence_intervals']['elm_llm'][1]:.1f}]

## 結論

LLMガイダンスは学習効率に{'統計的に有意な' if comparison['t_test']['significant'] else '有意でない'}影響を与えた。
効果量は{comparison['effect_size']['interpretation']}であり、実用的に{'重要' if abs(comparison['effect_size']['cohens_d']) > 0.5 else '限定的'}な改善を示している。

## 注意事項

この分析は過去の実験データに基づいており、現在の実験では技術的問題によりLLMガイダンス機能が正常に動作していない。
したがって、この結果は参考値として扱い、技術的問題を解決した上での再実験が推奨される。
"""
    
    return report

def main():
    """メイン分析実行"""
    print("Tower Defense ELM Learning Efficiency Experiment - 統計分析開始")
    
    # データ読み込み
    data = load_experiment_data()
    if data is None:
        return
    
    print(f"データ読み込み完了: ELMのみ {len(data['elm_only'])}回, ELM+LLM {len(data['elm_with_llm'])}回")
    
    # 分析実行
    elm_analysis = analyze_elm_only_performance(data['elm_only'])
    elm_llm_analysis = analyze_elm_llm_performance(data['elm_with_llm'])
    comparison = statistical_comparison(data['elm_only'], data['elm_with_llm'])
    
    print("統計分析完了")
    
    # 可視化作成
    create_visualizations(data)
    print("可視化作成完了")
    
    # レポート生成
    report = generate_report(elm_analysis, elm_llm_analysis, comparison)
    
    with open('statistical_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("分析レポート生成完了: statistical_analysis_report.md")
    
    # 結果の要約表示
    print("\n=== 分析結果要約 ===")
    print(f"ELMのみ平均スコア: {elm_analysis['score_stats']['mean']:.1f}点")
    print(f"ELM+LLM平均スコア: {elm_llm_analysis['score_stats']['mean']:.1f}点")
    print(f"改善: {comparison['descriptive']['difference']:.1f}点 ({comparison['descriptive']['improvement_rate']:.1f}%)")
    print(f"効果量: {comparison['effect_size']['cohens_d']:.3f} ({comparison['effect_size']['interpretation']})")
    print(f"統計的有意性: {'有意' if comparison['t_test']['significant'] else '非有意'} (p = {comparison['t_test']['p_value']:.6f})")

if __name__ == "__main__":
    main()
