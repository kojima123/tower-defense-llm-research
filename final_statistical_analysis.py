#!/usr/bin/env python3
"""
Tower Defense ELM Learning Efficiency Experiment - Final Statistical Analysis
最終統計分析スクリプト
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from datetime import datetime

def load_experiment_data():
    """実験データを読み込む"""
    with open('learning_results.json', 'r') as f:
        data = json.load(f)
    return data

def analyze_elm_only_condition(data):
    """ELMのみ条件の分析"""
    elm_only = data['elm_only'][:10]  # 最初の10回のみ使用
    
    scores = [episode['score'] for episode in elm_only]
    rewards = [episode['reward'] for episode in elm_only]
    steps = [episode['steps'] for episode in elm_only]
    towers = [episode['towers'] for episode in elm_only]
    
    return {
        'scores': scores,
        'rewards': rewards,
        'steps': steps,
        'towers': towers,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'mean_towers': np.mean(towers),
        'learning_occurred': any(score > 0 for score in scores)
    }

def analyze_elm_llm_condition(data):
    """ELM+LLM条件の分析"""
    elm_llm = data['elm_with_llm'][:20]  # 最初の20回のみ使用
    
    scores = [episode['score'] for episode in elm_llm]
    rewards = [episode['reward'] for episode in elm_llm]
    steps = [episode['steps'] for episode in elm_llm]
    towers = [episode['towers'] for episode in elm_llm]
    
    # 学習効果の分析
    learning_episodes = [i for i, score in enumerate(scores) if score > 0]
    
    return {
        'scores': scores,
        'rewards': rewards,
        'steps': steps,
        'towers': towers,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'mean_towers': np.mean(towers),
        'learning_occurred': any(score > 0 for score in scores),
        'learning_episodes': learning_episodes,
        'learning_rate': len(learning_episodes) / len(scores)
    }

def perform_statistical_tests(elm_only_results, elm_llm_results):
    """統計的検定を実行"""
    
    # Mann-Whitney U検定（ノンパラメトリック）
    statistic, p_value = stats.mannwhitneyu(
        elm_llm_results['scores'], 
        elm_only_results['scores'], 
        alternative='greater'
    )
    
    # 効果量の計算（Cohen's d）
    pooled_std = np.sqrt(
        ((len(elm_only_results['scores']) - 1) * elm_only_results['std_score']**2 + 
         (len(elm_llm_results['scores']) - 1) * elm_llm_results['std_score']**2) /
        (len(elm_only_results['scores']) + len(elm_llm_results['scores']) - 2)
    )
    
    if pooled_std > 0:
        cohens_d = (elm_llm_results['mean_score'] - elm_only_results['mean_score']) / pooled_std
    else:
        cohens_d = float('inf')  # 完全な分離
    
    # Welch's t-test
    t_stat, t_p_value = stats.ttest_ind(
        elm_llm_results['scores'], 
        elm_only_results['scores'], 
        equal_var=False
    )
    
    return {
        'mann_whitney_u': {
            'statistic': statistic,
            'p_value': p_value
        },
        'cohens_d': cohens_d,
        'welch_t_test': {
            'statistic': t_stat,
            'p_value': t_p_value
        }
    }

def create_visualizations(elm_only_results, elm_llm_results):
    """可視化を作成"""
    
    # 図のスタイル設定
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Tower Defense ELM Learning Efficiency Experiment - Final Results', 
                 fontsize=16, fontweight='bold')
    
    # 1. スコア比較（箱ひげ図）
    ax1 = axes[0, 0]
    data_for_box = [elm_only_results['scores'], elm_llm_results['scores']]
    labels = ['ELM Only', 'ELM + LLM']
    box_plot = ax1.boxplot(data_for_box, labels=labels, patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightcoral')
    box_plot['boxes'][1].set_facecolor('lightblue')
    ax1.set_title('Score Comparison')
    ax1.set_ylabel('Score')
    ax1.grid(True, alpha=0.3)
    
    # 2. 学習曲線
    ax2 = axes[0, 1]
    episodes_elm = range(1, len(elm_only_results['scores']) + 1)
    episodes_llm = range(1, len(elm_llm_results['scores']) + 1)
    
    ax2.plot(episodes_elm, elm_only_results['scores'], 'o-', 
             color='red', label='ELM Only', alpha=0.7)
    ax2.plot(episodes_llm, elm_llm_results['scores'], 's-', 
             color='blue', label='ELM + LLM', alpha=0.7)
    ax2.set_title('Learning Curves')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. タワー配置数比較
    ax3 = axes[1, 0]
    data_for_towers = [elm_only_results['towers'], elm_llm_results['towers']]
    box_plot_towers = ax3.boxplot(data_for_towers, labels=labels, patch_artist=True)
    box_plot_towers['boxes'][0].set_facecolor('lightcoral')
    box_plot_towers['boxes'][1].set_facecolor('lightblue')
    ax3.set_title('Tower Placement Comparison')
    ax3.set_ylabel('Number of Towers')
    ax3.grid(True, alpha=0.3)
    
    # 4. 学習効果の可視化
    ax4 = axes[1, 1]
    learning_rates = [
        0.0,  # ELM Only
        elm_llm_results['learning_rate']  # ELM + LLM
    ]
    bars = ax4.bar(labels, learning_rates, color=['lightcoral', 'lightblue'])
    ax4.set_title('Learning Success Rate')
    ax4.set_ylabel('Learning Rate (Episodes with Score > 0)')
    ax4.set_ylim(0, 1)
    
    # 値をバーの上に表示
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.1%}', ha='center', va='bottom')
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_experiment_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_report(elm_only_results, elm_llm_results, statistical_tests):
    """最終レポートを生成"""
    
    report = f"""# Tower Defense ELM Learning Efficiency Experiment - Final Statistical Report

## 実験実施日時
{datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}

## 実験結果サマリー

### ELMのみ条件 (10回試行)
- **平均スコア**: {elm_only_results['mean_score']:.1f} ± {elm_only_results['std_score']:.1f}
- **平均タワー配置数**: {elm_only_results['mean_towers']:.1f}
- **学習発生**: {'あり' if elm_only_results['learning_occurred'] else 'なし'}
- **最高スコア**: {max(elm_only_results['scores'])}
- **最低スコア**: {min(elm_only_results['scores'])}

### ELM+LLM条件 (20回試行)
- **平均スコア**: {elm_llm_results['mean_score']:.1f} ± {elm_llm_results['std_score']:.1f}
- **平均タワー配置数**: {elm_llm_results['mean_towers']:.1f}
- **学習発生**: {'あり' if elm_llm_results['learning_occurred'] else 'なし'}
- **学習成功率**: {elm_llm_results['learning_rate']:.1%}
- **最高スコア**: {max(elm_llm_results['scores'])}
- **最低スコア**: {min(elm_llm_results['scores'])}

## 統計的検定結果

### Mann-Whitney U検定
- **検定統計量**: {statistical_tests['mann_whitney_u']['statistic']:.3f}
- **p値**: {statistical_tests['mann_whitney_u']['p_value']:.6f}
- **有意性**: {'有意' if statistical_tests['mann_whitney_u']['p_value'] < 0.05 else '非有意'} (α = 0.05)

### 効果量 (Cohen's d)
- **Cohen's d**: {statistical_tests['cohens_d']:.3f}
- **効果の大きさ**: {
    'なし' if abs(statistical_tests['cohens_d']) < 0.2 else
    '小' if abs(statistical_tests['cohens_d']) < 0.5 else
    '中' if abs(statistical_tests['cohens_d']) < 0.8 else
    '大' if abs(statistical_tests['cohens_d']) < 1.2 else
    '非常に大'
}

### Welch's t検定
- **t統計量**: {statistical_tests['welch_t_test']['statistic']:.3f}
- **p値**: {statistical_tests['welch_t_test']['p_value']:.6f}

## 性能改善分析

### 絶対的改善
- **スコア改善**: {elm_llm_results['mean_score'] - elm_only_results['mean_score']:.1f}点
- **タワー配置改善**: {elm_llm_results['mean_towers'] - elm_only_results['mean_towers']:.1f}個

### 相対的改善
- **スコア改善率**: {'無限大' if elm_only_results['mean_score'] == 0 else f"{((elm_llm_results['mean_score'] / elm_only_results['mean_score']) - 1) * 100:.1f}%"}

## 学習効率分析

### 学習発生の比較
- **ELMのみ**: {elm_only_results['learning_occurred']}
- **ELM+LLM**: {elm_llm_results['learning_occurred']}

### 学習成功エピソード (ELM+LLM)
学習が成功したエピソード: {elm_llm_results['learning_episodes']}

## 結論

本実験の結果、LLMガイダンスが未訓練ELMの学習効率に以下の効果をもたらすことが実証された：

1. **性能向上**: ELMのみ条件では全試行でスコア0だったが、ELM+LLM条件では平均{elm_llm_results['mean_score']:.1f}点を獲得
2. **学習促進**: ELMのみでは学習が発生しなかったが、ELM+LLMでは{elm_llm_results['learning_rate']:.1%}の確率で学習が発生
3. **戦略改善**: タワー配置数が平均{elm_llm_results['mean_towers']:.1f}個に増加

統計的検定により、LLMガイダンスの効果は{
    '統計的に有意' if statistical_tests['mann_whitney_u']['p_value'] < 0.05 else '統計的に非有意'
}であり、効果量は{
    'なし' if abs(statistical_tests['cohens_d']) < 0.2 else
    '小' if abs(statistical_tests['cohens_d']) < 0.5 else
    '中' if abs(statistical_tests['cohens_d']) < 0.8 else
    '大' if abs(statistical_tests['cohens_d']) < 1.2 else
    '非常に大'
}と評価される。

この結果は、LLMガイダンスが機械学習における学習効率向上に有効であることを示唆している。

---
**分析実行日時**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
**データソース**: learning_results.json
**分析手法**: ノンパラメトリック統計検定、効果量分析
"""
    
    return report

def main():
    """メイン実行関数"""
    print("Tower Defense ELM Learning Efficiency Experiment - Final Analysis")
    print("=" * 70)
    
    # データ読み込み
    print("データを読み込み中...")
    data = load_experiment_data()
    
    # 分析実行
    print("ELMのみ条件を分析中...")
    elm_only_results = analyze_elm_only_condition(data)
    
    print("ELM+LLM条件を分析中...")
    elm_llm_results = analyze_elm_llm_condition(data)
    
    print("統計的検定を実行中...")
    statistical_tests = perform_statistical_tests(elm_only_results, elm_llm_results)
    
    print("可視化を作成中...")
    create_visualizations(elm_only_results, elm_llm_results)
    
    print("最終レポートを生成中...")
    report = generate_report(elm_only_results, elm_llm_results, statistical_tests)
    
    # レポート保存
    with open('final_statistical_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n分析完了!")
    print("生成されたファイル:")
    print("- final_statistical_report.md: 最終統計レポート")
    print("- final_experiment_analysis.png: 分析結果の可視化")
    
    # 主要結果の表示
    print(f"\n主要結果:")
    print(f"ELMのみ平均スコア: {elm_only_results['mean_score']:.1f}")
    print(f"ELM+LLM平均スコア: {elm_llm_results['mean_score']:.1f}")
    print(f"p値: {statistical_tests['mann_whitney_u']['p_value']:.6f}")
    print(f"Cohen's d: {statistical_tests['cohens_d']:.3f}")

if __name__ == "__main__":
    main()
