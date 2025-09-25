"""
Tower Defense Comparison Experiment System
ELMのみとELM+LLM教師の比較実験システム
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from typing import Dict, List
from elm_tower_defense import run_elm_experiment
from elm_llm_tower_defense import run_elm_llm_experiment
import matplotlib.font_manager as fm

# 日本語フォント設定
plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']

class TowerDefenseComparison:
    """Tower Defense比較実験システム"""
    
    def __init__(self):
        self.results = {}
        self.comparison_data = {}
    
    def run_comparison_experiment(self, episodes: int = 30, runs: int = 3) -> Dict:
        """比較実験を実行"""
        print(f"Tower Defense比較実験開始 - {runs}回実行、各{episodes}エピソード")
        
        elm_results = []
        hybrid_results = []
        
        for run in range(runs):
            print(f"\n=== 実行 {run + 1}/{runs} ===")
            
            # ELMのみの実験
            print("ELMのみの実験を実行中...")
            elm_result = run_elm_experiment(episodes)
            elm_results.append(elm_result)
            
            # ELM + LLM教師の実験
            print("ELM + LLM教師の実験を実行中...")
            hybrid_result = run_elm_llm_experiment(episodes)
            hybrid_results.append(hybrid_result)
        
        # 結果を統合
        self.results = {
            'elm_only': elm_results,
            'elm_llm_hybrid': hybrid_results,
            'experiment_config': {
                'episodes': episodes,
                'runs': runs,
                'timestamp': time.time()
            }
        }
        
        # 比較分析を実行
        self.comparison_data = self._analyze_results()
        
        return self.comparison_data
    
    def _analyze_results(self) -> Dict:
        """結果を分析"""
        elm_results = self.results['elm_only']
        hybrid_results = self.results['elm_llm_hybrid']
        
        # 各指標の統計を計算
        analysis = {
            'elm_only': self._calculate_statistics(elm_results),
            'elm_llm_hybrid': self._calculate_statistics(hybrid_results),
            'improvements': {},
            'statistical_significance': {}
        }
        
        # 改善率を計算
        for metric in ['final_avg_score', 'final_avg_efficiency']:
            elm_values = [result[metric] for result in elm_results]
            hybrid_values = [result[metric] for result in hybrid_results]
            
            elm_mean = np.mean(elm_values)
            hybrid_mean = np.mean(hybrid_values)
            
            if elm_mean > 0:
                improvement = ((hybrid_mean - elm_mean) / elm_mean) * 100
            else:
                improvement = 0 if hybrid_mean == 0 else float('inf')
            
            analysis['improvements'][metric] = improvement
            
            # 統計的有意性をテスト（簡易版）
            if len(elm_values) > 1 and len(hybrid_values) > 1:
                t_stat, p_value = self._simple_t_test(elm_values, hybrid_values)
                analysis['statistical_significance'][metric] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        # コスト・時間分析
        analysis['cost_analysis'] = self._analyze_costs()
        
        return analysis
    
    def _calculate_statistics(self, results: List[Dict]) -> Dict:
        """統計情報を計算"""
        metrics = ['final_avg_score', 'final_avg_efficiency']
        stats = {}
        
        for metric in metrics:
            values = [result[metric] for result in results]
            stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        # エピソード別の詳細統計
        all_scores = []
        all_survival_times = []
        all_towers = []
        all_enemies_killed = []
        
        for result in results:
            all_scores.extend(result['scores'])
            all_survival_times.extend(result['survival_times'])
            all_towers.extend(result['towers_built'])
            all_enemies_killed.extend(result['enemies_killed'])
        
        stats['detailed'] = {
            'scores': {
                'mean': np.mean(all_scores),
                'std': np.std(all_scores),
                'total_episodes': len(all_scores)
            },
            'survival_times': {
                'mean': np.mean(all_survival_times),
                'std': np.std(all_survival_times)
            },
            'towers_built': {
                'mean': np.mean(all_towers),
                'std': np.std(all_towers)
            },
            'enemies_killed': {
                'mean': np.mean(all_enemies_killed),
                'std': np.std(all_enemies_killed)
            }
        }
        
        return stats
    
    def _simple_t_test(self, sample1: List[float], sample2: List[float]) -> tuple:
        """簡易t検定"""
        n1, n2 = len(sample1), len(sample2)
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
        
        # プールされた標準偏差
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        # t統計量
        t_stat = (mean2 - mean1) / (pooled_std * np.sqrt(1/n1 + 1/n2))
        
        # 自由度
        df = n1 + n2 - 2
        
        # 簡易p値計算（正規近似）
        p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))
        
        return t_stat, p_value
    
    def _normal_cdf(self, x: float) -> float:
        """標準正規分布の累積分布関数（近似）"""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x * x / np.pi)))
    
    def _analyze_costs(self) -> Dict:
        """コスト分析"""
        elm_results = self.results['elm_only']
        hybrid_results = self.results['elm_llm_hybrid']
        
        # ELMのみのコスト
        elm_training_times = []
        elm_inference_times = []
        
        for result in elm_results:
            stats = result['agent_stats']
            elm_training_times.append(stats['avg_training_time'])
            elm_inference_times.append(stats['avg_inference_time'])
        
        # ハイブリッドのコスト
        hybrid_training_times = []
        hybrid_inference_times = []
        api_times = []
        api_calls = []
        
        for result in hybrid_results:
            stats = result['agent_stats']
            hybrid_training_times.append(stats['avg_training_time'])
            hybrid_inference_times.append(stats['avg_inference_time'])
            
            llm_stats = stats['llm_stats']
            api_times.append(llm_stats['avg_api_time'])
            api_calls.append(llm_stats['api_calls'])
        
        # APIコスト計算（GPT-4.1-miniの料金）
        input_cost_per_1k = 0.00015  # $0.00015 per 1K input tokens
        output_cost_per_1k = 0.0006  # $0.0006 per 1K output tokens
        avg_input_tokens = 300  # 推定
        avg_output_tokens = 100  # 推定
        
        cost_per_call = (avg_input_tokens * input_cost_per_1k / 1000 + 
                        avg_output_tokens * output_cost_per_1k / 1000)
        
        total_api_calls = sum(api_calls)
        total_api_cost = total_api_calls * cost_per_call
        
        return {
            'elm_only': {
                'avg_training_time': np.mean(elm_training_times),
                'avg_inference_time': np.mean(elm_inference_times),
                'total_cost': 0.0
            },
            'elm_llm_hybrid': {
                'avg_training_time': np.mean(hybrid_training_times),
                'avg_inference_time': np.mean(hybrid_inference_times),
                'avg_api_time': np.mean(api_times),
                'total_api_calls': total_api_calls,
                'total_api_cost': total_api_cost,
                'cost_per_episode': total_api_cost / (len(hybrid_results) * self.results['experiment_config']['episodes'])
            },
            'cost_efficiency': {
                'improvement_per_dollar': self.comparison_data.get('improvements', {}).get('final_avg_score', 0) / max(total_api_cost, 0.001),
                'roi_score': self.comparison_data.get('improvements', {}).get('final_avg_score', 0) / max(total_api_cost * 1000, 1)  # ROI per $0.001
            }
        }
    
    def create_visualizations(self):
        """可視化を作成"""
        if not self.comparison_data:
            print("比較データがありません。先に実験を実行してください。")
            return
        
        # 図1: 性能比較
        self._create_performance_comparison()
        
        # 図2: 学習曲線
        self._create_learning_curves()
        
        # 図3: コスト分析
        self._create_cost_analysis()
        
        # 図4: 統計的有意性
        self._create_statistical_analysis()
    
    def _create_performance_comparison(self):
        """性能比較グラフを作成"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Tower Defense: ELM vs ELM+LLM教師 性能比較', fontsize=16, fontweight='bold')
        
        elm_data = self.comparison_data['elm_only']
        hybrid_data = self.comparison_data['elm_llm_hybrid']
        
        # スコア比較
        ax1 = axes[0, 0]
        scores = [elm_data['final_avg_score']['mean'], hybrid_data['final_avg_score']['mean']]
        errors = [elm_data['final_avg_score']['std'], hybrid_data['final_avg_score']['std']]
        bars1 = ax1.bar(['ELMのみ', 'ELM+LLM教師'], scores, yerr=errors, 
                       color=['#3498db', '#e74c3c'], alpha=0.8, capsize=5)
        ax1.set_title('平均最終スコア', fontweight='bold')
        ax1.set_ylabel('スコア')
        ax1.grid(True, alpha=0.3)
        
        # 効率性比較
        ax2 = axes[0, 1]
        efficiency = [elm_data['final_avg_efficiency']['mean'], hybrid_data['final_avg_efficiency']['mean']]
        eff_errors = [elm_data['final_avg_efficiency']['std'], hybrid_data['final_avg_efficiency']['std']]
        bars2 = ax2.bar(['ELMのみ', 'ELM+LLM教師'], efficiency, yerr=eff_errors,
                       color=['#3498db', '#e74c3c'], alpha=0.8, capsize=5)
        ax2.set_title('平均効率性', fontweight='bold')
        ax2.set_ylabel('効率性')
        ax2.grid(True, alpha=0.3)
        
        # 改善率
        ax3 = axes[1, 0]
        improvements = [self.comparison_data['improvements']['final_avg_score'],
                       self.comparison_data['improvements']['final_avg_efficiency']]
        bars3 = ax3.bar(['スコア改善率', '効率性改善率'], improvements,
                       color=['#2ecc71', '#f39c12'], alpha=0.8)
        ax3.set_title('LLM教師による改善率', fontweight='bold')
        ax3.set_ylabel('改善率 (%)')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 詳細統計
        ax4 = axes[1, 1]
        metrics = ['平均スコア', '平均生存時間', '平均タワー数', '平均撃破数']
        elm_values = [
            elm_data['detailed']['scores']['mean'],
            elm_data['detailed']['survival_times']['mean'],
            elm_data['detailed']['towers_built']['mean'],
            elm_data['detailed']['enemies_killed']['mean']
        ]
        hybrid_values = [
            hybrid_data['detailed']['scores']['mean'],
            hybrid_data['detailed']['survival_times']['mean'],
            hybrid_data['detailed']['towers_built']['mean'],
            hybrid_data['detailed']['enemies_killed']['mean']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        bars4_1 = ax4.bar(x - width/2, elm_values, width, label='ELMのみ', color='#3498db', alpha=0.8)
        bars4_2 = ax4.bar(x + width/2, hybrid_values, width, label='ELM+LLM教師', color='#e74c3c', alpha=0.8)
        
        ax4.set_title('詳細統計比較', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/tower-defense-llm/tower_defense_performance_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_learning_curves(self):
        """学習曲線を作成"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Tower Defense: 学習曲線比較', fontsize=16, fontweight='bold')
        
        elm_results = self.results['elm_only']
        hybrid_results = self.results['elm_llm_hybrid']
        
        # スコアの学習曲線
        ax1 = axes[0, 0]
        for i, result in enumerate(elm_results):
            ax1.plot(result['scores'], alpha=0.6, color='#3498db', linewidth=1)
        for i, result in enumerate(hybrid_results):
            ax1.plot(result['scores'], alpha=0.6, color='#e74c3c', linewidth=1)
        
        # 平均線を追加
        elm_avg_scores = np.mean([result['scores'] for result in elm_results], axis=0)
        hybrid_avg_scores = np.mean([result['scores'] for result in hybrid_results], axis=0)
        ax1.plot(elm_avg_scores, color='#2980b9', linewidth=3, label='ELMのみ (平均)')
        ax1.plot(hybrid_avg_scores, color='#c0392b', linewidth=3, label='ELM+LLM教師 (平均)')
        
        ax1.set_title('スコア学習曲線', fontweight='bold')
        ax1.set_xlabel('エピソード')
        ax1.set_ylabel('スコア')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 効率性の学習曲線
        ax2 = axes[0, 1]
        for i, result in enumerate(elm_results):
            ax2.plot(result['efficiency'], alpha=0.6, color='#3498db', linewidth=1)
        for i, result in enumerate(hybrid_results):
            ax2.plot(result['efficiency'], alpha=0.6, color='#e74c3c', linewidth=1)
        
        elm_avg_eff = np.mean([result['efficiency'] for result in elm_results], axis=0)
        hybrid_avg_eff = np.mean([result['efficiency'] for result in hybrid_results], axis=0)
        ax2.plot(elm_avg_eff, color='#2980b9', linewidth=3, label='ELMのみ (平均)')
        ax2.plot(hybrid_avg_eff, color='#c0392b', linewidth=3, label='ELM+LLM教師 (平均)')
        
        ax2.set_title('効率性学習曲線', fontweight='bold')
        ax2.set_xlabel('エピソード')
        ax2.set_ylabel('効率性')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 生存時間
        ax3 = axes[1, 0]
        elm_avg_survival = np.mean([result['survival_times'] for result in elm_results], axis=0)
        hybrid_avg_survival = np.mean([result['survival_times'] for result in hybrid_results], axis=0)
        ax3.plot(elm_avg_survival, color='#2980b9', linewidth=3, label='ELMのみ')
        ax3.plot(hybrid_avg_survival, color='#c0392b', linewidth=3, label='ELM+LLM教師')
        
        ax3.set_title('平均生存時間', fontweight='bold')
        ax3.set_xlabel('エピソード')
        ax3.set_ylabel('生存時間 (秒)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 撃破数
        ax4 = axes[1, 1]
        elm_avg_kills = np.mean([result['enemies_killed'] for result in elm_results], axis=0)
        hybrid_avg_kills = np.mean([result['enemies_killed'] for result in hybrid_results], axis=0)
        ax4.plot(elm_avg_kills, color='#2980b9', linewidth=3, label='ELMのみ')
        ax4.plot(hybrid_avg_kills, color='#c0392b', linewidth=3, label='ELM+LLM教師')
        
        ax4.set_title('平均敵撃破数', fontweight='bold')
        ax4.set_xlabel('エピソード')
        ax4.set_ylabel('撃破数')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/tower-defense-llm/tower_defense_learning_curves.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_cost_analysis(self):
        """コスト分析グラフを作成"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Tower Defense: コスト・時間分析', fontsize=16, fontweight='bold')
        
        cost_data = self.comparison_data['cost_analysis']
        
        # 実行時間比較
        ax1 = axes[0, 0]
        elm_times = [cost_data['elm_only']['avg_training_time'], cost_data['elm_only']['avg_inference_time']]
        hybrid_times = [cost_data['elm_llm_hybrid']['avg_training_time'], cost_data['elm_llm_hybrid']['avg_inference_time']]
        
        x = np.arange(2)
        width = 0.35
        ax1.bar(x - width/2, elm_times, width, label='ELMのみ', color='#3498db', alpha=0.8)
        ax1.bar(x + width/2, hybrid_times, width, label='ELM+LLM教師', color='#e74c3c', alpha=0.8)
        
        ax1.set_title('平均実行時間比較', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['学習時間', '推論時間'])
        ax1.set_ylabel('時間 (秒)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # APIコスト
        ax2 = axes[0, 1]
        costs = [0, cost_data['elm_llm_hybrid']['total_api_cost']]
        bars = ax2.bar(['ELMのみ', 'ELM+LLM教師'], costs, color=['#3498db', '#e74c3c'], alpha=0.8)
        ax2.set_title('総APIコスト', fontweight='bold')
        ax2.set_ylabel('コスト ($)')
        ax2.grid(True, alpha=0.3)
        
        # ROI分析
        ax3 = axes[1, 0]
        roi_score = cost_data['cost_efficiency']['roi_score']
        improvement_per_dollar = cost_data['cost_efficiency']['improvement_per_dollar']
        
        ax3.bar(['ROIスコア', '改善率/$'], [roi_score, improvement_per_dollar], 
               color=['#2ecc71', '#f39c12'], alpha=0.8)
        ax3.set_title('投資対効果分析', fontweight='bold')
        ax3.set_ylabel('効率性指標')
        ax3.grid(True, alpha=0.3)
        
        # 時間オーバーヘッド
        ax4 = axes[1, 1]
        if cost_data['elm_llm_hybrid']['avg_api_time'] > 0:
            overhead = (cost_data['elm_llm_hybrid']['avg_api_time'] / 
                       cost_data['elm_only']['avg_inference_time']) * 100
        else:
            overhead = 0
        
        ax4.bar(['時間オーバーヘッド'], [overhead], color='#9b59b6', alpha=0.8)
        ax4.set_title('LLM統合による時間オーバーヘッド', fontweight='bold')
        ax4.set_ylabel('オーバーヘッド (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/tower-defense-llm/tower_defense_cost_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_statistical_analysis(self):
        """統計的有意性分析を作成"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('Tower Defense: 統計的有意性分析', fontsize=16, fontweight='bold')
        
        # p値の可視化
        ax1 = axes[0]
        metrics = list(self.comparison_data['statistical_significance'].keys())
        p_values = [self.comparison_data['statistical_significance'][metric]['p_value'] for metric in metrics]
        
        colors = ['#2ecc71' if p < 0.05 else '#e74c3c' for p in p_values]
        bars = ax1.bar(range(len(metrics)), p_values, color=colors, alpha=0.8)
        
        ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='有意水準 (p=0.05)')
        ax1.set_title('統計的有意性 (p値)', fontweight='bold')
        ax1.set_xticks(range(len(metrics)))
        ax1.set_xticklabels([m.replace('final_avg_', '') for m in metrics], rotation=45)
        ax1.set_ylabel('p値')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 効果サイズ
        ax2 = axes[1]
        improvements = [self.comparison_data['improvements'][metric] for metric in metrics]
        colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
        bars = ax2.bar(range(len(metrics)), improvements, color=colors, alpha=0.8)
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('改善効果サイズ', fontweight='bold')
        ax2.set_xticks(range(len(metrics)))
        ax2.set_xticklabels([m.replace('final_avg_', '') for m in metrics], rotation=45)
        ax2.set_ylabel('改善率 (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/tower-defense-llm/tower_defense_statistical_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, filename: str = 'tower_defense_comparison_results.json'):
        """結果をJSONファイルに保存"""
        filepath = f'/home/ubuntu/tower-defense-llm/{filename}'
        
        # NumPy配列をリストに変換
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        save_data = {
            'results': convert_numpy(self.results),
            'comparison_data': convert_numpy(self.comparison_data)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"結果を {filepath} に保存しました")
    
    def generate_report(self) -> str:
        """詳細レポートを生成"""
        if not self.comparison_data:
            return "比較データがありません。先に実験を実行してください。"
        
        report = f"""
# Tower Defense: ELM vs ELM+LLM教師 比較実験レポート

## 実験概要
- 実行回数: {self.results['experiment_config']['runs']}回
- エピソード数: {self.results['experiment_config']['episodes']}エピソード/回
- 実験日時: {time.ctime(self.results['experiment_config']['timestamp'])}

## 主要結果

### 性能比較
**最終平均スコア:**
- ELMのみ: {self.comparison_data['elm_only']['final_avg_score']['mean']:.2f} ± {self.comparison_data['elm_only']['final_avg_score']['std']:.2f}
- ELM+LLM教師: {self.comparison_data['elm_llm_hybrid']['final_avg_score']['mean']:.2f} ± {self.comparison_data['elm_llm_hybrid']['final_avg_score']['std']:.2f}
- **改善率: {self.comparison_data['improvements']['final_avg_score']:.1f}%**

**効率性:**
- ELMのみ: {self.comparison_data['elm_only']['final_avg_efficiency']['mean']:.3f} ± {self.comparison_data['elm_only']['final_avg_efficiency']['std']:.3f}
- ELM+LLM教師: {self.comparison_data['elm_llm_hybrid']['final_avg_efficiency']['mean']:.3f} ± {self.comparison_data['elm_llm_hybrid']['final_avg_efficiency']['std']:.3f}
- **改善率: {self.comparison_data['improvements']['final_avg_efficiency']:.1f}%**

### 統計的有意性
"""
        
        for metric, stats in self.comparison_data['statistical_significance'].items():
            significance = "有意" if stats['significant'] else "非有意"
            report += f"- {metric}: p={stats['p_value']:.4f} ({significance})\n"
        
        report += f"""
### コスト・時間分析
**実行時間:**
- ELM学習時間: {self.comparison_data['cost_analysis']['elm_only']['avg_training_time']:.4f}秒
- ELM推論時間: {self.comparison_data['cost_analysis']['elm_only']['avg_inference_time']:.4f}秒
- LLM API時間: {self.comparison_data['cost_analysis']['elm_llm_hybrid']['avg_api_time']:.4f}秒

**コスト:**
- 総APIコスト: ${self.comparison_data['cost_analysis']['elm_llm_hybrid']['total_api_cost']:.6f}
- エピソードあたりコスト: ${self.comparison_data['cost_analysis']['elm_llm_hybrid']['cost_per_episode']:.6f}
- ROIスコア: {self.comparison_data['cost_analysis']['cost_efficiency']['roi_score']:.2f}

## 結論
"""
        
        if self.comparison_data['improvements']['final_avg_score'] > 0:
            report += f"LLM教師システムは{self.comparison_data['improvements']['final_avg_score']:.1f}%の性能向上を実現し、"
        else:
            report += "LLM教師システムは性能向上を実現できませんでしたが、"
        
        report += f"コストは${self.comparison_data['cost_analysis']['elm_llm_hybrid']['cost_per_episode']:.6f}/エピソードと非常に低く、実用的な価値があります。"
        
        return report


def main():
    """メイン実行関数"""
    comparison = TowerDefenseComparison()
    
    # 比較実験を実行
    results = comparison.run_comparison_experiment(episodes=10, runs=3)
    
    # 可視化を作成
    comparison.create_visualizations()
    
    # 結果を保存
    comparison.save_results()
    
    # レポートを生成
    report = comparison.generate_report()
    print(report)
    
    # レポートをファイルに保存
    with open('/home/ubuntu/tower-defense-llm/tower_defense_comparison_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n比較実験が完了しました！")
    print("生成されたファイル:")
    print("- tower_defense_performance_comparison.png")
    print("- tower_defense_learning_curves.png") 
    print("- tower_defense_cost_analysis.png")
    print("- tower_defense_statistical_analysis.png")
    print("- tower_defense_comparison_results.json")
    print("- tower_defense_comparison_report.md")


if __name__ == "__main__":
    main()
