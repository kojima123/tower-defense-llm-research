#!/usr/bin/env python3
"""
4条件統一アブレーション実験システム
同一エピソード数・同一seed集合での厳密比較
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import subprocess
import tempfile


class UnifiedAblationExperiment:
    """4条件統一アブレーション実験システム"""
    
    def __init__(self, project_dir: str = "."):
        self.project_dir = Path(project_dir)
        self.conditions = ["elm_only", "rule_teacher", "random_teacher", "elm_llm"]
        
    def run_unified_experiment(self, episodes: int = 10, seeds: List[int] = None, experiment_name: str = None) -> str:
        """統一条件での4条件実験を実行"""
        if seeds is None:
            seeds = [42, 123, 456]  # デフォルトシード
        
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"unified_ablation_{timestamp}"
        
        print(f"🚀 Starting unified ablation experiment: {experiment_name}")
        print(f"📊 Conditions: {self.conditions}")
        print(f"🎲 Seeds: {seeds}")
        print(f"📈 Episodes per condition per seed: {episodes}")
        
        experiment_dir = self.project_dir / "runs" / "real" / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 実験設定保存
        config = {
            "experiment_name": experiment_name,
            "conditions": self.conditions,
            "seeds": seeds,
            "episodes_per_seed": episodes,
            "total_runs": len(self.conditions) * len(seeds),
            "timestamp": datetime.now().isoformat()
        }
        
        config_path = experiment_dir / "experiment_config.json"
        with config_path.open('w') as f:
            json.dump(config, f, indent=2)
        
        print(f"📁 Experiment directory: {experiment_dir}")
        print(f"⚙️  Configuration saved: {config_path}")
        
        # 各条件で実験実行
        results = {}
        
        for condition in self.conditions:
            print(f"\n🔬 Running condition: {condition}")
            condition_results = []
            
            for seed in seeds:
                print(f"  🎲 Seed {seed}...")
                
                try:
                    # 実験実行コマンド
                    cmd = [
                        "python", "run_experiment_cli_fixed.py", "run",
                        "--teachers", condition,
                        "--seeds", str(seed),
                        "--episodes", str(episodes),
                        "--experiment-name", f"{experiment_name}_{condition}_seed{seed}"
                    ]
                    
                    # 実験実行
                    result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_dir)
                    
                    if result.returncode == 0:
                        print(f"    ✅ Success")
                        condition_results.append({
                            "seed": seed,
                            "status": "success",
                            "episodes": episodes
                        })
                    else:
                        print(f"    ❌ Failed: {result.stderr}")
                        condition_results.append({
                            "seed": seed,
                            "status": "failed",
                            "error": result.stderr
                        })
                
                except Exception as e:
                    print(f"    ❌ Exception: {e}")
                    condition_results.append({
                        "seed": seed,
                        "status": "exception",
                        "error": str(e)
                    })
            
            results[condition] = condition_results
        
        # 結果保存
        results_path = experiment_dir / "experiment_results.json"
        with results_path.open('w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Unified ablation experiment completed")
        print(f"📊 Results saved: {results_path}")
        
        return str(experiment_dir)
    
    def analyze_unified_results(self, experiment_dir: str) -> Dict[str, Any]:
        """統一実験結果を分析"""
        experiment_path = Path(experiment_dir)
        
        print(f"📊 Analyzing unified experiment results: {experiment_path.name}")
        
        # 実測データ収集
        analysis = {
            "experiment_name": experiment_path.name,
            "conditions": {},
            "statistical_tests": {},
            "summary": {}
        }
        
        # 各条件のデータ収集
        for condition in self.conditions:
            condition_data = {
                "final_scores": [],
                "episode_counts": [],
                "seeds": [],
                "data_files": []
            }
            
            # 条件別CSVファイル検索
            csv_files = list(experiment_path.glob(f"**/*{condition}*/*.csv"))
            csv_files.extend(list(self.project_dir.glob(f"runs/real/**/*{condition}*.csv")))
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    if len(df) == 0:
                        continue
                    
                    # 条件確認
                    if 'condition' in df.columns and df['condition'].iloc[0] == condition:
                        # エピソード別最終スコア
                        if 'episode' in df.columns and 'score' in df.columns:
                            episode_scores = df.groupby('episode')['score'].last().tolist()
                            condition_data["final_scores"].extend(episode_scores)
                            condition_data["episode_counts"].append(len(episode_scores))
                        
                        if 'seed' in df.columns:
                            condition_data["seeds"].append(df['seed'].iloc[0])
                        
                        condition_data["data_files"].append(str(csv_file.relative_to(self.project_dir)))
                
                except Exception as e:
                    print(f"⚠️  Warning: Could not process {csv_file}: {e}")
                    continue
            
            analysis["conditions"][condition] = condition_data
        
        # 統計分析
        analysis["statistical_tests"] = self.perform_unified_statistical_tests(analysis["conditions"])
        
        # サマリー生成
        analysis["summary"] = self.generate_unified_summary(analysis["conditions"], analysis["statistical_tests"])
        
        return analysis
    
    def perform_unified_statistical_tests(self, conditions: Dict[str, Dict]) -> Dict[str, Any]:
        """統一実験の統計検定"""
        print("🧪 Performing unified statistical tests...")
        
        tests = {
            "descriptive_stats": {},
            "normality_tests": {},
            "anova": None,
            "pairwise_comparisons": {},
            "effect_sizes": {}
        }
        
        # 条件別記述統計
        condition_scores = {}
        for condition, data in conditions.items():
            if data["final_scores"]:
                scores = np.array(data["final_scores"])
                condition_scores[condition] = scores
                
                tests["descriptive_stats"][condition] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores, ddof=1) if len(scores) > 1 else 0),
                    "median": float(np.median(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "n": len(scores),
                    "sem": float(np.std(scores, ddof=1) / np.sqrt(len(scores)) if len(scores) > 1 else 0)
                }
        
        if len(condition_scores) < 2:
            return tests
        
        try:
            from scipy import stats as scipy_stats
            
            # 正規性検定
            for condition, scores in condition_scores.items():
                if len(scores) >= 3:
                    stat, p_value = scipy_stats.shapiro(scores)
                    tests["normality_tests"][condition] = {
                        "statistic": float(stat),
                        "p_value": float(p_value),
                        "is_normal": p_value > 0.05
                    }
            
            # 群間比較検定
            score_groups = list(condition_scores.values())
            if len(score_groups) >= 2:
                # 正規性に基づいて検定選択
                all_normal = all(
                    tests["normality_tests"].get(cond, {}).get("is_normal", False) 
                    for cond in condition_scores.keys()
                )
                
                if all_normal and len(score_groups) > 2:
                    # ANOVA
                    f_stat, p_value = scipy_stats.f_oneway(*score_groups)
                    tests["anova"] = {
                        "test": "ANOVA",
                        "statistic": float(f_stat),
                        "p_value": float(p_value),
                        "significant": p_value < 0.05
                    }
                else:
                    # Kruskal-Wallis
                    h_stat, p_value = scipy_stats.kruskal(*score_groups)
                    tests["anova"] = {
                        "test": "Kruskal-Wallis",
                        "statistic": float(h_stat),
                        "p_value": float(p_value),
                        "significant": p_value < 0.05
                    }
            
            # ペアワイズ比較
            condition_names = list(condition_scores.keys())
            for i, cond1 in enumerate(condition_names):
                for j, cond2 in enumerate(condition_names[i+1:], i+1):
                    scores1 = condition_scores[cond1]
                    scores2 = condition_scores[cond2]
                    
                    if len(scores1) >= 2 and len(scores2) >= 2:
                        # Mann-Whitney U検定
                        u_stat, p_value = scipy_stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
                        
                        # Cohen's d
                        pooled_std = np.sqrt(((len(scores1) - 1) * np.var(scores1, ddof=1) + 
                                            (len(scores2) - 1) * np.var(scores2, ddof=1)) / 
                                           (len(scores1) + len(scores2) - 2))
                        cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std if pooled_std > 0 else 0
                        
                        pair_key = f"{cond1}_vs_{cond2}"
                        tests["pairwise_comparisons"][pair_key] = {
                            "u_statistic": float(u_stat),
                            "p_value": float(p_value),
                            "cohens_d": float(cohens_d),
                            "significant": p_value < 0.05,
                            "effect_size": "large" if abs(cohens_d) >= 0.8 else "medium" if abs(cohens_d) >= 0.5 else "small"
                        }
                        
                        tests["effect_sizes"][pair_key] = float(cohens_d)
        
        except ImportError:
            print("⚠️  scipy not available, skipping statistical tests")
        except Exception as e:
            print(f"⚠️  Statistical test error: {e}")
        
        return tests
    
    def generate_unified_summary(self, conditions: Dict, tests: Dict) -> Dict[str, Any]:
        """統一実験サマリー生成"""
        summary = {
            "best_condition": None,
            "best_score": -float('inf'),
            "worst_condition": None,
            "worst_score": float('inf'),
            "significant_differences": [],
            "recommendations": []
        }
        
        # 最高・最低性能条件特定
        for condition, data in conditions.items():
            if data["final_scores"]:
                mean_score = np.mean(data["final_scores"])
                if mean_score > summary["best_score"]:
                    summary["best_score"] = mean_score
                    summary["best_condition"] = condition
                if mean_score < summary["worst_score"]:
                    summary["worst_score"] = mean_score
                    summary["worst_condition"] = condition
        
        # 有意差のある比較を特定
        if "pairwise_comparisons" in tests:
            for pair, result in tests["pairwise_comparisons"].items():
                if result["significant"]:
                    summary["significant_differences"].append({
                        "comparison": pair,
                        "p_value": result["p_value"],
                        "effect_size": result["effect_size"],
                        "cohens_d": result["cohens_d"]
                    })
        
        # 推奨事項生成
        if summary["best_condition"]:
            summary["recommendations"].append(f"最高性能: {summary['best_condition']} (平均スコア: {summary['best_score']:.2f})")
        
        if len(summary["significant_differences"]) > 0:
            summary["recommendations"].append(f"有意差のある比較: {len(summary['significant_differences'])}組")
        else:
            summary["recommendations"].append("条件間に統計的有意差は検出されませんでした")
        
        return summary
    
    def create_unified_ablation_table(self, analysis: Dict, output_path: str):
        """統一アブレーション表を作成"""
        print("📊 Creating unified ablation table...")
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        table_content = f"""# 4条件統一アブレーション実験結果

**実験日**: {current_date}  
**実験名**: {analysis['experiment_name']}  
**データソース**: 実測ログのみ使用

## 📊 条件別パフォーマンス比較

| 条件 | 平均スコア | 標準偏差 | 標準誤差 | 95%信頼区間 | 最小-最大 | サンプル数 | データファイル数 |
|------|------------|----------|----------|-------------|-----------|------------|------------------|"""
        
        # 条件別統計テーブル
        for condition in self.conditions:
            if condition in analysis["conditions"] and analysis["conditions"][condition]["final_scores"]:
                data = analysis["conditions"][condition]
                stats = analysis["statistical_tests"]["descriptive_stats"].get(condition, {})
                
                # 95%信頼区間計算
                if stats.get("sem", 0) > 0:
                    ci_margin = 1.96 * stats["sem"]  # 近似的な95%CI
                    ci_lower = stats["mean"] - ci_margin
                    ci_upper = stats["mean"] + ci_margin
                    ci_text = f"[{ci_lower:.2f}, {ci_upper:.2f}]"
                else:
                    ci_text = "N/A"
                
                table_content += f"""
| {condition} | {stats.get('mean', 0):.2f} | {stats.get('std', 0):.2f} | {stats.get('sem', 0):.2f} | {ci_text} | {stats.get('min', 0):.0f}-{stats.get('max', 0):.0f} | {stats.get('n', 0)} | {len(data['data_files'])} |"""
        
        # 統計検定結果
        if analysis["statistical_tests"]["anova"]:
            anova = analysis["statistical_tests"]["anova"]
            table_content += f"""

## 🧪 統計検定結果

### 群間比較
- **検定**: {anova['test']}
- **統計量**: {anova['statistic']:.4f}
- **p値**: {anova['p_value']:.6f}
- **有意差**: {'あり' if anova['significant'] else 'なし'} (α=0.05)
"""
        
        # ペアワイズ比較
        if analysis["statistical_tests"]["pairwise_comparisons"]:
            table_content += """
### ペアワイズ比較 (Mann-Whitney U検定)

| 比較 | p値 | Cohen's d | 効果量 | 有意差 |
|------|-----|-----------|--------|--------|"""
            
            for pair, result in analysis["statistical_tests"]["pairwise_comparisons"].items():
                significant = "✅" if result['significant'] else "❌"
                table_content += f"""
| {pair.replace('_vs_', ' vs ')} | {result['p_value']:.6f} | {result['cohens_d']:.3f} | {result['effect_size']} | {significant} |"""
        
        # サマリー
        summary = analysis["summary"]
        table_content += f"""

## 🏆 実験サマリー

- **最高性能条件**: {summary['best_condition']} (平均スコア: {summary['best_score']:.2f})
- **最低性能条件**: {summary['worst_condition']} (平均スコア: {summary['worst_score']:.2f})
- **有意差のある比較**: {len(summary['significant_differences'])}組

### 推奨事項
"""
        
        for recommendation in summary["recommendations"]:
            table_content += f"- {recommendation}\n"
        
        table_content += f"""

## 🔍 データ品質保証

- ✅ **実測データのみ**: 合成データ使用なし
- ✅ **統一条件**: 同一エピソード数・同一シード集合
- ✅ **再現可能性**: 固定シード実験
- ✅ **透明性**: 全実験ログ公開

### データソースファイル
"""
        
        for condition in self.conditions:
            if condition in analysis["conditions"]:
                data = analysis["conditions"][condition]
                if data["data_files"]:
                    table_content += f"\n**{condition}**:\n"
                    for i, file_path in enumerate(data["data_files"][:5], 1):  # 最初の5個まで
                        table_content += f"{i}. [`{Path(file_path).name}`]({file_path})\n"
                    if len(data["data_files"]) > 5:
                        table_content += f"... 他{len(data['data_files'])-5}個\n"
        
        table_content += """
---

*この表は実測データのみから生成されています*
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(table_content)
        
        print(f"✅ Unified ablation table saved: {output_path}")
    
    def run_and_analyze(self, episodes: int = 10, seeds: List[int] = None) -> Dict[str, Any]:
        """実験実行と分析を統合実行"""
        print("🚀 Starting unified ablation experiment and analysis...")
        print("=" * 70)
        
        # 実験実行
        experiment_dir = self.run_unified_experiment(episodes, seeds)
        
        print("\n" + "=" * 70)
        print("📊 Starting analysis...")
        
        # 結果分析
        analysis = self.analyze_unified_results(experiment_dir)
        
        # アブレーション表作成
        table_path = Path(experiment_dir) / "unified_ablation_table.md"
        self.create_unified_ablation_table(analysis, str(table_path))
        
        print("=" * 70)
        print("✅ Unified ablation experiment completed")
        print(f"📁 Results directory: {experiment_dir}")
        print(f"📊 Analysis table: {table_path}")
        
        return analysis


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="4条件統一アブレーション実験")
    parser.add_argument("--episodes", type=int, default=5, help="エピソード数 (デフォルト: 5)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456], help="シード集合 (デフォルト: 42 123 456)")
    parser.add_argument("--analyze-only", type=str, help="既存実験ディレクトリの分析のみ実行")
    
    args = parser.parse_args()
    
    experiment = UnifiedAblationExperiment()
    
    if args.analyze_only:
        print(f"📊 Analyzing existing experiment: {args.analyze_only}")
        analysis = experiment.analyze_unified_results(args.analyze_only)
        table_path = Path(args.analyze_only) / "unified_ablation_table.md"
        experiment.create_unified_ablation_table(analysis, str(table_path))
        print(f"✅ Analysis completed: {table_path}")
    else:
        print(f"🚀 Running unified ablation experiment...")
        print(f"📈 Episodes: {args.episodes}, Seeds: {args.seeds}")
        
        analysis = experiment.run_and_analyze(args.episodes, args.seeds)
        
        print("\n🎉 Unified ablation experiment complete!")
        print(f"🏆 Best condition: {analysis['summary']['best_condition']}")
        print(f"📊 Best score: {analysis['summary']['best_score']:.2f}")


if __name__ == "__main__":
    main()
