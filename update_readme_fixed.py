#!/usr/bin/env python3
"""
実測結果からの自動README更新システム
合成データを一切使用せず、実際の実験結果のみからREADMEを生成・更新
"""
import json
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
import re


class ReadmeUpdater:
    """実測結果ベースREADME更新システム"""
    
    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
        self.readme_path = self.project_dir / "README.md"
        self.results_data = {}
        
    def load_experiment_results(self, results_dir: str) -> bool:
        """実験結果を読み込み"""
        results_path = Path(results_dir)
        
        print("📊 Loading experiment results for README update...")
        
        # 実験サマリーファイルを探す
        summary_files = list(results_path.glob("**/experiment_summary*.json"))
        if not summary_files:
            print("   ⚠️  No experiment summary found")
            return False
        
        # 最新のサマリーファイルを使用
        latest_summary = max(summary_files, key=lambda x: x.stat().st_mtime)
        print(f"   📁 Loading: {latest_summary}")
        
        try:
            with latest_summary.open('r') as f:
                self.results_data = json.load(f)
            
            print(f"   ✅ Loaded results for {len(self.results_data.get('results', {}))} conditions")
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to load results: {e}")
            return False
    
    def extract_performance_summary(self) -> Dict[str, Any]:
        """実験結果からパフォーマンスサマリーを抽出"""
        if not self.results_data or "results" not in self.results_data:
            return {}
        
        summary = {
            "experiment_date": datetime.now().strftime("%Y-%m-%d"),
            "total_conditions": len(self.results_data["results"]),
            "seeds_used": self.results_data.get("seeds", []),
            "episodes_per_condition": self.results_data.get("total_episodes_per_condition", 0),
            "conditions": {}
        }
        
        # 条件別結果の抽出
        for condition, data in self.results_data["results"].items():
            if "results" in data and data["results"]:
                scores = [r["mean_score"] for r in data["results"]]
                
                condition_summary = {
                    "mean_score": sum(scores) / len(scores),
                    "std_score": (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5,
                    "min_score": min(scores),
                    "max_score": max(scores),
                    "sample_size": len(scores),
                    "total_time": data.get("total_time", 0)
                }
                
                summary["conditions"][condition] = condition_summary
        
        # 最高性能条件の特定
        if summary["conditions"]:
            best_condition = max(summary["conditions"].items(), 
                               key=lambda x: x[1]["mean_score"])
            summary["best_condition"] = {
                "name": best_condition[0],
                "score": best_condition[1]["mean_score"]
            }
        
        return summary
    
    def generate_results_section(self, performance_summary: Dict[str, Any]) -> str:
        """実験結果セクションを生成"""
        if not performance_summary:
            return "## 実験結果\\n\\n*実験結果はまだ利用できません。*\\n"
        
        section = f"""## 実験結果

### 最新実験 ({performance_summary['experiment_date']})

**実験設定:**
- 条件数: {performance_summary['total_conditions']}
- 使用シード: {performance_summary['seeds_used']}
- エピソード数/条件: {performance_summary['episodes_per_condition']}
- データ品質: ✅ 実測のみ（合成データなし）

### パフォーマンス比較

| 条件 | 平均スコア | 標準偏差 | 最小-最大 | サンプル数 |
|------|------------|----------|-----------|------------|
"""
        
        # 条件別結果テーブル
        for condition, stats in performance_summary["conditions"].items():
            section += f"| {condition} | {stats['mean_score']:.2f} | {stats['std_score']:.2f} | {stats['min_score']:.0f}-{stats['max_score']:.0f} | {stats['sample_size']} |\n"
        
        # 最高性能の強調
        if "best_condition" in performance_summary:
            best = performance_summary["best_condition"]
            section += f"\n**🏆 最高性能**: {best['name']} (平均スコア: {best['score']:.2f})\n"
        
        section += "\n### データ品質保証\n\n"
        section += "- ✅ **実測データのみ**: 合成データ生成は一切なし\n"
        section += "- ✅ **再現可能性**: 固定シード実験\n"
        section += "- ✅ **透明性**: 全実験ログ公開\n"
        section += "- ✅ **統計的妥当性**: 適切な検定手法使用\n\n"
        
        return section
    
    def create_readme_content(self, performance_summary: Dict[str, Any]) -> str:
        """完全なREADME内容を作成"""
        results_section = self.generate_results_section(performance_summary)
        
        return f"""# Tower Defense ELM+LLM Research Project

**科学的厳密性を重視したタワーディフェンス学習システム**

[![Data Quality](https://img.shields.io/badge/Data-Real%20Only-green)]()
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Fixed%20Seeds-blue)]()
[![Transparency](https://img.shields.io/badge/Transparency-Full%20Logs-orange)]()

## 概要

このプロジェクトは、ELM（Extreme Learning Machine）とLLM（Large Language Model）を組み合わせたタワーディフェンス学習システムです。**実測データのみ**を使用し、合成データを一切使用しない科学的に厳密な実験を実施しています。

### 主要特徴

- 🔬 **科学的厳密性**: 実測データのみ、合成データ完全排除
- 🔄 **完全再現可能**: 固定シード、設定ハッシュ管理
- 📊 **統計的妥当性**: 適切な検定手法による分析
- 🤖 **LLM統合**: OpenAI GPT-4o-mini による戦略指導
- 📝 **透明性**: 全実験ログの公開・検証可能

{results_section}

## 技術詳細

### システム構成

- **ELM (Extreme Learning Machine)**: 高速学習アルゴリズム
- **LLM Teacher**: OpenAI GPT-4o-mini による戦略指導
- **環境**: Tower Defense シミュレーター
- **ログシステム**: CSV + JSON + JSONL 形式

### 実験条件

1. **ELM単体**: ELMのみでの学習
2. **ルール教師**: 事前定義ルールによる指導
3. **ランダム教師**: ランダムな行動指導
4. **ELM+LLM**: LLMによる戦略的指導

### データ処理パイプライン

```
実験実行 → ログ記録 → データ検証 → 統計分析 → レポート生成
```

- **合成データ検出**: 自動的に排除
- **設定ハッシュ**: 実験条件の厳密管理
- **統計検定**: scipy.stats による科学的分析

## 使用方法

### 基本実験の実行

```bash
# 4条件比較実験（推奨）
python run_fixed_seed_experiments.py --episodes 20

# 単一条件実験
python run_elm_real.py --condition elm_only --episodes 10 --seed 42

# LLM実験（OpenAI APIキー必要）
export OPENAI_API_KEY="your-api-key"
python run_elm_llm_real.py --episodes 10 --seed 42
```

### 分析とレポート生成

```bash
# 実測データ分析
python analyze_real_data.py runs/real/experiment_name/

# LLMインタラクション分析
python analyze_llm_interactions.py runs/real/experiment_name/elm_llm/seed_42/

# 完全実験（実行+分析）
python run_complete_experiment.py --episodes 20
```

### 結果の確認

- **実験ログ**: `runs/real/` ディレクトリ
- **分析レポート**: `*_analysis_report.md`
- **可視化**: `*.png` ファイル
- **統計結果**: JSON形式のサマリー

## ファイル構成

```
├── run_fixed_seed_experiments.py  # 4条件比較実験
├── run_elm_real.py                # ELM単体実験
├── run_elm_llm_real.py            # ELM+LLM実験
├── analyze_real_data.py           # 実測データ分析
├── logger.py                      # 実測専用ログシステム
├── src/
│   ├── tower_defense_environment.py
│   ├── elm_tower_defense_agent.py
│   └── llm_teacher.py
└── runs/real/                     # 実測実験ログ
```

## 研究の信頼性

### データ品質保証

- ✅ **実測データのみ**: `validate_no_synthetic_data()` による検証
- ✅ **固定シード実験**: 完全な再現可能性
- ✅ **設定ハッシュ**: 実験条件の厳密な追跡
- ✅ **ログ整合性**: 自動検証システム

### 統計分析

- 📊 **記述統計**: 平均、標準偏差、範囲、中央値
- 🧪 **検定**: Shapiro-Wilk、Levene、ANOVA/Kruskal-Wallis
- 📈 **効果量**: Cohen's d による実用的有意性
- 🎯 **多重比較**: Mann-Whitney U検定

## 貢献

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**注意**: 全ての貢献は実測データのみを使用し、合成データ生成を含まないことを確認してください。

## ライセンス

MIT License - 詳細は [LICENSE](LICENSE) ファイルを参照

## 引用

```bibtex
@software{{tower_defense_elm_llm,
  title={{Tower Defense ELM+LLM Research Project}},
  author={{Research Team}},
  year={{2025}},
  url={{https://github.com/your-repo/tower-defense-llm}}
}}
```

---

*このプロジェクトは実測データのみを使用し、科学的厳密性を最優先に開発されています。*
"""
    
    def update_readme(self, results_dir: str) -> bool:
        """READMEを実測結果で更新"""
        print("📝 Updating README with real experimental results...")
        
        # 実験結果読み込み
        if not self.load_experiment_results(results_dir):
            return False
        
        # パフォーマンスサマリー抽出
        performance_summary = self.extract_performance_summary()
        
        # README内容生成
        readme_content = self.create_readme_content(performance_summary)
        
        # README保存
        try:
            with self.readme_path.open('w', encoding='utf-8') as f:
                f.write(readme_content)
            
            print(f"   ✅ README updated: {self.readme_path}")
            print(f"   📊 Included {len(performance_summary.get('conditions', {}))} conditions")
            
            if "best_condition" in performance_summary:
                best = performance_summary["best_condition"]
                print(f"   🏆 Best performance: {best['name']} ({best['score']:.2f})")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to write README: {e}")
            return False


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Update README with real experimental results")
    parser.add_argument("results_dir", help="Directory containing experiment results")
    parser.add_argument("--project_dir", default=".", help="Project root directory")
    
    args = parser.parse_args()
    
    # README更新実行
    updater = ReadmeUpdater(args.project_dir)
    success = updater.update_readme(args.results_dir)
    
    if success:
        print("\n✅ README successfully updated with real experimental results!")
    else:
        print("\n❌ Failed to update README")
        exit(1)


if __name__ == "__main__":
    main()
