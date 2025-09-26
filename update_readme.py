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
    
    def load_analysis_results(self, analysis_dir: str) -> Dict[str, Any]:
        """分析結果を読み込み"""
        analysis_path = Path(analysis_dir)
        analysis_data = {}
        
        # 分析レポートを探す
        report_files = list(analysis_path.glob("**/real_data_analysis_report.md"))
        if report_files:
            latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
            analysis_data["report_path"] = str(latest_report)
            print(f"   📊 Found analysis report: {latest_report}")
        
        # 可視化ファイルを探す
        viz_files = list(analysis_path.glob("**/real_data_analysis.png"))
        if viz_files:
            latest_viz = max(viz_files, key=lambda x: x.stat().st_mtime)
            analysis_data["visualization_path"] = str(latest_viz)
            print(f"   📈 Found visualization: {latest_viz}")
        
        # LLMインタラクション分析を探す
        llm_files = list(analysis_path.glob("**/llm_interaction_analysis.md"))
        if llm_files:
            latest_llm = max(llm_files, key=lambda x: x.stat().st_mtime)
            analysis_data["llm_analysis_path"] = str(latest_llm)
            print(f"   🤖 Found LLM analysis: {latest_llm}")
        
        return analysis_data
    
    def extract_performance_summary(self) -> Dict[str, Any]:
        """実験結果からパフォーマンスサマリーを抽出"""
        if not self.results_data or "results" not in self.results_data:
            return {}\n        \n        summary = {\n            "experiment_date": datetime.now().strftime("%Y-%m-%d"),\n            "total_conditions": len(self.results_data["results"]),\n            "seeds_used": self.results_data.get("seeds", []),\n            "episodes_per_condition": self.results_data.get("total_episodes_per_condition", 0),\n            "conditions": {}\n        }\n        \n        # 条件別結果の抽出\n        for condition, data in self.results_data["results"].items():\n            if "results" in data and data["results"]:\n                scores = [r["mean_score"] for r in data["results"]]\n                \n                condition_summary = {\n                    "mean_score": sum(scores) / len(scores),\n                    "std_score": (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5,\n                    "min_score": min(scores),\n                    "max_score": max(scores),\n                    "sample_size": len(scores),\n                    "total_time": data.get("total_time", 0)\n                }\n                \n                summary["conditions"][condition] = condition_summary\n        \n        # 最高性能条件の特定\n        if summary["conditions"]:\n            best_condition = max(summary["conditions"].items(), \n                               key=lambda x: x[1]["mean_score"])\n            summary["best_condition"] = {\n                "name": best_condition[0],\n                "score": best_condition[1]["mean_score"]\n            }\n        \n        return summary\n    \n    def generate_results_section(self, performance_summary: Dict[str, Any]) -> str:\n        """実験結果セクションを生成"""\n        if not performance_summary:\n            return "## 実験結果\\n\\n*実験結果はまだ利用できません。*\\n"\n        \n        section = f"""## 実験結果\n\n### 最新実験 ({performance_summary['experiment_date']})\n\n**実験設定:**\n- 条件数: {performance_summary['total_conditions']}\n- 使用シード: {performance_summary['seeds_used']}\n- エピソード数/条件: {performance_summary['episodes_per_condition']}\n- データ品質: ✅ 実測のみ（合成データなし）\n\n### パフォーマンス比較\n\n| 条件 | 平均スコア | 標準偏差 | 最小-最大 | サンプル数 |\n|------|------------|----------|-----------|------------|\n"""\n        \n        # 条件別結果テーブル\n        for condition, stats in performance_summary["conditions"].items():\n            section += f"| {condition} | {stats['mean_score']:.2f} | {stats['std_score']:.2f} | {stats['min_score']:.0f}-{stats['max_score']:.0f} | {stats['sample_size']} |\\n"\n        \n        # 最高性能の強調\n        if "best_condition" in performance_summary:\n            best = performance_summary["best_condition"]\n            section += f\"\\n**🏆 最高性能**: {best['name']} (平均スコア: {best['score']:.2f})\\n\"\n        \n        section += \"\\n### データ品質保証\\n\\n\"\n        section += \"- ✅ **実測データのみ**: 合成データ生成は一切なし\\n\"\n        section += \"- ✅ **再現可能性**: 固定シード実験\\n\"\n        section += \"- ✅ **透明性**: 全実験ログ公開\\n\"\n        section += \"- ✅ **統計的妥当性**: 適切な検定手法使用\\n\\n\"\n        \n        return section\n    \n    def generate_technical_details(self) -> str:\n        \"\"\"技術詳細セクションを生成\"\"\"\n        return \"\"\"## 技術詳細\n\n### システム構成\n\n- **ELM (Extreme Learning Machine)**: 高速学習アルゴリズム\n- **LLM Teacher**: OpenAI GPT-4o-mini による戦略指導\n- **環境**: Tower Defense シミュレーター\n- **ログシステム**: CSV + JSON + JSONL 形式\n\n### 実験条件\n\n1. **ELM単体**: ELMのみでの学習\n2. **ルール教師**: 事前定義ルールによる指導\n3. **ランダム教師**: ランダムな行動指導\n4. **ELM+LLM**: LLMによる戦略的指導\n\n### データ処理パイプライン\n\n```\n実験実行 → ログ記録 → データ検証 → 統計分析 → レポート生成\n```\n\n- **合成データ検出**: 自動的に排除\n- **設定ハッシュ**: 実験条件の厳密管理\n- **統計検定**: scipy.stats による科学的分析\n\n\"\"\"\n    \n    def generate_usage_section(self) -> str:\n        \"\"\"使用方法セクションを生成\"\"\"\n        return \"\"\"## 使用方法\n\n### 基本実験の実行\n\n```bash\n# 4条件比較実験（推奨）\npython run_fixed_seed_experiments.py --episodes 20\n\n# 単一条件実験\npython run_elm_real.py --condition elm_only --episodes 10 --seed 42\n\n# LLM実験（OpenAI APIキー必要）\nexport OPENAI_API_KEY=\"your-api-key\"\npython run_elm_llm_real.py --episodes 10 --seed 42\n```\n\n### 分析とレポート生成\n\n```bash\n# 実測データ分析\npython analyze_real_data.py runs/real/experiment_name/\n\n# LLMインタラクション分析\npython analyze_llm_interactions.py runs/real/experiment_name/elm_llm/seed_42/\n\n# 完全実験（実行+分析）\npython run_complete_experiment.py --episodes 20\n```\n\n### 結果の確認\n\n- **実験ログ**: `runs/real/` ディレクトリ\n- **分析レポート**: `*_analysis_report.md`\n- **可視化**: `*.png` ファイル\n- **統計結果**: JSON形式のサマリー\n\n\"\"\"\n    \n    def create_readme_template(self) -> str:\n        \"\"\"READMEテンプレートを作成\"\"\"\n        return \"\"\"# Tower Defense ELM+LLM Research Project\n\n**科学的厳密性を重視したタワーディフェンス学習システム**\n\n[![Data Quality](https://img.shields.io/badge/Data-Real%20Only-green)]()\n[![Reproducibility](https://img.shields.io/badge/Reproducibility-Fixed%20Seeds-blue)]()\n[![Transparency](https://img.shields.io/badge/Transparency-Full%20Logs-orange)]()\n\n## 概要\n\nこのプロジェクトは、ELM（Extreme Learning Machine）とLLM（Large Language Model）を組み合わせたタワーディフェンス学習システムです。**実測データのみ**を使用し、合成データを一切使用しない科学的に厳密な実験を実施しています。\n\n### 主要特徴\n\n- 🔬 **科学的厳密性**: 実測データのみ、合成データ完全排除\n- 🔄 **完全再現可能**: 固定シード、設定ハッシュ管理\n- 📊 **統計的妥当性**: 適切な検定手法による分析\n- 🤖 **LLM統合**: OpenAI GPT-4o-mini による戦略指導\n- 📝 **透明性**: 全実験ログの公開・検証可能\n\n{RESULTS_SECTION}\n\n{TECHNICAL_DETAILS}\n\n{USAGE_SECTION}\n\n## ファイル構成\n\n```\n├── run_fixed_seed_experiments.py  # 4条件比較実験\n├── run_elm_real.py                # ELM単体実験\n├── run_elm_llm_real.py            # ELM+LLM実験\n├── analyze_real_data.py           # 実測データ分析\n├── logger.py                      # 実測専用ログシステム\n├── src/\n│   ├── tower_defense_environment.py\n│   ├── elm_tower_defense_agent.py\n│   └── llm_teacher.py\n└── runs/real/                     # 実測実験ログ\n```\n\n## 研究の信頼性\n\n### データ品質保証\n\n- ✅ **実測データのみ**: `validate_no_synthetic_data()` による検証\n- ✅ **固定シード実験**: 完全な再現可能性\n- ✅ **設定ハッシュ**: 実験条件の厳密な追跡\n- ✅ **ログ整合性**: 自動検証システム\n\n### 統計分析\n\n- 📊 **記述統計**: 平均、標準偏差、範囲、中央値\n- 🧪 **検定**: Shapiro-Wilk、Levene、ANOVA/Kruskal-Wallis\n- 📈 **効果量**: Cohen's d による実用的有意性\n- 🎯 **多重比較**: Mann-Whitney U検定\n\n## 貢献\n\n1. Fork the repository\n2. Create your feature branch (`git checkout -b feature/AmazingFeature`)\n3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)\n4. Push to the branch (`git push origin feature/AmazingFeature`)\n5. Open a Pull Request\n\n**注意**: 全ての貢献は実測データのみを使用し、合成データ生成を含まないことを確認してください。\n\n## ライセンス\n\nMIT License - 詳細は [LICENSE](LICENSE) ファイルを参照\n\n## 引用\n\n```bibtex\n@software{tower_defense_elm_llm,\n  title={Tower Defense ELM+LLM Research Project},\n  author={Research Team},\n  year={2025},\n  url={https://github.com/your-repo/tower-defense-llm}\n}\n```\n\n---\n\n*このプロジェクトは実測データのみを使用し、科学的厳密性を最優先に開発されています。*\n\"\"\"\n    \n    def update_readme(self, results_dir: str, analysis_dir: str = None) -> bool:\n        \"\"\"READMEを実測結果で更新\"\"\"\n        print(\"📝 Updating README with real experimental results...\")\n        \n        # 実験結果読み込み\n        if not self.load_experiment_results(results_dir):\n            return False\n        \n        # 分析結果読み込み（オプション）\n        analysis_data = {}\n        if analysis_dir:\n            analysis_data = self.load_analysis_results(analysis_dir)\n        \n        # パフォーマンスサマリー抽出\n        performance_summary = self.extract_performance_summary()\n        \n        # セクション生成\n        results_section = self.generate_results_section(performance_summary)\n        technical_details = self.generate_technical_details()\n        usage_section = self.generate_usage_section()\n        \n        # READMEテンプレート作成\n        readme_content = self.create_readme_template()\n        \n        # プレースホルダー置換\n        readme_content = readme_content.replace(\"{RESULTS_SECTION}\", results_section)\n        readme_content = readme_content.replace(\"{TECHNICAL_DETAILS}\", technical_details)\n        readme_content = readme_content.replace(\"{USAGE_SECTION}\", usage_section)\n        \n        # 分析結果へのリンク追加\n        if analysis_data:\n            links_section = \"\\n## 詳細分析\\n\\n\"\n            if \"report_path\" in analysis_data:\n                links_section += f\"- 📊 [統計分析レポート]({analysis_data['report_path']})\\n\"\n            if \"visualization_path\" in analysis_data:\n                links_section += f\"- 📈 [結果可視化]({analysis_data['visualization_path']})\\n\"\n            if \"llm_analysis_path\" in analysis_data:\n                links_section += f\"- 🤖 [LLMインタラクション分析]({analysis_data['llm_analysis_path']})\\n\"\n            \n            readme_content = readme_content.replace(\"## ファイル構成\", links_section + \"\\n## ファイル構成\")\n        \n        # README保存\n        try:\n            with self.readme_path.open('w', encoding='utf-8') as f:\n                f.write(readme_content)\n            \n            print(f\"   ✅ README updated: {self.readme_path}\")\n            print(f\"   📊 Included {len(performance_summary.get('conditions', {}))} conditions\")\n            \n            if \"best_condition\" in performance_summary:\n                best = performance_summary[\"best_condition\"]\n                print(f\"   🏆 Best performance: {best['name']} ({best['score']:.2f})\")\n            \n            return True\n            \n        except Exception as e:\n            print(f\"   ❌ Failed to write README: {e}\")\n            return False\n    \n    def backup_existing_readme(self) -> bool:\n        \"\"\"既存のREADMEをバックアップ\"\"\"\n        if self.readme_path.exists():\n            backup_path = self.readme_path.with_suffix('.md.backup')\n            try:\n                backup_path.write_text(self.readme_path.read_text(encoding='utf-8'), encoding='utf-8')\n                print(f\"   💾 Existing README backed up to: {backup_path}\")\n                return True\n            except Exception as e:\n                print(f\"   ⚠️  Failed to backup README: {e}\")\n                return False\n        return True\n\n\ndef main():\n    \"\"\"メイン関数\"\"\"\n    parser = argparse.ArgumentParser(description=\"Update README with real experimental results\")\n    parser.add_argument(\"results_dir\", help=\"Directory containing experiment results\")\n    parser.add_argument(\"--analysis_dir\", help=\"Directory containing analysis results\")\n    parser.add_argument(\"--project_dir\", default=\".\", help=\"Project root directory\")\n    parser.add_argument(\"--backup\", action=\"store_true\", help=\"Backup existing README\")\n    \n    args = parser.parse_args()\n    \n    # README更新実行\n    updater = ReadmeUpdater(args.project_dir)\n    \n    if args.backup:\n        updater.backup_existing_readme()\n    \n    success = updater.update_readme(args.results_dir, args.analysis_dir)\n    \n    if success:\n        print(\"\\n✅ README successfully updated with real experimental results!\")\n    else:\n        print(\"\\n❌ Failed to update README\")\n        exit(1)\n\n\nif __name__ == \"__main__\":\n    main()
