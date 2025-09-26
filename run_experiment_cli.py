#!/usr/bin/env python3
"""
拡張CLI実験システム
教師選択、実験設定、分析オプションを統合したコマンドラインインターフェース
"""
import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import json
import time
from datetime import datetime

# プロジェクトモジュールのインポート
from run_fixed_seed_experiments import FixedSeedExperimentRunner
from analyze_real_data import RealDataAnalyzer
from analyze_llm_interactions import LLMInteractionAnalyzer
from update_readme_fixed import ReadmeUpdater


class ExperimentCLI:
    """拡張実験CLIシステム"""
    
    def __init__(self):
        self.available_teachers = ["elm_only", "rule_teacher", "random_teacher", "elm_llm"]
        self.available_seeds = [42, 123, 456, 789, 999]
        self.default_episodes = 10
        
    def create_parser(self) -> argparse.ArgumentParser:
        """CLIパーサーを作成"""
        parser = argparse.ArgumentParser(
            description="Tower Defense ELM+LLM Experiment CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
使用例:
  # 基本的な4条件比較実験
  python run_experiment_cli.py run --teachers all --episodes 20
  
  # 特定教師のみでの実験
  python run_experiment_cli.py run --teachers elm_only rule_teacher --episodes 10
  
  # カスタムシードでの実験
  python run_experiment_cli.py run --teachers elm_llm --seeds 42 123 --episodes 5
  
  # 実験+分析+README更新の完全パイプライン
  python run_experiment_cli.py full --teachers all --episodes 15 --update-readme
  
  # 既存データの分析のみ
  python run_experiment_cli.py analyze runs/real/my_experiment/
  
  # LLMインタラクション分析
  python run_experiment_cli.py llm-analysis runs/real/my_experiment/elm_llm/seed_42/
            """
        )
        
        # サブコマンド
        subparsers = parser.add_subparsers(dest='command', help='実行するコマンド')
        
        # run サブコマンド（実験実行）
        run_parser = subparsers.add_parser('run', help='実験を実行')
        self._add_experiment_args(run_parser)
        
        # full サブコマンド（実験+分析+README更新）
        full_parser = subparsers.add_parser('full', help='完全パイプライン（実験+分析+README更新）')
        self._add_experiment_args(full_parser)
        full_parser.add_argument('--update-readme', action='store_true', 
                               help='実験後にREADMEを自動更新')
        full_parser.add_argument('--skip-analysis', action='store_true',
                               help='統計分析をスキップ')
        
        # analyze サブコマンド（分析のみ）
        analyze_parser = subparsers.add_parser('analyze', help='既存データを分析')
        analyze_parser.add_argument('data_dir', help='分析対象のデータディレクトリ')
        analyze_parser.add_argument('--output', help='分析結果の出力ディレクトリ')
        
        # llm-analysis サブコマンド（LLM分析のみ）
        llm_parser = subparsers.add_parser('llm-analysis', help='LLMインタラクションを分析')
        llm_parser.add_argument('llm_dir', help='LLMログディレクトリ')
        llm_parser.add_argument('--output', help='分析結果の出力ディレクトリ')
        
        # readme サブコマンド（README更新のみ）
        readme_parser = subparsers.add_parser('readme', help='READMEを更新')
        readme_parser.add_argument('results_dir', help='実験結果ディレクトリ')
        readme_parser.add_argument('--project-dir', default='.', help='プロジェクトルートディレクトリ')
        
        # list サブコマンド（利用可能オプション表示）
        list_parser = subparsers.add_parser('list', help='利用可能なオプションを表示')
        list_parser.add_argument('--teachers', action='store_true', help='利用可能な教師を表示')
        list_parser.add_argument('--seeds', action='store_true', help='推奨シードを表示')
        list_parser.add_argument('--experiments', action='store_true', help='過去の実験を表示')
        
        return parser
    
    def _add_experiment_args(self, parser: argparse.ArgumentParser):
        """実験用引数を追加"""
        parser.add_argument('--teachers', nargs='+', 
                          choices=self.available_teachers + ['all'],
                          default=['all'],
                          help='使用する教師（デフォルト: all）')
        
        parser.add_argument('--seeds', nargs='+', type=int,
                          default=self.available_seeds[:3],  # デフォルトは最初の3つ
                          help=f'使用するシード（デフォルト: {self.available_seeds[:3]}）')
        
        parser.add_argument('--episodes', type=int, default=self.default_episodes,
                          help=f'エピソード数（デフォルト: {self.default_episodes}）')
        
        parser.add_argument('--base-dir', default='runs/real',
                          help='結果保存ベースディレクトリ（デフォルト: runs/real）')
        
        parser.add_argument('--experiment-name', 
                          help='実験名（デフォルト: 自動生成）')
        
        parser.add_argument('--parallel', action='store_true',
                          help='並列実行を有効化')
        
        parser.add_argument('--dry-run', action='store_true',
                          help='実際の実行なしで設定を表示')
        
        parser.add_argument('--openai-model', default='gpt-4o-mini',
                          choices=['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo'],
                          help='OpenAI モデル（デフォルト: gpt-4o-mini）')
        
        parser.add_argument('--timeout', type=int, default=3600,
                          help='実験タイムアウト（秒、デフォルト: 3600）')
    
    def validate_args(self, args) -> bool:
        """引数を検証"""
        if hasattr(args, 'teachers') and args.teachers:
            # 'all' の展開
            if 'all' in args.teachers:
                args.teachers = self.available_teachers
            
            # 重複除去
            args.teachers = list(set(args.teachers))
            
            # 無効な教師チェック
            invalid_teachers = set(args.teachers) - set(self.available_teachers)
            if invalid_teachers:
                print(f"❌ 無効な教師: {invalid_teachers}")
                print(f"利用可能な教師: {self.available_teachers}")
                return False
        
        if hasattr(args, 'seeds') and args.seeds:
            # シード範囲チェック
            if any(seed < 0 or seed > 999999 for seed in args.seeds):
                print("❌ シードは0-999999の範囲で指定してください")
                return False
        
        if hasattr(args, 'episodes') and args.episodes:
            if args.episodes < 1 or args.episodes > 1000:
                print("❌ エピソード数は1-1000の範囲で指定してください")
                return False
                # OpenAI APIキーチェック（LLM教師使用時）
        if hasattr(args, 'teachers') and 'elm_llm' in getattr(args, 'teachers', []):
            if not os.getenv('OPENAI_API_KEY'):
                print("⚠️  Warning: OPENAI_API_KEY not set. ELM+LLM will use fallback mode.")
                print("   Set OPENAI_API_KEY environment variable for full LLM functionality.")
        
        return True
    
    def run_experiment(self, args) -> str:
        """実験を実行"""
        # 実験名生成
        if not args.experiment_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            teachers_str = "_".join(args.teachers)
            args.experiment_name = f"{teachers_str}_{timestamp}"
        
        experiment_dir = Path(args.base_dir) / args.experiment_name
        
        print(f"🚀 Starting experiment: {args.experiment_name}")
        print(f"📊 Teachers: {args.teachers}")
        print(f"🎲 Seeds: {args.seeds}")
        print(f"📈 Episodes per seed: {args.episodes}")
        print(f"📁 Output directory: {experiment_dir}")
        print(f"⚡ Parallel execution: {args.parallel}")
        
        if args.dry_run:
            print("\n🔍 DRY RUN - No actual execution")
            return str(experiment_dir)
        
        # 実験実行
        runner = FixedSeedExperimentRunner(
            base_dir=str(experiment_dir),
            conditions=args.teachers,
            seeds=args.seeds
        )
        
        start_time = time.time()
        
        try:
            if args.parallel:
                summary = runner.run_all_conditions_parallel(args.episodes)
            else:
                summary = runner.run_all_conditions_sequential(args.episodes)
            
            # 実験整合性検証
            integrity_ok = runner.validate_experiment_integrity()
            
            total_time = time.time() - start_time
            
            print(f"\n✅ Experiment completed in {total_time:.2f}s")
            print(f"🔍 Data integrity: {'PASSED' if integrity_ok else 'FAILED'}")
            
            return str(experiment_dir)
            
        except KeyboardInterrupt:
            print("\n⚠️  Experiment interrupted by user")
            return str(experiment_dir)
        except Exception as e:
            print(f"\n❌ Experiment failed: {e}")
            raise\n    \n    def run_analysis(self, data_dir: str, output_dir: str = None) -> bool:\n        \"\"\"データ分析を実行\"\"\"\n        print(f\"📊 Analyzing data in: {data_dir}\")\n        \n        analyzer = RealDataAnalyzer(data_dir)\n        success = analyzer.run_complete_analysis(output_dir)\n        \n        if success:\n            print(\"✅ Analysis completed successfully\")\n        else:\n            print(\"❌ Analysis failed\")\n        \n        return success\n    \n    def run_llm_analysis(self, llm_dir: str, output_dir: str = None) -> bool:\n        \"\"\"LLMインタラクション分析を実行\"\"\"\n        print(f\"🤖 Analyzing LLM interactions in: {llm_dir}\")\n        \n        try:\n            analyzer = LLMInteractionAnalyzer(llm_dir)\n            \n            # 分析実行\n            analyzer.generate_interaction_report()\n            analyzer.create_visualization(output_dir)\n            \n            print(\"✅ LLM analysis completed successfully\")\n            return True\n            \n        except Exception as e:\n            print(f\"❌ LLM analysis failed: {e}\")\n            return False\n    \n    def update_readme(self, results_dir: str, project_dir: str = '.') -> bool:\n        \"\"\"READMEを更新\"\"\"\n        print(f\"📝 Updating README with results from: {results_dir}\")\n        \n        updater = ReadmeUpdater(project_dir)\n        success = updater.update_readme(results_dir)\n        \n        if success:\n            print(\"✅ README updated successfully\")\n        else:\n            print(\"❌ README update failed\")\n        \n        return success\n    \n    def list_options(self, args):\n        \"\"\"利用可能なオプションを表示\"\"\"\n        if args.teachers:\n            print(\"📚 利用可能な教師:\")\n            for teacher in self.available_teachers:\n                print(f\"  - {teacher}\")\n        \n        if args.seeds:\n            print(\"🎲 推奨シード:\")\n            for seed in self.available_seeds:\n                print(f\"  - {seed}\")\n        \n        if args.experiments:\n            print(\"🗂️  過去の実験:\")\n            runs_dir = Path(\"runs/real\")\n            if runs_dir.exists():\n                experiments = [d.name for d in runs_dir.iterdir() if d.is_dir()]\n                for exp in sorted(experiments):\n                    print(f\"  - {exp}\")\n            else:\n                print(\"  （実験履歴なし）\")\n    \n    def run_full_pipeline(self, args) -> bool:\n        \"\"\"完全パイプラインを実行\"\"\"\n        print(\"🔄 Running full experiment pipeline...\")\n        \n        # 1. 実験実行\n        experiment_dir = self.run_experiment(args)\n        \n        if args.dry_run:\n            return True\n        \n        # 2. 分析実行（スキップされない場合）\n        if not args.skip_analysis:\n            analysis_success = self.run_analysis(experiment_dir)\n            if not analysis_success:\n                print(\"⚠️  Analysis failed, but continuing...\")\n        \n        # 3. README更新（指定された場合）\n        if args.update_readme:\n            readme_success = self.update_readme(experiment_dir)\n            if not readme_success:\n                print(\"⚠️  README update failed, but continuing...\")\n        \n        print(\"\\n🎉 Full pipeline completed!\")\n        print(f\"📁 Results: {experiment_dir}\")\n        \n        return True\n    \n    def main(self):\n        \"\"\"メイン実行関数\"\"\"\n        parser = self.create_parser()\n        \n        if len(sys.argv) == 1:\n            parser.print_help()\n            return\n        \n        args = parser.parse_args()\n        \n        # 引数検証\n        if not self.validate_args(args):\n            sys.exit(1)\n        \n        try:\n            # コマンド実行\n            if args.command == 'run':\n                self.run_experiment(args)\n            \n            elif args.command == 'full':\n                self.run_full_pipeline(args)\n            \n            elif args.command == 'analyze':\n                output_dir = args.output or args.data_dir\n                success = self.run_analysis(args.data_dir, output_dir)\n                if not success:\n                    sys.exit(1)\n            \n            elif args.command == 'llm-analysis':\n                output_dir = args.output or args.llm_dir\n                success = self.run_llm_analysis(args.llm_dir, output_dir)\n                if not success:\n                    sys.exit(1)\n            \n            elif args.command == 'readme':\n                success = self.update_readme(args.results_dir, args.project_dir)\n                if not success:\n                    sys.exit(1)\n            \n            elif args.command == 'list':\n                self.list_options(args)\n            \n            else:\n                parser.print_help()\n        \n        except KeyboardInterrupt:\n            print(\"\\n⚠️  Operation interrupted by user\")\n            sys.exit(1)\n        except Exception as e:\n            print(f\"\\n❌ Error: {e}\")\n            sys.exit(1)\n\n\nif __name__ == \"__main__\":\n    cli = ExperimentCLI()\n    cli.main()
