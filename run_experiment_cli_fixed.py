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
        
        # analyze サブコマンド（分析のみ）
        analyze_parser = subparsers.add_parser('analyze', help='既存データを分析')
        analyze_parser.add_argument('data_dir', help='分析対象のデータディレクトリ')
        analyze_parser.add_argument('--output', help='分析結果の出力ディレクトリ')
        
        # list サブコマンド（利用可能オプション表示）
        list_parser = subparsers.add_parser('list', help='利用可能なオプションを表示')
        list_parser.add_argument('--teachers', action='store_true', help='利用可能な教師を表示')
        list_parser.add_argument('--seeds', action='store_true', help='推奨シードを表示')
        
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
        
        parser.add_argument('--dry-run', action='store_true',
                          help='実際の実行なしで設定を表示')
    
    def validate_args(self, args) -> bool:
        """引数を検証"""
        if hasattr(args, 'teachers') and args.teachers and isinstance(args.teachers, list):
            # 'all' の展開
            if 'all' in args.teachers:
                args.teachers = self.available_teachers
            
            # 重複除去（順序保持）
            args.teachers = list(dict.fromkeys(args.teachers))
            
            # 無効な教師チェック
            invalid_teachers = set(args.teachers) - set(self.available_teachers)
            if invalid_teachers:
                print(f"❌ 無効な教師: {invalid_teachers}")
                print(f"利用可能な教師: {self.available_teachers}")
                return False
        
        if hasattr(args, 'seeds') and args.seeds and isinstance(args.seeds, list):
            # シード範囲チェック
            if any(seed < 0 or seed > 999999 for seed in args.seeds):
                print("❌ シードは0-999999の範囲で指定してください")
                return False
        
        if hasattr(args, 'episodes') and args.episodes:
            if args.episodes < 1 or args.episodes > 1000:
                print("❌ エピソード数は1-1000の範囲で指定してください")
                return False
        
        # OpenAI APIキーチェック（LLM教師使用時）
        teachers_list = getattr(args, 'teachers', [])
        if hasattr(args, 'teachers') and isinstance(teachers_list, list) and 'elm_llm' in teachers_list:
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
        
        if args.dry_run:
            print("\n🔍 DRY RUN - No actual execution")
            return str(experiment_dir)
        
        # 実験実行
        runner = FixedSeedExperimentRunner(base_dir=str(experiment_dir))
        # 設定を上書き
        runner.conditions = args.teachers
        runner.seeds = args.seeds
        
        start_time = time.time()
        
        try:
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
            raise
    
    def run_analysis(self, data_dir: str, output_dir: str = None) -> bool:
        """データ分析を実行"""
        print(f"📊 Analyzing data in: {data_dir}")
        
        analyzer = RealDataAnalyzer(data_dir)
        success = analyzer.run_complete_analysis(output_dir)
        
        if success:
            print("✅ Analysis completed successfully")
        else:
            print("❌ Analysis failed")
        
        return success
    
    def update_readme(self, results_dir: str, project_dir: str = '.') -> bool:
        """READMEを更新"""
        print(f"📝 Updating README with results from: {results_dir}")
        
        updater = ReadmeUpdater(project_dir)
        success = updater.update_readme(results_dir)
        
        if success:
            print("✅ README updated successfully")
        else:
            print("❌ README update failed")
        
        return success
    
    def list_options(self, args):
        """利用可能なオプションを表示"""
        if args.teachers:
            print("📚 利用可能な教師:")
            for teacher in self.available_teachers:
                print(f"  - {teacher}")
        
        if args.seeds:
            print("🎲 推奨シード:")
            for seed in self.available_seeds:
                print(f"  - {seed}")
    
    def run_full_pipeline(self, args) -> bool:
        """完全パイプラインを実行"""
        print("🔄 Running full experiment pipeline...")
        
        # 1. 実験実行
        experiment_dir = self.run_experiment(args)
        
        if args.dry_run:
            return True
        
        # 2. 分析実行
        analysis_success = self.run_analysis(experiment_dir)
        if not analysis_success:
            print("⚠️  Analysis failed, but continuing...")
        
        # 3. README更新（指定された場合）
        if args.update_readme:
            readme_success = self.update_readme(experiment_dir)
            if not readme_success:
                print("⚠️  README update failed, but continuing...")
        
        print(f"\n🎉 Full pipeline completed!")
        print(f"📁 Results: {experiment_dir}")
        
        return True
    
    def main(self):
        """メイン実行関数"""
        parser = self.create_parser()
        
        if len(sys.argv) == 1:
            parser.print_help()
            return
        
        args = parser.parse_args()
        
        # 引数検証
        if not self.validate_args(args):
            sys.exit(1)
        
        try:
            # コマンド実行
            if args.command == 'run':
                self.run_experiment(args)
            
            elif args.command == 'full':
                self.run_full_pipeline(args)
            
            elif args.command == 'analyze':
                output_dir = args.output or args.data_dir
                success = self.run_analysis(args.data_dir, output_dir)
                if not success:
                    sys.exit(1)
            
            elif args.command == 'list':
                self.list_options(args)
            
            else:
                parser.print_help()
        
        except KeyboardInterrupt:
            print("\n⚠️  Operation interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    cli = ExperimentCLI()
    cli.main()
