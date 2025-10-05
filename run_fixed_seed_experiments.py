"""
固定シード実験ランナー
4条件（ELM単体、ルール教師、ランダム教師、LLM教師）を同一条件で比較
完全な再現性を保証する固定シード実験
"""
import argparse
import time
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from run_elm_real import run_elm_only_experiment, run_rule_teacher_experiment, run_random_teacher_experiment
from run_elm_llm_real import run_elm_llm_experiment


class FixedSeedExperimentRunner:
    """固定シード実験の管理・実行"""
    
    def __init__(self, base_dir: str = "runs/real"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 固定シード設定（完全な再現性のため）
        self.seeds = [42, 123, 456]
        self.conditions = ["elm_only", "rule_teacher", "random_teacher", "elm_llm"]
        
        # 実験設定
        self.episodes_per_seed = 20  # 各シードあたりのエピソード数
        self.total_episodes_per_condition = len(self.seeds) * self.episodes_per_seed  # 60エピソード
    
    def run_single_condition(self, condition: str, seed: int, episodes: int):
        """単一条件の実験を実行"""
        out_dir = self.base_dir / condition / f"seed_{seed}"
        
        print(f"Starting {condition} experiment with seed={seed}, episodes={episodes}")
        start_time = time.time()
        
        try:
            if condition == "elm_only":
                result = run_elm_only_experiment(episodes, seed, str(out_dir))
            elif condition == "rule_teacher":
                result = run_rule_teacher_experiment(episodes, seed, str(out_dir))
            elif condition == "random_teacher":
                result = run_random_teacher_experiment(episodes, seed, str(out_dir))
            elif condition == "elm_llm":
                # LLM実験はOpenAI APIキーが必要
                if not os.getenv("OPENAI_API_KEY"):
                    print(f"Warning: OPENAI_API_KEY not set, skipping {condition}")
                    return None
                result = run_elm_llm_experiment(episodes, seed, str(out_dir))
            else:
                raise ValueError(f"Unknown condition: {condition}")
            
            elapsed_time = time.time() - start_time
            result["experiment_time"] = elapsed_time
            result["seed"] = seed
            
            print(f"Completed {condition} (seed={seed}) in {elapsed_time:.2f}s: "
                  f"Mean score = {result['mean_score']:.2f}")
            
            return result
            
        except Exception as e:
            print(f"Error in {condition} (seed={seed}): {e}")
            return None
    
    def run_all_conditions_sequential(self, episodes_per_seed: int = None):
        """全条件を順次実行（安全・確実）"""
        if episodes_per_seed is None:
            episodes_per_seed = self.episodes_per_seed
        
        print(f"Starting fixed-seed experiments:")
        print(f"Conditions: {self.conditions}")
        print(f"Seeds: {self.seeds}")
        print(f"Episodes per seed: {episodes_per_seed}")
        print(f"Total episodes per condition: {len(self.seeds) * episodes_per_seed}")
        print("-" * 60)
        
        all_results = {}
        total_start_time = time.time()
        
        for condition in self.conditions:
            condition_results = []
            condition_start_time = time.time()
            
            for seed in self.seeds:
                result = self.run_single_condition(condition, seed, episodes_per_seed)
                if result:
                    condition_results.append(result)
            
            condition_time = time.time() - condition_start_time
            all_results[condition] = {
                "results": condition_results,
                "total_time": condition_time,
                "seeds": self.seeds,
                "episodes_per_seed": episodes_per_seed
            }
            
            # 条件別サマリー
            if condition_results:
                scores = [r["mean_score"] for r in condition_results]
                print(f"\n{condition.upper()} SUMMARY:")
                print(f"  Mean scores: {scores}")
                print(f"  Overall mean: {sum(scores)/len(scores):.2f}")
                print(f"  Std dev: {(sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5:.2f}")
                print(f"  Time: {condition_time:.2f}s")
        
        total_time = time.time() - total_start_time
        
        # 全体サマリー保存
        summary = {
            "experiment_type": "fixed_seed_comparison",
            "timestamp": time.time(),
            "total_time": total_time,
            "conditions": self.conditions,
            "seeds": self.seeds,
            "episodes_per_seed": episodes_per_seed,
            "total_episodes_per_condition": len(self.seeds) * episodes_per_seed,
            "results": all_results
        }
        
        summary_file = self.base_dir / "experiment_summary.json"
        with summary_file.open("w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"ALL EXPERIMENTS COMPLETED in {total_time:.2f}s")
        print(f"Results saved to: {summary_file}")
        print(f"{'='*60}")
        
        return summary
    
    def run_all_conditions_parallel(self, episodes_per_seed: int = None, max_workers: int = 2):
        """全条件を並列実行（高速だがリソース消費大）"""
        if episodes_per_seed is None:
            episodes_per_seed = self.episodes_per_seed
        
        print(f"Starting parallel fixed-seed experiments with {max_workers} workers:")
        print(f"Conditions: {self.conditions}")
        print(f"Seeds: {self.seeds}")
        print(f"Episodes per seed: {episodes_per_seed}")
        
        all_results = {}
        total_start_time = time.time()
        
        # 並列実行のタスクリスト作成
        tasks = []
        for condition in self.conditions:
            for seed in self.seeds:
                tasks.append((condition, seed, episodes_per_seed))
        
        # 並列実行
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self.run_single_condition, condition, seed, episodes): 
                (condition, seed) for condition, seed, episodes in tasks
            }
            
            for future in as_completed(future_to_task):
                condition, seed = future_to_task[future]
                try:
                    result = future.result()
                    if result:
                        if condition not in all_results:
                            all_results[condition] = []
                        all_results[condition].append(result)
                except Exception as e:
                    print(f"Error in {condition} (seed={seed}): {e}")
        
        total_time = time.time() - total_start_time
        
        # 結果整理・保存
        summary = {
            "experiment_type": "fixed_seed_comparison_parallel",
            "timestamp": time.time(),
            "total_time": total_time,
            "max_workers": max_workers,
            "conditions": self.conditions,
            "seeds": self.seeds,
            "episodes_per_seed": episodes_per_seed,
            "results": all_results
        }
        
        summary_file = self.base_dir / "experiment_summary_parallel.json"
        with summary_file.open("w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nPARALLEL EXPERIMENTS COMPLETED in {total_time:.2f}s")
        print(f"Results saved to: {summary_file}")
        
        return summary
    
    def validate_experiment_integrity(self):
        """実験の整合性を検証"""
        print("Validating experiment integrity...")
        
        issues = []
        
        for condition in self.conditions:
            condition_dir = self.base_dir / condition
            if not condition_dir.exists():
                issues.append(f"Missing condition directory: {condition}")
                continue
            
            for seed in self.seeds:
                seed_dir = condition_dir / f"seed_{seed}"
                if not seed_dir.exists():
                    issues.append(f"Missing seed directory: {condition}/seed_{seed}")
                    continue
                
                # ログファイルの存在確認
                steps_files = list(seed_dir.glob("steps_*.csv"))
                if not steps_files:
                    issues.append(f"Missing steps file: {condition}/seed_{seed}")
                
                summary_files = list(seed_dir.glob("summary_*.json"))
                if not summary_files:
                    issues.append(f"Missing summary file: {condition}/seed_{seed}")
        
        if issues:
            print("INTEGRITY ISSUES FOUND:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("✅ All experiments completed successfully!")
            return True


def main():
    parser = argparse.ArgumentParser(description="Run fixed-seed experiments for all conditions")
    parser.add_argument("--episodes", type=int, default=20, 
                       help="Episodes per seed (default: 20)")
    parser.add_argument("--parallel", action="store_true", 
                       help="Run experiments in parallel")
    parser.add_argument("--max_workers", type=int, default=2, 
                       help="Max parallel workers (default: 2)")
    parser.add_argument("--base_dir", type=str, default="runs/real", 
                       help="Base directory for results")
    parser.add_argument("--validate_only", action="store_true", 
                       help="Only validate existing experiments")
    
    args = parser.parse_args()
    
    runner = FixedSeedExperimentRunner(args.base_dir)
    
    if args.validate_only:
        runner.validate_experiment_integrity()
        return
    
    # OpenAI APIキーの確認
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. ELM+LLM experiments will be skipped.")
        print("Set OPENAI_API_KEY environment variable to run LLM experiments.")
    
    # 実験実行
    if args.parallel:
        summary = runner.run_all_conditions_parallel(args.episodes, args.max_workers)
    else:
        summary = runner.run_all_conditions_sequential(args.episodes)
    
    # 整合性検証
    runner.validate_experiment_integrity()
    
    print(f"\nExperiment summary saved to: {runner.base_dir}/experiment_summary.json")


if __name__ == "__main__":
    main()
