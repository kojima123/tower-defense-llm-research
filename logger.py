"""
実測専用ロガーシステム
合成データを一切使用せず、実際の実験ログのみを記録する
"""
from pathlib import Path
import csv
import time
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class StepLog:
    """1ステップごとの詳細ログ"""
    episode: int
    step: int
    timestamp: float
    seed: int
    condition: str  # elm_only, elm_llm, rule_teacher, random_teacher
    state_hash: str
    action: str
    reward: float
    score: int
    towers: int
    enemies_killed: int
    health: int
    llm_used: int  # 0 or 1
    llm_prompt_id: Optional[str] = None
    llm_decision: Optional[str] = None
    env_version: str = "1.0"
    agent_version: str = "1.0"
    prompt_version: str = "1.0"


@dataclass
class ExperimentConfig:
    """実験設定のハッシュ化用"""
    condition: str
    seed: int
    episodes: int
    env_version: str
    agent_version: str
    prompt_version: str
    model_name: str = "gpt-4o-mini"
    
    def get_hash(self) -> str:
        """設定のハッシュ値を生成"""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


class RunLogger:
    """実測ログの記録・管理"""
    
    def __init__(self, out_dir: str, config: ExperimentConfig):
        self.config = config
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # ファイルパス設定
        config_hash = config.get_hash()
        self.steps_file = self.out_dir / f"steps_{config_hash}.csv"
        self.config_file = self.out_dir / f"config_{config_hash}.json"
        self.summary_file = self.out_dir / f"summary_{config_hash}.json"
        
        # 設定ファイル保存
        with self.config_file.open("w") as f:
            json.dump(asdict(config), f, indent=2)
        
        # CSVヘッダー初期化
        if not self.steps_file.exists():
            with self.steps_file.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(StepLog.__annotations__.keys()))
                writer.writeheader()
    
    def log_step(self, step_log: StepLog):
        """ステップログを記録"""
        with self.steps_file.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(StepLog.__annotations__.keys()))
            writer.writerow(asdict(step_log))
    
    def log_summary(self, summary_data: Dict[str, Any]):
        """実験サマリーを記録"""
        summary_data["config_hash"] = self.config.get_hash()
        summary_data["timestamp"] = time.time()
        
        with self.summary_file.open("w") as f:
            json.dump(summary_data, f, indent=2)
    
    def get_state_hash(self, state: Any) -> str:
        """状態のハッシュ値を生成"""
        if hasattr(state, '__dict__'):
            state_str = json.dumps(state.__dict__, sort_keys=True, default=str)
        else:
            state_str = str(state)
        return hashlib.md5(state_str.encode()).hexdigest()[:8]


class LLMInteractionLogger:
    """LLM介入の詳細ログ"""
    
    def __init__(self, out_dir: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.interactions_file = self.out_dir / "llm_interactions.jsonl"
    
    def log_interaction(self, episode: int, step: int, prompt: str, 
                       response: str, decision: str, adopted: bool):
        """LLM介入の詳細を記録"""
        interaction = {
            "episode": episode,
            "step": step,
            "timestamp": time.time(),
            "prompt": prompt,
            "response": response,
            "decision": decision,
            "adopted": adopted,
            "prompt_id": hashlib.md5(prompt.encode()).hexdigest()[:8]
        }
        
        with self.interactions_file.open("a") as f:
            f.write(json.dumps(interaction) + "\n")


def create_run_directory(base_dir: str = "runs/real") -> Path:
    """実測ログ用ディレクトリを作成"""
    run_dir = Path(base_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def validate_no_synthetic_data(file_path: str) -> bool:
    """ファイルに合成データ生成コードが含まれていないことを確認"""
    # 自分自身（logger.py）は検証対象から除外
    if file_path.endswith('logger.py'):
        return True
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 統計分析用の合成データ生成パターンのみを検出
    synthetic_patterns = [
        'np.random.normal(',  # 統計分析用の偽データ生成
        'np.random.poisson(',  # 統計分析用の偽データ生成
        'simulate_experiment_data',  # 実験データシミュレーション
        'generate_fake_results',  # 偽結果生成
        'mock_statistical_data',  # 統計データモック
        'synthetic_performance_data',  # 合成パフォーマンスデータ
        'fake_llm_responses',  # 偽LLM応答
        'simulated_scores = np.random'  # スコアシミュレーション
    ]
    
    for pattern in synthetic_patterns:
        if pattern in content:
            return False
    
    return True


def validate_experiment_integrity(run_dir: str) -> Dict[str, Any]:
    """実験の整合性を検証し、合成データ使用をチェック"""
    validation_result = {
        "timestamp": time.time(),
        "run_directory": run_dir,
        "synthetic_data_detected": False,
        "files_checked": 0,
        "problematic_files": [],
        "data_quality_score": 100.0,
        "validation_passed": True
    }
    
    # 実験ランナーファイルをチェック
    check_files = [
        "run_elm_real.py",
        "run_elm_llm_real.py", 
        "run_fixed_seed_experiments.py",
        "analyze_real_data.py",
        "src/llm_teacher.py",
        "logger.py"
    ]
    
    for file_path in check_files:
        if Path(file_path).exists():
            validation_result["files_checked"] += 1
            if not validate_no_synthetic_data(file_path):
                validation_result["synthetic_data_detected"] = True
                validation_result["problematic_files"].append(file_path)
                validation_result["data_quality_score"] -= 20.0
    
    if validation_result["synthetic_data_detected"]:
        validation_result["validation_passed"] = False
    
    return validation_result
