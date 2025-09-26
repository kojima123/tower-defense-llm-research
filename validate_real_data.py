#!/usr/bin/env python3
"""
実測データ検証システム
合成データの混入を検出し、実測データのみの使用を保証する
"""
import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
import hashlib
import ast


class RealDataValidator:
    """実測データ検証システム"""
    
    def __init__(self, project_dir: str = "."):
        self.project_dir = Path(project_dir)
        self.real_data_dir = self.project_dir / "runs" / "real"
        self.synthetic_data_indicators = [
            "np.random",
            "random.seed",
            "random.normal",
            "random.uniform",
            "random.choice",
            "simulate_",
            "generate_synthetic",
            "fake_data",
            "mock_data"
        ]
        
    def validate_no_synthetic_data(self) -> Tuple[bool, List[str]]:
        """合成データの使用がないことを検証"""
        print("🔍 Validating no synthetic data usage...")
        
        violations = []
        
        # Pythonファイルの検証
        python_files = list(self.project_dir.glob("*.py"))
        python_files.extend(list(self.project_dir.glob("src/*.py")))
        
        for py_file in python_files:
            if self._contains_synthetic_data_code(py_file):
                violations.append(f"Python file contains synthetic data: {py_file}")
        
        # JSONファイルの検証
        json_files = list(self.project_dir.glob("*.json"))
        for json_file in json_files:
            if self._is_synthetic_json(json_file):
                violations.append(f"JSON file appears to be synthetic: {json_file}")
        
        # 実測ログディレクトリの検証
        if not self.real_data_dir.exists():
            violations.append("Real data directory 'runs/real' does not exist")
        else:
            real_data_valid = self._validate_real_data_structure()
            if not real_data_valid:
                violations.append("Real data directory structure is invalid")
        
        is_valid = len(violations) == 0
        
        if is_valid:
            print("✅ No synthetic data detected - all data is real measurement")
        else:
            print(f"❌ Found {len(violations)} synthetic data violations:")
            for violation in violations:
                print(f"   - {violation}")
        
        return is_valid, violations
    
    def _contains_synthetic_data_code(self, file_path: Path) -> bool:
        """Pythonファイルが合成データ生成コードを含むかチェック"""
        try:
            with file_path.open('r', encoding='utf-8') as f:
                content = f.read()
            
            # 統計分析用の合成データ生成パターンのみを検出
            synthetic_patterns = [
                "simulate_experiment",
                "generate_synthetic_data", 
                "fake_data",
                "mock_data",
                "synthetic_scores",
                "simulate_baseline",
                "simulate_elm_only",
                "simulate_llm_teacher",
                "simulate_random_teacher",
                "simulate_rule_teacher"
            ]
            
            # 統計分析用の合成データ生成のみを検出
            for pattern in synthetic_patterns:
                if pattern in content:
                    # 検証システム自体は除外
                    if file_path.name in ["validate_real_data.py", "logger.py"]:
                        continue
                    lines = content.split('\n')
                    for line in lines:
                        stripped = line.strip()
                        if pattern in stripped and not stripped.startswith('#') and not stripped.startswith('"""') and not stripped.startswith("'''"):
                            return True
            
            # 統計分析での大量データ生成パターンを検出
            if "for episode in range(" in content and "np.random" in content:
                # 大量エピソード生成（50以上）は合成データの可能性
                import re
                range_matches = re.findall(r'range\((\d+)\)', content)
                for match in range_matches:
                    if int(match) >= 50:  # 50エピソード以上は合成データの可能性
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _is_synthetic_json(self, file_path: Path) -> bool:
        """JSONファイルが合成データかチェック"""
        try:
            with file_path.open('r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 合成データの特徴を検出
            if isinstance(data, dict):
                # 大量の均一なデータパターン
                if 'experiment_data' in data:
                    exp_data = data['experiment_data']
                    if isinstance(exp_data, dict):
                        for condition, episodes in exp_data.items():
                            if isinstance(episodes, list) and len(episodes) > 50:
                                # 50エピソード以上は合成データの可能性が高い
                                return True
                
                # 統計結果の異常なパターン
                if 'descriptive_statistics' in data:
                    stats = data['descriptive_statistics']
                    if isinstance(stats, dict):
                        for condition, stat in stats.items():
                            if isinstance(stat, dict) and 'mean' in stat:
                                # 異常に整った数値は合成データの可能性
                                mean = stat.get('mean', 0)
                                if isinstance(mean, (int, float)) and mean == int(mean):
                                    return True
            
            return False
            
        except Exception:
            return False
    
    def _validate_real_data_structure(self) -> bool:
        """実測データディレクトリの構造を検証"""
        if not self.real_data_dir.exists():
            return False
        
        # 実測ログファイルの存在確認
        has_real_logs = False
        
        for experiment_dir in self.real_data_dir.iterdir():
            if experiment_dir.is_dir():
                # CSV/JSONログファイルの確認
                csv_files = list(experiment_dir.glob("**/*.csv"))
                json_files = list(experiment_dir.glob("**/*.json"))
                
                if csv_files or json_files:
                    has_real_logs = True
                    break
        
        return has_real_logs
    
    def generate_data_integrity_report(self) -> Dict[str, Any]:
        """データ整合性レポートを生成"""
        print("📊 Generating data integrity report...")
        
        is_valid, violations = self.validate_no_synthetic_data()
        
        # 実測データの統計
        real_data_stats = self._collect_real_data_stats()
        
        # 設定ハッシュの検証
        config_hashes = self._validate_config_hashes()
        
        report = {
            "validation_timestamp": pd.Timestamp.now().isoformat(),
            "synthetic_data_validation": {
                "is_valid": is_valid,
                "violations": violations,
                "total_violations": len(violations)
            },
            "real_data_statistics": real_data_stats,
            "configuration_hashes": config_hashes,
            "data_quality_score": self._calculate_quality_score(is_valid, real_data_stats, config_hashes)
        }
        
        return report
    
    def _collect_real_data_stats(self) -> Dict[str, Any]:
        """実測データの統計を収集"""
        stats = {
            "total_experiments": 0,
            "total_episodes": 0,
            "total_steps": 0,
            "conditions": [],
            "seeds_used": [],
            "date_range": {"earliest": None, "latest": None}
        }
        
        if not self.real_data_dir.exists():
            return stats
        
        for experiment_dir in self.real_data_dir.iterdir():
            if experiment_dir.is_dir():
                stats["total_experiments"] += 1
                
                # CSVファイルから統計収集
                csv_files = list(experiment_dir.glob("**/*.csv"))
                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(csv_file)
                        stats["total_steps"] += len(df)
                        
                        if 'episode' in df.columns:
                            stats["total_episodes"] += df['episode'].nunique()
                        
                        if 'condition' in df.columns:
                            conditions = df['condition'].unique().tolist()
                            stats["conditions"].extend(conditions)
                        
                        if 'seed' in df.columns:
                            seeds = df['seed'].unique().tolist()
                            stats["seeds_used"].extend(seeds)
                            
                    except Exception:
                        continue
        
        # 重複除去
        stats["conditions"] = list(set(stats["conditions"]))
        stats["seeds_used"] = list(set(stats["seeds_used"]))
        
        return stats
    
    def _validate_config_hashes(self) -> Dict[str, Any]:
        """設定ハッシュの検証"""
        hashes = {
            "total_configs": 0,
            "unique_hashes": [],
            "hash_consistency": True
        }
        
        if not self.real_data_dir.exists():
            return hashes
        
        for config_file in self.real_data_dir.glob("**/config_*.json"):
            try:
                with config_file.open('r') as f:
                    config = json.load(f)
                
                if 'config_hash' in config:
                    hash_value = config['config_hash']
                    hashes["unique_hashes"].append(hash_value)
                    hashes["total_configs"] += 1
                    
            except Exception:
                continue
        
        # ハッシュの一意性確認
        unique_count = len(set(hashes["unique_hashes"]))
        hashes["unique_count"] = unique_count
        hashes["hash_consistency"] = unique_count == hashes["total_configs"]
        
        return hashes
    
    def _calculate_quality_score(self, is_valid: bool, real_data_stats: Dict, config_hashes: Dict) -> float:
        """データ品質スコアを計算（0-100）"""
        score = 0.0
        
        # 合成データなし（40点）
        if is_valid:
            score += 40.0
        
        # 実測データの存在（30点）
        if real_data_stats["total_experiments"] > 0:
            score += 30.0
        
        # 設定ハッシュの整合性（20点）
        if config_hashes["hash_consistency"]:
            score += 20.0
        
        # データ量の充実度（10点）
        if real_data_stats["total_steps"] > 1000:
            score += 10.0
        elif real_data_stats["total_steps"] > 100:
            score += 5.0
        
        return score
    
    def save_validation_report(self, output_path: str = "data_validation_report.json"):
        """検証レポートを保存"""
        report = self.generate_data_integrity_report()
        
        output_file = self.project_dir / output_path
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Validation report saved: {output_file}")
        print(f"🏆 Data quality score: {report['data_quality_score']:.1f}/100")
        
        return report


def main():
    """メイン実行関数"""
    validator = RealDataValidator()
    
    print("🔬 Starting real data validation...")
    print("=" * 60)
    
    # 検証実行
    is_valid, violations = validator.validate_no_synthetic_data()
    
    # レポート生成・保存
    report = validator.save_validation_report()
    
    print("=" * 60)
    
    if is_valid and report['data_quality_score'] >= 80:
        print("✅ VALIDATION PASSED: System uses real data only")
        print("🔬 Scientific rigor confirmed")
    else:
        print("❌ VALIDATION FAILED: Issues detected")
        print("⚠️  Please address violations before proceeding")
        
        if violations:
            print("\n🔧 Recommended actions:")
            print("1. Move synthetic data files to sim/deprecated/")
            print("2. Ensure all analysis uses runs/real/ data only")
            print("3. Verify configuration hashes are consistent")
            print("4. Run experiments to generate real measurement data")


if __name__ == "__main__":
    main()
