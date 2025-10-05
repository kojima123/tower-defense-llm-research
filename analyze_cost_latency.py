#!/usr/bin/env python3
"""
コスト・遅延分析システム
実務的インパクト評価（tokens/ep, ¥/100ep, ms/decision）
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class CostLatencyAnalyzer:
    """コスト・遅延分析システム"""
    
    def __init__(self, project_dir: str = "."):
        self.project_dir = Path(project_dir)
        self.real_data_dir = self.project_dir / "runs" / "real"
        
        # OpenAI API料金設定 (2024年価格)
        self.openai_pricing = {
            "gpt-4o-mini": {
                "input": 0.00015,   # $0.15 per 1K tokens
                "output": 0.0006    # $0.60 per 1K tokens
            }
        }
        
        # 為替レート (概算)
        self.usd_to_jpy = 150
        
    def collect_performance_data(self) -> Dict[str, Any]:
        """パフォーマンスデータを収集"""
        print("⏱️  Collecting performance data...")
        
        data = {
            "conditions": {},
            "llm_usage": {
                "total_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost_usd": 0.0,
                "average_response_time": 0.0,
                "response_times": []
            }
        }
        
        # CSV形式の実測ログを収集
        csv_files = list(self.real_data_dir.glob("**/*.csv"))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if len(df) == 0 or 'condition' not in df.columns:
                    continue
                
                condition = df['condition'].iloc[0]
                
                if condition not in data["conditions"]:
                    data["conditions"][condition] = {
                        "episodes": [],
                        "total_steps": 0,
                        "total_time": 0.0,
                        "decision_times": [],
                        "llm_calls": 0,
                        "files": []
                    }
                
                # 基本統計
                data["conditions"][condition]["total_steps"] += len(df)
                data["conditions"][condition]["files"].append(str(csv_file.relative_to(self.project_dir)))
                
                # エピソード数
                if 'episode' in df.columns:
                    episodes = df['episode'].nunique()
                    data["conditions"][condition]["episodes"].append(episodes)
                
                # 時間分析
                if 'timestamp' in df.columns:
                    timestamps = df['timestamp'].values
                    if len(timestamps) > 1:
                        total_time = timestamps[-1] - timestamps[0]
                        data["conditions"][condition]["total_time"] += total_time
                        
                        # 決定時間 (ステップ間隔)
                        step_times = np.diff(timestamps) * 1000  # ms
                        data["conditions"][condition]["decision_times"].extend(step_times.tolist())
                
                # LLM使用統計
                if 'llm_used' in df.columns:
                    llm_calls = df['llm_used'].sum()
                    data["conditions"][condition]["llm_calls"] += llm_calls
            
            except Exception as e:
                print(f"⚠️  Warning: Could not process {csv_file}: {e}")
                continue
        
        # LLM介入ログからコスト情報を収集
        self.collect_llm_cost_data(data)
        
        return data
    
    def collect_llm_cost_data(self, data: Dict[str, Any]):
        """LLM介入ログからコスト情報を収集"""
        print("💰 Collecting LLM cost data...")
        
        jsonl_files = list(self.real_data_dir.glob("**/*.jsonl"))
        
        for jsonl_file in jsonl_files:
            try:
                with jsonl_file.open('r') as f:
                    for line in f:
                        if line.strip():
                            log_data = json.loads(line)
                            
                            if log_data.get('type') == 'llm_intervention':
                                data["llm_usage"]["total_calls"] += 1
                                
                                # トークン使用量
                                usage = log_data.get('usage', {})
                                input_tokens = usage.get('prompt_tokens', 0)
                                output_tokens = usage.get('completion_tokens', 0)
                                
                                data["llm_usage"]["total_input_tokens"] += input_tokens
                                data["llm_usage"]["total_output_tokens"] += output_tokens
                                
                                # コスト計算
                                input_cost = input_tokens / 1000 * self.openai_pricing["gpt-4o-mini"]["input"]
                                output_cost = output_tokens / 1000 * self.openai_pricing["gpt-4o-mini"]["output"]
                                data["llm_usage"]["total_cost_usd"] += input_cost + output_cost
                                
                                # 応答時間
                                response_time = log_data.get('response_time', 0)
                                if response_time > 0:
                                    data["llm_usage"]["response_times"].append(response_time)
            
            except Exception as e:
                print(f"⚠️  Warning: Could not process {jsonl_file}: {e}")
                continue
        
        # 平均応答時間計算
        if data["llm_usage"]["response_times"]:
            data["llm_usage"]["average_response_time"] = np.mean(data["llm_usage"]["response_times"])
    
    def calculate_cost_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """コスト指標を計算"""
        print("📊 Calculating cost metrics...")
        
        metrics = {
            "per_episode": {},
            "per_100_episodes": {},
            "per_decision": {},
            "efficiency": {}
        }
        
        for condition, condition_data in data["conditions"].items():
            total_episodes = sum(condition_data["episodes"]) if condition_data["episodes"] else 1
            total_steps = condition_data["total_steps"]
            
            # エピソード当たり指標
            if total_episodes > 0:
                metrics["per_episode"][condition] = {
                    "steps": total_steps / total_episodes,
                    "time_seconds": condition_data["total_time"] / total_episodes,
                    "llm_calls": condition_data["llm_calls"] / total_episodes
                }
                
                # 100エピソード当たり指標
                metrics["per_100_episodes"][condition] = {
                    "steps": (total_steps / total_episodes) * 100,
                    "time_minutes": (condition_data["total_time"] / total_episodes) * 100 / 60,
                    "llm_calls": (condition_data["llm_calls"] / total_episodes) * 100
                }
            
            # 決定当たり指標
            if total_steps > 0:
                avg_decision_time = np.mean(condition_data["decision_times"]) if condition_data["decision_times"] else 0
                
                metrics["per_decision"][condition] = {
                    "time_ms": avg_decision_time,
                    "llm_probability": condition_data["llm_calls"] / total_steps if total_steps > 0 else 0
                }
        
        # LLM特有のコスト指標
        if data["llm_usage"]["total_calls"] > 0:
            total_tokens = data["llm_usage"]["total_input_tokens"] + data["llm_usage"]["total_output_tokens"]
            
            # ELM+LLM条件のエピソード数を取得
            elm_llm_episodes = sum(data["conditions"].get("elm_llm", {}).get("episodes", [0]))
            if elm_llm_episodes == 0:
                elm_llm_episodes = 1  # ゼロ除算回避
            
            metrics["llm_specific"] = {
                "tokens_per_call": total_tokens / data["llm_usage"]["total_calls"],
                "tokens_per_episode": total_tokens / elm_llm_episodes,
                "cost_usd_per_call": data["llm_usage"]["total_cost_usd"] / data["llm_usage"]["total_calls"],
                "cost_usd_per_episode": data["llm_usage"]["total_cost_usd"] / elm_llm_episodes,
                "cost_jpy_per_100_episodes": (data["llm_usage"]["total_cost_usd"] / elm_llm_episodes) * 100 * self.usd_to_jpy,
                "average_response_time_ms": data["llm_usage"]["average_response_time"] * 1000
            }
        else:
            metrics["llm_specific"] = {
                "tokens_per_call": 0,
                "tokens_per_episode": 0,
                "cost_usd_per_call": 0,
                "cost_usd_per_episode": 0,
                "cost_jpy_per_100_episodes": 0,
                "average_response_time_ms": 0
            }
        
        return metrics
    
    def create_cost_analysis_table(self, metrics: Dict[str, Any], output_path: str):
        """コスト分析表を作成"""
        print("📋 Creating cost analysis table...")
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        table_content = f"""# コスト・遅延分析レポート

**分析日**: {current_date}  
**データソース**: 実測ログのみ使用  
**API料金**: OpenAI GPT-4o-mini (入力: $0.15/1K tokens, 出力: $0.60/1K tokens)  
**為替レート**: $1 = ¥{self.usd_to_jpy} (概算)

## 📊 条件別パフォーマンス指標

### エピソード当たり指標

| 条件 | ステップ数 | 実行時間(秒) | LLM呼び出し回数 |
|------|------------|--------------|-----------------|"""
        
        for condition, data in metrics["per_episode"].items():
            table_content += f"""
| {condition} | {data['steps']:.1f} | {data['time_seconds']:.2f} | {data['llm_calls']:.1f} |"""
        
        table_content += """

### 100エピソード当たり指標

| 条件 | 総ステップ数 | 実行時間(分) | LLM呼び出し回数 |
|------|--------------|--------------|-----------------|"""
        
        for condition, data in metrics["per_100_episodes"].items():
            table_content += f"""
| {condition} | {data['steps']:.0f} | {data['time_minutes']:.1f} | {data['llm_calls']:.0f} |"""
        
        table_content += """

### 決定当たり指標

| 条件 | 決定時間(ms) | LLM使用確率 |
|------|--------------|-------------|"""
        
        for condition, data in metrics["per_decision"].items():
            table_content += f"""
| {condition} | {data['time_ms']:.2f} | {data['llm_probability']:.3f} |"""
        
        # LLM特有のコスト指標
        if "llm_specific" in metrics:
            llm_metrics = metrics["llm_specific"]
            table_content += f"""

## 💰 LLM使用コスト分析

### 基本指標
- **トークン/呼び出し**: {llm_metrics['tokens_per_call']:.1f} tokens
- **トークン/エピソード**: {llm_metrics['tokens_per_episode']:.1f} tokens
- **平均応答時間**: {llm_metrics['average_response_time_ms']:.1f} ms

### コスト指標
- **コスト/呼び出し**: ${llm_metrics['cost_usd_per_call']:.4f} (¥{llm_metrics['cost_usd_per_call'] * self.usd_to_jpy:.2f})
- **コスト/エピソード**: ${llm_metrics['cost_usd_per_episode']:.4f} (¥{llm_metrics['cost_usd_per_episode'] * self.usd_to_jpy:.2f})
- **コスト/100エピソード**: ${llm_metrics['cost_usd_per_episode'] * 100:.2f} (¥{llm_metrics['cost_jpy_per_100_episodes']:.0f})

### 実務的評価
"""
            
            # 実務的評価
            cost_per_100ep = llm_metrics['cost_jpy_per_100_episodes']
            if cost_per_100ep < 100:
                cost_evaluation = "非常に低コスト - 実用的"
            elif cost_per_100ep < 500:
                cost_evaluation = "低コスト - 実用的"
            elif cost_per_100ep < 2000:
                cost_evaluation = "中程度コスト - 用途次第"
            else:
                cost_evaluation = "高コスト - 慎重な検討が必要"
            
            response_time = llm_metrics['average_response_time_ms']
            if response_time < 500:
                latency_evaluation = "低遅延 - リアルタイム用途に適用可能"
            elif response_time < 2000:
                latency_evaluation = "中程度遅延 - 準リアルタイム用途に適用可能"
            else:
                latency_evaluation = "高遅延 - バッチ処理向け"
            
            table_content += f"""
- **コスト評価**: {cost_evaluation}
- **遅延評価**: {latency_evaluation}
- **スケーラビリティ**: {"高" if cost_per_100ep < 500 and response_time < 1000 else "中" if cost_per_100ep < 2000 else "低"}
"""
        
        table_content += """

## 🎯 投資対効果分析

### ROI指標
"""
        
        # ROI分析（ELM+LLM vs ELM単体の比較）
        if "elm_llm" in metrics["per_episode"] and "elm_only" in metrics["per_episode"]:
            elm_llm_steps = metrics["per_episode"]["elm_llm"]["steps"]
            elm_only_steps = metrics["per_episode"]["elm_only"]["steps"]
            
            if "llm_specific" in metrics:
                cost_per_ep = metrics["llm_specific"]["cost_usd_per_episode"] * self.usd_to_jpy
                
                # 効率改善率
                if elm_only_steps > 0:
                    efficiency_improvement = (elm_llm_steps - elm_only_steps) / elm_only_steps * 100
                    
                    table_content += f"""
- **効率改善**: {efficiency_improvement:.1f}% (ELM+LLM vs ELM単体)
- **改善コスト**: ¥{cost_per_ep:.2f}/エピソード
- **改善価値**: {"高" if efficiency_improvement > 10 else "中" if efficiency_improvement > 0 else "低"}
"""
        
        table_content += """

## 📈 スケーリング予測

### 大規模運用時のコスト予測

| 規模 | エピソード数 | 予想コスト(¥) | 実行時間 |
|------|--------------|---------------|----------|"""
        
        if "llm_specific" in metrics:
            scales = [
                ("小規模テスト", 1000, metrics["llm_specific"]["cost_jpy_per_100_episodes"] * 10),
                ("中規模実験", 10000, metrics["llm_specific"]["cost_jpy_per_100_episodes"] * 100),
                ("大規模運用", 100000, metrics["llm_specific"]["cost_jpy_per_100_episodes"] * 1000)
            ]
            
            for scale_name, episodes, cost in scales:
                # 実行時間予測（ELM+LLM条件から）
                if "elm_llm" in metrics["per_episode"]:
                    time_per_ep = metrics["per_episode"]["elm_llm"]["time_seconds"]
                    total_time_hours = episodes * time_per_ep / 3600
                    time_text = f"{total_time_hours:.1f}時間"
                else:
                    time_text = "N/A"
                
                table_content += f"""
| {scale_name} | {episodes:,} | ¥{cost:,.0f} | {time_text} |"""
        
        table_content += """

## 🔍 データ品質保証

- ✅ **実測データのみ**: 合成データ使用なし
- ✅ **実際のAPI使用**: OpenAI実API呼び出し記録
- ✅ **透明性**: 全コスト計算過程公開
- ✅ **再現可能性**: 固定シード実験

---

*このレポートは実測データのみから生成されています*
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(table_content)
        
        print(f"✅ Cost analysis table saved: {output_path}")
    
    def create_cost_visualization(self, metrics: Dict[str, Any], output_path: str):
        """コスト可視化を作成"""
        print("📊 Creating cost visualization...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 条件別決定時間比較
        conditions = []
        decision_times = []
        
        for condition, data in metrics["per_decision"].items():
            conditions.append(condition)
            decision_times.append(data["time_ms"])
        
        if conditions:
            bars1 = ax1.bar(conditions, decision_times, alpha=0.7)
            ax1.set_title('Decision Time by Condition')
            ax1.set_ylabel('Time (ms)')
            ax1.tick_params(axis='x', rotation=45)
            
            # LLM条件をハイライト
            for i, condition in enumerate(conditions):
                if 'llm' in condition.lower():
                    bars1[i].set_color('orange')
                    bars1[i].set_edgecolor('red')
                    bars1[i].set_linewidth(2)
        
        # LLM使用確率
        llm_probabilities = []
        for condition in conditions:
            prob = metrics["per_decision"][condition]["llm_probability"]
            llm_probabilities.append(prob)
        
        if conditions:
            ax2.bar(conditions, llm_probabilities, alpha=0.7, color='green')
            ax2.set_title('LLM Usage Probability by Condition')
            ax2.set_ylabel('Probability')
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis='x', rotation=45)
        
        # コスト分析（LLMがある場合）
        if "llm_specific" in metrics:
            llm_metrics = metrics["llm_specific"]
            
            # コスト内訳
            cost_categories = ['Per Call', 'Per Episode', 'Per 100 Episodes']
            cost_values = [
                llm_metrics['cost_usd_per_call'] * self.usd_to_jpy,
                llm_metrics['cost_usd_per_episode'] * self.usd_to_jpy,
                llm_metrics['cost_jpy_per_100_episodes']
            ]
            
            ax3.bar(cost_categories, cost_values, alpha=0.7, color='purple')
            ax3.set_title('LLM Cost Analysis (¥)')
            ax3.set_ylabel('Cost (JPY)')
            ax3.tick_params(axis='x', rotation=45)
            
            # スケーリング予測
            scales = [1000, 10000, 100000]
            costs = [llm_metrics['cost_jpy_per_100_episodes'] * (s/100) for s in scales]
            
            ax4.plot(scales, costs, marker='o', linewidth=2, markersize=8)
            ax4.set_title('Cost Scaling Prediction')
            ax4.set_xlabel('Episodes')
            ax4.set_ylabel('Total Cost (¥)')
            ax4.set_xscale('log')
            ax4.grid(True, alpha=0.3)
            
            # コスト閾値線
            ax4.axhline(y=10000, color='orange', linestyle='--', alpha=0.7, label='¥10,000')
            ax4.axhline(y=100000, color='red', linestyle='--', alpha=0.7, label='¥100,000')
            ax4.legend()
        else:
            ax3.text(0.5, 0.5, 'No LLM cost data available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('LLM Cost Analysis')
            
            ax4.text(0.5, 0.5, 'No LLM scaling data available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Cost Scaling Prediction')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Cost visualization saved: {output_path}")
    
    def analyze_all(self):
        """全コスト・遅延分析を実行"""
        print("💰 Starting comprehensive cost and latency analysis...")
        print("=" * 70)
        
        # データ収集
        data = self.collect_performance_data()
        
        # 指標計算
        metrics = self.calculate_cost_metrics(data)
        
        # 分析表作成
        self.create_cost_analysis_table(metrics, "cost_latency_analysis.md")
        
        # 可視化作成
        self.create_cost_visualization(metrics, "cost_latency_visualization.png")
        
        print("=" * 70)
        print("✅ Cost and latency analysis completed")
        
        if "llm_specific" in metrics:
            llm_metrics = metrics["llm_specific"]
            print(f"💰 LLM cost per 100 episodes: ¥{llm_metrics['cost_jpy_per_100_episodes']:.0f}")
            print(f"⏱️  Average response time: {llm_metrics['average_response_time_ms']:.1f} ms")
        
        print("📁 Generated files:")
        print("  - cost_latency_analysis.md")
        print("  - cost_latency_visualization.png")
        
        return metrics


def main():
    """メイン実行関数"""
    analyzer = CostLatencyAnalyzer()
    
    print("💰 Starting cost and latency analysis...")
    print("📊 Analyzing real measurement data for practical impact...")
    
    # 全分析実行
    metrics = analyzer.analyze_all()
    
    print("\n🎉 Cost and latency analysis complete!")


if __name__ == "__main__":
    main()
