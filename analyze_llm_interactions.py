#!/usr/bin/env python3
"""
LLMインタラクション分析ツール
実測ログからLLMの介入パターンと効果を分析
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any
import numpy as np


class LLMInteractionAnalyzer:
    """LLMインタラクション分析クラス"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.interactions = []
        self.load_interactions()
    
    def load_interactions(self):
        """LLMインタラクションログを読み込み"""
        interaction_file = self.log_dir / "llm_interactions.jsonl"
        
        if not interaction_file.exists():
            print(f"Warning: No LLM interaction log found at {interaction_file}")
            return
        
        with interaction_file.open('r') as f:
            for line in f:
                try:
                    interaction = json.loads(line.strip())
                    self.interactions.append(interaction)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}")
        
        print(f"Loaded {len(self.interactions)} LLM interactions")
    
    def analyze_interaction_patterns(self) -> Dict[str, Any]:
        """インタラクションパターンを分析"""
        if not self.interactions:
            return {}
        
        df = pd.DataFrame(self.interactions)
        
        analysis = {
            "total_interactions": len(self.interactions),
            "unique_episodes": df['episode'].nunique(),
            "unique_prompts": df['prompt_id'].nunique(),
            "adoption_rate": df['adopted'].mean() if 'adopted' in df.columns else 0,
            "interactions_per_episode": df.groupby('episode').size().describe().to_dict(),
            "decision_types": df['decision'].value_counts().to_dict() if 'decision' in df.columns else {}
        }
        
        # 時系列分析
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        analysis["time_span"] = {
            "start": df['timestamp'].min().isoformat(),
            "end": df['timestamp'].max().isoformat(),
            "duration_seconds": (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
        }
        
        # プロンプトの多様性分析
        prompt_lengths = df['prompt'].str.len()
        analysis["prompt_analysis"] = {
            "avg_length": prompt_lengths.mean(),
            "min_length": prompt_lengths.min(),
            "max_length": prompt_lengths.max(),
            "std_length": prompt_lengths.std()
        }
        
        # レスポンス分析
        response_lengths = df['response'].str.len()
        analysis["response_analysis"] = {
            "avg_length": response_lengths.mean(),
            "min_length": response_lengths.min(),
            "max_length": response_lengths.max(),
            "std_length": response_lengths.std()
        }
        
        return analysis
    
    def extract_action_recommendations(self) -> Dict[str, int]:
        """LLMの行動推奨を抽出・分類"""
        action_counts = {
            "place_tower": 0,
            "upgrade_tower": 0,
            "wait": 0,
            "focus_defense": 0,
            "other": 0
        }
        
        for interaction in self.interactions:
            response = interaction.get('response', '').lower()
            
            if 'place_tower' in response:
                action_counts["place_tower"] += 1
            elif 'upgrade_tower' in response or 'upgrade' in response:
                action_counts["upgrade_tower"] += 1
            elif 'wait' in response or 'save' in response:
                action_counts["wait"] += 1
            elif 'focus_defense' in response or 'defense' in response:
                action_counts["focus_defense"] += 1
            else:
                action_counts["other"] += 1
        
        return action_counts
    
    def analyze_adoption_patterns(self) -> Dict[str, Any]:
        """LLM推奨の採用パターンを分析"""
        if not self.interactions:
            return {}
        
        df = pd.DataFrame(self.interactions)
        
        if 'adopted' not in df.columns:
            return {"error": "No adoption data available"}
        
        # エピソード別採用率
        episode_adoption = df.groupby('episode')['adopted'].agg(['count', 'sum', 'mean']).reset_index()
        episode_adoption['adoption_rate'] = episode_adoption['mean']
        
        # ステップ別採用パターン
        step_adoption = df.groupby('step')['adopted'].agg(['count', 'sum', 'mean']).reset_index()
        
        return {
            "overall_adoption_rate": df['adopted'].mean(),
            "episode_adoption": episode_adoption.to_dict('records'),
            "step_adoption": step_adoption.to_dict('records'),
            "adoption_by_decision": df.groupby('decision')['adopted'].mean().to_dict() if 'decision' in df.columns else {}
        }
    
    def generate_interaction_report(self, output_file: str = None):
        """インタラクション分析レポートを生成"""
        if output_file is None:
            output_file = self.log_dir / "llm_interaction_analysis.md"
        
        # 分析実行
        patterns = self.analyze_interaction_patterns()
        actions = self.extract_action_recommendations()
        adoption = self.analyze_adoption_patterns()
        
        # レポート生成
        report = f"""# LLMインタラクション分析レポート

## 基本統計

- **総インタラクション数**: {patterns.get('total_interactions', 0)}
- **対象エピソード数**: {patterns.get('unique_episodes', 0)}
- **ユニークプロンプト数**: {patterns.get('unique_prompts', 0)}
- **全体採用率**: {patterns.get('adoption_rate', 0):.2%}

## 時系列情報

- **開始時刻**: {patterns.get('time_span', {}).get('start', 'N/A')}
- **終了時刻**: {patterns.get('time_span', {}).get('end', 'N/A')}
- **実行時間**: {patterns.get('time_span', {}).get('duration_seconds', 0):.2f}秒

## プロンプト分析

- **平均長**: {patterns.get('prompt_analysis', {}).get('avg_length', 0):.0f}文字
- **最小長**: {patterns.get('prompt_analysis', {}).get('min_length', 0)}文字
- **最大長**: {patterns.get('prompt_analysis', {}).get('max_length', 0)}文字

## レスポンス分析

- **平均長**: {patterns.get('response_analysis', {}).get('avg_length', 0):.0f}文字
- **最小長**: {patterns.get('response_analysis', {}).get('min_length', 0)}文字
- **最大長**: {patterns.get('response_analysis', {}).get('max_length', 0)}文字

## 行動推奨分析

"""
        
        for action, count in actions.items():
            percentage = (count / sum(actions.values()) * 100) if sum(actions.values()) > 0 else 0
            report += f"- **{action}**: {count}回 ({percentage:.1f}%)\n"
        
        report += f"""
## 採用パターン分析

- **全体採用率**: {adoption.get('overall_adoption_rate', 0):.2%}

### エピソード別採用率

"""
        
        for ep_data in adoption.get('episode_adoption', []):
            report += f"- エピソード{ep_data['episode']}: {ep_data['adoption_rate']:.2%} ({ep_data['sum']}/{ep_data['count']})\n"
        
        # ファイル保存
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📊 LLM interaction analysis report saved to: {output_file}")
        return report
    
    def create_visualization(self, output_dir: str = None):
        """インタラクション可視化を作成"""
        if not self.interactions:
            print("No interactions to visualize")
            return
        
        if output_dir is None:
            output_dir = self.log_dir
        
        df = pd.DataFrame(self.interactions)
        
        # 図のセットアップ
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('LLM Interaction Analysis', fontsize=16)
        
        # 1. 行動推奨分布
        actions = self.extract_action_recommendations()
        axes[0, 0].pie(actions.values(), labels=actions.keys(), autopct='%1.1f%%')
        axes[0, 0].set_title('Action Recommendations Distribution')
        
        # 2. エピソード別インタラクション数
        episode_counts = df['episode'].value_counts().sort_index()
        axes[0, 1].bar(episode_counts.index, episode_counts.values)
        axes[0, 1].set_title('Interactions per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Interaction Count')
        
        # 3. 採用率（エピソード別）
        if 'adopted' in df.columns:
            adoption_by_episode = df.groupby('episode')['adopted'].mean()
            axes[1, 0].plot(adoption_by_episode.index, adoption_by_episode.values, marker='o')
            axes[1, 0].set_title('Adoption Rate by Episode')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Adoption Rate')
            axes[1, 0].set_ylim(0, 1)
        
        # 4. レスポンス長分布
        response_lengths = df['response'].str.len()
        axes[1, 1].hist(response_lengths, bins=20, alpha=0.7)
        axes[1, 1].set_title('Response Length Distribution')
        axes[1, 1].set_xlabel('Response Length (characters)')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # 保存
        output_file = Path(output_dir) / "llm_interaction_visualization.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 LLM interaction visualization saved to: {output_file}")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze LLM interactions from experiment logs")
    parser.add_argument("log_dir", help="Directory containing LLM interaction logs")
    parser.add_argument("--output", help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    # 分析実行
    analyzer = LLMInteractionAnalyzer(args.log_dir)
    
    output_dir = args.output or args.log_dir
    
    # レポート生成
    analyzer.generate_interaction_report(Path(output_dir) / "llm_interaction_analysis.md")
    
    # 可視化作成
    analyzer.create_visualization(output_dir)
    
    print(f"✅ LLM interaction analysis completed!")


if __name__ == "__main__":
    main()
