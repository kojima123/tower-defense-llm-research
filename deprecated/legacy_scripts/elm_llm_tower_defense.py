"""
ELM + LLM Teacher Tower Defense System
ELMとLLM教師を組み合わせたタワーディフェンス学習システム
"""

import numpy as np
import json
import time
import requests
import os
from typing import Dict, List, Tuple, Optional
from elm_tower_defense import TowerDefenseEnvironment, ELMTowerDefenseAgent

class LLMTeacher:
    """LLM教師システム"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-4o-mini"
        
        # 評価履歴
        self.evaluation_history = []
        self.api_calls = 0
        self.total_api_time = 0
    
    def evaluate_game_state(self, env: TowerDefenseEnvironment, agent_stats: Dict) -> Dict:
        """ゲーム状態を評価し、戦略的指導を提供"""
        if not self.api_key:
            return self._fallback_evaluation(env, agent_stats)
        
        start_time = time.time()
        
        # ゲーム状況の分析
        game_analysis = self._analyze_game_situation(env)
        
        # LLMプロンプトを構築
        prompt = self._build_evaluation_prompt(env, agent_stats, game_analysis)
        
        try:
            # OpenAI APIを呼び出し
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0.7
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                evaluation = self._parse_llm_response(result['choices'][0]['message']['content'])
            else:
                evaluation = self._fallback_evaluation(env, agent_stats)
        
        except Exception as e:
            print(f"LLM API Error: {e}")
            evaluation = self._fallback_evaluation(env, agent_stats)
        
        # 統計情報を更新
        self.api_calls += 1
        self.total_api_time += time.time() - start_time
        self.evaluation_history.append(evaluation)
        
        return evaluation
    
    def _analyze_game_situation(self, env: TowerDefenseEnvironment) -> Dict:
        """ゲーム状況を分析"""
        # 戦略的指標を計算
        tower_density = len(env.towers) / len(env.valid_positions) if env.valid_positions else 0
        resource_efficiency = env.score / max(1, len(env.towers) * env.TOWER_COST)
        threat_level = len(env.enemies) / max(1, env.wave)
        defense_coverage = self._calculate_defense_coverage(env)
        
        return {
            'tower_density': tower_density,
            'resource_efficiency': resource_efficiency,
            'threat_level': threat_level,
            'defense_coverage': defense_coverage,
            'economic_status': 'good' if env.money > 100 else 'tight' if env.money > 50 else 'critical',
            'health_status': 'excellent' if env.health > 80 else 'good' if env.health > 50 else 'critical'
        }
    
    def _calculate_defense_coverage(self, env: TowerDefenseEnvironment) -> float:
        """防御カバレッジを計算"""
        if not env.towers:
            return 0.0
        
        # パス上の各点での防御力を計算
        coverage_points = 0
        total_points = 100
        
        for i in range(total_points):
            progress = i / total_points
            pos = env._get_enemy_position(progress)
            
            # この位置での防御力を計算
            defense_strength = 0
            for tower in env.towers:
                distance = np.sqrt((pos[0] - tower['x'])**2 + (pos[1] - tower['y'])**2)
                if distance <= tower['range']:
                    defense_strength += tower['damage']
            
            if defense_strength > 0:
                coverage_points += 1
        
        return coverage_points / total_points
    
    def _build_evaluation_prompt(self, env: TowerDefenseEnvironment, agent_stats: Dict, analysis: Dict) -> str:
        """LLM評価用プロンプトを構築"""
        prompt = f"""
Tower Defense戦略分析と指導をお願いします。

【現在の状況】
- 資金: ${env.money}
- ヘルス: {env.health}/100
- ウェーブ: {env.wave}
- スコア: {env.score}
- タワー数: {len(env.towers)}
- 敵数: {len(env.enemies)}
- ゲーム時間: {env.game_time}秒

【戦略分析】
- タワー密度: {analysis['tower_density']:.2f}
- 資源効率: {analysis['resource_efficiency']:.2f}
- 脅威レベル: {analysis['threat_level']:.2f}
- 防御カバレッジ: {analysis['defense_coverage']:.2f}
- 経済状況: {analysis['economic_status']}
- ヘルス状況: {analysis['health_status']}

【エージェント統計】
- 探索率: {agent_stats.get('epsilon', 0):.3f}
- 経験バッファ: {agent_stats.get('experience_buffer_size', 0)}

以下の形式で戦略評価を提供してください：

STRATEGY: [最適な戦略 - expand/defend/economy/emergency]
PRIORITY: [優先度 - low/medium/high/urgent]
ACTION: [推奨行動 - build_tower/save_money/wait/focus_defense]
REASONING: [戦略的理由を1-2文で]
SCORE: [現在の戦略の評価 0-100]
"""
        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict:
        """LLMレスポンスを解析"""
        evaluation = {
            'strategy': 'defend',
            'priority': 'medium',
            'action': 'build_tower',
            'reasoning': 'バランスの取れた防御戦略を継続',
            'score': 50
        }
        
        lines = response.strip().split('\n')
        for line in lines:
            if line.startswith('STRATEGY:'):
                evaluation['strategy'] = line.split(':', 1)[1].strip()
            elif line.startswith('PRIORITY:'):
                evaluation['priority'] = line.split(':', 1)[1].strip()
            elif line.startswith('ACTION:'):
                evaluation['action'] = line.split(':', 1)[1].strip()
            elif line.startswith('REASONING:'):
                evaluation['reasoning'] = line.split(':', 1)[1].strip()
            elif line.startswith('SCORE:'):
                try:
                    evaluation['score'] = int(line.split(':', 1)[1].strip())
                except:
                    evaluation['score'] = 50
        
        return evaluation
    
    def _fallback_evaluation(self, env: TowerDefenseEnvironment, agent_stats: Dict) -> Dict:
        """APIが利用できない場合のフォールバック評価"""
        # ルールベースの簡単な評価
        if env.health < 30:
            return {
                'strategy': 'emergency',
                'priority': 'urgent',
                'action': 'build_tower',
                'reasoning': 'ヘルスが危険レベル、緊急防御が必要',
                'score': 20
            }
        elif env.money >= env.TOWER_COST * 2 and len(env.towers) < 6:
            return {
                'strategy': 'expand',
                'priority': 'high',
                'action': 'build_tower',
                'reasoning': '十分な資金があり、防御拡張が有効',
                'score': 70
            }
        elif len(env.enemies) > 8:
            return {
                'strategy': 'defend',
                'priority': 'high',
                'action': 'build_tower',
                'reasoning': '敵数が多く、防御強化が必要',
                'score': 60
            }
        else:
            return {
                'strategy': 'economy',
                'priority': 'medium',
                'action': 'save_money',
                'reasoning': '現状維持で経済発展を優先',
                'score': 55
            }
    
    def get_stats(self) -> Dict:
        """LLM教師の統計情報を取得"""
        return {
            'api_calls': self.api_calls,
            'avg_api_time': self.total_api_time / max(1, self.api_calls),
            'total_api_time': self.total_api_time,
            'evaluation_count': len(self.evaluation_history)
        }


class ELMLLMHybridAgent(ELMTowerDefenseAgent):
    """ELM + LLM教師ハイブリッドエージェント"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 100):
        super().__init__(state_size, action_size, hidden_size)
        
        self.llm_teacher = LLMTeacher()
        self.evaluation_interval = 180  # 180ステップ間隔でLLM評価
        self.last_evaluation_time = 0
        self.current_guidance = None
        
        # LLM指導の効果を追跡
        self.guidance_history = []
        self.action_modifications = 0
    
    def predict_with_guidance(self, state: np.ndarray, env: TowerDefenseEnvironment) -> int:
        """LLM指導を考慮した行動予測"""
        # 基本的なELM予測
        base_action = self.predict(state)
        
        # LLM評価のタイミングをチェック
        if env.game_time - self.last_evaluation_time >= self.evaluation_interval:
            self.current_guidance = self.llm_teacher.evaluate_game_state(env, self.get_stats())
            self.last_evaluation_time = env.game_time
            self.guidance_history.append(self.current_guidance)
        
        # LLM指導に基づいて行動を修正
        if self.current_guidance:
            modified_action = self._apply_guidance(base_action, env, self.current_guidance)
            if modified_action != base_action:
                self.action_modifications += 1
            return modified_action
        
        return base_action
    
    def _apply_guidance(self, base_action: int, env: TowerDefenseEnvironment, guidance: Dict) -> int:
        """LLM指導に基づいて行動を修正"""
        action_type = guidance.get('action', 'build_tower')
        priority = guidance.get('priority', 'medium')
        
        # 緊急度が高い場合は強制的に指導に従う
        if priority == 'urgent':
            if action_type == 'build_tower' and env.money >= env.TOWER_COST:
                # 最適な位置を選択
                return self._select_optimal_tower_position(env)
            elif action_type == 'save_money':
                return 0  # 何もしない
        
        # 高優先度の場合は70%の確率で指導に従う
        elif priority == 'high' and np.random.random() < 0.7:
            if action_type == 'build_tower' and env.money >= env.TOWER_COST:
                return self._select_optimal_tower_position(env)
            elif action_type == 'save_money':
                return 0
        
        # 中優先度の場合は40%の確率で指導に従う
        elif priority == 'medium' and np.random.random() < 0.4:
            if action_type == 'build_tower' and env.money >= env.TOWER_COST:
                return self._select_optimal_tower_position(env)
        
        # その他の場合は基本行動を維持
        return base_action
    
    def _select_optimal_tower_position(self, env: TowerDefenseEnvironment) -> int:
        """最適なタワー配置位置を選択"""
        if not env.valid_positions:
            return 0
        
        best_position = 0
        best_score = -1
        
        for i, pos in enumerate(env.valid_positions):
            # この位置での戦略的価値を計算
            score = self._evaluate_position_value(pos, env)
            if score > best_score:
                best_score = score
                best_position = i + 1  # action 0は何もしない
        
        return best_position
    
    def _evaluate_position_value(self, pos: Tuple[int, int], env: TowerDefenseEnvironment) -> float:
        """位置の戦略的価値を評価"""
        x, y = pos
        value = 0
        
        # パスとの距離（近いほど良い）
        min_path_distance = float('inf')
        for i in range(len(env.path) - 1):
            p1 = env.path[i]
            p2 = env.path[i + 1]
            dist = env._distance_to_line(x, y, p1[0], p1[1], p2[0], p2[1])
            min_path_distance = min(min_path_distance, dist)
        
        # パスに近いほど高価値（ただし配置可能範囲内）
        if min_path_distance < env.TOWER_RANGE:
            value += (env.TOWER_RANGE - min_path_distance) / env.TOWER_RANGE * 100
        
        # 他のタワーとの距離（適度な間隔が良い）
        for tower in env.towers:
            distance = np.sqrt((x - tower['x'])**2 + (y - tower['y'])**2)
            if distance < 60:  # 近すぎる
                value -= 50
            elif distance < 120:  # 適度な距離
                value += 20
        
        # 敵の密度が高い場所の近くは高価値
        for enemy in env.enemies:
            enemy_pos = env._get_enemy_position(enemy['path_progress'])
            distance = np.sqrt((x - enemy_pos[0])**2 + (y - enemy_pos[1])**2)
            if distance < env.TOWER_RANGE:
                value += 30
        
        return value
    
    def get_hybrid_stats(self) -> Dict:
        """ハイブリッドシステムの統計情報を取得"""
        base_stats = self.get_stats()
        llm_stats = self.llm_teacher.get_stats()
        
        return {
            **base_stats,
            'llm_stats': llm_stats,
            'action_modifications': self.action_modifications,
            'guidance_count': len(self.guidance_history),
            'modification_rate': self.action_modifications / max(1, self.episode_count)
        }


def run_elm_llm_experiment(episodes: int = 30) -> Dict:
    """ELM + LLM教師ハイブリッド実験を実行"""
    env = TowerDefenseEnvironment()
    state_size = len(env.get_state())
    action_size = env.get_action_size()
    
    agent = ELMLLMHybridAgent(state_size, action_size)
    
    results = {
        'scores': [],
        'survival_times': [],
        'towers_built': [],
        'enemies_killed': [],
        'final_health': [],
        'efficiency': [],
        'guidance_history': []
    }
    
    print(f"ELM + LLM教師ハイブリッド実験開始 - {episodes}エピソード")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.predict_with_guidance(state, env)
            next_state, reward, done = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done or steps > 1000:  # 最大1000ステップ
                break
        
        # 結果を記録
        results['scores'].append(env.score)
        results['survival_times'].append(env.game_time)
        results['towers_built'].append(len(env.towers))
        results['enemies_killed'].append(env.enemies_killed)
        results['final_health'].append(env.health)
        
        # 効率性を計算
        efficiency = env.score / max(1, len(env.towers) * env.TOWER_COST)
        results['efficiency'].append(efficiency)
        
        agent.total_reward += total_reward
        agent.episode_count += 1
        
        if (episode + 1) % 5 == 0:
            print(f"Episode {episode + 1}: Score={env.score}, Health={env.health}, Towers={len(env.towers)}")
    
    # 統計情報を追加
    results['agent_stats'] = agent.get_hybrid_stats()
    results['guidance_history'] = agent.guidance_history
    results['final_avg_score'] = np.mean(results['scores'][-5:])  # 最後5エピソードの平均
    results['final_avg_efficiency'] = np.mean(results['efficiency'][-5:])
    
    print(f"ELM + LLM教師実験完了 - 平均スコア: {results['final_avg_score']:.2f}")
    
    return results


if __name__ == "__main__":
    # テスト実行
    results = run_elm_llm_experiment(10)
    print(json.dumps(results, indent=2, default=str))
