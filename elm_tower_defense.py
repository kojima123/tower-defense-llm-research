"""
Tower Defense ELM Learning System
ELMを使用したタワーディフェンス学習システム
"""

import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional

class TowerDefenseEnvironment:
    """Tower Defense環境シミュレーター"""
    
    def __init__(self):
        # ゲーム設定（バランス調整済み）
        self.TOWER_COST = 50
        self.TOWER_DAMAGE = 60  # ダメージを大幅増加
        self.TOWER_RANGE = 150  # 射程を大幅拡大
        self.ENEMY_HEALTH = 80  # 敵の体力を減少
        self.ENEMY_SPEED = 0.7  # 敵の速度を減少
        self.ENEMY_REWARD = 30  # 報酬を3倍に増加
        self.INITIAL_MONEY = 250  # 初期資金を増加
        self.INITIAL_HEALTH = 100
        self.CANVAS_WIDTH = 800
        self.CANVAS_HEIGHT = 600
        
        # パス定義
        self.path = [
            (0, 300), (200, 300), (200, 150), (400, 150),
            (400, 450), (600, 450), (600, 300), (800, 300)
        ]
        
        # 配置可能位置（パスから離れた場所）
        self.valid_positions = self._generate_valid_positions()
        
        # 初期化
        self.reset()
    
    def _generate_valid_positions(self) -> List[Tuple[int, int]]:
        """タワー配置可能位置を生成"""
        positions = []
        for x in range(50, self.CANVAS_WIDTH - 50, 50):
            for y in range(50, self.CANVAS_HEIGHT - 50, 50):
                if self._is_valid_position(x, y):
                    positions.append((x, y))
        return positions
    
    def _is_valid_position(self, x: int, y: int) -> bool:
        """位置がタワー配置可能かチェック"""
        # パスから十分離れているかチェック
        for i in range(len(self.path) - 1):
            p1 = self.path[i]
            p2 = self.path[i + 1]
            dist = self._distance_to_line(x, y, p1[0], p1[1], p2[0], p2[1])
            if dist < 40:  # パスから40ピクセル以上離れている必要
                return False
        return True
    
    def _distance_to_line(self, px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
        """点から線分への距離を計算"""
        A = px - x1
        B = py - y1
        C = x2 - x1
        D = y2 - y1
        
        dot = A * C + B * D
        len_sq = C * C + D * D
        if len_sq == 0:
            return np.sqrt(A * A + B * B)
        
        param = dot / len_sq
        if param < 0:
            xx, yy = x1, y1
        elif param > 1:
            xx, yy = x2, y2
        else:
            xx = x1 + param * C
            yy = y1 + param * D
        
        dx = px - xx
        dy = py - yy
        return np.sqrt(dx * dx + dy * dy)
    
    def reset(self):
        """環境をリセット"""
        self.money = self.INITIAL_MONEY
        self.health = self.INITIAL_HEALTH
        self.wave = 1
        self.score = 0
        self.towers = []
        self.enemies = []
        self.game_time = 0
        self.wave_timer = 0
        self.total_enemies_spawned = 0
        self.enemies_killed = 0
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """現在の状態を特徴ベクトルとして取得"""
        features = [
            self.money / 1000.0,  # 正規化された資金
            self.health / 100.0,  # 正規化されたヘルス
            self.wave / 10.0,     # 正規化されたウェーブ
            len(self.towers) / 20.0,  # 正規化されたタワー数
            len(self.enemies) / 10.0,  # 正規化された敵数
            self.score / 10000.0,  # 正規化されたスコア
            self.game_time / 300.0,  # 正規化された時間（5分）
        ]
        
        # タワーの配置情報（グリッド形式）
        tower_grid = np.zeros((16, 12))  # 50x50グリッド
        for tower in self.towers:
            grid_x = min(15, max(0, int(tower['x'] // 50)))
            grid_y = min(11, max(0, int(tower['y'] // 50)))
            tower_grid[grid_x][grid_y] = 1.0
        
        # 敵の密度情報
        enemy_density = np.zeros(8)  # パスを8セクションに分割
        for enemy in self.enemies:
            section = min(7, max(0, int(enemy['path_progress'] * 8)))
            enemy_density[section] += 1.0
        enemy_density = enemy_density / 10.0  # 正規化
        
        # 全特徴を結合
        state = np.concatenate([
            features,
            tower_grid.flatten(),
            enemy_density
        ])
        
        return state
    
    def get_action_space_size(self) -> int:
        """行動空間のサイズを取得"""
        return len(self.valid_positions) + 1  # 各位置への配置 + 何もしない
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """1ステップ実行"""
        reward = 0
        
        # 行動を実行
        if action > 0 and action <= len(self.valid_positions):
            pos_index = action - 1
            x, y = self.valid_positions[pos_index]
            if self.money >= self.TOWER_COST:
                self.towers.append({
                    'x': x, 'y': y, 'damage': self.TOWER_DAMAGE,
                    'range': self.TOWER_RANGE, 'last_shot': 0, 'kills': 0
                })
                self.money -= self.TOWER_COST
                reward += 10  # タワー配置報酬
        
        # ゲーム状態を更新
        self._update_game_state()
        
        # 報酬を計算
        reward += self._calculate_reward()
        
        # 終了条件をチェック
        done = self.health <= 0 or self.game_time > 300  # 5分でタイムアウト
        
        return self.get_state(), reward, done
    
    def _update_game_state(self):
        """ゲーム状態を更新"""
        self.game_time += 1
        self.wave_timer += 1
        
        # 新しいウェーブの生成
        if len(self.enemies) == 0 and self.wave_timer > 180:  # 180ステップ後
            self._spawn_wave()
            self.wave_timer = 0
        
        # 敵の移動
        for enemy in self.enemies[:]:
            enemy['path_progress'] += enemy['speed'] / 800.0  # パス全長で正規化
            if enemy['path_progress'] >= 1.0:
                # 敵がゴールに到達
                self.health -= 10
                self.enemies.remove(enemy)
        
        # タワーの攻撃
        for tower in self.towers:
            if self.game_time - tower['last_shot'] > 60:  # 1秒間隔
                target = self._find_nearest_enemy(tower)
                if target:
                    target['health'] -= tower['damage']
                    tower['last_shot'] = self.game_time
                    if target['health'] <= 0:
                        self.money += self.ENEMY_REWARD
                        self.score += 100
                        self.enemies_killed += 1
                        tower['kills'] += 1
                        if target in self.enemies:
                            self.enemies.remove(target)
    
    def _spawn_wave(self):
        """新しいウェーブを生成"""
        wave_size = 5 + self.wave  # ウェーブごとに敵数増加
        for i in range(wave_size):
            enemy_health = self.ENEMY_HEALTH * (1 + self.wave * 0.2)
            self.enemies.append({
                'health': enemy_health,
                'max_health': enemy_health,
                'speed': self.ENEMY_SPEED * (1 + self.wave * 0.1),
                'path_progress': -i * 0.1,  # 時間差で出現
                'reward': self.ENEMY_REWARD
            })
            self.total_enemies_spawned += 1
        self.wave += 1
    
    def _find_nearest_enemy(self, tower: Dict) -> Optional[Dict]:
        """タワーから最も近い敵を見つける"""
        nearest = None
        min_distance = tower['range']
        
        for enemy in self.enemies:
            if enemy['path_progress'] < 0:  # まだ出現していない
                continue
            
            # 敵の現在位置を計算
            enemy_pos = self._get_enemy_position(enemy['path_progress'])
            distance = np.sqrt((enemy_pos[0] - tower['x'])**2 + (enemy_pos[1] - tower['y'])**2)
            
            if distance < min_distance:
                min_distance = distance
                nearest = enemy
        
        return nearest
    
    def _get_enemy_position(self, progress: float) -> Tuple[float, float]:
        """パス上の進行度から敵の位置を計算"""
        if progress <= 0:
            return self.path[0]
        if progress >= 1:
            return self.path[-1]
        
        # パス上の位置を線形補間で計算
        total_segments = len(self.path) - 1
        segment_progress = progress * total_segments
        segment_index = int(segment_progress)
        local_progress = segment_progress - segment_index
        
        if segment_index >= total_segments:
            return self.path[-1]
        
        p1 = self.path[segment_index]
        p2 = self.path[segment_index + 1]
        
        x = p1[0] + (p2[0] - p1[0]) * local_progress
        y = p1[1] + (p2[1] - p1[1]) * local_progress
        
        return (x, y)
    
    def _calculate_reward(self) -> float:
        """報酬を計算"""
        reward = 0
        
        # 基本的な生存報酬
        reward += 0.1
        
        # ヘルス維持報酬
        reward += (self.health / self.INITIAL_HEALTH) * 0.5
        
        # 効率性報酬（スコア/コスト比）
        if len(self.towers) > 0:
            efficiency = self.score / (len(self.towers) * self.TOWER_COST)
            reward += efficiency * 0.1
        
        # 敵撃破報酬は既にstepで加算済み
        
        return reward


class ELMTowerDefenseAgent:
    """ELMを使用したTower Defenseエージェント"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 100):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        # ELMパラメータ
        self.input_weights = np.random.randn(state_size, hidden_size) * 0.1
        self.biases = np.random.randn(hidden_size) * 0.1
        self.output_weights = np.zeros((hidden_size, action_size))
        
        # 経験バッファ
        self.experience_buffer = []
        self.buffer_size = 1000
        
        # 学習パラメータ
        self.learning_rate = 0.01
        self.epsilon = 0.1  # 探索率
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # 統計情報
        self.total_reward = 0
        self.episode_count = 0
        self.training_time = 0
        self.inference_time = 0
    
    def _activation(self, x: np.ndarray) -> np.ndarray:
        """活性化関数（tanh）"""
        return np.tanh(x)
    
    def _forward(self, state: np.ndarray) -> np.ndarray:
        """順伝播"""
        hidden = self._activation(np.dot(state, self.input_weights) + self.biases)
        output = np.dot(hidden, self.output_weights)
        return output, hidden
    
    def predict(self, state: np.ndarray) -> int:
        """行動を予測"""
        start_time = time.time()
        
        q_values, _ = self._forward(state)
        
        # ε-greedy探索
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.action_size)
        else:
            action = np.argmax(q_values)
        
        self.inference_time += time.time() - start_time
        return action
    
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """経験を記録"""
        experience = (state, action, reward, next_state, done)
        self.experience_buffer.append(experience)
        
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)
    
    def learn(self):
        """ELMで学習"""
        if len(self.experience_buffer) < 32:
            return
        
        start_time = time.time()
        
        # バッチサンプリング
        batch_size = min(32, len(self.experience_buffer))
        batch = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        
        states = []
        targets = []
        
        for i in batch:
            state, action, reward, next_state, done = self.experience_buffer[i]
            
            # Q値を計算
            q_values, hidden = self._forward(state)
            target = q_values.copy()
            
            if done:
                target[action] = reward
            else:
                next_q_values, _ = self._forward(next_state)
                target[action] = reward + 0.95 * np.max(next_q_values)  # γ=0.95
            
            states.append(hidden)
            targets.append(target)
        
        # ELM学習（最小二乗法）
        H = np.array(states)
        T = np.array(targets)
        
        try:
            # 正則化項を追加して数値安定性を向上
            HTH = np.dot(H.T, H) + np.eye(self.hidden_size) * 1e-6
            HTH_inv = np.linalg.inv(HTH)
            self.output_weights = np.dot(np.dot(HTH_inv, H.T), T)
        except np.linalg.LinAlgError:
            # 特異行列の場合は疑似逆行列を使用
            self.output_weights = np.dot(np.linalg.pinv(H), T)
        
        # 探索率を減衰
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.training_time += time.time() - start_time
    
    def get_stats(self) -> Dict:
        """統計情報を取得"""
        return {
            'total_reward': self.total_reward,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'avg_training_time': self.training_time / max(1, self.episode_count),
            'avg_inference_time': self.inference_time / max(1, self.episode_count),
            'experience_buffer_size': len(self.experience_buffer)
        }


def run_elm_experiment(episodes: int = 30) -> Dict:
    """ELMのみの実験を実行"""
    env = TowerDefenseEnvironment()
    state_size = len(env.get_state())
    action_size = env.get_action_space_size()
    
    agent = ELMTowerDefenseAgent(state_size, action_size)
    
    results = {
        'scores': [],
        'survival_times': [],
        'towers_built': [],
        'enemies_killed': [],
        'final_health': [],
        'efficiency': []
    }
    
    print(f"ELM Tower Defense実験開始 - {episodes}エピソード")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.predict(state)
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
    results['agent_stats'] = agent.get_stats()
    results['final_avg_score'] = np.mean(results['scores'][-5:])  # 最後5エピソードの平均
    results['final_avg_efficiency'] = np.mean(results['efficiency'][-5:])
    
    print(f"ELM実験完了 - 平均スコア: {results['final_avg_score']:.2f}")
    
    return results


if __name__ == "__main__":
    # テスト実行
    results = run_elm_experiment(10)
    print(json.dumps(results, indent=2, default=str))
