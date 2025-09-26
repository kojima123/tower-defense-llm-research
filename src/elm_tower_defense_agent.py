"""
ELM Tower Defense エージェント（実験ランナー用）
elm_tower_defense.pyから抽出・独立化
"""

import numpy as np
from typing import Optional, Dict, Any


class ELMTowerDefenseAgent:
    """ELM（Extreme Learning Machine）を使用したタワーディフェンスエージェント"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: Optional[int] = None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 個別RNG生成器（グローバルシードに依存しない）
        self.rng = np.random.default_rng(seed)
        
        # ELMの重み（入力層→隠れ層）
        self.input_weights = self.rng.normal(0, 0.5, (input_size, hidden_size))
        self.biases = self.rng.normal(0, 0.5, hidden_size)
        
        # 出力重み（隠れ層→出力層）- 学習で更新される
        self.output_weights = self.rng.normal(0, 0.1, (hidden_size, output_size))
        
        # 学習用データ蓄積
        self.training_data = []
        self.training_targets = []
        
        # 統計情報
        self.total_actions = 0
        self.successful_actions = 0
        
    def _activation(self, x: np.ndarray) -> np.ndarray:
        """活性化関数（シグモイド）"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def _forward(self, state: np.ndarray) -> np.ndarray:
        """順伝播"""
        # 隠れ層の計算
        hidden = self._activation(np.dot(state, self.input_weights) + self.biases)
        
        # 出力層の計算
        output = np.dot(hidden, self.output_weights)
        
        return output, hidden
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """状態から行動値を予測"""
        output, _ = self._forward(state)
        return output
    
    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """ε-greedy行動選択"""
        self.total_actions += 1
        
        if self.rng.random() < epsilon:
            # ランダム行動
            return self.rng.integers(0, self.output_size)
        else:
            # 最適行動
            q_values = self.predict(state)
            return np.argmax(q_values)
    
    def add_experience(self, state: np.ndarray, action: int, reward: float, 
                      next_state: np.ndarray, done: bool):
        """経験を蓄積"""
        # Q学習のターゲット計算
        if done:
            target = reward
        else:
            next_q_values = self.predict(next_state)
            target = reward + 0.95 * np.max(next_q_values)  # γ=0.95
        
        # ターゲットベクトル作成
        current_q_values = self.predict(state)
        target_vector = current_q_values.copy()
        target_vector[action] = target
        
        self.training_data.append(state)
        self.training_targets.append(target_vector)
        
        # バッチサイズ制限
        if len(self.training_data) > 1000:
            self.training_data = self.training_data[-1000:]
            self.training_targets = self.training_targets[-1000:]
    
    def train(self, regularization: float = 1e-6):
        """ELM学習（最小二乗法）"""
        if len(self.training_data) < 10:
            return
        
        # データ準備
        X = np.array(self.training_data)
        Y = np.array(self.training_targets)
        
        # 隠れ層の出力計算
        H = self._activation(np.dot(X, self.input_weights) + self.biases)
        
        # 最小二乗法で出力重みを更新
        try:
            # 正則化項付き最小二乗法
            HTH = np.dot(H.T, H)
            HTY = np.dot(H.T, Y)
            
            # 正則化
            I = np.eye(HTH.shape[0])
            self.output_weights = np.linalg.solve(HTH + regularization * I, HTY)
            
        except np.linalg.LinAlgError:
            # 特異行列の場合は擬似逆行列を使用
            H_pinv = np.linalg.pinv(H)
            self.output_weights = np.dot(H_pinv, Y)
    
    def get_stats(self) -> Dict[str, Any]:
        """エージェントの統計情報を取得"""
        success_rate = self.successful_actions / max(1, self.total_actions)
        return {
            'total_actions': self.total_actions,
            'successful_actions': self.successful_actions,
            'success_rate': success_rate,
            'training_samples': len(self.training_data),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size
        }
    
    def reset_stats(self):
        """統計情報をリセット"""
        self.total_actions = 0
        self.successful_actions = 0
    
    def save_model(self, filepath: str):
        """モデルを保存"""
        model_data = {
            'input_weights': self.input_weights.tolist(),
            'biases': self.biases.tolist(),
            'output_weights': self.output_weights.tolist(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'stats': self.get_stats()
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, filepath: str):
        """モデルを読み込み"""
        import json
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.input_weights = np.array(model_data['input_weights'])
        self.biases = np.array(model_data['biases'])
        self.output_weights = np.array(model_data['output_weights'])
        self.input_size = model_data['input_size']
        self.hidden_size = model_data['hidden_size']
        self.output_size = model_data['output_size']


class RuleBasedTeacher:
    """ルールベース教師エージェント"""
    
    def __init__(self):
        self.name = "rule_teacher"
    
    def get_action(self, env) -> int:
        """ルールベースの行動決定"""
        info = env.get_game_info()
        
        # 基本戦略：
        # 1. 資金が十分あり、敵が多い場合はタワー配置
        # 2. パスの前半部分を優先
        # 3. 既存タワーの近くに配置して集中攻撃
        
        if info['money'] < env.TOWER_COST:
            return len(env.valid_positions)  # 何もしない
        
        if len(env.enemies) < 2 and info['wave'] < 5:
            return len(env.valid_positions)  # 序盤は様子見
        
        # 優先配置位置を計算
        best_position = self._find_best_position(env)
        
        if best_position is not None:
            return best_position
        else:
            return len(env.valid_positions)  # 何もしない
    
    def _find_best_position(self, env) -> Optional[int]:
        """最適な配置位置を探す"""
        scores = []
        
        for i, pos in enumerate(env.valid_positions):
            # 既にタワーがある位置はスキップ
            occupied = any(abs(tower['x'] - pos[0]) < 30 and abs(tower['y'] - pos[1]) < 30 
                          for tower in env.towers)
            if occupied:
                scores.append(-1)
                continue
            
            score = 0
            
            # パスとの距離（近いほど良い）
            min_path_dist = float('inf')
            for j in range(len(env.path) - 1):
                p1 = env.path[j]
                p2 = env.path[j + 1]
                dist = env._distance_to_line(pos[0], pos[1], p1[0], p1[1], p2[0], p2[1])
                min_path_dist = min(min_path_dist, dist)
            
            if min_path_dist < env.TOWER_RANGE:
                score += (env.TOWER_RANGE - min_path_dist) / env.TOWER_RANGE * 100
            
            # 既存タワーとの距離（適度に近いほど良い）
            if env.towers:
                min_tower_dist = min(
                    np.sqrt((tower['x'] - pos[0])**2 + (tower['y'] - pos[1])**2)
                    for tower in env.towers
                )
                if 50 < min_tower_dist < 150:
                    score += 50
            
            # パスの前半部分を優先
            if pos[0] < 400:
                score += 30
            
            scores.append(score)
        
        if not scores or max(scores) <= 0:
            return None
        
        return np.argmax(scores)


class RandomTeacher:
    """ランダム教師エージェント"""
    
    def __init__(self, seed: Optional[int] = None):
        self.name = "random_teacher"
        # 個別RNG生成器（グローバルシードに依存しない）
        self.rng = np.random.default_rng(seed)
    
    def get_action(self, env) -> int:
        """ランダムな行動決定"""
        info = env.get_game_info()
        
        # 資金不足の場合は何もしない
        if info['money'] < env.TOWER_COST:
            return len(env.valid_positions)
        
        # 70%の確率でタワー配置、30%で待機
        if self.rng.random() < 0.7:
            # 利用可能な位置からランダム選択
            available_positions = []
            for i, pos in enumerate(env.valid_positions):
                occupied = any(abs(tower['x'] - pos[0]) < 30 and abs(tower['y'] - pos[1]) < 30 
                              for tower in env.towers)
                if not occupied:
                    available_positions.append(i)
            
            if available_positions:
                return self.rng.choice(available_positions)
        
        return len(env.valid_positions)  # 何もしない
