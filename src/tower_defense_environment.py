"""
Tower Defense環境クラス（実験ランナー用）
elm_tower_defense.pyから抽出・独立化
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
    
    def reset(self, seed: Optional[int] = None):
        """環境をリセット"""
        if seed is not None:
            np.random.seed(seed)
        
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
            self.health / 100.0,  # 正規化された体力
            self.wave / 20.0,     # 正規化されたウェーブ
            len(self.towers) / 20.0,  # 正規化されたタワー数
            len(self.enemies) / 10.0,  # 正規化された敵数
            self.score / 10000.0,  # 正規化されたスコア
            self.enemies_killed / 100.0,  # 正規化された撃破数
        ]
        
        # タワー密度マップ（8x6グリッド）
        tower_density = np.zeros((8, 6))
        for tower in self.towers:
            grid_x = min(7, int(tower['x'] / 100))
            grid_y = min(5, int(tower['y'] / 100))
            tower_density[grid_x][grid_y] += 1
        
        features.extend(tower_density.flatten() / 5.0)  # 正規化
        
        return np.array(features, dtype=np.float32)
    
    def get_state_size(self) -> int:
        """状態ベクトルのサイズを取得"""
        return len(self.get_state())
    
    def get_action_size(self) -> int:
        """行動空間のサイズを取得"""
        return len(self.valid_positions) + 1  # タワー配置 + 何もしない
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """1ステップ実行"""
        reward = 0
        info = {}
        
        # 行動実行
        if action < len(self.valid_positions):
            # タワー配置
            pos = self.valid_positions[action]
            if self.money >= self.TOWER_COST:
                self.towers.append({
                    'x': pos[0],
                    'y': pos[1],
                    'damage': self.TOWER_DAMAGE,
                    'range': self.TOWER_RANGE
                })
                self.money -= self.TOWER_COST
                reward += 10  # タワー配置報酬
                info['action_type'] = 'place_tower'
                info['tower_position'] = pos
            else:
                reward -= 5  # 資金不足ペナルティ
                info['action_type'] = 'invalid_placement'
        else:
            # 何もしない
            info['action_type'] = 'wait'
        
        # ゲーム進行
        self._update_game()
        
        # 報酬計算
        reward += self._calculate_reward()
        
        # 終了判定
        done = self.health <= 0 or self.wave > 20
        
        info.update({
            'money': self.money,
            'health': self.health,
            'wave': self.wave,
            'score': self.score,
            'towers': len(self.towers),
            'enemies': len(self.enemies),
            'enemies_killed': self.enemies_killed
        })
        
        return self.get_state(), reward, done, info
    
    def _update_game(self):
        """ゲーム状態を更新"""
        self.game_time += 1
        self.wave_timer += 1
        
        # 敵のスポーン
        if self.wave_timer % 30 == 0 and len(self.enemies) < 10:
            self._spawn_enemy()
        
        # 敵の移動
        self._update_enemies()
        
        # タワーの攻撃
        self._tower_attacks()
        
        # ウェーブ進行
        if len(self.enemies) == 0 and self.wave_timer > 100:
            self.wave += 1
            self.wave_timer = 0
            self.money += 50  # ウェーブクリア報酬
    
    def _spawn_enemy(self):
        """敵をスポーン"""
        enemy_health = self.ENEMY_HEALTH + (self.wave - 1) * 10
        self.enemies.append({
            'x': 0,
            'y': 300,
            'health': enemy_health,
            'max_health': enemy_health,
            'path_index': 0,
            'progress': 0.0
        })
        self.total_enemies_spawned += 1
    
    def _update_enemies(self):
        """敵の移動処理"""
        for enemy in self.enemies[:]:
            # パス上を移動
            if enemy['path_index'] < len(self.path) - 1:
                current_pos = self.path[enemy['path_index']]
                next_pos = self.path[enemy['path_index'] + 1]
                
                enemy['progress'] += self.ENEMY_SPEED / 100.0
                
                if enemy['progress'] >= 1.0:
                    enemy['path_index'] += 1
                    enemy['progress'] = 0.0
                
                # 位置更新
                if enemy['path_index'] < len(self.path) - 1:
                    t = enemy['progress']
                    enemy['x'] = current_pos[0] + t * (next_pos[0] - current_pos[0])
                    enemy['y'] = current_pos[1] + t * (next_pos[1] - current_pos[1])
            else:
                # ゴール到達
                self.health -= 10
                self.enemies.remove(enemy)
    
    def _tower_attacks(self):
        """タワーの攻撃処理"""
        for tower in self.towers:
            # 射程内の敵を探す
            for enemy in self.enemies[:]:
                distance = np.sqrt((tower['x'] - enemy['x'])**2 + (tower['y'] - enemy['y'])**2)
                if distance <= tower['range']:
                    # 攻撃
                    enemy['health'] -= tower['damage']
                    if enemy['health'] <= 0:
                        # 敵撃破
                        self.enemies.remove(enemy)
                        self.money += self.ENEMY_REWARD
                        self.score += 100
                        self.enemies_killed += 1
                    break  # 1ターンに1体のみ攻撃
    
    def _calculate_reward(self) -> float:
        """報酬を計算"""
        reward = 0
        
        # 基本生存報酬
        reward += 1
        
        # 体力維持報酬
        reward += self.health / 100.0
        
        # 効率的な資金使用報酬
        if self.money < 100:
            reward += 2
        
        return reward
    
    def get_game_info(self) -> Dict:
        """ゲーム情報を取得"""
        return {
            'money': self.money,
            'health': self.health,
            'wave': self.wave,
            'score': self.score,
            'towers': len(self.towers),
            'enemies': len(self.enemies),
            'enemies_killed': self.enemies_killed,
            'total_spawned': self.total_enemies_spawned
        }
    
    def render_text(self) -> str:
        """テキスト形式でゲーム状態を表示"""
        return (f"Wave: {self.wave}, Money: {self.money}, Health: {self.health}, "
                f"Score: {self.score}, Towers: {len(self.towers)}, "
                f"Enemies: {len(self.enemies)}, Killed: {self.enemies_killed}")
