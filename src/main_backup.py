#!/usr/bin/env python3
"""
Flask server for Tower Defense LLM Trainer - Simple Deployment Version
Handles rule-based guidance without external dependencies
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import time
import random
import math

app = Flask(__name__, static_folder='../static', static_url_path='')
CORS(app)

# Get the absolute path to the project directory
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(PROJECT_DIR, 'static')

# Simple ELM implementation without external dependencies
class SimpleTowerDefenseELM:
    def __init__(self, input_size=8, hidden_size=20, output_size=2, random_state=42):
        """
        Simple ELM for Tower Defense strategy
        """
        random.seed(random_state)
        
        # Initialize weights with random values
        self.input_weights = []
        for i in range(input_size):
            row = []
            for j in range(hidden_size):
                row.append(random.gauss(0, 0.5))
            self.input_weights.append(row)
        
        self.hidden_bias = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        
        self.output_weights = []
        for i in range(hidden_size):
            row = []
            for j in range(output_size):
                row.append(random.gauss(0, 0.1))
            self.output_weights.append(row)
        
        self.learning_rate = 0.02
        
    def sigmoid(self, x):
        try:
            return 1 / (1 + math.exp(-max(-500, min(500, x))))
        except:
            return 0.5
    
    def tanh(self, x):
        try:
            return math.tanh(max(-500, min(500, x)))
        except:
            return 0
    
    def predict(self, x):
        # Normalize inputs
        x_norm = []
        for val in x:
            if abs(val) > 1e-8:
                x_norm.append(val / abs(val))
            else:
                x_norm.append(val)
        
        # Forward pass
        hidden = []
        for j in range(len(self.hidden_bias)):
            sum_val = self.hidden_bias[j]
            for i in range(len(x_norm)):
                sum_val += x_norm[i] * self.input_weights[i][j]
            hidden.append(self.tanh(sum_val))
        
        # Output layer
        output = []
        for j in range(len(self.output_weights[0])):
            sum_val = 0
            for i in range(len(hidden)):
                sum_val += hidden[i] * self.output_weights[i][j]
            output.append(sum_val)
        
        # Apply activations
        output[0] = self.sigmoid(output[0])
        output[1] = self.sigmoid(output[1])
        
        return output
    
    def update(self, x, target, learning_rate=None):
        if learning_rate is None:
            learning_rate = self.learning_rate
        
        # Simple weight update (simplified)
        prediction = self.predict(x)
        error = [target[i] - prediction[i] for i in range(len(target))]
        
        # Update output weights
        for i in range(len(self.output_weights)):
            for j in range(len(self.output_weights[i])):
                self.output_weights[i][j] += learning_rate * error[j] * 0.1

# Global model instances
baseline_elm = SimpleTowerDefenseELM(random_state=42)
llm_guided_elm = SimpleTowerDefenseELM(random_state=43)

@app.route('/')
def index():
    """Serve the main game page"""
    # Try multiple paths to find the static HTML file
    possible_paths = [
        os.path.join(STATIC_DIR, 'index.html'),
        os.path.join(PROJECT_DIR, 'static.html'),
        os.path.join(PROJECT_DIR, 'static', 'index.html')
    ]
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            print(f"Failed to read {path}: {e}")
            continue
    
    # Fallback HTML if no static file is found
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tower Defense LLM Trainer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            color: white;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5rem;
            background: linear-gradient(45deg, #4f46e5, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        .header p {
            color: #9ca3af;
            font-size: 1.1rem;
        }
        
        .game-layout {
            display: grid;
            grid-template-columns: 1fr 300px;
            gap: 20px;
            align-items: start;
        }
        
        .game-area {
            background: rgba(31, 41, 55, 0.8);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #374151;
        }
        
        .canvas-container {
            position: relative;
            display: inline-block;
        }
        
        #gameCanvas {
            border: 2px solid #4b5563;
            border-radius: 8px;
            background: #111827;
            cursor: crosshair;
        }
        
        .controls {
            margin-top: 15px;
            display: flex;
            gap: 10px;
        }
        
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .btn-primary {
            background: #10b981;
            color: white;
        }
        
        .btn-primary:hover {
            background: #059669;
        }
        
        .btn-danger {
            background: #ef4444;
            color: white;
        }
        
        .btn-danger:hover {
            background: #dc2626;
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .panel {
            background: rgba(31, 41, 55, 0.8);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #374151;
        }
        
        .panel h3 {
            margin-bottom: 15px;
            color: #f3f4f6;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .stat-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            padding: 5px 0;
        }
        
        .stat-label {
            color: #9ca3af;
        }
        
        .stat-value {
            font-weight: 600;
            color: #f3f4f6;
        }
        
        .llm-guidance {
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid #3b82f6;
            border-radius: 8px;
            padding: 15px;
            margin-top: 10px;
        }
        
        .priority-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 8px;
        }
        
        .priority-urgent {
            background: #ef4444;
            color: white;
        }
        
        .priority-high {
            background: #f59e0b;
            color: white;
        }
        
        .priority-medium {
            background: #3b82f6;
            color: white;
        }
        
        .priority-low {
            background: #6b7280;
            color: white;
        }
        
        .mode-buttons {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .mode-btn {
            padding: 8px 12px;
            border: 1px solid #4b5563;
            background: rgba(75, 85, 99, 0.3);
            color: #d1d5db;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 0.9rem;
        }
        
        .mode-btn.active {
            background: #3b82f6;
            border-color: #3b82f6;
            color: white;
        }
        
        .mode-btn:hover {
            background: rgba(59, 130, 246, 0.2);
        }
        
        .toggle-switch {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
        }
        
        .switch {
            position: relative;
            width: 50px;
            height: 24px;
            background: #4b5563;
            border-radius: 12px;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .switch.active {
            background: #10b981;
        }
        
        .switch-handle {
            position: absolute;
            top: 2px;
            left: 2px;
            width: 20px;
            height: 20px;
            background: white;
            border-radius: 50%;
            transition: transform 0.2s;
        }
        
        .switch.active .switch-handle {
            transform: translateX(26px);
        }
        
        @media (max-width: 1200px) {
            .game-layout {
                grid-template-columns: 1fr;
            }
            
            .sidebar {
                grid-row: 1;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Tower Defense LLM Trainer</h1>
            <p>LLM指導型学習システムでタワーディフェンスの戦略を最適化</p>
        </div>
        
        <div class="game-layout">
            <div class="game-area">
                <h3>🎮 ゲーム画面</h3>
                <div class="canvas-container">
                    <canvas id="gameCanvas" width="800" height="600"></canvas>
                </div>
                <div class="controls">
                    <button id="startBtn" class="btn btn-primary">ゲーム開始</button>
                    <button id="stopBtn" class="btn btn-danger" disabled>ゲーム停止</button>
                </div>
            </div>
            
            <div class="sidebar">
                <div class="panel">
                    <h3>📊 ゲーム状況</h3>
                    <div class="stat-row">
                        <span class="stat-label">💰 資金</span>
                        <span class="stat-value" id="money">$250</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">❤️ ヘルス</span>
                        <span class="stat-value" id="health">100</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">⚡ ウェーブ</span>
                        <span class="stat-value" id="wave">1</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">🎯 スコア</span>
                        <span class="stat-value" id="score">0</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">🏗️ タワー数</span>
                        <span class="stat-value" id="towers">0</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">👹 敵数</span>
                        <span class="stat-value" id="enemies">0</span>
                    </div>
                </div>
                
                <div class="panel">
                    <h3>🤖 戦略指導システム</h3>
                    <div class="toggle-switch">
                        <div class="switch" id="llmToggle">
                            <div class="switch-handle"></div>
                        </div>
                        <span>戦略指導システム</span>
                    </div>
                    <div id="llmGuidance" class="llm-guidance" style="display: none;">
                        <div class="priority-badge priority-medium" id="priorityBadge">中程度</div>
                        <p id="recommendation">戦略を分析中...</p>
                        <p style="margin-top: 8px; font-size: 0.9rem; color: #9ca3af;">
                            <strong>理由:</strong> <span id="reasoning">システムを初期化しています</span>
                        </p>
                    </div>
                </div>
                
                <div class="panel">
                    <h3>⚙️ 実験制御</h3>
                    <div class="mode-buttons">
                        <button class="mode-btn active" data-mode="manual">🎮 手動プレイ</button>
                        <button class="mode-btn" data-mode="elm">🤖 ELMのみ</button>
                        <button class="mode-btn" data-mode="elm_llm">🧠 ELM+指導システム</button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Game configuration
        const GAME_CONFIG = {
            CANVAS_WIDTH: 800,
            CANVAS_HEIGHT: 600,
            TOWER_COST: 50,
            TOWER_DAMAGE: 60,
            TOWER_RANGE: 150,
            ENEMY_HEALTH: 80,
            ENEMY_SPEED: 0.7,
            ENEMY_REWARD: 30,
            WAVE_SIZE: 3,
            INITIAL_MONEY: 250,
            INITIAL_HEALTH: 100
        };

        // Game path
        const PATH = [
            { x: 0, y: 300 },
            { x: 200, y: 300 },
            { x: 200, y: 150 },
            { x: 400, y: 150 },
            { x: 400, y: 450 },
            { x: 600, y: 450 },
            { x: 600, y: 300 },
            { x: 800, y: 300 }
        ];

        // Game state
        class GameState {
            constructor() {
                this.reset();
            }

            reset() {
                this.money = GAME_CONFIG.INITIAL_MONEY;
                this.health = GAME_CONFIG.INITIAL_HEALTH;
                this.wave = 1;
                this.score = 0;
                this.towers = [];
                this.enemies = [];
                this.projectiles = [];
                this.gameTime = 0;
                this.waveTimer = 0;
                this.isGameRunning = false;
                this.gameOver = false;
            }

            addTower(x, y) {
                if (this.money >= GAME_CONFIG.TOWER_COST && this.canPlaceTower(x, y)) {
                    this.towers.push({
                        x: x,
                        y: y,
                        damage: GAME_CONFIG.TOWER_DAMAGE,
                        range: GAME_CONFIG.TOWER_RANGE,
                        lastShot: 0,
                        kills: 0
                    });
                    this.money -= GAME_CONFIG.TOWER_COST;
                    return true;
                }
                return false;
            }

            canPlaceTower(x, y) {
                // Check path collision
                for (let i = 0; i < PATH.length - 1; i++) {
                    const p1 = PATH[i];
                    const p2 = PATH[i + 1];
                    const dist = this.distanceToLine(x, y, p1.x, p1.y, p2.x, p2.y);
                    if (dist < 30) return false;
                }
                
                // Check tower collision
                for (const tower of this.towers) {
                    if (Math.hypot(x - tower.x, y - tower.y) < 40) return false;
                }
                
                return true;
            }

            distanceToLine(px, py, x1, y1, x2, y2) {
                const A = px - x1;
                const B = py - y1;
                const C = x2 - x1;
                const D = y2 - y1;
                
                const dot = A * C + B * D;
                const lenSq = C * C + D * D;
                let param = -1;
                if (lenSq !== 0) param = dot / lenSq;
                
                let xx, yy;
                if (param < 0) {
                    xx = x1;
                    yy = y1;
                } else if (param > 1) {
                    xx = x2;
                    yy = y2;
                } else {
                    xx = x1 + param * C;
                    yy = y1 + param * D;
                }
                
                const dx = px - xx;
                const dy = py - yy;
                return Math.sqrt(dx * dx + dy * dy);
            }

            spawnWave() {
                for (let i = 0; i < GAME_CONFIG.WAVE_SIZE; i++) {
                    setTimeout(() => {
                        this.enemies.push({
                            x: PATH[0].x,
                            y: PATH[0].y,
                            health: GAME_CONFIG.ENEMY_HEALTH * (1 + this.wave * 0.2),
                            maxHealth: GAME_CONFIG.ENEMY_HEALTH * (1 + this.wave * 0.2),
                            speed: GAME_CONFIG.ENEMY_SPEED * (1 + this.wave * 0.1),
                            pathIndex: 0,
                            progress: 0
                        });
                    }, i * 1000);
                }
            }

            update() {
                if (!this.isGameRunning || this.gameOver) return;

                this.gameTime += 1/60;
                this.waveTimer += 1/60;

                // Spawn new wave
                if (this.enemies.length === 0 && this.waveTimer > 3) {
                    this.wave++;
                    this.spawnWave();
                    this.waveTimer = 0;
                }

                // Update enemies
                this.enemies.forEach((enemy, enemyIndex) => {
                    if (enemy.pathIndex < PATH.length - 1) {
                        const current = PATH[enemy.pathIndex];
                        const next = PATH[enemy.pathIndex + 1];
                        const dx = next.x - current.x;
                        const dy = next.y - current.y;
                        const distance = Math.hypot(dx, dy);
                        
                        enemy.progress += enemy.speed / distance;
                        
                        if (enemy.progress >= 1) {
                            enemy.pathIndex++;
                            enemy.progress = 0;
                        }
                        
                        if (enemy.pathIndex < PATH.length - 1) {
                            const currentPath = PATH[enemy.pathIndex];
                            const nextPath = PATH[enemy.pathIndex + 1];
                            enemy.x = currentPath.x + (nextPath.x - currentPath.x) * enemy.progress;
                            enemy.y = currentPath.y + (nextPath.y - currentPath.y) * enemy.progress;
                        }
                    } else {
                        // Enemy reached goal
                        this.health -= 10;
                        this.enemies.splice(enemyIndex, 1);
                        if (this.health <= 0) {
                            this.gameOver = true;
                        }
                    }
                });

                // Tower attacks
                this.towers.forEach(tower => {
                    if (this.gameTime - tower.lastShot > 0.5) {
                        const target = this.findNearestEnemy(tower);
                        if (target) {
                            this.projectiles.push({
                                x: tower.x,
                                y: tower.y,
                                targetX: target.x,
                                targetY: target.y,
                                damage: tower.damage,
                                speed: 5
                            });
                            tower.lastShot = this.gameTime;
                        }
                    }
                });

                // Update projectiles
                this.projectiles.forEach((projectile, projIndex) => {
                    const dx = projectile.targetX - projectile.x;
                    const dy = projectile.targetY - projectile.y;
                    const distance = Math.hypot(dx, dy);
                    
                    if (distance < projectile.speed) {
                        const target = this.enemies.find(enemy => 
                            Math.hypot(enemy.x - projectile.targetX, enemy.y - projectile.targetY) < 20
                        );
                        if (target) {
                            target.health -= projectile.damage;
                            if (target.health <= 0) {
                                this.money += GAME_CONFIG.ENEMY_REWARD;
                                this.score += 100;
                                const enemyIndex = this.enemies.indexOf(target);
                                if (enemyIndex > -1) this.enemies.splice(enemyIndex, 1);
                            }
                        }
                        this.projectiles.splice(projIndex, 1);
                    } else {
                        projectile.x += (dx / distance) * projectile.speed;
                        projectile.y += (dy / distance) * projectile.speed;
                    }
                });
            }

            findNearestEnemy(tower) {
                let nearest = null;
                let minDistance = tower.range;
                
                this.enemies.forEach(enemy => {
                    const distance = Math.hypot(enemy.x - tower.x, enemy.y - tower.y);
                    if (distance < minDistance) {
                        minDistance = distance;
                        nearest = enemy;
                    }
                });
                
                return nearest;
            }

            getState() {
                return {
                    money: this.money,
                    health: this.health,
                    wave: this.wave,
                    score: this.score,
                    towers: this.towers.length,
                    enemies: this.enemies.length,
                    gameTime: this.gameTime,
                    efficiency: this.towers.length > 0 ? this.score / (this.towers.length * GAME_CONFIG.TOWER_COST) : 0,
                    survival: this.health / GAME_CONFIG.INITIAL_HEALTH,
                    progress: this.wave
                };
            }
        }

        // Game instance
        const game = new GameState();
        let animationId = null;
        let isGuidanceEnabled = false;
        let gameMode = 'manual';

        // Canvas and context
        const canvas = document.getElementById('gameCanvas');
        const ctx = canvas.getContext('2d');

        // UI elements
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const guidanceToggle = document.getElementById('llmToggle');
        const guidancePanel = document.getElementById('llmGuidance');
        const modeButtons = document.querySelectorAll('.mode-btn');

        // Game loop
        function gameLoop() {
            game.update();
            draw();
            updateUI();
            
            if (game.isGameRunning && !game.gameOver) {
                animationId = requestAnimationFrame(gameLoop);
            } else if (game.gameOver) {
                alert(`ゲームオーバー！スコア: ${game.score}`);
                stopGame();
            }
        }

        // Drawing function
        function draw() {
            // Clear canvas
            ctx.fillStyle = '#1a1a1a';
            ctx.fillRect(0, 0, GAME_CONFIG.CANVAS_WIDTH, GAME_CONFIG.CANVAS_HEIGHT);
            
            // Draw path
            ctx.strokeStyle = '#444';
            ctx.lineWidth = 30;
            ctx.beginPath();
            ctx.moveTo(PATH[0].x, PATH[0].y);
            for (let i = 1; i < PATH.length; i++) {
                ctx.lineTo(PATH[i].x, PATH[i].y);
            }
            ctx.stroke();
            
            // Draw towers
            game.towers.forEach(tower => {
                // Tower range
                ctx.strokeStyle = '#333';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.arc(tower.x, tower.y, tower.range, 0, Math.PI * 2);
                ctx.stroke();
                
                // Tower body
                ctx.fillStyle = '#4ade80';
                ctx.beginPath();
                ctx.arc(tower.x, tower.y, 15, 0, Math.PI * 2);
                ctx.fill();
            });
            
            // Draw enemies
            game.enemies.forEach(enemy => {
                // Enemy body
                ctx.fillStyle = '#ef4444';
                ctx.beginPath();
                ctx.arc(enemy.x, enemy.y, 12, 0, Math.PI * 2);
                ctx.fill();
                
                // Health bar
                const healthPercent = enemy.health / enemy.maxHealth;
                ctx.fillStyle = '#333';
                ctx.fillRect(enemy.x - 15, enemy.y - 20, 30, 4);
                ctx.fillStyle = healthPercent > 0.5 ? '#4ade80' : healthPercent > 0.25 ? '#fbbf24' : '#ef4444';
                ctx.fillRect(enemy.x - 15, enemy.y - 20, 30 * healthPercent, 4);
            });
            
            // Draw projectiles
            game.projectiles.forEach(projectile => {
                ctx.fillStyle = '#fbbf24';
                ctx.beginPath();
                ctx.arc(projectile.x, projectile.y, 3, 0, Math.PI * 2);
                ctx.fill();
            });
        }

        // Update UI
        function updateUI() {
            const state = game.getState();
            document.getElementById('money').textContent = `$${state.money}`;
            document.getElementById('health').textContent = state.health;
            document.getElementById('wave').textContent = state.wave;
            document.getElementById('score').textContent = state.score;
            document.getElementById('towers').textContent = state.towers;
            document.getElementById('enemies').textContent = state.enemies;
        }

        // Start game
        function startGame() {
            game.reset();
            game.isGameRunning = true;
            game.spawnWave();
            startBtn.disabled = true;
            stopBtn.disabled = false;
            gameLoop();
        }

        // Stop game
        function stopGame() {
            game.isGameRunning = false;
            startBtn.disabled = false;
            stopBtn.disabled = true;
            if (animationId) {
                cancelAnimationFrame(animationId);
            }
        }

        // Canvas click handler
        canvas.addEventListener('click', (event) => {
            if (gameMode !== 'manual' || !game.isGameRunning) return;
            
            const rect = canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            
            game.addTower(x, y);
        });

        // Event listeners
        startBtn.addEventListener('click', startGame);
        stopBtn.addEventListener('click', stopGame);

        guidanceToggle.addEventListener('click', () => {
            isGuidanceEnabled = !isGuidanceEnabled;
            guidanceToggle.classList.toggle('active');
            guidancePanel.style.display = isGuidanceEnabled ? 'block' : 'none';
        });

        modeButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                modeButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                gameMode = btn.dataset.mode;
            });
        });

        // Guidance system
        setInterval(() => {
            if (!isGuidanceEnabled || !game.isGameRunning) return;
            
            const state = game.getState();
            
            // Get guidance from server
            fetch('/api/llm-guidance', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    game_state: state
                })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('recommendation').textContent = data.recommendation;
                document.getElementById('reasoning').textContent = data.reasoning;
                
                const badge = document.getElementById('priorityBadge');
                badge.className = `priority-badge priority-${data.priority}`;
                badge.textContent = data.priority === 'urgent' ? '緊急' :
                                   data.priority === 'high' ? '重要' :
                                   data.priority === 'medium' ? '中程度' : '低';
            })
            .catch(error => {
                console.error('Guidance request failed:', error);
            });
        }, 3000);

        // Initial draw
        draw();
        updateUI();
    </script>
</body>
</html>
    """

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'openai_available': False,
        'timestamp': time.time(),
        'project_dir': PROJECT_DIR,
        'static_dir': STATIC_DIR
    })

@app.route('/api/llm-guidance', methods=['POST'])
def get_llm_guidance():
    """Get strategic guidance based on current game state"""
    try:
        data = request.json
        game_state = data['game_state']
        
        # Rule-based guidance system
        return get_rule_based_guidance(game_state)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_rule_based_guidance(game_state):
    """Rule-based guidance system"""
    money = game_state['money']
    health = game_state['health']
    wave = game_state['wave']
    enemies = game_state['enemies']
    towers = game_state['towers']
    
    if health < 30:
        return jsonify({
            'recommendation': '緊急でタワーを配置してください！',
            'reasoning': 'ヘルスが危険な状態です',
            'priority': 'urgent',
            'source': 'rule_based'
        })
    elif money >= 100 and towers < wave:
        return jsonify({
            'recommendation': 'タワーを追加配置しましょう',
            'reasoning': '十分な資金があり、ウェーブに対してタワーが不足しています',
            'priority': 'high',
            'source': 'rule_based'
        })
    elif enemies > 5 and towers < 3:
        return jsonify({
            'recommendation': 'タワーを配置して防御を強化してください',
            'reasoning': '敵の数が多く、防御が不十分です',
            'priority': 'high',
            'source': 'rule_based'
        })
    elif money >= 50:
        return jsonify({
            'recommendation': 'タワーを配置して防御を拡張しましょう',
            'reasoning': '資金に余裕があります',
            'priority': 'medium',
            'source': 'rule_based'
        })
    else:
        return jsonify({
            'recommendation': '現在の戦略を継続してください',
            'reasoning': '良好な状態を維持しています',
            'priority': 'low',
            'source': 'rule_based'
        })

@app.route('/api/elm-predict', methods=['POST'])
def elm_predict():
    """Get ELM prediction for tower placement"""
    try:
        data = request.json
        game_state = data['game_state']
        model_type = data.get('model_type', 'baseline')
        
        # Prepare input features
        features = [
            game_state['money'] / 1000.0,  # Normalize money
            game_state['health'] / 100.0,  # Normalize health
            game_state['wave'] / 10.0,     # Normalize wave
            game_state['enemies'] / 10.0,  # Normalize enemies
            game_state['towers'] / 10.0,   # Normalize towers
            game_state.get('efficiency', 0),
            game_state.get('survival', 1),
            game_state.get('progress', 1) / 10.0
        ]
        
        if model_type == 'llm_guided':
            prediction = llm_guided_elm.predict(features)
        else:
            prediction = baseline_elm.predict(features)
        
        # Convert prediction to actionable format
        should_place_tower = prediction[0] > 0.5
        tower_position_ratio = prediction[1]
        
        return jsonify({
            'should_place_tower': bool(should_place_tower),
            'placement_probability': float(prediction[0]),
            'position_ratio': float(tower_position_ratio),
            'confidence': float(abs(prediction[0] - 0.5) * 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/elm-update', methods=['POST'])
def elm_update():
    """Update ELM model based on performance feedback"""
    try:
        data = request.json
        game_state = data['game_state']
        action_taken = data['action_taken']
        performance_score = data['performance_score']
        model_type = data.get('model_type', 'llm_guided')
        
        # Prepare input features
        features = [
            game_state['money'] / 1000.0,
            game_state['health'] / 100.0,
            game_state['wave'] / 10.0,
            game_state['enemies'] / 10.0,
            game_state['towers'] / 10.0,
            game_state.get('efficiency', 0),
            game_state.get('survival', 1),
            game_state.get('progress', 1) / 10.0
        ]
        
        # Create target based on performance
        target = [
            performance_score if action_taken['placed_tower'] else 1 - performance_score,
            action_taken.get('position_ratio', 0.5)
        ]
        
        # Adaptive learning rate
        learning_rate = 0.01 * (performance_score + 0.1)
        
        if model_type == 'llm_guided':
            llm_guided_elm.update(features, target, learning_rate)
        else:
            baseline_elm.update(features, target, learning_rate)
        
        return jsonify({
            'status': 'updated',
            'learning_rate': learning_rate,
            'target': target
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/evaluate-performance', methods=['POST'])
def evaluate_performance():
    """Evaluate current game performance"""
    try:
        data = request.json
        game_state = data['game_state']
        previous_state = data.get('previous_state', {})
        
        # Calculate performance metrics
        score_improvement = game_state['score'] - previous_state.get('score', 0)
        health_ratio = game_state['health'] / 100.0
        efficiency = game_state.get('efficiency', 0)
        
        # Composite performance score
        performance = (
            min(score_improvement / 100.0, 1.0) * 0.4 +  # Score improvement
            health_ratio * 0.3 +                          # Health preservation
            min(efficiency, 1.0) * 0.3                    # Tower efficiency
        )
        
        performance = max(0.0, min(1.0, performance))
        
        return jsonify({
            'performance_score': performance,
            'metrics': {
                'score_improvement': score_improvement,
                'health_ratio': health_ratio,
                'efficiency': efficiency
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset-models', methods=['POST'])
def reset_models():
    """Reset both ELM models to initial state"""
    global baseline_elm, llm_guided_elm
    baseline_elm = SimpleTowerDefenseELM(random_state=42)
    llm_guided_elm = SimpleTowerDefenseELM(random_state=43)
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    print("🚀 Starting Tower Defense LLM Trainer Server (Simple Deployment)")
    print(f"📁 Project Directory: {PROJECT_DIR}")
    print(f"📁 Static Directory: {STATIC_DIR}")
    print("🎮 Game available at: http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
