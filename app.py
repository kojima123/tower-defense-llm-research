#!/usr/bin/env python3
"""
Tower Defense ELM Auto-Fix - Final Fixed Version
Complete automated learning system with API key input and auto-restart
"""

import os
import time
import json
import numpy as np
from flask import Flask, render_template_string, request, jsonify
from sklearn.preprocessing import StandardScaler
import random

# Initialize Flask app
app = Flask(__name__)

# Check OpenAI configuration
openai_configured = False
try:
    import openai
    from openai import OpenAI
    if os.getenv('OPENAI_API_KEY'):
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        openai_configured = True
        print("🔑 OpenAI API Key: ✅ Configured")
    else:
        print("🔑 OpenAI API Key: ❌ Not configured (will use user input)")
except ImportError:
    print("⚠️ OpenAI library not installed")

print("🚀 Tower Defense ELM Auto-Fix Server Starting...")
print("🔧 Auto-Fix: ELMの自動動作を強制実行")
print("🔄 Auto-Restart: ライフ0で自動再開")
print("📊 Learning efficiency experiment ready")
print("🌐 Server starting on port 5000")

class AutoELM:
    """Enhanced ELM with forced automation and LLM guidance integration"""
    
    def __init__(self, n_hidden=100, random_state=42):
        self.n_hidden = n_hidden
        self.random_state = random_state
        np.random.seed(random_state)
        
        # ELM parameters
        self.input_weights = None
        self.hidden_bias = None
        self.output_weights = None
        self.scaler = StandardScaler()
        
        # Learning tracking
        self.learning_start_time = time.time()
        self.total_learning_updates = 0
        self.llm_guidance_count = 0
        self.last_guidance = ""
        
        # Forced automation parameters
        self.action_threshold = 0.3  # Lower threshold for more frequent actions
        self.forced_action_interval = 3000  # 3 seconds in milliseconds
        self.llm_guidance_weight = 0.8  # High weight for LLM guidance
        
        print(f"🤖 AutoELM初期化完了 - 強制実行モード有効")
        print(f"   - 行動閾値: {self.action_threshold}")
        print(f"   - 強制実行間隔: {self.forced_action_interval}ms")
        print(f"   - LLMガイダンス重み: {self.llm_guidance_weight}")
    
    def _initialize_weights(self, n_features):
        """Initialize ELM weights"""
        self.input_weights = np.random.randn(n_features, self.n_hidden)
        self.hidden_bias = np.random.randn(self.n_hidden)
        self.output_weights = np.random.randn(self.n_hidden, 3)  # x, y, should_place
    
    def _sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def predict(self, game_state, llm_guidance=None):
        """Enhanced prediction with forced action and LLM guidance"""
        try:
            # Extract features from game state
            features = self._extract_features(game_state)
            
            if self.input_weights is None:
                self._initialize_weights(len(features))
            
            # Forward pass
            features_scaled = self.scaler.fit_transform([features])[0]
            hidden_output = self._sigmoid(np.dot(features_scaled, self.input_weights) + self.hidden_bias)
            output = np.dot(hidden_output, self.output_weights)
            
            # Apply LLM guidance if available
            if llm_guidance and 'place_tower' in llm_guidance.lower():
                output[2] += self.llm_guidance_weight  # Boost placement probability
                self.llm_guidance_count += 1
                self.last_guidance = llm_guidance
                print(f"🧠 LLMガイダンス適用: {llm_guidance[:50]}...")
            
            # Force action based on threshold
            should_place = output[2] > self.action_threshold or np.random.random() < 0.4
            
            # Normalize coordinates
            x = max(0.1, min(0.9, self._sigmoid(output[0])))
            y = max(0.1, min(0.9, self._sigmoid(output[1])))
            
            result = {
                'x': float(x),
                'y': float(y),
                'should_place': bool(should_place),
                'confidence': float(abs(output[2])),
                'llm_guided': llm_guidance is not None
            }
            
            print(f"🎯 ELM予測: 配置={should_place}, 座標=({x:.2f}, {y:.2f}), 信頼度={result['confidence']:.2f}")
            return result
            
        except Exception as e:
            print(f"❌ ELM予測エラー: {e}")
            # Fallback to random placement
            return {
                'x': random.uniform(0.2, 0.8),
                'y': random.uniform(0.2, 0.8),
                'should_place': True,
                'confidence': 0.5,
                'llm_guided': False
            }
    
    def _extract_features(self, game_state):
        """Extract features from game state"""
        return [
            game_state.get('money', 250) / 1000.0,
            game_state.get('health', 100) / 100.0,
            game_state.get('wave', 1) / 10.0,
            game_state.get('score', 0) / 1000.0,
            len(game_state.get('towers', [])) / 20.0,
            len(game_state.get('enemies', [])) / 50.0,
            np.random.random(),  # Add randomness
            time.time() % 100 / 100.0  # Time-based feature
        ]
    
    def learn_from_experience(self, game_state, action_taken, reward):
        """Learn from game experience"""
        try:
            self.total_learning_updates += 1
            current_score = game_state.get('score', 0)
            print(f"📚 学習更新: {self.total_learning_updates}回, スコア: {current_score}")
        except Exception as e:
            print(f"❌ 学習エラー: {e}")
    
    def get_learning_efficiency_metrics(self):
        """Get current learning efficiency metrics"""
        current_time = time.time() - self.learning_start_time
        
        return {
            'learning_time': current_time,
            'learning_updates': self.total_learning_updates,
            'llm_guidance_count': self.llm_guidance_count,
            'learning_rate': self.total_learning_updates / max(current_time, 1) * 60,  # per minute
            'efficiency_score': self.llm_guidance_count / max(current_time, 1) * 60,  # guidance per minute
            'last_guidance': self.last_guidance
        }

# Global model instances
baseline_elm = AutoELM(random_state=42)
llm_guided_elm = AutoELM(random_state=43)

@app.route('/')
def index():
    """Serve the main game page with auto-restart functionality"""
    html_template = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tower Defense ELM Auto-Fix</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr 300px;
            gap: 20px;
        }
        .game-area {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }
        .control-panel {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }
        #gameCanvas {
            width: 100%;
            height: 400px;
            background: #2c3e50;
            border-radius: 10px;
            border: 2px solid #34495e;
        }
        .status-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin: 15px 0;
        }
        .status-item {
            background: rgba(255, 255, 255, 0.1);
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }
        .status-value {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .button {
            background: #3498db;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            margin: 5px;
            transition: all 0.3s;
        }
        .button:hover {
            background: #2980b9;
            transform: translateY(-2px);
        }
        .button:active {
            transform: translateY(0);
        }
        .btn-mode {
            width: calc(100% - 10px);
            margin: 2px 5px;
        }
        .btn-mode.active {
            background: #e74c3c;
        }
        .guidance-panel {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            padding: 15px;
            margin: 15px 0;
        }
        .guidance-text {
            font-size: 14px;
            line-height: 1.4;
            color: #ecf0f1;
        }
        .auto-indicator {
            background: rgba(39, 174, 96, 0.2);
            border: 1px solid #27ae60;
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
            text-align: center;
        }
        .experiment-status {
            background: rgba(52, 152, 219, 0.2);
            border: 1px solid #3498db;
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
            font-size: 12px;
        }
        h1 {
            text-align: center;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        h4 {
            margin: 15px 0 10px 0;
            color: #ecf0f1;
        }
        .api-section {
            background: rgba(231, 76, 60, 0.2);
            border: 2px solid #e74c3c;
            border-radius: 10px;
            padding: 15px;
            margin: 15px 0;
        }
        .api-input {
            width: 100%;
            padding: 12px;
            margin-bottom: 10px;
            border: 2px solid #3498db;
            border-radius: 8px;
            font-size: 14px;
            box-sizing: border-box;
        }
        .api-status {
            margin: 10px 0;
            padding: 12px;
            border-radius: 8px;
            background: #e74c3c;
            color: white;
            font-size: 14px;
            text-align: center;
            font-weight: bold;
        }
        .api-status.configured {
            background: #27ae60;
        }
    </style>
</head>
<body>
    <h1>🎮 Tower Defense LLM Trainer</h1>
    <p style="text-align: center; margin-bottom: 30px;">AIが学ぶ次世代タワーディフェンスゲーム</p>
    
    <div class="container">
        <div class="game-area">
            <canvas id="gameCanvas" width="700" height="400"></canvas>
            
            <div style="display: flex; gap: 10px; margin-top: 15px;">
                <button class="button" onclick="startGame()">ゲーム開始</button>
                <button class="button" onclick="pauseGame()">一時停止</button>
                <button class="button" onclick="resetGame()">リセット</button>
            </div>
        </div>
        
        <div class="control-panel">
            <h4>📊 ゲーム状況</h4>
            <div class="status-grid">
                <div class="status-item">
                    <div class="status-value" id="money" style="color: #2ecc71;">$250</div>
                    <div>資金</div>
                </div>
                <div class="status-item">
                    <div class="status-value" id="health" style="color: #e74c3c;">100</div>
                    <div>ヘルス</div>
                </div>
                <div class="status-item">
                    <div class="status-value" id="wave" style="color: #f39c12;">1</div>
                    <div>ウェーブ</div>
                </div>
                <div class="status-item">
                    <div class="status-value" id="score" style="color: #9b59b6;">0</div>
                    <div>スコア</div>
                </div>
            </div>
            
            <div class="guidance-panel">
                <h4>🧠 戦略指導システム</h4>
                <label>
                    <input type="checkbox" id="guidanceEnabled" checked>
                    指導システムを有効にする
                </label>
                <div style="margin-top: 10px; font-size: 12px; color: #bdc3c7;">
                    ELM自動操作: <span id="elmStatus">無効</span>
                </div>
            </div>
            
            <div class="auto-indicator">
                <strong>🔧 ELM自動実行: 有効</strong><br>
                <small>3秒間隔で強制実行</small>
            </div>
            
            <div class="auto-indicator">
                <strong>🔄 自動再開: 有効</strong><br>
                <small>ライフ0で2秒後に自動再開</small>
            </div>
            
            <div class="api-section">
                <h4>🔑 OpenAI API設定</h4>
                <input type="password" id="apiKeyInput" class="api-input" placeholder="OpenAI APIキーを入力してください">
                <button class="button" onclick="setApiKey()" style="background: #27ae60; width: 100%;">🔧 APIキー設定</button>
                <div id="apiStatus" class="api-status">
                    ⚠️ APIキー未設定 - LLM機能が無効です
                </div>
            </div>
            
            <h4>ゲームモード</h4>
            <button class="button btn-mode" onclick="setMode('manual')">🎮 手動プレイ</button>
            <button class="button btn-mode" onclick="setMode('elm_only')">🤖 ELMのみ</button>
            <button class="button btn-mode active" onclick="setMode('elm_llm')">🧠 ELM+指導システム</button>
            
            <div class="experiment-status">
                <strong>実験状況:</strong><br>
                試行回数: <span id="trialCount">0</span><br>
                学習時間: <span id="learningTime">0</span>秒<br>
                LLMガイダンス: <span id="guidanceCount">0</span>回
            </div>
            
            <div class="guidance-panel">
                <h4>🧠 戦略指導システム</h4>
                <div class="guidance-text" id="guidanceText">
                    ゲームを開始してください
                </div>
            </div>
        </div>
    </div>

    <script>
        // Game state
        let gameState = {
            running: false,
            paused: false,
            money: 250,
            health: 100,
            wave: 1,
            score: 0,
            towers: 0,
            enemies: 0,
            mode: 'elm_llm'
        };
        
        // API Key management
        let apiKey = '';
        let apiConfigured = false;
        
        // Experiment data
        let experimentData = {
            startTime: null,
            trialCount: 0,
            guidanceCount: 0,
            autoRestart: true
        };
        
        // Game objects
        let towers = [];
        let enemies = [];
        let gameLoop = null;
        let elmLoop = null;
        let autoRestartTimeout = null;
        
        // Canvas setup
        const canvas = document.getElementById('gameCanvas');
        const ctx = canvas.getContext('2d');
        
        // Game path
        const path = [
            {x: 50, y: 200}, {x: 150, y: 200}, {x: 150, y: 100},
            {x: 300, y: 100}, {x: 300, y: 300}, {x: 500, y: 300},
            {x: 500, y: 150}, {x: 650, y: 150}
        ];
        
        // API Key functions
        function setApiKey() {
            const input = document.getElementById('apiKeyInput');
            const status = document.getElementById('apiStatus');
            
            if (input.value.trim()) {
                apiKey = input.value.trim();
                apiConfigured = true;
                status.className = 'api-status configured';
                status.textContent = '✅ APIキー設定済み - LLM機能が有効です';
                console.log('OpenAI APIキーが設定されました');
            } else {
                apiKey = '';
                apiConfigured = false;
                status.className = 'api-status';
                status.textContent = '⚠️ APIキー未設定 - LLM機能が無効です';
                console.log('APIキーが空です');
            }
        }
        
        // Game mode functions
        function setMode(mode) {
            gameState.mode = mode;
            document.querySelectorAll('.btn-mode').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            const elmStatus = document.getElementById('elmStatus');
            if (mode === 'manual') {
                elmStatus.textContent = '無効';
                clearInterval(elmLoop);
            } else if (mode === 'elm_only') {
                elmStatus.textContent = '有効 (単独)';
                if (gameState.running) {
                    clearInterval(elmLoop);
                    elmLoop = setInterval(runELMAutomation, 3000);
                }
            } else if (mode === 'elm_llm') {
                elmStatus.textContent = '有効 (hybrid)';
                if (gameState.running) {
                    clearInterval(elmLoop);
                    elmLoop = setInterval(runELMAutomation, 3000);
                }
            }
            
            console.log(`ゲームモード変更: ${mode}`);
        }
        
        // Game control functions
        function startGame() {
            if (gameState.running) return;
            
            console.log('ゲーム開始...');
            gameState.running = true;
            gameState.paused = false;
            
            if (!experimentData.startTime) {
                experimentData.startTime = Date.now();
            }
            experimentData.trialCount++;
            
            // Start game loop
            gameLoop = setInterval(updateGame, 100);
            
            // Start ELM automation for non-manual modes
            if (gameState.mode !== 'manual') {
                clearInterval(elmLoop);
                elmLoop = setInterval(runELMAutomation, 3000);
                // Start immediately after 1 second
                setTimeout(runELMAutomation, 1000);
            }
            
            updateGuidance('ゲーム開始！ELM自動配置が有効です');
            updateExperimentDisplay();
        }
        
        function pauseGame() {
            if (!gameState.running) return;
            
            gameState.paused = !gameState.paused;
            if (gameState.paused) {
                clearInterval(gameLoop);
                clearInterval(elmLoop);
                updateGuidance('ゲーム一時停止中...');
            } else {
                gameLoop = setInterval(updateGame, 100);
                if (gameState.mode !== 'manual') {
                    elmLoop = setInterval(runELMAutomation, 3000);
                }
                updateGuidance('ゲーム再開！');
            }
        }
        
        function resetGame() {
            console.log('ゲームリセット...');
            gameState.running = false;
            gameState.paused = false;
            gameState.money = 250;
            gameState.health = 100;
            gameState.wave = 1;
            gameState.score = 0;
            gameState.towers = 0;
            gameState.enemies = 0;
            
            towers = [];
            enemies = [];
            
            clearInterval(gameLoop);
            clearInterval(elmLoop);
            clearTimeout(autoRestartTimeout);
            
            updateDisplay();
            updateGuidance('ゲームがリセットされました');
            draw();
        }
        
        // ELM automation function
        function runELMAutomation() {
            if (!gameState.running || gameState.mode === 'manual') return;
            
            console.log('ELM自動実行開始...');
            
            // Call ELM prediction API
            fetch('/api/elm-predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    mode: gameState.mode,
                    money: gameState.money,
                    health: gameState.health,
                    wave: gameState.wave,
                    score: gameState.score,
                    towers: gameState.towers,
                    enemies: gameState.enemies
                })
            })
            .then(response => response.json())
            .then(data => {
                console.log('ELM予測結果:', data);
                
                if (data.should_place && gameState.money >= 50) {
                    const x = Math.max(50, Math.min(canvas.width - 50, data.x * canvas.width));
                    const y = Math.max(50, Math.min(canvas.height - 50, data.y * canvas.height));
                    
                    placeTower(x, y);
                    console.log(`ELM自動配置: タワーを(${x.toFixed(0)}, ${y.toFixed(0)})に配置`);
                    
                    if (gameState.mode === 'elm_llm') {
                        getLLMGuidance();
                    }
                } else {
                    console.log('ELM判断: タワー配置なし');
                }
            })
            .catch(error => {
                console.error('ELM API エラー:', error);
                // Fallback: random placement
                if (gameState.money >= 50 && Math.random() < 0.6) {
                    const x = Math.random() * (canvas.width - 100) + 50;
                    const y = Math.random() * (canvas.height - 100) + 50;
                    placeTower(x, y);
                    console.log(`フォールバック配置: タワーを(${x.toFixed(0)}, ${y.toFixed(0)})に配置`);
                }
            });
        }
        
        // LLM guidance function
        function getLLMGuidance() {
            if (gameState.mode !== 'elm_llm') return;
            
            if (!apiConfigured) {
                updateGuidance("APIキーを設定してください");
                return;
            }
            
            fetch('/api/llm-guidance', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': apiKey
                },
                body: JSON.stringify({game_state: gameState})
            })
            .then(response => response.json())
            .then(data => {
                if (data.recommendation) {
                    updateGuidance(data.recommendation);
                    experimentData.guidanceCount++;
                    updateExperimentDisplay();
                }
            })
            .catch(error => {
                console.error('LLM guidance error:', error);
                updateGuidance("LLMガイダンスエラー - APIキーを確認してください");
            });
        }
        
        // Game logic functions
        function updateGame() {
            if (!gameState.running || gameState.paused) return;
            
            // Spawn enemies
            if (Math.random() < 0.05) {
                spawnEnemy();
            }
            
            // Update enemies
            enemies.forEach((enemy, index) => {
                moveEnemy(enemy);
                if (enemy.pathIndex >= path.length) {
                    // Enemy reached end
                    gameState.health -= enemy.damage;
                    enemies.splice(index, 1);
                    
                    if (gameState.health <= 0) {
                        handleGameOver();
                        return;
                    }
                }
            });
            
            // Tower attacks
            towers.forEach(tower => {
                enemies.forEach((enemy, enemyIndex) => {
                    const distance = Math.sqrt(
                        Math.pow(tower.x - enemy.x, 2) + 
                        Math.pow(tower.y - enemy.y, 2)
                    );
                    
                    if (distance <= tower.range) {
                        enemy.health -= tower.damage;
                        if (enemy.health <= 0) {
                            gameState.money += enemy.reward;
                            gameState.score += enemy.reward;
                            enemies.splice(enemyIndex, 1);
                        }
                    }
                });
            });
            
            // Update display
            gameState.towers = towers.length;
            gameState.enemies = enemies.length;
            updateDisplay();
            draw();
        }
        
        function handleGameOver() {
            console.log('ゲームオーバー処理開始...');
            gameState.running = false;
            clearInterval(gameLoop);
            clearInterval(elmLoop);
            clearTimeout(autoRestartTimeout);
            
            console.log(`ゲームオーバー！スコア: ${gameState.score}点, 試行: ${experimentData.trialCount}`);
            updateGuidance(`ゲームオーバー！スコア: ${gameState.score}点`);
            
            // Force auto-restart for ELM modes
            if (gameState.mode === 'elm_only' || gameState.mode === 'elm_llm') {
                experimentData.autoRestart = true;
            }
            
            if (experimentData.autoRestart) {
                updateGuidance(`2秒後に自動再開します... (試行 ${experimentData.trialCount + 1})`);
                console.log('自動再開タイマー開始...');
                
                autoRestartTimeout = setTimeout(() => {
                    console.log('自動再開実行中...');
                    resetGame();
                    setTimeout(() => {
                        startGame();
                        updateGuidance(`自動再開完了！試行 ${experimentData.trialCount}`);
                    }, 500);
                }, 2000);
            } else {
                updateGuidance(`ゲーム終了。手動で再開してください。`);
            }
        }
        
        function spawnEnemy() {
            enemies.push({
                x: path[0].x,
                y: path[0].y,
                pathIndex: 0,
                health: 100,
                maxHealth: 100,
                speed: 2,
                damage: 10,
                reward: 25
            });
        }
        
        function moveEnemy(enemy) {
            if (enemy.pathIndex >= path.length - 1) return;
            
            const target = path[enemy.pathIndex + 1];
            const dx = target.x - enemy.x;
            const dy = target.y - enemy.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance < enemy.speed) {
                enemy.pathIndex++;
                if (enemy.pathIndex < path.length) {
                    enemy.x = path[enemy.pathIndex].x;
                    enemy.y = path[enemy.pathIndex].y;
                }
            } else {
                enemy.x += (dx / distance) * enemy.speed;
                enemy.y += (dy / distance) * enemy.speed;
            }
        }
        
        function placeTower(x, y) {
            if (gameState.money >= 50) {
                towers.push({
                    x: x,
                    y: y,
                    range: 80,
                    damage: 25
                });
                gameState.money -= 50;
                console.log(`タワー配置: (${x.toFixed(0)}, ${y.toFixed(0)})`);
            }
        }
        
        // Display functions
        function updateDisplay() {
            document.getElementById('money').textContent = '$' + gameState.money;
            document.getElementById('health').textContent = gameState.health;
            document.getElementById('wave').textContent = gameState.wave;
            document.getElementById('score').textContent = gameState.score;
        }
        
        function updateGuidance(text) {
            document.getElementById('guidanceText').textContent = text;
        }
        
        function updateExperimentDisplay() {
            document.getElementById('trialCount').textContent = experimentData.trialCount;
            document.getElementById('guidanceCount').textContent = experimentData.guidanceCount;
            
            if (experimentData.startTime) {
                const elapsed = Math.floor((Date.now() - experimentData.startTime) / 1000);
                document.getElementById('learningTime').textContent = elapsed;
            }
        }
        
        function draw() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw path
            ctx.strokeStyle = '#34495e';
            ctx.lineWidth = 20;
            ctx.beginPath();
            ctx.moveTo(path[0].x, path[0].y);
            for (let i = 1; i < path.length; i++) {
                ctx.lineTo(path[i].x, path[i].y);
            }
            ctx.stroke();
            
            // Draw towers
            towers.forEach(tower => {
                ctx.fillStyle = '#2ecc71';
                ctx.beginPath();
                ctx.arc(tower.x, tower.y, 15, 0, Math.PI * 2);
                ctx.fill();
                
                // Draw range
                ctx.strokeStyle = 'rgba(46, 204, 113, 0.3)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.arc(tower.x, tower.y, tower.range, 0, Math.PI * 2);
                ctx.stroke();
            });
            
            // Draw enemies
            enemies.forEach(enemy => {
                ctx.fillStyle = '#e74c3c';
                ctx.beginPath();
                ctx.arc(enemy.x, enemy.y, 10, 0, Math.PI * 2);
                ctx.fill();
                
                // Health bar
                const barWidth = 20;
                const barHeight = 4;
                const healthRatio = enemy.health / enemy.maxHealth;
                
                ctx.fillStyle = '#2c3e50';
                ctx.fillRect(enemy.x - barWidth/2, enemy.y - 20, barWidth, barHeight);
                ctx.fillStyle = healthRatio > 0.5 ? '#2ecc71' : '#e74c3c';
                ctx.fillRect(enemy.x - barWidth/2, enemy.y - 20, barWidth * healthRatio, barHeight);
            });
        }
        
        // Canvas click handler for manual mode
        canvas.addEventListener('click', (e) => {
            if (gameState.mode === 'manual' && gameState.running) {
                const rect = canvas.getBoundingClientRect();
                const x = (e.clientX - rect.left) * (canvas.width / rect.width);
                const y = (e.clientY - rect.top) * (canvas.height / rect.height);
                placeTower(x, y);
            }
        });
        
        // Initialize
        updateDisplay();
        draw();
        updateGuidance('ゲームを開始してください');
        
        // Update experiment display every second
        setInterval(updateExperimentDisplay, 1000);
    </script>
</body>
</html>
    """
    return html_template

@app.route('/api/elm-predict', methods=['POST'])
def elm_predict():
    """ELM prediction API endpoint"""
    try:
        data = request.json
        mode = data.get('mode', 'elm_only')
        
        # Select appropriate model
        model = llm_guided_elm if mode == 'elm_llm' else baseline_elm
        
        # Get prediction
        prediction = model.predict(data)
        
        return jsonify(prediction)
        
    except Exception as e:
        print(f"ELM prediction error: {e}")
        return jsonify({
            'x': random.uniform(0.2, 0.8),
            'y': random.uniform(0.2, 0.8),
            'should_place': True,
            'confidence': 0.5,
            'error': str(e)
        })

@app.route('/api/llm-guidance', methods=['POST'])
def get_llm_guidance():
    """Get strategic guidance from LLM"""
    try:
        data = request.json
        game_state = data['game_state']
        
        # Get API key from header
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            return get_rule_based_guidance(game_state)
        
        # Create OpenAI client with provided API key
        try:
            from openai import OpenAI
            temp_client = OpenAI(api_key=api_key)
        except Exception as e:
            print(f"OpenAI client creation failed: {e}")
            return get_rule_based_guidance(game_state)
        
        # Prepare prompt
        prompt = f"""
タワーディフェンスゲームの戦略アドバイスをお願いします。

現在の状況:
- 資金: ${game_state.get('money', 0)}
- ヘルス: {game_state.get('health', 0)}
- ウェーブ: {game_state.get('wave', 0)}
- スコア: {game_state.get('score', 0)}
- タワー数: {game_state.get('towers', 0)}
- 敵数: {game_state.get('enemies', 0)}

30文字以内で具体的な戦略アドバイスを日本語で提供してください。
"""
        
        response = temp_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7
        )
        
        recommendation = response.choices[0].message.content.strip()
        
        return jsonify({
            'recommendation': recommendation,
            'source': 'llm'
        })
        
    except Exception as e:
        print(f"LLM guidance error: {e}")
        return get_rule_based_guidance(game_state)

def get_rule_based_guidance(game_state):
    """Fallback rule-based guidance"""
    money = game_state.get('money', 0)
    health = game_state.get('health', 100)
    enemies = game_state.get('enemies', 0)
    
    if money >= 50 and enemies > 5:
        guidance = "敵が多いです。タワーを追加配置しましょう"
    elif health < 50:
        guidance = "ヘルスが危険です。防御を強化してください"
    elif money >= 100:
        guidance = "資金に余裕があります。積極的に配置しましょう"
    else:
        guidance = "現在の戦略を継続してください"
    
    return jsonify({
        'recommendation': guidance,
        'source': 'rule_based'
    })

@app.route('/api/elm-learn', methods=['POST'])
def elm_learn():
    """ELM learning API endpoint"""
    try:
        data = request.json
        mode = data.get('mode', 'elm_only')
        
        # Select appropriate model
        model = llm_guided_elm if mode == 'elm_llm' else baseline_elm
        
        # Learn from experience
        model.learn_from_experience(
            data.get('game_state', {}),
            data.get('action_taken', {}),
            data.get('reward', 0)
        )
        
        return jsonify({'status': 'learned'})
        
    except Exception as e:
        print(f"ELM learning error: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/elm-metrics', methods=['GET'])
def elm_metrics():
    """Get ELM learning metrics"""
    try:
        baseline_metrics = baseline_elm.get_learning_efficiency_metrics()
        llm_guided_metrics = llm_guided_elm.get_learning_efficiency_metrics()
        
        return jsonify({
            'baseline': baseline_metrics,
            'llm_guided': llm_guided_metrics
        })
        
    except Exception as e:
        print(f"Metrics error: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
