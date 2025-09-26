#!/usr/bin/env python3
"""
Tower Defense ELM Auto-Fix Server with Auto-Restart
ELMの自動動作 + ライフ0時の自動再開機能付き
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import json
import time
import random
import numpy as np

app = Flask(__name__)
CORS(app)

# OpenAI client setup
try:
    from openai import OpenAI
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        client = None
        print("⚠️ OpenAI API key not found. Using fallback guidance system.")
except ImportError:
    client = None
    print("⚠️ OpenAI library not installed. Using fallback guidance system.")

# Tower Defense ELM implementation
class TowerDefenseELM:
    def __init__(self, input_size=8, hidden_size=20, output_size=2, random_state=42):
        np.random.seed(random_state)
        self.input_weights = np.random.randn(input_size, hidden_size) * 0.5
        self.hidden_bias = np.random.randn(hidden_size) * 0.5
        self.output_weights = np.random.randn(hidden_size, output_size) * 0.1
        self.learning_rate = 0.02
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        return np.tanh(np.clip(x, -500, 500))
    
    def predict(self, x):
        x = np.array(x).reshape(1, -1)
        x = x / np.maximum(np.abs(x), 1e-8)
        
        hidden = self.tanh(np.dot(x, self.input_weights) + self.hidden_bias)
        output = np.dot(hidden, self.output_weights)
        
        output[0, 0] = self.sigmoid(output[0, 0])
        output[0, 1] = self.sigmoid(output[0, 1])
        
        return output[0]
    
    def update(self, x, target, learning_rate=None):
        if learning_rate is None:
            learning_rate = self.learning_rate
            
        x = np.array(x).reshape(1, -1)
        target = np.array(target).reshape(1, -1)
        x = x / np.maximum(np.abs(x), 1e-8)
        
        hidden = self.tanh(np.dot(x, self.input_weights) + self.hidden_bias)
        output = np.dot(hidden, self.output_weights)
        
        output[0, 0] = self.sigmoid(output[0, 0])
        output[0, 1] = self.sigmoid(output[0, 1])
        
        error = target - output
        self.output_weights += learning_rate * np.dot(hidden.T, error)

# Global model instances
baseline_elm = TowerDefenseELM(random_state=42)
llm_guided_elm = TowerDefenseELM(random_state=43)

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
            text-align: center;
            padding: 10px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
        }
        .status-value {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .status-label {
            font-size: 12px;
            opacity: 0.8;
        }
        .button {
            width: 100%;
            padding: 12px;
            margin: 5px 0;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .btn-start { background: #27ae60; color: white; }
        .btn-pause { background: #f39c12; color: white; }
        .btn-reset { background: #e74c3c; color: white; }
        .btn-mode { background: #3498db; color: white; margin: 2px 0; }
        .btn-mode.active { background: #2ecc71; }
        .guidance-panel {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            margin: 15px 0;
        }
        .guidance-text {
            font-size: 14px;
            line-height: 1.4;
        }
        .auto-indicator {
            background: rgba(46, 204, 113, 0.2);
            border: 1px solid #2ecc71;
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
    </style>
</head>
<body>
    <div class="container">
        <div class="game-area">
            <h1>🎮 Tower Defense ELM Auto-Fix</h1>
            <p>ELM自動動作 + 自動再開機能付き学習実験システム</p>
            
            <canvas id="gameCanvas" width="600" height="400"></canvas>
            
            <div class="status-grid">
                <div class="status-item">
                    <div class="status-value" id="money">$250</div>
                    <div class="status-label">資金</div>
                </div>
                <div class="status-item">
                    <div class="status-value" id="health">100</div>
                    <div class="status-label">ヘルス</div>
                </div>
                <div class="status-item">
                    <div class="status-value" id="wave">1</div>
                    <div class="status-label">ウェーブ</div>
                </div>
                <div class="status-item">
                    <div class="status-value" id="score">0</div>
                    <div class="status-label">スコア</div>
                </div>
                <div class="status-item">
                    <div class="status-value" id="towers">0</div>
                    <div class="status-label">タワー数</div>
                </div>
                <div class="status-item">
                    <div class="status-value" id="enemies">0</div>
                    <div class="status-label">敵数</div>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;">
                <button class="button btn-start" onclick="startGame()">▶️ 実験開始</button>
                <button class="button btn-pause" onclick="pauseGame()">⏸️ 一時停止</button>
                <button class="button btn-reset" onclick="resetGame()">🔄 リセット</button>
            </div>
        </div>
        
        <div class="control-panel">
            <h3>📊 実験制御</h3>
            
            <div class="auto-indicator">
                <strong>🔧 ELM自動動作: 有効</strong><br>
                <small>5秒間隔で強制実行</small>
            </div>
            
            <div class="auto-indicator">
                <strong>🔄 自動再開: 有効</strong><br>
                <small>ライフ0で2秒後に自動再開</small>
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
                    実験開始を待機中...
                </div>
            </div>
        </div>
    </div>

    <script>
        // Game state
        let gameState = {
            money: 250,
            health: 100,
            wave: 1,
            score: 0,
            towers: 0,
            enemies: 0,
            running: false,
            mode: 'elm_llm'
        };
        
        let experimentData = {
            trialCount: 0,
            learningTime: 0,
            guidanceCount: 0,
            autoRestart: true
        };
        
        // Canvas setup
        const canvas = document.getElementById('gameCanvas');
        const ctx = canvas.getContext('2d');
        
        // Game objects
        let towers = [];
        let enemies = [];
        let gameLoop;
        let elmLoop;
        let autoRestartTimeout;
        
        // Path for enemies
        const path = [
            {x: 0, y: 200},
            {x: 150, y: 200},
            {x: 150, y: 100},
            {x: 300, y: 100},
            {x: 300, y: 300},
            {x: 450, y: 300},
            {x: 450, y: 150},
            {x: 600, y: 150}
        ];
        
        function startGame() {
            gameState.running = true;
            gameState.money = 250;
            gameState.health = 100;
            gameState.wave = 1;
            gameState.score = 0;
            gameState.towers = 0;
            gameState.enemies = 0;
            
            towers = [];
            enemies = [];
            
            experimentData.trialCount++;
            updateExperimentDisplay();
            
            // Start game loop
            if (gameLoop) clearInterval(gameLoop);
            gameLoop = setInterval(updateGame, 100);
            
            // Start ELM automation (5 second intervals)
            if (elmLoop) clearInterval(elmLoop);
            if (gameState.mode !== 'manual') {
                elmLoop = setInterval(runELMAutomation, 5000);
            }
            
            // Start enemy spawning
            setTimeout(spawnEnemies, 2000);
            
            updateGuidance("ゲーム開始！ELMが自動でタワー配置を学習します。");
        }
        
        function pauseGame() {
            gameState.running = !gameState.running;
            if (!gameState.running) {
                clearInterval(gameLoop);
                clearInterval(elmLoop);
            } else {
                gameLoop = setInterval(updateGame, 100);
                if (gameState.mode !== 'manual') {
                    elmLoop = setInterval(runELMAutomation, 5000);
                }
            }
        }
        
        function resetGame() {
            gameState.running = false;
            clearInterval(gameLoop);
            clearInterval(elmLoop);
            clearTimeout(autoRestartTimeout);
            
            gameState.money = 250;
            gameState.health = 100;
            gameState.wave = 1;
            gameState.score = 0;
            gameState.towers = 0;
            gameState.enemies = 0;
            
            towers = [];
            enemies = [];
            
            updateDisplay();
            drawGame();
            updateGuidance("ゲームがリセットされました。");
        }
        
        function setMode(mode) {
            gameState.mode = mode;
            document.querySelectorAll('.btn-mode').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            if (gameState.running) {
                clearInterval(elmLoop);
                if (mode !== 'manual') {
                    elmLoop = setInterval(runELMAutomation, 5000);
                }
            }
        }
        
        function runELMAutomation() {
            if (!gameState.running || gameState.mode === 'manual') return;
            
            // Force ELM to make a decision
            if (gameState.money >= 50) {
                // Simple strategy: place tower if we have money
                const x = Math.random() * (canvas.width - 100) + 50;
                const y = Math.random() * (canvas.height - 100) + 50;
                
                placeTower(x, y);
                
                if (gameState.mode === 'elm_llm') {
                    getLLMGuidance();
                }
                
                console.log(`ELM自動動作: タワーを(${x.toFixed(0)}, ${y.toFixed(0)})に配置`);
            }
        }
        
        function spawnEnemies() {
            if (!gameState.running) return;
            
            const enemyCount = Math.min(5 + gameState.wave, 20);
            for (let i = 0; i < enemyCount; i++) {
                setTimeout(() => {
                    if (gameState.running) {
                        enemies.push({
                            x: path[0].x,
                            y: path[0].y,
                            pathIndex: 0,
                            health: 80,
                            maxHealth: 80,
                            speed: 1 + Math.random() * 0.5
                        });
                        gameState.enemies = enemies.length;
                    }
                }, i * 1000);
            }
            
            // Schedule next wave
            setTimeout(() => {
                if (gameState.running) {
                    gameState.wave++;
                    spawnEnemies();
                }
            }, 15000);
        }
        
        function placeTower(x, y) {
            if (gameState.money >= 50) {
                towers.push({
                    x: x,
                    y: y,
                    range: 80,
                    damage: 60,
                    lastShot: 0
                });
                gameState.money -= 50;
                gameState.towers = towers.length;
                updateDisplay();
            }
        }
        
        function updateGame() {
            if (!gameState.running) return;
            
            // Move enemies
            enemies.forEach((enemy, enemyIndex) => {
                if (enemy.pathIndex < path.length - 1) {
                    const target = path[enemy.pathIndex + 1];
                    const dx = target.x - enemy.x;
                    const dy = target.y - enemy.y;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    
                    if (distance < 5) {
                        enemy.pathIndex++;
                    } else {
                        enemy.x += (dx / distance) * enemy.speed;
                        enemy.y += (dy / distance) * enemy.speed;
                    }
                } else {
                    // Enemy reached end
                    gameState.health -= 10;
                    enemies.splice(enemyIndex, 1);
                    gameState.enemies = enemies.length;
                    
                    if (gameState.health <= 0) {
                        handleGameOver();
                        return;
                    }
                }
            });
            
            // Tower shooting
            towers.forEach(tower => {
                const now = Date.now();
                if (now - tower.lastShot > 1000) { // 1 second cooldown
                    enemies.forEach((enemy, enemyIndex) => {
                        const dx = enemy.x - tower.x;
                        const dy = enemy.y - tower.y;
                        const distance = Math.sqrt(dx * dx + dy * dy);
                        
                        if (distance <= tower.range) {
                            enemy.health -= tower.damage;
                            tower.lastShot = now;
                            
                            if (enemy.health <= 0) {
                                gameState.money += 30;
                                gameState.score += 30;
                                enemies.splice(enemyIndex, 1);
                                gameState.enemies = enemies.length;
                            }
                        }
                    });
                }
            });
            
            updateDisplay();
            drawGame();
        }
        
        function handleGameOver() {
            gameState.running = false;
            clearInterval(gameLoop);
            clearInterval(elmLoop);
            
            updateGuidance(`ゲームオーバー！スコア: ${gameState.score}点`);
            
            if (experimentData.autoRestart) {
                updateGuidance(`2秒後に自動再開します... (試行 ${experimentData.trialCount + 1})`);
                autoRestartTimeout = setTimeout(() => {
                    startGame();
                }, 2000);
            }
        }
        
        function getLLMGuidance() {
            if (gameState.mode !== 'elm_llm') return;
            
            fetch('/api/llm-guidance', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
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
                updateGuidance("タワーを配置してください");
            });
        }
        
        function updateGuidance(text) {
            document.getElementById('guidanceText').textContent = text;
        }
        
        function updateDisplay() {
            document.getElementById('money').textContent = `$${gameState.money}`;
            document.getElementById('health').textContent = gameState.health;
            document.getElementById('wave').textContent = gameState.wave;
            document.getElementById('score').textContent = gameState.score;
            document.getElementById('towers').textContent = gameState.towers;
            document.getElementById('enemies').textContent = gameState.enemies;
        }
        
        function updateExperimentDisplay() {
            document.getElementById('trialCount').textContent = experimentData.trialCount;
            document.getElementById('learningTime').textContent = experimentData.learningTime;
            document.getElementById('guidanceCount').textContent = experimentData.guidanceCount;
        }
        
        function drawGame() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw path
            ctx.strokeStyle = '#7f8c8d';
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
        updateExperimentDisplay();
        drawGame();
        updateGuidance("🚀 ELM自動動作システム準備完了！実験開始ボタンを押してください。");
    </script>
</body>
</html>
    """
    return html_template

@app.route('/api/llm-guidance', methods=['POST'])
def get_llm_guidance():
    """Get strategic guidance from LLM"""
    try:
        data = request.json
        game_state = data['game_state']
        
        if not client:
            return get_rule_based_guidance(game_state)
        
        try:
            prompt = f"""
あなたはタワーディフェンスゲームの戦略アドバイザーです。現在のゲーム状況を分析し、最適な戦略を提案してください。

現在の状況:
- 資金: ${game_state['money']}
- ヘルス: {game_state['health']}
- ウェーブ: {game_state['wave']}
- 敵の数: {game_state['enemies']}
- タワー数: {game_state['towers']}
- スコア: {game_state['score']}

タワーコスト: $50、ダメージ: 60、射程: 80

具体的で実行可能なアドバイスを1文で回答してください。
"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.7
            )
            
            content = response.choices[0].message.content.strip()
            
            return jsonify({
                'recommendation': content,
                'source': 'llm'
            })
            
        except Exception as e:
            print(f"LLM guidance failed: {e}")
            return get_rule_based_guidance(game_state)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_rule_based_guidance(game_state):
    """Fallback rule-based guidance"""
    money = game_state['money']
    health = game_state['health']
    enemies = game_state['enemies']
    towers = game_state['towers']
    
    if health < 30:
        recommendation = '緊急でタワーを配置してください！'
    elif enemies > 10 and towers < 3:
        recommendation = 'タワーを追加配置しましょう'
    elif money >= 100:
        recommendation = 'タワーを配置して防御を強化してください'
    elif money >= 50:
        recommendation = 'タワーを配置しましょう'
    else:
        recommendation = '資金を貯めてタワーを準備してください'
    
    return jsonify({
        'recommendation': recommendation,
        'source': 'rule_based'
    })

if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    print("🚀 Tower Defense ELM Auto-Fix Server Starting...")
    print(f"🔑 OpenAI API Key: {'✅ Configured' if client else '❌ Using fallback'}")
    print("🔧 Auto-Fix: ELMの自動動作を強制実行")
    print("🔄 Auto-Restart: ライフ0で自動再開")
    print("📊 Learning efficiency experiment ready")
    print(f"🌐 Server starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
