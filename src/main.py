#!/usr/bin/env python3
"""
Tower Defense LLM Trainer with ELM Automation
ELMの予測結果を実際のゲーム操作に反映させる完全版
"""

from flask import Flask, request, jsonify, render_template_string
import json
import random
import math
import requests
import time

app = Flask(__name__)

# Game constants
TOWER_COST = 50
TOWER_DAMAGE = 60
TOWER_RANGE = 150
ENEMY_HEALTH = 80
ENEMY_SPEED = 0.7
ENEMY_REWARD = 30
ATTACK_INTERVAL = 500

# OpenAI configuration
OPENAI_API_KEY = "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"

def get_real_llm_guidance(game_state):
    """Get real LLM guidance from OpenAI GPT-4o-mini"""
    try:
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        prompt = f"""
あなたはタワーディフェンスゲームの戦略アドバイザーです。現在の状況を分析して、最適な戦略を提案してください。

現在の状況:
- 資金: ${game_state['money']}
- ヘルス: {game_state['health']}
- ウェーブ: {game_state['wave']}
- スコア: {game_state['score']}
- タワー数: {game_state['towers']}
- 敵数: {game_state['enemies']}

ゲーム設定:
- タワーコスト: $50
- タワー攻撃力: 60
- 敵体力: 80
- 撃破報酬: $30

以下の形式で回答してください:
推奨行動: [具体的な行動]
理由: [詳細な理由]
優先度: [urgent/high/medium/low]
"""

        data = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.7
        }
        
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Parse the response
            lines = content.strip().split('\n')
            recommendation = ""
            reasoning = ""
            priority = "medium"
            
            for line in lines:
                if line.startswith('推奨行動:'):
                    recommendation = line.replace('推奨行動:', '').strip()
                elif line.startswith('理由:'):
                    reasoning = line.replace('理由:', '').strip()
                elif line.startswith('優先度:'):
                    priority = line.replace('優先度:', '').strip()
            
            return {
                'recommendation': recommendation,
                'reasoning': reasoning,
                'priority': priority,
                'source': 'openai_gpt_4o_mini'
            }
        else:
            raise Exception(f"OpenAI API error: {response.status_code}")
            
    except Exception as e:
        print(f"LLM guidance error: {e}")
        return None

# Enhanced ELM implementation
class LLMGuidedTowerDefenseELM:
    def __init__(self, input_size=8, hidden_size=20, output_size=3, random_state=42):
        """
        Enhanced ELM for Tower Defense strategy with LLM guidance integration
        Output: [should_place_tower, position_x_ratio, position_y_ratio]
        """
        random.seed(random_state)
        
        # Initialize weights
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
        self.llm_guidance_weight = 0.4  # Increased weight for LLM guidance
        self.last_guidance = None
        
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
    
    def predict(self, x, llm_guidance=None):
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
        output[0] = self.sigmoid(output[0])  # should_place_tower
        output[1] = self.sigmoid(output[1])  # position_x_ratio
        output[2] = self.sigmoid(output[2])  # position_y_ratio
        
        # Apply LLM guidance if available
        if llm_guidance:
            self.last_guidance = llm_guidance
            guidance_influence = self._interpret_llm_guidance(llm_guidance)
            
            # Modify output based on LLM guidance
            output[0] = output[0] * (1 - self.llm_guidance_weight) + guidance_influence['should_place'] * self.llm_guidance_weight
            output[1] = output[1] * (1 - self.llm_guidance_weight) + guidance_influence['pos_x'] * self.llm_guidance_weight
            output[2] = output[2] * (1 - self.llm_guidance_weight) + guidance_influence['pos_y'] * self.llm_guidance_weight
        
        return output
    
    def _interpret_llm_guidance(self, guidance):
        """Interpret LLM guidance into actionable parameters"""
        priority = guidance.get('priority', 'medium')
        recommendation = guidance.get('recommendation', '').lower()
        
        # Map priority to urgency
        priority_map = {
            'urgent': 0.9,
            'high': 0.8,
            'medium': 0.5,
            'low': 0.3
        }
        urgency = priority_map.get(priority, 0.5)
        
        # Determine if should place tower based on recommendation
        should_place = 0.5
        if 'タワーを配置' in recommendation or 'タワーを' in recommendation or 'tower' in recommendation:
            should_place = urgency
        elif '継続' in recommendation or 'continue' in recommendation:
            should_place = 0.2
        
        # Strategic positions (these would be learned over time)
        pos_x = 0.3 + random.random() * 0.4  # Avoid edges
        pos_y = 0.3 + random.random() * 0.4  # Avoid edges
        
        return {
            'should_place': should_place,
            'pos_x': pos_x,
            'pos_y': pos_y
        }

# Initialize ELM models
baseline_elm = LLMGuidedTowerDefenseELM(random_state=42)
llm_guided_elm = LLMGuidedTowerDefenseELM(random_state=123)

@app.route('/')
def index():
    """Serve the main game interface with ELM automation"""
    return render_template_string("""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tower Defense LLM Trainer</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
            color: white;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            min-height: 100vh;
        }
        
        .header {
            text-align: center;
            margin-bottom: 20px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
            margin: 10px 0;
        }
        
        .game-container {
            display: flex;
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .game-area {
            flex: 1;
        }
        
        .canvas-container {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        #gameCanvas {
            border: 2px solid #4f46e5;
            border-radius: 10px;
            background: #1f2937;
            display: block;
            margin: 0 auto;
        }
        
        .controls {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .btn-primary {
            background: linear-gradient(45deg, #10b981, #059669);
            color: white;
        }
        
        .btn-secondary {
            background: linear-gradient(45deg, #f59e0b, #d97706);
            color: white;
        }
        
        .btn-danger {
            background: linear-gradient(45deg, #ef4444, #dc2626);
            color: white;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        .mode-selection {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        
        .mode-btn {
            padding: 10px 20px;
            border: 2px solid #4f46e5;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.1);
            color: white;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .mode-btn.active {
            background: #4f46e5;
            box-shadow: 0 0 20px rgba(79, 70, 229, 0.5);
        }
        
        .sidebar {
            width: 300px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .panel {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .panel h3 {
            margin: 0 0 15px 0;
            font-size: 1.3em;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        
        .stat-item {
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.8em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .stat-label {
            font-size: 0.9em;
            opacity: 0.8;
        }
        
        .money { color: #10b981; }
        .health { color: #ef4444; }
        .wave { color: #f59e0b; }
        .score { color: #8b5cf6; }
        .towers { color: #06b6d4; }
        .enemies { color: #f97316; }
        
        .guidance-content {
            min-height: 100px;
        }
        
        .guidance-recommendation {
            font-size: 1.1em;
            font-weight: bold;
            margin-bottom: 10px;
            padding: 10px;
            border-radius: 8px;
        }
        
        .priority-urgent {
            background: rgba(239, 68, 68, 0.2);
            border-left: 4px solid #ef4444;
        }
        
        .priority-high {
            background: rgba(245, 158, 11, 0.2);
            border-left: 4px solid #f59e0b;
        }
        
        .priority-medium {
            background: rgba(16, 185, 129, 0.2);
            border-left: 4px solid #10b981;
        }
        
        .priority-low {
            background: rgba(107, 114, 128, 0.2);
            border-left: 4px solid #6b7280;
        }
        
        .guidance-reasoning {
            font-size: 0.95em;
            opacity: 0.9;
            line-height: 1.5;
        }
        
        .guidance-toggle {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
        }
        
        .automation-status {
            background: rgba(16, 185, 129, 0.2);
            border: 1px solid #10b981;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 15px;
            text-align: center;
        }
        
        .automation-status.inactive {
            background: rgba(107, 114, 128, 0.2);
            border-color: #6b7280;
        }
        
        @media (max-width: 768px) {
            .game-container {
                flex-direction: column;
            }
            
            .sidebar {
                width: 100%;
            }
            
            #gameCanvas {
                width: 100%;
                height: auto;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎮 Tower Defense LLM Trainer</h1>
        <p>AIがAIを教える次世代タワーディフェンスゲーム</p>
    </div>
    
    <div class="game-container">
        <div class="game-area">
            <div class="canvas-container">
                <canvas id="gameCanvas" width="800" height="600"></canvas>
            </div>
            
            <div class="controls">
                <button id="startBtn" class="btn btn-primary">ゲーム開始</button>
                <button id="pauseBtn" class="btn btn-secondary">一時停止</button>
                <button id="resetBtn" class="btn btn-danger">リセット</button>
            </div>
            
            <div class="mode-selection">
                <button id="manualMode" class="mode-btn active">手動プレイ</button>
                <button id="elmMode" class="mode-btn">ELMのみ</button>
                <button id="hybridMode" class="mode-btn">ELM+指導システム</button>
            </div>
        </div>
        
        <div class="sidebar">
            <div class="panel">
                <h3>📊 ゲーム状況</h3>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div id="money" class="stat-value money">$250</div>
                        <div class="stat-label">資金</div>
                    </div>
                    <div class="stat-item">
                        <div id="health" class="stat-value health">100</div>
                        <div class="stat-label">ヘルス</div>
                    </div>
                    <div class="stat-item">
                        <div id="wave" class="stat-value wave">1</div>
                        <div class="stat-label">ウェーブ</div>
                    </div>
                    <div class="stat-item">
                        <div id="score" class="stat-value score">0</div>
                        <div class="stat-label">スコア</div>
                    </div>
                    <div class="stat-item">
                        <div id="towers" class="stat-value towers">0</div>
                        <div class="stat-label">タワー数</div>
                    </div>
                    <div class="stat-item">
                        <div id="enemies" class="stat-value enemies">0</div>
                        <div class="stat-label">敵数</div>
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <h3>🧠 戦略指導システム</h3>
                <div class="guidance-toggle">
                    <input type="checkbox" id="guidanceToggle" checked>
                    <label for="guidanceToggle">指導システムを有効にする</label>
                </div>
                <div id="automationStatus" class="automation-status inactive">
                    ELM自動操作: 無効
                </div>
                <div id="guidanceContent" class="guidance-content">
                    <div class="guidance-recommendation priority-medium">
                        ゲームを開始してください
                    </div>
                    <div class="guidance-reasoning">
                        戦略指導システムが有効になっています。ゲームを開始すると、リアルタイムで戦略アドバイスを提供します。
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Game state
        let gameState = {
            isRunning: false,
            money: 250,
            health: 100,
            wave: 1,
            score: 0,
            towers: [],
            enemies: [],
            mode: 'manual',
            guidanceEnabled: true,
            automationEnabled: false,
            lastAutomationTime: 0
        };

        // Game constants
        const TOWER_COST = 50;
        const TOWER_DAMAGE = 60;
        const TOWER_RANGE = 150;
        const ENEMY_HEALTH = 80;
        const ENEMY_SPEED = 0.7;
        const ENEMY_REWARD = 30;
        const ATTACK_INTERVAL = 500;
        const AUTOMATION_INTERVAL = 2000; // ELM automation every 2 seconds

        // Path for enemies
        const PATH = [
            {x: 50, y: 300},
            {x: 200, y: 300},
            {x: 200, y: 150},
            {x: 400, y: 150},
            {x: 400, y: 450},
            {x: 600, y: 450},
            {x: 600, y: 300},
            {x: 750, y: 300}
        ];

        // Canvas setup
        const canvas = document.getElementById('gameCanvas');
        const ctx = canvas.getContext('2d');

        // Initialize game
        function initGame() {
            gameState = {
                isRunning: false,
                money: 250,
                health: 100,
                wave: 1,
                score: 0,
                towers: [],
                enemies: [],
                mode: 'manual',
                guidanceEnabled: true,
                automationEnabled: false,
                lastAutomationTime: 0
            };
            updateUI();
            drawGame();
        }

        // Update UI
        function updateUI() {
            document.getElementById('money').textContent = '$' + gameState.money;
            document.getElementById('health').textContent = gameState.health;
            document.getElementById('wave').textContent = gameState.wave;
            document.getElementById('score').textContent = gameState.score;
            document.getElementById('towers').textContent = gameState.towers.length;
            document.getElementById('enemies').textContent = gameState.enemies.length;
            
            // Update automation status
            const statusEl = document.getElementById('automationStatus');
            if (gameState.automationEnabled) {
                statusEl.textContent = 'ELM自動操作: 有効 (' + gameState.mode + ')';
                statusEl.classList.remove('inactive');
            } else {
                statusEl.textContent = 'ELM自動操作: 無効';
                statusEl.classList.add('inactive');
            }
        }

        // Draw game
        function drawGame() {
            // Clear canvas
            ctx.fillStyle = '#1f2937';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Draw path
            ctx.strokeStyle = '#4b5563';
            ctx.lineWidth = 40;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.beginPath();
            ctx.moveTo(PATH[0].x, PATH[0].y);
            for (let i = 1; i < PATH.length; i++) {
                ctx.lineTo(PATH[i].x, PATH[i].y);
            }
            ctx.stroke();
            
            // Draw towers
            gameState.towers.forEach(tower => {
                // Tower range (when hovered or selected)
                ctx.strokeStyle = 'rgba(16, 185, 129, 0.3)';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(tower.x, tower.y, TOWER_RANGE, 0, 2 * Math.PI);
                ctx.stroke();
                
                // Tower
                ctx.fillStyle = '#10b981';
                ctx.beginPath();
                ctx.arc(tower.x, tower.y, 20, 0, 2 * Math.PI);
                ctx.fill();
                
                // Tower border
                ctx.strokeStyle = '#059669';
                ctx.lineWidth = 3;
                ctx.stroke();
            });
            
            // Draw enemies
            gameState.enemies.forEach(enemy => {
                // Enemy
                ctx.fillStyle = '#ef4444';
                ctx.beginPath();
                ctx.arc(enemy.x, enemy.y, 15, 0, 2 * Math.PI);
                ctx.fill();
                
                // Enemy border
                ctx.strokeStyle = '#dc2626';
                ctx.lineWidth = 2;
                ctx.stroke();
                
                // Health bar
                const barWidth = 30;
                const barHeight = 4;
                const healthRatio = enemy.health / ENEMY_HEALTH;
                
                ctx.fillStyle = '#1f2937';
                ctx.fillRect(enemy.x - barWidth/2, enemy.y - 25, barWidth, barHeight);
                
                ctx.fillStyle = healthRatio > 0.5 ? '#10b981' : healthRatio > 0.25 ? '#f59e0b' : '#ef4444';
                ctx.fillRect(enemy.x - barWidth/2, enemy.y - 25, barWidth * healthRatio, barHeight);
            });
        }

        // Spawn enemy
        function spawnEnemy() {
            gameState.enemies.push({
                x: PATH[0].x,
                y: PATH[0].y,
                health: ENEMY_HEALTH,
                pathIndex: 0,
                progress: 0
            });
        }

        // Move enemies
        function moveEnemies() {
            gameState.enemies.forEach((enemy, enemyIndex) => {
                if (enemy.pathIndex < PATH.length - 1) {
                    const current = PATH[enemy.pathIndex];
                    const next = PATH[enemy.pathIndex + 1];
                    
                    const dx = next.x - current.x;
                    const dy = next.y - current.y;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    
                    enemy.progress += ENEMY_SPEED;
                    
                    if (enemy.progress >= distance) {
                        enemy.pathIndex++;
                        enemy.progress = 0;
                        enemy.x = next.x;
                        enemy.y = next.y;
                    } else {
                        const ratio = enemy.progress / distance;
                        enemy.x = current.x + dx * ratio;
                        enemy.y = current.y + dy * ratio;
                    }
                } else {
                    // Enemy reached the end
                    gameState.health -= 10;
                    gameState.enemies.splice(enemyIndex, 1);
                }
            });
        }

        // Tower attacks
        function towerAttacks() {
            gameState.towers.forEach(tower => {
                if (Date.now() - (tower.lastAttack || 0) < ATTACK_INTERVAL) return;
                
                // Find enemies in range
                const enemiesInRange = gameState.enemies.filter(enemy => {
                    const dx = enemy.x - tower.x;
                    const dy = enemy.y - tower.y;
                    return Math.sqrt(dx * dx + dy * dy) <= TOWER_RANGE;
                });
                
                if (enemiesInRange.length > 0) {
                    // Attack first enemy
                    const target = enemiesInRange[0];
                    target.health -= TOWER_DAMAGE;
                    tower.lastAttack = Date.now();
                    
                    // Remove dead enemies
                    if (target.health <= 0) {
                        const index = gameState.enemies.indexOf(target);
                        gameState.enemies.splice(index, 1);
                        gameState.money += ENEMY_REWARD;
                        gameState.score += ENEMY_REWARD;
                    }
                }
            });
        }

        // ELM Automation
        async function performELMAutomation() {
            if (!gameState.automationEnabled || !gameState.isRunning) return;
            if (Date.now() - gameState.lastAutomationTime < AUTOMATION_INTERVAL) return;
            
            try {
                const modelType = gameState.mode === 'hybrid' ? 'llm_guided' : 'baseline';
                
                const response = await fetch('/api/elm-predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        game_state: {
                            money: gameState.money,
                            health: gameState.health,
                            wave: gameState.wave,
                            score: gameState.score,
                            towers: gameState.towers.length,
                            enemies: gameState.enemies.length
                        },
                        model_type: modelType
                    })
                });
                
                if (response.ok) {
                    const prediction = await response.json();
                    
                    // Execute ELM decision
                    if (prediction.should_place_tower && gameState.money >= TOWER_COST) {
                        // Calculate position based on ELM prediction
                        const x = 100 + (prediction.position_ratio || 0.5) * 600;
                        const y = 100 + (Math.random()) * 400;
                        
                        placeTower(x, y);
                        
                        console.log('🤖 ELM自動配置:', {
                            probability: prediction.placement_probability,
                            confidence: prediction.confidence,
                            position: {x, y}
                        });
                    }
                }
                
                gameState.lastAutomationTime = Date.now();
                
            } catch (error) {
                console.error('ELM automation error:', error);
            }
        }

        // Game loop
        function gameLoop() {
            if (!gameState.isRunning) return;
            
            // Spawn enemies periodically
            if (Math.random() < 0.02) {
                spawnEnemy();
            }
            
            moveEnemies();
            towerAttacks();
            
            // ELM automation
            performELMAutomation();
            
            drawGame();
            updateUI();
            
            // Get guidance if enabled
            if (gameState.guidanceEnabled && Math.random() < 0.1) {
                getGuidance();
            }
            
            // Check game over
            if (gameState.health <= 0) {
                gameState.isRunning = false;
                alert('ゲームオーバー！ スコア: ' + gameState.score);
            }
            
            requestAnimationFrame(gameLoop);
        }

        // Get LLM guidance
        async function getGuidance() {
            try {
                const response = await fetch('/api/llm-guidance', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        game_state: {
                            money: gameState.money,
                            health: gameState.health,
                            wave: gameState.wave,
                            score: gameState.score,
                            towers: gameState.towers.length,
                            enemies: gameState.enemies.length
                        }
                    })
                });
                
                const guidance = await response.json();
                displayGuidance(guidance);
            } catch (error) {
                console.error('Error getting guidance:', error);
            }
        }

        // Display guidance
        function displayGuidance(guidance) {
            const content = document.getElementById('guidanceContent');
            const priorityClass = 'priority-' + guidance.priority;
            
            content.innerHTML = `
                <div class="guidance-recommendation ${priorityClass}">
                    ${guidance.recommendation}
                </div>
                <div class="guidance-reasoning">
                    ${guidance.reasoning}
                </div>
            `;
        }

        // Place tower
        function placeTower(x, y) {
            if (gameState.money >= TOWER_COST) {
                // Check if position is valid (not on path)
                let validPosition = true;
                for (let point of PATH) {
                    const dx = x - point.x;
                    const dy = y - point.y;
                    if (Math.sqrt(dx * dx + dy * dy) < 50) {
                        validPosition = false;
                        break;
                    }
                }
                
                if (validPosition) {
                    gameState.towers.push({
                        x: x,
                        y: y,
                        lastAttack: 0
                    });
                    gameState.money -= TOWER_COST;
                    updateUI();
                    drawGame();
                }
            }
        }

        // Event listeners
        document.getElementById('startBtn').addEventListener('click', () => {
            gameState.isRunning = true;
            gameLoop();
        });

        document.getElementById('pauseBtn').addEventListener('click', () => {
            gameState.isRunning = !gameState.isRunning;
            if (gameState.isRunning) gameLoop();
        });

        document.getElementById('resetBtn').addEventListener('click', () => {
            initGame();
        });

        document.getElementById('guidanceToggle').addEventListener('change', (e) => {
            gameState.guidanceEnabled = e.target.checked;
        });

        // Mode selection
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                
                const mode = e.target.id.replace('Mode', '');
                gameState.mode = mode;
                
                // Enable automation for ELM modes
                gameState.automationEnabled = (mode === 'elm' || mode === 'hybrid');
                updateUI();
            });
        });

        // Canvas click for manual tower placement
        canvas.addEventListener('click', (e) => {
            if (gameState.mode === 'manual') {
                const rect = canvas.getBoundingClientRect();
                const x = (e.clientX - rect.left) * (canvas.width / rect.width);
                const y = (e.clientY - rect.top) * (canvas.height / rect.height);
                placeTower(x, y);
            }
        });

        // Initialize game on load
        initGame();
    </script>
</body>
</html>""")

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'llm_integration': 'enabled',
        'openai_configured': bool(OPENAI_API_KEY),
        'model': 'gpt-4o-mini',
        'api_key_length': len(OPENAI_API_KEY) if OPENAI_API_KEY else 0,
        'elm_automation': 'enabled',
        'timestamp': time.time()
    })

@app.route('/api/llm-guidance', methods=['POST'])
def llm_guidance():
    """Get LLM guidance for tower defense strategy"""
    try:
        data = request.json
        game_state = data['game_state']
        
        # Try to get real LLM guidance
        guidance = get_real_llm_guidance(game_state)
        
        if guidance:
            return jsonify(guidance)
        else:
            # Fallback to rule-based guidance
            return get_fallback_guidance(game_state)
            
    except Exception as e:
        print(f"LLM guidance error: {e}")
        return get_fallback_guidance(request.json['game_state'])

def get_fallback_guidance(game_state):
    """Fallback rule-based guidance"""
    money = game_state['money']
    health = game_state['health']
    wave = game_state['wave']
    towers = game_state['towers']
    enemies = game_state['enemies']
    
    if health < 30 and money >= 50:
        return jsonify({
            'recommendation': '緊急でタワーを配置してください！',
            'reasoning': 'ヘルスが危険な状態です',
            'priority': 'urgent',
            'source': 'rule_based_fallback'
        })
    elif money >= 100 and towers < wave:
        return jsonify({
            'recommendation': 'タワーを追加配置しましょう',
            'reasoning': '十分な資金があり、ウェーブに対してタワーが不足しています',
            'priority': 'high',
            'source': 'rule_based_fallback'
        })
    elif money >= 50:
        return jsonify({
            'recommendation': 'タワーを配置して防御を拡張しましょう',
            'reasoning': '資金に余裕があります',
            'priority': 'medium',
            'source': 'rule_based_fallback'
        })
    else:
        return jsonify({
            'recommendation': '現在の戦略を継続してください',
            'reasoning': '良好な状態を維持しています',
            'priority': 'low',
            'source': 'rule_based_fallback'
        })

@app.route('/api/elm-predict', methods=['POST'])
def elm_predict():
    """Get ELM prediction for tower placement with optional LLM guidance"""
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
            game_state.get('towers', 0) / 10.0,   # Normalize towers (optional)
            game_state.get('efficiency', 0),
            game_state.get('survival', 1),
            game_state.get('progress', 1) / 10.0
        ]
        
        # Get LLM guidance for llm_guided and hybrid models
        llm_guidance = None
        if model_type in ['llm_guided', 'hybrid']:
            try:
                llm_guidance = get_real_llm_guidance(game_state)
            except Exception as e:
                print(f"LLM guidance error: {e}")
                llm_guidance = None
        
        if model_type in ['llm_guided', 'hybrid']:
            prediction = llm_guided_elm.predict(features, llm_guidance)
        else:
            prediction = baseline_elm.predict(features)
        
        # Convert prediction to actionable format
        should_place_tower = prediction[0] > 0.5
        placement_probability = prediction[0]
        position_ratio = prediction[1] if len(prediction) > 1 else 0.5
        
        return jsonify({
            'should_place_tower': bool(should_place_tower),
            'placement_probability': float(placement_probability),
            'position_ratio': float(position_ratio),
            'confidence': float(abs(placement_probability - 0.5) * 2),
            'llm_guidance_applied': llm_guidance is not None,
            'model_type': model_type
        })
        
    except Exception as e:
        print(f"ELM prediction error: {e}")
        return jsonify({
            'should_place_tower': False,
            'placement_probability': 0.5,
            'position_ratio': 0.5,
            'confidence': 0.0,
            'llm_guidance_applied': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
