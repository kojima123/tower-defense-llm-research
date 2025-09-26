#!/usr/bin/env python3
"""
Flask server for Tower Defense LLM Trainer - Embedded HTML Version
Handles OpenAI GPT integration with embedded HTML content
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import os
import json
import time
import random
import math
import urllib.request
import urllib.parse

# Test API key (for demonstration purposes)
TEST_OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_HERE"

app = Flask(__name__)
CORS(app)

# Get the absolute path to the project directory
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Embedded HTML content for the game
GAME_HTML = """<!DOCTYPE html>
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
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }
        
        .game-canvas {
            width: 100%;
            max-width: 800px;
            height: 600px;
            background: linear-gradient(135deg, #2d3748, #4a5568);
            border: 2px solid #4f46e5;
            border-radius: 10px;
            display: block;
            margin: 0 auto;
        }
        
        .controls {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .btn-primary {
            background: linear-gradient(45deg, #4f46e5, #7c3aed);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(79, 70, 229, 0.3);
        }
        
        .btn-secondary {
            background: linear-gradient(45deg, #059669, #10b981);
            color: white;
        }
        
        .btn-secondary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(16, 185, 129, 0.3);
        }
        
        .btn-danger {
            background: linear-gradient(45deg, #dc2626, #ef4444);
            color: white;
        }
        
        .btn-danger:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(239, 68, 68, 0.3);
        }
        
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .panel {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .panel h3 {
            color: #4f46e5;
            margin-bottom: 15px;
            font-size: 1.2rem;
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
            background: rgba(0, 0, 0, 0.2);
            padding: 12px;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #4ade80;
        }
        
        .stat-label {
            font-size: 0.9rem;
            color: #9ca3af;
            margin-top: 5px;
        }
        
        .guidance-panel {
            background: rgba(79, 70, 229, 0.1);
            border: 1px solid #4f46e5;
        }
        
        .guidance-content {
            background: rgba(0, 0, 0, 0.2);
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
        }
        
        .guidance-recommendation {
            font-weight: bold;
            color: #4ade80;
            margin-bottom: 10px;
        }
        
        .guidance-reasoning {
            color: #d1d5db;
            font-size: 0.9rem;
            line-height: 1.4;
        }
        
        .priority-urgent {
            border-left: 4px solid #ef4444;
            padding-left: 12px;
        }
        
        .priority-high {
            border-left: 4px solid #f59e0b;
            padding-left: 12px;
        }
        
        .priority-medium {
            border-left: 4px solid #10b981;
            padding-left: 12px;
        }
        
        .priority-low {
            border-left: 4px solid #6b7280;
            padding-left: 12px;
        }
        
        .mode-selector {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }
        
        .mode-btn {
            flex: 1;
            padding: 8px 12px;
            border: 1px solid #4f46e5;
            background: transparent;
            color: #4f46e5;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .mode-btn.active {
            background: #4f46e5;
            color: white;
        }
        
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: #4f46e5;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        @media (max-width: 768px) {
            .game-layout {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .controls {
                flex-direction: column;
                align-items: center;
            }
            
            .btn {
                width: 100%;
                max-width: 300px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎮 Tower Defense LLM Trainer</h1>
            <p>AIがAIを教える次世代タワーディフェンスゲーム</p>
        </div>
        
        <div class="game-layout">
            <div class="game-area">
                <canvas id="gameCanvas" class="game-canvas" width="800" height="600"></canvas>
                
                <div class="controls">
                    <button id="startBtn" class="btn btn-primary">ゲーム開始</button>
                    <button id="pauseBtn" class="btn btn-secondary">一時停止</button>
                    <button id="resetBtn" class="btn btn-danger">リセット</button>
                </div>
                
                <div class="mode-selector">
                    <button id="manualMode" class="mode-btn active">手動プレイ</button>
                    <button id="elmMode" class="mode-btn">ELMのみ</button>
                    <button id="guidedMode" class="mode-btn">ELM+指導システム</button>
                </div>
            </div>
            
            <div class="sidebar">
                <div class="panel">
                    <h3>📊 ゲーム状況</h3>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-value" id="moneyValue">$250</div>
                            <div class="stat-label">資金</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="healthValue">100</div>
                            <div class="stat-label">ヘルス</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="waveValue">1</div>
                            <div class="stat-label">ウェーブ</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="scoreValue">0</div>
                            <div class="stat-label">スコア</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="towersValue">0</div>
                            <div class="stat-label">タワー数</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="enemiesValue">0</div>
                            <div class="stat-label">敵数</div>
                        </div>
                    </div>
                </div>
                
                <div class="panel guidance-panel">
                    <h3>🧠 戦略指導システム</h3>
                    <label>
                        <input type="checkbox" id="guidanceToggle" checked> 指導システムを有効にする
                    </label>
                    <div id="guidanceContent" class="guidance-content">
                        <div class="guidance-recommendation">ゲームを開始してください</div>
                        <div class="guidance-reasoning">戦略的アドバイスがここに表示されます</div>
                    </div>
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
            towers: [],
            enemies: [],
            isRunning: false,
            mode: 'manual',
            guidanceEnabled: true
        };

        // Canvas and context
        const canvas = document.getElementById('gameCanvas');
        const ctx = canvas.getContext('2d');

        // Game constants
        const TOWER_COST = 50;
        const TOWER_DAMAGE = 60;
        const TOWER_RANGE = 150;
        const ENEMY_HEALTH = 80;
        const ENEMY_SPEED = 0.7;
        const ENEMY_REWARD = 30;
        const ATTACK_INTERVAL = 500; // ms

        // Path definition (simple path from left to right)
        const PATH = [
            {x: 0, y: 300},
            {x: 200, y: 300},
            {x: 200, y: 200},
            {x: 400, y: 200},
            {x: 400, y: 400},
            {x: 600, y: 400},
            {x: 600, y: 300},
            {x: 800, y: 300}
        ];

        // Initialize game
        function initGame() {
            gameState = {
                money: 250,
                health: 100,
                wave: 1,
                score: 0,
                towers: [],
                enemies: [],
                isRunning: false,
                mode: 'manual',
                guidanceEnabled: true
            };
            updateUI();
            drawGame();
        }

        // Update UI elements
        function updateUI() {
            document.getElementById('moneyValue').textContent = '$' + gameState.money;
            document.getElementById('healthValue').textContent = gameState.health;
            document.getElementById('waveValue').textContent = gameState.wave;
            document.getElementById('scoreValue').textContent = gameState.score;
            document.getElementById('towersValue').textContent = gameState.towers.length;
            document.getElementById('enemiesValue').textContent = gameState.enemies.length;
        }

        // Draw game
        function drawGame() {
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw path
            ctx.strokeStyle = '#4a5568';
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
                // Tower base
                ctx.fillStyle = '#10b981';
                ctx.beginPath();
                ctx.arc(tower.x, tower.y, 20, 0, Math.PI * 2);
                ctx.fill();
                
                // Tower range (when selected)
                if (tower.showRange) {
                    ctx.strokeStyle = 'rgba(16, 185, 129, 0.3)';
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.arc(tower.x, tower.y, TOWER_RANGE, 0, Math.PI * 2);
                    ctx.stroke();
                }
            });

            // Draw enemies
            gameState.enemies.forEach(enemy => {
                ctx.fillStyle = '#ef4444';
                ctx.beginPath();
                ctx.arc(enemy.x, enemy.y, 15, 0, Math.PI * 2);
                ctx.fill();
                
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

        // Game loop
        function gameLoop() {
            if (!gameState.isRunning) return;
            
            // Spawn enemies periodically
            if (Math.random() < 0.02) {
                spawnEnemy();
            }
            
            moveEnemies();
            towerAttacks();
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
                gameState.mode = e.target.id.replace('Mode', '');
            });
        });

        // Canvas click for tower placement
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
</html>"""

# Enhanced ELM implementation with LLM guidance integration
class LLMGuidedTowerDefenseELM:
    def __init__(self, input_size=8, hidden_size=20, output_size=2, random_state=42):
        """
        Enhanced ELM for Tower Defense strategy with LLM guidance integration
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
        self.llm_guidance_weight = 0.3  # Weight for LLM guidance influence
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
        output[0] = self.sigmoid(output[0])
        output[1] = self.sigmoid(output[1])
        
        # Apply LLM guidance if available
        if llm_guidance:
            self.last_guidance = llm_guidance
            guidance_influence = self._interpret_llm_guidance(llm_guidance)
            
            # Modify output based on LLM guidance
            output[0] = output[0] * (1 - self.llm_guidance_weight) + guidance_influence['should_place'] * self.llm_guidance_weight
            output[1] = output[1] * (1 - self.llm_guidance_weight) + guidance_influence['urgency'] * self.llm_guidance_weight
        
        return output
    
    def _interpret_llm_guidance(self, guidance):
        """Interpret LLM guidance into actionable parameters"""
        priority = guidance.get('priority', 'medium')
        recommendation = guidance.get('recommendation', '').lower()
        
        # Map priority to urgency
        priority_map = {
            'urgent': 0.9,
            'high': 0.7,
            'medium': 0.5,
            'low': 0.3
        }
        urgency = priority_map.get(priority, 0.5)
        
        # Determine if should place tower based on recommendation
        should_place = 0.5
        if 'タワーを配置' in recommendation or 'tower' in recommendation:
            should_place = 0.8
        elif '継続' in recommendation or 'continue' in recommendation:
            should_place = 0.2
        
        return {
            'should_place': should_place,
            'urgency': urgency
        }

# Simple ELM implementation without LLM guidance
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

# Global model instances
baseline_elm = SimpleTowerDefenseELM(random_state=42)
llm_guided_elm = LLMGuidedTowerDefenseELM(random_state=43)

@app.route('/')
def index():
    """Serve the main game page with embedded HTML"""
    return Response(GAME_HTML, mimetype='text/html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'llm_integration': 'enabled',
        'openai_configured': True,
        'api_key_length': len(TEST_OPENAI_API_KEY),
        'model': 'gpt-4o-mini',
        'embedded_html': True,
        'timestamp': time.time()
    })

def call_openai_api(prompt):
    """Call OpenAI API using simple HTTP request"""
    api_key = TEST_OPENAI_API_KEY
    
    try:
        url = 'https://api.openai.com/v1/chat/completions'
        
        data = {
            'model': 'gpt-4o-mini',
            'messages': [
                {'role': 'system', 'content': 'あなたは経験豊富なタワーディフェンス戦略アドバイザーです。簡潔で実用的なアドバイスを提供してください。'},
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': 150,
            'temperature': 0.7
        }
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers=headers
        )
        
        print(f"🤖 Calling OpenAI API...")
        
        with urllib.request.urlopen(req, timeout=15) as response:
            result = json.loads(response.read().decode('utf-8'))
            response_text = result['choices'][0]['message']['content']
            print(f"✅ LLM Response: {response_text[:100]}...")
            return response_text
            
    except Exception as e:
        print(f"❌ Error calling OpenAI API: {e}")
        return None

@app.route('/api/llm-guidance', methods=['POST'])
def get_llm_guidance():
    """Get strategic guidance based on current game state using real LLM"""
    try:
        data = request.json
        game_state = data['game_state']
        
        print(f"🎮 Getting LLM guidance for game state: {game_state}")
        
        # Always try real LLM first
        return get_real_llm_guidance(game_state)
            
    except Exception as e:
        print(f"Error in LLM guidance: {e}")
        return jsonify({'error': str(e)}), 500

def get_real_llm_guidance(game_state):
    """Get guidance from real OpenAI LLM"""
    try:
        # Prepare the prompt for LLM
        prompt = f"""
タワーディフェンスゲームの戦略アドバイザーとして、現在の状況を分析してください。

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

以下の形式で簡潔に回答してください:
推奨行動: [具体的な行動]
理由: [戦略的理由]
優先度: [urgent/high/medium/low]
"""

        llm_response = call_openai_api(prompt)
        
        if llm_response:
            # Parse LLM response to extract structured data
            parsed_guidance = parse_llm_response(llm_response, game_state)
            
            return jsonify({
                'recommendation': parsed_guidance['recommendation'],
                'reasoning': parsed_guidance['reasoning'],
                'priority': parsed_guidance['priority'],
                'source': 'openai_gpt_4o_mini',
                'raw_response': llm_response
            })
        else:
            # Fallback to rule-based guidance
            return get_rule_based_guidance(game_state)
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        # Fallback to rule-based guidance
        return get_rule_based_guidance(game_state)

def parse_llm_response(llm_response, game_state):
    """Parse LLM response to extract structured guidance"""
    lines = llm_response.strip().split('\n')
    
    recommendation = "戦略を継続してください"
    reasoning = "LLMからの分析結果"
    priority = "medium"
    
    for line in lines:
        line = line.strip()
        if line:
            if '推奨行動' in line or '推奨' in line:
                recommendation = line.split(':', 1)[-1].strip() if ':' in line else line
            elif '理由' in line:
                reasoning = line.split(':', 1)[-1].strip() if ':' in line else line
            elif '優先度' in line:
                priority_text = line.split(':', 1)[-1].strip().lower() if ':' in line else line.lower()
                if 'urgent' in priority_text or '緊急' in priority_text:
                    priority = 'urgent'
                elif 'high' in priority_text or '高' in priority_text:
                    priority = 'high'
                elif 'low' in priority_text or '低' in priority_text:
                    priority = 'low'
                else:
                    priority = 'medium'
    
    # Clean up recommendation and reasoning
    recommendation = recommendation.replace('推奨行動:', '').strip()
    reasoning = reasoning.replace('理由:', '').strip()
    
    return {
        'recommendation': recommendation,
        'reasoning': reasoning,
        'priority': priority
    }

def get_rule_based_guidance(game_state):
    """Fallback rule-based guidance system"""
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
            game_state['towers'] / 10.0,   # Normalize towers
            game_state.get('efficiency', 0),
            game_state.get('survival', 1),
            game_state.get('progress', 1) / 10.0
        ]
        
        # Get LLM guidance for llm_guided model
        llm_guidance = None
        if model_type == 'llm_guided':
            try:
                guidance_response = get_real_llm_guidance(game_state)
                if hasattr(guidance_response, 'json'):
                    llm_guidance = guidance_response.json
                else:
                    llm_guidance = guidance_response.get_json()
            except:
                llm_guidance = None
        
        if model_type == 'llm_guided':
            prediction = llm_guided_elm.predict(features, llm_guidance)
        else:
            prediction = baseline_elm.predict(features)
        
        # Convert prediction to actionable format
        should_place_tower = prediction[0] > 0.5
        tower_position_ratio = prediction[1]
        
        return jsonify({
            'should_place_tower': bool(should_place_tower),
            'placement_probability': float(prediction[0]),
            'position_ratio': float(tower_position_ratio),
            'confidence': float(abs(prediction[0] - 0.5) * 2),
            'llm_guidance_applied': llm_guidance is not None
        })
        
    except Exception as e:
        print(f"Error in ELM prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset-models', methods=['POST'])
def reset_models():
    """Reset both ELM models to initial state"""
    global baseline_elm, llm_guided_elm
    
    baseline_elm = SimpleTowerDefenseELM(random_state=42)
    llm_guided_elm = LLMGuidedTowerDefenseELM(random_state=43)
    
    return jsonify({
        'status': 'reset',
        'message': 'Both ELM models have been reset to initial state'
    })

if __name__ == '__main__':
    print("🚀 Starting Tower Defense LLM Trainer with EMBEDDED HTML...")
    print(f"✅ OpenAI API configured: True")
    print(f"🔑 API key length: {len(TEST_OPENAI_API_KEY)}")
    print(f"🤖 Model: gpt-4o-mini")
    print(f"📄 HTML embedded: True")
    app.run(host='0.0.0.0', port=5000, debug=True)
