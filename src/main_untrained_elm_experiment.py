"""
Tower Defense ELM Experiment - Untrained ELM Comparison
未訓練ELMでのLLMガイダンス効果測定実験
"""

from flask import Flask, request, jsonify, render_template_string
import json
import random
import math
import requests
import time
import os
from datetime import datetime

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
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def get_real_llm_guidance(game_state):
    """Get real LLM guidance from OpenAI GPT-4o-mini"""
    if not OPENAI_API_KEY:
        return None
        
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

以下の形式で回答してください:
{{
    "priority": "urgent/high/medium/low",
    "recommendation": "具体的な行動提案",
    "reasoning": "判断理由"
}}
"""
        
        data = {
            'model': 'gpt-4o-mini',
            'messages': [
                {'role': 'system', 'content': 'あなたはタワーディフェンスゲームの専門戦略アドバイザーです。'},
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': 300,
            'temperature': 0.7
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
            
            # Try to parse JSON response
            try:
                guidance = json.loads(content)
                return guidance
            except:
                # Fallback parsing
                return {
                    'priority': 'medium',
                    'recommendation': content[:100],
                    'reasoning': 'LLM response parsing fallback'
                }
        else:
            print(f"LLM API error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"LLM guidance error: {e}")
        return None

# Untrained ELM implementation for experiments
class UntrainedTowerDefenseELM:
    def __init__(self, input_size=8, hidden_size=20, output_size=3, random_state=None):
        """
        Untrained ELM for Tower Defense strategy
        Output: [should_place_tower, position_x_ratio, position_y_ratio]
        """
        if random_state is not None:
            random.seed(random_state)
        
        # Initialize random weights (untrained state)
        self.input_weights = []
        for i in range(input_size):
            row = []
            for j in range(hidden_size):
                row.append(random.gauss(0, 1.0))  # Random initialization
            self.input_weights.append(row)
        
        self.hidden_bias = [random.gauss(0, 1.0) for _ in range(hidden_size)]
        
        self.output_weights = []
        for i in range(hidden_size):
            row = []
            for j in range(output_size):
                row.append(random.gauss(0, 1.0))  # Random initialization
            self.output_weights.append(row)
        
        # No learning - keep weights fixed for experiment
        self.is_trained = False
        self.llm_guidance_weight = 0.5  # Weight for LLM guidance integration
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
        """Predict action with optional LLM guidance"""
        # Normalize inputs
        x_norm = []
        for val in x:
            if abs(val) > 1e-8:
                x_norm.append(max(-10, min(10, val / (abs(val) + 1))))
            else:
                x_norm.append(val)
        
        # Forward pass through untrained network
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
            
            # Blend ELM output with LLM guidance
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
        if any(keyword in recommendation for keyword in ['タワーを配置', 'タワーを', 'tower', '建設', '設置']):
            should_place = urgency
        elif any(keyword in recommendation for keyword in ['継続', 'continue', '待機', 'wait']):
            should_place = 0.2
        
        # Strategic positions (random for untrained ELM)
        pos_x = 0.2 + random.random() * 0.6  # Avoid edges
        pos_y = 0.2 + random.random() * 0.6  # Avoid edges
        
        return {
            'should_place': should_place,
            'pos_x': pos_x,
            'pos_y': pos_y
        }

# Global experiment state
experiment_state = {
    'mode': 'elm_only',  # 'elm_only' or 'elm_llm'
    'trial_count': 0,
    'results': [],
    'start_time': None,
    'current_trial_start': None
}

# Initialize untrained ELM models with different random seeds
elm_only_model = UntrainedTowerDefenseELM(random_state=42)
elm_llm_model = UntrainedTowerDefenseELM(random_state=123)

@app.route('/')
def index():
    """Serve the experiment interface"""
    return render_template_string("""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tower Defense ELM Experiment</title>
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
        
        .experiment-info {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
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
        }
        
        .btn-primary {
            background: linear-gradient(45deg, #4f46e5, #7c3aed);
            color: white;
        }
        
        .btn-secondary {
            background: linear-gradient(45deg, #059669, #0d9488);
            color: white;
        }
        
        .btn-danger {
            background: linear-gradient(45deg, #dc2626, #b91c1c);
            color: white;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        
        .info-panel {
            width: 350px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            height: fit-content;
        }
        
        .info-section {
            margin-bottom: 20px;
        }
        
        .info-section h3 {
            margin: 0 0 10px 0;
            color: #fbbf24;
            font-size: 1.2em;
        }
        
        .stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        
        .stat-item {
            background: rgba(255, 255, 255, 0.1);
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-label {
            font-size: 0.9em;
            opacity: 0.8;
            margin-bottom: 5px;
        }
        
        .stat-value {
            font-size: 1.4em;
            font-weight: bold;
        }
        
        .experiment-status {
            background: rgba(34, 197, 94, 0.2);
            border: 1px solid #22c55e;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }
        
        .mode-indicator {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-left: 10px;
        }
        
        .mode-elm-only {
            background: #dc2626;
            color: white;
        }
        
        .mode-elm-llm {
            background: #059669;
            color: white;
        }
        
        .guidance-panel {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            margin-top: 15px;
        }
        
        .guidance-recommendation {
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .guidance-reasoning {
            font-size: 0.9em;
            opacity: 0.8;
        }
        
        .priority-urgent {
            border-left: 4px solid #dc2626;
            padding-left: 10px;
        }
        
        .priority-high {
            border-left: 4px solid #f59e0b;
            padding-left: 10px;
        }
        
        .priority-medium {
            border-left: 4px solid #3b82f6;
            padding-left: 10px;
        }
        
        .priority-low {
            border-left: 4px solid #6b7280;
            padding-left: 10px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏰 Tower Defense ELM Experiment</h1>
        <p>未訓練ELMでのLLMガイダンス効果測定実験</p>
    </div>
    
    <div class="experiment-info">
        <div class="experiment-status">
            <strong>実験モード:</strong>
            <span id="currentMode">ELM Only</span>
            <span id="modeIndicator" class="mode-indicator mode-elm-only">ELM のみ</span>
            <br>
            <strong>試行回数:</strong> <span id="trialCount">0</span> / 20
            <br>
            <strong>実験時間:</strong> <span id="experimentTime">00:00</span>
        </div>
    </div>
    
    <div class="game-container">
        <div class="game-area">
            <div class="canvas-container">
                <canvas id="gameCanvas" width="800" height="600"></canvas>
            </div>
            
            <div class="controls">
                <button class="btn btn-primary" onclick="startExperiment()">実験開始</button>
                <button class="btn btn-secondary" onclick="switchMode()">モード切替</button>
                <button class="btn btn-danger" onclick="resetExperiment()">実験リセット</button>
                <button class="btn btn-primary" onclick="exportResults()">結果エクスポート</button>
            </div>
        </div>
        
        <div class="info-panel">
            <div class="info-section">
                <h3>🎮 ゲーム状態</h3>
                <div class="stat-grid">
                    <div class="stat-item">
                        <div class="stat-label">資金</div>
                        <div class="stat-value" id="money">$100</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">ヘルス</div>
                        <div class="stat-value" id="health">100</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">スコア</div>
                        <div class="stat-value" id="score">0</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">ウェーブ</div>
                        <div class="stat-value" id="wave">1</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">タワー</div>
                        <div class="stat-value" id="towers">0</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">敵</div>
                        <div class="stat-value" id="enemies">0</div>
                    </div>
                </div>
            </div>
            
            <div class="info-section">
                <h3>🤖 ELM状態</h3>
                <div id="elmStatus">
                    <p>未訓練ELMが実行中...</p>
                    <p>ランダム初期パラメータ使用</p>
                </div>
            </div>
            
            <div class="info-section" id="guidanceSection" style="display: none;">
                <h3>🧠 LLMガイダンス</h3>
                <div id="guidanceContent" class="guidance-panel">
                    ガイダンス待機中...
                </div>
            </div>
            
            <div class="info-section">
                <h3>📊 実験結果</h3>
                <div id="experimentResults">
                    <p>実験データ収集中...</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Game constants
        const TOWER_COST = 50;
        const TOWER_DAMAGE = 60;
        const TOWER_RANGE = 150;
        const ENEMY_HEALTH = 80;
        const ENEMY_SPEED = 0.7;
        const ENEMY_REWARD = 30;
        const ATTACK_INTERVAL = 500;
        
        // Game state
        let gameState = {
            money: 100,
            health: 100,
            score: 0,
            wave: 1,
            towers: [],
            enemies: [],
            projectiles: [],
            isRunning: false,
            lastAutomationTime: 0,
            automationInterval: 2000,
            guidanceEnabled: false
        };
        
        // Experiment state
        let experimentState = {
            mode: 'elm_only',
            trialCount: 0,
            results: [],
            startTime: null,
            currentTrialStart: null,
            trialDuration: 60000  // 1 minute per trial
        };
        
        // Canvas setup
        const canvas = document.getElementById('gameCanvas');
        const ctx = canvas.getContext('2d');
        
        // Enemy path (simple straight line for now)
        const enemyPath = [
            {x: 0, y: 300},
            {x: 800, y: 300}
        ];
        
        // Initialize game
        function initGame() {
            gameState = {
                money: 100,
                health: 100,
                score: 0,
                wave: 1,
                towers: [],
                enemies: [],
                projectiles: [],
                isRunning: false,
                lastAutomationTime: 0,
                automationInterval: 2000,
                guidanceEnabled: experimentState.mode === 'elm_llm'
            };
            
            updateUI();
        }
        
        // Start experiment
        function startExperiment() {
            if (!experimentState.startTime) {
                experimentState.startTime = Date.now();
            }
            
            experimentState.currentTrialStart = Date.now();
            experimentState.trialCount++;
            
            initGame();
            gameState.isRunning = true;
            
            // Update UI
            updateExperimentUI();
            
            // Start game loop
            gameLoop();
            
            // Auto-stop trial after duration
            setTimeout(() => {
                stopTrial();
            }, experimentState.trialDuration);
        }
        
        // Stop current trial
        function stopTrial() {
            gameState.isRunning = false;
            
            // Record trial results
            const trialResult = {
                mode: experimentState.mode,
                trial: experimentState.trialCount,
                score: gameState.score,
                health: gameState.health,
                towers: gameState.towers.length,
                duration: Date.now() - experimentState.currentTrialStart,
                timestamp: new Date().toISOString()
            };
            
            experimentState.results.push(trialResult);
            
            console.log('Trial completed:', trialResult);
            updateExperimentResults();
            
            // Auto-start next trial if under 20 trials
            if (experimentState.trialCount < 20) {
                setTimeout(() => {
                    startExperiment();
                }, 2000);
            } else {
                alert('実験完了！20回の試行が終了しました。');
            }
        }
        
        // Switch experiment mode
        function switchMode() {
            if (gameState.isRunning) {
                alert('実験中はモードを変更できません。');
                return;
            }
            
            experimentState.mode = experimentState.mode === 'elm_only' ? 'elm_llm' : 'elm_only';
            updateExperimentUI();
        }
        
        // Reset experiment
        function resetExperiment() {
            if (confirm('実験をリセットしますか？すべてのデータが失われます。')) {
                experimentState = {
                    mode: 'elm_only',
                    trialCount: 0,
                    results: [],
                    startTime: null,
                    currentTrialStart: null,
                    trialDuration: 60000
                };
                
                initGame();
                updateExperimentUI();
                updateExperimentResults();
            }
        }
        
        // Export results
        function exportResults() {
            if (experimentState.results.length === 0) {
                alert('エクスポートするデータがありません。');
                return;
            }
            
            const data = {
                experiment_info: {
                    total_trials: experimentState.trialCount,
                    start_time: experimentState.startTime,
                    trial_duration_ms: experimentState.trialDuration
                },
                results: experimentState.results
            };
            
            const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `elm_experiment_results_${new Date().toISOString().slice(0,19).replace(/:/g, '-')}.json`;
            a.click();
            URL.revokeObjectURL(url);
        }
        
        // Update experiment UI
        function updateExperimentUI() {
            document.getElementById('currentMode').textContent = 
                experimentState.mode === 'elm_only' ? 'ELM Only' : 'ELM + LLM Guidance';
            
            const indicator = document.getElementById('modeIndicator');
            if (experimentState.mode === 'elm_only') {
                indicator.textContent = 'ELM のみ';
                indicator.className = 'mode-indicator mode-elm-only';
            } else {
                indicator.textContent = 'ELM + LLM';
                indicator.className = 'mode-indicator mode-elm-llm';
            }
            
            document.getElementById('trialCount').textContent = experimentState.trialCount;
            
            // Show/hide guidance section
            const guidanceSection = document.getElementById('guidanceSection');
            guidanceSection.style.display = experimentState.mode === 'elm_llm' ? 'block' : 'none';
            
            // Update experiment time
            if (experimentState.startTime) {
                const elapsed = Date.now() - experimentState.startTime;
                const minutes = Math.floor(elapsed / 60000);
                const seconds = Math.floor((elapsed % 60000) / 1000);
                document.getElementById('experimentTime').textContent = 
                    `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            }
        }
        
        // Update experiment results display
        function updateExperimentResults() {
            const resultsDiv = document.getElementById('experimentResults');
            
            if (experimentState.results.length === 0) {
                resultsDiv.innerHTML = '<p>実験データ収集中...</p>';
                return;
            }
            
            // Calculate statistics
            const elmOnlyResults = experimentState.results.filter(r => r.mode === 'elm_only');
            const elmLlmResults = experimentState.results.filter(r => r.mode === 'elm_llm');
            
            const avgScoreElmOnly = elmOnlyResults.length > 0 ? 
                elmOnlyResults.reduce((sum, r) => sum + r.score, 0) / elmOnlyResults.length : 0;
            const avgScoreElmLlm = elmLlmResults.length > 0 ? 
                elmLlmResults.reduce((sum, r) => sum + r.score, 0) / elmLlmResults.length : 0;
            
            resultsDiv.innerHTML = `
                <p><strong>ELM Only:</strong> ${elmOnlyResults.length}回, 平均スコア: ${avgScoreElmOnly.toFixed(1)}</p>
                <p><strong>ELM + LLM:</strong> ${elmLlmResults.length}回, 平均スコア: ${avgScoreElmLlm.toFixed(1)}</p>
                <p><strong>最新スコア:</strong> ${experimentState.results[experimentState.results.length - 1]?.score || 0}</p>
            `;
        }
        
        // Spawn enemy
        function spawnEnemy() {
            gameState.enemies.push({
                x: enemyPath[0].x,
                y: enemyPath[0].y,
                health: ENEMY_HEALTH,
                maxHealth: ENEMY_HEALTH,
                speed: ENEMY_SPEED,
                pathIndex: 0
            });
        }
        
        // Move enemies
        function moveEnemies() {
            for (let i = gameState.enemies.length - 1; i >= 0; i--) {
                const enemy = gameState.enemies[i];
                
                // Move towards next path point
                const target = enemyPath[1]; // Simple straight path
                const dx = target.x - enemy.x;
                const dy = target.y - enemy.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance > 5) {
                    enemy.x += (dx / distance) * enemy.speed;
                    enemy.y += (dy / distance) * enemy.speed;
                } else {
                    // Enemy reached end
                    gameState.health -= 10;
                    gameState.enemies.splice(i, 1);
                }
            }
        }
        
        // Tower attacks
        function towerAttacks() {
            gameState.towers.forEach(tower => {
                if (Date.now() - tower.lastAttack < ATTACK_INTERVAL) return;
                
                // Find nearest enemy in range
                let nearestEnemy = null;
                let nearestDistance = TOWER_RANGE;
                
                gameState.enemies.forEach(enemy => {
                    const dx = enemy.x - tower.x;
                    const dy = enemy.y - tower.y;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    
                    if (distance < nearestDistance) {
                        nearestEnemy = enemy;
                        nearestDistance = distance;
                    }
                });
                
                if (nearestEnemy) {
                    // Attack enemy
                    nearestEnemy.health -= TOWER_DAMAGE;
                    tower.lastAttack = Date.now();
                    
                    // Create projectile for visual effect
                    gameState.projectiles.push({
                        x: tower.x,
                        y: tower.y,
                        targetX: nearestEnemy.x,
                        targetY: nearestEnemy.y,
                        speed: 5
                    });
                    
                    // Check if enemy is killed
                    if (nearestEnemy.health <= 0) {
                        gameState.score += ENEMY_REWARD;
                        gameState.money += ENEMY_REWARD;
                        
                        // Remove enemy
                        const index = gameState.enemies.indexOf(nearestEnemy);
                        if (index > -1) {
                            gameState.enemies.splice(index, 1);
                        }
                    }
                }
            });
        }
        
        // ELM automation
        async function performELMAutomation() {
            if (Date.now() - gameState.lastAutomationTime < gameState.automationInterval) {
                return;
            }
            
            try {
                // Prepare game state for ELM
                const elmInput = {
                    money: gameState.money,
                    health: gameState.health,
                    wave: gameState.wave,
                    score: gameState.score,
                    towers: gameState.towers.length,
                    enemies: gameState.enemies.length,
                    mode: experimentState.mode
                };
                
                // Get ELM prediction
                const response = await fetch('/api/elm-predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(elmInput)
                });
                
                const prediction = await response.json();
                
                // Execute ELM decision
                if (prediction.should_place_tower && gameState.money >= TOWER_COST) {
                    // Calculate position based on ELM prediction
                    const x = 100 + prediction.position_x_ratio * 600;
                    const y = 100 + prediction.position_y_ratio * 400;
                    
                    placeTower(x, y);
                    
                    console.log('🤖 ELM自動配置:', {
                        mode: experimentState.mode,
                        should_place: prediction.should_place_tower,
                        position: {x, y}
                    });
                }
                
                gameState.lastAutomationTime = Date.now();
                
            } catch (error) {
                console.error('ELM automation error:', error);
            }
        }
        
        // Place tower
        function placeTower(x, y) {
            if (gameState.money >= TOWER_COST) {
                // Check if position is valid (not too close to path)
                const pathDistance = Math.abs(y - 300);
                if (pathDistance > 50) {
                    gameState.towers.push({
                        x: x,
                        y: y,
                        lastAttack: 0
                    });
                    gameState.money -= TOWER_COST;
                }
            }
        }
        
        // Get LLM guidance (only in elm_llm mode)
        async function getGuidance() {
            if (experimentState.mode !== 'elm_llm') return;
            
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
        
        // Draw game
        function drawGame() {
            // Clear canvas
            ctx.fillStyle = '#1f2937';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Draw path
            ctx.strokeStyle = '#6b7280';
            ctx.lineWidth = 40;
            ctx.beginPath();
            ctx.moveTo(enemyPath[0].x, enemyPath[0].y);
            ctx.lineTo(enemyPath[1].x, enemyPath[1].y);
            ctx.stroke();
            
            // Draw towers
            gameState.towers.forEach(tower => {
                ctx.fillStyle = '#3b82f6';
                ctx.fillRect(tower.x - 15, tower.y - 15, 30, 30);
                
                // Draw range circle
                ctx.strokeStyle = 'rgba(59, 130, 246, 0.3)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.arc(tower.x, tower.y, TOWER_RANGE, 0, 2 * Math.PI);
                ctx.stroke();
            });
            
            // Draw enemies
            gameState.enemies.forEach(enemy => {
                ctx.fillStyle = '#dc2626';
                ctx.fillRect(enemy.x - 10, enemy.y - 10, 20, 20);
                
                // Health bar
                const healthRatio = enemy.health / enemy.maxHealth;
                ctx.fillStyle = '#ef4444';
                ctx.fillRect(enemy.x - 15, enemy.y - 20, 30, 5);
                ctx.fillStyle = '#22c55e';
                ctx.fillRect(enemy.x - 15, enemy.y - 20, 30 * healthRatio, 5);
            });
            
            // Draw projectiles
            gameState.projectiles.forEach((projectile, index) => {
                ctx.fillStyle = '#fbbf24';
                ctx.beginPath();
                ctx.arc(projectile.x, projectile.y, 3, 0, 2 * Math.PI);
                ctx.fill();
                
                // Move projectile
                const dx = projectile.targetX - projectile.x;
                const dy = projectile.targetY - projectile.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance > projectile.speed) {
                    projectile.x += (dx / distance) * projectile.speed;
                    projectile.y += (dy / distance) * projectile.speed;
                } else {
                    gameState.projectiles.splice(index, 1);
                }
            });
        }
        
        // Update UI
        function updateUI() {
            document.getElementById('money').textContent = '$' + gameState.money;
            document.getElementById('health').textContent = gameState.health;
            document.getElementById('score').textContent = gameState.score;
            document.getElementById('wave').textContent = gameState.wave;
            document.getElementById('towers').textContent = gameState.towers.length;
            document.getElementById('enemies').textContent = gameState.enemies.length;
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
            updateExperimentUI();
            
            // Get guidance if enabled
            if (gameState.guidanceEnabled && Math.random() < 0.1) {
                getGuidance();
            }
            
            // Check game over
            if (gameState.health <= 0) {
                stopTrial();
                return;
            }
            
            requestAnimationFrame(gameLoop);
        }
        
        // Initialize
        initGame();
        updateExperimentUI();
        updateExperimentResults();
        
        // Update experiment time periodically
        setInterval(updateExperimentUI, 1000);
    </script>
</body>
</html>
    """)

@app.route('/api/elm-predict', methods=['POST'])
def elm_predict():
    """Get ELM prediction for tower placement"""
    try:
        data = request.json
        
        # Prepare input features
        features = [
            data['money'] / 1000.0,  # Normalized money
            data['health'] / 100.0,  # Normalized health
            data['wave'] / 10.0,     # Normalized wave
            data['score'] / 1000.0,  # Normalized score
            data['towers'] / 10.0,   # Normalized tower count
            data['enemies'] / 10.0,  # Normalized enemy count
            1.0 if data['mode'] == 'elm_llm' else 0.0,  # Mode indicator
            random.random()  # Random factor
        ]
        
        # Get LLM guidance if in elm_llm mode
        llm_guidance = None
        if data['mode'] == 'elm_llm':
            llm_guidance = get_real_llm_guidance({
                'money': data['money'],
                'health': data['health'],
                'wave': data['wave'],
                'score': data['score'],
                'towers': data['towers'],
                'enemies': data['enemies']
            })
        
        # Select appropriate ELM model
        if data['mode'] == 'elm_only':
            prediction = elm_only_model.predict(features)
        else:
            prediction = elm_llm_model.predict(features, llm_guidance)
        
        return jsonify({
            'should_place_tower': prediction[0] > 0.5,
            'position_x_ratio': prediction[1],
            'position_y_ratio': prediction[2],
            'raw_output': prediction,
            'llm_guidance_used': llm_guidance is not None
        })
        
    except Exception as e:
        print(f"ELM prediction error: {e}")
        return jsonify({
            'should_place_tower': False,
            'position_x_ratio': 0.5,
            'position_y_ratio': 0.5,
            'error': str(e)
        }), 500

@app.route('/api/llm-guidance', methods=['POST'])
def llm_guidance():
    """Get LLM guidance for the current game state"""
    try:
        data = request.json
        guidance = get_real_llm_guidance(data['game_state'])
        
        if guidance:
            return jsonify(guidance)
        else:
            return jsonify({
                'priority': 'medium',
                'recommendation': 'LLMガイダンスが利用できません',
                'reasoning': 'API接続エラーまたは設定問題'
            })
            
    except Exception as e:
        print(f"LLM guidance error: {e}")
        return jsonify({
            'priority': 'low',
            'recommendation': 'エラーが発生しました',
            'reasoning': str(e)
        }), 500

@app.route('/api/experiment-status')
def experiment_status():
    """Get current experiment status"""
    return jsonify(experiment_state)

if __name__ == '__main__':
    print("🚀 Tower Defense ELM Experiment Server Starting...")
    print(f"🔑 OpenAI API Key: {'✅ Configured' if OPENAI_API_KEY else '❌ Not configured'}")
    print("📊 Experiment ready for untrained ELM comparison")
    app.run(host='0.0.0.0', port=5000, debug=True)
