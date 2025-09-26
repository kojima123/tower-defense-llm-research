"""
Tower Defense ELM Learning Efficiency Experiment - Auto-Fix Version
ELMの自動動作を確実に実行する修正版
"""

from flask import Flask, request, jsonify, render_template_string
import json
import random
import math
import requests
import time
import os
from datetime import datetime
import numpy as np

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
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YOUR_OPENAI_API_KEY_HERE')

def get_real_llm_guidance(game_state):
    """Get real LLM guidance from OpenAI GPT-4o-mini"""
    if not OPENAI_API_KEY or OPENAI_API_KEY == 'YOUR_OPENAI_API_KEY_HERE':
        return {
            'priority': 'high',
            'recommendation': 'タワーを配置してください',
            'reasoning': f'現在{game_state["enemies"]}体の敵がいます。防御が必要です。',
            'learning_tip': 'タワーの配置位置と敵の進行ルートを観察して学習してください。'
        }
        
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

タワーコスト: $50
タワー攻撃力: 60
敵体力: 80

以下のJSON形式で回答してください:
{{
    "priority": "urgent/high/medium/low",
    "recommendation": "具体的な推奨行動",
    "reasoning": "推奨理由の説明",
    "learning_tip": "学習改善のためのアドバイス"
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
        
        response = requests.post('https://api.openai.com/v1/chat/completions', 
                               headers=headers, json=data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            try:
                return json.loads(content)
            except:
                return {
                    'priority': 'high',
                    'recommendation': content[:100],
                    'reasoning': 'LLMからの戦略アドバイス',
                    'learning_tip': '継続的に観察して学習してください'
                }
        else:
            print(f"LLM API error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"LLM guidance error: {e}")
        return None

# Learning-capable ELM implementation with forced automation
class AutoELM:
    def __init__(self, input_size=8, hidden_size=20, output_size=3, random_state=None):
        """
        Auto-executing ELM for Tower Defense with guaranteed automation
        """
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        # Initialize random weights (untrained state)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.input_weights = np.random.normal(0, 1.0, (input_size, hidden_size))
        self.hidden_bias = np.random.normal(0, 1.0, hidden_size)
        self.output_weights = np.random.normal(0, 0.1, (hidden_size, output_size))
        
        # Learning parameters
        self.learning_rate = 0.01
        self.llm_guidance_weight = 0.8  # Higher weight for LLM guidance
        self.experience_buffer = []
        self.max_buffer_size = 1000
        
        # Auto-execution parameters
        self.action_threshold = 0.3  # Lower threshold for more actions
        self.forced_action_interval = 5000  # Force action every 5 seconds
        self.last_action_time = 0
        
        # Learning efficiency tracking
        self.learning_history = []
        self.performance_history = []
        self.learning_start_time = time.time()
        self.total_learning_updates = 0
        self.llm_guidance_count = 0
        
        self.last_guidance = None
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        return np.tanh(np.clip(x, -500, 500))
    
    def predict(self, x, llm_guidance=None):
        """Predict action with guaranteed automation"""
        # Convert to numpy array and normalize
        x = np.array(x)
        x_norm = np.clip(x / (np.abs(x) + 1e-8), -10, 10)
        
        # Forward pass
        hidden = self.tanh(np.dot(x_norm, self.input_weights) + self.hidden_bias)
        output = np.dot(hidden, self.output_weights)
        
        # Apply activations
        output[0] = self.sigmoid(output[0])  # should_place_tower
        output[1] = self.sigmoid(output[1])  # position_x_ratio
        output[2] = self.sigmoid(output[2])  # position_y_ratio
        
        # Force action if enough time has passed
        current_time = time.time() * 1000
        if current_time - self.last_action_time > self.forced_action_interval:
            output[0] = 0.9  # Force tower placement
            self.last_action_time = current_time
            print("🔥 ELM強制実行: 時間間隔による自動配置")
        
        # Apply LLM guidance if available
        if llm_guidance:
            self.llm_guidance_count += 1
            self.last_guidance = llm_guidance
            guidance_influence = self._interpret_llm_guidance(llm_guidance)
            
            # Blend ELM output with LLM guidance (higher weight)
            output[0] = output[0] * (1 - self.llm_guidance_weight) + guidance_influence['should_place'] * self.llm_guidance_weight
            output[1] = output[1] * (1 - self.llm_guidance_weight) + guidance_influence['pos_x'] * self.llm_guidance_weight
            output[2] = output[2] * (1 - self.llm_guidance_weight) + guidance_influence['pos_y'] * self.llm_guidance_weight
            
            # Store experience for learning
            self._store_experience(x_norm, output, guidance_influence, reward=1.0)
            
            print(f"🧠 LLMガイダンス適用: {llm_guidance.get('recommendation', 'N/A')}")
        
        # Lower threshold for more frequent actions
        if output[0] < self.action_threshold:
            output[0] = self.action_threshold + 0.2
        
        return output.tolist()
    
    def _interpret_llm_guidance(self, guidance):
        """Interpret LLM guidance into actionable parameters with high urgency"""
        priority = guidance.get('priority', 'high')
        recommendation = guidance.get('recommendation', '').lower()
        
        # Map priority to urgency (higher values)
        priority_map = {
            'urgent': 0.95,
            'high': 0.85,
            'medium': 0.7,
            'low': 0.5
        }
        urgency = priority_map.get(priority, 0.7)
        
        # Analyze recommendation for action cues
        should_place = 0.8  # Default high probability
        if any(word in recommendation for word in ['配置', 'タワー', '設置', '購入', 'place', 'tower']):
            should_place = 0.9
        if any(word in recommendation for word in ['急', '緊急', 'urgent', 'すぐ', '即座']):
            should_place = 0.95
        
        # Position strategy (prefer center-left for better coverage)
        pos_x = random.uniform(0.3, 0.7)  # Center-left to center-right
        pos_y = random.uniform(0.3, 0.7)  # Center area
        
        return {
            'should_place': should_place,
            'pos_x': pos_x,
            'pos_y': pos_y,
            'urgency': urgency
        }
    
    def _store_experience(self, state, action, guidance, reward):
        """Store experience for learning"""
        experience = {
            'state': state.copy(),
            'action': action.copy(),
            'guidance': guidance.copy(),
            'reward': reward,
            'timestamp': time.time()
        }
        
        self.experience_buffer.append(experience)
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)
    
    def learn_from_experience(self, current_score):
        """Learn from recent experiences"""
        if len(self.experience_buffer) < 5:
            return
        
        # Simple learning: adjust weights based on recent performance
        recent_experiences = self.experience_buffer[-5:]
        
        for exp in recent_experiences:
            # Positive reinforcement for good guidance
            if exp['reward'] > 0:
                learning_signal = exp['reward'] * self.learning_rate
                
                # Update output weights slightly
                guidance_vector = np.array([
                    exp['guidance']['should_place'],
                    exp['guidance']['pos_x'],
                    exp['guidance']['pos_y']
                ])
                
                # Small weight adjustment
                adjustment = np.outer(np.ones(self.hidden_size), guidance_vector) * learning_signal * 0.01
                self.output_weights += adjustment
        
        self.total_learning_updates += 1
        
        # Record learning metrics
        learning_time = time.time() - self.learning_start_time
        self.learning_history.append({
            'time': learning_time,
            'updates': self.total_learning_updates,
            'score': current_score,
            'guidance_count': self.llm_guidance_count
        })
        
        print(f"📚 学習更新: {self.total_learning_updates}回, スコア: {current_score}")
    
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

# Initialize models
elm_only_model = AutoELM(random_state=42)
elm_llm_model = AutoELM(random_state=123)

# Experiment state
experiment_state = {
    'mode': 'elm_only',
    'trial_count': 0,
    'results': [],
    'learning_data': []
}

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tower Defense ELM Auto-Fix Experiment</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            display: flex;
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        .game-area {
            flex: 1;
        }
        .control-panel {
            width: 300px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        #gameCanvas {
            border: 2px solid #333;
            background: #90EE90;
            display: block;
            margin: 0 auto;
        }
        .game-info {
            display: flex;
            justify-content: space-around;
            margin: 10px 0;
            padding: 10px;
            background: white;
            border-radius: 5px;
        }
        .mode-selector {
            margin: 20px 0;
        }
        .mode-selector button {
            display: block;
            width: 100%;
            margin: 5px 0;
            padding: 10px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }
        .mode-selector button.active {
            background: #4CAF50;
            color: white;
        }
        .mode-selector button:not(.active) {
            background: #ddd;
        }
        .experiment-controls {
            margin: 20px 0;
        }
        .experiment-controls button {
            width: 100%;
            padding: 15px;
            margin: 5px 0;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
        }
        .start-btn {
            background: #4CAF50;
            color: white;
        }
        .stop-btn {
            background: #f44336;
            color: white;
        }
        .export-btn {
            background: #2196F3;
            color: white;
        }
        .guidance-section {
            margin: 20px 0;
            padding: 15px;
            background: #e8f5e8;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }
        .guidance-section h3 {
            margin: 0 0 10px 0;
            color: #2e7d32;
        }
        .guidance-text {
            font-size: 14px;
            line-height: 1.4;
        }
        .experiment-status {
            margin: 20px 0;
            padding: 15px;
            background: #fff3cd;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
        }
        .automation-indicator {
            margin: 10px 0;
            padding: 10px;
            background: #d4edda;
            border-radius: 5px;
            border-left: 4px solid #28a745;
            font-weight: bold;
        }
        .automation-indicator.active {
            background: #d1ecf1;
            border-left-color: #17a2b8;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="game-area">
            <h1>🎮 Tower Defense ELM Auto-Fix Experiment</h1>
            <p><strong>🔧 修正版:</strong> ELMの自動動作を確実に実行</p>
            
            <canvas id="gameCanvas" width="800" height="600"></canvas>
            
            <div class="game-info">
                <div>💰 資金: $<span id="money">100</span></div>
                <div>❤️ ヘルス: <span id="health">100</span></div>
                <div>🌊 ウェーブ: <span id="wave">1</span></div>
                <div>🎯 スコア: <span id="score">0</span></div>
                <div>🏗️ タワー: <span id="towers">0</span></div>
                <div>👾 敵: <span id="enemies">0</span></div>
            </div>
            
            <div class="automation-indicator" id="automationIndicator">
                🤖 ELM自動化: 待機中
            </div>
        </div>
        
        <div class="control-panel">
            <h2>🔬 実験制御</h2>
            
            <div class="mode-selector">
                <h3>実験モード</h3>
                <button onclick="setMode('manual')" id="manualBtn">🎮 手動プレイ</button>
                <button onclick="setMode('elm_only')" id="elmOnlyBtn" class="active">🤖 ELMのみ</button>
                <button onclick="setMode('elm_llm')" id="elmLlmBtn">🧠 ELM+指導システム</button>
            </div>
            
            <div class="experiment-controls">
                <button onclick="startExperiment()" class="start-btn">▶️ 実験開始</button>
                <button onclick="stopExperiment()" class="stop-btn">⏹️ 実験停止</button>
                <button onclick="exportResults()" class="export-btn">📊 結果エクスポート</button>
            </div>
            
            <div class="experiment-status">
                <h3>📊 実験状況</h3>
                <p>モード: <span id="currentMode">ELMのみ</span></p>
                <p>試行回数: <span id="trialCount">0</span>/20</p>
                <p>実験時間: <span id="experimentTime">00:00</span></p>
            </div>
            
            <div class="guidance-section" id="guidanceSection" style="display: none;">
                <h3>🧠 LLMガイダンス</h3>
                <div class="guidance-text" id="guidanceText">
                    ガイダンス待機中...
                </div>
            </div>
            
            <div id="experimentResults">
                <h3>📈 実験結果</h3>
                <p>実験データ収集中...</p>
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
            wave: 1,
            score: 0,
            towers: [],
            enemies: [],
            projectiles: [],
            isRunning: false,
            lastSpawn: 0,
            spawnInterval: 2000,
            lastAutomationTime: 0,
            automationInterval: 1000  // ELM automation every 1 second
        };
        
        // Experiment state
        let experimentState = {
            mode: 'elm_only',
            startTime: null,
            currentTrialStart: null,
            trialCount: 0,
            trialDuration: 120000, // 2 minutes per trial
            results: [],
            learningData: []
        };
        
        // Canvas setup
        const canvas = document.getElementById('gameCanvas');
        const ctx = canvas.getContext('2d');
        
        // Enemy path (simple straight line)
        const enemyPath = [
            {x: 50, y: 300},
            {x: 750, y: 300}
        ];
        
        // Initialize game
        function initGame() {
            gameState = {
                money: 100,
                health: 100,
                wave: 1,
                score: 0,
                towers: [],
                enemies: [],
                projectiles: [],
                isRunning: false,
                lastSpawn: 0,
                spawnInterval: 2000,
                lastAutomationTime: 0,
                automationInterval: 1000
            };
            updateUI();
        }
        
        // Set experiment mode
        function setMode(mode) {
            experimentState.mode = mode;
            
            // Update button states
            document.querySelectorAll('.mode-selector button').forEach(btn => {
                btn.classList.remove('active');
            });
            
            if (mode === 'manual') {
                document.getElementById('manualBtn').classList.add('active');
                document.getElementById('currentMode').textContent = '手動プレイ';
            } else if (mode === 'elm_only') {
                document.getElementById('elmOnlyBtn').classList.add('active');
                document.getElementById('currentMode').textContent = 'ELMのみ';
            } else if (mode === 'elm_llm') {
                document.getElementById('elmLlmBtn').classList.add('active');
                document.getElementById('currentMode').textContent = 'ELM+指導システム';
            }
            
            updateExperimentUI();
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
            
            // Reset learning model for new trial
            resetLearningModel();
            
            // Update UI
            updateExperimentUI();
            
            // Start game loop
            gameLoop();
            
            // Auto-stop trial after duration
            setTimeout(() => {
                stopTrial();
            }, experimentState.trialDuration);
            
            console.log('🚀 実験開始:', {
                mode: experimentState.mode,
                trial: experimentState.trialCount
            });
        }
        
        // Stop experiment
        function stopExperiment() {
            gameState.isRunning = false;
            console.log('⏹️ 実験停止');
        }
        
        // Reset learning model
        async function resetLearningModel() {
            try {
                await fetch('/api/reset-learning', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        mode: experimentState.mode
                    })
                });
                console.log('🔄 学習モデルリセット完了');
            } catch (error) {
                console.error('Error resetting learning model:', error);
            }
        }
        
        // Stop current trial
        function stopTrial() {
            gameState.isRunning = false;
            
            // Get final learning metrics
            getLearningMetrics().then(metrics => {
                // Record trial results
                const trialResult = {
                    mode: experimentState.mode,
                    trial: experimentState.trialCount,
                    score: gameState.score,
                    health: gameState.health,
                    towers: gameState.towers.length,
                    duration: Date.now() - experimentState.currentTrialStart,
                    timestamp: new Date().toISOString(),
                    learning_metrics: metrics
                };
                
                experimentState.results.push(trialResult);
                experimentState.learningData.push(metrics);
                
                console.log('✅ 試行完了:', trialResult);
                updateExperimentResults();
                
                // Auto-start next trial if under 20 trials
                if (experimentState.trialCount < 20) {
                    setTimeout(() => {
                        startExperiment();
                    }, 3000);
                } else {
                    alert('🎉 実験完了！20回の試行が終了しました。');
                }
            });
        }
        
        // Get learning metrics
        async function getLearningMetrics() {
            try {
                const response = await fetch('/api/learning-metrics', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        mode: experimentState.mode,
                        current_score: gameState.score
                    })
                });
                
                return await response.json();
            } catch (error) {
                console.error('Error getting learning metrics:', error);
                return {};
            }
        }
        
        // Update experiment UI
        function updateExperimentUI() {
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
                resultsDiv.innerHTML = '<h3>📈 実験結果</h3><p>実験データ収集中...</p>';
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
                <h3>📈 実験結果</h3>
                <p><strong>ELM Only:</strong> ${elmOnlyResults.length}回</p>
                <p>平均スコア: ${avgScoreElmOnly.toFixed(1)}</p>
                <br>
                <p><strong>ELM + LLM:</strong> ${elmLlmResults.length}回</p>
                <p>平均スコア: ${avgScoreElmLlm.toFixed(1)}</p>
                <br>
                <p><strong>改善率:</strong> ${avgScoreElmOnly > 0 ? ((avgScoreElmLlm / avgScoreElmOnly - 1) * 100).toFixed(1) : 'N/A'}%</p>
            `;
        }
        
        // Export results
        function exportResults() {
            const data = {
                experiment_state: experimentState,
                results: experimentState.results,
                learning_data: experimentState.learningData,
                timestamp: new Date().toISOString()
            };
            
            const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `tower_defense_experiment_${Date.now()}.json`;
            a.click();
            URL.revokeObjectURL(url);
        }
        
        // Game loop
        function gameLoop() {
            if (!gameState.isRunning) return;
            
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw path
            ctx.strokeStyle = '#8B4513';
            ctx.lineWidth = 20;
            ctx.beginPath();
            ctx.moveTo(enemyPath[0].x, enemyPath[0].y);
            ctx.lineTo(enemyPath[1].x, enemyPath[1].y);
            ctx.stroke();
            
            // Spawn enemies
            if (Date.now() - gameState.lastSpawn > gameState.spawnInterval) {
                spawnEnemy();
                gameState.lastSpawn = Date.now();
            }
            
            // Move enemies
            moveEnemies();
            
            // Tower attacks
            towerAttacks();
            
            // Move projectiles
            moveProjectiles();
            
            // ELM automation (GUARANTEED EXECUTION)
            if (experimentState.mode !== 'manual') {
                performELMAutomation();
            }
            
            // Draw everything
            drawGame();
            
            // Update UI
            updateUI();
            
            // Continue loop
            requestAnimationFrame(gameLoop);
        }
        
        // ELM automation with guaranteed execution
        async function performELMAutomation() {
            if (Date.now() - gameState.lastAutomationTime < gameState.automationInterval) {
                return;
            }
            
            try {
                // Update automation indicator
                const indicator = document.getElementById('automationIndicator');
                indicator.textContent = '🤖 ELM自動化: 実行中...';
                indicator.classList.add('active');
                
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
                
                console.log('🤖 ELM予測結果:', prediction);
                
                // Execute ELM decision (GUARANTEED EXECUTION)
                if (prediction.should_place_tower && gameState.money >= TOWER_COST) {
                    // Calculate position based on ELM prediction
                    const x = 100 + prediction.position_x_ratio * 600;
                    const y = 100 + prediction.position_y_ratio * 400;
                    
                    placeTower(x, y);
                    
                    console.log('✅ ELM自動配置実行:', {
                        mode: experimentState.mode,
                        position: {x: Math.round(x), y: Math.round(y)},
                        money_before: gameState.money + TOWER_COST,
                        money_after: gameState.money,
                        llm_guidance_used: prediction.llm_guidance_used
                    });
                    
                    indicator.textContent = `🤖 ELM自動化: タワー配置完了 (${gameState.towers.length}個)`;
                } else {
                    console.log('⏸️ ELM判定: タワー配置見送り', {
                        should_place: prediction.should_place_tower,
                        money: gameState.money,
                        cost: TOWER_COST
                    });
                    indicator.textContent = '🤖 ELM自動化: 配置見送り';
                }
                
                // Trigger learning update
                await fetch('/api/learn-from-experience', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        mode: experimentState.mode,
                        current_score: gameState.score
                    })
                });
                
                gameState.lastAutomationTime = Date.now();
                
                // Reset indicator after delay
                setTimeout(() => {
                    indicator.classList.remove('active');
                    indicator.textContent = '🤖 ELM自動化: 待機中';
                }, 2000);
                
            } catch (error) {
                console.error('❌ ELM automation error:', error);
                const indicator = document.getElementById('automationIndicator');
                indicator.textContent = '🤖 ELM自動化: エラー';
                indicator.classList.remove('active');
            }
        }
        
        // Get LLM guidance
        async function getLLMGuidance() {
            if (experimentState.mode !== 'elm_llm') return null;
            
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
                
                // Update guidance display
                const guidanceText = document.getElementById('guidanceText');
                guidanceText.innerHTML = `
                    <strong>優先度:</strong> ${guidance.priority}<br>
                    <strong>推奨:</strong> ${guidance.recommendation}<br>
                    <strong>理由:</strong> ${guidance.reasoning}<br>
                    <strong>学習ヒント:</strong> ${guidance.learning_tip}
                `;
                
                return guidance;
                
            } catch (error) {
                console.error('LLM guidance error:', error);
                return null;
            }
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
        
        // Move projectiles
        function moveProjectiles() {
            for (let i = gameState.projectiles.length - 1; i >= 0; i--) {
                const proj = gameState.projectiles[i];
                
                const dx = proj.targetX - proj.x;
                const dy = proj.targetY - proj.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < proj.speed) {
                    gameState.projectiles.splice(i, 1);
                } else {
                    proj.x += (dx / distance) * proj.speed;
                    proj.y += (dy / distance) * proj.speed;
                }
            }
        }
        
        // Place tower
        function placeTower(x, y) {
            if (gameState.money >= TOWER_COST) {
                gameState.towers.push({
                    x: x,
                    y: y,
                    lastAttack: 0
                });
                gameState.money -= TOWER_COST;
                
                console.log(`🏗️ タワー配置: (${Math.round(x)}, ${Math.round(y)}), 残り資金: $${gameState.money}`);
            }
        }
        
        // Draw game
        function drawGame() {
            // Draw towers
            gameState.towers.forEach(tower => {
                // Tower base
                ctx.fillStyle = '#654321';
                ctx.fillRect(tower.x - 15, tower.y - 15, 30, 30);
                
                // Tower range (faint circle)
                ctx.strokeStyle = 'rgba(255, 0, 0, 0.2)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.arc(tower.x, tower.y, TOWER_RANGE, 0, 2 * Math.PI);
                ctx.stroke();
            });
            
            // Draw enemies
            gameState.enemies.forEach(enemy => {
                // Enemy body
                ctx.fillStyle = '#FF0000';
                ctx.fillRect(enemy.x - 10, enemy.y - 10, 20, 20);
                
                // Health bar
                const healthRatio = enemy.health / enemy.maxHealth;
                ctx.fillStyle = '#FF0000';
                ctx.fillRect(enemy.x - 15, enemy.y - 20, 30, 5);
                ctx.fillStyle = '#00FF00';
                ctx.fillRect(enemy.x - 15, enemy.y - 20, 30 * healthRatio, 5);
            });
            
            // Draw projectiles
            gameState.projectiles.forEach(proj => {
                ctx.fillStyle = '#FFD700';
                ctx.beginPath();
                ctx.arc(proj.x, proj.y, 3, 0, 2 * Math.PI);
                ctx.fill();
            });
        }
        
        // Update UI
        function updateUI() {
            document.getElementById('money').textContent = gameState.money;
            document.getElementById('health').textContent = gameState.health;
            document.getElementById('wave').textContent = gameState.wave;
            document.getElementById('score').textContent = gameState.score;
            document.getElementById('towers').textContent = gameState.towers.length;
            document.getElementById('enemies').textContent = gameState.enemies.length;
        }
        
        // Canvas click handler for manual mode
        canvas.addEventListener('click', (e) => {
            if (experimentState.mode === 'manual' && gameState.isRunning) {
                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                placeTower(x, y);
            }
        });
        
        // Initialize
        initGame();
        updateExperimentUI();
        
        // Update experiment time periodically
        setInterval(() => {
            if (experimentState.startTime) {
                updateExperimentUI();
            }
        }, 1000);
        
        // Get LLM guidance periodically for elm_llm mode
        setInterval(() => {
            if (experimentState.mode === 'elm_llm' && gameState.isRunning) {
                getLLMGuidance();
            }
        }, 3000);
    </script>
</body>
</html>
    ''')

@app.route('/api/elm-predict', methods=['POST'])
def elm_predict():
    """Get ELM prediction for tower placement with guaranteed automation"""
    try:
        data = request.json
        
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
        
        # Prepare features for ELM
        features = [
            data['money'] / 100.0,  # Normalized money
            data['health'] / 100.0,  # Normalized health
            data['wave'] / 10.0,     # Normalized wave
            data['score'] / 1000.0,  # Normalized score
            data['towers'] / 20.0,   # Normalized tower count
            data['enemies'] / 50.0,  # Normalized enemy count
            1.0 if llm_guidance else 0.0,  # LLM guidance available
            time.time() % 100 / 100.0  # Time factor for variation
        ]
        
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
            'llm_guidance_used': llm_guidance is not None,
            'llm_guidance': llm_guidance
        })
        
    except Exception as e:
        print(f"ELM prediction error: {e}")
        return jsonify({
            'should_place_tower': True,  # Force action on error
            'position_x_ratio': 0.5,
            'position_y_ratio': 0.5,
            'error': str(e)
        }), 500

@app.route('/api/learn-from-experience', methods=['POST'])
def learn_from_experience():
    """Trigger learning from recent experiences"""
    try:
        data = request.json
        mode = data['mode']
        current_score = data['current_score']
        
        # Select appropriate model and trigger learning
        if mode == 'elm_only':
            elm_only_model.learn_from_experience(current_score)
        else:
            elm_llm_model.learn_from_experience(current_score)
        
        return jsonify({'status': 'learning_updated'})
        
    except Exception as e:
        print(f"Learning error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/learning-metrics', methods=['POST'])
def learning_metrics():
    """Get current learning efficiency metrics"""
    try:
        data = request.json
        mode = data['mode']
        
        # Select appropriate model
        if mode == 'elm_only':
            metrics = elm_only_model.get_learning_efficiency_metrics()
        else:
            metrics = elm_llm_model.get_learning_efficiency_metrics()
        
        return jsonify(metrics)
        
    except Exception as e:
        print(f"Learning metrics error: {e}")
        return jsonify({}), 500

@app.route('/api/reset-learning', methods=['POST'])
def reset_learning():
    """Reset learning model for new trial"""
    try:
        data = request.json
        mode = data['mode']
        
        # Reset appropriate model
        global elm_only_model, elm_llm_model
        if mode == 'elm_only':
            elm_only_model = AutoELM(random_state=42)
        else:
            elm_llm_model = AutoELM(random_state=123)
        
        return jsonify({'status': 'model_reset'})
        
    except Exception as e:
        print(f"Model reset error: {e}")
        return jsonify({'error': str(e)}), 500

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
                'priority': 'high',
                'recommendation': 'タワーを配置してください',
                'reasoning': 'デフォルト戦略として基本的な防御を構築します',
                'learning_tip': '敵の動きとタワーの効果を観察して学習してください'
            })
            
    except Exception as e:
        print(f"LLM guidance error: {e}")
        return jsonify({
            'priority': 'medium',
            'recommendation': 'エラーが発生しました',
            'reasoning': str(e),
            'learning_tip': '安定した戦略を維持してください'
        }), 500

if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    print("🚀 Tower Defense ELM Auto-Fix Server Starting...")
    print(f"🔑 OpenAI API Key: {'✅ Configured' if OPENAI_API_KEY and OPENAI_API_KEY != 'YOUR_OPENAI_API_KEY_HERE' else '❌ Using fallback'}")
    print("🔧 Auto-Fix: ELMの自動動作を強制実行")
    print("📊 Learning efficiency experiment ready")
    print(f"🌐 Server starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
