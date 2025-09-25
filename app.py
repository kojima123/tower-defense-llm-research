#!/usr/bin/env python3
"""
Flask server for Tower Defense LLM Trainer - Deployment Version
Handles LLM guidance requests and ELM integration for tower defense game
"""

from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
import os
import json
import time
import random
import numpy as np
from openai import OpenAI

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Initialize OpenAI client
client = None
if os.getenv('OPENAI_API_KEY'):
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        print("✅ OpenAI client initialized")
    except Exception as e:
        print(f"❌ OpenAI initialization failed: {e}")

# Tower Defense ELM implementation
class TowerDefenseELM:
    def __init__(self, input_size=8, hidden_size=20, output_size=2, random_state=42):
        """
        ELM for Tower Defense strategy
        Input: [money, health, wave, enemies, towers, efficiency, survival, progress]
        Output: [place_tower_probability, tower_x_position]
        """
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
        # Normalize inputs
        x = x / np.maximum(np.abs(x), 1e-8)
        
        hidden = self.tanh(np.dot(x, self.input_weights) + self.hidden_bias)
        output = np.dot(hidden, self.output_weights)
        
        # Apply sigmoid to tower placement probability
        output[0, 0] = self.sigmoid(output[0, 0])
        # Normalize tower position to [0, 1]
        output[0, 1] = self.sigmoid(output[0, 1])
        
        return output[0]
    
    def update(self, x, target, learning_rate=None):
        if learning_rate is None:
            learning_rate = self.learning_rate
            
        x = np.array(x).reshape(1, -1)
        target = np.array(target).reshape(1, -1)
        
        # Normalize inputs
        x = x / np.maximum(np.abs(x), 1e-8)
        
        # Forward pass
        hidden = self.tanh(np.dot(x, self.input_weights) + self.hidden_bias)
        output = np.dot(hidden, self.output_weights)
        
        # Apply activations
        output[0, 0] = self.sigmoid(output[0, 0])
        output[0, 1] = self.sigmoid(output[0, 1])
        
        # Simple gradient update
        error = target - output
        self.output_weights += learning_rate * np.dot(hidden.T, error)

# Global model instances
baseline_elm = TowerDefenseELM(random_state=42)
llm_guided_elm = TowerDefenseELM(random_state=43)

# Game state tracking
game_sessions = {}

@app.route('/')
def index():
    """Serve the main game page"""
    try:
        return send_from_directory('static', 'index.html')
    except:
        # Fallback to static.html in root directory
        return send_from_directory('.', 'static.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'openai_available': client is not None,
        'timestamp': time.time()
    })

@app.route('/api/llm-guidance', methods=['POST'])
def get_llm_guidance():
    """Get strategic guidance from LLM based on current game state"""
    try:
        data = request.json
        game_state = data['game_state']
        
        if not client:
            # Fallback to rule-based guidance
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
- 効率性: {game_state.get('efficiency', 0):.2f}
- 生存率: {game_state.get('survival', 1):.2f}

タワーコスト: $50
タワーダメージ: 60
タワー射程: 150

以下の形式で回答してください:
優先度: [urgent/high/medium/low]
推奨行動: [具体的な行動]
理由: [戦略的な理由]
"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse LLM response
            lines = content.split('\n')
            priority = 'medium'
            recommendation = 'タワーを配置してください'
            reasoning = '防御を強化する必要があります'
            
            for line in lines:
                if '優先度:' in line:
                    priority_text = line.split(':', 1)[1].strip()
                    if 'urgent' in priority_text or '緊急' in priority_text:
                        priority = 'urgent'
                    elif 'high' in priority_text or '高' in priority_text:
                        priority = 'high'
                    elif 'low' in priority_text or '低' in priority_text:
                        priority = 'low'
                elif '推奨行動:' in line or '推奨' in line:
                    recommendation = line.split(':', 1)[1].strip()
                elif '理由:' in line:
                    reasoning = line.split(':', 1)[1].strip()
            
            return jsonify({
                'recommendation': recommendation,
                'reasoning': reasoning,
                'priority': priority,
                'source': 'llm'
            })
            
        except Exception as e:
            print(f"LLM guidance failed: {e}")
            return get_rule_based_guidance(game_state)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
    baseline_elm = TowerDefenseELM(random_state=42)
    llm_guided_elm = TowerDefenseELM(random_state=43)
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    print("🚀 Starting Tower Defense LLM Trainer Server (Deployment)")
    print(f"🔑 OpenAI API: {'✅ Available' if client else '❌ Not available'}")
    print("🎮 Game available at: http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
