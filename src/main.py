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
    try:
        # Try to serve from static directory first
        with open('../static/index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except:
        try:
            # Fallback to static.html in root directory
            with open('../static.html', 'r', encoding='utf-8') as f:
                return f.read()
        except:
            return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tower Defense LLM Trainer</title>
</head>
<body>
    <h1>Tower Defense LLM Trainer</h1>
    <p>ゲームを読み込み中...</p>
    <p>静的ファイルの読み込みに問題が発生しました。</p>
</body>
</html>
            """

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'openai_available': False,
        'timestamp': time.time()
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
    print("🎮 Game available at: http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
