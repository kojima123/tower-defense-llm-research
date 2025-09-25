#!/usr/bin/env python3
"""
Flask server for Tower Defense LLM Trainer - Test LLM Integration
Handles OpenAI GPT integration for testing purposes
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import time
import random
import math
import urllib.request
import urllib.parse

app = Flask(__name__, static_folder='../static', static_url_path='')
CORS(app)

# Test API key (for demonstration purposes)
TEST_OPENAI_API_KEY = "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"

# Get the absolute path to the project directory
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(PROJECT_DIR, 'static')

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
    
    def update(self, x, target, learning_rate=None):
        if learning_rate is None:
            learning_rate = self.learning_rate
        
        # Enhanced learning with LLM guidance consideration
        prediction = self.predict(x, self.last_guidance)
        error = [target[i] - prediction[i] for i in range(len(target))]
        
        # Adjust learning rate based on LLM guidance confidence
        if self.last_guidance:
            priority = self.last_guidance.get('priority', 'medium')
            if priority == 'urgent':
                learning_rate *= 1.5  # Learn faster for urgent situations
            elif priority == 'high':
                learning_rate *= 1.2
        
        # Update output weights
        for i in range(len(self.output_weights)):
            for j in range(len(self.output_weights[i])):
                self.output_weights[i][j] += learning_rate * error[j] * 0.1

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
    
    def update(self, x, target, learning_rate=None):
        if learning_rate is None:
            learning_rate = self.learning_rate
        
        # Simple weight update
        prediction = self.predict(x)
        error = [target[i] - prediction[i] for i in range(len(target))]
        
        # Update output weights
        for i in range(len(self.output_weights)):
            for j in range(len(self.output_weights[i])):
                self.output_weights[i][j] += learning_rate * error[j] * 0.1

# Global model instances
baseline_elm = SimpleTowerDefenseELM(random_state=42)
llm_guided_elm = LLMGuidedTowerDefenseELM(random_state=43)

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
    
    # Return a simple message if no static file is found
    return """
    <h1>Tower Defense LLM Trainer</h1>
    <p>Game server is running with REAL LLM integration!</p>
    <p>OpenAI API Key configured: True</p>
    <p>Testing actual GPT-4o-mini integration...</p>
    """

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'llm_integration': 'enabled',
        'openai_configured': True,
        'api_key_length': len(TEST_OPENAI_API_KEY),
        'model': 'gpt-4o-mini',
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
    print("🚀 Starting Tower Defense LLM Trainer with REAL OpenAI integration...")
    print(f"✅ OpenAI API configured: True")
    print(f"🔑 API key length: {len(TEST_OPENAI_API_KEY)}")
    print(f"🤖 Model: gpt-4o-mini")
    app.run(host='0.0.0.0', port=5000, debug=True)
