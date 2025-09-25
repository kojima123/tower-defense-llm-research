#!/usr/bin/env python3
"""
Learning ELM System for Tower Defense
Implements actual learning with experience replay and LLM guidance integration
"""

import numpy as np
import random
import json
import time
from typing import List, Dict, Tuple, Optional
from collections import deque
import requests

class LearningTowerDefenseELM:
    """ELM with actual learning capabilities and LLM guidance integration"""
    
    def __init__(self, input_size: int = 8, hidden_size: int = 100, output_size: int = 3, 
                 learning_rate: float = 0.01, memory_size: int = 1000):
        """
        Initialize learning ELM
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden neurons
            output_size: Number of outputs (placement_prob, pos_x, pos_y)
            learning_rate: Learning rate for weight updates
            memory_size: Size of experience replay buffer
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights
        self.input_weights = np.random.randn(input_size, hidden_size) * 0.1
        self.output_weights = np.random.randn(hidden_size, output_size) * 0.1
        self.hidden_bias = np.random.randn(hidden_size) * 0.1
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Learning statistics
        self.learning_stats = {
            'episodes': 0,
            'total_reward': 0,
            'avg_reward': 0,
            'learning_updates': 0,
            'llm_guidance_used': 0
        }
        
        # Performance tracking
        self.performance_history = []
        
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""
        # Input to hidden
        hidden_input = np.dot(x, self.input_weights) + self.hidden_bias
        hidden_output = self.sigmoid(hidden_input)
        
        # Hidden to output
        output = np.dot(hidden_output, self.output_weights)
        
        # Apply appropriate activations
        output[0] = self.sigmoid(output[0])  # Placement probability [0,1]
        output[1] = self.sigmoid(output[1])  # Position X [0,1]
        output[2] = self.sigmoid(output[2])  # Position Y [0,1]
        
        return output, hidden_output
    
    def predict(self, features: List[float], llm_guidance: Optional[Dict] = None) -> np.ndarray:
        """Make prediction with optional LLM guidance"""
        x = np.array(features)
        output, _ = self.forward(x)
        
        # Apply LLM guidance if available
        if llm_guidance:
            self.learning_stats['llm_guidance_used'] += 1
            output = self._apply_llm_guidance(output, llm_guidance)
        
        return output
    
    def _apply_llm_guidance(self, base_output: np.ndarray, llm_guidance: Dict) -> np.ndarray:
        """Apply LLM guidance to base ELM output"""
        guided_output = base_output.copy()
        
        # Extract guidance signals
        should_place = llm_guidance.get('should_place', 0.5)
        pos_x = llm_guidance.get('pos_x', 0.5)
        pos_y = llm_guidance.get('pos_y', 0.5)
        
        # Blend ELM output with LLM guidance (weighted average)
        guidance_weight = 0.3  # How much to trust LLM vs ELM
        
        guided_output[0] = (1 - guidance_weight) * base_output[0] + guidance_weight * should_place
        guided_output[1] = (1 - guidance_weight) * base_output[1] + guidance_weight * pos_x
        guided_output[2] = (1 - guidance_weight) * base_output[2] + guidance_weight * pos_y
        
        return guided_output
    
    def store_experience(self, state: List[float], action: List[float], 
                        reward: float, next_state: List[float], done: bool):
        """Store experience in replay buffer"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'timestamp': time.time()
        }
        self.memory.append(experience)
    
    def learn_from_experience(self, batch_size: int = 32):
        """Learn from stored experiences using experience replay"""
        if len(self.memory) < batch_size:
            return
        
        # Sample random batch
        batch = random.sample(self.memory, batch_size)
        
        total_loss = 0
        for experience in batch:
            state = np.array(experience['state'])
            action = np.array(experience['action'])
            reward = experience['reward']
            next_state = np.array(experience['next_state'])
            done = experience['done']
            
            # Forward pass
            current_output, hidden = self.forward(state)
            
            # Calculate target (simplified Q-learning style)
            if done:
                target = reward
            else:
                next_output, _ = self.forward(next_state)
                target = reward + 0.9 * np.max(next_output)  # Gamma = 0.9
            
            # Calculate loss (mean squared error)
            target_output = current_output.copy()
            target_output[0] = target  # Update placement probability based on reward
            
            loss = np.mean((current_output - target_output) ** 2)
            total_loss += loss
            
            # Backpropagation (simplified)
            output_error = current_output - target_output
            
            # Update output weights
            self.output_weights -= self.learning_rate * np.outer(hidden, output_error)
            
            # Update input weights (simplified backprop)
            hidden_error = np.dot(output_error, self.output_weights.T) * hidden * (1 - hidden)
            self.input_weights -= self.learning_rate * np.outer(state, hidden_error)
            self.hidden_bias -= self.learning_rate * hidden_error
        
        self.learning_stats['learning_updates'] += 1
        avg_loss = total_loss / batch_size
        
        return avg_loss
    
    def update_performance(self, episode_reward: float, episode_score: int):
        """Update performance statistics"""
        self.learning_stats['episodes'] += 1
        self.learning_stats['total_reward'] += episode_reward
        self.learning_stats['avg_reward'] = self.learning_stats['total_reward'] / self.learning_stats['episodes']
        
        self.performance_history.append({
            'episode': self.learning_stats['episodes'],
            'reward': episode_reward,
            'score': episode_score,
            'avg_reward': self.learning_stats['avg_reward']
        })
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def get_learning_stats(self) -> Dict:
        """Get current learning statistics"""
        recent_performance = self.performance_history[-10:] if self.performance_history else []
        recent_avg_score = np.mean([p['score'] for p in recent_performance]) if recent_performance else 0
        
        return {
            **self.learning_stats,
            'recent_avg_score': recent_avg_score,
            'memory_size': len(self.memory),
            'performance_trend': self._calculate_trend()
        }
    
    def _calculate_trend(self) -> str:
        """Calculate performance trend"""
        if len(self.performance_history) < 5:
            return "insufficient_data"
        
        recent_scores = [p['score'] for p in self.performance_history[-5:]]
        older_scores = [p['score'] for p in self.performance_history[-10:-5]] if len(self.performance_history) >= 10 else recent_scores
        
        recent_avg = np.mean(recent_scores)
        older_avg = np.mean(older_scores)
        
        if recent_avg > older_avg * 1.1:
            return "improving"
        elif recent_avg < older_avg * 0.9:
            return "declining"
        else:
            return "stable"

class TowerDefenseGameSimulator:
    """Simplified game simulator for training"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset game to initial state"""
        self.state = {
            'money': 250,
            'health': 100,
            'wave': 1,
            'score': 0,
            'towers': 0,
            'enemies': 3,
            'efficiency': 0,
            'survival': 1
        }
        self.step_count = 0
        self.max_steps = 100
        return self._get_features()
    
    def _get_features(self) -> List[float]:
        """Convert game state to feature vector"""
        return [
            self.state['money'] / 1000.0,
            self.state['health'] / 100.0,
            self.state['wave'] / 10.0,
            self.state['enemies'] / 10.0,
            self.state['towers'] / 10.0,
            self.state['efficiency'],
            self.state['survival'],
            self.step_count / self.max_steps
        ]
    
    def step(self, action: np.ndarray) -> Tuple[List[float], float, bool]:
        """Execute one game step"""
        placement_prob, pos_x, pos_y = action
        
        reward = 0
        
        # Decide whether to place tower based on probability
        if placement_prob > 0.5 and self.state['money'] >= 50:
            # Place tower
            self.state['towers'] += 1
            self.state['money'] -= 50
            
            # Simulate tower effectiveness based on position
            effectiveness = self._calculate_tower_effectiveness(pos_x, pos_y)
            
            # Simulate enemy kills
            kills = min(int(effectiveness * 3), self.state['enemies'])
            self.state['score'] += kills * 30
            self.state['money'] += kills * 30
            self.state['enemies'] = max(1, self.state['enemies'] - kills + 1)
            
            reward += kills * 10  # Reward for kills
            reward += effectiveness * 5  # Reward for good positioning
        
        # Simulate game progression
        self.step_count += 1
        self.state['wave'] += 0.1
        self.state['enemies'] += random.randint(0, 2)
        
        # Calculate efficiency
        if self.state['towers'] > 0:
            self.state['efficiency'] = self.state['score'] / (self.state['towers'] * 50)
        
        # Health loss simulation
        if self.state['towers'] < self.state['enemies'] / 3:
            health_loss = random.randint(1, 5)
            self.state['health'] = max(0, self.state['health'] - health_loss)
            reward -= health_loss  # Penalty for health loss
        
        # Update survival
        self.state['survival'] = self.state['health'] / 100.0
        
        # Check if done
        done = (self.state['health'] <= 0 or 
                self.step_count >= self.max_steps or
                self.state['score'] > 1000)
        
        # Final reward calculation
        if done:
            if self.state['health'] > 0:
                reward += self.state['score'] / 10  # Bonus for surviving
            else:
                reward -= 50  # Penalty for dying
        
        return self._get_features(), reward, done
    
    def _calculate_tower_effectiveness(self, pos_x: float, pos_y: float) -> float:
        """Calculate tower effectiveness based on position"""
        # Simplified effectiveness calculation
        # Better positions are away from edges and in strategic locations
        center_distance = abs(pos_x - 0.5) + abs(pos_y - 0.5)
        edge_penalty = min(pos_x, pos_y, 1-pos_x, 1-pos_y)
        
        effectiveness = (1 - center_distance) * edge_penalty * 2
        return max(0.1, min(1.0, effectiveness))

def train_elm_with_llm_guidance(base_url: str, episodes: int = 50):
    """Train ELM with LLM guidance"""
    
    # Initialize systems
    elm_only = LearningTowerDefenseELM()
    elm_with_llm = LearningTowerDefenseELM()
    simulator = TowerDefenseGameSimulator()
    
    results = {
        'elm_only': [],
        'elm_with_llm': []
    }
    
    print(f"Training ELM systems for {episodes} episodes...")
    print("=" * 60)
    
    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}")
        
        # Train ELM only
        state = simulator.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = elm_only.predict(state)
            next_state, reward, done = simulator.step(action)
            
            elm_only.store_experience(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        elm_only.update_performance(total_reward, simulator.state['score'])
        elm_only.learn_from_experience()
        
        results['elm_only'].append({
            'episode': episode + 1,
            'score': simulator.state['score'],
            'reward': total_reward,
            'steps': steps,
            'towers': simulator.state['towers']
        })
        
        # Train ELM with LLM guidance
        state = simulator.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Get LLM guidance (simplified)
            llm_guidance = None
            if random.random() < 0.3:  # 30% chance to get LLM guidance
                try:
                    # Simulate LLM guidance request
                    game_state = {
                        'money': int(state[0] * 1000),
                        'health': int(state[1] * 100),
                        'wave': int(state[2] * 10),
                        'enemies': int(state[3] * 10),
                        'towers': int(state[4] * 10)
                    }
                    
                    response = requests.post(
                        f"{base_url}/api/llm-guidance",
                        json={'game_state': game_state},
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        guidance_data = response.json()
                        llm_guidance = {
                            'should_place': 0.8 if 'タワー' in guidance_data.get('recommendation', '') else 0.3,
                            'pos_x': 0.3 + random.random() * 0.4,
                            'pos_y': 0.3 + random.random() * 0.4
                        }
                except:
                    pass
            
            action = elm_with_llm.predict(state, llm_guidance)
            next_state, reward, done = simulator.step(action)
            
            # Bonus reward for using LLM guidance effectively
            if llm_guidance and reward > 0:
                reward += 2
            
            elm_with_llm.store_experience(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        elm_with_llm.update_performance(total_reward, simulator.state['score'])
        elm_with_llm.learn_from_experience()
        
        results['elm_with_llm'].append({
            'episode': episode + 1,
            'score': simulator.state['score'],
            'reward': total_reward,
            'steps': steps,
            'towers': simulator.state['towers']
        })
        
        # Print progress
        if (episode + 1) % 10 == 0:
            elm_only_stats = elm_only.get_learning_stats()
            elm_llm_stats = elm_with_llm.get_learning_stats()
            
            print(f"  ELM Only - Avg Score: {elm_only_stats['recent_avg_score']:.1f}, Trend: {elm_only_stats['performance_trend']}")
            print(f"  ELM+LLM  - Avg Score: {elm_llm_stats['recent_avg_score']:.1f}, Trend: {elm_llm_stats['performance_trend']}")
            print()
    
    return results, elm_only, elm_with_llm

def analyze_learning_results(results: Dict, elm_only, elm_with_llm) -> str:
    """Analyze learning results and generate report"""
    
    # Calculate statistics
    elm_only_scores = [r['score'] for r in results['elm_only']]
    elm_llm_scores = [r['score'] for r in results['elm_with_llm']]
    
    elm_only_final = np.mean(elm_only_scores[-10:])
    elm_llm_final = np.mean(elm_llm_scores[-10:])
    
    elm_only_initial = np.mean(elm_only_scores[:10])
    elm_llm_initial = np.mean(elm_llm_scores[:10])
    
    # Get learning statistics
    elm_only_stats = elm_only.get_learning_stats()
    elm_llm_stats = elm_with_llm.get_learning_stats()
    
    report = []
    report.append("# Learning ELM System Analysis Report")
    report.append("=" * 50)
    report.append("")
    
    report.append("## Performance Comparison")
    report.append("")
    report.append(f"**ELM Only:**")
    report.append(f"- Initial Performance (Episodes 1-10): {elm_only_initial:.1f}")
    report.append(f"- Final Performance (Episodes 41-50): {elm_only_final:.1f}")
    report.append(f"- Improvement: {elm_only_final - elm_only_initial:.1f} ({((elm_only_final/elm_only_initial-1)*100):.1f}%)")
    report.append(f"- Learning Trend: {elm_only_stats['performance_trend']}")
    report.append("")
    
    report.append(f"**ELM + LLM Guidance:**")
    report.append(f"- Initial Performance (Episodes 1-10): {elm_llm_initial:.1f}")
    report.append(f"- Final Performance (Episodes 41-50): {elm_llm_final:.1f}")
    report.append(f"- Improvement: {elm_llm_final - elm_llm_initial:.1f} ({((elm_llm_final/elm_llm_initial-1)*100):.1f}%)")
    report.append(f"- Learning Trend: {elm_llm_stats['performance_trend']}")
    report.append(f"- LLM Guidance Usage: {elm_llm_stats['llm_guidance_used']} times")
    report.append("")
    
    report.append("## Learning Statistics")
    report.append("")
    report.append(f"**ELM Only:**")
    report.append(f"- Total Episodes: {elm_only_stats['episodes']}")
    report.append(f"- Learning Updates: {elm_only_stats['learning_updates']}")
    report.append(f"- Memory Size: {elm_only_stats['memory_size']}")
    report.append(f"- Average Reward: {elm_only_stats['avg_reward']:.2f}")
    report.append("")
    
    report.append(f"**ELM + LLM:**")
    report.append(f"- Total Episodes: {elm_llm_stats['episodes']}")
    report.append(f"- Learning Updates: {elm_llm_stats['learning_updates']}")
    report.append(f"- Memory Size: {elm_llm_stats['memory_size']}")
    report.append(f"- Average Reward: {elm_llm_stats['avg_reward']:.2f}")
    report.append(f"- LLM Guidance Rate: {(elm_llm_stats['llm_guidance_used']/elm_llm_stats['episodes']*100):.1f}%")
    report.append("")
    
    # Comparative analysis
    improvement_difference = (elm_llm_final - elm_llm_initial) - (elm_only_final - elm_only_initial)
    final_performance_difference = elm_llm_final - elm_only_final
    
    report.append("## Comparative Analysis")
    report.append("")
    report.append(f"**Learning Efficiency:**")
    report.append(f"- ELM+LLM vs ELM Only improvement difference: {improvement_difference:.1f}")
    report.append(f"- Final performance difference: {final_performance_difference:.1f}")
    report.append("")
    
    if final_performance_difference > 10:
        conclusion = "LLM guidance shows significant positive effect on learning"
    elif final_performance_difference > 0:
        conclusion = "LLM guidance shows modest positive effect on learning"
    elif abs(final_performance_difference) < 10:
        conclusion = "No significant difference between learning approaches"
    else:
        conclusion = "ELM only performs better (possible LLM guidance interference)"
    
    report.append(f"**Conclusion:** {conclusion}")
    
    return "\n".join(report)

if __name__ == "__main__":
    base_url = "https://kkh7ikc7ewn0.manus.space"
    
    print("Starting Learning ELM Training with LLM Guidance")
    print("This will train both ELM-only and ELM+LLM systems with actual learning")
    print()
    
    # Run training
    results, elm_only, elm_with_llm = train_elm_with_llm_guidance(base_url, episodes=50)
    
    # Save results
    with open('/home/ubuntu/tower-defense-llm/learning_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate analysis report
    report = analyze_learning_results(results, elm_only, elm_with_llm)
    with open('/home/ubuntu/tower-defense-llm/learning_analysis_report.md', 'w') as f:
        f.write(report)
    
    print("\n" + "="*60)
    print("LEARNING ANALYSIS COMPLETE")
    print("="*60)
    print(report)
