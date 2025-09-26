"""
実測専用LLMTeacherクラス
実際のOpenAI APIコールのみを実行し、合成データを一切使用しない
"""
import os
import time
import json
import hashlib
import requests
from typing import Dict, List, Optional, Any


class LLMTeacher:
    """実測専用LLM教師システム"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"
        
        # 実測統計
        self.api_calls = 0
        self.total_api_time = 0.0
        self.successful_calls = 0
        self.failed_calls = 0
        
        # ログ用
        self.last_prompt = ""
        self.last_response = ""
        self.last_prompt_id = ""
        
        if not self.api_key:
            print("Warning: No OpenAI API key provided. LLM teacher will use fallback mode.")
    
    def evaluate_state_and_recommend(self, env) -> str:
        """ゲーム状態を評価し、推奨行動を返す（実測のみ）"""
        if not self.api_key:
            return self._fallback_evaluation(env)
        
        # ゲーム状況の分析
        game_analysis = self._analyze_game_situation(env)
        
        # プロンプト構築
        prompt = self._build_strategic_prompt(env, game_analysis)
        self.last_prompt = prompt
        self.last_prompt_id = hashlib.md5(prompt.encode()).hexdigest()[:8]
        
        # OpenAI API呼び出し
        start_time = time.time()
        
        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 150,
                    "timeout": 10
                },
                timeout=15
            )
            
            api_time = time.time() - start_time
            self.total_api_time += api_time
            self.api_calls += 1
            
            if response.status_code == 200:
                result = response.json()
                evaluation = result['choices'][0]['message']['content'].strip()
                self.last_response = evaluation
                self.successful_calls += 1
                
                return evaluation
            else:
                print(f"API Error {response.status_code}: {response.text}")
                self.failed_calls += 1
                return self._fallback_evaluation(env)
                
        except Exception as e:
            api_time = time.time() - start_time
            self.total_api_time += api_time
            self.api_calls += 1
            self.failed_calls += 1
            
            print(f"LLM API call failed: {e}")
            return self._fallback_evaluation(env)
    
    def _analyze_game_situation(self, env) -> Dict[str, Any]:
        """ゲーム状況を分析（実測データのみ）"""
        return {
            "score": env.score,
            "health": env.health,
            "money": env.money,
            "wave": getattr(env, 'wave', 1),
            "towers": len(env.towers) if hasattr(env, 'towers') else 0,
            "enemies": len(env.enemies) if hasattr(env, 'enemies') else 0,
            "enemies_killed": getattr(env, 'enemies_killed', 0),
            "tower_positions": [(t['x'], t['y']) for t in env.towers] if hasattr(env, 'towers') else [],
            "enemy_positions": [(e['x'], e['y']) for e in env.enemies] if hasattr(env, 'enemies') else []
        }
    
    def _build_strategic_prompt(self, env, analysis: Dict) -> str:
        """戦略的プロンプトを構築"""
        prompt = f"""You are a tower defense strategy expert. Analyze the current game state and provide a specific action recommendation.

Current Game State:
- Score: {analysis['score']}
- Health: {analysis['health']}
- Money: ${analysis['money']}
- Wave: {analysis['wave']}
- Towers: {analysis['towers']}
- Active Enemies: {analysis['enemies']}
- Enemies Killed: {analysis['enemies_killed']}

Tower Positions: {analysis['tower_positions']}
Enemy Positions: {analysis['enemy_positions']}

Based on this situation, recommend ONE specific action:
1. "place_tower" - if we should build a new tower
2. "upgrade_tower" - if we should upgrade an existing tower
3. "wait" - if we should save money and wait
4. "focus_defense" - if we need to strengthen current defenses

Provide your recommendation with a brief reason (max 2 sentences).
Format: ACTION: [action] REASON: [reason]"""
        
        return prompt
    
    def _fallback_evaluation(self, env) -> str:
        """APIが利用できない場合のフォールバック評価"""
        analysis = self._analyze_game_situation(env)
        
        # シンプルなルールベース判断
        if analysis['health'] < 50:
            return "ACTION: place_tower REASON: Health is low, need more defense."
        elif analysis['money'] > 100 and analysis['towers'] < 10:
            return "ACTION: place_tower REASON: Sufficient money available for expansion."
        elif analysis['enemies'] > 5:
            return "ACTION: place_tower REASON: Many enemies present, strengthen defense."
        else:
            return "ACTION: wait REASON: Current situation is stable."
    
    def get_last_prompt(self) -> str:
        """最後のプロンプトを取得"""
        return self.last_prompt
    
    def get_last_prompt_id(self) -> str:
        """最後のプロンプトIDを取得"""
        return self.last_prompt_id
    
    def get_last_response(self) -> str:
        """最後のレスポンスを取得"""
        return self.last_response
    
    def get_api_stats(self) -> Dict[str, Any]:
        """API使用統計を取得"""
        return {
            "total_calls": self.api_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.successful_calls / max(self.api_calls, 1),
            "total_api_time": self.total_api_time,
            "avg_api_time": self.total_api_time / max(self.api_calls, 1),
            "model": self.model
        }
    
    def reset_stats(self):
        """統計をリセット"""
        self.api_calls = 0
        self.total_api_time = 0.0
        self.successful_calls = 0
        self.failed_calls = 0
