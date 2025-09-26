"""
実測専用ELM+LLM実験ランナー
合成データを一切使用せず、実際のLLM APIコールのみを実行
"""
import argparse
import time
import os
from pathlib import Path
from logger import RunLogger, StepLog, ExperimentConfig, LLMInteractionLogger
from src.tower_defense_environment import TowerDefenseEnvironment
from src.elm_tower_defense_agent import ELMTowerDefenseAgent
from src.llm_teacher import LLMTeacher


class ELMLLMHybridAgentReal(ELMTowerDefenseAgent):
    """実測専用ELM+LLMハイブリッドエージェント"""
    
    def __init__(self, input_size, hidden_size, output_size, llm_teacher, run_logger=None, llm_logger=None):
        super().__init__(input_size, hidden_size, output_size)
        self.llm_teacher = llm_teacher
        self.run_logger = run_logger
        self.llm_logger = llm_logger
        self.llm_call_count = 0
        self.llm_adoption_count = 0
        # 呼び出し間引き：30ステップに1回（コスト・レート制限対策）
        self.eval_interval_steps = 30
    
    def predict_with_guidance(self, state, env, episode, step):
        """LLMガイダンス付きの予測（実測のみ）"""
        # ELMの基本行動
        base_action = self.select_action(state)
        
        # 呼び出し間引き：30ステップに1回のみLLMを呼び出し
        if step % self.eval_interval_steps != 0:
            return base_action, False, None
        
        # LLM教師の評価・推奨
        try:
            evaluation = self.llm_teacher.evaluate_state_and_recommend(env)
            self.llm_call_count += 1
            
            # LLM介入ログ
            if self.llm_logger:
                prompt = self.llm_teacher.get_last_prompt()
                self.llm_logger.log_interaction(
                    episode=episode,
                    step=step,
                    prompt=prompt,
                    response=evaluation,
                    decision=evaluation,
                    adopted=True
                )
            
            # ガイダンスの適用
            final_action = self._apply_guidance(base_action, evaluation, env)
            
            if final_action != base_action:
                self.llm_adoption_count += 1
            
            return final_action, True, evaluation
            
        except Exception as e:
            print(f"LLM call failed: {e}")
            # LLM失敗時はELMの行動をそのまま使用
            return base_action, False, None
    
    def _apply_guidance(self, base_action, evaluation, env):
        """LLMガイダンスを実際の行動に適用"""
        if not evaluation:
            return base_action
        
        # LLMの推奨行動を解析
        if "place_tower" in evaluation.lower():
            # 利用可能な位置からランダム選択
            if env.money >= env.TOWER_COST:
                available_positions = []
                for i, pos in enumerate(env.valid_positions):
                    occupied = any(abs(tower['x'] - pos[0]) < 30 and abs(tower['y'] - pos[1]) < 30 
                                  for tower in env.towers)
                    if not occupied:
                        available_positions.append(i)
                
                if available_positions:
                    import random
                    return random.choice(available_positions)
        
        elif "wait" in evaluation.lower() or "save" in evaluation.lower():
            # 待機の推奨
            return len(env.valid_positions)  # 何もしない行動
        
        # デフォルトはELMの行動
        return base_action
    
    def get_llm_stats(self):
        """LLM使用統計を取得"""
        return {
            "llm_calls": self.llm_call_count,
            "llm_adoptions": self.llm_adoption_count,
            "adoption_rate": self.llm_adoption_count / max(self.llm_call_count, 1)
        }


def run_elm_llm_experiment(episodes: int, seed: int, out_dir: str = "runs/real/elm_llm"):
    """ELM+LLM実験を実行（実測のみ）"""
    
    # OpenAI APIキーの確認
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    # 実験設定
    config = ExperimentConfig(
        condition="elm_llm",
        seed=seed,
        episodes=episodes,
        env_version="1.0",
        agent_version="1.0",
        prompt_version="llm_v1.0",
        model_name="gpt-4o-mini"
    )
    
    # ロガー初期化
    logger = RunLogger(out_dir, config)
    llm_logger = LLMInteractionLogger(out_dir)
    
    # 環境とエージェント初期化
    env = TowerDefenseEnvironment()
    llm_teacher = LLMTeacher()
    agent = ELMLLMHybridAgentReal(
        input_size=env.get_state_size(),
        hidden_size=100,
        output_size=env.get_action_size(),
        llm_teacher=llm_teacher,
        run_logger=logger,
        llm_logger=llm_logger
    )
    
    print(f"Starting ELM+LLM experiment: {episodes} episodes, seed={seed}")
    print(f"Using model: {config.model_name}")
    
    # 実験実行
    episode_scores = []
    start_time = time.time()
    total_llm_calls = 0
    
    for episode in range(episodes):
        episode_start = time.time()
        state = env.reset(seed + episode)
        done = False
        step = 0
        episode_reward = 0
        episode_llm_calls = 0
        
        while not done and step < 1000:  # 最大ステップ制限
            # LLMガイダンス付き行動選択
            action, llm_used, llm_decision = agent.predict_with_guidance(
                state, env, episode, step
            )
            
            if llm_used:
                episode_llm_calls += 1
                total_llm_calls += 1
            
            # 環境ステップ
            next_state, reward, done, info = env.step(action)
            
            # エージェント学習
            agent.add_experience(state, action, reward, next_state, done)
            if step % 10 == 0:
                agent.train()
            
            # ログ記録
            step_log = StepLog(
                episode=episode,
                step=step,
                timestamp=time.time(),
                seed=seed + episode,
                condition="elm_llm",
                state_hash=logger.get_state_hash(state),
                action=str(action),
                reward=reward,
                score=env.score,
                towers=len(env.towers),
                enemies_killed=env.enemies_killed,
                health=env.health,
                llm_used=1 if llm_used else 0,
                llm_prompt_id=llm_teacher.get_last_prompt_id() if llm_used else None,
                llm_decision=str(llm_decision) if llm_decision else None,
                env_version=config.env_version,
                agent_version=config.agent_version,
                prompt_version=config.prompt_version
            )
            logger.log_step(step_log)
            
            state = next_state
            step += 1
            episode_reward += reward
        
        episode_scores.append(env.score)
        episode_time = time.time() - episode_start
        
        print(f"Episode {episode+1}/{episodes}: Score={env.score}, "
              f"Towers={len(env.towers)}, Health={env.health}, "
              f"LLM calls={episode_llm_calls}, Time={episode_time:.2f}s")
    
    # サマリー記録
    total_time = time.time() - start_time
    llm_stats = agent.get_llm_stats()
    
    summary = {
        "condition": "elm_llm",
        "episodes": episodes,
        "seed": seed,
        "total_time": total_time,
        "episode_scores": episode_scores,
        "mean_score": sum(episode_scores) / len(episode_scores),
        "final_score": episode_scores[-1] if episode_scores else 0,
        "max_score": max(episode_scores) if episode_scores else 0,
        "min_score": min(episode_scores) if episode_scores else 0,
        "learning_success_rate": sum(1 for s in episode_scores if s > 0) / len(episode_scores),
        "llm_statistics": {
            "total_llm_calls": total_llm_calls,
            "llm_calls_per_episode": total_llm_calls / episodes,
            "llm_adoption_rate": llm_stats["adoption_rate"],
            "model_name": config.model_name
        },
        "agent_parameters": {
            "input_size": agent.input_size,
            "hidden_size": agent.hidden_size,
            "output_size": agent.output_size,
            "training_samples": len(agent.training_data)
        }
    }
    
    logger.log_summary(summary)
    
    print(f"\nExperiment completed!")
    print(f"Mean score: {summary['mean_score']:.2f}")
    print(f"Learning success rate: {summary['learning_success_rate']:.2%}")
    print(f"Total LLM calls: {total_llm_calls}")
    print(f"LLM adoption rate: {llm_stats['adoption_rate']:.2%}")
    print(f"Results saved to: {out_dir}")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run real ELM+LLM experiments (no synthetic data)")
    parser.add_argument("--episodes", type=int, default=30, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out_dir", type=str, default="runs/real/elm_llm", help="Output directory")
    
    args = parser.parse_args()
    
    run_elm_llm_experiment(args.episodes, args.seed, args.out_dir)
