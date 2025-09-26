"""
実測専用ELM実験ランナー
合成データを一切使用せず、実際の実験のみを実行
"""
import argparse
import time
from pathlib import Path
from logger import RunLogger, StepLog, ExperimentConfig, LLMInteractionLogger
from src.tower_defense_environment import TowerDefenseEnvironment
from src.elm_tower_defense_agent import ELMTowerDefenseAgent


def run_elm_only_experiment(episodes: int, seed: int, out_dir: str = "runs/real/elm_only"):
    """ELMのみの実験を実行（実測のみ）"""
    
    # 実験設定
    config = ExperimentConfig(
        condition="elm_only",
        seed=seed,
        episodes=episodes,
        env_version="1.0",
        agent_version="1.0",
        prompt_version="N/A"
    )
    
    # ロガー初期化
    logger = RunLogger(out_dir, config)
    
    # 環境とエージェント初期化
    env = TowerDefenseEnvironment()
    agent = ELMTowerDefenseAgent(
        input_size=env.get_state_size(),
        hidden_size=100,
        output_size=env.get_action_size()
    )
    
    print(f"Starting ELM-only experiment: {episodes} episodes, seed={seed}")
    
    # 実験実行
    episode_scores = []
    start_time = time.time()
    
    for episode in range(episodes):
        episode_start = time.time()
        state = env.reset(seed + episode)
        done = False
        step = 0
        episode_reward = 0
        
        while not done and step < 1000:  # 最大ステップ制限
            # 行動選択
            action = agent.select_action(state)
            
            # 環境ステップ
            next_state, reward, done, info = env.step(action)
            
            # エージェント学習
            agent.add_experience(state, action, reward, next_state, done)
            if step % 10 == 0:  # 10ステップごとに学習
                agent.train()
            
            # ログ記録
            step_log = StepLog(
                episode=episode,
                step=step,
                timestamp=time.time(),
                seed=seed + episode,
                condition="elm_only",
                state_hash=logger.get_state_hash(state),
                action=str(action),
                reward=reward,
                score=env.score,
                towers=len(env.towers),
                enemies_killed=env.enemies_killed,
                health=env.health,
                llm_used=0,
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
              f"Time={episode_time:.2f}s")
    
    # サマリー記録
    total_time = time.time() - start_time
    summary = {
        "condition": "elm_only",
        "episodes": episodes,
        "seed": seed,
        "total_time": total_time,
        "episode_scores": episode_scores,
        "mean_score": sum(episode_scores) / len(episode_scores),
        "final_score": episode_scores[-1] if episode_scores else 0,
        "max_score": max(episode_scores) if episode_scores else 0,
        "min_score": min(episode_scores) if episode_scores else 0,
        "learning_success_rate": sum(1 for s in episode_scores if s > 0) / len(episode_scores),
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
    print(f"Results saved to: {out_dir}")
    
    return summary


def run_rule_teacher_experiment(episodes: int, seed: int, out_dir: str = "runs/real/rule_teacher"):
    """ルール教師の実験を実行（実測のみ）"""
    
    config = ExperimentConfig(
        condition="rule_teacher",
        seed=seed,
        episodes=episodes,
        env_version="1.0",
        agent_version="1.0",
        prompt_version="rule_v1.0"
    )
    
    logger = RunLogger(out_dir, config)
    env = TowerDefenseEnvironment()
    agent = ELMTowerDefenseAgent(
        input_size=env.get_state_size(),
        hidden_size=100,
        output_size=env.get_action_size()
    )
    
    print(f"Starting Rule Teacher experiment: {episodes} episodes, seed={seed}")
    
    episode_scores = []
    start_time = time.time()
    
    for episode in range(episodes):
        state = env.reset(seed + episode)
        done = False
        step = 0
        
        while not done and step < 1000:
            # ELMの基本行動
            base_action = agent.select_action(state)
            
            # ルールベース教師の指導
            from src.elm_tower_defense_agent import RuleBasedTeacher
            rule_teacher = RuleBasedTeacher()
            final_action = rule_teacher.get_action(env)
            
            # 環境ステップ
            next_state, reward, done, info = env.step(final_action)
            agent.add_experience(state, final_action, reward, next_state, done)
            if step % 10 == 0:
                agent.train()
            
            # ログ記録
            step_log = StepLog(
                episode=episode,
                step=step,
                timestamp=time.time(),
                seed=seed + episode,
                condition="rule_teacher",
                state_hash=logger.get_state_hash(state),
                action=str(final_action),
                reward=reward,
                score=env.score,
                towers=len(env.towers),
                enemies_killed=env.enemies_killed,
                health=env.health,
                llm_used=0,
                llm_prompt_id="rule_guidance",
                llm_decision=f"rule_action_{final_action}",
                env_version=config.env_version,
                agent_version=config.agent_version,
                prompt_version=config.prompt_version
            )
            logger.log_step(step_log)
            
            state = next_state
            step += 1
        
        episode_scores.append(env.score)
        print(f"Episode {episode+1}/{episodes}: Score={env.score}")
    
    # サマリー記録
    summary = {
        "condition": "rule_teacher",
        "episodes": episodes,
        "seed": seed,
        "total_time": time.time() - start_time,
        "episode_scores": episode_scores,
        "mean_score": sum(episode_scores) / len(episode_scores),
        "learning_success_rate": sum(1 for s in episode_scores if s > 0) / len(episode_scores)
    }
    
    logger.log_summary(summary)
    print(f"Rule Teacher experiment completed! Mean score: {summary['mean_score']:.2f}")
    
    return summary


def apply_rule_guidance(env, base_action):
    """ルールベースのガイダンスを適用"""
    from src.elm_tower_defense_agent import RuleBasedTeacher
    
    rule_teacher = RuleBasedTeacher()
    recommended_action = rule_teacher.get_action(env)
    
    # ルール教師の推奨行動を返す
    if recommended_action < len(env.valid_positions):
        pos = env.valid_positions[recommended_action]
        return f"place_tower_{pos[0]}_{pos[1]}"
    else:
        return "wait"


def run_random_teacher_experiment(episodes: int, seed: int, out_dir: str = "runs/real/random_teacher"):
    """ランダム教師の実験を実行（実測のみ）"""
    import random
    
    config = ExperimentConfig(
        condition="random_teacher",
        seed=seed,
        episodes=episodes,
        env_version="1.0",
        agent_version="1.0",
        prompt_version="random_v1.0"
    )
    
    logger = RunLogger(out_dir, config)
    env = TowerDefenseEnvironment()
    agent = ELMTowerDefenseAgent(
        input_size=env.get_state_size(),
        hidden_size=100,
        output_size=env.get_action_size()
    )
    
    print(f"Starting Random Teacher experiment: {episodes} episodes, seed={seed}")
    
    episode_scores = []
    start_time = time.time()
    
    for episode in range(episodes):
        random.seed(seed + episode)  # 再現性のため
        state = env.reset(seed + episode)
        done = False
        step = 0
        
        while not done and step < 1000:
            base_action = agent.select_action(state)
            
            # ランダム教師の指導
            from src.elm_tower_defense_agent import RandomTeacher
            random_teacher = RandomTeacher(seed + episode)
            final_action = random_teacher.get_action(env)
            random_guidance = True
            
            next_state, reward, done, info = env.step(final_action)
            agent.add_experience(state, final_action, reward, next_state, done)
            if step % 10 == 0:
                agent.train()
            
            # ログ記録
            step_log = StepLog(
                episode=episode,
                step=step,
                timestamp=time.time(),
                seed=seed + episode,
                condition="random_teacher",
                state_hash=logger.get_state_hash(state),
                action=str(final_action),
                reward=reward,
                score=env.score,
                towers=len(env.towers),
                enemies_killed=env.enemies_killed,
                health=env.health,
                llm_used=0,
                llm_prompt_id="random_guidance" if random_guidance else None,
                llm_decision=str(final_action) if random_guidance else "no_guidance",
                env_version=config.env_version,
                agent_version=config.agent_version,
                prompt_version=config.prompt_version
            )
            logger.log_step(step_log)
            
            state = next_state
            step += 1
        
        episode_scores.append(env.score)
        print(f"Episode {episode+1}/{episodes}: Score={env.score}")
    
    # サマリー記録
    summary = {
        "condition": "random_teacher",
        "episodes": episodes,
        "seed": seed,
        "total_time": time.time() - start_time,
        "episode_scores": episode_scores,
        "mean_score": sum(episode_scores) / len(episode_scores),
        "learning_success_rate": sum(1 for s in episode_scores if s > 0) / len(episode_scores)
    }
    
    logger.log_summary(summary)
    print(f"Random Teacher experiment completed! Mean score: {summary['mean_score']:.2f}")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run real ELM experiments (no synthetic data)")
    parser.add_argument("--condition", choices=["elm_only", "rule_teacher", "random_teacher"], 
                       required=True, help="Experiment condition")
    parser.add_argument("--episodes", type=int, default=30, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out_dir", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    if args.out_dir is None:
        args.out_dir = f"runs/real/{args.condition}"
    
    if args.condition == "elm_only":
        run_elm_only_experiment(args.episodes, args.seed, args.out_dir)
    elif args.condition == "rule_teacher":
        run_rule_teacher_experiment(args.episodes, args.seed, args.out_dir)
    elif args.condition == "random_teacher":
        run_random_teacher_experiment(args.episodes, args.seed, args.out_dir)
