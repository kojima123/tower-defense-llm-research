#!/usr/bin/env python3
"""
LLMインタラクションログのテストスクリプト
実際のOpenAI APIを使用してログ機能を検証
"""
import os
import time
from pathlib import Path
from logger import LLMInteractionLogger
from src.llm_teacher import LLMTeacher
from src.tower_defense_environment import TowerDefenseEnvironment


def test_llm_interaction_logging():
    """LLMインタラクションログのテスト"""
    
    # APIキーの確認
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return False
    
    print(f"✅ OpenAI API Key configured: {api_key[:10]}...")
    
    # テスト用ディレクトリ作成
    test_dir = Path("runs/real/llm_logging_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # ロガー初期化
    llm_logger = LLMInteractionLogger(str(test_dir))
    
    # LLM教師初期化
    llm_teacher = LLMTeacher(api_key=api_key, model="gpt-4o-mini")
    
    # 環境初期化
    env = TowerDefenseEnvironment()
    env.reset(seed=42)
    
    print("\n🧪 Testing LLM interaction logging...")
    
    # テストケース1: 基本的なLLM呼び出し
    print("\n1. Basic LLM call test:")
    try:
        start_time = time.time()
        evaluation = llm_teacher.evaluate_state_and_recommend(env)
        api_time = time.time() - start_time
        
        print(f"   ✅ LLM Response: {evaluation[:100]}...")
        print(f"   ⏱️  API Time: {api_time:.2f}s")
        
        # ログ記録
        llm_logger.log_interaction(
            episode=1,
            step=1,
            prompt=llm_teacher.get_last_prompt(),
            response=evaluation,
            decision=evaluation,
            adopted=True
        )
        
        print("   ✅ Interaction logged successfully")
        
    except Exception as e:
        print(f"   ❌ LLM call failed: {e}")
        return False
    
    # テストケース2: 複数回の呼び出し
    print("\n2. Multiple LLM calls test:")
    for i in range(3):
        try:
            # 環境状態を少し変更
            env.money += 100 * i
            env.wave += i
            
            evaluation = llm_teacher.evaluate_state_and_recommend(env)
            
            llm_logger.log_interaction(
                episode=1,
                step=i+2,
                prompt=llm_teacher.get_last_prompt(),
                response=evaluation,
                decision=f"test_decision_{i}",
                adopted=i % 2 == 0  # 交互に採用/非採用
            )
            
            print(f"   ✅ Call {i+1}: {evaluation[:50]}...")
            
        except Exception as e:
            print(f"   ❌ Call {i+1} failed: {e}")
    
    # テストケース3: API統計の確認
    print("\n3. API statistics test:")
    stats = llm_teacher.get_api_stats()
    print(f"   📊 Total calls: {stats['total_calls']}")
    print(f"   ✅ Successful calls: {stats['successful_calls']}")
    print(f"   ❌ Failed calls: {stats['failed_calls']}")
    print(f"   📈 Success rate: {stats['success_rate']:.2%}")
    print(f"   ⏱️  Average API time: {stats['avg_api_time']:.2f}s")
    
    # ログファイルの確認
    print("\n4. Log file verification:")
    log_file = test_dir / "llm_interactions.jsonl"
    if log_file.exists():
        with log_file.open('r') as f:
            lines = f.readlines()
        print(f"   ✅ Log file created: {len(lines)} interactions logged")
        
        # 最初のログエントリを表示
        if lines:
            import json
            first_entry = json.loads(lines[0])
            print(f"   📝 First entry keys: {list(first_entry.keys())}")
            print(f"   🔑 Prompt ID: {first_entry.get('prompt_id', 'N/A')}")
    else:
        print("   ❌ Log file not found")
        return False
    
    print(f"\n✅ LLM interaction logging test completed successfully!")
    print(f"📁 Test results saved to: {test_dir}")
    
    return True


def test_llm_teacher_fallback():
    """LLM教師のフォールバック機能テスト"""
    print("\n🔄 Testing LLM teacher fallback mode...")
    
    # APIキーなしでテスト
    llm_teacher = LLMTeacher(api_key=None)
    env = TowerDefenseEnvironment()
    env.reset(seed=42)
    
    try:
        evaluation = llm_teacher.evaluate_state_and_recommend(env)
        print(f"   ✅ Fallback response: {evaluation}")
        
        stats = llm_teacher.get_api_stats()
        print(f"   📊 Fallback stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Fallback test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting LLM interaction logging tests...")
    
    # メインテスト
    success = test_llm_interaction_logging()
    
    # フォールバックテスト
    fallback_success = test_llm_teacher_fallback()
    
    if success and fallback_success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Some tests failed!")
        exit(1)
