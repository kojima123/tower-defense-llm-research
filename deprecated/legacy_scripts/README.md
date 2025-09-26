# 非推奨スクリプト

このディレクトリには、現在のシステム構成と互換性のない古いスクリプトが含まれています。

## 非推奨ファイル

### elm_llm_tower_defense.py
- **理由**: 古い`elm_tower_defense.py`の環境クラスを使用
- **問題**: `get_action_space_size()`メソッドが現在の`src/tower_defense_environment.py`の`get_action_size()`と不整合
- **代替**: `run_elm_llm_real.py`または`run_experiment_cli_fixed.py`を使用

## 推奨実行方法

現在のシステムでは以下のエントリポイントを使用してください：

```bash
# ELM+LLM実験
python run_elm_llm_real.py --episodes 10 --seed 42

# 統合実験システム
python run_experiment_cli_fixed.py run --teachers elm_llm --episodes 10

# Make統合
make run-llm EPISODES=10
```

## 注意事項

- これらの古いスクリプトは直接実行しないでください
- ビルドエラーや不整合の原因となります
- 現在のsrc/構成に基づく新しいシステムを使用してください
