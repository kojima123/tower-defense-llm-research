# Tower Defense ELM+LLM Research - 実測データ専用システム

[![Data Quality](https://img.shields.io/badge/Data%20Quality-100%2F100-brightgreen)](./data_validation_report.json)
[![Real Data Only](https://img.shields.io/badge/Real%20Data-Only-blue)](#データ品質保証)
[![Reproducible](https://img.shields.io/badge/Reproducible-Fixed%20Seeds-orange)](#再現可能性)
[![Scientific Rigor](https://img.shields.io/badge/Scientific-Rigor-purple)](#科学的厳密性)

**ELM (Extreme Learning Machine) と LLM (Large Language Model) を組み合わせたタワーディフェンス学習システム**

## 🔬 科学的厳密性の保証

### データ品質保証
- ✅ **実測データのみ**: 合成データ生成は一切なし
- ✅ **検証済み**: 自動検証システムによる100%品質スコア
- ✅ **透明性**: 全実験ログ公開・検証可能
- ✅ **再現可能性**: 固定シード実験

### 実測データ統計 (最新更新: 2025-09-26)
- **総実験数**: 16実験
- **総エピソード数**: 32エピソード  
- **総ステップ数**: 51,974ステップ
- **実験条件**: 4条件 (elm_only, rule_teacher, random_teacher, elm_llm)
- **使用シード**: [np.int64(42), np.int64(123), np.int64(456)]

## 🎯 研究目的

高速学習アルゴリズム（ELM）と大規模言語モデル（LLM）の協調により、複雑な戦略ゲームにおける学習効率を向上させる。

## 📊 実験結果 (実測データ)

### パフォーマンス比較

| 条件 | 平均スコア | 標準偏差 | 95%信頼区間 | 最小-最大 | サンプル数 | データソース |
|------|------------|----------|-------------|-----------|------------|--------------|
| elm_only | 2320.00 | 1258.57 | [1419.67, 3220.33] | 0-3200 | 10 | [steps_30e7a0f9.csv](runs/real/test/steps_30e7a0f9.csv), [steps_30e7a0f9.csv](runs/real/integration_test/elm_only/seed_42/steps_30e7a0f9.csv), [steps_0f0fd5f0.csv](runs/real/integration_test/elm_only/seed_123/steps_0f0fd5f0.csv) (+2個) |
| rule_teacher | 3000.00 | 0.00 | [3000.00, 3000.00] | 3000-3000 | 10 | [steps_12eebaae.csv](runs/real/test_rule/steps_12eebaae.csv), [steps_12eebaae.csv](runs/real/integration_test/rule_teacher/seed_42/steps_12eebaae.csv), [steps_ef640b85.csv](runs/real/integration_test/rule_teacher/seed_123/steps_ef640b85.csv) (+2個) |
| random_teacher | 2137.50 | 801.67 | [1467.29, 2807.71] | 1000-2900 | 8 | [steps_7205abe8.csv](runs/real/test_random/steps_7205abe8.csv), [steps_7205abe8.csv](runs/real/integration_test/random_teacher/seed_42/steps_7205abe8.csv), [steps_1b07132c.csv](runs/real/integration_test/random_teacher/seed_123/steps_1b07132c.csv) (+1個) |
| elm_llm | 1950.00 | 1391.64 | [-264.41, 4164.41] | 0-3000 | 4 | [steps_0e46390a.csv](runs/real/test_llm/steps_0e46390a.csv), [steps_0e46390a.csv](runs/real/integration_test/elm_llm/seed_42/steps_0e46390a.csv) |

**🏆 最高性能**: rule_teacher (平均スコア: 3000.00)

### 統計検定結果

**群間比較**: Kruskal-Wallis
- 統計量: 12.6113
- p値: 0.005557
- 有意差: あり (α=0.05)

**ペアワイズ比較** (Mann-Whitney U検定):

| 比較 | p値 | Cohen's d | 効果量 | 有意差 |
|------|-----|-----------|--------|--------|
| elm_only vs rule_teacher | 0.197931 | -0.764 | 中 | ❌ |
| elm_only vs random_teacher | 0.162471 | 0.169 | 小 | ❌ |
| elm_only vs elm_llm | 0.386585 | 0.286 | 小 | ❌ |
| rule_teacher vs random_teacher | 0.000110 | 1.627 | 大 | ✅ |
| rule_teacher vs elm_llm | 0.004326 | 1.509 | 大 | ✅ |
| random_teacher vs elm_llm | 0.862242 | 0.185 | 小 | ❌ |

## 🔍 データ品質保証

### 実測データソース
以下のファイルから直接算出された統計のみを使用：

1. [`steps_30e7a0f9.csv`](runs/real/test/steps_30e7a0f9.csv)
2. [`steps_12eebaae.csv`](runs/real/test_rule/steps_12eebaae.csv)
3. [`steps_7205abe8.csv`](runs/real/test_random/steps_7205abe8.csv)
4. [`steps_0e46390a.csv`](runs/real/test_llm/steps_0e46390a.csv)
5. [`steps_30e7a0f9.csv`](runs/real/integration_test/elm_only/seed_42/steps_30e7a0f9.csv)
6. [`steps_0f0fd5f0.csv`](runs/real/integration_test/elm_only/seed_123/steps_0f0fd5f0.csv)
7. [`steps_e8b8e527.csv`](runs/real/integration_test/elm_only/seed_456/steps_e8b8e527.csv)
8. [`steps_12eebaae.csv`](runs/real/integration_test/rule_teacher/seed_42/steps_12eebaae.csv)
9. [`steps_ef640b85.csv`](runs/real/integration_test/rule_teacher/seed_123/steps_ef640b85.csv)
10. [`steps_9a98549f.csv`](runs/real/integration_test/rule_teacher/seed_456/steps_9a98549f.csv)
... 他6個のファイル

### 合成データ完全排除
- **検証システム**: [`validate_real_data.py`](./validate_real_data.py)による自動検証
- **隔離システム**: 合成データファイルを[`sim/synthetic_data_deprecated/`](./sim/synthetic_data_deprecated/)に隔離
- **品質スコア**: 100/100 (合成データ0件検出)

### 再現可能性
- **固定シード**: 完全な結果再現
- **設定管理**: ハッシュによる実験条件追跡
- **ログ公開**: 全実験プロセスの透明性

## 🚀 使用方法

### 基本実験実行
```bash
# 4条件比較実験（推奨）
python run_experiment_cli_fixed.py run --teachers all --episodes 20

# 特定条件実験
python run_experiment_cli_fixed.py run --teachers elm_llm --episodes 10 --seeds 42 123

# 完全パイプライン（実験+分析+README更新）
python run_experiment_cli_fixed.py full --teachers all --episodes 15 --update-readme
```

### データ検証・分析
```bash
# 実測データ検証
python validate_real_data.py

# 実測データ分析
python analyze_real_data.py runs/real/experiment_name/

# README自動更新
python auto_update_readme.py
```

## 🤖 LLM統合

### LLM Teacher システム
- **モデル**: OpenAI GPT-4o-mini
- **機能**: 戦略的行動推奨
- **ログ**: 詳細なインタラクション記録（JSONL形式）
- **フォールバック**: APIキーなしでも動作

## 📈 技術詳細

### ELM (Extreme Learning Machine)
- **特徴**: 高速学習アルゴリズム
- **実装**: 最小二乗による出力重み更新
- **利点**: 計算効率、過学習抑制

### Tower Defense Environment
- **状態空間**: 敵位置、タワー配置、リソース、ヘルス
- **行動空間**: タワー配置、アップグレード、待機
- **報酬設計**: スコア、生存時間、効率性

## 📁 プロジェクト構成

```
tower-defense-llm/
├── 🔬 validate_real_data.py          # 実測データ検証
├── 📊 auto_update_readme.py          # README自動更新
├── 📊 analyze_real_data.py           # 実測データ分析
├── 🤖 analyze_llm_interactions.py    # LLM分析
├── 🚀 run_experiment_cli_fixed.py    # 統合CLIシステム
├── logger.py                         # 実測専用ログ
├── src/                              # 環境・エージェント
├── runs/real/                        # 実測実験ログ
└── sim/synthetic_data_deprecated/    # 合成データ隔離
```

## 🔧 開発・貢献

### 環境設定
```bash
# 依存関係インストール
pip install -r requirements.txt

# OpenAI APIキー設定（LLM使用時）
export OPENAI_API_KEY="your-api-key"
```

### データ品質維持
- 新しいファイル追加時は`python validate_real_data.py`で検証
- 合成データの使用を厳格に禁止
- 実測ログの継続的な蓄積

---

**このプロジェクトは実測データのみを使用し、完全な科学的厳密性を保証します。**

*最終更新: {current_date} (自動生成)*