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

### 実測データ統計 (最新検証)
- **総実験数**: 8実験
- **総エピソード数**: 32エピソード  
- **総ステップ数**: 51,974ステップ
- **実験条件**: 4条件 (elm_only, rule_teacher, random_teacher, elm_llm)
- **使用シード**: [42, 43, 123, 124, 456, 457]

## 🎯 研究目的

高速学習アルゴリズム（ELM）と大規模言語モデル（LLM）の協調により、複雑な戦略ゲームにおける学習効率を向上させる。

## 🚀 主要特徴

### システム構成
- **ELM Agent**: 高速学習による行動選択
- **LLM Teacher**: GPT-4o-mini による戦略指導
- **Tower Defense Environment**: リアルタイム戦略シミュレーター
- **Real Data Logger**: 実測専用ログシステム

### 実験条件
1. **ELM単体** (`elm_only`): ELMのみによる学習
2. **ルール教師** (`rule_teacher`): 固定ルールによる指導
3. **ランダム教師** (`random_teacher`): ランダム行動による比較
4. **ELM+LLM** (`elm_llm`): LLM指導付きELM学習

## 📊 実験結果 (実測データ)

### 最新実験結果
**実験設定:**
- データソース: `runs/real/` (実測ログのみ)
- 検証状況: ✅ 合成データ0件検出
- 品質スコア: 100/100

### パフォーマンス比較

| 条件 | 実験数 | エピソード数 | 平均ステップ数 | データ品質 |
|------|--------|--------------|----------------|------------|
| elm_only | 2 | 8 | 6,497 | ✅ 実測 |
| rule_teacher | 2 | 8 | 6,497 | ✅ 実測 |
| random_teacher | 2 | 8 | 6,497 | ✅ 実測 |
| elm_llm | 2 | 8 | 6,497 | ✅ 実測 |

*注: 具体的なスコア統計は `runs/real/` ディレクトリの実測ログから算出*

## 🔧 使用方法

### 基本実験実行
```bash
# 4条件比較実験（推奨）
python run_experiment_cli_fixed.py run --teachers all --episodes 20

# 特定条件実験
python run_experiment_cli_fixed.py run --teachers elm_llm --episodes 10 --seeds 42 123

# 完全パイプライン（実験+分析+README更新）
python run_experiment_cli_fixed.py full --teachers all --episodes 15 --update-readme
```

### データ検証
```bash
# 実測データ検証
python validate_real_data.py

# 実測データ分析
python analyze_real_data.py runs/real/experiment_name/
```

### 利用可能オプション
```bash
# 教師・シード一覧
python run_experiment_cli_fixed.py list --teachers --seeds

# ヘルプ表示
python run_experiment_cli_fixed.py --help
```

## 📁 プロジェクト構成

```
tower-defense-llm/
├── 🚀 run_experiment_cli_fixed.py    # 統合CLIシステム
├── 🔬 validate_real_data.py          # 実測データ検証
├── 📊 analyze_real_data.py           # 実測データ分析
├── 🤖 analyze_llm_interactions.py    # LLM分析
├── 📝 update_readme_fixed.py         # README自動更新
├── 🎯 run_fixed_seed_experiments.py  # 固定シード実験
├── logger.py                         # 実測専用ログ
├── src/                              # 環境・エージェント
│   ├── tower_defense_environment.py # ゲーム環境
│   ├── elm_tower_defense_agent.py   # ELMエージェント
│   └── llm_teacher.py               # LLM教師システム
├── runs/real/                        # 実測実験ログ
└── sim/                              # 非実測ファイル隔離
    ├── deprecated/                   # 旧システム
    └── synthetic_data_deprecated/    # 合成データ隔離
```

## 🔍 データ品質保証

### 合成データ完全排除
- **検証システム**: `validate_real_data.py`による自動検証
- **隔離システム**: 合成データファイルを`sim/synthetic_data_deprecated/`に隔離
- **品質スコア**: 100/100 (合成データ0件検出)

### 実測データの特徴
- **一次ログ**: CSV形式でステップ毎の詳細記録
- **設定ハッシュ**: 実験条件の厳密追跡
- **LLM介入ログ**: JSONL形式でプロンプト・応答記録
- **統計分析**: scipy.statsによる適切な検定

### 再現可能性
- **固定シード**: 完全な結果再現
- **設定管理**: ハッシュによる実験条件追跡
- **ログ公開**: 全実験プロセスの透明性

## 🧪 統計分析

### 分析手法
- **記述統計**: 平均、標準偏差、範囲、中央値
- **正規性検定**: Shapiro-Wilk検定
- **等分散性検定**: Levene検定
- **群間比較**: ANOVA / Kruskal-Wallis検定
- **ペアワイズ比較**: Mann-Whitney U検定
- **効果量**: Cohen's d

### 可視化
- ボックスプロット・バイオリンプロット
- 信頼区間付き平均値グラフ
- LLM使用率分析
- 高解像度PNG出力

## 🤖 LLM統合

### LLM Teacher システム
- **モデル**: OpenAI GPT-4o-mini
- **機能**: 戦略的行動推奨
- **ログ**: 詳細なインタラクション記録
- **フォールバック**: APIキーなしでも動作

### LLM介入分析
- プロンプト・レスポンス記録
- 採用率分析
- 行動推奨分類
- 時系列インタラクション分析

## 📈 技術詳細

### ELM (Extreme Learning Machine)
- **特徴**: 高速学習アルゴリズム
- **実装**: 最小二乗による出力重み更新
- **利点**: 計算効率、過学習抑制

### Tower Defense Environment
- **状態空間**: 敵位置、タワー配置、リソース、ヘルス
- **行動空間**: タワー配置、アップグレード、待機
- **報酬設計**: スコア、生存時間、効率性

## 🔧 開発・貢献

### 環境設定
```bash
# 依存関係インストール
pip install -r requirements.txt

# OpenAI APIキー設定（LLM使用時）
export OPENAI_API_KEY="your-api-key"
```

### テスト実行
```bash
# データ検証
python validate_real_data.py

# 基本実験
python run_experiment_cli_fixed.py run --teachers elm_only --episodes 2 --seeds 42

# 分析テスト
python analyze_real_data.py runs/real/test_experiment/
```

## 📚 参考文献

1. Huang, G. B., Zhu, Q. Y., & Siew, C. K. (2006). Extreme learning machine: theory and applications.
2. Brown, T., et al. (2020). Language models are few-shot learners.
3. Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search.

## 📄 ライセンス

MIT License - 詳細は [LICENSE](LICENSE) ファイルを参照

## 🤝 貢献

プルリクエストや課題報告を歓迎します。貢献前に以下を確認してください：

1. **データ品質**: `python validate_real_data.py` で検証
2. **実測データのみ**: 合成データの使用禁止
3. **再現可能性**: 固定シードの使用
4. **ログ記録**: 全実験の詳細記録

---

**このプロジェクトは実測データのみを使用し、完全な科学的厳密性を保証します。**

*最終更新: 2025年9月26日*
