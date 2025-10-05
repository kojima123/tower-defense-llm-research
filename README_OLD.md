# Tower Defense ELM+LLM Research Project

**科学的厳密性を重視したタワーディフェンス学習システム**

[![Data Quality](https://img.shields.io/badge/Data-Real%20Only-green)]()
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Fixed%20Seeds-blue)]()
[![Transparency](https://img.shields.io/badge/Transparency-Full%20Logs-orange)]()

## 概要

このプロジェクトは、ELM（Extreme Learning Machine）とLLM（Large Language Model）を組み合わせたタワーディフェンス学習システムです。**実測データのみ**を使用し、合成データを一切使用しない科学的に厳密な実験を実施しています。

### 主要特徴

- 🔬 **科学的厳密性**: 実測データのみ、合成データ完全排除
- 🔄 **完全再現可能**: 固定シード、設定ハッシュ管理
- 📊 **統計的妥当性**: 適切な検定手法による分析
- 🤖 **LLM統合**: OpenAI GPT-4o-mini による戦略指導
- 📝 **透明性**: 全実験ログの公開・検証可能

## 実験結果

### 最新実験 (2025-09-26)

**実験設定:**
- 条件数: 4
- 使用シード: [42, 123, 456]
- エピソード数/条件: 6
- データ品質: ✅ 実測のみ（合成データなし）

### パフォーマンス比較

| 条件 | 平均スコア | 標準偏差 | 最小-最大 | サンプル数 |
|------|------------|----------|-----------|------------|
| elm_only | 1783.33 | 1265.13 | 0-2800 | 3 |
| rule_teacher | 3000.00 | 0.00 | 3000-3000 | 3 |
| random_teacher | 2300.00 | 511.53 | 1650-2900 | 3 |

**🏆 最高性能**: rule_teacher (平均スコア: 3000.00)

### データ品質保証

- ✅ **実測データのみ**: 合成データ生成は一切なし
- ✅ **再現可能性**: 固定シード実験
- ✅ **透明性**: 全実験ログ公開
- ✅ **統計的妥当性**: 適切な検定手法使用



## 技術詳細

### システム構成

- **ELM (Extreme Learning Machine)**: 高速学習アルゴリズム
- **LLM Teacher**: OpenAI GPT-4o-mini による戦略指導
- **環境**: Tower Defense シミュレーター
- **ログシステム**: CSV + JSON + JSONL 形式

### 実験条件

1. **ELM単体**: ELMのみでの学習
2. **ルール教師**: 事前定義ルールによる指導
3. **ランダム教師**: ランダムな行動指導
4. **ELM+LLM**: LLMによる戦略的指導

### データ処理パイプライン

```
実験実行 → ログ記録 → データ検証 → 統計分析 → レポート生成
```

- **合成データ検出**: 自動的に排除
- **設定ハッシュ**: 実験条件の厳密管理
- **統計検定**: scipy.stats による科学的分析

## 使用方法

### 基本実験の実行

```bash
# 4条件比較実験（推奨）
python run_fixed_seed_experiments.py --episodes 20

# 単一条件実験
python run_elm_real.py --condition elm_only --episodes 10 --seed 42

# LLM実験（OpenAI APIキー必要）
export OPENAI_API_KEY="your-api-key"
python run_elm_llm_real.py --episodes 10 --seed 42
```

### 分析とレポート生成

```bash
# 実測データ分析
python analyze_real_data.py runs/real/experiment_name/

# LLMインタラクション分析
python analyze_llm_interactions.py runs/real/experiment_name/elm_llm/seed_42/

# 完全実験（実行+分析）
python run_complete_experiment.py --episodes 20
```

### 結果の確認

- **実験ログ**: `runs/real/` ディレクトリ
- **分析レポート**: `*_analysis_report.md`
- **可視化**: `*.png` ファイル
- **統計結果**: JSON形式のサマリー

## ファイル構成

```
├── run_fixed_seed_experiments.py  # 4条件比較実験
├── run_elm_real.py                # ELM単体実験
├── run_elm_llm_real.py            # ELM+LLM実験
├── analyze_real_data.py           # 実測データ分析
├── logger.py                      # 実測専用ログシステム
├── src/
│   ├── tower_defense_environment.py
│   ├── elm_tower_defense_agent.py
│   └── llm_teacher.py
└── runs/real/                     # 実測実験ログ
```

## 研究の信頼性

### データ品質保証

- ✅ **実測データのみ**: `validate_no_synthetic_data()` による検証
- ✅ **固定シード実験**: 完全な再現可能性
- ✅ **設定ハッシュ**: 実験条件の厳密な追跡
- ✅ **ログ整合性**: 自動検証システム

### 統計分析

- 📊 **記述統計**: 平均、標準偏差、範囲、中央値
- 🧪 **検定**: Shapiro-Wilk、Levene、ANOVA/Kruskal-Wallis
- 📈 **効果量**: Cohen's d による実用的有意性
- 🎯 **多重比較**: Mann-Whitney U検定

## 貢献

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**注意**: 全ての貢献は実測データのみを使用し、合成データ生成を含まないことを確認してください。

## ライセンス

MIT License - 詳細は [LICENSE](LICENSE) ファイルを参照

## 引用

```bibtex
@software{tower_defense_elm_llm,
  title={Tower Defense ELM+LLM Research Project},
  author={Research Team},
  year={2025},
  url={https://github.com/your-repo/tower-defense-llm}
}
```

---

*このプロジェクトは実測データのみを使用し、科学的厳密性を最優先に開発されています。*
