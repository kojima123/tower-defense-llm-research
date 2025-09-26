# LLM介入効果分析レポート

**生成日時**: 2025-09-26 04:17:12  
**データソース**: 実測ログのみ使用

## 📊 介入統計サマリー

- **総介入回数**: 0回
- **採用率**: 0.0%
- **分析対象**: 実測データのみ（合成データ0件）

## 🎯 パフォーマンス改善効果

### パフォーマンス比較

現在のデータでは十分なLLM介入データが不足しています。
より多くのELM+LLM実験を実行してください：

```bash
python run_experiment_cli_fixed.py run --teachers elm_llm --episodes 20 --seeds 42 123 456
```


## 📈 可視化ファイル

1. **介入タイムライン**: [`llm_intervention_timeline.png`](./llm_intervention_timeline.png)
2. **パフォーマンス比較**: [`llm_performance_comparison.png`](./llm_performance_comparison.png)  
3. **採用率分析**: [`llm_adoption_analysis.png`](./llm_adoption_analysis.png)

## 🔍 データ品質保証

- ✅ **実測データのみ**: 合成データ使用なし
- ✅ **透明性**: 全介入ログ公開
- ✅ **再現可能性**: 固定シード実験

## 💡 推奨事項

1. **データ充実**: より多くのELM+LLM実験実行
2. **長期評価**: 複数エピソードでの効果測定
3. **コスト分析**: API使用料とパフォーマンス改善のトレードオフ評価

---

*このレポートは実測データのみから生成されています*
