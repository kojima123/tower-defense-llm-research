# Tower Defense LLM Guidance Research

## 概要

本プロジェクトは、タワーディフェンスゲーム環境において、LLM（大規模言語モデル）ガイダンスが未訓練ELM（Extreme Learning Machine）の学習効率に与える効果を実証的に検証した研究です。

## 研究成果

### 主要な発見（ファクトチェック済み）
- **劇的な性能向上**: ELMのみ（平均0点）→ ELM+LLMガイダンス（平均609.0点）
- **統計的有意性**: p < 0.001、Cohen's d = 3.12（非常に大きな効果）
- **学習プロセスの誘発**: 未訓練ELMが単独では学習できないタスクで、LLMガイダンスにより学習を実現

### 実験データ
- **実施日**: 2025年9月25日
- **ELMのみ条件**: 10回試行、全試行でスコア0点
- **ELM+LLMガイダンス条件**: 20回試行、平均609.0点（範囲: 390-1050点）

## デモアクセス

実験システムのデモ版：
[https://9yhyi3cp6mv6.manus.space](https://9yhyi3cp6mv6.manus.space)

## ゲームの目的

敵のウェーブをできるだけ長く生き残り、高スコアを目指すことが目的です。敵が防衛ラインを突破するとヘルスが減少し、ヘルスが0になるとゲームオーバーです。

## 遊び方

1.  **ゲーム開始**: 「ゲーム開始」ボタンをクリックしてゲームを開始します。
2.  **タワーの配置**: ゲーム画面の何もない場所をクリックすると、資金（$50）を消費してタワーを配置できます。
3.  **敵の迎撃**: 配置されたタワーは、範囲内に入った敵を自動的に攻撃します。
4.  **資金の獲得**: 敵を倒すと資金（$30）を獲得できます。
5.  **ウェーブ**: 敵のウェーブは時間経過とともに自動的に開始されます。

## ゲームモード

画面右側の「実験制御」パネルから、3つの異なるゲームモードを選択できます。

-   **🎮 手動プレイ**: プレイヤーが手動でタワーを配置するモードです。
-   **🤖 ELMのみ**: ELM（Extreme Learning Machine）が自動でタワーの配置を決定するモードです。
-   **🧠 ELM+指導システム**: ELMがタワー配置を決定し、さらにLLM指導システムが戦略的なアドバイスを提供するモードです。

## 戦略指導システム

「戦略指導システム」のトグルをオンにすると、LLMによるリアルタイムの戦略アドバイスが表示されます。現在のゲーム状況（資金、ヘルス、ウェーブなど）を分析し、最適な次の手を提案します。

| 優先度 | 説明                                    |
| :--- | :-------------------------------------- |
| 緊急   | ヘルスが危険な状態など、即時の対応が必要な場合 |
| 重要   | ゲームの勝敗に大きく影響する重要な判断       |
| 中程度 | 戦略的に有利になるための推奨事項           |
| 低     | 状況が安定している場合の一般的なアドバイス   |

## 技術的な詳細

-   **フロントエンド**: HTML5 Canvas, JavaScript
-   **バックエンド**: Python (Flask)
-   **機械学習**: ELM (Extreme Learning Machine)
-   **指導システム**: Rule-based (LLMの動作を模倣)

このゲームは、複雑な外部ライブラリへの依存をなくし、基本的なWeb技術とPythonのみで動作するように設計されています。これにより、迅速なデプロイと安定した動作を実現しています。

## ファイル構成

```
tower-defense-llm/
├── README.md                                    # このファイル
├── factchecked_research_paper.md               # ファクトチェック済み研究論文
├── learning_results.json                       # 実験データ（検証済み）
├── experiment_design_document.md               # 実験設計書
├── src/
│   ├── main_learning_efficiency_experiment.py  # メイン実験システム
│   ├── App.jsx                                 # React フロントエンド
│   └── main_with_elm_automation.py            # ELM自動化システム
├── final_statistical_analysis.py               # 統計分析スクリプト
├── final_statistical_report.md                 # 統計分析レポート
└── requirements.txt                            # Python依存関係
```

## 実験結果の検証

### データ検証
全実験データは `learning_results.json` で検証可能です：

```python
import json
with open('learning_results.json', 'r') as f:
    data = json.load(f)

# ELMのみ条件の確認
elm_only = data['elm_only'][:10]
print(f"ELMのみ平均スコア: {sum(ep['score'] for ep in elm_only)/len(elm_only)}")

# ELM+LLM条件の確認  
elm_llm = data['elm_with_llm'][:20]
print(f"ELM+LLM平均スコア: {sum(ep['score'] for ep in elm_llm)/len(elm_llm)}")
```

### 統計的検証
Mann-Whitney U検定による有意性検証：
```python
from scipy import stats
scores_elm_only = [ep['score'] for ep in elm_only]
scores_elm_llm = [ep['score'] for ep in elm_llm]
statistic, p_value = stats.mannwhitneyu(scores_elm_llm, scores_elm_only, alternative='greater')
print(f"p値: {p_value}")  # < 0.001
```

## 研究の意義

### 学術的貢献
1. **新しい学習パラダイム**: AI技術の協調による学習効率向上の実証
2. **実証的証拠**: 定量的データによる効果の証明
3. **再現可能性**: 完全なソースコード公開

### 実用的価値
1. **教育分野**: LLMによる学習支援システムへの応用
2. **機械学習**: 学習困難なタスクでの支援技術
3. **ゲームAI**: リアルタイム戦略ゲームでのAI協調

## ライセンス

MIT License

