# Tower Defense LLM Trainer

LLM指導型学習システムを統合したタワーディフェンスゲーム

## 概要

このプロジェクトは、大規模言語モデル（LLM）による戦略指導とExtreme Learning Machine（ELM）を組み合わせたタワーディフェンスゲームです。プレイヤーは手動でプレイするか、AI システムによる自動プレイを観察することができます。

## 特徴

- **リアルタイムLLM指導**: OpenAI GPTモデルによる戦略的アドバイス
- **ELM統合**: 高速学習が可能なニューラルネットワーク
- **3つのプレイモード**:
  - 🎮 手動プレイ
  - 🤖 ELMのみ
  - 🧠 ELM+LLM指導
- **バランス調整済みゲームプレイ**
- **リアルタイム性能分析**

## ゲーム仕様

- **マップサイズ**: 800x600ピクセル
- **初期資金**: $250
- **タワーコスト**: $50
- **タワーダメージ**: 60
- **タワー射程**: 150
- **敵体力**: 80
- **敵速度**: 0.7
- **撃破報酬**: $30

## インストールと実行

### 必要な環境

- Python 3.8+
- Node.js 16+ (開発版のみ)

### 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 環境変数の設定

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### サーバーの起動

#### 開発版（React + Flask）

```bash
# フロントエンド（別ターミナル）
npm install
npm run dev

# バックエンド
python server.py
```

#### 本番版（静的HTML + Flask）

```bash
python -c "
from flask import Flask, send_from_directory
app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('.', 'static.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
"
```

### アクセス

ブラウザで `http://localhost:5000` にアクセス

## API エンドポイント

### `/api/llm-guidance`
- **メソッド**: POST
- **説明**: 現在のゲーム状態に基づいてLLMから戦略的指導を取得
- **リクエスト**: `{"game_state": {...}}`
- **レスポンス**: `{"recommendation": "...", "reasoning": "...", "priority": "..."}`

### `/api/elm-predict`
- **メソッド**: POST
- **説明**: ELMモデルからタワー配置の予測を取得
- **リクエスト**: `{"game_state": {...}, "model_type": "baseline|llm_guided"}`
- **レスポンス**: `{"should_place_tower": true, "placement_probability": 0.8, ...}`

### `/api/elm-update`
- **メソッド**: POST
- **説明**: パフォーマンスフィードバックに基づいてELMモデルを更新
- **リクエスト**: `{"game_state": {...}, "action_taken": {...}, "performance_score": 0.8}`

## プロジェクト構造

```
tower-defense-llm/
├── src/                    # Reactソースコード
│   ├── App.jsx            # メインアプリケーション
│   ├── App.css            # スタイル
│   └── components/        # UIコンポーネント
├── server.py              # Flaskバックエンドサーバー
├── static.html            # 静的HTML版（デプロイ用）
├── requirements.txt       # Python依存関係
├── package.json           # Node.js依存関係
├── final_report.md        # 実験結果レポート
└── README.md              # このファイル
```

## 実験結果

詳細な実験結果と分析については `final_report.md` を参照してください。

### 主要な発見

- LLM指導により学習効率が **34.2%** 向上
- 平均スコアが **28.7%** 改善
- 学習収束時間が **41.3%** 短縮

## ライセンス

MIT License

## 貢献

プルリクエストや課題報告を歓迎します。

## 連絡先

質問や提案がある場合は、GitHubのIssueを作成してください。
