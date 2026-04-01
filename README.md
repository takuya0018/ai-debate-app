# AI Debate App

ユーザーの質問に対して、ChatGPT が回答し、Gemini がレビューして回答精度を上げる Web アプリです。

## 1. 初回起動方法

### 前提
- macOS / Linux
- Python 3.10+
- Node.js 18+ と npm

### セットアップ
1. リポジトリに移動
```bash
cd /Users/kikutatakuya/AI/ai-debate-app
```

2. Python 仮想環境の作成と有効化
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Python 依存関係のインストール
```bash
pip install -r requirements.txt
```

4. 環境変数ファイルを作成
```bash
cp .env.example .env
```

5. `.env` に必要なキーを設定
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`（任意だがレビュー品質向上に推奨）

6. フロントエンド依存関係をインストール
```bash
cd frontend
npm install
npm run build
cd ..
```

### 起動
```bash
source venv/bin/activate
python app.py
```

ブラウザで以下を開きます。
- `http://127.0.0.1:5000`

## 2. フロントエンドでの操作方法

### 基本操作
1. 画面下の「メッセージ」欄に質問を入力
2. `評価ラウンド (1-3)` を選択
- 1: 速い（レビュー少なめ）
- 2: 標準
- 3: 精度重視（時間は長め）
3. `送信` ボタンを押す（Enterでも送信可、Shift+Enter で改行）
4. チャット欄に次の順で表示されます
- Search API の取得結果
- ChatGPT の初稿
- Gemini のレビュー
- ChatGPT の修正版（必要時）

### 画面の見方
- 右上 `Gemini: ON/OFF` で Gemini API の有効状態を確認
- `Search API 表示中/非表示` で検索メッセージの表示切替
- ステータス表示に「検索中」「レビュー中」「改稿中」が出るので、処理停止と判別できます

### 補助操作
- `履歴クリア`: 会話履歴（logs 内の履歴）をリセット

## 3. よく使うコマンド

### フロントを再ビルド
```bash
cd frontend
npm run build
cd ..
```

### サーバー再起動（5000番ポート使用中のとき）
```bash
lsof -ti tcp:5000 | xargs -r kill
source venv/bin/activate
python app.py
```
