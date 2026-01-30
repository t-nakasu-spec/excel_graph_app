# Excel グラフ化ツール

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

## 🌐 オンライン版（推奨）
このアプリケーションは **Streamlit Community Cloud** で公開されています。
インストール不要でブラウザから直接使用できます。

🔗 **[アプリを開く](https://share.streamlit.io/)** （デプロイ後にURLを更新してください）

## 💻 ローカル環境での実行

### 📋 必要なローカル環境
- Windows 10/11
- Python 3.9 以上（https://www.python.org/downloads/ からインストール）

## 🚀 初回セットアップ（1回だけ）
1. `setup.bat` をダブルクリック
2. 完了するまで待つ（数分かかります）

## ▶️ 起動方法（2回目以降）
1. `run.bat` をダブルクリック
2. 自動でブラウザが開きます
3. Excelファイルをアップロードして使用

## 🛑 終了方法
- ブラウザを閉じる
- コマンドプロンプトで `Ctrl + C` を押す

## 📁 必要なファイル
- `app.py` - メインプログラム
- `requirements.txt` - 必要なライブラリリスト
- `setup.bat` - 初回セットアップ用
- `run.bat` - 起動用

## 📊 使い方
### Excelファイルの形式
#### 条件シート
- B列：出荷品番
- C列以降：グラフ番号（列名は任意）

#### データシート（39シート）
必要な列：
- `状態`（日付列、デフォルト）
- `出荷品番`
- `生産済`
- `所要時間`
- `異常値`（0=正常、1=異常）

## ❓ トラブルシューティング
### エラーが出る場合
1. `venv` フォルダを削除
2. `setup.bat` を再実行

### Pythonがない場合
https://www.python.org/downloads/ から最新版をインストール

### グラフが表示されない場合
- 日付列にデータが入っているか確認
- 生産済、所要時間に数値が入っているか確認

## 🚀 Streamlit Community Cloud へのデプロイ方法

このアプリケーションを自分のStreamlit Community Cloudアカウントでデプロイできます：

1. [Streamlit Community Cloud](https://streamlit.io/cloud) にアクセスしてGitHubアカウントでログイン
2. 「New app」をクリック
3. このリポジトリ（`t-nakasu-spec/excel_graph_app`）を選択
4. メインファイルとして `app.py` を指定
5. 「Deploy」をクリック

デプロイ後、自動的に公開URLが生成されます。

### デプロイの特徴
- ✅ 無料で公開可能
- ✅ HTTPSで安全にアクセス
- ✅ 自動的にSSL証明書を提供
- ✅ リポジトリ更新時に自動再デプロイ
- ✅ インターネット経由でどこからでもアクセス可能
