@echo off
chcp 65001 > nul
echo ================================
echo Excel グラフ化ツール - 初回セットアップ
echo ================================
echo.

echo [1/3] Python のバージョンを確認中...
python --version
if %errorlevel% neq 0 (
    echo エラー: Python がインストールされていません。
    echo https://www.python.org/downloads/ からインストールしてください。
    pause
    exit /b 1
)
echo.

echo [2/3] 仮想環境を作成中...
python -m venv venv
if %errorlevel% neq 0 (
    echo エラー: 仮想環境の作成に失敗しました。
    pause
    exit /b 1
)
echo.

echo [3/3] 必要なライブラリをインストール中...
call venv\Scripts\activate.bat
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo エラー: ライブラリのインストールに失敗しました。
    pause
    exit /b 1
)
echo.

echo ================================
echo セットアップ完了！
echo 次回から run.bat をダブルクリックで起動できます。
echo ================================
pause
