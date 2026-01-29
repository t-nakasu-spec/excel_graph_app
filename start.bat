@echo off
chcp 65001 > nul
color 0A

echo.
echo ================================
echo Excel グラフ化ツール
echo ================================
echo.

REM 仮想環境の確認
if not exist venv (
    echo エラー: 仮想環境が見つかりません。
    echo 先に setup.bat を実行してください。
    echo.
    pause
    exit /b 1
)

REM 仮想環境の有効化
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo エラー: 仮想環境の有効化に失敗しました。
    pause
    exit /b 1
)

echo ✓ 仮想環境を有効化しました
echo.
echo Streamlit を起動中...
echo ブラウザが自動的に開きます。
echo.

REM Streamlitアプリを起動
streamlit run app.py

if %errorlevel% neq 0 (
    echo.
    echo エラーが発生しました。Enterキーを押して終了...
    pause
)
