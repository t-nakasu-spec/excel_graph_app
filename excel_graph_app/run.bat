@echo off
chcp 65001 > nul
echo ================================
echo Excel グラフ化ツールを起動中...
echo ================================
echo.

if not exist venv (
    echo エラー: 仮想環境が見つかりません。
    echo 先に setup.bat を実行してください。
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
streamlit run app.py --server.headless true

if %errorlevel% neq 0 (
    echo.
    echo エラーが発生しました。
    pause
)
