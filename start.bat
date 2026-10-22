@echo off
chcp 65001 >nul
title Trade Journal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] 首次运行，正在创建虚拟环境...
    python -m venv .venv
    if errorlevel 1 (
        echo 创建虚拟环境失败，请确认已安装 Python 3.10 及以上版本。
        pause
        exit /b 1
    )
    echo [2/3] 正在安装依赖，请稍候...
    ".venv\Scripts\python.exe" -m pip install -q --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -q -e .
    if errorlevel 1 (
        echo 安装依赖失败，请检查网络。
        pause
        exit /b 1
    )
) else (
    echo [1/3] 虚拟环境已存在，跳过安装
)

echo [3/3] 启动服务，浏览器将自动打开...
echo.
echo     地址: http://127.0.0.1:8765
echo     关闭: 按 Ctrl+C 或直接关闭本窗口
echo.
".venv\Scripts\python.exe" -m trade_journal

pause
