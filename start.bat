@echo off
chcp 65001 >nul
title Adaptive-RAG 智能客服工作台

echo ==============================================================================
echo 🚀 正在启动 Adaptive-RAG 全栈服务 (FastAPI + React Web Workbench)...
echo ==============================================================================

if not exist ".venv\Scripts\python.exe" (
    echo ❌ 错误: 未找到 .venv 虚拟环境，请先初始化环境！
    pause
    exit /b 1
)

:: 使用虚拟环境中的 Python 启动带有健康检查与自动开浏览器的启动器
.\.venv\Scripts\python.exe scripts\start_server.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ⚠️ 服务已停止运行。
    pause
)
