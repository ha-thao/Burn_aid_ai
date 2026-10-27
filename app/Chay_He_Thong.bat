@echo off
chcp 65001 > nul
title HE THONG DANH GIA MUC DO BONG & TRO LY SO CUU BURNAID
echo ===================================================================
echo   HE THONG DANH GIA MUC DO BONG & TRO LY SO CUU BURNAID
echo   Tich hop Web UI + AI Vision 2-Stage + RAG Knowledge Base + Groq LLM
echo ===================================================================
echo.

:: 1. Chuyen ve thu muc goc chua file .bat
cd /d "%~dp0"

:: 2. Kiem tra Python
echo [1/3] Dang kiem tra moi truong Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [CANH BAO] Khong tim thay lenh python! Dang thu python3...
    python3 --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo [LOI] May cua ban chua cai dat Python hoac chua tick "Add Python to PATH".
        echo Vui long cai dat Python tai: https://www.python.org/downloads/
        pause
        exit /b
    )
)

:: 3. Kiem tra thu vien can thiet (uvicorn, starlette, torch, timm, groq...)
echo [2/3] Dang kiem tra thu vien AI...
python -c "import uvicorn, starlette, torch, timm, albumentations, groq" >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] Phat hien thieu thu vien, dang tu dong cai dat tu requirements.txt...
    pip install -r requirements.txt
)

:: 4. Giai phong cong 8000 neu co tien trinh cu dang chay
echo [3/3] Dang khoi chay He thong Hop nhat (Unified Web App & AI Server)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /f /pid %%a >nul 2>&1
)

echo Trinh duyet se tu dong mo tai: http://127.0.0.1:8000
python api_server.py
pause
