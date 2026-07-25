@echo off
chcp 65001 >nul
cd /d "%~dp0\..\.."

echo ========================================
echo   Cloudflare Tunnel 内网穿透
echo ========================================
echo.

set "CF=tools\cloudflared.exe"

if not exist "%CF%" (
    echo [错误] 未找到 %CF%
    pause
    exit /b 1
)

echo [启动] 穿透 127.0.0.1:5000 ...
echo 按 Ctrl+C 停止
echo ========================================
echo.

%CF% tunnel --url http://127.0.0.1:5000

pause
