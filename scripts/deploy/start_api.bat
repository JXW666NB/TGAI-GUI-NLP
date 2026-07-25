@echo off
chcp 65001 >nul
cd /d "%~dp0\..\.."

:: 本地 CPU 测试 (会弹警告)
:: python scripts\api_server.py --checkpoint checkpoints\tgai_sft.pt --port 5000 --force-cpu

:: AutoDL / GPU 环境用下面这行
python scripts\api_server.py --checkpoint checkpoints\tgai_sft.pt --port 5000

pause
