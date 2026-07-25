@echo off
chcp 65001 >nul
echo ====================================
echo   TGAI SFT 数据下载
echo ====================================
echo.
echo 正在下载 5 个数据集，约 10 分钟...
echo 输出: data/sft/tgai_sft_data.jsonl
echo.
cd /d "%~dp0\..\.."
python scripts\data\download_sft_data.py --mirror
echo.
echo 完成! 按任意键退出...
pause >nul
