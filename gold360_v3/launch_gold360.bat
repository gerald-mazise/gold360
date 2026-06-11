@echo off
title GOLD360 V3 (Leakage-Controlled)
echo.
echo  ========================================
echo   GOLD360 V3 (Leakage-Controlled)
echo   Launching on port 8503...
echo  ========================================
echo.
cd /d "%~dp0gold360\dashboard"
streamlit run app.py --server.port 8503 --server.headless true
pause
