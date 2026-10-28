@echo off
title BHOOMIQ - Real Estate Valuation & Market Intelligence Platform
echo =====================================================================
echo           BHOOMIQ Real Estate Valuation & Analytics Engine
echo                   Anna University / Sri Eshwar CE
echo =====================================================================
echo.
echo Launching Streamlit Interactive Dashboard...
echo Opening application at http://localhost:8501
echo Press Ctrl+C in this window when you wish to stop the application.
echo.

python -m streamlit run app.py --server.port 8501
pause
