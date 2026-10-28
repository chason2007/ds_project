# BHOOMIQ Streamlit Launcher PowerShell Script
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "       BHOOMIQ Real Estate Valuation & Analytics Engine" -ForegroundColor Green
Write-Host "               Anna University / Sri Eshwar CE" -ForegroundColor Yellow
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Launching Streamlit Interactive Dashboard on http://localhost:8501..." -ForegroundColor White
python -m streamlit run app.py --server.port 8501
