# Customer Churn Predictor - Launcher
# Ejecutar con doble clic o desde PowerShell
Write-Host "============================================" -ForegroundColor Red
Write-Host "  Customer Churn Predictor - ENTRENAMIENTO" -ForegroundColor White
Write-Host "============================================" -ForegroundColor Red
Write-Host ""
Write-Host "Instalando dependencias..." -ForegroundColor Yellow
pip install -r requirements.txt --quiet 2>$null
Write-Host "Iniciando... Abre http://localhost:8501" -ForegroundColor Green
streamlit run 01_model_training.py --server.headless true
