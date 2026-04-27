@echo off
title Customer Churn Predictor - MVP
echo ============================================
echo   Customer Churn Predictor - MVP
echo ============================================
echo.
echo Verificando dependencias...
pip install -r requirements.txt --quiet
echo.
echo Iniciando aplicacion principal...
echo Abre tu navegador en: http://localhost:8502
echo.
streamlit run 02_churn_predictor.py --server.port 8502 --server.headless true
pause
