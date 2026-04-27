@echo off
title Customer Churn Predictor - Model Training
echo ============================================
echo   Customer Churn Predictor - ENTRENAMIENTO
echo ============================================
echo.
echo Verificando dependencias...
pip install -r requirements.txt --quiet
echo.
echo Iniciando aplicacion de entrenamiento...
echo Abre tu navegador en: http://localhost:8501
echo.
streamlit run 01_model_training.py --server.headless true
pause
