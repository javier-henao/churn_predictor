# Despliegue en macOS (Customer Churn Predictor)

Este documento explica cómo ejecutar el proyecto en **macOS** de forma local.

## 1. Requisitos previos

- macOS con Terminal
- Python 3.10+ instalado
- `pip` disponible

Para validar:

```bash
python3 --version
pip3 --version
```

## 2. Ir a la carpeta del proyecto

Abre Terminal y navega a la carpeta donde tengas clonado o copiado este proyecto:

```bash
cd /ruta/a/churn-predictor
```

Ejemplo:

```bash
cd ~/Documents/churn-predictor
```

## 3. Crear y activar entorno virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Cuando esté activo, verás `(.venv)` al inicio de la línea en Terminal.

## 4. Instalar dependencias

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Ejecutar la aplicación principal

```bash
streamlit run 02_churn_predictor.py
```

La app abrirá en el navegador, normalmente en:

- `http://localhost:8501`

## 6. Ejecutar módulo de entrenamiento (opcional)

```bash
streamlit run 01_model_training.py
```

## 7. Detener la ejecución

En la Terminal donde corre Streamlit:

- Presiona `Ctrl + C`

## 8. Errores comunes y solución

### Error: `command not found: python3`

Instala Python (por ejemplo, con Homebrew):

```bash
brew install python
```

### Error: `ModuleNotFoundError`

Faltan dependencias o no está activo el entorno virtual.

1. Activa entorno:
   ```bash
   source .venv/bin/activate
   ```
2. Reinstala dependencias:
   ```bash
   pip install -r requirements.txt
   ```

### Error con puerto ocupado (`8501`)

Ejecuta Streamlit en otro puerto:

```bash
streamlit run 02_churn_predictor.py --server.port 8502
```

## 9. Recomendación de uso diario

Cada vez que abras el proyecto:

```bash
cd /ruta/a/churn-predictor
source .venv/bin/activate
streamlit run 02_churn_predictor.py
```
