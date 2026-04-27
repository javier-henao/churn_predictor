"""
=================================================================
  CUSTOMER CHURN PREDICTOR — MVP Deployment
  Main Streamlit Application
  Modules: Documentación | Churn Prediction
=================================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import io
import joblib
from datetime import datetime
from fpdf import FPDF

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report)

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ──────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = None

# ──────────────────────────────────────────────
# THEME
# ──────────────────────────────────────────────
def get_colors():
    if st.session_state.dark_mode:
        return {"bg":"#0E1117","card":"#1E2130","text":"#FAFAFA",
                "primary":"#FF4B4B","secondary":"#FF8C8C","accent":"#C62828",
                "success":"#4CAF50","warning":"#FFC107","danger":"#EF5350"}
    return {"bg":"#FFFFFF","card":"#F8F9FA","text":"#1E1E1E",
            "primary":"#C62828","secondary":"#EF5350","accent":"#B71C1C",
            "success":"#2E7D32","warning":"#F57F17","danger":"#D32F2F"}

# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=60)
    st.markdown("### 🛒 Customer Churn Predictor")

    dark = st.toggle("🌙 Modo Oscuro", value=st.session_state.dark_mode)
    if dark != st.session_state.dark_mode:
        st.session_state.dark_mode = dark
        st.rerun()

    st.divider()
    store_name = st.text_input("🏪 Nombre del comercio", "Mi E-Commerce")
    analyst_name = st.text_input("👤 Analista", "Equipo Data Science")

    st.divider()
    page = st.radio("📑 Navegación", ["🏠 Inicio", "📖 Documentación", "🔮 Churn Prediction"])

    st.divider()
    st.caption("Customer Churn Predictor v1.0")
    st.caption(f"© {datetime.now().year} | {store_name}")

# ──────────────────────────────────────────────
colors = get_colors()
# ──────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────
st.markdown(f"""
<style>
 
/* Fondo principal */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {{
    background-color: {colors['bg']} !important;
}}
 
/* Bloques internos transparentes */
[data-testid="stVerticalBlock"] {{
    background-color: transparent !important;
}}
 
/* Sidebar */
[data-testid="stSidebar"] > div {{
    background-color: {colors['card']} !important;
}}
 
/* Texto — solo sobre selectores específicos de Streamlit */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
.stMetric label,
.stMetric div[data-testid="stMetricValue"] {{
    color: {colors['text']} !important;
}}
 
/* Header personalizado */
.main-header {{
    background: linear-gradient(135deg, {colors['primary']}, {colors['accent']});
    color: white;
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    text-align: center;
}}
.main-header h1 {{ color: white !important; margin: 0; font-size: 2.2rem; }}
.main-header p  {{ color: #FFCCCCdd !important; margin: 0.5rem 0 0 0; }}
 
/* Badge ganador */
.winner-badge {{
    background: linear-gradient(135deg, #4CAF50, #2E7D32);
    color: white;
    padding: 0.5rem 1.2rem;
    border-radius: 20px;
    display: inline-block;
    font-weight: 700;
    font-size: 1rem;
}}
 
/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px 8px 0 0;
    padding: 10px 20px;
}}
 
</style>
""", unsafe_allow_html=True)
# ──────────────────────────────────────────────
# FEATURES DEFINITION (from EDA)
# ──────────────────────────────────────────────
REQUIRED_FEATURES = [
    'Total_Spend', 'Years_as_Customer', 'Num_of_Purchases',
    'Num_of_Returns', 'Num_of_Support_Contacts',
    'Satisfaction_Score', 'Last_Purchase_Days_Ago', 'Email_Opt_In', 'Promotion_Response'
]

FEATURE_DESCRIPTIONS = {
    'Total_Spend': 'Gasto total del cliente ($)',
    'Years_as_Customer': 'Años como cliente',
    'Num_of_Purchases': 'Número de compras realizadas',
    'Num_of_Returns': 'Número de devoluciones',
    'Num_of_Support_Contacts': 'Contactos con soporte',
    'Satisfaction_Score': 'Puntuación de satisfacción (1-5)',
    'Last_Purchase_Days_Ago': 'Días desde la última compra',
    'Email_Opt_In': 'Suscrito a emails (0=No, 1=Sí)',
    'Promotion_Response': 'Respuesta a promociones (0=Desuscrito, 1=Respondió, 2=Ignoró)'
}

# ──────────────────────────────────────────────
# DATA LOADING & MODEL FUNCTIONS
# ──────────────────────────────────────────────
@st.cache_data
def load_raw_dataset(path):
    return pd.read_csv(path)

@st.cache_data
def process_eda_dataset(path):
    df = pd.read_csv(path)
    df = df.dropna().drop_duplicates()

    # Drop Customer_ID
    if 'Customer_ID' in df.columns:
        df = df.drop('Customer_ID', axis=1)

    # Encode solo columnas que SI se usan en el modelo
    df['Email_Opt_In'] = df['Email_Opt_In'].astype(int)
    df['Target_Churn'] = df['Target_Churn'].astype(int)
    df['Promotion_Response'] = df['Promotion_Response'].replace(
        {'Unsubscribed':0,'Responded':1,'Ignored':2}).astype(int)

    # df_full conserva todas las columnas para el EDA
    df_full = df.copy()

    # Drop columnas excluidas por EDA (sesgo y redundancia)
    cols_to_drop = [c for c in ['Age', 'Gender', 'Annual_Income', 'Average_Transaction_Amount']
                    if c in df.columns]
    df = df.drop(cols_to_drop, axis=1)

    return df, df_full

def train_best_model(df, random_state=42):
    """Train all 3 models with Pipeline+GridSearchCV — sin leakage."""
    X = df.drop('Target_Churn', axis=1)
    y = df['Target_Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    pipelines = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=random_state))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(random_state=random_state))
        ]),
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(random_state=random_state))
        ]),
    }

    param_grids = {
        'Logistic Regression': {'clf__C': [0.1, 1, 10], 'clf__solver': ['lbfgs']},
        'Random Forest':       {'clf__n_estimators': [100, 200], 'clf__max_depth': [None, 10]},
        'Gradient Boosting':   {'clf__n_estimators': [100], 'clf__learning_rate': [0.05, 0.1], 'clf__max_depth': [3, 5]},
    }

    all_results = {}
    for name, pipe in pipelines.items():
        gs = GridSearchCV(pipe, param_grids[name], cv=cv, scoring='f1', n_jobs=-1)
        gs.fit(X_train, y_train)
        best_pipe = gs.best_estimator_
        best_index = gs.best_index_
        y_pred  = best_pipe.predict(X_test)
        y_proba = best_pipe.predict_proba(X_test)[:, 1]
        all_results[name] = {
            'model': best_pipe, 'best_params': gs.best_params_,
            'y_pred': y_pred, 'y_proba': y_proba,
            'accuracy':   accuracy_score(y_test, y_pred),
            'precision':  precision_score(y_test, y_pred),
            'recall':     recall_score(y_test, y_pred),
            'f1':         f1_score(y_test, y_pred),
            'auc_roc':    roc_auc_score(y_test, y_proba),
            'cv_f1_mean': gs.cv_results_['mean_test_score'][best_index],
            'cv_f1_std':  gs.cv_results_['std_test_score'][best_index],
            'confusion':  confusion_matrix(y_test, y_pred)
        }

    best_name = max(all_results, key=lambda k: all_results[k]['f1'])
    return all_results, best_name, X_test, y_test, X.columns.tolist()

def preprocess_new_data(df_new, feature_names, scaler):
    """Preprocess new customer data for prediction."""
    errors = []

    # Check required columns
    # Map common alternative names
    col_map = {c.lower().replace(' ','_'): c for c in df_new.columns}

    for feat in feature_names:
        if feat not in df_new.columns:
            alt = feat.lower().replace(' ','_')
            if alt in col_map:
                df_new = df_new.rename(columns={col_map[alt]: feat})
            else:
                errors.append(feat)

    if errors:
        return None, errors

    df_proc = df_new[feature_names].copy()

    # Encode categoricals robustly (text, bool, category, mixed).
    email_map = {
        'true': 1, 'false': 0, 'yes': 1, 'no': 0, 'si': 1, 'sí': 1, '1': 1, '0': 0
    }
    promo_map = {
        'unsubscribed': 0, 'responded': 1, 'ignored': 2,
        'desuscrito': 0, 'respondio': 1, 'respondió': 1, 'ignoro': 2, 'ignoró': 2,
        '0': 0, '1': 1, '2': 2
    }

    email_text = df_proc['Email_Opt_In'].astype(str).str.strip().str.lower()
    email_num = pd.to_numeric(df_proc['Email_Opt_In'], errors='coerce')
    df_proc['Email_Opt_In'] = email_text.map(email_map).where(email_text.map(email_map).notna(), email_num)

    promo_text = df_proc['Promotion_Response'].astype(str).str.strip().str.lower()
    promo_num = pd.to_numeric(df_proc['Promotion_Response'], errors='coerce')
    df_proc['Promotion_Response'] = promo_text.map(promo_map).where(promo_text.map(promo_map).notna(), promo_num)

    # Ensure every feature is numeric before inference.
    for col in feature_names:
        df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce')

    invalid_cols = [col for col in feature_names if df_proc[col].isna().any()]
    if invalid_cols:
        return None, [f"{c} (valor inválido o vacío)" for c in invalid_cols]

    # No escalar aqui — el Pipeline del modelo lo hace internamente
    return df_proc, []

def generate_recommendations(row, prob):
    """Generate loyalty recommendations based on SurtiClientes model."""
    recs = []
    if prob > 0.7:
        recs.append("🔴 **Programa de rescate urgente**: Contacto personalizado con oferta exclusiva "
                    "de descuento del 15-20% en próxima compra (modelo Tarjeta SurtiCliente).")
        recs.append("🎁 **Bono de reactivación**: Enviar bono digital de recompensa por fidelidad "
                    "acumulada, similar al 'Bono Millón' de Surtifamiliar.")
        recs.append("📞 **Llamada de servicio proactiva**: Contacto del equipo de soporte para "
                    "resolver cualquier inconveniente pendiente.")
    elif prob > 0.4:
        recs.append("🟡 **Programa de puntos acelerado**: Duplicar puntos de fidelización en las "
                    "próximas 3 compras (modelo acumulación SurtiCliente).")
        recs.append("📧 **Campaña de re-engagement**: Enviar newsletter personalizado con ofertas "
                    "basadas en historial de compras.")
        recs.append("🏷️ **Descuento por día especial**: Ofrecer descuento en su categoría favorita "
                    "(como los 'Martes de Redención' de Surtifamiliar).")
    else:
        recs.append("🟢 **Programa de fidelización estándar**: Mantener en programa de acumulación "
                    "de puntos y redención de beneficios.")
        recs.append("⭐ **Club de clientes VIP**: Invitar a programa exclusivo con beneficios como "
                    "acceso anticipado a ofertas (modelo Club Amas de Casa Surtifamiliar).")
        recs.append("🎉 **Reconocimiento de lealtad**: Enviar comunicación de agradecimiento con "
                    "regalo sorpresa en fechas especiales.")

    # Specific feature-based recs
    if 'Satisfaction_Score' in row.index and row.get('Satisfaction_Score', 5) <= 2:
        recs.append("⚠️ **Encuesta de satisfacción**: Cliente con baja satisfacción — realizar "
                    "seguimiento personalizado para identificar pain points.")
    if 'Num_of_Support_Contacts' in row.index and row.get('Num_of_Support_Contacts', 0) >= 5:
        recs.append("🛠️ **Atención prioritaria**: Alto volumen de contactos con soporte — asignar "
                    "agente dedicado para resolución proactiva.")
    if 'Last_Purchase_Days_Ago' in row.index and row.get('Last_Purchase_Days_Ago', 0) >= 200:
        recs.append("⏰ **Campaña de reactivación**: Cliente inactivo — enviar oferta de 'Te extrañamos' "
                    "con incentivo de regreso.")
    if 'Email_Opt_In' in row.index and row.get('Email_Opt_In', 0) == 0:
        recs.append("📬 **Opt-in campaign**: Incentivar suscripción a comunicaciones con descuento "
                    "inicial del 5% (como Tarjeta Colpatria Surtifamiliar).")

    return recs

def generate_pdf_report(results_df, store_name, analyst, model_name):
    """Generate PDF report of predictions."""
    def _safe_pdf_text(value):
        # Helvetica in FPDF is latin-1; replace unsupported unicode chars (e.g., emojis).
        return str(value).encode('latin-1', errors='replace').decode('latin-1')

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 18)
    pdf.cell(0, 15, _safe_pdf_text('Customer Churn Prediction Report'), ln=True, align='C')
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 8, _safe_pdf_text(f'Comercio: {store_name} | Analista: {analyst}'), ln=True, align='C')
    pdf.cell(0, 8, _safe_pdf_text(f'Modelo: {model_name} | Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M")}'), ln=True, align='C')
    pdf.ln(5)

    # Summary
    total = len(results_df)
    churn_count = results_df['Prediccion'].sum() if 'Prediccion' in results_df.columns else 0
    pdf.set_font('Helvetica', 'B', 13)
    pdf.cell(0, 10, _safe_pdf_text('Resumen Ejecutivo'), ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 7, _safe_pdf_text(f'Total clientes evaluados: {total}'), ln=True)
    pdf.cell(0, 7, _safe_pdf_text(f'Clientes en riesgo de churn: {churn_count} ({churn_count/max(total,1)*100:.1f}%)'), ln=True)
    pdf.cell(0, 7, _safe_pdf_text(f'Clientes estables: {total - churn_count} ({(total-churn_count)/max(total,1)*100:.1f}%)'), ln=True)
    pdf.ln(5)

    # Table
    pdf.set_font('Helvetica', 'B', 10)
    cols_to_show = [c for c in ['Cliente', 'Probabilidad_Churn', 'Prediccion', 'Nivel_Riesgo']
                    if c in results_df.columns]
    if not cols_to_show:
        cols_to_show = results_df.columns[:5].tolist()

    col_w = 180 / max(len(cols_to_show), 1)
    for c in cols_to_show:
        pdf.cell(col_w, 8, _safe_pdf_text(str(c)[:20]), border=1, align='C')
    pdf.ln()

    pdf.set_font('Helvetica', '', 9)
    for _, row in results_df.head(100).iterrows():
        for c in cols_to_show:
            val = row.get(c, '')
            if isinstance(val, float):
                val = f'{val:.4f}'
            pdf.cell(col_w, 7, _safe_pdf_text(str(val)[:25]), border=1, align='C')
        pdf.ln()

    out = pdf.output()
    if isinstance(out, bytearray):
        return bytes(out)
    if isinstance(out, str):
        return out.encode('latin-1', errors='replace')
    return out

# ──────────────────────────────────────────────
# LOAD DATASET
# ──────────────────────────────────────────────
dataset_path = None
for p in ['Dataset.csv', os.path.join('..','Dataset.csv'), '/mnt/user-data/uploads/Dataset.csv']:
    if os.path.exists(p):
        dataset_path = p
        break

# ──────────────────────────────────────────────
# LOAD OR TRAIN MODEL
# ──────────────────────────────────────────────
@st.cache_resource
def get_model_and_data(path):
    saved = 'saved_model'
    df_proc, df_full = process_eda_dataset(path)

    if (os.path.exists(os.path.join(saved, 'best_model.pkl')) and
        os.path.exists(os.path.join(saved, 'feature_names.pkl'))):
        # Cargar Pipeline completo (ya incluye el scaler internamente)
        model    = joblib.load(os.path.join(saved, 'best_model.pkl'))
        features = joblib.load(os.path.join(saved, 'feature_names.pkl'))
        metadata = joblib.load(os.path.join(saved, 'metadata.pkl'))

        X = df_proc.drop('Target_Churn', axis=1)
        y = df_proc['Target_Churn']
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # El pipeline escala internamente — no usar scaler.transform()
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]

        result = {
            'model': model, 'y_pred': y_pred, 'y_proba': y_proba,
            'accuracy':  accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall':    recall_score(y_test, y_pred),
            'f1':        f1_score(y_test, y_pred),
            'auc_roc':   roc_auc_score(y_test, y_proba),
            'confusion': confusion_matrix(y_test, y_pred)
        }
        model_name = metadata.get('model_name', 'Best Model')
        return {model_name: result}, model_name, X_test, y_test, features, df_proc, df_full
    else:
        # Fallback: entrenar desde cero con Pipeline + GridSearchCV
        all_results, best_name, X_test, y_test, features = train_best_model(df_proc)
        return all_results, best_name, X_test, y_test, features, df_proc, df_full

# ═══════════════════════════════════════════════
#                    PAGES
# ═══════════════════════════════════════════════

# ──────────────────────────────────────────────
# HOME
# ──────────────────────────────────────────────
if page == "🏠 Inicio":
    st.markdown(f"""
    <div class="main-header">
        <h1>🛒 Customer Churn Predictor</h1>
        <p>{store_name} — Sistema de Predicción de Abandono de Clientes</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    ¡Bienvenido al **Customer Churn Predictor** de **{store_name}**! Esta herramienta permite
    predecir qué clientes están en riesgo de abandonar tu e-commerce, utilizando modelos de
    aprendizaje automático supervisado entrenados con datos históricos de comportamiento.

    **¿Qué puedes hacer aquí?**

    🔹 **Documentación** — Revisa el análisis exploratorio de datos (EDA), la ficha técnica del dataset
    y los detalles del modelo de machine learning seleccionado.

    🔹 **Churn Prediction** — Sube un archivo CSV con datos de clientes o evalúa un cliente individual
    para obtener predicciones de riesgo de abandono con recomendaciones de fidelización.

    ---
    *Analista: {analyst_name} | Fecha: {datetime.now().strftime("%d/%m/%Y")}*
    """)

    if dataset_path is None:
        st.warning("⚠️ No se encontró el Dataset.csv. Por favor colóquelo en el mismo directorio de la app.")

# ──────────────────────────────────────────────
# DOCUMENTACIÓN
# ──────────────────────────────────────────────
elif page == "📖 Documentación":
    st.markdown(f"""
    <div class="main-header">
        <h1>📖 Documentación del Proyecto</h1>
        <p>EDA + Ficha Técnica del Modelo — {store_name}</p>
    </div>
    """, unsafe_allow_html=True)

    if dataset_path is None:
        st.error("No se encontró el Dataset.csv.")
        st.stop()

    data = get_model_and_data(dataset_path)
    all_results, best_name, X_test, y_test, feature_names, df_proc, df_full = data
    raw_df = load_raw_dataset(dataset_path)

    doc_tab1, doc_tab2 = st.tabs(["📊 EDA — Análisis Exploratorio", "🤖 Modelo Seleccionado"])

    # ────── EDA TAB ──────
    with doc_tab1:
        st.subheader("1. Ficha Técnica del Dataset")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Fuente", "Kaggle")
        c2.metric("Instancias", f"{len(raw_df):,}")
        c3.metric("Atributos", raw_df.shape[1])
        c4.metric("Nulos", raw_df.isnull().sum().sum())
        c5.metric("Variable Objetivo", "Target_Churn")

        st.markdown("""
        **Tipo de matriz**: Datos tabulares (CSV) con variables numéricas, categóricas y booleanas.
        El dataset proviene de `online_retail_customer_churn.csv` (Kaggle — Eskikri, 2023)
        con 1,000 instancias y 15 atributos originales.

        **Features descartadas**: `Customer_ID` (identificador), `Age` y `Gender`
        (excluidas para evitar sesgo demográfico), `Annual_Income` y `Average_Transaction_Amount`
        (redundantes — la última es derivable de Total_Spend / Num_of_Purchases).

        **Procesamiento aplicado**: Conversión de booleanos a 0/1, encoding de categóricas,
        eliminación de nulos y duplicados, detección de outliers por IQR.
        """)

        st.divider()
        st.subheader("2. Distribución de Clases (Balance)")

        y_eda = df_proc['Target_Churn']
        pcts = y_eda.value_counts(normalize=True) * 100
        bal = "BALANCEADO" if pcts.min() >= 40 else "DESBALANCEADO"

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Estado", bal)
            st.metric("No Churn (0)", f"{pcts.get(0,0):.1f}%")
            st.metric("Churn (1)", f"{pcts.get(1,0):.1f}%")
        with col2:
            fig, ax = plt.subplots(figsize=(5, 5))
            labels = [f"No Churn ({pcts.get(0,0):.1f}%)", f"Churn ({pcts.get(1,0):.1f}%)"]
            ax.pie(y_eda.value_counts().sort_index(), labels=labels,
                   colors=['#FF5252', '#B71C1C'], startangle=90)
            ax.set_title('Distribución de clases — Target Churn', fontweight='bold')
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.divider()
        st.subheader("3. Detección de Outliers (IQR)")

        variables_num = ['Total_Spend', 'Years_as_Customer', 'Num_of_Purchases',
                        'Average_Transaction_Amount', 'Num_of_Returns', 'Num_of_Support_Contacts',
                        'Satisfaction_Score', 'Last_Purchase_Days_Ago']
        vars_in_df = [v for v in variables_num if v in df_proc.columns]

        Q1 = df_proc[vars_in_df].quantile(0.25)
        Q3 = df_proc[vars_in_df].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df_proc[vars_in_df] < (Q1 - 1.5 * IQR)) |
                    (df_proc[vars_in_df] > (Q3 + 1.5 * IQR))).sum()
        st.dataframe(outliers.to_frame('Outliers'), use_container_width=True)

        st.divider()
        st.subheader("4. Boxplots por Clase")

        box_data = df_proc.melt(id_vars='Target_Churn', value_vars=vars_in_df,
                               var_name='Variable', value_name='Valor')
        fig_box = sns.FacetGrid(box_data, col='Variable', col_wrap=4,
                               sharex=False, sharey=False, height=3.5)
        fig_box.map_dataframe(sns.boxplot, x='Target_Churn', y='Valor',
                             hue='Target_Churn', palette=['#FF5252','#B71C1C'],
                             order=[0,1], legend=False)
        fig_box.set_axis_labels('Churn', 'Valor')
        fig_box.set_titles('{col_name}')
        plt.tight_layout()
        st.pyplot(fig_box.figure, use_container_width=True)
        plt.close('all')

        st.divider()
        st.subheader("5. Histogramas por Clase")

        n_bins = int(1 + math.log2(len(df_proc)))
        all_vars = [v for v in vars_in_df + ['Email_Opt_In','Promotion_Response'] if v in df_proc.columns]
        hist_data = df_proc.melt(id_vars='Target_Churn', value_vars=all_vars,
                                var_name='Variable', value_name='Valor')
        fig_hist = sns.FacetGrid(hist_data, col='Variable', col_wrap=4,
                                hue='Target_Churn', sharex=False, sharey=False,
                                height=3.5, palette={0:'#FF5252',1:'#B71C1C'})
        fig_hist.map_dataframe(sns.histplot, x='Valor', bins=n_bins, alpha=0.5, element='step')
        fig_hist.set_axis_labels('', 'Frecuencia')
        fig_hist.set_titles('{col_name}')
        fig_hist.add_legend()
        plt.tight_layout()
        st.pyplot(fig_hist.figure, use_container_width=True)
        plt.close('all')

        st.divider()
        st.subheader("6. Matriz de Correlación")

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_proc.corr(), annot=True, fmt='.2f', annot_kws={'size':8},
                   cmap='Reds', ax=ax)
        ax.set_title('Mapa de Calor — Correlaciones', fontweight='bold')
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.divider()
        st.subheader("7. Importancia de Características")

        importance_vars = [v for v in feature_names if v in df_proc.columns]
        importancia = df_proc[importance_vars].corrwith(df_proc['Target_Churn']).abs()
        pct_imp = (importancia / importancia.sum()) * 100

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(pct_imp.round(2).to_frame('Importancia (%)'), use_container_width=True)
        with col2:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(pct_imp, labels=pct_imp.index,
                   colors=sns.color_palette('Reds', len(pct_imp)),
                   autopct='%1.1f%%', startangle=90, pctdistance=0.85,
                   textprops={'fontsize':9})
            ax.set_title('Importancia de Características', fontweight='bold')
            st.pyplot(fig, use_container_width=True)
            plt.close()

    # ────── MODEL TAB ──────
    with doc_tab2:
        st.subheader(f"🤖 Modelo Seleccionado: {best_name}")

        best = all_results[best_name]

        # Model description
        descriptions = {
            "Logistic Regression": """**Regresión Logística** es un modelo lineal que aprende una frontera de decisión 
            lineal para clasificación binaria. Los coeficientes indican el peso de cada característica. 
            Es altamente interpretable y rápido de entrenar. Ideal como línea base.""",
            "Random Forest": """**Random Forest** es un ensemble de 100 árboles de decisión que votan por la clase final.
            Captura relaciones no lineales entre variables y proporciona importancia de características nativa.
            Técnica: Bagging (Bootstrap Aggregating).""",
            "Gradient Boosting": """**Gradient Boosting** construye árboles de decisión de forma secuencial, donde cada árbol
            corrige los errores del anterior. Incluye regularización contra overfitting.
            Técnica: Boosting (aprendizaje secuencial con corrección de errores)."""
        }

        st.markdown(descriptions.get(best_name, "Modelo de clasificación supervisada."))

        st.divider()
        st.subheader("📊 Métricas de Evaluación")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{best['accuracy']:.4f}")
        c2.metric("Precision", f"{best['precision']:.4f}")
        c3.metric("Recall", f"{best['recall']:.4f}")
        c4.metric("F1-Score", f"{best['f1']:.4f}")
        c5.metric("AUC-ROC", f"{best['auc_roc']:.4f}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Matriz de Confusión**")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(best['confusion'], annot=True, fmt='d', cmap='Reds', ax=ax,
                       xticklabels=['No Churn','Churn'], yticklabels=['No Churn','Churn'])
            ax.set_xlabel('Predicho'); ax.set_ylabel('Real')
            ax.set_title(f'Matriz de Confusión — {best_name}')
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col2:
            st.markdown("**Curva ROC**")
            fig, ax = plt.subplots(figsize=(5, 4))
            fpr, tpr, _ = roc_curve(y_test, best['y_proba'])
            ax.plot(fpr, tpr, color='#C62828', lw=2, label=f'AUC={best["auc_roc"]:.4f}')
            ax.plot([0,1],[0,1],'k--',lw=1)
            ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
            ax.set_title(f'ROC — {best_name}'); ax.legend()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        if hasattr(best['model'], 'feature_importances_'):
            st.markdown("**Importancia de Características (modelo)**")
            imp = pd.Series(best['model'].feature_importances_, index=feature_names).sort_values()
            fig, ax = plt.subplots(figsize=(8, 4))
            imp.plot(kind='barh', color='#C62828', ax=ax)
            ax.set_title(f'Feature Importance — {best_name}')
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.divider()
        st.markdown("**Ficha Técnica del Modelo**")
        st.markdown(f"""
        | Parámetro | Valor |
        |---|---|
        | Algoritmo | {best_name} |
        | Train/Test Split | 80/20 |
        | Estratificación | Sí (stratify=y) |
        | Escalado | StandardScaler |
        | Features | {len(feature_names)} |
        | Random State | 42 |
        """)

# ──────────────────────────────────────────────
# CHURN PREDICTION
# ──────────────────────────────────────────────
elif page == "🔮 Churn Prediction":
    st.markdown(f"""
    <div class="main-header">
        <h1>🔮 Churn Prediction</h1>
        <p>Predicción de abandono de clientes — {store_name}</p>
    </div>
    """, unsafe_allow_html=True)

    if dataset_path is None:
        st.error("No se encontró el Dataset.csv para entrenar el modelo.")
        st.stop()

    data = get_model_and_data(dataset_path)
    all_results, best_name, X_test, y_test, feature_names, df_proc, df_full = data
    best_model = all_results[best_name]['model']

    pred_mode = st.radio("Modo de predicción", ["📄 Subir dataset (CSV/XLSX)", "👤 Evaluar cliente individual"],
                        horizontal=True)

    # ────── DATASET UPLOAD ──────
    if pred_mode == "📄 Subir dataset (CSV/XLSX)":
        st.markdown("Sube un archivo CSV o XLSX con datos de clientes para predecir su riesgo de churn.")

        with st.expander("ℹ️ Características requeridas en el dataset"):
            for feat, desc in FEATURE_DESCRIPTIONS.items():
                st.markdown(f"- **{feat}**: {desc}")

        uploaded_file = st.file_uploader("Cargar archivo", type=['csv', 'xlsx'])

        if uploaded_file:
            # Validate format
            fname = uploaded_file.name.lower()
            if fname.endswith('.csv'):
                try:
                    df_new = pd.read_csv(uploaded_file)
                except Exception as e:
                    st.markdown(f'<div class="alert-friendly">📋 No pudimos leer este archivo CSV. '
                               f'Verifica que esté bien formado y vuelve a intentarlo.</div>',
                               unsafe_allow_html=True)
                    st.stop()
            elif fname.endswith('.xlsx'):
                try:
                    df_new = pd.read_excel(uploaded_file)
                except Exception as e:
                    st.markdown(f'<div class="alert-friendly">📋 No pudimos leer este archivo Excel. '
                               f'Verifica que sea .xlsx y vuelve a intentarlo.</div>',
                               unsafe_allow_html=True)
                    st.stop()
            else:
                st.markdown('<div class="alert-friendly">📎 Este formato no es compatible. '
                           'Por favor usa archivos .csv o .xlsx únicamente.</div>',
                           unsafe_allow_html=True)
                st.stop()

            if len(df_new) == 0:
                st.markdown('<div class="alert-friendly">📋 El archivo está vacío. '
                           'Sube un archivo con al menos un registro.</div>',
                           unsafe_allow_html=True)
                st.stop()

            st.success(f"✅ Archivo cargado: {len(df_new)} registros, {df_new.shape[1]} columnas")

            # Preprocess — valida columnas y encodea, el Pipeline escala internamente
            df_clean, missing = preprocess_new_data(df_new.copy(), feature_names, None)

            if missing:
                st.markdown(f"""
                <div class="alert-friendly">
                    📋 <strong>No se pudo procesar el archivo</strong><br>
                    Revisa estas características (faltantes, vacías o con valores no válidos):<br>
                    <strong>{', '.join(missing)}</strong><br><br>
                    Asegúrate de incluir todas las columnas requeridas y usar valores numéricos
                    (o categorías válidas en variables categóricas).
                    Puedes ver la lista completa en el desplegable "Características requeridas" arriba.
                </div>
                """, unsafe_allow_html=True)
                st.stop()

            # Predict
            if st.button("🚀 Ejecutar Predicción", type="primary", use_container_width=True):
                proba = best_model.predict_proba(df_clean)[:, 1]
                preds = (proba >= 0.5).astype(int)

                results_df = df_new.copy()
                results_df['Probabilidad_Churn'] = proba.round(4)
                results_df['Prediccion'] = preds
                results_df['Nivel_Riesgo'] = pd.cut(proba, bins=[0, 0.3, 0.6, 1.0],
                                                     labels=['🟢 Bajo', '🟡 Medio', '🔴 Alto'])
                results_df.insert(0, 'Cliente', range(1, len(results_df)+1))

                st.session_state.prediction_results = results_df

        # Show results
        if st.session_state.prediction_results is not None:
            results_df = st.session_state.prediction_results

            st.divider()
            st.subheader("📊 Informe Predictivo")

            # Summary metrics
            total = len(results_df)
            churn_count = results_df['Prediccion'].sum()
            stable = total - churn_count

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Clientes", total)
            c2.metric("🔴 En Riesgo", int(churn_count))
            c3.metric("🟢 Estables", int(stable))
            c4.metric("% Churn", f"{churn_count/max(total,1)*100:.1f}%")

            # Distribution chart
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(5, 4))
                results_df['Nivel_Riesgo'].value_counts().plot(
                    kind='bar', color=['#4CAF50','#FFC107','#C62828'], ax=ax)
                ax.set_title('Distribución por Nivel de Riesgo')
                ax.set_xlabel(''); ax.set_ylabel('Clientes')
                plt.xticks(rotation=0)
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with col2:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.hist(results_df['Probabilidad_Churn'], bins=20, color='#C62828', alpha=0.7, edgecolor='white')
                ax.axvline(0.5, color='black', linestyle='--', label='Umbral (0.5)')
                ax.set_title('Distribución de Probabilidad de Churn')
                ax.set_xlabel('Probabilidad'); ax.set_ylabel('Frecuencia')
                ax.legend()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            # Results table
            st.markdown("**Detalle por Cliente**")
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            # Top at risk
            st.subheader("⚠️ Top Clientes en Mayor Riesgo")
            top_risk = results_df.nlargest(10, 'Probabilidad_Churn')
            st.dataframe(top_risk, use_container_width=True, hide_index=True)

            # Recommendations for top risk
            st.subheader("💡 Recomendaciones de Fidelización")
            st.markdown("*Basadas en el modelo SurtiClientes de Surtifamiliar — Programa de fidelización retail*")

            for _, row in top_risk.head(5).iterrows():
                prob = row['Probabilidad_Churn']
                recs = generate_recommendations(row, prob)
                with st.expander(f"Cliente #{int(row['Cliente'])} — Prob: {prob:.2%} — {row['Nivel_Riesgo']}"):
                    for rec in recs:
                        st.markdown(rec)

            # Export
            st.divider()
            st.subheader("📥 Exportar Resultados")

            c1, c2, c3 = st.columns(3)

            with c1:
                csv_data = results_df.to_csv(index=False).encode('utf-8')
                st.download_button("📄 Descargar CSV", csv_data,
                                  f"churn_prediction_{datetime.now().strftime('%Y%m%d')}.csv",
                                  "text/csv", use_container_width=True)

            with c2:
                buf = io.BytesIO()
                results_df.to_excel(buf, index=False, engine='openpyxl')
                st.download_button("📊 Descargar Excel", buf.getvalue(),
                                  f"churn_prediction_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                  use_container_width=True)

            with c3:
                pdf_bytes = generate_pdf_report(results_df, store_name, analyst_name, best_name)
                st.download_button("📕 Descargar PDF", pdf_bytes,
                                  f"churn_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                                  "application/pdf", use_container_width=True)

            # Clear button
            st.divider()
            if st.button("🗑️ Limpiar resultados para nueva predicción", use_container_width=True):
                st.session_state.prediction_results = None
                st.rerun()

    # ────── INDIVIDUAL CLIENT ──────
    else:
        st.markdown("Evalúa un cliente específico ingresando sus datos manualmente.")

        with st.form("individual_client"):
            st.markdown("### Datos del Cliente")
            c1, c2 = st.columns(2)

            with c1:
                total_spend    = st.number_input("💰 Gasto Total ($)", 0.0, 100000.0, 500.0, 50.0)
                years_customer = st.slider("📅 Años como cliente", 0, 20, 3)
                num_purchases  = st.number_input("🛍️ Número de compras", 0, 500, 15)
                num_returns    = st.number_input("📦 Número de devoluciones", 0, 100, 2)

            with c2:
                support_contacts = st.number_input("📞 Contactos con soporte", 0, 50, 2)
                satisfaction     = st.slider("⭐ Satisfacción (1-10)", 1, 10, 7)
                last_purchase    = st.number_input("📆 Días desde última compra", 0, 730, 30)
                email_opt        = st.selectbox("📧 Suscrito a emails", ['Sí', 'No'])
                promo_resp       = st.selectbox("🏷️ Respuesta a promociones",
                                                ['Respondió', 'Ignoró', 'Desuscrito'])

            submitted = st.form_submit_button("🔮 Predecir", type="primary", use_container_width=True)

        if submitted:
            email_val = 1 if email_opt == 'Sí' else 0
            promo_val = {'Desuscrito':0, 'Respondió':1, 'Ignoró':2}[promo_resp]

            client_data = pd.DataFrame([{
                'Total_Spend':             total_spend,
                'Years_as_Customer':       years_customer,
                'Num_of_Purchases':        num_purchases,
                'Num_of_Returns':          num_returns,
                'Num_of_Support_Contacts': support_contacts,
                'Satisfaction_Score':      satisfaction,
                'Last_Purchase_Days_Ago':  last_purchase,
                'Email_Opt_In':            email_val,
                'Promotion_Response':      promo_val
            }])

            # El Pipeline escala internamente — no usar scaler.transform()
            prob = best_model.predict_proba(client_data[feature_names])[0][1]
            pred = 1 if prob >= 0.5 else 0

            st.divider()

            if pred == 1:
                risk_color = "#C62828"
                risk_label = "🔴 ALTO RIESGO DE CHURN"
            elif prob > 0.3:
                risk_color = "#F57F17"
                risk_label = "🟡 RIESGO MEDIO"
            else:
                risk_color = "#2E7D32"
                risk_label = "🟢 CLIENTE ESTABLE"

            st.markdown(f"""
            <div style="text-align:center; padding:2rem; background:linear-gradient(135deg, {risk_color}22, {risk_color}11);
                        border-radius:12px; border-left:6px solid {risk_color};">
                <h2 style="color:{risk_color}; margin:0;">{risk_label}</h2>
                <p style="font-size:2.5rem; font-weight:700; color:{risk_color}; margin:0.5rem 0;">
                    {prob:.1%}
                </p>
                <p style="color:#666;">Probabilidad de abandono</p>
            </div>
            """, unsafe_allow_html=True)

            # Recommendations
            st.subheader("💡 Recomendaciones de Fidelización")
            recs = generate_recommendations(client_data.iloc[0], prob)
            for rec in recs:
                st.markdown(f'<div class="rec-card">{rec}</div>', unsafe_allow_html=True)
