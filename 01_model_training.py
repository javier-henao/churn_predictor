"""
========================================================
  CUSTOMER CHURN PREDICTOR — Model Training & Selection
  Streamlit App for training and comparing ML models
========================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from datetime import datetime
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
    page_title="Customer Churn — Model Training",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# THEME / DARK MODE
# ──────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def get_colors():
    if st.session_state.dark_mode:
        return {
            "bg": "#0E1117", "card": "#1E2130", "text": "#FAFAFA",
            "primary": "#FF4B4B", "secondary": "#FF8C8C", "accent": "#C62828",
            "success": "#4CAF50", "warning": "#FFC107"
        }
    return {
        "bg": "#FFFFFF", "card": "#F8F9FA", "text": "#1E1E1E",
        "primary": "#C62828", "secondary": "#EF5350", "accent": "#B71C1C",
        "success": "#2E7D32", "warning": "#F57F17"
    }
# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=60)
    st.markdown("### ⚙️ Configuración")

    dark = st.toggle("🌙 Modo Oscuro", value=st.session_state.dark_mode)
    if dark != st.session_state.dark_mode:
        st.session_state.dark_mode = dark
        st.rerun()

    st.divider()
    st.markdown("**Parámetros de entrenamiento**")
    test_size = st.slider("% Datos de prueba", 10, 40, 20, 5) / 100
    random_state = st.number_input("Random State", 0, 999, 42)
    cv_folds = st.slider("K-Fold CV (folds)", 3, 10, 5)

    st.divider()
    store_name = st.text_input("Nombre del comercio", "Mi E-Commerce")
    analyst_name = st.text_input("Analista", "Equipo Data Science")

    st.divider()
    st.caption("Customer Churn Predictor v1.0")
    st.caption(f"© {datetime.now().year}")
# ──────────────────────────────────────────────
# COLOR
# ──────────────────────────────────────────────
colors = get_colors() 
# ──────────────────────────────────────────────
# CUSTOM CSS
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
 
/* Texto — solo sobre selectores específicos de Streamlit,
   NO un selector global que rompe widgets */
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
# HEADER
# ──────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <h1>🛒 Customer Churn Predictor — Training Lab</h1>
    <p>{store_name} | Analista: {analyst_name}</p>
</div>
""", unsafe_allow_html=True)

st.markdown("Bienvenido al módulo de **entrenamiento y selección de modelos**. "
            "Aquí se entrenan tres algoritmos de aprendizaje supervisado y se selecciona "
            "el mejor según múltiples métricas de evaluación.")

# ──────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ──────────────────────────────────────────────
@st.cache_data
def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Nulls & duplicates
    df = df.dropna().drop_duplicates()

    # Drop Customer_ID and others features
    if 'Customer_ID' in df.columns:
        df = df.drop('Customer_ID', axis=1)
    
    # Drop columnas excluidas por sesgo y redundancia
    cols_to_drop = [c for c in ['Age', 'Gender', 'Annual_Income', 'Average_Transaction_Amount'] if c in df.columns]
    df = df.drop(cols_to_drop, axis=1)
            
    # Encode
    df['Email_Opt_In'] = df['Email_Opt_In'].astype(int)
    df['Target_Churn'] = df['Target_Churn'].astype(int)
    df['Promotion_Response'] = df['Promotion_Response'].replace(
        {'Unsubscribed': 0, 'Responded': 1, 'Ignored': 2}).astype(int)

    return df

# Try loading default dataset
dataset_path = None
#Aqui se borro
for p in ['Dataset.csv', os.path.join('..', 'Dataset.csv')]:
    if os.path.exists(p):
        dataset_path = p
        break

uploaded = st.file_uploader("📂 Cargar Dataset (CSV)", type=['csv'])
if uploaded:
    import tempfile
    temp_path = os.path.join(tempfile.gettempdir(), 'dataset_training.csv')
    with open(temp_path, 'wb') as f:
        f.write(uploaded.getvalue())
    dataset_path = temp_path

if dataset_path is None:
    st.info("Por favor, cargue el archivo Dataset.csv para comenzar el entrenamiento.")
    st.stop()

df = load_and_preprocess(dataset_path)

# Features & Target
X = df.drop('Target_Churn', axis=1)
y = df['Target_Churn']
feature_names = X.columns.tolist()

# ──────────────────────────────────────────────
# DATASET OVERVIEW
# ──────────────────────────────────────────────
with st.expander("📊 Ficha Técnica del Dataset", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Instancias", f"{len(df):,}")
    c2.metric("Características", len(feature_names))
    c3.metric("Variable Objetivo", "Target_Churn")
    balance = y.value_counts(normalize=True)
    bal_label = "Balanceado" if balance.min() >= 0.4 else "Desbalanceado"
    c4.metric("Balance", bal_label)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Distribución de clases**")
        fig, ax = plt.subplots(figsize=(4, 4))
        pcts = y.value_counts(normalize=True) * 100
        labels = [f"No Churn ({pcts.get(0,0):.1f}%)", f"Churn ({pcts.get(1,0):.1f}%)"]
        ax.pie(y.value_counts().sort_index(), labels=labels,
               colors=['#EF5350', '#B71C1C'], startangle=90)
        ax.set_title("Target Churn", fontweight='bold')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("**Características utilizadas**")
        st.dataframe(pd.DataFrame({
            'Feature': feature_names,
            'Tipo': [str(X[c].dtype) for c in feature_names],
            'Min': [X[c].min() for c in feature_names],
            'Max': [X[c].max() for c in feature_names],
            'Media': [X[c].mean().round(2) for c in feature_names]
        }), use_container_width=True, hide_index=True)
        
# ──────────────────────────────────────────────
# TRAIN MODELS
# ──────────────────────────────────────────────
st.divider()
st.subheader("🚀 Entrenamiento de Modelos")

if st.button("▶️ Entrenar los 3 modelos", type="primary", use_container_width=True):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    pipelines = {
        "Logistic Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=random_state))
        ]),
        "Random Forest": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(random_state=random_state))
        ]),
        "Gradient Boosting": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(random_state=random_state))
        ]),
    }

    param_grids = {
        "Logistic Regression": {
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__solver': ['lbfgs', 'liblinear']
        },
        "Random Forest": {
            'clf__n_estimators': [50, 100, 200],
            'clf__max_depth': [None, 5, 10],
        },
        "Gradient Boosting": {
            'clf__n_estimators': [50, 100],
            'clf__learning_rate': [0.05, 0.1, 0.2],
            'clf__max_depth': [3, 5],
        },
    }

    results = {}
    progress = st.progress(0, text="Entrenando modelos...")

    for i, (name, pipe) in enumerate(pipelines.items()):
        progress.progress(i / 3, text=f"Entrenando {name}...")

        # GridSearchCV sobre el pipeline completo
        gs = GridSearchCV(pipe, param_grids[name], cv=cv, scoring='f1', n_jobs=-1)
        gs.fit(X_train, y_train)

        best_pipe  = gs.best_estimator_
        best_index = gs.best_index_
        y_pred     = best_pipe.predict(X_test)
        y_proba    = best_pipe.predict_proba(X_test)[:, 1]

        results[name] = {
            'model':       best_pipe,
            'best_params': gs.best_params_,
            'y_pred':      y_pred,
            'y_proba':     y_proba,
            'accuracy':    accuracy_score(y_test, y_pred),
            'precision':   precision_score(y_test, y_pred),
            'recall':      recall_score(y_test, y_pred),
            'f1':          f1_score(y_test, y_pred),
            'auc_roc':     roc_auc_score(y_test, y_proba),
            'cv_f1_mean':  gs.cv_results_['mean_test_score'][best_index],
            'cv_f1_std':   gs.cv_results_['std_test_score'][best_index],
            'confusion':   confusion_matrix(y_test, y_pred),
            'report':      classification_report(y_test, y_pred, output_dict=True)
        }
        
    progress.progress(1.0, text="✅ Entrenamiento completado")

    # Store in session
    st.session_state['results'] = results
    st.session_state['X_test'] = X_test
    st.session_state['y_test'] = y_test
    st.session_state['X_train'] = X_train
    st.session_state['feature_names'] = feature_names

# ──────────────────────────────────────────────
# SHOW RESULTS
# ──────────────────────────────────────────────
if 'results' in st.session_state:
    results = st.session_state['results']
    y_test = st.session_state['y_test']

    # Comparison table
    st.subheader("📋 Comparación de Modelos")
    comp_df = pd.DataFrame({
        name: {
            'Accuracy': f"{r['accuracy']:.4f}",
            'Precision': f"{r['precision']:.4f}",
            'Recall': f"{r['recall']:.4f}",
            'F1-Score': f"{r['f1']:.4f}",
            'AUC-ROC': f"{r['auc_roc']:.4f}",
            'CV F1 (mean±std)': f"{r['cv_f1_mean']:.4f} ± {r['cv_f1_std']:.4f}"
        } for name, r in results.items()
    }).T
    st.dataframe(comp_df, use_container_width=True)

    # Winner
    best_name = max(results, key=lambda k: results[k]['f1'])
    best = results[best_name]
    st.markdown(f"""
    <div style="text-align:center; margin: 1rem 0;">
        <span class="winner-badge">🏆 Modelo Seleccionado: {best_name} — F1-Score: {best['f1']:.4f}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Mejor hiperparámetros
    with st.expander("🔧 Mejores hiperparámetros por modelo"):
        for name, r in results.items():
            st.markdown(f"**{name}:** `{r['best_params']}`")

    # Tabs for each model
    tabs = st.tabs(list(results.keys()) + ["📈 ROC Comparativa"])

    for idx, (name, r) in enumerate(results.items()):
        with tabs[idx]:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy", f"{r['accuracy']:.4f}")
            c2.metric("Precision", f"{r['precision']:.4f}")
            c3.metric("Recall", f"{r['recall']:.4f}")
            c4.metric("F1-Score", f"{r['f1']:.4f}")
            c5.metric("AUC-ROC", f"{r['auc_roc']:.4f}")

            # Indicadore de overfitting basado en la brecha entre CV F1 y Test F1
            gap = abs(r['f1'] - r['cv_f1_mean'])
            overfitting = "⚠️ Posible sobreajuste" if gap > 0.05 else "✅ Generaliza bien"
            st.caption(f"CV F1: {r['cv_f1_mean']:.4f} ± {r['cv_f1_std']:.4f} — {overfitting}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Matriz de Confusión**")
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(r['confusion'], annot=True, fmt='d',
                           cmap='Reds', ax=ax,
                           xticklabels=['No Churn', 'Churn'],
                           yticklabels=['No Churn', 'Churn'])
                ax.set_xlabel('Predicho')
                ax.set_ylabel('Real')
                ax.set_title(f'Matriz de Confusión — {name}')
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with col2:
                st.markdown("**Curva ROC**")
                fig, ax = plt.subplots(figsize=(5, 4))
                fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
                ax.plot(fpr, tpr, color='#C62828', lw=2,
                       label=f'AUC = {r["auc_roc"]:.4f}')
                ax.plot([0, 1], [0, 1], 'k--', lw=1)
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'ROC — {name}')
                ax.legend()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            # Feature importance for tree models
            clf = r['model']['clf']
            if hasattr(clf, 'feature_importances_'):
                st.markdown("**Importancia de Características**")
                imp = pd.Series(clf.feature_importances_,
                                index=st.session_state['feature_names'])
                imp = imp.sort_values(ascending=True)
                fig, ax = plt.subplots(figsize=(8, 4))
                imp.plot(kind='barh', color='#C62828', ax=ax)
                ax.set_title(f'Feature Importance — {name}')
                st.pyplot(fig, use_container_width=True)
                plt.close()
            elif hasattr(clf, 'coef_'):
                st.markdown("**Coeficientes del Modelo**")
                coef = pd.Series(clf.coef_[0],
                                index=st.session_state['feature_names'])
                coef = coef.sort_values(ascending=True)
                colors_bar = ['#2E7D32' if v < 0 else '#C62828' for v in coef.values]
                fig, ax = plt.subplots(figsize=(8, 4))
                coef.plot(kind='barh', color=colors_bar, ax=ax)
                ax.set_title(f'Coeficientes — {name}')
                st.pyplot(fig, use_container_width=True)
                plt.close()

    # Combined ROC
    with tabs[-1]:
        fig, ax = plt.subplots(figsize=(8, 6))
        palette = ['#C62828', '#1565C0', '#2E7D32']
        for idx2, (name, r) in enumerate(results.items()):
            fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
            ax.plot(fpr, tpr, color=palette[idx2], lw=2,
                   label=f'{name} (AUC={r["auc_roc"]:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Curvas ROC Comparativas')
        ax.legend(loc='lower right')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ──────────────────────────────────────────
    # EXPORT BEST MODEL
    # ──────────────────────────────────────────
    st.divider()
    st.subheader("💾 Exportar Modelo Seleccionado")

    if st.button("Guardar modelo ganador", type="primary"):
        save_dir = "saved_model"
        os.makedirs(save_dir, exist_ok=True)

        best_model = results[best_name]['model']

        joblib.dump(best_model, os.path.join(save_dir, 'best_model.pkl'))
        joblib.dump(st.session_state['feature_names'], os.path.join(save_dir, 'feature_names.pkl'))

        # Save metadata
        metadata = {
            'model_name':  best_name,
            'best_params': results[best_name]['best_params'],
            'accuracy':    best['accuracy'],
            'precision':   best['precision'],
            'recall':      best['recall'],
            'f1':          best['f1'],
            'auc_roc':     best['auc_roc'],
            'cv_f1_mean':  best['cv_f1_mean'],
            'cv_f1_std':   best['cv_f1_std'],
            'features':    st.session_state['feature_names'],
            'train_date':  datetime.now().isoformat(),
            'test_size':   test_size,
            'random_state': random_state,
            'cv_folds':    cv_folds,
            'store_name':  store_name,
            'analyst':     analyst_name,
            'all_results': {
                name: {k: v for k, v in r.items()
                       if k not in ['model', 'y_pred', 'y_proba', 'confusion', 'report']}
                for name, r in results.items()
            }
        }
        joblib.dump(metadata, os.path.join(save_dir, 'metadata.pkl'))

        st.success(f"✅ Modelo **{best_name}** guardado en `{save_dir}/`")
        st.info("Ahora puede ejecutar la aplicación principal (`02_churn_predictor.py`) "
                "que cargará automáticamente este modelo.")
