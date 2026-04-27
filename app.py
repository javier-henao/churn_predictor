import streamlit as st

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Customer Churn Predictor")
st.markdown(
    "Selecciona un módulo desde el menú lateral de páginas:\n\n"
    "- `Churn Predictor`\n"
    "- `Model Training`"
)

