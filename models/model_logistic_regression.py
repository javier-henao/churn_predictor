"""
Modelo 1: Logistic Regression — Customer Churn Prediction
Pipeline: StandardScaler → LogisticRegression
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline
import joblib


def load_and_preprocess(path='Dataset.csv'):
    df = pd.read_csv(path)
    df = df.dropna().drop_duplicates()
    if 'Customer_ID' in df.columns:
        df = df.drop('Customer_ID', axis=1)
    df['Email_Opt_In'] = df['Email_Opt_In'].astype(int)
    df['Target_Churn'] = df['Target_Churn'].astype(int)
    df['Promotion_Response'] = df['Promotion_Response'].replace(
        {'Unsubscribed':0,'Responded':1,'Ignored':2}).astype(int)
    # Drop columnas excluidas por sesgo y redundancia
    cols_to_drop = [c for c in ['Age', 'Gender', 'Annual_Income', 'Average_Transaction_Amount'] if c in df.columns]
    df = df.drop(cols_to_drop, axis=1)
    return df


def train_logistic_regression(df, test_size=0.2, random_state=42):
    X = df.drop('Target_Churn', axis=1)
    y = df['Target_Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)

    # Pipeline: Scaler + Model
    pipeline = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, random_state=random_state))])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1')

    print("=" * 50)
    print("LOGISTIC REGRESSION — Resultados")
    print("=" * 50)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC:   {roc_auc_score(y_test, y_proba):.4f}")
    print(f"CV F1:     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Coeficientes
    model = pipeline.named_steps['clf']
    coefs = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)
    print("\nCoeficientes (ordenados):")
    print(coefs.round(4))

    return pipeline


if __name__ == '__main__':
    df = load_and_preprocess()
    model = train_logistic_regression(df)
    joblib.dump(model, 'model_logistic_regression.pkl')
    print("\nModelo guardado en: model_logistic_regression.pkl")
