"""
Modelo 2: Random Forest — Customer Churn Prediction
Pipeline: StandardScaler → RandomForestClassifier + GridSearchCV
Consistente con 01_model_training.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline
import joblib
 
 
def load_and_preprocess(path='Dataset.csv'):
    df = pd.read_csv(path)
    df = df.dropna().drop_duplicates()
 
    # Drop Customer_ID
    if 'Customer_ID' in df.columns:
        df = df.drop('Customer_ID', axis=1)
 
    # Drop columnas excluidas por sesgo y redundancia
    cols_to_drop = [c for c in ['Age', 'Gender', 'Annual_Income', 'Average_Transaction_Amount']
                    if c in df.columns]
    df = df.drop(cols_to_drop, axis=1)
 
    # Encode
    df['Email_Opt_In']       = df['Email_Opt_In'].astype(int)
    df['Target_Churn']       = df['Target_Churn'].astype(int)
    df['Promotion_Response'] = df['Promotion_Response'].replace(
        {'Unsubscribed': 0, 'Responded': 1, 'Ignored': 2}).astype(int)
 
    return df
 
 
def train_random_forest(df, test_size=0.2, random_state=42, cv_folds=5):
    X = df.drop('Target_Churn', axis=1)
    y = df['Target_Churn']
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
 
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
 
    # Pipeline: Scaler + Modelo (sin leakage)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=random_state))
    ])
 
    # GridSearchCV sobre el pipeline completo
    param_grid = {
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth':    [None, 5, 10],
    }
 
    gs = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1)
    gs.fit(X_train, y_train)
 
    best_pipe  = gs.best_estimator_
    best_index = gs.best_index_
    y_pred     = best_pipe.predict(X_test)
    y_proba    = best_pipe.predict_proba(X_test)[:, 1]
 
    # Métricas
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    cv_f1_mean = gs.cv_results_['mean_test_score'][best_index]
    cv_f1_std  = gs.cv_results_['std_test_score'][best_index]
 
    # Indicador de overfitting
    gap         = abs(f1 - cv_f1_mean)
    overfitting = "⚠️  Posible sobreajuste" if gap > 0.05 else "✅  Generaliza bien"
 
    print("=" * 55)
    print("RANDOM FOREST — Resultados")
    print("=" * 55)
    print(f"Accuracy:       {acc:.4f}")
    print(f"Precision:      {prec:.4f}")
    print(f"Recall:         {rec:.4f}")
    print(f"F1-Score:       {f1:.4f}")
    print(f"AUC-ROC:        {auc:.4f}")
    print(f"CV F1 (mean):   {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")
    print(f"Overfitting gap:{gap:.4f} — {overfitting}")
    print(f"\nMejores hiperparámetros: {gs.best_params_}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
 
    # Feature importance
    clf = best_pipe.named_steps['clf']
    imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nFeature Importance:")
    print(imp.round(4))
 
    return best_pipe
 
 
if __name__ == '__main__':
    df    = load_and_preprocess()
    model = train_random_forest(df)
    joblib.dump(model, 'model_random_forest.pkl')
    print("\nModelo guardado en: model_random_forest.pkl")
