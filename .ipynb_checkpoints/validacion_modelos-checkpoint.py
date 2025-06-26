"""
Script: validacion_modelos.py
Descripción: Comparación de algoritmos para generación de horarios (Random Forest, Lineal, Genético simulado)
Autor: Julio César Santa Rita Ávila
Requisitos:
    pip install numpy pandas matplotlib scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Simulación de datos
np.random.seed(42)
n_samples = 200
X = pd.DataFrame({
    'docente_id': np.random.randint(1, 10, size=n_samples),
    'materia_id': np.random.randint(1, 25, size=n_samples),
    'salon_id': np.random.randint(1, 15, size=n_samples),
    'hora_inicio': np.random.randint(8, 21, size=n_samples),
    'dia': np.random.randint(1, 6, size=n_samples),
    'duracion': np.random.choice([1.5, 2.5, 3.0], size=n_samples)
})
y = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelo 1: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_proba_rf)

# Modelo 2: Regresión Lineal
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
y_proba_lr = lr_model.predict_proba(X_test)[:, 1]
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_proba_lr)

# Modelo 3: Genético simulado
y_pred_ga = np.random.choice([0, 1], size=len(y_test), p=[0.4, 0.6])
y_proba_ga = np.random.rand(len(y_test))
precision_ga = precision_score(y_test, y_pred_ga)
recall_ga = recall_score(y_test, y_pred_ga)
f1_ga = f1_score(y_test, y_pred_ga)
auc_ga = roc_auc_score(y_test, y_proba_ga)

# Resultados
results = pd.DataFrame({
    "Algoritmo": ["Lineal", "Genético", "Random Forest"],
    "Precisión": [precision_lr, precision_ga, precision_rf],
    "Recall": [recall_lr, recall_ga, recall_rf],
    "F1-Score": [f1_lr, f1_ga, f1_rf],
    "AUC": [auc_lr, auc_ga, auc_rf]
})
print(results)

# Gráfica comparativa
labels = ["Precisión", "Recall", "F1-Score", "AUC"]
x = np.arange(len(labels))
width = 0.25
fig, ax = plt.subplots()
ax.bar(x - width, results.iloc[0, 1:], width, label='Lineal')
ax.bar(x, results.iloc[1, 1:], width, label='Genético')
ax.bar(x + width, results.iloc[2, 1:], width, label='Random Forest')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim([0, 1])
ax.legend()
plt.title("Comparación de Métricas")
plt.tight_layout()
plt.show()
