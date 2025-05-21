import os
import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def cargar_datos():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ruta_csv = os.path.join(script_dir, "emails.csv")

    try:
        df = pd.read_csv(ruta_csv)
        print("✅ Datos cargados desde:", ruta_csv)
        print("🧾 Columnas encontradas:", df.columns.tolist())
        return df
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo 'emails.csv' en:", ruta_csv)
        exit()

def preprocesar_datos(df):
    if not set(['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Spam']).issubset(df.columns):
        raise ValueError("❌ El archivo debe contener las columnas 'Feature1' a 'Feature5' y 'Spam'.")

    X = df[['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']].values
    y = df['Spam'].values
    return X, y

def evaluar(individuo, X, y):
    umbral = individuo[0]

    # Entrenamos el clasificador
    clf = LogisticRegression()
    clf.fit(X, y)

    # Probabilidad de ser spam
    y_prob = clf.predict_proba(X)[:, 1]
    y_pred = (y_prob > umbral).astype(int)

    return f1_score(y, y_pred),  # DEAP espera tuplas

def mutacion_colina(individuo, sigma=0.05):
    nuevo = np.array(individuo) + np.random.normal(0, sigma, size=len(individuo))
    nuevo = np.clip(nuevo, 0, 1)  # mantener dentro del rango válido
    return nuevo.tolist()

def hill_climbing(evaluar, X, y, generaciones=50):
    individuo = [random.uniform(0.3, 0.7)]  # solo umbral
    mejor_fitness = evaluar(individuo, X, y)[0]

    for _ in range(generaciones):
        vecino = mutacion_colina(individuo)
        fitness_vecino = evaluar(vecino, X, y)[0]

        if fitness_vecino > mejor_fitness:
            individuo = vecino
            mejor_fitness = fitness_vecino

    return individuo, mejor_fitness

# === MAIN ===
df = cargar_datos()
X, y = preprocesar_datos(df)
mejor_individuo, mejor_f1 = hill_climbing(evaluar, X, y)

print("\n🏁 Mejor configuración encontrada:")
print("📏 Umbral de decisión:", round(mejor_individuo[0], 3))
print("🎯 F1-Score obtenido:", round(mejor_f1, 4))
