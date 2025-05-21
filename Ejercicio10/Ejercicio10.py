import os
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# --- Cargar datos ---
def cargar_datos():
    # Ajusta aquí la ruta a tu archivo CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ruta_csv = os.path.join(script_dir, "matriculas.csv")
    df = pd.read_csv(ruta_csv, sep="\t")
    print("✅ Datos cargados desde:", ruta_csv)
    return df

# --- Preprocesamiento ---
def preprocesar(df):
    X = df[['Credits', 'Prev_GPA', 'Extracurricular_hours']].values
    y = df['Category'].values
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42), encoder

# --- Crear modelo dinámico con sklearn MLPClassifier ---
def crear_modelo(n_capas, n_neuronas, lr):
    # La estructura oculta es una tupla con n_capas veces n_neuronas
    hidden_layers = tuple([n_neuronas] * n_capas)
    model = MLPClassifier(hidden_layer_sizes=hidden_layers, learning_rate_init=lr, max_iter=20, random_state=42)
    return model

# --- Evaluar genotipo ---
def evaluar(genotipo, X_train, y_train, X_test, y_test):
    n_capas, n_neuronas, lr = genotipo
    model = crear_modelo(n_capas, n_neuronas, lr)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred), model

# --- Mutar genotipo ---
def mutar(genotipo):
    n_capas, n_neuronas, lr = genotipo
    if random.random() < 0.33:
        n_capas = max(1, min(3, n_capas + random.choice([-1, 1])))
    if random.random() < 0.33:
        n_neuronas = max(4, min(128, n_neuronas + random.choice([-4, 4])))
    if random.random() < 0.33:
        lr = max(0.0001, min(0.1, lr + random.uniform(-0.01, 0.01)))
    return (n_capas, n_neuronas, lr)

# --- Evolución con hill climbing ---
def evolucionar(X_train, y_train, X_test, y_test, n_iter=30):
    mejor_genotipo = (random.randint(1, 3), random.randint(4, 128), round(random.uniform(0.001, 0.05), 4))
    mejor_fitness, mejor_modelo = evaluar(mejor_genotipo, X_train, y_train, X_test, y_test)

    print(f"Iteración 0: fitness={mejor_fitness:.4f}, genotipo={mejor_genotipo}")

    for i in range(1, n_iter + 1):
        nuevo_genotipo = mutar(mejor_genotipo)
        nuevo_fitness, nuevo_modelo = evaluar(nuevo_genotipo, X_train, y_train, X_test, y_test)

        if nuevo_fitness > mejor_fitness:
            mejor_genotipo, mejor_fitness, mejor_modelo = nuevo_genotipo, nuevo_fitness, nuevo_modelo
            print(f"🔁 Iteración {i}: NUEVO MEJOR fitness = {mejor_fitness:.4f} con genotipo = {mejor_genotipo}")

    return mejor_genotipo, mejor_fitness, mejor_modelo

# --- Main ---
if __name__ == "__main__":
    df = cargar_datos()
    (X_train, X_test, y_train, y_test), encoder = preprocesar(df)
    mejor_genotipo, mejor_fitness, modelo_final = evolucionar(X_train, y_train, X_test, y_test)

    print("\n✅ Arquitectura final:")
    print(f"- Capas ocultas: {mejor_genotipo[0]}")
    print(f"- Neuronas por capa: {mejor_genotipo[1]}")
    print(f"- Learning rate: {mejor_genotipo[2]:.5f}")
    print(f"🎯 Accuracy final: {mejor_fitness:.4f}")
