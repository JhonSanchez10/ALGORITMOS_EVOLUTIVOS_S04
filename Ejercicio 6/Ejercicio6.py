import os
import pandas as pd
import random

# Paso 1: Cargar datos desde la misma carpeta del script
def cargar_datos():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ruta_csv = os.path.join(script_dir, "preguntas_examen.csv")
    try:
        df = pd.read_csv(ruta_csv)
        print("✅ Archivo cargado desde:\n", ruta_csv)
        return df
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{ruta_csv}'")
        exit()

# Paso 2: Función de aptitud
def evaluar(solucion, df):
    tiempo_total = sum(df['Time_min'][i] for i in range(len(solucion)) if solucion[i] == 1)
    dificultad_total = sum(df['Difficulty'][i] for i in range(len(solucion)) if solucion[i] == 1)
    
    if tiempo_total > 90 or not (180 <= dificultad_total <= 200):
        return float('-inf')  # Penaliza si no cumple restricciones
    
    return dificultad_total  # Se busca maximizar dificultad dentro del rango

# Paso 3: Generar vecino cambiando 1 bit
def vecino(solucion):
    vecino = solucion.copy()
    i = random.randint(0, len(solucion) - 1)
    vecino[i] = 1 - vecino[i]
    return vecino

# Paso 4: Algoritmo Hill Climbing
def hill_climbing(df, iteraciones=1000):
    actual = [random.randint(0, 1) for _ in range(len(df))]
    mejor_aptitud = evaluar(actual, df)
    
    for _ in range(iteraciones):
        nuevo = vecino(actual)
        aptitud_nuevo = evaluar(nuevo, df)
        if aptitud_nuevo > mejor_aptitud:
            actual = nuevo
            mejor_aptitud = aptitud_nuevo
    
    return actual, mejor_aptitud

# Paso 5: Ejecutar
df = cargar_datos()
mejor_solucion, mejor_valor = hill_climbing(df)

# Paso 6: Mostrar resultados
print("\n📘 Preguntas seleccionadas para el examen:")
preguntas = []
tiempo_total = 0
dificultad_total = 0
for i, bit in enumerate(mejor_solucion):
    if bit == 1:
        fila = df.iloc[i]
        preguntas.append(fila['QuestionID'])
        tiempo_total += fila['Time_min']
        dificultad_total += fila['Difficulty']
        print(f" - {fila['QuestionID']} (Dif: {fila['Difficulty']}, Tiempo: {fila['Time_min']} min)")

print(f"\n📊 Total de preguntas: {len(preguntas)}")
print(f"🕒 Tiempo total: {tiempo_total} min")
print(f"🔥 Dificultad total: {dificultad_total}")
