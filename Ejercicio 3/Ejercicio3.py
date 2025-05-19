import pandas as pd
import random
import os

# Paso 1: Leer archivo CSV desde la misma carpeta del script
def cargar_matriz_distancias():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ruta_csv = os.path.join(script_dir, "distancias.csv")
    try:
        df = pd.read_csv(ruta_csv, index_col=0)
        print("✅ Archivo 'distancias.csv' cargado correctamente desde:\n", ruta_csv)
        return df
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{ruta_csv}'. Verifica que exista y esté en la misma carpeta que el script.")
        exit()

# Paso 2: Calcular distancia total de una ruta
def calcular_distancia_total(ruta, matriz):
    distancia_total = 0
    for i in range(len(ruta) - 1):
        desde = ruta[i]
        hasta = ruta[i + 1]
        distancia_total += matriz.loc[desde, hasta]
    # Retornar al punto de inicio
    distancia_total += matriz.loc[ruta[-1], ruta[0]]
    return distancia_total

# Paso 3: Generar vecino (intercambio de 2 nodos)
def generar_vecino(ruta_actual):
    vecino = ruta_actual[:]
    i, j = random.sample(range(len(vecino)), 2)
    vecino[i], vecino[j] = vecino[j], vecino[i]
    return vecino

# Paso 4: Algoritmo Hill Climbing
def hill_climbing(matriz, max_iter=1000):
    laboratorios = list(matriz.index)
    actual = random.sample(laboratorios, len(laboratorios))
    mejor = actual
    mejor_distancia = calcular_distancia_total(mejor, matriz)

    for _ in range(max_iter):
        vecino = generar_vecino(mejor)
        distancia_vecino = calcular_distancia_total(vecino, matriz)
        if distancia_vecino < mejor_distancia:
            mejor = vecino
            mejor_distancia = distancia_vecino

    return mejor, mejor_distancia

# Ejecutar el algoritmo
matriz_distancias = cargar_matriz_distancias()
ruta_optima, distancia_total = hill_climbing(matriz_distancias, max_iter=1000)

# Mostrar resultado
print("\n🚶 Ruta óptima de revisión de laboratorios:")
for i, lab in enumerate(ruta_optima):
    print(f"{i+1}. {lab}")
print(f"\n📏 Distancia total recorrida: {distancia_total:.2f} metros")
