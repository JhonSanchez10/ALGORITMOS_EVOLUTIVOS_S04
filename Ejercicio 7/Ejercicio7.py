import pandas as pd
import numpy as np
import random
import os

# Cargar CSV desde la ruta del script
def cargar_datos():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ruta_csv = os.path.join(script_dir, "estudiantes.csv")
    try:
        df = pd.read_csv(ruta_csv)
        print("✅ Datos cargados desde:", ruta_csv)
        return df
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo 'estudiantes.csv'")
        exit()

# Calcular función de aptitud: varianza GPA + penalización por desbalance de skills
def calcular_aptitud(equipos, df):
    varianzas = []
    skill_counts = []
    for equipo in equipos:
        gpas = df.loc[equipo, 'GPA']
        skills = df.loc[equipo, 'Skill'].value_counts()
        varianzas.append(np.var(gpas))
        skill_counts.append(skills)

    total_varianza = sum(varianzas)

    # Penalización: desviación del promedio ideal por habilidad
    habilidades = df['Skill'].unique()
    ideal_por_equipo = {h: df['Skill'].value_counts()[h] / len(equipos) for h in habilidades}
    penalizacion = 0
    for counts in skill_counts:
        for h in habilidades:
            real = counts.get(h, 0)
            ideal = ideal_por_equipo[h]
            penalizacion += abs(real - ideal)

    return total_varianza + penalizacion

# Crear una solución inicial aleatoria: 5 equipos de 4
def generar_solucion_inicial(num_estudiantes=20, num_equipos=5):
    indices = list(range(num_estudiantes))
    random.shuffle(indices)
    equipos = [indices[i::num_equipos] for i in range(num_equipos)]
    return equipos

# Generar vecino: intercambiar dos alumnos de distintos equipos
def generar_vecino(equipos):
    equipos_copia = [list(e) for e in equipos]
    eq1, eq2 = random.sample(range(len(equipos)), 2)
    if not equipos_copia[eq1] or not equipos_copia[eq2]:
        return equipos_copia  # evitar errores
    idx1 = random.randint(0, len(equipos_copia[eq1]) - 1)
    idx2 = random.randint(0, len(equipos_copia[eq2]) - 1)
    equipos_copia[eq1][idx1], equipos_copia[eq2][idx2] = equipos_copia[eq2][idx2], equipos_copia[eq1][idx1]
    return equipos_copia

# Ejecutar Hill Climbing
def hill_climbing(df, iteraciones=1000):
    equipos = generar_solucion_inicial()
    mejor_aptitud = calcular_aptitud(equipos, df)

    for _ in range(iteraciones):
        vecino = generar_vecino(equipos)
        aptitud_vecino = calcular_aptitud(vecino, df)
        if aptitud_vecino < mejor_aptitud:
            equipos = vecino
            mejor_aptitud = aptitud_vecino

    return equipos, mejor_aptitud

# Mostrar resultados
def mostrar_resultados(equipos, df):
    print("\n📋 Equipos formados:")
    for i, equipo in enumerate(equipos):
        print(f"\n🔹 Equipo {i+1}:")
        for idx in equipo:
            fila = df.loc[idx]
            print(f"  {fila['StudentID']} - GPA: {fila['GPA']} - Skill: {fila['Skill']}")

# Ejecutar
df = cargar_datos()
equipos, aptitud = hill_climbing(df)
mostrar_resultados(equipos, df)
print(f"\n✅ Aptitud final: {aptitud:.4f}")
