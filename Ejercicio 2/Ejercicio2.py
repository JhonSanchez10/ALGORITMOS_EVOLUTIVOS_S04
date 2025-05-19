import pandas as pd
import random
import os

# Paso 1: Leer archivo CSV desde la misma carpeta del script
def cargar_datos():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ruta_csv = os.path.join(script_dir, "disponibilidad.csv")
    try:
        df = pd.read_csv(ruta_csv)
        print("✅ Archivo 'disponibilidad.csv' cargado correctamente desde:\n", ruta_csv)
        return df
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{ruta_csv}'. Verifica que exista y esté en la misma carpeta que el script.")
        exit()

# Paso 2: Cargar la disponibilidad de los mentores
df = cargar_datos()

# Paso 3: Preparar datos
horarios = df.columns[1:]  # Slot1 a Slot10
mentores = df['MentorID'].tolist()
disponibilidad = df[horarios].values  # matriz 20x10

# Paso 4: Calcular choques
def calcular_choques(asignacion):
    contador_slots = [0] * len(horarios)
    for slot_inicio in asignacion:
        if slot_inicio != -1:
            contador_slots[slot_inicio] += 1
            contador_slots[slot_inicio + 1] += 1
    choques = sum(1 for count in contador_slots if count > 1)
    return choques

# Paso 5: Generar solución aleatoria válida
def generar_solucion_valida():
    asignacion = []
    for i in range(len(mentores)):
        posibles = []
        for j in range(len(horarios) - 1):
            if disponibilidad[i][j] == 1 and disponibilidad[i][j + 1] == 1:
                posibles.append(j)
        if posibles:
            asignacion.append(random.choice(posibles))
        else:
            asignacion.append(-1)  # no se encontró bloque válido
    return asignacion

# Paso 6: Vecindario
def generar_vecino(solucion_actual):
    nuevo = solucion_actual[:]
    mentor = random.randint(0, len(mentores) - 1)
    posibles = []
    for j in range(len(horarios) - 1):
        if disponibilidad[mentor][j] == 1 and disponibilidad[mentor][j + 1] == 1:
            posibles.append(j)
    if posibles:
        nuevo[mentor] = random.choice(posibles)
    return nuevo

# Paso 7: Algoritmo Hill Climbing
def hill_climbing(max_iter=1000):
    actual = generar_solucion_valida()
    if -1 in actual:
        print("⚠️ Hay mentores sin bloques disponibles de 2h seguidas. Revisa la disponibilidad.")
        return actual, calcular_choques(actual)

    mejor = actual
    costo_mejor = calcular_choques(mejor)

    for _ in range(max_iter):
        vecino = generar_vecino(mejor)
        costo_vecino = calcular_choques(vecino)

        if costo_vecino < costo_mejor:
            mejor = vecino
            costo_mejor = costo_vecino

        if costo_mejor == 0:
            break

    return mejor, costo_mejor

# Paso 8: Ejecutar y mostrar resultados
asignacion_final, choques_finales = hill_climbing()

print("\n🧠 Asignación final de bloques de 2h por mentor (slot inicial):")
for i, slot in enumerate(asignacion_final):
    if slot != -1:
        print(f"Mentor {mentores[i]}: Slot {slot + 1} y {slot + 2}")
    else:
        print(f"Mentor {mentores[i]}: ❌ No se pudo asignar bloque válido")

print(f"\n🔧 Choques totales: {choques_finales}")
