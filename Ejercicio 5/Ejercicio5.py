import pandas as pd
import numpy as np
import random
import os

# Paso 1: Leer archivo CSV desde la misma carpeta del script
def cargar_datos():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ruta_csv = os.path.join(script_dir, "tesistas.csv")
    try:
        df = pd.read_csv(ruta_csv)
        print("✅ Archivo cargado correctamente desde:\n", ruta_csv)
        return df
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{ruta_csv}'. Verifica que exista.")
        exit()

# Paso 2: Asignación secuencial inicial (heurística)
def asignacion_inicial(df):
    franjas = [f"F{i+1}" for i in range(6)]
    solucion = {}
    sala_actual = 0
    franja_actual = 0
    for _, row in df.iterrows():
        while not row[franjas[franja_actual]]:
            franja_actual = (franja_actual + 1) % 6
            sala_actual = (sala_actual + (franja_actual == 0)) % 6
        solucion[row['TesistaID']] = (sala_actual, franjas[franja_actual])
        franja_actual = (franja_actual + 1) % 6
        sala_actual = (sala_actual + (franja_actual == 0)) % 6
    return solucion

# Paso 3: Función de evaluación
def evaluar(solucion):
    uso_salas = {}
    solapamientos = 0
    franjas_por_sala = {sala: [] for sala in range(6)}
    
    for tesista, (sala, franja) in solucion.items():
        clave = (sala, franja)
        if clave in uso_salas:
            solapamientos += 1
        else:
            uso_salas[clave] = tesista
        franjas_por_sala[sala].append(int(franja[1]))  # Extraer número de franja

    huecos = 0
    for franjas in franjas_por_sala.values():
        if franjas:
            franjas.sort()
            # Medir huecos entre franjas usadas
            for i in range(len(franjas) - 1):
                diff = franjas[i+1] - franjas[i]
                if diff > 1:
                    huecos += diff - 1
            # Penalización si excede 4 franjas usadas
            if len(franjas) > 4:
                huecos += 1000

    return -(solapamientos + huecos), solapamientos, huecos

# Paso 4: Generar vecino
def generar_vecino(solucion, df):
    vecino = solucion.copy()
    tesista_random = random.choice(list(solucion.keys()))
    row = df[df['TesistaID'] == tesista_random].iloc[0]
    franjas = [f"F{j+1}" for j in range(6)]
    opciones_validas = [(sala, franja) for sala in range(6) for franja in franjas if row[franja] == 1]
    if opciones_validas:
        vecino[tesista_random] = random.choice(opciones_validas)
    return vecino

# Paso 5: Hill Climbing
def hill_climbing(df, iteraciones=1000):
    solucion_actual = asignacion_inicial(df)
    mejor_score, _, _ = evaluar(solucion_actual)
    for _ in range(iteraciones):
        vecino = generar_vecino(solucion_actual, df)
        score, _, _ = evaluar(vecino)
        if score > mejor_score:
            solucion_actual = vecino
            mejor_score = score
    return solucion_actual, evaluar(solucion_actual)

# Paso 6: Guardar calendario
def guardar_calendario(solucion):
    df = pd.DataFrame([
        {"TesistaID": tesista, "Sala": sala + 1, "Franja": franja}
        for tesista, (sala, franja) in solucion.items()
    ])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ruta_salida = os.path.join(script_dir, "calendario_tesis.csv")
    df.to_csv(ruta_salida, index=False)
    print("📁 Calendario guardado como:", ruta_salida)
    print(df.sort_values(by=["Franja", "Sala"]).to_string(index=False))

# Ejecutar todo
if __name__ == "__main__":
    df_tesistas = cargar_datos()
    solucion_final, (score, solapamientos, huecos) = hill_climbing(df_tesistas, iteraciones=1000)
    print(f"\n✅ Métricas Finales:")
    print(f"   Solapamientos: {solapamientos}")
    print(f"   Huecos: {huecos}")
    print(f"   Score final: {score}")
    guardar_calendario(solucion_final)
