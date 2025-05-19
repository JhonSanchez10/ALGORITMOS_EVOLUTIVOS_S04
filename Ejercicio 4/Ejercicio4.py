import pandas as pd
import random
import os

# Paso 1: Cargar archivo CSV desde la misma carpeta
def cargar_datos():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ruta_csv = os.path.join(script_dir, "proyectos.csv")
    try:
        df = pd.read_csv(ruta_csv)
        print("✅ Archivo cargado correctamente desde:\n", ruta_csv)
        return df
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{ruta_csv}'. Verifica que exista y esté bien nombrado.")
        exit()

# Paso 2: Función de aptitud (beneficio si cumple presupuesto, si no → -inf)
def fitness(bitstring, costos, beneficios, presupuesto=10000):
    total_costo = sum(c for c, s in zip(costos, bitstring) if s == 1)
    if total_costo > presupuesto:
        return float('-inf')
    return sum(b for b, s in zip(beneficios, bitstring) if s == 1)

# Paso 3: Generar vecino (voltear un bit aleatorio)
def generar_vecino(bitstring):
    vecino = bitstring[:]
    i = random.randint(0, len(vecino) - 1)
    vecino[i] = 1 - vecino[i]
    return vecino

# Paso 4: Algoritmo Hill Climbing
def hill_climbing(costos, beneficios, presupuesto=10000, iteraciones=1000):
    actual = [random.randint(0, 1) for _ in range(len(costos))]
    mejor = actual
    mejor_fit = fitness(mejor, costos, beneficios, presupuesto)

    for _ in range(iteraciones):
        vecino = generar_vecino(mejor)
        vecino_fit = fitness(vecino, costos, beneficios, presupuesto)
        if vecino_fit > mejor_fit:
            mejor = vecino
            mejor_fit = vecino_fit
    return mejor, mejor_fit

def main():
    df = cargar_datos()
    proyectos = df['ProjectID'].tolist()
    costos = df['Cost_Soles'].tolist()
    beneficios = df['Benefit_Soles'].tolist()

    solucion, beneficio_total = hill_climbing(costos, beneficios)

    seleccionados = [p for p, s in zip(proyectos, solucion) if s == 1]
    costo_total = sum(c for c, s in zip(costos, solucion) if s == 1)

    print("\n🎯 Proyectos seleccionados:")
    for p in seleccionados:
        print(f"  - {p}")
    print(f"\n📊 Costo total:     S/ {costo_total}")
    print(f"🏆 Beneficio total: S/ {beneficio_total}")

if __name__ == "__main__":
    main()
