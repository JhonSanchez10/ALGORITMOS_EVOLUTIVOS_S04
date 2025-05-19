import pandas as pd
import numpy as np
import os

# Paso 1: Leer archivo CSV desde la misma carpeta del script
def cargar_datos():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ruta_csv = os.path.join(script_dir, "notas.csv")
    try:
        df = pd.read_csv(ruta_csv)
        print("✅ Archivo cargado correctamente desde:\n", ruta_csv)
        return df
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{ruta_csv}'. Verifica que exista y la ruta esté bien escrita.")
        exit()

# Paso 2: Función de aptitud
def calcular_aptitud(df, offset):
    df_adjusted = df.copy()

    # Aplicar offset solo a las columnas numéricas (evita 'StudentID')
    cols_numericas = df_adjusted.select_dtypes(include=[np.number]).columns
    df_adjusted[cols_numericas] = df_adjusted[cols_numericas] + offset
    df_adjusted[cols_numericas] = df_adjusted[cols_numericas].clip(lower=0, upper=20)

    promedio_general = df_adjusted[cols_numericas].mean().mean()
    total_alumnos = len(df_adjusted)

    promedios_alumno = df_adjusted[cols_numericas].mean(axis=1)
    aprobados = (promedios_alumno >= 11).sum()
    porcentaje_aprobados = aprobados / total_alumnos

    # Penalización si promedio > 14
    if promedio_general > 14:
        return porcentaje_aprobados - (promedio_general - 14) * 0.1
    else:
        return porcentaje_aprobados

# Paso 3: Hill Climbing
def hill_climbing(df):
    mejor_offset = 0
    mejor_aptitud = calcular_aptitud(df, mejor_offset)
    cambios = [-0.5, 0.5]
    iteraciones = 0

    while True:
        mejoras = False
        for cambio in cambios:
            nuevo_offset = mejor_offset + cambio
            if -5 <= nuevo_offset <= 5:
                nueva_aptitud = calcular_aptitud(df, nuevo_offset)
                if nueva_aptitud > mejor_aptitud:
                    mejor_offset = nuevo_offset
                    mejor_aptitud = nueva_aptitud
                    mejoras = True
                    break
        iteraciones += 1
        if not mejoras or iteraciones > 100:
            break

    return mejor_offset, mejor_aptitud

# Paso 4: Aplicar y mostrar resultados
def aplicar_offset(df, offset):
    df_final = df.copy()
    cols_numericas = df_final.select_dtypes(include=[np.number]).columns

    df_final[cols_numericas] = df_final[cols_numericas] + offset
    df_final[cols_numericas] = df_final[cols_numericas].clip(lower=0, upper=20)

    print("\n🎯 Offset óptimo encontrado:", offset)
    print("📊 Nuevo promedio general:", round(df_final[cols_numericas].mean().mean(), 2))

    promedios_alumno = df_final[cols_numericas].mean(axis=1)
    porcentaje_aprobados = (promedios_alumno >= 11).sum() / len(df_final)
    print("✅ Porcentaje de aprobados:", round(porcentaje_aprobados * 100, 2), "%")
    print("\n📝 Distribución de notas ajustadas:")
    print(df_final.head())

    return df_final

# Función principal
def main():
    df = cargar_datos()
    print("📄 Primeras filas del archivo CSV:")
    print(df.head())

    mejor_offset, _ = hill_climbing(df)
    df_ajustado = aplicar_offset(df, mejor_offset)

    # Guardar nuevo CSV con notas ajustadas
    ruta_salida = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notas_ajustadas.csv")
    df_ajustado.to_csv(ruta_salida, index=False)
    print("\n💾 Archivo 'notas_ajustadas.csv' guardado exitosamente.")

# Ejecutar
if __name__ == "__main__":
    main()
