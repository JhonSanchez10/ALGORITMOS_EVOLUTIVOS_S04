import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from deap import base, creator, tools
import random
import matplotlib.pyplot as plt

def cargar_datos():
    script_dir = os.path.dirname(os.path.abspath(__file__))  # directorio del script
    ruta_csv = os.path.join(script_dir, "HousePricesUNS.csv")  # nombre del archivo csv
    try:
        df = pd.read_csv(ruta_csv)
        print("✅ Datos cargados desde:", ruta_csv)
        X = df[['Rooms', 'Area_m2']].values
        y = df['Price_Soles'].values
        return X, y
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo 'HousePricesUNS.csv' en:", ruta_csv)
        exit()

def evaluar_individuo(alpha, X_train, y_train, X_val, y_val):
    alpha_val = max(alpha[0], 1e-5)  # extraer valor float del individuo
    model = Ridge(alpha=alpha_val)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return (rmse,)

def mut_gauss_small(individual):
    sigma = 0.1
    individual[0] += random.gauss(0, sigma)
    # Limitar alpha entre 1e-5 y 10
    individual[0] = min(max(individual[0], 1e-5), 10.0)
    return (individual,)

def main():
    X, y = cargar_datos()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Cada individuo tiene un solo valor alpha entre 1e-5 y 10
    toolbox.register("attr_alpha", random.uniform, 1e-5, 10.0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_alpha, 1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluar_individuo, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    toolbox.register("mutate", mut_gauss_small)

    pop = toolbox.population(n=20)

    # Evaluar población inicial
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    max_gens = 100
    best_rmse_progress = []

    for gen in range(max_gens):
        new_pop = []

        for ind in pop:
            mutant = toolbox.clone(ind)
            toolbox.mutate(mutant)
            mutant.fitness.values = toolbox.evaluate(mutant)

            # Selección greedy: elegir al que tenga menor RMSE
            if mutant.fitness.values[0] < ind.fitness.values[0]:
                new_pop.append(mutant)
            else:
                new_pop.append(ind)

        pop = new_pop
        best_ind = tools.selBest(pop, 1)[0]
        best_rmse_progress.append(best_ind.fitness.values[0])
        print(f"Gen {gen+1}: Mejor RMSE = {best_ind.fitness.values[0]:.4f} con alpha={best_ind[0]:.5f}")

    best_ind = tools.selBest(pop, 1)[0]
    print(f"\nAlpha óptimo: {best_ind[0]:.5f} con RMSE: {best_ind.fitness.values[0]:.4f}")

    plt.plot(best_rmse_progress, label='RMSE mínimo por generación')
    plt.xlabel('Generación')
    plt.ylabel('RMSE')
    plt.title('Curva de convergencia Hill Climbing en Ridge Regression')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
