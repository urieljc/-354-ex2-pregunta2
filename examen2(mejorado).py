# -*- coding: utf-8 -*-
"""
Created on Sun May 30 15:39:20 2021

@author: BazanJuanCarlos
"""

import array
import random

import pandas as pd
import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools



#datos del data sest
#lectura de datos
datos=pd.read_csv('datos2.csv',sep=';',header=0)


#distancia =datos["peso"]
#poblacion=len(datos.axes[0])

datos.columns=["A","B","C","D","E"]
matriz=datos[["A","B","C","D","E"]]
matriz=np.array(matriz)
print("________________________________________________")
print("matriz de distancia")
print(matriz)
print("________________________________________________")
#codigo del agente viajero 

distance_map = matriz
IND_SIZE = 5

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# atributo generador
toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE)

# inicializando la estructura
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#eimpleementacion del agente viajero a la matriz
def evalTSP(individual):
    distance = distance_map[individual[-1]][individual[0]]
    for gene1, gene2 in zip(individual[0:-1], individual[1:]):
        distance += distance_map[gene1][gene2]
    return distance,

toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evalTSP)

def main():
    random.seed(169)

    pop = toolbox.population(n=300)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 40, stats=stats, 
                        halloffame=hof)
    
    print("________________________________________________")
    print("individuos mutados")
    print(pop)
    print("________________________________________________")
    print("mejor recorrido ")
    print(hof)
    return pop, stats, hof

if __name__ == "__main__":
    main()
    print("________________________________________________")
    #print(main.pop())