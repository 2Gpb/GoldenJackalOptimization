from GJO import *

if __name__ == "__main__":
    gjo = GoldenJackalOptimization()
    best_solution, best_fitness = gjo.optimize()
    print("Best solution found:", best_solution)
    print("Target function value:", best_fitness)
