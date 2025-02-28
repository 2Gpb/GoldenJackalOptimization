import math
from Jackal import *


class GoldenJackalOptimization:
    def __init__(self, fitness_function, pop_size=30, dim=10, bounds=(-5, 5), max_iter=400):
        self.pop_size = pop_size
        self.dim = dim
        self.bounds = bounds
        self.max_iter = max_iter
        self.__fitness_function = fitness_function
        self.population = [Jackal(dim, bounds, fitness_function) for _ in range(pop_size)]
        self.male_jackal = min(self.population, key=lambda j: j.fitness)
        self.female_jackal = sorted(self.population, key=lambda j: j.fitness)[1]

    @staticmethod
    def levy_flight(beta=1.5, size=1):
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        return u / np.abs(v) ** (1 / beta)

    def optimize(self):
        for _ in range(self.max_iter):
            evading_energy = 2 * np.random.rand() - 1
            levy_factor = self.levy_flight(beta=1.5, size=self.dim)

            for jackal in self.population:
                jackal.update_position(self.male_jackal, self.female_jackal, evading_energy, levy_factor)

            self.male_jackal = min(self.population, key=lambda j: j.fitness)
            self.female_jackal = sorted(self.population, key=lambda j: j.fitness)[1]

        return self.male_jackal.position, self.male_jackal.fitness
