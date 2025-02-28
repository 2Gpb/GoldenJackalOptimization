import numpy as np
import math


class GoldenJackalOptimization:
    def __init__(self, pop_size=20, dim=10, bounds=(-10, 10), max_iter=100):
        self.pop_size = pop_size
        self.dim = dim
        self.bounds = bounds
        self.max_iter = max_iter
        self.population = self.initialize_population()
        self.fitness = np.array([self.fitness_function(ind) for ind in self.population])
        self.male_jackal = self.population[np.argmin(self.fitness)]
        self.female_jackal = self.population[np.argsort(self.fitness)[1]]

    @staticmethod
    def fitness_function(x):
        return np.sum(x ** 2)

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))

    @staticmethod
    def update_position(male_jackal, female_jackal, prey, evading_energy, levy_factor):
        center_of_leaders = (male_jackal + female_jackal) / 2
        return center_of_leaders - evading_energy * np.abs(levy_factor * prey - center_of_leaders)

    @staticmethod
    def levy_flight(beta=1.5, size=1):
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        return u / np.abs(v) ** (1 / beta)

    def optimize(self):
        for t in range(self.max_iter):
            evading_energy = 2 * np.random.rand() - 1
            levy_factor = self.levy_flight(beta=1.5, size=self.dim)

            for i in range(self.pop_size):
                prey = self.population[i]
                new_position = self.update_position(self.male_jackal, self.female_jackal, prey, evading_energy,
                                                    levy_factor)

                new_position = np.clip(new_position, self.bounds[0], self.bounds[1])
                new_fitness = self.fitness_function(new_position)

                if new_fitness < self.fitness[i]:
                    self.population[i] = new_position
                    self.fitness[i] = new_fitness

            self.male_jackal = self.population[np.argmin(self.fitness)]
            self.female_jackal = self.population[np.argsort(self.fitness)[1]]

        return self.male_jackal, self.fitness_function(self.male_jackal)
