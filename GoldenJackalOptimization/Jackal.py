import numpy as np


class Jackal:
    def __init__(self, dim, bounds):
        self.dim = dim
        self.bounds = bounds
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.fitness = self.evaluate()

    def evaluate(self):
        return np.sum(self.position ** 2)

    def update_position(self, best_male, best_female, evading_energy, levy_factor):
        center_of_leaders = (best_male.position + best_female.position) / 2
        new_position = center_of_leaders - evading_energy * np.abs(levy_factor * self.position - center_of_leaders)
        new_position = np.clip(new_position, self.bounds[0], self.bounds[1])
        self.position = new_position
        self.fitness = self.evaluate()
