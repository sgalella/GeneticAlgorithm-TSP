import numpy as np
from tqdm import tqdm


class GeneticAlgorithm:
    """
    Genetic algorithm for TSP.
    """
    def __init__(self, loc_cities, num_iterations, num_selected, num_children, num_immigrants, num_mutated):
        """
        Initializes the algorithm.
        """
        self.loc_cities = loc_cities
        self.N = len(self.loc_cities)
        self.num_iterations = num_iterations
        self.num_selected = num_selected
        self.num_children = num_children
        self.num_immigrants = num_immigrants
        self.num_mutated = num_mutated
        self.population_size = self.num_selected + self.num_children + self.num_immigrants + self.num_mutated

    def __repr__(self):
        """
        Visualizes algorithm parameters when printing.
        """
        return (f"Iterations: {self.num_iterations}\n"
                f"Population size: {self.population_size}\n"
                f"Num selected: {self.num_selected}\n"
                f"Num children: {self.num_children}\n"
                f"Num immigrants: {self.num_immigrants}\n"
                f"Num mutations: {self.num_mutated}\n")

    def random_population(self, num_individuals):
        """
        Generates random population of individuals

        Args:
            num_individuals (int): Number of individuals to be created.

        Returns:
            population (np.array): Population containg the different individuals.
        """
        # Initialize population
        population = np.array([np.zeros([self.N, ], dtype=int) for _ in range(num_individuals)])

        # Apply different inhibition to each individual
        for individual in range(num_individuals):
            population[individual] = np.random.permutation(self.N, )

        return population

    def compute_fitness(self, population):
        """
        Computes the fitness for each individual by calculating the distances of the cities.

        Args:
            population (np.array): Population containg the different individuals.

        Returns:
            fitness_population (np.array): Fitness of the population.
        """
        fitness_population = np.zeros([self.population_size, 1])
        for idx, individual in enumerate(population):

            if np.unique(individual).shape[0] == self.N:
                fitness_individual = 0
                for i in range(1, self.N - 1):
                    fitness_individual += ((self.loc_cities[individual[i]][0] - self.loc_cities[individual[i - 1]][0])**2
                                           + (self.loc_cities[individual[i]][1] - self.loc_cities[individual[i - 1]][1])**2)**0.5
                fitness_individual += ((self.loc_cities[individual[-1]][0] - self.loc_cities[individual[0]][0])**2
                                       + (self.loc_cities[individual[-1]][1] - self.loc_cities[individual[0]][1])**2)**0.5
            else:
                fitness_individual = np.inf
            fitness_population[idx] = 4 * max(max(self.loc_cities)) / fitness_individual

        return fitness_population.flatten()

    def recombination(self, parent1, parent2):
        """
        Creates a new individual by recombinating two parents.

        Args:
            parent1 (np.array): First parent.
            parent2 (np.array): Second parent.

        Returns:
            new_individual: Recombinated individual.
        """
        random_pos = np.random.randint(self.N)
        new_individual = np.concatenate((parent1[:random_pos], parent2[random_pos:]))
        return new_individual

    def mutation(self, individual, num_mutations):
        """
        Mutates indidividual by changing the position of different cities.

        Args:
            individual (np.array): Individual to be mutated.
            num_mutations (int): Number of cities to be switched.

        Returns:
            mutated_individual (np.array): New mutated individual.
        """
        mutated_individual = np.copy(individual)
        for mutation in range(num_mutations):
            city1, city2 = np.random.choice(self.N, 2, replace=False)
            mutated_individual[city1], mutated_individual[city2] = mutated_individual[city2], mutated_individual[city1]

        return mutated_individual

    def next_population(self, population, fitness):
        """
        Generates the population for the next iteration.

        Args:
            population (np.array): Population containg the different individuals.
            fitness (np.array): Fitness of the population

        Returns:
            next_populatioin (np.array): Population for the next iteration.
        """
        # Initialize next population
        next_population = np.array([np.zeros([self.N, ], dtype=int) for _ in range(self.population_size)])

        # Select best individuals
        best_individuals = np.argsort(fitness)[::-1]

        for individual in range(self.num_selected):
            next_population[individual] = population[best_individuals[individual]]
            # next_population[individual] = population[np.where(best_individuals == individual)]

        # Recombinate best individuals
        for individual in range(self.num_selected, self.num_selected + self.num_children):
            pair = np.random.choice(self.num_selected, size=2, replace=False)
            next_population[individual] = self.recombination(next_population[pair[0]], next_population[pair[1]])

        # Add immigration
        for individual in range(self.num_selected + self.num_children,
                                self.num_selected + self.num_children + self.num_immigrants):
            next_population[individual] = self.random_population(1)

        # Add mutation
        for idx_best, individual in enumerate(range(self.num_selected + self.num_children + self.num_immigrants,
                                                    self.population_size)):
            next_population[individual] = self.mutation(next_population[idx_best], 1)

        return next_population

    def run(self):
        """
        Runs the algorithm.

        Args:
            iter_info (int, optional): Frequency of information in screen. Defaults to 10.

        Returns:
            best_individual (np.array): Returns the best individual (path) generated.
        """
        # Initialize first population
        population = self.random_population(self.population_size)

        # Initialize fitness variables
        mean_fitness = []
        max_fitness = []

        for iteration in tqdm(range(self.num_iterations), ncols=75):
            fitness = self.compute_fitness(population)
            population = self.next_population(population, fitness)
            max_fitness.append(np.max(fitness))
            mean_fitness.append(np.mean(fitness))

        return population[0], max_fitness, mean_fitness
