import numpy as np
import matplotlib.pyplot as plt
from genetic_algorithm import GeneticAlgorithm


# Set rng for reproducibility
seed = 1234
np.random.seed(seed)

# Initialize cities
num_cities = 15
map_size = 100
loc_cities = []

while len(loc_cities) < num_cities:
    new_city = tuple(np.random.choice((map_size), size=2))
    if new_city not in loc_cities:
        loc_cities.append(new_city)

# Plot different cities
fig = plt.figure()
for i, city in enumerate(loc_cities):
    x, y = city
    plt.plot(x, y, 'r.', markersize=10)
    plt.text(x + 1, y + 1, i)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Cities location')
plt.xlim([0, map_size])
plt.ylim([0, map_size])
plt.grid(alpha=0.3)
plt.savefig('images/cities.png')
plt.draw()

# Algorithm settings
num_iterations = 200
num_selected = 100
num_children = 50
num_immigrants = 25
num_mutated = 25

# Initialize algorithm
genetic = GeneticAlgorithm(loc_cities, num_selected, num_children, num_immigrants, num_mutated)
best_path, max_fitness, mean_fitness = genetic.run(num_iterations)

# Plot algorithm convergence
fig = plt.figure()
plt.plot(range(len(mean_fitness)), mean_fitness, 'b')
plt.plot(range(len(max_fitness)), max_fitness, 'r--')
plt.legend(("mean fitness", "max fitness"))
plt.xlabel('iterations')
plt.ylabel('fitness')
plt.title('Fitness convergence')
plt.grid(alpha=0.3)
plt.savefig('images/convergence.png')
plt.draw()

# Plot best path
fig = plt.figure()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Best path')
plt.xlim([0, map_size])
plt.ylim([0, map_size])
plt.grid(alpha=0.3)

for i in range(1, len(best_path)):
    x0, y0 = loc_cities[best_path[i - 1]]
    x1, y1 = loc_cities[best_path[i]]
    plt.text(x0 + 1, y0 + 1, best_path[i - 1])
    plt.plot([x0, x1], [y0, y1], 'ro-')
plt.text(x1, y1, best_path[-1])

# Complete the tour
x0, y0 = loc_cities[best_path[-1]]
x1, y1 = loc_cities[best_path[0]]
plt.plot([x0, x1], [y0, y1], 'ro-')
plt.savefig('images/best_path.png')
plt.show()
