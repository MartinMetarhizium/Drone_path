import numpy as np
import random
import matplotlib.pyplot as plt

FIELD_SIZE = (100, 100)
WORKERS = 16
POPULATION_SIZE = 30
NUM_GENERATIONS = 100

def divide_field(field_size, workers):
    width, height = field_size
    num_divisions = int(np.sqrt(workers))
    block_w = width // num_divisions
    block_h = height // num_divisions
    zones = []
    for i in range(num_divisions):
        for j in range(num_divisions):
            x_start = j * block_w
            x_end = (j+1) * block_w
            y_start = i * block_h
            y_end = (i+1) * block_h
            zones.append((x_start, x_end, y_start, y_end))
    return zones

def serpentine_path(x_start, x_end, y_start, y_end):
    path = []
    for y in range(y_start, y_end):
        if (y - y_start) % 2 == 0:
            for x in range(x_start, x_end):
                path.append((x, y))
        else:
            for x in reversed(range(x_start, x_end)):
                path.append((x, y))
    return path

def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def merge_paths(worker_paths, order):
    merged_path = []
    for idx in order:
        path = worker_paths[idx]
        if not merged_path:
            merged_path += path
        else:
            last = merged_path[-1]
            first = path[0]
            intermediate = []
            x, y = last
            fx, fy = first
            while x != fx:
                x += 1 if fx > x else -1
                intermediate.append((x, y))
            while y != fy:
                y += 1 if fy > y else -1
                intermediate.append((x, y))
            merged_path += intermediate + path
    # Return to start
    last = merged_path[-1]
    return_to_start = []
    x, y = last
    while x != 0:
        x += -1
        return_to_start.append((x, y))
    while y != 0:
        y += -1
        return_to_start.append((x, y))
    merged_path += return_to_start
    return merged_path

class WorkerOrder:
    def __init__(self, worker_paths, order=None):
        self.worker_paths = worker_paths
        self.num_workers = len(worker_paths)
        self.order = order if order else random.sample(range(self.num_workers), self.num_workers)
        self.path = merge_paths(worker_paths, self.order)
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        return len(self.path)  # minimize total path length

    def mutate(self):
        a, b = random.sample(range(self.num_workers), 2)
        new_order = self.order.copy()
        new_order[a], new_order[b] = new_order[b], new_order[a]
        return WorkerOrder(self.worker_paths, new_order)

    def crossover(self, other):
        # Order crossover (OX)
        size = self.num_workers
        a, b = sorted(random.sample(range(size), 2))
        child_order = [None] * size
        child_order[a:b] = self.order[a:b]
        fill = [w for w in other.order if w not in child_order]
        j = 0
        for i in range(size):
            if child_order[i] is None:
                child_order[i] = fill[j]
                j += 1
        return WorkerOrder(self.worker_paths, child_order)

def plot_path(path, field_size, title="Path"):
    width, height = field_size
    grid = np.zeros((height, width))
    for i, (x, y) in enumerate(path):
        grid[y][x] = i + 1

    plt.imshow(grid, cmap='Blues', origin='upper')
    x_coords, y_coords = zip(*path)
    plt.plot(x_coords, y_coords, color='red')
    plt.scatter(x_coords[0], y_coords[0], color='green', label='Start')
    plt.scatter(x_coords[-1], y_coords[-1], color='blue', label='End')
    plt.title(title)
    plt.legend()
    plt.show()

def evolve_worker_order(field_size, workers):
    zones = divide_field(field_size, workers)
    worker_paths = [serpentine_path(*zone) for zone in zones]

    population = [WorkerOrder(worker_paths) for _ in range(POPULATION_SIZE)]

    best_fitness_per_generation = []  # <-- Track best fitness

    for gen in range(NUM_GENERATIONS):
        population.sort(key=lambda w: w.fitness)
        best = population[0]
        best_fitness_per_generation.append(best.fitness)  # <-- Save it
        print(f"Gen {gen+1}: Path Length = {best.fitness}")
        
        if (gen + 1) % 20 == 0:
            plot_path(best.path, field_size, title=f"Generation {gen+1}")
        
        survivors = population[:10]
        new_population = survivors.copy()
        while len(new_population) < POPULATION_SIZE:
            p1, p2 = random.sample(survivors, 2)
            child = p1.crossover(p2)
            if random.random() < 0.3:
                child = child.mutate()
            new_population.append(child)
        population = new_population

    # Plot final path
    best = population[0]
    plot_path(best.path, field_size, title="Final Path")
    print(f"Final Path Length: {best.fitness}")

    # Plot the progress chart
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, NUM_GENERATIONS + 1), best_fitness_per_generation, marker='o')
    plt.xlabel('Generation')
    plt.ylabel('Best Path Length')
    plt.title('Progress of Best Fitness Over Generations')
    plt.grid(True)
    plt.show()

# Run
evolve_worker_order(FIELD_SIZE, WORKERS)


def theoretical_min_path(field_size, workers):
    width, height = field_size
    total_cells = width * height
    worker_blocks = int(workers)
    cells_per_worker = total_cells // worker_blocks
    steps_per_worker = cells_per_worker - 1
    total_worker_steps = steps_per_worker * worker_blocks
    minimal_worker_connections = worker_blocks - 1  
    return_to_start = (width - 1) + (height - 1)  

    minimal_total_steps = total_worker_steps + minimal_worker_connections + return_to_start
    return minimal_total_steps

# Example
print(theoretical_min_path(FIELD_SIZE, WORKERS))
