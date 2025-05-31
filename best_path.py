import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

# Constants
FIELD_SIZE = (20, 20)
NUM_OBSTACLES = 0
POPULATION_SIZE = 30
OBSTACLE_PENALTY = 10000
REVISIT_PENALTY = 1000
NUM_GENERATIONS = 5000
MUTATION_RATE = 0.5
MAX_PATH_LENGTH = int(1 * FIELD_SIZE[0] * FIELD_SIZE[1])
ZONES = 10  # 2x2 = 4 zones

# Field with obstacles
class Field:
    def __init__(self, width, height, num_obstacles):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))
        self.place_obstacles(num_obstacles)

    def place_obstacles(self, num):
        count = 0
        while count < num:
            x = random.randint(1, self.width - 2)
            y = random.randint(1, self.height - 2)
            if self.grid[y][x] == 0:
                self.grid[y-1:y+2, x-1:x+2] = -1  # 3x3 block
                count += 1

    def is_valid(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height and self.grid[y][x] != -1

    def zone_index(self, x, y):
        return int(y / (self.height / ZONES)) * ZONES + int(x / (self.width / ZONES))

# Smart path generator
def generate_valid_path(field, start=(0, 0), max_length=MAX_PATH_LENGTH):
    x, y = start
    path = [(x, y)]
    visited = set(path)

    for _ in range(max_length - 1):
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        random.shuffle(neighbors)
        best = None
        for nx, ny in neighbors:
            if field.is_valid(nx, ny) and (nx, ny) not in visited:
                best = (nx, ny)
                break
        if best:
            x, y = best
            path.append((x, y))
            visited.add((x, y))
        else:
            break
    return path

class DronePath:
    def __init__(self, field, path=None):
        self.field = field
        self.path = path if path else generate_valid_path(field)
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        visited_once = set()
        visited_more_than_once = set()
        zones = set()
        obstacle_hits = 0

        for x, y in self.path:
            if not self.field.is_valid(x, y):
                obstacle_hits += 1
            if (x, y) in visited_once:
                visited_more_than_once.add((x, y))
            else:
                visited_once.add((x, y))
                zones.add(self.field.zone_index(x, y))

        coverage = len(visited_once) / (self.field.width * self.field.height)
        zone_bonus = len(zones) / (ZONES * ZONES)

        # Penalizaciones: cada obstáculo resta fuerte, y cada revisita más leve
        penalty = REVISIT_PENALTY * len(visited_more_than_once) + OBSTACLE_PENALTY * obstacle_hits

        # Fitness final
        return (coverage + 0.2 * zone_bonus) - penalty * 0.001

    def mutate(self):
        if len(self.path) < 10:
            return DronePath(self.field, generate_valid_path(self.field))

        if random.random() < 0.1:  # fuerte 10% de las veces
            return DronePath(self.field, generate_valid_path(self.field))  # reemplazo total

        idx = random.randint(5, len(self.path) - 5)
        head = self.path[:idx]
        tail_start = head[-1]
        tail = generate_valid_path(self.field, tail_start, MAX_PATH_LENGTH - len(head))
        return DronePath(self.field, head + tail)

    def crossover(self, other):
        idx = random.randint(1, min(len(self.path), len(other.path)) - 2)
        base = [p for p in self.path[:idx]]
        seen = set(base)
        for p in other.path:
            if p not in seen and self.field.is_valid(*p):
                base.append(p)
                seen.add(p)
            if len(base) >= MAX_PATH_LENGTH:
                break
        return DronePath(self.field, base)

class GeneticAlgorithm:
    def __init__(self, field):
        self.field = field
        self.population = [DronePath(field) for _ in range(POPULATION_SIZE)]

    def evolve(self):
        for gen in range(NUM_GENERATIONS):
            value = gen % 1000
            self.population.sort(key=lambda p: p.fitness, reverse=True)
            best = self.population[0]
            coverage = len(set(best.path)) / (self.field.width * self.field.height)
            print(f"Gen {gen+1}: Fitness={best.fitness:.4f}, Coverage={coverage*100:.1f}%")
            if (value == 0):
                self.enhanced_plot_path(best.path, f"Gen {gen+1}")
            new_population = self.population[:2]
            while len(new_population) < POPULATION_SIZE:
                p1, p2 = random.choices(self.population[:10], k=2)
                child = p1.crossover(p2).mutate()
                new_population.append(child)
            self.population = new_population




    def enhanced_plot_path(self, path, title="Drone Path", value=0):
        grid = np.zeros_like(self.field.grid)
        
        # Mark obstacles (white)
        grid[self.field.grid == -1] = 1.0

        # Mark visited points
        for x, y in path:
            if 0 <= y < self.field.height and 0 <= x < self.field.width:
                if self.field.is_valid(x, y):
                    grid[y][x] = 0.5
                else:
                    grid[y][x] = 0.8  # stepped on obstacle

        plt.imshow(grid, cmap='gray', origin='upper')

        x_coords, y_coords = zip(*path)
        plt.plot(x_coords, y_coords, color='red', linewidth=1)
        plt.scatter(x_coords[0], y_coords[0], color='green', label='Start', zorder=5)
        plt.scatter(x_coords[-1], y_coords[-1], color='blue', label='End', zorder=5)

        plt.title(title)
        plt.legend()
        plt.grid(True, linewidth=0.2, color='gray')
        
        plt.show()

# Run
field = Field(*FIELD_SIZE, NUM_OBSTACLES)
ga = GeneticAlgorithm(field)
ga.evolve()
