import numpy as np
import random
import matplotlib.pyplot as plt

FIELD_SIZE = (10, 10)
POPULATION_SIZE = 50
NUM_GENERATIONS = 300
MUTATION_RATE = 0.3
MAX_PATH_LENGTH = FIELD_SIZE[0] * FIELD_SIZE[1] * 2

MOVES = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up

class DronePath:
    def __init__(self, field_size, path=None):
        self.width, self.height = field_size
        self.start = (0, 0)
        self.path = path if path else self.smart_random_path()
        self.fitness = self.calculate_fitness()

    def smart_random_path(self):
        path = [self.start]
        current_direction = random.choice(MOVES)

        for _ in range(MAX_PATH_LENGTH):
            if random.random() < 0.2:
                current_direction = random.choice(MOVES)

            x, y = path[-1]
            dx, dy = current_direction
            nx, ny = x + dx, y + dy

            if 0 <= nx < self.width and 0 <= ny < self.height:
                path.append((nx, ny))
            else:
                current_direction = random.choice(MOVES)

        path.append((0, 0))
        return path

    def calculate_fitness(self):
        visited = {}
        for (x, y) in self.path:
            visited[(x, y)] = visited.get((x, y), 0) + 1

        total_cells = self.width * self.height
        unique_visited = len(visited)
        coverage = unique_visited / total_cells

        # Penalize revisits
        revisits = sum(1 for v in visited.values() if v > 1)

        # Penalize not ending at start
        loop_penalty = 0 if self.path[-1] == self.start else 1

        # Penalize turns
        turns = 0
        for i in range(1, len(self.path) - 1):
            dx1 = self.path[i][0] - self.path[i-1][0]
            dy1 = self.path[i][1] - self.path[i-1][1]
            dx2 = self.path[i+1][0] - self.path[i][0]
            dy2 = self.path[i+1][1] - self.path[i][1]
            if (dx1, dy1) != (dx2, dy2):
                turns += 1

        steps = len(self.path)

        if coverage < 1.0:
            return coverage - revisits * 10.0  # Early: cover as much as possible but punish revisits

        return (5.0
                - (steps / total_cells)
                - (turns / 100.0)
                - loop_penalty * 5.0
                - revisits * 10.0)  # Strongly punish revisits

    def mutate(self):
        path = self.path[:-1]

        if random.random() < 0.5:
            idx = random.randint(1, len(path) - 2)
            x, y = path[idx-1]
            dx, dy = random.choice(MOVES)
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                path[idx] = (nx, ny)
        else:
            if len(path) > 2:
                idx = random.randint(1, len(path) - 2)
                del path[idx]

        path.append((0, 0))
        return DronePath((self.width, self.height), path)

    def crossover(self, other):
        idx = random.randint(1, min(len(self.path), len(other.path)) - 2)
        new_path = self.path[:idx] + other.path[idx:]
        new_path[-1] = (0, 0)
        return DronePath((self.width, self.height), new_path)

    def plot(self, title="Drone Path"):
        grid = np.zeros((self.height, self.width))
        for i, (x, y) in enumerate(self.path):
            grid[y][x] = i + 1

        plt.imshow(grid, cmap='Blues', origin='upper')
        x_coords, y_coords = zip(*self.path)
        plt.plot(x_coords, y_coords, color='red')
        plt.scatter(x_coords[0], y_coords[0], color='green', label='Start')
        plt.scatter(x_coords[-1], y_coords[-1], color='blue', label='End')
        plt.title(title)
        plt.legend()
        plt.show()


class GeneticAlgorithm:
    def __init__(self, field_size):
        self.field_size = field_size
        self.population = [DronePath(field_size) for _ in range(POPULATION_SIZE)]

    def evolve(self):
        for gen in range(NUM_GENERATIONS):
            self.population.sort(key=lambda p: p.fitness, reverse=True)
            best = self.population[0]
            unique_cells = len(set(best.path))
            print(f"Gen {gen+1}: Fitness={best.fitness:.4f}, Steps={len(best.path)}, Coverage={unique_cells / (self.field_size[0] * self.field_size[1]) * 100:.2f}% Revisits={len(best.path) - unique_cells}")
            if (gen + 1) % 50 == 0:
                best.plot(f"Generation {gen+1}")

            if best.fitness > 4.0:
                print("Optimal path found!")
                best.plot(f"Final Solution Gen {gen+1}")
                break

            survivors = self.population[:10]
            new_population = survivors.copy()
            while len(new_population) < POPULATION_SIZE:
                parent1, parent2 = random.sample(survivors, 2)
                child = parent1.crossover(parent2)
                if random.random() < MUTATION_RATE:
                    child = child.mutate()
                new_population.append(child)
            self.population = new_population

# Run
ga = GeneticAlgorithm(FIELD_SIZE)
ga.evolve()
