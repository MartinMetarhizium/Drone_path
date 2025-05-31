import numpy as np
import matplotlib.pyplot as plt

FIELD_SIZE = (10, 10)
WORKERS = 4  # must be a perfect square: 4 -> 2x2, 9 -> 3x3

# Divide field among workers
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

# Create a serpentine path inside a block
def serpentine_path(x_start, x_end, y_start, y_end):
    path = []
    for y in range(y_start, y_end):
        if (y - y_start) % 2 == 0:
            # left to right
            for x in range(x_start, x_end):
                path.append((x, y))
        else:
            # right to left
            for x in reversed(range(x_start, x_end)):
                path.append((x, y))
    return path

# Merge paths
def merge_paths(worker_paths):
    merged_path = []
    for idx, path in enumerate(worker_paths):
        if idx == 0:
            merged_path += path
        else:
            # Connect the last point of merged_path to first point of next path
            last = merged_path[-1]
            first = path[0]
            # Try direct Manhattan connection
            intermediate_path = []
            x, y = last
            fx, fy = first

            while x != fx:
                x += 1 if fx > x else -1
                intermediate_path.append((x, y))
            while y != fy:
                y += 1 if fy > y else -1
                intermediate_path.append((x, y))
            merged_path += intermediate_path + path
    return merged_path

# Generate the full path
def generate_full_path(field_size, workers):
    zones = divide_field(field_size, workers)
    worker_paths = []
    for zone in zones:
        x_start, x_end, y_start, y_end = zone
        worker_path = serpentine_path(x_start, x_end, y_start, y_end)
        worker_paths.append(worker_path)

    full_path = merge_paths(worker_paths)
    # Return to start
    full_path.append((0, 0))
    return full_path

# Plot the path
def plot_path(path, field_size):
    width, height = field_size
    grid = np.zeros((height, width))
    for i, (x, y) in enumerate(path):
        grid[y][x] = i + 1

    plt.imshow(grid, cmap='Blues', origin='upper')
    x_coords, y_coords = zip(*path)
    plt.plot(x_coords, y_coords, color='red')
    plt.scatter(x_coords[0], y_coords[0], color='green', label='Start')
    plt.scatter(x_coords[-1], y_coords[-1], color='blue', label='End')
    plt.title("Worker-divided Path")
    plt.legend()
    plt.show()

# Run
full_path = generate_full_path(FIELD_SIZE, WORKERS)
plot_path(full_path, FIELD_SIZE)

print(f"Total steps: {len(full_path)} | Unique visited: {len(set(full_path))}")
