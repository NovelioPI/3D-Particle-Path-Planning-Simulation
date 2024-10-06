import numpy as np
import heapq
import random
import time
import os
import pickle

# Dijkstra's Algorithm in 3D
def dijkstra_3d(grid, start, goal):
    rows, cols, depth = grid.shape
    
    if grid[start] == 1:
        raise ValueError(f"Start point {start} is inside an obstacle.")
    if grid[goal] == 1:
        raise ValueError(f"Goal point {goal} is inside an obstacle.")
    
    dist = np.full_like(grid, np.inf, dtype=float)
    dist[start] = 0
    pq = [(0, start)]
    prev = {start: None}
    
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]  # 6 possible moves in 3D

    while pq:
        current_dist, current_node = heapq.heappop(pq)
        if current_node == goal:
            break
        
        for d in directions:
            neighbor = tuple(np.add(current_node, d))
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and 0 <= neighbor[2] < depth:
                if grid[neighbor] == 0:  # Check if neighbor is not an obstacle
                    alt = current_dist + 1  # Cost of moving to a neighbor
                    if alt < dist[neighbor]:
                        dist[neighbor] = alt
                        prev[neighbor] = current_node
                        heapq.heappush(pq, (alt, neighbor))

    if dist[goal] == np.inf:
        print(f"No path found from {start} to {goal}.")
        return [], np.inf

    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()

    return path, dist[goal]

# A* Algorithm in 3D
def a_star_3d(grid, start, goal):
    rows, cols, depth = grid.shape
    
    if grid[start] == 1:
        raise ValueError(f"Start point {start} is inside an obstacle.")
    if grid[goal] == 1:
        raise ValueError(f"Goal point {goal} is inside an obstacle.")
    
    def heuristic(node1, node2):
        return np.linalg.norm(np.array(node1) - np.array(node2))
    
    dist = np.full_like(grid, np.inf, dtype=float)
    dist[start] = 0
    pq = [(heuristic(start, goal), 0, start)]
    prev = {start: None}
    
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    while pq:
        _, current_dist, current_node = heapq.heappop(pq)
        if current_node == goal:
            break
        
        for d in directions:
            neighbor = tuple(np.add(current_node, d))
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and 0 <= neighbor[2] < depth:
                if grid[neighbor] == 0:
                    g_score = current_dist + 1
                    f_score = g_score + heuristic(neighbor, goal)
                    if g_score < dist[neighbor]:
                        dist[neighbor] = g_score
                        prev[neighbor] = current_node
                        heapq.heappush(pq, (f_score, g_score, neighbor))

    if dist[goal] == np.inf:
        print(f"No path found from {start} to {goal}.")
        return [], np.inf

    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()

    return path, dist[goal]

# Genetic Algorithm for Pathfinding in 3D
def genetic_algorithm_3d(grid, start, goal, population_size=100, generations=500, mutation_rate=0.01, save_dir="results/genetic", loaded_model=None):
    rows, cols, depth = grid.shape
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    def generate_random_path():
        path = [start]
        current_node = start
        while current_node != goal:
            possible_moves = []
            for d in directions:
                neighbor = tuple(np.add(current_node, d))
                if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and 0 <= neighbor[2] < depth and 
                    grid[neighbor] == 0):  # Ensure valid and non-obstacle move
                    possible_moves.append(neighbor)

            if not possible_moves:
                break
            
            # Select the move that gets closer to the goal
            current_node = min(possible_moves, key=lambda n: np.linalg.norm(np.array(n) - np.array(goal)))
            path.append(current_node)
            
            if current_node == goal:
                break
        return path

    def fitness(path):
        last_node = path[-1]
        distance_to_goal = np.linalg.norm(np.array(last_node) - np.array(goal))
        
        if last_node == goal:
            return 10 / (1 + len(path))
        else:
            return 1 / (1 + distance_to_goal)

    def selection(population):
        weights = [fitness(p) for p in population]
        total = sum(weights)
        probabilities = [w / total for w in weights]
        selected = random.choices(population, weights=probabilities, k=2)
        return selected

    def crossover(parent1, parent2):
        common_nodes = set(parent1) & set(parent2)
        if common_nodes:
            common_node = random.choice(list(common_nodes))
            index1 = parent1.index(common_node)
            index2 = parent2.index(common_node)
            child = parent1[:index1] + parent2[index2:]
        else:
            split_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
            child = parent1[:split_point] + parent2[split_point:]
        return child

    def mutate(path):
        if random.random() < mutation_rate:
            idx = random.randint(1, len(path) - 2)
            neighbor = tuple(np.add(path[idx], random.choice(directions)))
            if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and 0 <= neighbor[2] < depth and 
                grid[neighbor] == 0):
                path[idx] = neighbor
        return path
        
    # If a model is loaded, use its saved state
    if loaded_model:
        print("Resuming from the loaded model...")
        with open(f'{save_dir}/{loaded_model}', 'rb') as f:
            loaded_model = pickle.load(f)
        best_path = loaded_model['best_path']
        best_fitness = loaded_model['fitness']
        population_size = loaded_model['population_size']  # Override with saved population size
        mutation_rate = loaded_model['mutation_rate']  # Override with saved mutation rate
        population = [generate_random_path() for _ in range(population_size)]  # Recreate the population
    else:
        best_path = None
        best_fitness = float('-inf')
        population = [generate_random_path() for _ in range(population_size)]
    
    start_time = time.time()

    for generation in range(generations):
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = selection(population)
            child1, child2 = crossover(parent1, parent2), crossover(parent2, parent1)
            child1, child2 = mutate(child1), mutate(child2)
            new_population.extend([child1, child2])
        
        population = new_population

        generation_best_path = max(population, key=fitness)
        generation_best_fitness = fitness(generation_best_path)
        if generation_best_fitness > best_fitness:
            best_fitness = generation_best_fitness
            best_path = generation_best_path
        
        print(f"Generation {generation}, Best Fitness: {best_fitness}, Path Length: {len(best_path)}")

        if best_fitness >= 0.99:
            print(f"Optimal solution found at generation {generation}")
            break

    end_time = time.time()
    time_taken = round(end_time - start_time, 2)
    path_length = len(best_path)

    # Save the model (best path and fitness) using pickle
    model_filename = f"{best_fitness}_{path_length}.pkl"
    model_filepath = os.path.join(save_dir, model_filename)
    model_data = {
        'best_path': best_path,
        'fitness': best_fitness,
        'path_length': path_length,
        'time_taken': time_taken,
        'population_size': population_size,
        'generations': generations,
        'mutation_rate': mutation_rate
    }
    
    with open(model_filepath, 'wb') as model_file:
        pickle.dump(model_data, model_file)
    
    print(f"Model saved to {model_filepath}")

    return best_path, path_length