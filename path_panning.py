import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import time
import pickle
from algorithms import dijkstra_3d, a_star_3d, genetic_algorithm_3d

# Define a 3D grid and set obstacles
def create_grid(size, start, goal, obstacle_density=0.2):
    grid = np.zeros(size, dtype=int)
    num_obstacles = int(np.prod(size) * obstacle_density)
    
    # Randomly place obstacles in the grid
    obstacles = np.random.choice(np.arange(np.prod(size)), size=num_obstacles, replace=False)
    for obs in obstacles:
        z, y, x = np.unravel_index(obs, size)
        if (x, y, z) != start and (x, y, z) != goal:
            grid[z, y, x] = 1  # Mark obstacle as 1
    return grid

# Animate the particle along the path with line tracking
def animate_particle(path, grid, start, goal, save_dir, algorithm_name):
    if not path:
        print("No path to animate.")
        return
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([0, grid.shape[2]])
    ax.set_ylim([0, grid.shape[1]])
    ax.set_zlim([0, grid.shape[0]])
    
    # Plot the obstacles
    obstacles = np.where(grid == 1)
    ax.scatter(obstacles[2], obstacles[1], obstacles[0], c='red', label="Obstacles")

    # Plot the start and goal points
    ax.scatter(*start[::-1], c='green', label="Start")
    ax.scatter(*goal[::-1], c='blue', label="Goal")
    
    # Create particle and the line track
    particle, = ax.plot([], [], [], 'bo', label="Particle")
    line, = ax.plot([], [], [], 'b-', label="Path")  # Line to trace the path

    # Arrays to store path coordinates for line tracking
    xdata, ydata, zdata = [], [], []

    # Start time of the animation
    start_time = time.time()

    # Create a text box for live time tracking
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

    def update(i):
        point = path[i]

        # Update the particle's position
        particle.set_data([point[2]], [point[1]])  # x, y coordinates
        particle.set_3d_properties([point[0]])     # z coordinate
        
        # Update the path line
        xdata.append(point[2])
        ydata.append(point[1])
        zdata.append(point[0])
        line.set_data(xdata, ydata)
        line.set_3d_properties(zdata)

        # Update the live elapsed time
        elapsed_time = time.time() - start_time
        time_text.set_text(f"Elapsed Time: {elapsed_time:.2f} seconds")
        
        return particle, line, time_text

    # Set the plot title to include the algorithm name
    ax.set_title(f"{algorithm_name} Algorithm", fontsize=12)

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(path), interval=200, blit=True)

    # Save the animation as a gif with live time
    gif_path = os.path.join(save_dir, "3d_particle_animation.gif")
    writer = PillowWriter(fps=10)
    ani.save(gif_path, writer=writer)
    print(f"Animation saved to {gif_path}")

    # Save the last frame as an image
    img_path = os.path.join(save_dir, "3d_particle_last_frame.png")
    plt.savefig(img_path)
    print(f"Last frame image saved to {img_path}")

    # Keep the last frame displayed for inspection
    plt.show()

# Main simulation function
def run_simulation(algorithm='dijkstra', **kwargs):
    size = (10, 10, 10)  # Size of the 3D grid
    start = (0, 0, 0)
    goal = (9, 9, 9)

    # Create the results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create subdirectory for the specific algorithm
    algorithm_dir = os.path.join(results_dir, algorithm)
    if not os.path.exists(algorithm_dir):
        os.makedirs(algorithm_dir)

    # Generate the grid with obstacles, ensuring no obstacle at start or goal
    grid = create_grid(size, start, goal)

    # Run the selected pathfinding algorithm
    try:
        start_time = time.time()
        if algorithm == 'dijkstra':
            print("Running Dijkstra's Algorithm...")
            path, total_distance = dijkstra_3d(grid, start, goal)
        elif algorithm == 'a_star':
            print("Running A* Algorithm...")
            path, total_distance = a_star_3d(grid, start, goal)
        elif algorithm == 'genetic':
            print("Running Genetic Algorithm...")
            path, total_distance = genetic_algorithm_3d(grid, start, goal, population_size=kwargs.get('population_size', 100), generations=kwargs.get('generations', 500), mutation_rate=kwargs.get('mutation_rate', 0.01), save_dir=kwargs.get('save_dir', "results/genetic"), loaded_model=kwargs.get('loaded_model', None))
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        end_time = time.time()

        if not path:
            print("No valid path found in the simulation.")
            return

        # Animate the particle moving along the path and save to file
        animate_particle(path, grid, start, goal, algorithm_dir, algorithm)

        # Save simulation results to a text file
        simulation_data = {
            'algorithm': algorithm,
            'path': path,
            'total_distance': total_distance,
            'time_taken': end_time - start_time
        }
        results_file = os.path.join(algorithm_dir, "simulation_results.json")
        with open(results_file, "w") as f:
            f.write(str(simulation_data))
        print(f"Simulation results saved to {results_file}")

    except ValueError as e:
        print(e)


# Example usage:
if __name__ == "__main__":
    # Run simulation with Dijkstra
    run_simulation('dijkstra')

    # Run simulation with A*
    run_simulation('a_star')

    # Run simulation with Genetic Algorithm
    run_simulation('genetic', population_size=500, generations=1000, mutation_rate=0.1, loaded_model="1.0_9.pkl")
