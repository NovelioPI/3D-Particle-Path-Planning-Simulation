# 3D Path Planning Simulation with Genetic Algorithm, Dijkstra, and A* Algorithms

This project simulates 3D path planning using different algorithms:
- **Dijkstra's Algorithm**
- **A\* Algorithm**
- **Genetic Algorithm**

The simulation allows for visualizing the path of a particle from a start point to a goal point while avoiding randomly placed obstacles in a 3D space. The simulation is saved as a GIF with live time displayed during the animation.

## Features
- **Dijkstra's Algorithm**: A classical shortest path algorithm.
- **A\* Algorithm**: A heuristic-based pathfinding algorithm that is more efficient than Dijkstra.
- **Genetic Algorithm (GA)**: An evolutionary algorithm that uses natural selection, crossover, and mutation to find a path.
- **Live Time Display**: Shows the live time elapsed during the simulation both in real-time and in the saved GIF animation.
- **Save and Resume GA Training**: Save the model of the genetic algorithm and continue training from the saved model.

## Setup

### Requirements
Make sure you have the following dependencies installed:

```bash
pip install numpy matplotlib
```

## Project Struture
```
.
├── algorithms.py            # Contains the implementations of Dijkstra, A*, and Genetic Algorithm
├── simulation.py            # Contains the simulation and visualization logic
├── results/                 # Directory where results (GIF, image, model) are saved
├── README.md                # This README file
```

## How to Use
# 1. Clone the Repository
```
git clone https://github.com/NovelioPI/3D-Particle-Path-Planning/
cd path-planning-simulation
```
# 2. Run the Simulations
You can run the simulation.py script and choose between different algorithms for 3D path planning.
```
run_simulation(algorithm='dijkstra')
run_simulation(algorithm='a_star')
run_simulation(algorithm='genetic')
```
# 3. Resume Genetic Algorithm Training
You can load a saved model and resume training from where it left off by calling last model.
```
run_simulation('genetic', loaded_model="1.0_9.pkl")
```

## Visualization
The animation of the pathfinding process is displayed as a 3D plot. The plot shows:
- Obstacles in the grid (in red).
- The start point (in green) and the goal point (in blue).
- A live timer showing the elapsed time during the animation.
- The path traced by the particle as it moves toward the goal.
# Example Outputs
- GIF: The animation is saved as a GIF in the results folder.
- Final Frame Image: The final state of the path is saved as an image in the results folder.
- Model: The best path found by the Genetic Algorithm is saved as a .pkl file.

# Customization
You can adjust various parameters like grid size, obstacle density, population size, mutation rate, and more by modifying the relevant functions in `simulation.py` and `algorithms.py`.
