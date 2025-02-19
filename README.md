# Automatic data clustering using nature inspired symbiotic organism search algorithm

implementation of Automatic data clustering using nature inspired symbiotic organism search algorithm paper
[https://www.sciencedirect.com/science/article/pii/S0950705118304647](link)

## Features

- Support for multiple metaheuristic optimization algorithms:
  - ABC (Artificial Bee Colony)
  - CSA (Crow Search Algorithm)
  - DE (Differential Evolution)
  - FPA (Flower Pollination Algorithm)
  - MVO (Multi-Verse Optimizer)
  - PSO (Particle Swarm Optimization)
  - SOS (Symbiotic Organisms Search)

- Built-in support for various datasets:
  - Iris Dataset
  - Breast Cancer
  - Balance Scale
  - Seeds
  - Statlog
  - Contraceptive Method Choice
  - Haberman's Survival
  - Wine

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ADCSOS.git
cd ADCSOS
cd runer
python main.py --help
# Install required dependencies
pip install requirements.txt
```

## Project Structure

```
ADCSOS/
├── runer/
│   ├── main.py      # Main execution script
│   └── problem.py   # Problem definition for clustering
├── utils/
│   ├── datasets.py
│   ├── generate_bound.py
│   ├── saver_model.py
│   └── visualize.py
├── datasets/     # Contains dataset files
├── figs
```

## Usage

Run the clustering algorithm using the command line interface:

```bash
python runer/main.py <epoch> <pop_size> <times_run> <algorithm_name> <save_models>
```

### Parameters:

- `epoch`: Number of iterations (int)
- `pop_size`: Population size (int)
- `times_run`: Number of times to run each dataset (int)
- `algorithm_name`: Algorithm type (ABC/CSA/DE/FPA/MVO/PSO/SOS)
- `save_models`: Whether to save models (yes/no)

### Example:

```bash
python runer/main.py 100 50 5 PSO yes
```

## How It Works

1. The program loads the specified dataset from the datasets directory
2. Generates appropriate bounds for the clustering problem
3. Initializes the selected optimization algorithm
4. Runs the optimization process for the specified number of epochs
5. Saves results and (optionally) the trained models
6. Results are stored in `all_results.pkl`

## Output

The program generates:
- Optimization results stored in pickle format
- (Optional) Saved models for each run
- Performance metrics including:
  - Best solution found
  - Fitness value
  - Execution time

## Technical Details

The clustering problem is formulated as an optimization task where:
- Each solution represents cluster centers
- The objective function minimizes the sum of distances between data points and their nearest cluster centers
- The implementation uses the `mealpy` library for optimization algorithms
- The problem is defined as a minimization task using Euclidean distance

## Dependencies

- mealpy
- numpy
- pickle
- argparse

