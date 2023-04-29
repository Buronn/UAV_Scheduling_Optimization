# UAV Scheduling Optimization

This repository contains an implementation of optimization algorithms to efficiently schedule UAVs while minimizing the cost associated with deviation from preferred times.

## Overview

The code features deterministic and stochastic greedy approaches as well as local search methods, including hill climbing with first improvement and best improvement strategies. Users can experiment with different input scenarios and seed values to find the optimal scheduling solution and compare the performance of different optimization techniques.

## Usage

1. Define an input file with UAV data and separation times.
2. Update the `input_file` variable with the path to the input file.
3. Run the script to see the results of the different optimization methods.

## Functions

- `read_input_file(filename)`: Reads the input file containing UAV data and separation times.
- `deterministic_greedy(n_uavs, uav_data, separation_times)`: Implements the deterministic greedy approach.
- `stochastic_greedy(n_uavs, uav_data, separation_times, seed=None)`: Implements the stochastic greedy approach with optional seed value.
- `evaluate_solution(schedule, uav_data)`: Evaluates a given schedule and calculates the total penalty.
- `generate_neighbors(schedule, min_delta=-5, max_delta=5)`: Generates neighbor schedules within a delta range.
- `hill_climbing_first_improvement(start_schedule, uav_data)`: Local search method that finds the first improvement in the search space.
- `hill_climbing_best_improvement(start_schedule, uav_data)`: Local search method that finds the best improvement in the search space.

## Results

The script will output the results of deterministic and stochastic greedy approaches, as well as hill climbing first and best improvement strategies. It will also output the time taken for the different methods.
