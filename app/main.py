import time
import random


def read_input_file(filename):
    with open(filename, 'r') as f:
        n_uavs_line = f.readline()
        n_uavs = int(n_uavs_line)

        uav_data = []
        separation_times = []
        for i in range(n_uavs):
            uav_line = f.readline()
            min_time, pref_time, max_time = map(int, uav_line.split())
            uav_data.append((min_time, pref_time, max_time))

            separation_row = []
            while len(separation_row) < n_uavs:
                separation_line = f.readline()
                separation_row.extend(list(map(int, separation_line.split())))
            separation_times.append(separation_row)

    return n_uavs, uav_data, separation_times


def deterministic_greedy(n_uavs, uav_data, separation_times):
    costo = 0
    schedule = []
    for i in range(n_uavs):
        min_time, pref_time, max_time = uav_data[i]

        if not schedule:
            schedule.append(pref_time)
            continue

        start_time = max(min_time, schedule[-1] + separation_times[i-1][i])
        if start_time <= max_time:
            costo += 1
            schedule.append(start_time)
        else:
            schedule.append(pref_time)

    return schedule, costo


def stochastic_greedy(n_uavs, uav_data, separation_times, seed=None):
    if seed is not None:
        random.seed(seed)
    costo = 0

    schedule = []
    for i in range(n_uavs):
        min_time, pref_time, max_time = uav_data[i]

        if not schedule:
            schedule.append(pref_time)
            continue

        start_time = max(min_time, schedule[-1] + separation_times[i-1][i])
        random_start_time = random.randint(min_time, max_time)

        if start_time <= max_time:
            costo += 1
            schedule.append(random_start_time)
        else:
            schedule.append(start_time)

    return schedule, costo


input_file = "input/t2_Deimos.txt"

n_uavs, uav_data, separation_times = read_input_file(input_file)

det_greedy_schedule, det_cost = deterministic_greedy(
    n_uavs, uav_data, separation_times)
print("Deterministic Greedy Schedule:",
      det_greedy_schedule, "\n \t Costo", det_cost)

seeds = [42, 45, 47, 49, 51]
res_stoch_greedy = []
for i in seeds:
    stoch_greedy_schedule, stoch_cost = stochastic_greedy(
        n_uavs, uav_data, separation_times, seed=i)
    print(f"Stochastic Greedy Schedule (Seed {i}):",
          stoch_greedy_schedule, "\n \t Costo", stoch_cost)
    res_stoch_greedy.append((stoch_greedy_schedule, stoch_cost))

def evaluate_solution(schedule, uav_data):
    total_penalty = 0
    for i, start_time in enumerate(schedule):
        _, pref_time, _ = uav_data[i]
        total_penalty += abs(pref_time - start_time)
    return total_penalty

def generate_neighbors(schedule, min_delta=-5, max_delta=5):
    neighbors = []
    for i, start_time in enumerate(schedule):
        for delta in range(min_delta, max_delta+1):
            if delta == 0:
                continue
            new_schedule = schedule[:]
            new_schedule[i] = start_time + delta
            neighbors.append(new_schedule)
    return neighbors


def hill_climbing_first_improvement(start_schedule, uav_data):
    current_schedule = start_schedule
    current_evaluation = evaluate_solution(
        current_schedule, uav_data)

    while True:
        improvement_found = False
        neighbors = generate_neighbors(current_schedule)
        for neighbor in neighbors:
            neighbor_evaluation = evaluate_solution(
                neighbor, uav_data)
            if neighbor_evaluation < current_evaluation:
                current_schedule = neighbor
                current_evaluation = neighbor_evaluation
                improvement_found = True
                break

        if not improvement_found:
            break

    return current_schedule


def hill_climbing_best_improvement(start_schedule, uav_data):
    current_schedule = start_schedule
    current_evaluation = evaluate_solution(
        current_schedule, uav_data)

    while True:
        improvement_found = False
        best_neighbor = None
        best_evaluation = float('inf')

        neighbors = generate_neighbors(current_schedule)
        for neighbor in neighbors:
            neighbor_evaluation = evaluate_solution(
                neighbor, uav_data)
            if neighbor_evaluation < best_evaluation:
                best_neighbor = neighbor
                best_evaluation = neighbor_evaluation
                improvement_found = True

        if improvement_found and best_evaluation < current_evaluation:
            current_schedule = best_neighbor
            current_evaluation = best_evaluation
        else:
            break

    return current_schedule


# Hill Climbing First Improvement

start_time = time.time()
hc_first_improvement_schedule = hill_climbing_first_improvement(
    det_greedy_schedule, uav_data)
end_time = time.time()
print("Time:", end_time - start_time)

# Hill Climbing Best Improvement

start_time = time.time()
hc_best_improvement_schedule = hill_climbing_best_improvement(
    det_greedy_schedule, uav_data)
end_time = time.time()
print("Time:", end_time - start_time)


if hc_first_improvement_schedule == hc_best_improvement_schedule:
    print("Hill Climbing First Improvement and Hill Climbing Best Improvement found the same solution")

for stoch_greedy_schedule,cost in res_stoch_greedy:
    start_time = time.time()
    hc_best_improvement_schedule = hill_climbing_best_improvement(
        stoch_greedy_schedule, uav_data)
    end_time = time.time()
    print("Time:", end_time - start_time)

    if hc_best_improvement_schedule == stoch_greedy_schedule:
        print("Hill Climbing First Improvement and Stochastic Greedy found the same solution")