import time
import random
import os
import sys
import numpy as np
import bisect
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter


def read_input_file(filename):
    with open(filename, 'r') as f:
        n_uavs_line = f.readline()
        n_uavs = int(n_uavs_line)

        uav_data = []
        separation_times = []
        for i in range(n_uavs):
            uav_line = f.readline()
            min_time, pref_time, max_time = map(int, uav_line.split())
            uav_data.append((min_time, pref_time, max_time, i))

            separation_row = []
            while len(separation_row) < n_uavs:
                separation_line = f.readline()
                separation_row.extend(list(map(int, separation_line.split())))
            separation_times.append(separation_row)

    return n_uavs, uav_data, separation_times


def check_schedule(uav_data, separation_times, schedule):
    for i in range(len(schedule)):
        time, id = schedule[i]
        min_time, _, max_time, _ = uav_data[id]
        print("\tUAV", id, "scheduled at", time)
        print("\tTime window:", min_time, "-", max_time)
        if time < min_time or time > max_time:
            print("\tUAV", id, "scheduled at", time,
                  "but its time window is", min_time, "-", max_time)
            return False
        if i > 0:
            prev_time, prev_id = schedule[i-1]
            if time < prev_time + separation_times[prev_id][id]:
                print("\t separation_times[prev_id][id]",
                      separation_times[prev_id][id])
                print("\tUAV", id, "scheduled at", time,
                      "but it is too close to UAV", prev_id)
                return False
    return True


def deterministic_greedy(n_uavs, uav_data, separation_times):
    costo = 0
    schedule = []
    # Order uavs according to pref_time
    tmp_data = uav_data[:]
    tmp_data.sort(key=lambda x: x[0])
    # print("separation_times", separation_times)
    # print("ID\tMin\tPref\tMax")

    for i in range(n_uavs):
        min_time, pref_time, max_time, id = tmp_data[i]
        # print(id,"\t", min_time,"\t", pref_time,"\t", max_time)
        if not schedule:
            # print("Tiempo escogido:", pref_time, "para el UAV", id)
            schedule.append((pref_time, id))
            continue
        # print("\tTiempo del anterior", schedule[i-1][0])
        # print("\tseparation_times[i-1][i]", separation_times[schedule[i-1][1]][id])
        start_time = max(
            min_time, schedule[i-1][0] + separation_times[schedule[i-1][1]][id])
        # print("Tiempo escogido:", start_time, "para el UAV", id)
        if start_time <= max_time:
            costo += abs(pref_time - start_time)
            schedule.append((start_time, id))

    return schedule, costo


def stochastic_greedy(n_uavs, uav_data, separation_times, seed=None):
    costo = 0
    schedule = []
    backjumped = set()

    tmp_data = uav_data[:]

    while tmp_data:
        if seed is not None:
            random.seed(seed)
        # Select a UAV with probability proportional to inverse of its min_time and pref_time
        weights = [1 / (min_time + pref_time + 1e-10) if id not in backjumped else 0 for min_time, pref_time, _, id in tmp_data]
        total_weight = sum(weights)
        probabilities = [weight / total_weight for weight in weights]
        selected_index = random.choices(range(len(tmp_data)), probabilities)[0]
        min_time, pref_time, max_time, id = tmp_data.pop(selected_index)
        if not schedule:
            schedule.append((pref_time, id))
            continue

        start_time = max(min_time, schedule[-1][0] + separation_times[schedule[-1][1]][id])

        if start_time <= max_time:
            costo += abs(pref_time - start_time)
            schedule.append((start_time, id))
            backjumped.clear()  # Clear backjumped set when a UAV is successfully scheduled
        else:
            if schedule:
                # Backjump
                deleted = schedule.pop()
                costo -= abs((uav_data[deleted[1]])[1] - deleted[0]) # Devolvemos el costo de la programación anterior
                tmp_data.append((uav_data[deleted[1]])) # Devolvemos el UAV anterior a la lista tmp
                tmp_data.append((uav_data[id])) # Devolvemos el UAV actual a la lista tmp
                backjumped.add(deleted[1])  # Add the deleted UAV to the backjumped set
                seed = random.randint(0, 1000000)

                continue

    return schedule, costo

archivos_txt = [filename for filename in os.listdir(
    'input/') if filename.endswith('.txt')]
completer = WordCompleter(archivos_txt)
opcion = prompt("Seleccione un archivo: ", completer=completer)
if opcion not in archivos_txt:
    exit("El archivo seleccionado no existe.")
else:
    nombre_archivo = opcion
input_file = f"input/{nombre_archivo}"
n_uavs, uav_data, separation_times = read_input_file(input_file)


det_greedy_schedule, det_cost = deterministic_greedy(
    n_uavs, uav_data, separation_times)
print("Deterministic Greedy Schedule:",
      det_greedy_schedule, "\n \t Costo", det_cost)
print("\tChecking if the schedule is valid...")
print("\tSchedule is valid:", check_schedule(
    uav_data, separation_times, det_greedy_schedule))

seeds = [42]
#seeds = [42, 45, 47, 48, 51]

res_stoch_greedy = []
print("\n \t \t --------------------------- Stochastic Greedy --------------------------- \n")
for i in seeds:
    stoch_greedy_schedule, stoch_cost = stochastic_greedy(
        n_uavs, uav_data, separation_times, seed=i)
    print(f"Stochastic Greedy Schedule (Seed {i}):",
          stoch_greedy_schedule, "\n \t Costo", stoch_cost)

    print("\n")
    print("\tChecking if the schedule is valid...")
    print("\tSchedule is valid:", check_schedule(
        uav_data, separation_times, stoch_greedy_schedule))

    res_stoch_greedy.append((stoch_greedy_schedule, stoch_cost))

sys.exit(0)


def get_solution_cost(solution, uav_data, separation_times):
    cost = 0
    for i in range(len(solution) - 1):
        cost += abs(uav_data[solution[i][1]][1] - solution[i][0])
        cost += max(0, solution[i + 1][0] - (solution[i][0] +
                                             separation_times[solution[i][1]][solution[i + 1][1]]))
    cost += abs(uav_data[solution[-1][1]][1] - solution[-1][0])
    return cost


def generate_neighbors(solution):
    neighbors = []
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            neighbor = solution.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)
    return neighbors


def hill_climbing_first_improvement(initial_solution, uav_data, separation_times, max_iterations=1000):
    current_solution = initial_solution.copy()
    current_cost = get_solution_cost(
        current_solution, uav_data, separation_times)

    for _ in range(max_iterations):
        neighbors = generate_neighbors(current_solution)
        improvement_found = False

        for neighbor in neighbors:
            neighbor_cost = get_solution_cost(
                neighbor, uav_data, separation_times)

            if neighbor_cost < current_cost:
                current_solution = neighbor.copy()
                current_cost = neighbor_cost
                improvement_found = True
                break

        if not improvement_found:
            break

    return current_solution, current_cost


def hill_climbing_best_improvement(initial_solution, uav_data, separation_times, max_iterations=1000):
    current_solution = initial_solution.copy()
    current_cost = get_solution_cost(
        current_solution, uav_data, separation_times)

    for _ in range(max_iterations):
        neighbors = generate_neighbors(current_solution)
        best_neighbor = None
        best_cost = current_cost

        for neighbor in neighbors:
            neighbor_cost = get_solution_cost(
                neighbor, uav_data, separation_times)

            if neighbor_cost < best_cost:
                best_neighbor = neighbor.copy()
                best_cost = neighbor_cost

        if best_neighbor is None:
            break

        current_solution = best_neighbor
        current_cost = best_cost

    return current_solution, current_cost


# Hill Climbing first improvement
print("\nHill Climbing (First Improvement):")
det_hc_first_schedule, det_hc_first_cost = hill_climbing_first_improvement(
    det_greedy_schedule, uav_data, separation_times)
print("From Deterministic Greedy Schedule:",
      det_hc_first_schedule, "\n \t Cost", det_hc_first_cost)

for idx, stoch_greedy_result in enumerate(res_stoch_greedy):
    stoch_hc_first_schedule, stoch_hc_first_cost = hill_climbing_first_improvement(
        stoch_greedy_result[0], uav_data, separation_times)
    print(f"From Stochastic Greedy Schedule (Seed {seeds[idx]}):",
          stoch_hc_first_schedule, "\n \t Cost", stoch_hc_first_cost)

# Hill Climbing best improvement
print("\nHill Climbing (Best Improvement):")
det_hc_best_schedule, det_hc_best_cost = hill_climbing_best_improvement(
    det_greedy_schedule, uav_data, separation_times)
print("From Deterministic Greedy Schedule:",
      det_hc_best_schedule, "\n \t Cost", det_hc_best_cost)

for idx, stoch_greedy_result in enumerate(res_stoch_greedy):
    stoch_hc_best_schedule, stoch_hc_best_cost = hill_climbing_best_improvement(
        stoch_greedy_result[0], uav_data, separation_times)
    print(f"From Stochastic Greedy Schedule (Seed {seeds[idx]}):",
          stoch_hc_best_schedule, "\n \t Cost", stoch_hc_best_cost)


def tabu_search(initial_solution, uav_data, separation_times, max_iterations=1000, tabu_size=5, aspiration_value=None):
    best_solution = initial_solution.copy()
    best_cost = get_solution_cost(initial_solution, uav_data, separation_times)
    tabu_list = []

    current_solution = initial_solution.copy()
    current_cost = best_cost

    for _ in range(max_iterations):
        neighbors = generate_neighbors(current_solution)
        best_neighbor = None
        best_neighbor_cost = float("inf")

        for neighbor in neighbors:
            neighbor_cost = get_solution_cost(
                neighbor, uav_data, separation_times)

            if neighbor_cost < best_cost:
                if aspiration_value and neighbor_cost < aspiration_value:
                    best_neighbor = neighbor
                    best_neighbor_cost = neighbor_cost
                    break

                if neighbor not in tabu_list:
                    best_neighbor = neighbor
                    best_neighbor_cost = neighbor_cost
                    break

            elif neighbor not in tabu_list and neighbor_cost < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = neighbor_cost

        if best_neighbor is not None:
            current_solution = best_neighbor
            current_cost = best_neighbor_cost

            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost

            tabu_list.append(best_neighbor)
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)

    return best_solution, best_cost


# Tabu Search
print("\nTabu Search:")
det_tabu_schedule, det_tabu_cost = tabu_search(
    det_greedy_schedule, uav_data, separation_times)
print("From Deterministic Greedy Schedule:",
      det_tabu_schedule, "\n \t Cost", det_tabu_cost)

for idx, stoch_greedy_result in enumerate(res_stoch_greedy):
    stoch_tabu_schedule, stoch_tabu_cost = tabu_search(
        stoch_greedy_result[0], uav_data, separation_times)
    print(f"From Stochastic Greedy Schedule (Seed {seeds[idx]}):",
          stoch_tabu_schedule, "\n \t Cost", stoch_tabu_cost)
