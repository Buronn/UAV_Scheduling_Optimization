import time
import random
import os
import sys
import numpy as np
import bisect
import itertools
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
        # print("\tUAV", id, "scheduled at", time)
        # print("\tTime window:", min_time, "-", max_time)
        if time < min_time or time > max_time:
            # print("\tUAV", id, "scheduled at", time,
            #       "but its time window is", min_time, "-", max_time)
            return False
        if i > 0:
            prev_time, prev_id = schedule[i-1]
            if time < prev_time + separation_times[prev_id][id]:
                # print("\t separation_times[prev_id][id]",
                #       separation_times[prev_id][id])
                # print("\tUAV", id, "scheduled at", time,
                #       "but it is too close to UAV", prev_id)
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
    if seed is not None:
        random.seed(seed)
    costo = 0
    schedule = []
    # Order uavs according to pref_time
    tmp_data = uav_data[:]
    tmp_data.sort(key=lambda x: x[1])
    # print("id\tmin\tpref\tmax\tchosen")
    i = 0
    while i < n_uavs:
        min_time, pref_time, max_time, id = tmp_data[i]
        # print(id,"\t", min_time,"\t", pref_time,"\t", max_time, end="\t")
        if not schedule:
            schedule.append((pref_time, id))
            # print(pref_time)
            i += 1
            continue
        
        start_time = max(min_time, schedule[-1][0] + separation_times[schedule[-1][1]][id])
        
        if start_time <= max_time:
            weights = []
            for j in range(start_time, max_time+1):
                weights.append(1 / (abs(pref_time - j) + 1e-10))
            total_weight = sum(weights)
            probabilities = [weight / total_weight for weight in weights]
            chosen_time = random.choices(range(start_time, max_time+1), probabilities)[0]
            # print(chosen_time)
            costo += abs(pref_time - chosen_time)
            schedule.append((chosen_time, id))
            i += 1
        else:
            schedule.append((max_time, id))
            costo += 2 * abs(pref_time - max_time)
            # print(max_time, "PENALTY")
            i += 1

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
print("Algoritmo utilizado\t\t\t\t\t\tCosto\tFactible")
print("Deterministic Greedy Schedule\t\t\t\t\t", det_cost, "\t", check_schedule(
    uav_data, separation_times, det_greedy_schedule))

seeds = [42, 45, 47, 48, 51]

res_stoch_greedy = []
for i in seeds:
    stoch_greedy_schedule, stoch_cost = stochastic_greedy(
        n_uavs, uav_data, separation_times, seed=i)
    print(f"Stochastic Greedy Schedule (Seed {i})\t\t\t\t", stoch_cost, "\t", check_schedule(
        uav_data, separation_times, stoch_greedy_schedule))

    res_stoch_greedy.append((stoch_greedy_schedule, stoch_cost))

def swap_positions(schedule, i, j):
    """Swap the positions of two UAVs in the schedule."""
    schedule[i], schedule[j] = schedule[j], schedule[i]

def calculate_cost(uav_data, schedule):
    """Calculate the cost of a schedule."""
    cost = 0
    for time, id in schedule:
        _, pref_time, _, _ = uav_data[id]
        cost += abs(pref_time - time)
    return cost

def first_choice_hill_climbing(n_uavs, uav_data, separation_times, initial_schedule, initial_cost):
    """Implement the first-choice hill climbing algorithm."""
    current_schedule = initial_schedule
    current_cost = initial_cost
    while True:
        for i in range(n_uavs):
            for j in range(i+1, n_uavs):
                # Swap two UAVs and recalculate the schedule
                swap_positions(current_schedule, i, j)
                new_schedule = recalculate_schedule(uav_data, separation_times, current_schedule)
                if new_schedule is not None:
                    new_cost = calculate_cost(uav_data, new_schedule)
                    if new_cost < current_cost:
                        # Found a better solution, update the current solution and break
                        current_schedule = new_schedule
                        current_cost = new_cost
                        break
                # Swap back if the new schedule is not better
                swap_positions(current_schedule, i, j)
            else:
                continue  # Continue if the inner loop wasn't broken
            break  # Inner loop was broken, break the outer loop
        else:
            # No better solution found, return the current solution
            return current_schedule, current_cost

def recalculate_schedule(uav_data, separation_times, schedule):
    """Recalculate the landing times in the schedule."""
    new_schedule = []
    for i, (time, id) in enumerate(schedule):
        min_time, _, max_time, _ = uav_data[id]
        if i > 0:
            prev_time, prev_id = new_schedule[i-1]
            time = max(min_time, prev_time + separation_times[prev_id][id])
        if time > max_time:
            return None  # The schedule is not feasible
        new_schedule.append((time, id))
    return new_schedule

def steepest_ascent_hill_climbing(n_uavs, uav_data, separation_times, initial_schedule, initial_cost):
    """Implement the steepest-ascent hill climbing algorithm."""
    current_schedule = initial_schedule
    current_cost = initial_cost
    while True:
        best_schedule = current_schedule
        best_cost = current_cost
        for i in range(n_uavs):
            for j in range(i+1, n_uavs):
                # Swap two UAVs and recalculate the schedule
                swap_positions(current_schedule, i, j)
                new_schedule = recalculate_schedule(uav_data, separation_times, current_schedule)
                if new_schedule is not None:
                    new_cost = calculate_cost(uav_data, new_schedule)
                    if new_cost < best_cost:
                        best_schedule = new_schedule
                        best_cost = new_cost
                # Swap back to continue the search
                swap_positions(current_schedule, i, j)
        if best_cost < current_cost:
            # Found a better solution, update the current solution
            current_schedule = best_schedule
            current_cost = best_cost
        else:
            # No better solution found, return the current solution
            return current_schedule, current_cost


res_first_choice_hill_climbing = []
res_steepest_ascent_hill_climbing = []

i = 0
first_choice_schedule, first_choice_cost = first_choice_hill_climbing(
    n_uavs, uav_data, separation_times, det_greedy_schedule, det_cost)
print("Hill Climbing first choice (Deterministic Greedy)\t\t", first_choice_cost, "\t", check_schedule(
    uav_data, separation_times, first_choice_schedule))
res_first_choice_hill_climbing.append((first_choice_schedule, first_choice_cost))
for seed in seeds:
    first_choice_schedule, first_choice_cost = first_choice_hill_climbing(
        n_uavs, uav_data, separation_times, res_stoch_greedy[i][0], res_stoch_greedy[i][1])
    print(f"Hill Climbing first choice (Stochastic Greedy, Seed {seed})\t\t", first_choice_cost, "\t", check_schedule(
        uav_data, separation_times, first_choice_schedule))

    res_first_choice_hill_climbing.append(
        (first_choice_schedule, first_choice_cost))
    i += 1

better_choice_schedule, better_choice_cost = steepest_ascent_hill_climbing(
    n_uavs, uav_data, separation_times, det_greedy_schedule, det_cost)
print("Hill Climbing steepest ascent (Deterministic Greedy)\t\t", better_choice_cost, "\t", check_schedule(
    uav_data, separation_times, better_choice_schedule))
res_steepest_ascent_hill_climbing.append(
    (better_choice_schedule, better_choice_cost))
i = 0
for seed in seeds:
    steepest_ascent_schedule, steepest_ascent_cost = steepest_ascent_hill_climbing(
        n_uavs, uav_data, separation_times, res_stoch_greedy[i][0], res_stoch_greedy[i][1])
    print(f"Hill Climbing steepest ascent (Stochastic Greedy, Seed {seed})\t", steepest_ascent_cost, "\t", check_schedule(
        uav_data, separation_times, steepest_ascent_schedule))

    res_steepest_ascent_hill_climbing.append(
        (steepest_ascent_schedule, steepest_ascent_cost))
    i += 1


def tabu_search(n_uavs, uav_data, separation_times, initial_schedule, initial_cost, max_iterations, tabu_size):
    """Implement the tabu search algorithm."""
    current_schedule = initial_schedule
    current_cost = initial_cost
    best_schedule = current_schedule
    best_cost = current_cost
    tabu_list = []

    for _ in range(max_iterations):
        best_candidate = None
        best_candidate_cost = float('inf')

        for i in range(n_uavs):
            for j in range(i+1, n_uavs):
                # Swap two UAVs and recalculate the schedule
                swap_positions(current_schedule, i, j)
                new_schedule = recalculate_schedule(uav_data, separation_times, current_schedule)
                if new_schedule is not None and (new_schedule not in tabu_list or calculate_cost(uav_data, new_schedule) < best_cost):
                    new_cost = calculate_cost(uav_data, new_schedule)
                    if new_cost < best_candidate_cost:
                        best_candidate = new_schedule
                        best_candidate_cost = new_cost
                # Swap back to continue the search
                swap_positions(current_schedule, i, j)

        if best_candidate is not None:
            current_schedule = best_candidate
            current_cost = best_candidate_cost
            tabu_list.append(best_candidate)
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)

            if current_cost < best_cost:
                best_schedule = current_schedule
                best_cost = current_cost

    return best_schedule, best_cost

tabu_schedule, tabu_cost = tabu_search(
    n_uavs, uav_data, separation_times, det_greedy_schedule, det_cost, 100, 10)
print("Tabu Search (Deterministic Greedy)\t\t\t\t", tabu_cost, "\t", check_schedule(
    uav_data, separation_times, tabu_schedule))
i = 0
for seed in seeds:
    tabu_schedule, tabu_cost = tabu_search(
        n_uavs, uav_data, separation_times, res_stoch_greedy[i][0], res_stoch_greedy[i][1], 100, 10)
    print(f"Tabu Search (Stochastic Greedy, Seed {seed})\t\t\t", tabu_cost, "\t", check_schedule(
        uav_data, separation_times, tabu_schedule))
    

    i += 1