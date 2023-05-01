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
            uav_data.append((min_time, pref_time, max_time, i))

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
        min_time, pref_time, max_time, id = uav_data[i]
        if not schedule:
            schedule.append((pref_time,id))
            continue

        start_time = max(min_time, schedule[-1][0] + separation_times[i-1][i])
        if start_time <= max_time:
            if start_time >= min_time:
                costo += abs(pref_time - start_time)
                schedule.append((start_time,id))
            else:
                schedule.append((max_time,id))

    return schedule, costo

def greedy_stochastic(vehicles, adjacency_matrix, iterations=100):
    n = len(vehicles)
    assigned = set()
    schedule = []
    total_cost = 0

    for i in range(n):
        vehicle = None
        best_diff = float('inf')
        best_time = None

        # Find the best time for the current vehicle
        for j in range(n):
            if j not in assigned:
                for _ in range(iterations):
                    t = random.randint(vehicles[j][0], vehicles[j][2])
                    diff = abs(t - vehicles[j][1])
                    wait_time = max(adjacency_matrix[j][k] for k in assigned) if assigned else 0
                    if diff + wait_time < best_diff:
                        best_diff = diff + wait_time
                        vehicle = j
                        best_time = t

        # Assign the vehicle to the best time
        assigned.add(vehicle)
        schedule.append((vehicle, best_time))
        total_cost += best_diff

    return schedule, total_cost

def stochastic_greedy(n_uavs, uav_data, separation_times, seed=None):
    if seed is not None:
        random.seed(seed)
    costo = 0
    schedule = []
    # Disorder uavs
    uav_data = random.sample(uav_data, len(uav_data))
    print(uav_data)
    for i in range(n_uavs):
        min_time, pref_time, max_time, id = uav_data[i]

        if not schedule:
            schedule.append((pref_time,id))
            continue

        start_time = max(min_time, schedule[-1][0] + separation_times[id-1][id])
        
        if start_time <= max_time:
            
            costo += abs(pref_time - start_time)

            schedule.append((start_time,id))
        else:
            print("UAV", id, "not scheduled")
            schedule.append((start_time,id))

    return schedule, costo

import os

from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

# Obtener todos los nombres de archivo que terminan en .txt en el directorio actual
archivos_txt = [filename for filename in os.listdir('input/') if filename.endswith('.txt')]

# Crear un completer para el menú desplegable
completer = WordCompleter(archivos_txt)

# Mostrar el menú desplegable y obtener la selección del usuario
opcion = prompt("Seleccione un archivo: ", completer=completer)

# Realizar una acción basada en la selección del usuario
if opcion not in archivos_txt:
    print("Opción inválida.")
else:
    nombre_archivo = opcion
    print(f"Ha seleccionado el archivo '{nombre_archivo}'.")
    # Realizar la acción deseada con el archivo seleccionado
    # por ejemplo, puedes abrirlo con la función open() de Python
input_file = f"input/{nombre_archivo}"
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
        _, pref_time, _, _ = uav_data[i]
        total_penalty += abs(pref_time - start_time)
    return total_penalty

def generate_neighbors(schedule, min_delta=-10, max_delta=10):
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
print("Time of hc_first-det_greedy:", end_time - start_time)

# Hill Climbing Best Improvement

start_time = time.time()
hc_best_improvement_schedule = hill_climbing_best_improvement(
    det_greedy_schedule, uav_data)
end_time = time.time()
print("Time of hc_best_improv-det_greedy:", end_time - start_time)


if hc_first_improvement_schedule == hc_best_improvement_schedule:
    print("Hill Climbing First Improvement and Hill Climbing Best Improvement found the same solution")

# cont = 0
# for stoch_greedy_schedule,cost in res_stoch_greedy:

#     print(f'Iteration number {cont} \n')
#     start_time = time.time()
#     hc_best_improvement_schedule = hill_climbing_best_improvement(
#         stoch_greedy_schedule, uav_data)
#     end_time = time.time()
#     print("\tTime of hc_best-stock_greed:", end_time - start_time)
#     print("\t\tSolution : ", hc_best_improvement_schedule)

    
#     start_time = time.time()
#     hc_first_improvement_schedule = hill_climbing_best_improvement(
#         stoch_greedy_schedule, uav_data)
#     end_time = time.time()
#     print("\tTime of first_improv-stock_greedy:", end_time - start_time)
#     print("\t\tSolution : ", hc_first_improvement_schedule)

#     if hc_first_improvement_schedule == hc_best_improvement_schedule:
#         print("\tBoth found the same solution")

#     cont += 1
    


def tabu_search(start_schedule, uav_data, max_iterations=100, tabu_list_size=10):
    current_schedule = start_schedule
    current_evaluation = evaluate_solution(current_schedule, uav_data)

    best_schedule = current_schedule
    best_evaluation = current_evaluation

    tabu_list = []
    iteration = 0

    while iteration < max_iterations:
        neighbors = generate_neighbors(current_schedule)
        best_neighbor = None
        best_neighbor_evaluation = float('inf')

        for neighbor in neighbors:
            if (neighbor not in tabu_list) or (evaluate_solution(neighbor, uav_data) < best_evaluation):
                neighbor_evaluation = evaluate_solution(neighbor, uav_data)
                if neighbor_evaluation < best_neighbor_evaluation:
                    best_neighbor = neighbor
                    best_neighbor_evaluation = neighbor_evaluation

        if best_neighbor is None:
            break

        if best_neighbor_evaluation < best_evaluation:
            best_schedule = best_neighbor
            best_evaluation = best_neighbor_evaluation

        current_schedule = best_neighbor

        if len(tabu_list) >= tabu_list_size:
            tabu_list.pop(0)
        tabu_list.append(current_schedule)

        iteration += 1

    return best_schedule

# # Tabu Search on Deterministic Greedy
# ts_det_greedy_schedule = tabu_search(det_greedy_schedule, uav_data)
# ts_det_greedy_cost = evaluate_solution(ts_det_greedy_schedule, uav_data)
# print("Tabu Search on Deterministic Greedy Schedule:", ts_det_greedy_schedule, "\n\tCost:", ts_det_greedy_cost)

# # Tabu Search on Stochastic Greedy
# for stoch_greedy_schedule, cost in res_stoch_greedy:
#     ts_stoch_greedy_schedule = tabu_search(stoch_greedy_schedule, uav_data)
#     ts_stoch_greedy_cost = evaluate_solution(ts_stoch_greedy_schedule, uav_data)
#     print("Tabu Search on Stochastic Greedy Schedule:", ts_stoch_greedy_schedule, "\n\tCost:", ts_stoch_greedy_cost)
