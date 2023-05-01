import random

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

def greedy_deterministic(vehicles, adjacency_matrix):
    n = len(vehicles)
    assigned = set()
    schedule = []
    total_cost = 0

    for i in range(n):
        vehicle = None
        min_diff = float('inf')
        best_time = None

        # Find the best time for the current vehicle
        for j in range(n):
            if j not in assigned:
                for t in range(vehicles[j][0], vehicles[j][2] + 1):
                    diff = abs(t - vehicles[j][1])
                    wait_time = max(adjacency_matrix[j][k] for k in assigned) if assigned else 0
                    if diff + wait_time < min_diff:
                        min_diff = diff + wait_time
                        vehicle = j
                        best_time = t

        # Assign the vehicle to the best time
        assigned.add(vehicle)
        schedule.append((vehicle, best_time))
        total_cost += min_diff

    return schedule, total_cost



def hill_climbing(schedule, adjacency_matrix, use_deterministic_greedy=True):
    n = len(schedule)
    vehicles = [None] * n
    for i, (vehicle, time) in enumerate(schedule):
        vehicles[i] = (time, adjacency_matrix[i])

    if use_deterministic_greedy:
        greedy = greedy_deterministic
    else:
        greedy = greedy_stochastic

    best_cost = cost(schedule)
    best_schedule = schedule

    while True:
        improved = False
        for i in range(n):
            for j in range(i+1, n):
                new_schedule = swap(schedule, i, j)
                new_cost = cost(new_schedule)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_schedule = new_schedule
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

        schedule = best_schedule
        vehicles = [None] * n
        for i, (vehicle, time) in enumerate(schedule):
            vehicles[i] = (time, adjacency_matrix[i])

        schedule, cost = greedy(vehicles, adjacency_matrix)
        if cost < best_cost:
            best_cost = cost
            best_schedule = schedule

    return best_schedule, best_cost

def cost(schedule):
    total_cost = 0
    for i, (vehicle, time) in enumerate(schedule):
        total_cost += abs(time - vehicles[i][0])
        for j in range(i):
            wait_time = max(0, time - schedule[j][1])
            total_cost += wait_time * vehicles[j][1]
    return total_cost

def swap(schedule, i, j):
    new_schedule = schedule[:]
    new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]
    return new_schedule
