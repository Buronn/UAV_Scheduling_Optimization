import random
def stochastic_greedy(n_uavs, uav_data, separation_times, seed=None):
    costo = 0
    schedule = []
    backjumped = set()
    if seed is not None:
            random.seed(seed)

    tmp_data = uav_data[:]

    while tmp_data:
        
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