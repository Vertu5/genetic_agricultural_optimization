# -*- coding: utf-8 -*-
import numpy as np
from mapfunctions import subgroups
from evaluation import calculate_fitness_globale

def select_proximity(proximity_map, decision_map, cost_map, budget):
    sorted_vals = sorted(set(proximity_map.flatten()))  # Trier les valeurs uniques par ordre croissant
    decision_proximity = np.zeros_like(proximity_map)  # Initialiser la matrice de decision avec des zeros
    total_cost = 0  # Initialiser le cout total a zero
    for val in sorted_vals:
        if val == 0:  # Ignorer les zeros
            continue
        eligible_coords = []
        for coord in zip(*np.where(proximity_map == val)):
            if decision_map[coord] == 1:  # Verifier si la decision est eligible
                eligible_coords.append(coord)

        for coord in eligible_coords:
            if total_cost + val <= budget:
                decision_proximity[coord] = 2  # Ajouter la decision
                total_cost += val
            else:
                break  # Sortir de la boucle si le budget est depasse
        if total_cost >= budget:
            break  # Sortir de la boucle si le budget est atteint

    return decision_proximity

def select_productivity(productivity_map, decision_map, cost_map, budget):
    sorted_vals = sorted(set(productivity_map.flatten()), reverse=True)  # Trier les valeurs uniques par ordre décroissant
    decision_productivity = np.zeros_like(productivity_map)  # Initialiser la matrice de decision avec des zeros
    total_cost = 0  # Initialiser le cout total a zero
    for val in sorted_vals:
        if val == 0:  # Ignorer les zeros
            continue
        eligible_coords = []
        for coord in zip(*np.where(productivity_map == val)):
            if decision_map[coord] == 1:  # Verifier si la decision est eligible
                eligible_coords.append(coord)
        for coord in eligible_coords:
            if total_cost + cost_map[coord] <= budget:
                decision_productivity[coord] = 2  # Ajouter la decision
                total_cost += cost_map[coord]
            else:
                break  # Sortir de la boucle si le budget est depasse
        if total_cost >= budget:
            break  # Sortir de la boucle si le budget est atteint

    return decision_productivity

def selection(population, subgroup_coords, proximity_map, productivity_map, fitness_threshold, elite_ratio, random_ratio):
    """
    This function selects the next generation of subgroups based on their fitness scores.

    Parameters:
        population (list): A list of numpy arrays representing the current population of subgroups.
        subgroup_coords (list): A list of lists, where each list contains the coordinates of the cells in a subgroup.
        proximity_map (numpy.ndarray): A numpy array representing the proximity map.
        productivity_map (numpy.ndarray): A numpy array representing the productivity map.
        fitness_threshold (float): A threshold value for fitness score, below which subgroups will be rejected.
        elite_ratio (float): The proportion of elite subgroups to be carried over to the next generation.
        random_ratio (float): The proportion of randomly selected subgroups to be added to the next generation.

    Returns:
        next_generation (list): A list of numpy arrays representing the subgroups in the next generation.
    """
    fitness_scores = []
    for subgroup in population:
        fitness = calculate_fitness_globale(subgroup_coords, proximity_map, productivity_map)
        fitness_scores.append(fitness)

    elite_size = int(elite_ratio * len(population))
    random_size = int(random_ratio * len(population))
    selected_size = len(population) - elite_size - random_size

    elite_indices = np.argsort(fitness_scores)[-elite_size:]
    random_indices = np.random.choice(range(len(population)), random_size, replace=False)
    selected_indices = np.argsort(fitness_scores)[:selected_size]

    next_generation = [population[i] for i in elite_indices]
    next_generation += [population[i] for i in random_indices]
    next_generation += [population[i] for i in selected_indices if fitness_scores[i] > fitness_threshold]

    return next_generation
