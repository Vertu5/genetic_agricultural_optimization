# -*- coding: utf-8 -*-
import random
import numpy as np
from mapfunctions import *
from utils import *
from evaluation import *
from generator import *


def generate_population(Map, cost_map, pop_size, budget):
    population = []
    while len(population) < pop_size:
        Map_pop = np.zeros((len(Map), len(Map[0])))
        sum_cost = 0
        while sum_cost <= budget:
            j, k = np.random.randint(0, Map.shape[0]), np.random.randint(0, Map.shape[1])
            if Map[j, k] == 1 and sum_cost + cost_map[j, k] <= budget:
                Map_pop[j, k] = 2
                sum_cost += cost_map[j, k]
            elif sum_cost + cost_map[j, k] >= budget: 
                break

        population.append(Map_pop)
    return population


def Pareto_dominates(fitness1, fitness2):
    """
    Returns True if fitness1 Pareto-dominates fitness2:
      - Objective 1: Compactness C(S)   [MINIMIZE: c1 <= c2]
      - Objective 2: Proximity P(S)     [MINIMIZE: p1 <= p2]
      - Objective 3: Productivity R(S)  [MAXIMIZE: r1 >= r2]
    """
    c1, p1, r1 = fitness1
    c2, p2, r2 = fitness2

    not_worse = (c1 <= c2) and (p1 <= p2) and (r1 >= r2)
    strictly_better = (c1 < c2) or (p1 < p2) or (r1 > r2)
    return not_worse and strictly_better



def fast_non_dominated_sort(population_fitness):
    """
    Fast Non-Dominated Sorting for multi-objective optimization (NSGA-II).
    Returns fronts (list of lists of indices) and ranks dict (idx -> rank).
    """
    pop_size = len(population_fitness)
    S = [[] for _ in range(pop_size)]
    n = [0] * pop_size
    fronts = [[]]
    ranks = {}

    for i in range(pop_size):
        for j in range(pop_size):
            if i == j:
                continue
            if Pareto_dominates(population_fitness[i], population_fitness[j]):
                S[i].append(j)
            elif Pareto_dominates(population_fitness[j], population_fitness[i]):
                n[i] += 1
        if n[i] == 0:
            ranks[i] = 0
            fronts[0].append(i)

    i_front = 0
    while len(fronts[i_front]) > 0:
        next_front = []
        for p in fronts[i_front]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    ranks[q] = i_front + 1
                    next_front.append(q)
        i_front += 1
        fronts.append(next_front)

    if not fronts[-1]:
        fronts.pop()

    return fronts, ranks


def calculate_crowding_distance(front_indices, population_fitness):
    """
    Calculates Crowding Distance for individuals within a single Pareto front.
    """
    if not front_indices:
        return {}

    num_obj = len(population_fitness[0])
    distances = {idx: 0.0 for idx in front_indices}
    num_indivs = len(front_indices)

    if num_indivs <= 2:
        for idx in front_indices:
            distances[idx] = float('inf')
        return distances

    for m in range(num_obj):
        sorted_front = sorted(front_indices, key=lambda idx: population_fitness[idx][m])
        distances[sorted_front[0]] = float('inf')
        distances[sorted_front[-1]] = float('inf')

        obj_min = population_fitness[sorted_front[0]][m]
        obj_max = population_fitness[sorted_front[-1]][m]
        range_m = obj_max - obj_min

        if range_m > 0:
            for i in range(1, num_indivs - 1):
                prev_val = population_fitness[sorted_front[i - 1]][m]
                next_val = population_fitness[sorted_front[i + 1]][m]
                distances[sorted_front[i]] += (next_val - prev_val) / range_m

    return distances


def nsga2_tournament_selection(population, ranks, crowding_distances, num_parents, tournament_size=2):
    """
    NSGA-II Crowded Comparison Tournament Selection.
    """
    selected_indices = []
    pop_indices = list(range(len(population)))

    for _ in range(num_parents):
        candidates = random.sample(pop_indices, min(tournament_size, len(pop_indices)))
        winner = candidates[0]
        for c in candidates[1:]:
            if ranks[c] < ranks[winner]:
                winner = c
            elif ranks[c] == ranks[winner]:
                if crowding_distances.get(c, 0) > crowding_distances.get(winner, 0):
                    winner = c
        selected_indices.append(winner)

    return selected_indices


def nsga2_replacement(parent_population, parent_fitness, offspring_population, offspring_fitness, pop_size):
    """
    NSGA-II Environmental Selection (Combining parents + offspring, sorting by Pareto Front & Crowding Distance).
    """
    combined_population = parent_population + offspring_population
    combined_fitness = parent_fitness + offspring_fitness

    fronts, ranks = fast_non_dominated_sort(combined_fitness)

    new_population = []
    new_fitness = []
    new_crowding = {}
    new_ranks = {}

    for front in fronts:
        front_crowding = calculate_crowding_distance(front, combined_fitness)

        if len(new_population) + len(front) <= pop_size:
            for idx in front:
                new_population.append(combined_population[idx])
                new_fitness.append(combined_fitness[idx])
                new_idx = len(new_population) - 1
                new_crowding[new_idx] = front_crowding[idx]
                new_ranks[new_idx] = ranks[idx]
        else:
            remaining_slots = pop_size - len(new_population)
            sorted_front = sorted(front, key=lambda idx: front_crowding[idx], reverse=True)
            for idx in sorted_front[:remaining_slots]:
                new_population.append(combined_population[idx])
                new_fitness.append(combined_fitness[idx])
                new_idx = len(new_population) - 1
                new_crowding[new_idx] = front_crowding[idx]
                new_ranks[new_idx] = ranks[idx]
            break

    return new_population, new_fitness, new_ranks, new_crowding


def crossover(parent1, parent2, productivity_map, proximity_map, cost_map, budget, generation):
    """
    Performs domain-specific multi-aspect crossover combining subgroups.
    """
    subgroup_coords1, _ = subgroups(parent1.copy())
    subgroup_coords2, _ = subgroups(parent2.copy())

    compactness1 = calculate_compactness(subgroup_coords1)
    compactness2 = calculate_compactness(subgroup_coords2)
    compactness_total = compactness1 + compactness2
    subgroup_coords_total = subgroup_coords1 + subgroup_coords2

    # Child 1: Combine subgroups based on compactness
    sorted_subgroups = [subgroup for _, subgroup in sorted(zip(compactness_total, subgroup_coords_total), reverse=True)]
    child1 = np.zeros_like(parent1)
    child1_cost = 0
    for subgroup in sorted_subgroups:
        for coord in subgroup:
            if child1_cost + cost_map[coord] <= budget:
                if child1[coord] == 0:
                    child1[coord] = parent1[coord] if subgroup in subgroup_coords1 else parent2[coord]
                    child1_cost += cost_map[coord]
            else:
                break

    # Child 2: Combine subgroups based on proximity
    proximity1 = calculate_pro(subgroup_coords1, proximity_map)
    proximity2 = calculate_pro(subgroup_coords2, proximity_map)
    proximity_total = proximity1 + proximity2
    sorted_subgroups = [subgroup for _, subgroup in sorted(zip(proximity_total, subgroup_coords_total))]
    child2 = np.zeros_like(parent1)
    child2_cost = 0
    for subgroup in sorted_subgroups:
        for coord in subgroup:
            if child2_cost + cost_map[coord] <= budget:
                if child2[coord] == 0:
                    child2[coord] = parent1[coord] if subgroup in subgroup_coords1 else parent2[coord]
                    child2_cost += cost_map[coord]
            else:
                break

    # Child 3: Combine subgroups based on productivity
    productivity1 = calculate_pro(subgroup_coords1, productivity_map)
    productivity2 = calculate_pro(subgroup_coords2, productivity_map)
    productivity_total = productivity1 + productivity2
    sorted_subgroups = [subgroup for _, subgroup in sorted(zip(productivity_total, subgroup_coords_total), reverse=True)]
    child3 = np.zeros_like(parent1)
    child3_cost = 0
    for subgroup in sorted_subgroups:
        for coord in subgroup:
            if child3_cost + cost_map[coord] <= budget:
                if child3[coord] == 0:
                    child3[coord] = parent1[coord] if subgroup in subgroup_coords1 else parent2[coord]
                    child3_cost += cost_map[coord]
            else:
                break

    return child1, child2, child3


def mutate(individual, Decision_Map, mutation_prob, cost_map, budget):
    mutated_individual = individual.copy()
    if np.random.rand() < mutation_prob:
        i = np.random.randint(len(mutated_individual))
        j = np.random.randint(len(mutated_individual[0]))
        if Decision_Map[i, j] == 1:
            mutated_individual[i, j] = np.random.choice([0, 2])
            sum_cost = np.sum(cost_map[(mutated_individual == 2)])
            if sum_cost <= budget:
                mutated_individual[i, j] = 2
        else:
            mutated_individual[i, j] = 0

    return mutated_individual


def multi_objective_ga(Map, cost_map, productivity_map, proximity_map, budget=500, pop_size=50, num_generations=35, tournament_max_size=4, mutation_prob=0.1):
    if pop_size % 2 != 0:
        pop_size = pop_size + 1

    population = generate_population(Map, cost_map, pop_size, budget)

    # Evaluate fitness of each member of initial population
    fitness_values = [evaluate_individual(ind, proximity_map, productivity_map, cost_map, budget) for ind in population]

    fronts, ranks = fast_non_dominated_sort(fitness_values)
    crowding_distances = {}
    for front in fronts:
        crowding_distances.update(calculate_crowding_distance(front, fitness_values))

    all_solutions = list(population)
    history_per_gen = [{
        'generation': 0,
        'population': list(population),
        'fitness_values': list(fitness_values),
        'pareto_front': [population[idx] for idx in fronts[0]] if fronts else [],
        'pareto_fitness': [fitness_values[idx] for idx in fronts[0]] if fronts else []
    }]

    for i in range(num_generations):
        print("Generation :", i + 1, "/", num_generations)

        # Select parents using NSGA-II tournament
        parents_selected = nsga2_tournament_selection(population, ranks, crowding_distances, pop_size, tournament_size=tournament_max_size)

        offspring = []
        for j in range(0, pop_size, 2):
            parent1 = population[parents_selected[j]]
            parent2 = population[parents_selected[j + 1]]

            if np.array_equal(parent1, parent2):
                continue

            children = crossover(parent1, parent2, productivity_map, proximity_map, cost_map, budget, i)
            for child in children:
                if not np.all(child == 0) and np.sum(cost_map[(child == 2)]) <= budget:
                    offspring.append(child)

        # Mutate offspring
        for k in range(len(offspring)):
            mutated_individual = mutate(offspring[k], Map, mutation_prob, cost_map, budget)
            if np.sum(cost_map[(mutated_individual == 2)]) <= budget:
                offspring[k] = mutated_individual

        # Evaluate fitness of offspring
        offspring_fitness = [evaluate_individual(ind, proximity_map, productivity_map, cost_map, budget) for ind in offspring]

        # NSGA-II Replacement
        population, fitness_values, ranks, crowding_distances = nsga2_replacement(
            population, fitness_values, offspring, offspring_fitness, pop_size
        )

        all_solutions.extend(population)
        
        # Save snapshot for GIF visualization
        cur_fronts, _ = fast_non_dominated_sort(fitness_values)
        history_per_gen.append({
            'generation': i + 1,
            'population': list(population),
            'fitness_values': list(fitness_values),
            'pareto_front': [population[idx] for idx in cur_fronts[0]] if cur_fronts else [],
            'pareto_fitness': [fitness_values[idx] for idx in cur_fronts[0]] if cur_fronts else []
        })

    return all_solutions, history_per_gen

