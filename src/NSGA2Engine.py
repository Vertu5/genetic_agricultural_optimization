# ==============================================================================
# 🌾 Project: Genetic Agricultural Optimization (NSGA-II + PROMETHEE II)
# 👨‍💻 Author: Olivier Vertu Ndingaoba
# 🌐 Portfolio: https://ndingaoba-oliviervertu.vercel.app/
# 📅 Date: August 2026
# 📝 Description: Moteur algorithmique génétique multi-objectif NSGA-II
#
# Ce code est le moteur back-end d'optimisation spatiale. Il a été conçu, 
# architecturé et développé de A à Z par mes soins pour être intégré 
# comme API Serverless sur ma plateforme interactive.
# ==============================================================================

import random
import numpy as np

class NSGA2Engine:
    """
    Moteur de recherche génétique multi-objectif basé sur NSGA-II :
    - Tri non-dominé rapide (Fast Non-Dominated Sorting)
    - Distance de tassement (Crowding Distance)
    - Sélection par tournoi encombré (Crowded Comparison Tournament)
    - Croisement et Mutation spatiaux sous contrainte budgétaire
    """

    def __init__(self, map_manager, fitness_evaluator, pop_size=30, num_generations=20, budget=None):
        self.map_manager = map_manager
        self.fitness_evaluator = fitness_evaluator
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.budget = float(budget) if budget is not None else float(map_manager.budget)

    def generate_initial_population(self):
        population = []
        cultivable = self.map_manager.cultivable_cells
        if not cultivable:
            return [np.zeros((self.map_manager.rows, self.map_manager.cols)) for _ in range(self.pop_size)]

        for _ in range(self.pop_size):
            individual = np.zeros((self.map_manager.rows, self.map_manager.cols), dtype=int)
            shuffled = list(cultivable)
            random.shuffle(shuffled)
            current_cost = 0.0

            for r, c in shuffled:
                cell_cost = self.map_manager.cost_map[r, c] if self.map_manager.cost_map is not None else 1.0
                if current_cost + cell_cost <= self.budget:
                    individual[r, c] = 2
                    current_cost += cell_cost
                if current_cost >= self.budget:
                    break

            population.append(individual)
        return population

    def pareto_dominates(self, fit1, fit2):
        c1, p1, r1 = fit1
        c2, p2, r2 = fit2
        not_worse = (c1 <= c2) and (p1 <= p2) and (r1 >= r2)
        strictly_better = (c1 < c2) or (p1 < p2) or (r1 > r2)
        return not_worse and strictly_better

    def fast_non_dominated_sort(self, population_fitness):
        pop_len = len(population_fitness)
        S = [[] for _ in range(pop_len)]
        n = [0] * pop_len
        fronts = [[]]
        ranks = {}

        for i in range(pop_len):
            for j in range(pop_len):
                if i == j:
                    continue
                if self.pareto_dominates(population_fitness[i], population_fitness[j]):
                    S[i].append(j)
                elif self.pareto_dominates(population_fitness[j], population_fitness[i]):
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

    def calculate_crowding_distance(self, front_indices, population_fitness):
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

    def tournament_selection(self, population, ranks, crowding_distances):
        i, j = random.sample(range(len(population)), 2)
        if ranks[i] < ranks[j]:
            return population[i]
        elif ranks[j] < ranks[i]:
            return population[j]
        else:
            return population[i] if crowding_distances[i] >= crowding_distances[j] else population[j]

    def crossover(self, parent1, parent2):
        child = np.zeros_like(parent1)
        mask = np.random.rand(*parent1.shape) < 0.5
        child[mask] = parent1[mask]
        child[~mask] = parent2[~mask]
        return self._repair_budget(child)

    def mutate(self, individual, mutation_rate=0.1):
        mutated = individual.copy()
        cultivable = self.map_manager.cultivable_cells
        for r, c in cultivable:
            if random.random() < mutation_rate:
                mutated[r, c] = 2 if mutated[r, c] == 0 else 0
        return self._repair_budget(mutated)

    def _repair_budget(self, individual):
        if self.map_manager.cost_map is None:
            return individual
        selected = np.argwhere(individual == 2)
        current_cost = np.sum(self.map_manager.cost_map[individual == 2])
        if current_cost > self.budget:
            indices = list(range(len(selected)))
            random.shuffle(indices)
            for idx in indices:
                r, c = selected[idx]
                individual[r, c] = 0
                current_cost -= self.map_manager.cost_map[r, c]
                if current_cost <= self.budget:
                    break
        return individual

    def run(self):
        population = self.generate_initial_population()
        fitnesses = [self.fitness_evaluator.evaluate(ind, self.budget) for ind in population]

        for gen in range(self.num_generations):
            fronts, ranks = self.fast_non_dominated_sort(fitnesses)
            crowding_distances = {}
            for front in fronts:
                cd = self.calculate_crowding_distance(front, fitnesses)
                crowding_distances.update(cd)

            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self.tournament_selection(population, ranks, crowding_distances)
                p2 = self.tournament_selection(population, ranks, crowding_distances)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                offspring.append(child)

            combined_pop = population + offspring
            combined_fitnesses = [self.fitness_evaluator.evaluate(ind, self.budget) for ind in combined_pop]

            combined_fronts, combined_ranks = self.fast_non_dominated_sort(combined_fitnesses)

            new_population = []
            new_fitnesses = []

            for front in combined_fronts:
                cd = self.calculate_crowding_distance(front, combined_fitnesses)
                if len(new_population) + len(front) <= self.pop_size:
                    for idx in front:
                        new_population.append(combined_pop[idx])
                        new_fitnesses.append(combined_fitnesses[idx])
                else:
                    sorted_front = sorted(front, key=lambda idx: cd[idx], reverse=True)
                    needed = self.pop_size - len(new_population)
                    for idx in sorted_front[:needed]:
                        new_population.append(combined_pop[idx])
                        new_fitnesses.append(combined_fitnesses[idx])
                    break

            population = new_population
            fitnesses = new_fitnesses

        fronts, _ = self.fast_non_dominated_sort(fitnesses)
        pareto_population = [population[i] for i in fronts[0]]
        pareto_fitnesses = [fitnesses[i] for i in fronts[0]]

        return pareto_population, pareto_fitnesses
