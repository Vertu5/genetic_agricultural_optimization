# -*- coding: utf-8 -*-
import random
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata

from mapfunctions import *
from utils import *
from evaluation import * 
from genetic_algo import *
from prometh import *

np.random.seed(2)
random.seed(2)

# 1. Modelisation
Map = create_map(read_file("Usage_map.txt"))
cost_map = read_ProdCost("Cost_map.txt", Map)
productivity_map = read_ProdCost("Production_map.txt", Map)

# Find the indices of the cells with values 1 and 2
indices_1 = np.argwhere(Map == 1)
indices_2 = np.argwhere(Map == 2)

if len(indices_1) > 0 and len(indices_2) > 0:
    distances = cdist(indices_1, indices_2)
    min_distances = np.min(distances, axis=0)
    new_map = np.copy(Map)
    new_map[new_map == 2] = min_distances
    min_val = np.min(new_map)
    max_val = np.max(new_map)
    proximity_map = np.interp(new_map, (min_val, max_val), (1, 9)).astype(float)
else:
    proximity_map = np.ones_like(Map, dtype=float)

# Decision variable
DecisionMap = Decision_map(read_file("Usage_map.txt"))
DecisionMap = preprocess(DecisionMap, productivity_map, proximity_map)

budget_limit = 500
Min_prox = select_proximity(proximity_map, DecisionMap, cost_map, budget=budget_limit)
Max_prod = select_productivity(productivity_map, DecisionMap, cost_map, budget=budget_limit)

# Visualisation des cartes
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs[0, 0].imshow(Map, cmap="coolwarm")
axs[0, 0].set_title("Usage Map")

axs[0, 1].imshow(proximity_map, cmap="coolwarm")
axs[0, 1].set_title("Proximity Map")

axs[1, 0].imshow(productivity_map, cmap="coolwarm")
axs[1, 0].set_title("Productivity Map")

axs[1, 1].imshow(cost_map, cmap="coolwarm")
axs[1, 1].set_title("Cost Map")

plt.tight_layout()
plt.savefig("input_maps.png")
plt.close()

# 2. Execution de l'algorithme génétique multi-objectifs (NSGA-II)
print("--- Lancement du GA Multi-Objectifs (NSGA-II) ---")
best_solutions = multi_objective_ga(
    DecisionMap, 
    cost_map, 
    productivity_map, 
    proximity_map, 
    budget=budget_limit, 
    pop_size=40, 
    num_generations=20
)

# 3. Évaluation des objectifs pour chaque solution trouvée
global_obj = []
for solution in best_solutions:
    subgroup_coords, _ = subgroups(solution.copy()) 
    global_obj.append(calculate_global_objectives(subgroup_coords, proximity_map, productivity_map))

# 4. Calcul de la frontière de Pareto avec Tri par Dominance (NSGA-II)
fronts, _ = fast_non_dominated_sort(global_obj)
if fronts:
    indices_pareto = fronts[0]
else:
    indices_pareto = list(range(len(global_obj)))

pareto_points = [global_obj[i] for i in indices_pareto]
data = np.array(pareto_points).reshape(-1, 3)

# Suppression des doublons
data_unique, unique_indices = np.unique(data, axis=0, return_index=True)
indices_pareto = [indices_pareto[idx] for idx in unique_indices]

pareto_x = [p[0] for p in data_unique]
pareto_y = [p[1] for p in data_unique]
pareto_z = [p[2] for p in data_unique]

# 5. Classement des solutions par la méthode PROMETHEE II
weights = np.array([0.2, 0.4, 0.4]) # [Compactness, Proximity, Productivity]
indices = promethee(data_unique, weights)

sorted_data = [data_unique[i] for i in indices]
sorted_pareto = [indices_pareto[i] for i in indices]

# Sauvegarde des résultats Pareto dans pareto.csv
headers = ["compactness_score", "proximity_score", "productivity_score"]
with open("pareto.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(data_unique)

print(f"Frontière de Pareto extraite: {len(data_unique)} solutions non-dominées enregistrées dans pareto.csv.")

# 6. Visualisation de la Frontière de Pareto
sorted_x = [p[0] for p in sorted_data]
sorted_y = [p[1] for p in sorted_data]
sorted_z = [p[2] for p in sorted_data]

try:
    import mpl_toolkits.mplot3d
    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.scatter3D(sorted_x, sorted_y, sorted_z, c=range(len(sorted_data)), cmap='coolwarm', s=50)
    ax1.set_xlabel('1 / Compactness')
    ax1.set_ylabel('1 / Proximity')
    ax1.set_zlabel('Productivity')
    ax1.set_title('Frontière de Pareto - Vue 3D')

    ax2 = fig.add_subplot(2, 2, 2)
    sc2 = ax2.scatter(sorted_x, sorted_y, c=sorted_z, cmap='viridis', s=50)
    ax2.set_xlabel('1 / Compactness')
    ax2.set_ylabel('1 / Proximity')
    ax2.set_title('Proximity vs Compactness (Color=Productivity)')
    plt.colorbar(sc2, ax=ax2)

    ax3 = fig.add_subplot(2, 2, 3)
    sc3 = ax3.scatter(sorted_x, sorted_z, c=sorted_y, cmap='plasma', s=50)
    ax3.set_xlabel('1 / Compactness')
    ax3.set_ylabel('Productivity')
    ax3.set_title('Productivity vs Compactness (Color=Proximity)')
    plt.colorbar(sc3, ax=ax3)

    ax4 = fig.add_subplot(2, 2, 4)
    sc4 = ax4.scatter(sorted_y, sorted_z, c=sorted_x, cmap='magma', s=50)
    ax4.set_xlabel('1 / Proximity')
    ax4.set_ylabel('Productivity')
    ax4.set_title('Productivity vs Proximity (Color=Compactness)')
    plt.colorbar(sc4, ax=ax4)
except Exception as e:
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    sc0 = axs[0, 0].scatter(sorted_x, sorted_y, c=sorted_z, cmap='viridis', s=50)
    axs[0, 0].set_xlabel('1 / Compactness')
    axs[0, 0].set_ylabel('1 / Proximity')
    axs[0, 0].set_title('Proximity vs Compactness')
    plt.colorbar(sc0, ax=axs[0, 0])

    sc1 = axs[0, 1].scatter(sorted_x, sorted_z, c=sorted_y, cmap='plasma', s=50)
    axs[0, 1].set_xlabel('1 / Compactness')
    axs[0, 1].set_ylabel('Productivity')
    axs[0, 1].set_title('Productivity vs Compactness')
    plt.colorbar(sc1, ax=axs[0, 1])

    sc2 = axs[1, 0].scatter(sorted_y, sorted_z, c=sorted_x, cmap='magma', s=50)
    axs[1, 0].set_xlabel('1 / Proximity')
    axs[1, 0].set_ylabel('Productivity')
    axs[1, 0].set_title('Productivity vs Proximity')
    plt.colorbar(sc2, ax=axs[1, 0])

    axs[1, 1].plot(range(len(sorted_data)), sorted_x, label='1 / Compactness', marker='o')
    axs[1, 1].plot(range(len(sorted_data)), sorted_y, label='1 / Proximity', marker='s')
    axs[1, 1].plot(range(len(sorted_data)), sorted_z, label='Productivity', marker='^')
    axs[1, 1].set_xlabel('PROMETHEE Rank Index')
    axs[1, 1].set_ylabel('Objective Value')
    axs[1, 1].set_title('Solutions Pareto classées (PROMETHEE II)')
    axs[1, 1].legend()

plt.tight_layout()
plt.savefig("pareto_frontier.png")
plt.close()

print("Execution terminee avec succes! Figures enregistrees sous input_maps.png et pareto_frontier.png.")

