# -*- coding: utf-8 -*-
import os
import glob
import random
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.distance import cdist

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

# Visualisation des cartes d'entrée
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
best_solutions, history_per_gen = multi_objective_ga(
    DecisionMap, 
    cost_map, 
    productivity_map, 
    proximity_map, 
    budget=budget_limit, 
    pop_size=40, 
    num_generations=20
)

# 3. Évaluation des objectifs physiques bruts pour chaque solution trouvée
global_obj = []
for solution in best_solutions:
    subgroup_coords, _ = subgroups(solution.copy()) 
    global_obj.append(calculate_global_objectives(subgroup_coords, proximity_map, productivity_map))

# 4. Calcul de la frontière de Pareto avec Tri par Dominance Directe (NSGA-II)
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

# 5. Classement des solutions par la méthode PROMETHEE II
# Poids: [Compacité C (Min), Proximité P (Min), Productivité R (Max)]
weights = np.array([0.2, 0.4, 0.4])
indices = promethee(data_unique, weights)

sorted_data = [data_unique[i] for i in indices]
sorted_pareto = [indices_pareto[i] for i in indices]

# Sauvegarde des résultats Pareto dans pareto.csv (Valeurs physiques brutes)
headers = ["compactness_C", "proximity_P", "productivity_R"]
with open("pareto.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(data_unique)

print(f"Frontière de Pareto extraite: {len(data_unique)} solutions non-dominées enregistrées dans pareto.csv.")

# 6. Génération des GIFs Animés (Évolution Spatiale & Convergence Pareto)
print("--- Génération des animations GIF ---")
temp_dir = "temp_frames"
os.makedirs(temp_dir, exist_ok=True)

spatial_frames = []
pareto_frames = []

for step in history_per_gen:
    gen = step['generation']
    pareto_front = step['pareto_front']
    pareto_fits = step['pareto_fitness']

    # --- Frame 1: Évolution de la Carte Spatiale ---
    fig_sp, ax_sp = plt.subplots(figsize=(8, 6))
    if pareto_front:
        top_sol = pareto_front[0]
        ax_sp.imshow(top_sol, cmap='coolwarm')
    else:
        ax_sp.imshow(Map, cmap='coolwarm')
    ax_sp.set_title(f"NSGA-II - Génération {gen:02d} : Configuration Spatiale")
    sp_path = os.path.join(temp_dir, f"spatial_{gen:02d}.png")
    plt.tight_layout()
    plt.savefig(sp_path)
    plt.close(fig_sp)
    spatial_frames.append(sp_path)

    # --- Frame 2: Convergence du Front de Pareto ---
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    if pareto_fits:
        px = [f[0] for f in pareto_fits] # Compactness C (Min)
        py = [f[1] for f in pareto_fits] # Proximity P (Min)
        pz = [f[2] for f in pareto_fits] # Productivity R (Max)
        sc = ax_pr.scatter(px, py, c=pz, cmap='viridis', s=60, edgecolors='k')
        plt.colorbar(sc, ax=ax_pr, label='Productivité R (Max)')
    ax_pr.set_xlabel('Compacité C (Min - périmètre²/surface)')
    ax_pr.set_ylabel('Proximité P (Min - distance)')
    ax_pr.set_title(f"NSGA-II - Génération {gen:02d} : Convergence Front de Pareto")
    pr_path = os.path.join(temp_dir, f"pareto_{gen:02d}.png")
    plt.tight_layout()
    plt.savefig(pr_path)
    plt.close(fig_pr)
    pareto_frames.append(pr_path)

# Assemblage des fichiers GIF avec PIL
def build_gif(frame_list, output_name, duration=350):
    images = [Image.open(f) for f in frame_list]
    if images:
        images[0].save(output_name, save_all=True, append_images=images[1:], duration=duration, loop=0)
        print(f"GIF créé avec succès : {output_name}")

build_gif(spatial_frames, "spatial_evolution.gif")
build_gif(pareto_frames, "pareto_convergence.gif")

# Nettoyage des images temporaires
for f in spatial_frames + pareto_frames:
    if os.path.exists(f):
        os.remove(f)
if os.path.exists(temp_dir):
    os.rmdir(temp_dir)

# 7. Visualisation finale statique de la Frontière de Pareto
sorted_x = [p[0] for p in sorted_data] # Compactness C
sorted_y = [p[1] for p in sorted_data] # Proximity P
sorted_z = [p[2] for p in sorted_data] # Productivity R

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
sc0 = axs[0, 0].scatter(sorted_x, sorted_y, c=sorted_z, cmap='viridis', s=50)
axs[0, 0].set_xlabel('Compacité C (Min)')
axs[0, 0].set_ylabel('Proximité P (Min)')
axs[0, 0].set_title('Proximité vs Compacité')
plt.colorbar(sc0, ax=axs[0, 0])

sc1 = axs[0, 1].scatter(sorted_x, sorted_z, c=sorted_y, cmap='plasma', s=50)
axs[0, 1].set_xlabel('Compacité C (Min)')
axs[0, 1].set_ylabel('Productivité R (Max)')
axs[0, 1].set_title('Productivité vs Compacité')
plt.colorbar(sc1, ax=axs[0, 1])

sc2 = axs[1, 0].scatter(sorted_y, sorted_z, c=sorted_x, cmap='magma', s=50)
axs[1, 0].set_xlabel('Proximité P (Min)')
axs[1, 0].set_ylabel('Productivité R (Max)')
axs[1, 0].set_title('Productivité vs Proximité')
plt.colorbar(sc2, ax=axs[1, 0])

axs[1, 1].plot(range(len(sorted_data)), sorted_x, label='Compacité C (Min)', marker='o')
axs[1, 1].plot(range(len(sorted_data)), sorted_y, label='Proximité P (Min)', marker='s')
axs[1, 1].plot(range(len(sorted_data)), sorted_z, label='Productivité R (Max)', marker='^')
axs[1, 1].set_xlabel('Rang PROMETHEE II')
axs[1, 1].set_ylabel('Valeur Physique Brute')
axs[1, 1].set_title('Solutions Pareto classées (PROMETHEE II)')
axs[1, 1].legend()

plt.tight_layout()
plt.savefig("pareto_frontier.png")
plt.close()

print("Execution terminee avec succes! Animations GIF et figures générées.")
