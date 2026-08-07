# -*- coding: utf-8 -*-
import sys
import site
try:
    import mpl_toolkits
    site_packages = site.getusersitepackages()
    if site_packages + '/mpl_toolkits' not in mpl_toolkits.__path__:
        mpl_toolkits.__path__.insert(0, site_packages + '/mpl_toolkits')
    from mpl_toolkits.mplot3d import Axes3D
except Exception:
    pass

import os
import glob
import random
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.distance import cdist
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lang', type=str, default='fr', choices=['fr', 'en'])
args = parser.parse_args()
LANG = args.lang

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

from mapfunctions import *
from utils import *
from evaluation import * 
from genetic_algo import *
from prometh import *

np.random.seed(2)
random.seed(2)

# Paths resolution (works regardless of CWD)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", LANG)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get custom colormap and discrete legend patches
cmap_land, patches_land = get_custom_colormap_and_legend(lang=LANG)

# 1. Modelisation
usage_map_path = os.path.join(DATA_DIR, "Usage_map.txt")
cost_map_path = os.path.join(DATA_DIR, "Cost_map.txt")
prod_map_path = os.path.join(DATA_DIR, "Production_map.txt")

Map = create_map(read_file(usage_map_path))
cost_map = read_ProdCost(cost_map_path, Map)
productivity_map = read_ProdCost(prod_map_path, Map)

# Find the indices of the cells with values 1 and 2
indices_1 = np.argwhere(Map == 1)
indices_2 = np.argwhere(Map == 2)

if len(indices_1) > 0 and len(indices_2) > 0:
    distances = cdist(indices_1, indices_2)
    # axis=1 : Distance minimale entre CHAQUE candidat (1) et le point existant (2) le plus proche
    min_distances = np.min(distances, axis=1) 
    
    proximity_map = np.ones_like(Map, dtype=float) * 9.0 # Proximité par défaut (très mauvaise)
    
    min_val = np.min(min_distances)
    max_val = np.max(min_distances)
    
    # Interpolation uniquement sur les zones candidates
    if max_val > min_val:
        proximity_map[Map == 1] = np.interp(min_distances, (min_val, max_val), (1, 9))
    else:
        proximity_map[Map == 1] = 1.0
else:
    proximity_map = np.ones_like(Map, dtype=float) * 9.0

# Decision variable
budget_limit = 1000
DecisionMap = Decision_map(read_file(usage_map_path))
DecisionMap = preprocess(DecisionMap, productivity_map, proximity_map, cost_map, budget_limit)
Min_prox = select_proximity(proximity_map, DecisionMap, cost_map, budget=budget_limit)
Max_prod = select_productivity(productivity_map, DecisionMap, cost_map, budget=budget_limit)

# Visualisation des cartes d'entrée avec légende explicite
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# 4-state visual map for base Usage Map
vis_base = get_visualization_map(Map)
axs[0, 0].imshow(vis_base, cmap=cmap_land, vmin=0, vmax=3)
axs[0, 0].set_title("Land Use Map" if LANG == 'en' else "Carte d'Occupation des Sols")
axs[0, 0].legend(handles=patches_land[:3], loc='lower center', bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=12)

sc_prox = axs[0, 1].imshow(proximity_map, cmap="viridis")
axs[0, 1].set_title("Proximity Map" if LANG == 'en' else "Carte de Proximité")
plt.colorbar(sc_prox, ax=axs[0, 1], label="Proximity (1=Close, 9=Far)" if LANG == 'en' else "Proximité (1=Proche, 9=Éloigné)")

sc_prod = axs[1, 0].imshow(productivity_map, cmap="YlGn")
axs[1, 0].set_title("Productivity Map" if LANG == 'en' else "Carte de Productivité")
plt.colorbar(sc_prod, ax=axs[1, 0], label="Productivity (R)" if LANG == 'en' else "Productivité (R)")

sc_cost = axs[1, 1].imshow(cost_map, cmap="YlOrRd")
axs[1, 1].set_title("Land Cost Map" if LANG == 'en' else "Carte de Coût d'Acquisition")
plt.colorbar(sc_cost, ax=axs[1, 1], label="Cost (units)" if LANG == 'en' else "Coût (en unités)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "input_maps.png"), dpi=300)
plt.close()

# 2. Execution de l'algorithme génétique multi-objectifs (NSGA-II)
print("--- Lancement du GA Multi-Objectifs (NSGA-II) ---")
best_solutions, history_per_gen = multi_objective_ga(
    DecisionMap, 
    cost_map, 
    productivity_map, 
    proximity_map, 
    budget=budget_limit, 
    pop_size=100, 
    num_generations=100
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
pareto_csv_path = os.path.join(OUTPUT_DIR, "pareto.csv")
headers = ["compactness_C", "proximity_P", "productivity_R"]
with open(pareto_csv_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(data_unique)

print(f"Frontière de Pareto extraite: {len(data_unique)} solutions non-dominées enregistrées dans {pareto_csv_path}.")

# 6. Génération des GIFs Animés (Évolution Spatiale & Convergence Pareto)
print("--- Génération des animations GIF ---")
temp_dir = os.path.join(OUTPUT_DIR, "temp_frames")
os.makedirs(temp_dir, exist_ok=True)

spatial_frames = []
pareto_frames = []

for step in history_per_gen:
    gen = step['generation']
    pareto_front = step['pareto_front']
    pareto_fits = step['pareto_fitness']

    # --- Frame 1: Évolution de la Carte Spatiale avec Légende ---
    fig_sp, ax_sp = plt.subplots(figsize=(10, 8))
    if pareto_front:
        top_sol = pareto_front[0]
        vis_sol = get_visualization_map(Map, top_sol)
        ax_sp.imshow(vis_sol, cmap=cmap_land, vmin=0, vmax=3)
    else:
        vis_base = get_visualization_map(Map)
        ax_sp.imshow(vis_base, cmap=cmap_land, vmin=0, vmax=3)
    
    title = f"NSGA-II - Generation {gen:02d}: Spatial Allocation" if LANG == 'en' else f"NSGA-II - Génération {gen:02d} : Allocation Spatiale"
    ax_sp.set_title(title, fontsize=16, fontweight='bold')
    ax_sp.legend(handles=patches_land, loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=14)
    sp_path = os.path.join(temp_dir, f"spatial_{gen:02d}.png")
    plt.tight_layout()
    plt.savefig(sp_path, dpi=120)
    plt.close(fig_sp)
    spatial_frames.append(sp_path)

    # --- Frame 2: Convergence du Front de Pareto ---
    fig_pr, ax_pr = plt.subplots(figsize=(10, 8))
    if pareto_fits:
        px = [f[0] for f in pareto_fits] # Compactness C (Min)
        py = [f[1] for f in pareto_fits] # Proximity P (Min)
        pz = [f[2] for f in pareto_fits] # Productivity R (Max)
        sc = ax_pr.scatter(px, py, c=pz, cmap='viridis', s=80, edgecolors='k')
        plt.colorbar(sc, ax=ax_pr, label='Productivity R (Max)' if LANG == 'en' else 'Productivité R (Max)')
        
    ax_pr.set_xlabel('Compactness C (Min - perim²/area)' if LANG == 'en' else 'Compacité C (Min - périmètre²/surface)')
    ax_pr.set_ylabel('Proximity P (Min - dist)' if LANG == 'en' else 'Proximité P (Min - distance)')
    title_pr = f"NSGA-II - Gen {gen:02d}: Pareto Convergence" if LANG == 'en' else f"NSGA-II - Génération {gen:02d} : Convergence Pareto"
    ax_pr.set_title(title_pr, fontsize=16)
    pr_path = os.path.join(temp_dir, f"pareto_{gen:02d}.png")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=120)
    plt.close(fig_pr)
    pareto_frames.append(pr_path)

# Assemblage des fichiers GIF avec PIL
def build_gif(frame_list, output_name, duration=350):
    images = [Image.open(f) for f in frame_list]
    if images:
        out_path = os.path.join(OUTPUT_DIR, output_name)
        images[0].save(out_path, save_all=True, append_images=images[1:], duration=duration, loop=0)
        print(f"GIF créé avec succès : {out_path}")

build_gif(spatial_frames, "spatial_evolution.gif")
build_gif(pareto_frames, "pareto_convergence.gif")

# Nettoyage des images temporaires
for f in spatial_frames + pareto_frames:
    if os.path.exists(f):
        os.remove(f)
if os.path.exists(temp_dir):
    os.rmdir(temp_dir)

# 7. Visualisation finale statique 2D de la Frontière de Pareto
sorted_x = [p[0] for p in sorted_data] # Compactness C
sorted_y = [p[1] for p in sorted_data] # Proximity P
sorted_z = [p[2] for p in sorted_data] # Productivity R

fig, axs = plt.subplots(2, 2, figsize=(16, 12))
sc0 = axs[0, 0].scatter(sorted_x, sorted_y, c=sorted_z, cmap='viridis', s=70)
axs[0, 0].set_xlabel('Compactness C (Min)' if LANG == 'en' else 'Compacité C (Min)')
axs[0, 0].set_ylabel('Proximity P (Min)' if LANG == 'en' else 'Proximité P (Min)')
axs[0, 0].set_title('Proximity vs Compactness' if LANG == 'en' else 'Proximité vs Compacité')
plt.colorbar(sc0, ax=axs[0, 0])

sc1 = axs[0, 1].scatter(sorted_x, sorted_z, c=sorted_y, cmap='plasma', s=70)
axs[0, 1].set_xlabel('Compactness C (Min)' if LANG == 'en' else 'Compacité C (Min)')
axs[0, 1].set_ylabel('Productivity R (Max)' if LANG == 'en' else 'Productivité R (Max)')
axs[0, 1].set_title('Productivity vs Compactness' if LANG == 'en' else 'Productivité vs Compacité')
plt.colorbar(sc1, ax=axs[0, 1])

sc2 = axs[1, 0].scatter(sorted_y, sorted_z, c=sorted_x, cmap='magma', s=70)
axs[1, 0].set_xlabel('Proximity P (Min)' if LANG == 'en' else 'Proximité P (Min)')
axs[1, 0].set_ylabel('Productivity R (Max)' if LANG == 'en' else 'Productivité R (Max)')
axs[1, 0].set_title('Productivity vs Proximity' if LANG == 'en' else 'Productivité vs Proximité')
plt.colorbar(sc2, ax=axs[1, 0])

axs[1, 1].plot(range(len(sorted_data)), sorted_x, label='Compactness C' if LANG == 'en' else 'Compacité C', marker='o')
axs[1, 1].plot(range(len(sorted_data)), sorted_y, label='Proximity P' if LANG == 'en' else 'Proximité P', marker='s')
axs[1, 1].plot(range(len(sorted_data)), sorted_z, label='Productivity R' if LANG == 'en' else 'Productivité R', marker='^')
axs[1, 1].set_xlabel('PROMETHEE II Rank' if LANG == 'en' else 'Rang PROMETHEE II')
axs[1, 1].set_ylabel('Raw Physical Value' if LANG == 'en' else 'Valeur Physique Brute')
axs[1, 1].set_title('Ranked Pareto Solutions' if LANG == 'en' else 'Solutions Pareto classées')
axs[1, 1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pareto_frontier.png"), dpi=300)
plt.close()

# 8. Verification & 3D Plot / Pareto Tour Generation
from verify_solutions import run_verification
run_verification(best_solutions, lang=LANG)

print("Execution terminee avec succes! Animations GIF, surface 3D et rapport générés.")
