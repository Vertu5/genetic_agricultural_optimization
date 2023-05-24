# -*- coding: utf-8 -*-
from mapfunctions import*
from utils import*
from evaluation import* 
from genetic_algo import*
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
import numpy as np
from prometh import *

np.random.seed(2)
random.seed(2)

# Modelisation
Map = create_map(read_file("Usage_map.txt"))
cost_map = read_ProdCost("Cost_map.txt", Map)
productivity_map = read_ProdCost("Production_map.txt", Map)

# Find the indices of the cells with values 1 and 2
indices_1 = np.argwhere(Map == 1)
indices_2 = np.argwhere(Map == 2)

# Compute the distances between all pairs of cells with values 1 and 2
distances = cdist(indices_1, indices_2)

# Find the minimum distance from each cell with value 2 to a cell with value 1
min_distances = np.min(distances, axis=0)

# Create a new matrix where cells with value 2 are replaced with the minimum distance to a cell with value 1
new_map = np.copy(Map)
new_map[new_map == 2] = min_distances

min_val = np.min(new_map)
max_val = np.max(new_map)
proximity_map = np.interp(new_map, (min_val, max_val), (1, 9)).astype(float)

#Decision variable
DecisionMap = Decision_map(read_file("Usage_map.txt"))
DecisionMap = preprocess(DecisionMap, productivity_map, proximity_map)

Min_prox = select_proximity(proximity_map, DecisionMap, cost_map, budget = 5000)
Max_prod = select_productivity(productivity_map, DecisionMap, cost_map, budget = 5000)

# Création du subplot
fig, axs = plt.subplots(2, 2, figsize=(12, 4))

# Plot de la carte Map
axs[0, 0].imshow(Map, cmap="coolwarm")
axs[0, 0].set_title("MAP")

# Plot de la carte Proximity
axs[0, 1].imshow(proximity_map, cmap="coolwarm")
axs[0, 1].set_title("Proximity MAP")

# Plot de la carte DecisionMap
axs[1, 0].imshow(productivity_map, cmap="coolwarm")
axs[1, 0].set_title("Productivity Map")

axs[1, 1].imshow(cost_map, cmap="coolwarm")
axs[1, 1].set_title("Cost Map")

plt.tight_layout()
plt.show()



# Création du subplot
fig, axs = plt.subplots(2, figsize=(12, 4))

axs[0].imshow(DecisionMap, cmap="coolwarm")
axs[0].set_title("Decision Map")

# Plot de la carte Proximity
axs[1].imshow(Map, cmap="coolwarm")
axs[1].set_title("MAP")

plt.tight_layout()
plt.show()

# Création du subplot
fig, axs = plt.subplots(2, 2, figsize=(12, 4))
# Plot de la carte Map
axs[0, 0].imshow(proximity_map, cmap="coolwarm")
axs[0, 0].set_title("Proximity") 

# Plot de la carte Proximity
axs[0, 1].imshow(productivity_map , cmap="coolwarm")
axs[0, 1].set_title("Productivity Map")

# Plot de la carte DecisionMap
axs[1, 0].imshow(Min_prox , cmap="coolwarm")
axs[1, 0].set_title("Minimize Proximity") 

axs[1, 1].imshow(Max_prod, cmap="coolwarm")
axs[1, 1].set_title("Maximize Productivity")

plt.tight_layout()
plt.show()



best_solutions = multi_objective_ga(DecisionMap, cost_map, productivity_map, proximity_map)

# Show the last 6 solutions in best_solutions using imshow
#for i in range(-6, 0):
#   plt.imshow(best_solutions[i])
#   plt.show()

global_obj = []
data = []
for solution in best_solutions:
    subgroup_coords, _ = subgroups(solution.copy()) 
    global_obj.append(calculate_global_objectives(subgroup_coords, proximity_map, productivity_map))
    # Création de la matrice data en utilisant les valeurs globales de compactness, proximity et productivity
    
# Calcul de la frontière de Pareto
def pareto_frontier(global_objectives):
    """
    This function takes in a list of tuples containing global values for compactness, proximity, and productivity,
    and a list of corresponding fitness scores. The function then returns the Pareto frontier as a list of tuples.

    Parameters:
        global_objectives (list): A list of tuples containing global values for compactness, proximity, and productivity.

    Returns:
        pareto_frontier (list): A list of tuples representing the points on the Pareto frontier.
    """

    pareto_frontier = []
    indices_pareto = []
    pareto_frontier.append(global_objectives[0])

    for i in range(1, len(global_objectives)):
        dominated = False
        for j in range(len(pareto_frontier)):
            if global_objectives[i][0] <= pareto_frontier[j][0] and \
               global_objectives[i][1] <= pareto_frontier[j][1] and \
               global_objectives[i][2] <= pareto_frontier[j][2]:
                dominated = True
                break
            elif global_objectives[i][0] >= pareto_frontier[j][0] and \
                 global_objectives[i][1] >= pareto_frontier[j][1] and \
                 global_objectives[i][2] >= pareto_frontier[j][2]:
                pareto_frontier.remove(pareto_frontier[j])
                break
        if not dominated:
            pareto_frontier.append(global_objectives[i])
            indices_pareto.append(i)
    
    return pareto_frontier, indices_pareto

pareto, indices_pareto = pareto_frontier(global_obj)

data = np.array(pareto).reshape(-1, 3)

data_unique, unique_indices = np.unique(data, axis=0, return_index=True)

indices_pareto = np.delete(indices_pareto, np.setdiff1d(indices_pareto, unique_indices))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

pareto_x = [p[0] for p in data_unique]
pareto_y = [p[1] for p in data_unique]
pareto_z = [p[2] for p in data_unique]

# Calcul des indices de Promethee
weights = np.array([10/100, 90/100, 10/100])
indices = promethee(data_unique, weights)

# Tri des données en fonction des indices de Promethee
sorted_data = [d for _, d in sorted(zip(indices, data_unique))]
sorted_pareto = [d for _, d in sorted(zip(indices_pareto, data_unique))]

plt.imshow(best_solutions[indices_pareto[-1]])
plt.show()

sorted_x = [p[0] for p in sorted_data]
sorted_y = [p[1] for p in sorted_data]
sorted_z = [p[2] for p in sorted_data]

import csv
headers = ["compactness", "proximity", "productivity"]
save_data  = list(zip(pareto_x, pareto_y, pareto_z))

with open("pareto.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(data)

# Création de la grille régulière pour la surface
sgr = np.linspace(min(sorted_x), max(sorted_x), 100)
ygr = np.linspace(min(sorted_y), max(sorted_y), 100)
XX, YY = np.meshgrid(sgr, ygr)

# Création de la fonction interpolante
F = griddata((sorted_x, sorted_y), sorted_z, (XX, YY), method='linear')

# Création de la figure avec les sous-graphiques
fig = plt.figure()

# Sous-graphique 1
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.plot_surface(XX, YY, F, cmap='coolwarm', edgecolor='none')
ax1.scatter3D(sorted_x, sorted_y, sorted_z, c=range(len(sorted_data)), cmap='coolwarm')
ax1.set_title('Vue 1')

# Sous-graphique 2
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.plot_surface(XX, YY, F, cmap='cool', edgecolor='none')
ax2.scatter3D(sorted_x, sorted_y, sorted_z, c=range(len(sorted_data)), cmap='cool')
ax2.view_init(-148, 8)
ax2.set_title('Vue 2')

# Sous-graphique 3
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax3.plot_surface(XX, YY, F, cmap='jet', edgecolor='none')
ax3.scatter3D(sorted_x, sorted_y, sorted_z, c=range(len(sorted_x)), cmap='jet')
ax3.view_init(-180, 8)
ax3.set_title('Vue 3')

# Sous-graphique 4
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
ax4.plot_surface(XX, YY, F, cmap='rainbow', edgecolor='none')
ax4.scatter3D(sorted_x, sorted_y, sorted_z, c=range(len(sorted_x)), cmap='rainbow')
ax4.view_init(-300, 8)
ax4.set_title('Vue 4')

# Ajuster les paramètres de mise en page
plt.tight_layout()

# Afficher la figure
plt.show()
