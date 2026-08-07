# -*- coding: utf-8 -*-
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

def read_ProdCost(filename, Map=None):
    """
    Reads a grid map text file containing numeric values into a numpy array.
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    
    matrix = []
    for line in lines:
        line_str = line.strip()
        if line_str:
            row = [float(char) for char in line_str if char.isdigit()]
            matrix.append(row)
            
    matrix = np.array(matrix, dtype=float)
    return matrix

def Decision_map(data):
    """
    Constructs a DecisionMap matrix where:
      0 = Restricted (R)
      1 = Cultivable candidate (C)
      2 = Existing agricultural land (A)
    """
    if isinstance(data, list):
        Map = np.zeros((len(data), len(data[0])))
        for i in range(len(data)):
            for j in range(len(data[0])):
                if data[i][j] == "R":
                    Map[i][j] = 0
                elif data[i][j] == "C":
                    Map[i][j] = 1
                else:   
                    Map[i][j] = 2
        return Map
    elif isinstance(data, np.ndarray):
        return data.copy()
    return data

def preprocess(DecisionMap, productivity_map, proximity_map, cost_map=None, budget=None):
    """
    Preprocesses the decision map by filtering strictly unviable candidate cells.
    Deactivates candidates (sets to 0) if productivity <= 0 or cost > budget.
    """
    optimized_map = DecisionMap.copy()
    
    if cost_map is None or budget is None:
        return optimized_map

    # Vectorized filtering of candidate cells (C = 1)
    unviable_mask = (optimized_map == 1) & ((productivity_map <= 0) | (cost_map > budget))
    optimized_map[unviable_mask] = 0
    
    filtered_count = np.sum(unviable_mask)
    if filtered_count > 0:
        print(f"[Preprocessing] Filtered out {filtered_count} unviable candidate cells (sterile or over budget).")
        
    return optimized_map

def get_visualization_map(base_map, solution_map=None):
    """
    Creates a 4-state visualization matrix:
      0 = Restricted (R) -> Charcoal Grey
      1 = Cultivable Candidate (C - Available) -> Sky Blue
      2 = Existing Agricultural (A - Owned) -> Forest Green
      3 = Newly Bought Parcel (Extension) -> Gold / Orange
    """
    vis_map = np.zeros_like(base_map, dtype=int)
    vis_map[base_map == 0] = 0
    vis_map[base_map == 1] = 1
    vis_map[base_map == 2] = 2
    
    if solution_map is not None:
        vis_map[(base_map == 1) & (solution_map == 2)] = 3
        
    return vis_map

def get_custom_colormap_and_legend(lang='fr'):
    """
    Returns discrete colormap and patches for land classification legend.
    """
    colors = ['#374151', '#BAE6FD', '#15803D', '#F59E0B']
    cmap = ListedColormap(colors)
    
    if lang == 'en':
        patches = [
            mpatches.Patch(color='#374151', label='Restricted R (Not purchasable)'),
            mpatches.Patch(color='#BAE6FD', label='Candidate C (Available)'),
            mpatches.Patch(color='#15803D', label='Existing Farm A (Owned)'),
            mpatches.Patch(color='#F59E0B', label='New Extension (Purchased)')
        ]
    else:
        patches = [
            mpatches.Patch(color='#374151', label='Restreint R (Non achetable)'),
            mpatches.Patch(color='#BAE6FD', label='Candidate C (Disponible)'),
            mpatches.Patch(color='#15803D', label='Ferme Existante A (Possédée)'),
            mpatches.Patch(color='#F59E0B', label='Nouvelle Extension (Achetée)')
        ]
    return cmap, patches
