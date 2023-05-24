import random
import numpy as np
import math
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from collections import deque
import mapfunctions

def read_file(filename):
    with open(filename) as file:
        data = file.read().splitlines()
    return data

#lecture cost_map.txt en matrice
def create_map(data):
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

def subgroups(Map):
    """
    This function takes in a 2D numpy array Map as input, where each element represents the value of a cell in a map. 
        It returns a 2D numpy array with the same dimensions as the input array, where all the connected subgroups of cells 
        with value 2 are replaced with unique values starting from 2.

    Parameters:
        Map (numpy array): A 2D numpy array representing the map.

    Returns:
        Map (numpy array): A 2D numpy array with the same dimensions as the input array, 
            where all the connected subgroups of cells with value 2 are replaced with unique values starting from 2.
    """
    # Create a binary matrix where 1 indicates cells with value 2
    bin_map = (Map == 2).astype(int)

    # Initialize visited matrix
    visited = np.zeros_like(bin_map)

    # Define BFS function
    def bfs(start):
        queue = deque([start])
        subgroup = set()
        while queue:
            i, j = queue.popleft()
            if visited[i, j]:
                continue
            visited[i, j] = True
            if bin_map[i, j]:
                subgroup.add((i, j))
                for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < bin_map.shape[0] and 0 <= nj < bin_map.shape[1]:
                        queue.append((ni, nj))
        if len(subgroup) >= 1:
            return subgroup
        else:
            return set()

    # Apply BFS to each cell with value 2
    subgroups = [bfs((i, j)) for i, j in np.argwhere(bin_map)]

    # subgroups and save their coordinates
    subgroup_coords = []
    for subgroup in subgroups:
        if not subgroup:
            continue
        if len(subgroup) < 1:
            continue
        for i, j in subgroup:
            idx = i * bin_map.shape[1] + j
        subgroup_coords.append(list(subgroup))

    for i, subgroup in enumerate(subgroup_coords):
        for coord in subgroup:
            Map[coord[0], coord[1]] = i + 2

    return subgroup_coords, Map
