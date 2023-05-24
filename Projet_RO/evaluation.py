import numpy as np


def calculate_compactness(subgroup_coords):
    compactness = []
    for subgroup_coord in subgroup_coords:
        boundary_len = sum(
            (ni, nj) not in subgroup_coord
            for i, j in subgroup_coord
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for ni, nj in [(i + di, j + dj)]
        )
        area = len(subgroup_coord)
        compactness.append((boundary_len ** 2) / (4 * np.pi * area))
    return compactness


def calculate_pro(subgroup_coords, pro_map):
    pro = []
    for subgroup_coord in subgroup_coords:
        # Convert the tuple to a numpy array
        subgroup_coord = np.array(subgroup_coord)
        # Use the indices of the subgroup to extract the corresponding values from pro_map
        values = pro_map[subgroup_coord[:, 0], subgroup_coord[:, 1]]
        pro.append(np.mean(values))
    return pro


def calculate_global_objectives(subgroup_coords, proximity_map, productivity_map):
    total_area = sum(len(subgroup) for subgroup in subgroup_coords)
    weights = np.array([float(len(subgroup)) / total_area for subgroup in subgroup_coords])

    local_compactnesses = calculate_compactness(subgroup_coords)
    global_compactness = np.average(local_compactnesses, weights=weights)

    local_proximities = calculate_pro(subgroup_coords, proximity_map)
    global_proximity = np.average(local_proximities)

    local_productivities = calculate_pro(subgroup_coords, productivity_map)
    global_productivity = np.average(local_productivities)

    return (global_compactness, 1/global_proximity, global_productivity)


def calculate_global_compactness(subgroup_coords):
    total_area = sum(len(subgroup) for subgroup in subgroup_coords)
    weights = np.array([float(len(subgroup)) / total_area for subgroup in subgroup_coords])

    local_compactnesses = calculate_compactness(subgroup_coords)
    global_compactness = np.average(local_compactnesses, weights=weights)

    return global_compactness


def calculate_global_pro(subgroup_coords, proximity_map):
    local_pro = calculate_pro(subgroup_coords, proximity_map)
    global_pro = np.average(local_pro)

    return global_pro

def calculate_fitness_globale(subgroup_coords, proximity_map, productivity_map):
   
    fitness = calculate_global_pro(subgroup_coords, productivity_map) + 1/calculate_global_pro(subgroup_coords, proximity_map) + calculate_global_compactness(subgroup_coords)

    return fitness
    