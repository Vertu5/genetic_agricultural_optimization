import numpy as np
from mapfunctions import subgroups


def calculate_compactness(subgroup_coords):
    compactness = []
    for subgroup_coord in subgroup_coords:
        if not subgroup_coord:
            continue
        boundary_len = sum(
            (ni, nj) not in subgroup_coord
            for i, j in subgroup_coord
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for ni, nj in [(i + di, j + dj)]
        )
        area = len(subgroup_coord)
        c_val = (boundary_len ** 2) / (4 * np.pi * area) if area > 0 else 1.0
        compactness.append(c_val)
    return compactness if compactness else [1.0]


def calculate_pro(subgroup_coords, pro_map):
    pro = []
    for subgroup_coord in subgroup_coords:
        if not subgroup_coord:
            continue
        subgroup_coord_arr = np.array(subgroup_coord)
        values = pro_map[subgroup_coord_arr[:, 0], subgroup_coord_arr[:, 1]]
        pro.append(np.mean(values))
    return pro if pro else [1.0]


def calculate_global_objectives(subgroup_coords, proximity_map, productivity_map):
    """
    Returns a tuple of 3 raw physical objective values:
    1. Compactness C(S) = Perimeter^2 / (4 * pi * Area)  [MINIMIZE - closer to 1.0 is more compact]
    2. Proximity P(S) = Average distance to existing parcels [MINIMIZE - closer is better]
    3. Productivity R(S) = Average crop yield               [MAXIMIZE - higher yield is better]
    """
    if not subgroup_coords:
        return (float('inf'), float('inf'), 0.0)

    total_area = sum(len(subgroup) for subgroup in subgroup_coords)
    if total_area == 0:
        return (float('inf'), float('inf'), 0.0)

    weights = np.array([float(len(subgroup)) / total_area for subgroup in subgroup_coords])

    local_compactnesses = calculate_compactness(subgroup_coords)
    global_compactness = np.average(local_compactnesses, weights=weights) if len(local_compactnesses) > 0 else float('inf')

    local_proximities = calculate_pro(subgroup_coords, proximity_map)
    global_proximity = np.average(local_proximities) if len(local_proximities) > 0 else float('inf')

    local_productivities = calculate_pro(subgroup_coords, productivity_map)
    global_productivity = np.average(local_productivities) if len(local_productivities) > 0 else 0.0

    return (global_compactness, global_proximity, global_productivity)


def evaluate_individual(individual, proximity_map, productivity_map, cost_map=None, budget=None):
    """
    Multi-objective evaluation returning raw tuple (compactness_C, proximity_P, productivity_R).
    Applies penalty if budget constraint is violated.
    """
    if cost_map is not None and budget is not None:
        cost = np.sum(cost_map[individual == 2])
        if cost > budget:
            return (float('inf'), float('inf'), 0.0)

    subgroup_coords, _ = subgroups(individual.copy())
    return calculate_global_objectives(subgroup_coords, proximity_map, productivity_map)



def calculate_global_compactness(subgroup_coords):
    if not subgroup_coords:
        return 1.0
    total_area = sum(len(subgroup) for subgroup in subgroup_coords)
    if total_area == 0:
        return 1.0
    weights = np.array([float(len(subgroup)) / total_area for subgroup in subgroup_coords])
    local_compactnesses = calculate_compactness(subgroup_coords)
    return np.average(local_compactnesses, weights=weights)


def calculate_global_pro(subgroup_coords, proximity_map):
    if not subgroup_coords:
        return 1.0
    local_pro = calculate_pro(subgroup_coords, proximity_map)
    return np.average(local_pro) if len(local_pro) > 0 else 1.0


def calculate_fitness_globale(subgroup_coords, proximity_map, productivity_map):
    """
    Legacy composite scalar fitness (kept for compatibility).
    """
    comp, prox, prod = calculate_global_objectives(subgroup_coords, proximity_map, productivity_map)
    return comp + prox + prod

    