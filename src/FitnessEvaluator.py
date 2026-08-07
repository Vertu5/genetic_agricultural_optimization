# ==============================================================================
# 🌾 Project: Genetic Agricultural Optimization (NSGA-II + PROMETHEE II)
# 👨‍💻 Author: Olivier Vertu Ndingaoba
# 🌐 Portfolio: https://ndingaoba-oliviervertu.vercel.app/
# 📅 Date: August 2026
# 📝 Description: Évaluateur des objectifs physiques (Compacité, Proximité, Productivité)
#
# Ce code est le moteur back-end d'optimisation spatiale. Il a été conçu, 
# architecturé et développé de A à Z par mes soins pour être intégré 
# comme API Serverless sur ma plateforme interactive.
# ==============================================================================

import numpy as np

class FitnessEvaluator:
    """
    Classe chargée d'évaluer les 3 objectifs physiques non-dominés d'une configuration :
    1. Compactness C(S) [MINIMIZE]
    2. Proximity P(S)    [MINIMIZE]
    3. Productivity R(S) [MAXIMIZE]
    """

    def __init__(self, map_manager):
        self.map_manager = map_manager

    def find_subgroups(self, grid_solution):
        """
        Subgrouping spatial par BFS pour détecter les blocs contigus sélectionnés.
        """
        rows, cols = grid_solution.shape
        visited = np.zeros((rows, cols), dtype=bool)
        subgroups = []

        for r in range(rows):
            for c in range(cols):
                if grid_solution[r, c] == 2 and not visited[r, c]:
                    # BFS pour former la composante connexe
                    queue = [(r, c)]
                    visited[r, c] = True
                    cluster = [(r, c)]

                    while queue:
                        curr_r, curr_c = queue.pop(0)
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if grid_solution[nr, nc] == 2 and not visited[nr, nc]:
                                    visited[nr, nc] = True
                                    queue.append((nr, nc))
                                    cluster.append((nr, nc))
                    subgroups.append(cluster)
        return subgroups

    def calculate_compactness(self, subgroups):
        compactness_vals = []
        for cluster in subgroups:
            if not cluster:
                continue
            cluster_set = set(cluster)
            boundary_len = sum(
                (r + dr, c + dc) not in cluster_set
                for r, c in cluster
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]
            )
            area = len(cluster)
            c_val = (boundary_len ** 2) / (4.0 * np.pi * area) if area > 0 else 1.0
            compactness_vals.append(c_val)
        return compactness_vals if compactness_vals else [1.0]

    def calculate_average_values(self, subgroups, data_map):
        vals = []
        for cluster in subgroups:
            if not cluster:
                continue
            arr = np.array(cluster)
            cluster_vals = data_map[arr[:, 0], arr[:, 1]]
            vals.append(np.mean(cluster_vals))
        return vals if vals else [1.0]

    def evaluate(self, grid_solution, budget=None):
        """
        Évalue la solution et retourne le triplet (Compactness, Proximity, Productivity).
        Applique une pénalité sévère si le budget est dépassé.
        """
        if budget is not None and self.map_manager.cost_map is not None:
            total_cost = np.sum(self.map_manager.cost_map[grid_solution == 2])
            if total_cost > budget:
                return (float('inf'), float('inf'), 0.0)

        subgroups = self.find_subgroups(grid_solution)
        if not subgroups:
            return (float('inf'), float('inf'), 0.0)

        total_area = sum(len(sg) for sg in subgroups)
        if total_area == 0:
            return (float('inf'), float('inf'), 0.0)

        weights = np.array([len(sg) / total_area for sg in subgroups])

        # 1. Compactness C
        local_comp = self.calculate_compactness(subgroups)
        global_comp = np.average(local_comp, weights=weights)

        # 2. Proximity P
        if self.map_manager.dist_matrix is not None:
            # calcul via dist_matrix
            cell_indices = {coord: idx for idx, coord in enumerate(self.map_manager.cultivable_cells)}
            selected_dist = [self.map_manager.dist_matrix[cell_indices[(r, c)]] 
                             for r, c in self.map_manager.cultivable_cells 
                             if grid_solution[r, c] == 2 and (r, c) in cell_indices]
            global_prox = np.mean(selected_dist) if selected_dist else float('inf')
        else:
            global_prox = float('inf')

        # 3. Productivity R
        if self.map_manager.prod_map is not None:
            local_prod = self.calculate_average_values(subgroups, self.map_manager.prod_map)
            global_prod = np.average(local_prod)
        else:
            global_prod = 0.0

        return (float(global_comp), float(global_prox), float(global_prod))
