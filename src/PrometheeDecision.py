# ==============================================================================
# 🌾 Project: Genetic Agricultural Optimization (NSGA-II + PROMETHEE II)
# 👨‍💻 Author: Olivier Vertu Ndingaoba
# 🌐 Portfolio: https://ndingaoba-oliviervertu.vercel.app/
# 📅 Date: August 2026
# 📝 Description: Moteur de décision MCDA PROMETHEE II pour classer le front de Pareto
#
# Ce code est le moteur back-end d'optimisation spatiale. Il a été conçu, 
# architecturé et développé de A à Z par mes soins pour être intégré 
# comme API Serverless sur ma plateforme interactive.
# ==============================================================================

import numpy as np

class PrometheeDecision:
    """
    Module d'analyse décisionnelle multi-critère (MCDA) PROMETHEE II :
    - Normalisation adaptative des 3 objectifs (Compactness, Proximity, Productivity)
    - Calcul des matrices de flux de préférence (Phi+, Phi-)
    - Classement absolu par flux net (Phi = Phi+ - Phi-)
    """

    def __init__(self, fitness_matrix=None, weights=None):
        self.fitness_matrix = fitness_matrix
        self.weights = np.array(weights) if weights is not None else np.array([0.33, 0.33, 0.34])

    def normalize(self, fitness_matrix):
        data = np.array(fitness_matrix)
        if len(data) <= 1:
            return np.ones_like(data)

        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        range_val = np.where((max_val - min_val) == 0, 1.0, (max_val - min_val))

        norm = (data - min_val) / range_val
        norm[:, 0] = 1.0 - norm[:, 0]  # Compactness C: plus petit = meilleur
        norm[:, 1] = 1.0 - norm[:, 1]  # Proximity P: plus petit = meilleur
        return norm

    def rank(self, fitness_matrix=None, weights=None):
        data_to_rank = fitness_matrix if fitness_matrix is not None else self.fitness_matrix
        weights_to_use = np.array(weights) if weights is not None else self.weights

        if data_to_rank is None or len(data_to_rank) == 0:
            return [], []

        if len(data_to_rank) == 1:
            return [0], [1.0]

        norm_data = self.normalize(data_to_rank)
        weighed_data = norm_data * weights_to_use
        n = len(weighed_data)

        p_matrix = np.zeros((n, n))
        n_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                p_matrix[i, j] = np.sum(np.maximum(weighed_data[i] - weighed_data[j], 0))
                n_matrix[i, j] = np.sum(np.maximum(weighed_data[j] - weighed_data[i], 0))

        s_plus = np.sum(p_matrix, axis=1) / (n - 1)
        s_minus = np.sum(n_matrix, axis=1) / (n - 1)
        net_flows = s_plus - s_minus

        ranking_indices = np.argsort(net_flows)[::-1].tolist()
        return ranking_indices, net_flows.tolist()

    def get_best_solution(self, weights=None, fitness_matrix=None):
        ranking_indices, _ = self.rank(fitness_matrix=fitness_matrix, weights=weights)
        return ranking_indices[0] if ranking_indices else 0
