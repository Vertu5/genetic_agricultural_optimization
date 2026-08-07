# ==============================================================================
# 🌾 Project: Genetic Agricultural Optimization (NSGA-II + PROMETHEE II)
# 👨‍💻 Author: Olivier Vertu Ndingaoba
# 🌐 Portfolio: https://ndingaoba-oliviervertu.vercel.app/
# 📅 Date: August 2026
# 📝 Description: Gestionnaire de données géographiques et chargement des cartes
#
# Ce code est le moteur back-end d'optimisation spatiale. Il a été conçu, 
# architecturé et développé de A à Z par mes soins pour être intégré 
# comme API Serverless sur ma plateforme interactive.
# ==============================================================================

import numpy as np

class MapManager:
    """
    Classe responsable du chargement, du stockage et du prétraitement 
    des cartes géographiques (Usage, Coût, Production).
    """
    def __init__(self, usage_data=None, cost_data=None, prod_data=None, budget=500.0):
        self.usage_path = usage_data if isinstance(usage_data, str) else None
        self.cost_path = cost_data if isinstance(cost_data, str) else None
        self.prod_path = prod_data if isinstance(prod_data, str) else None
        self.budget = float(budget) if budget is not None else 500.0

        self.usage_map = None
        self.cost_map = None
        self.prod_map = None

        self.rows = 0
        self.cols = 0
        self.cultivable_cells = []
        self.existing_farm_cells = []
        self.restricted_cells = []
        self.dist_matrix = None

        # Si des listes ou arrays directes sont fournies
        if usage_data is not None and not isinstance(usage_data, str):
            self.usage_map = np.array(usage_data, dtype=str)
        if cost_data is not None and not isinstance(cost_data, str):
            self.cost_map = np.array(cost_data, dtype=float)
        if prod_data is not None and not isinstance(prod_data, str):
            self.prod_map = np.array(prod_data, dtype=float)

        if self.usage_map is not None:
            self.rows, self.cols = self.usage_map.shape
            self._parse_cell_types()
            self.compute_distance_matrix()

        # Si des chemins de fichiers texte sont fournis
        elif self.usage_path:
            self.load_maps(self.usage_path, self.cost_path, self.prod_path)

    def load_maps(self, usage_path: str = None, cost_path: str = None, prod_path: str = None):
        """Charge les cartes depuis des fichiers texte."""
        if usage_path: self.usage_path = usage_path
        if cost_path: self.cost_path = cost_path
        if prod_path: self.prod_path = prod_path

        if self.usage_path:
            self.usage_map = self._read_text_map(self.usage_path, dtype=str)
            self.rows, self.cols = self.usage_map.shape
            self._parse_cell_types()

        if self.cost_path:
            self.cost_map = self._read_text_map(self.cost_path, dtype=float)

        if self.prod_path:
            self.prod_map = self._read_text_map(self.prod_path, dtype=float)

        self.compute_distance_matrix()

    def _read_text_map(self, filepath: str, dtype):
        grid = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    grid.append([dtype(x) for x in parts])
        return np.array(grid)

    def _parse_cell_types(self):
        self.cultivable_cells = []
        self.existing_farm_cells = []
        self.restricted_cells = []

        for r in range(self.rows):
            for c in range(self.cols):
                val = str(self.usage_map[r, c]).strip().upper()
                if val == 'C':
                    self.cultivable_cells.append((r, c))
                elif val == 'A':
                    self.existing_farm_cells.append((r, c))
                elif val == 'R':
                    self.restricted_cells.append((r, c))

    def compute_distance_matrix(self):
        if not self.cultivable_cells or not self.existing_farm_cells:
            if self.cultivable_cells:
                self.dist_matrix = np.ones(len(self.cultivable_cells))
            return self.dist_matrix

        c_coords = np.array(self.cultivable_cells)
        a_coords = np.array(self.existing_farm_cells)

        diff = c_coords[:, np.newaxis, :] - a_coords[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))

        self.dist_matrix = np.min(dists, axis=1)
        return self.dist_matrix
