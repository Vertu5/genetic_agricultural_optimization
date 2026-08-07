# ==============================================================================
# 🌾 Project: Genetic Agricultural Optimization (NSGA-II + PROMETHEE II)
# 👨‍💻 Author: Olivier Vertu Ndingaoba
# 🌐 Portfolio: https://ndingaoba-oliviervertu.vercel.app/
# 📅 Date: August 2026
# 📝 Description: API Serverless FastAPI pour l'intégration Vercel et web front-end
#
# Ce code est le moteur back-end d'optimisation spatiale. Il a été conçu, 
# architecturé et développé de A à Z par mes soins pour être intégré 
# comme API Serverless sur ma plateforme interactive.
# ==============================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import sys

# Ajouter le dossier parent au path pour importer la suite de classes src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.MapManager import MapManager
from src.FitnessEvaluator import FitnessEvaluator
from src.NSGA2Engine import NSGA2Engine
from src.PrometheeDecision import PrometheeDecision

app = FastAPI(
    title="Genetic Agricultural Optimization API",
    description="API Serverless backend pour le moteur NSGA-II + PROMETHEE II",
    version="2.0.0"
)

class OptimizationRequest(BaseModel):
    usage_grid: Optional[List[List[str]]] = None
    cost_grid: Optional[List[List[float]]] = None
    prod_grid: Optional[List[List[float]]] = None
    budget: float = 500.0
    generations: int = 20
    pop_size: int = 30

@app.get("/")
def read_root():
    return {
        "status": "online",
        "project": "Genetic Agricultural Optimization API",
        "author": "Olivier Vertu Ndingaoba",
        "portfolio": "https://ndingaoba-oliviervertu.vercel.app/"
    }

@app.post("/api/optimize")
def optimize_land(request: OptimizationRequest):
    try:
        map_mgr = MapManager()

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        usage_path = os.path.join(base_dir, "Usage_map.txt")
        cost_path = os.path.join(base_dir, "Cost_map.txt")
        prod_path = os.path.join(base_dir, "Production_map.txt")

        map_mgr.load_maps(usage_path=usage_path, cost_path=cost_path, prod_path=prod_path)

        # Si des grilles dynamiques sont passées par le front-end
        if request.usage_grid:
            map_mgr.usage_map = np.array(request.usage_grid)
            map_mgr.rows, map_mgr.cols = map_mgr.usage_map.shape
            map_mgr._parse_cell_types()

        if request.cost_grid:
            map_mgr.cost_map = np.array(request.cost_grid)

        if request.prod_grid:
            map_mgr.prod_map = np.array(request.prod_grid)

        map_mgr.compute_distance_matrix()

        # Évaluation et Algorithme Génétique
        evaluator = FitnessEvaluator(map_mgr)
        engine = NSGA2Engine(
            map_manager=map_mgr,
            fitness_evaluator=evaluator,
            pop_size=request.pop_size,
            num_generations=request.generations,
            budget=request.budget
        )

        pareto_pop, pareto_fits = engine.run()

        # Classement PROMETHEE II
        decision_maker = PrometheeDecision()
        ranking, net_flows = decision_maker.rank(pareto_fits)

        best_idx = ranking[0] if ranking else 0
        best_solution_grid = pareto_pop[best_idx].tolist() if pareto_pop else []

        solutions_payload = []
        for idx in range(len(pareto_pop)):
            rank_pos = ranking.index(idx) + 1 if idx in ranking else len(pareto_pop)
            fit = pareto_fits[idx]
            solutions_payload.append({
                "id": idx,
                "rank": rank_pos,
                "compactness": fit[0],
                "proximity": fit[1],
                "productivity": fit[2],
                "net_flow": net_flows[idx] if idx < len(net_flows) else 0.0,
                "grid": pareto_pop[idx].tolist()
            })

        return {
            "status": "success",
            "best_solution": {
                "rank": 1,
                "compactness": pareto_fits[best_idx][0] if pareto_fits else 0,
                "proximity": pareto_fits[best_idx][1] if pareto_fits else 0,
                "productivity": pareto_fits[best_idx][2] if pareto_fits else 0,
                "grid": best_solution_grid
            },
            "pareto_solutions": solutions_payload
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
