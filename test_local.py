# ==============================================================================
# 🌾 Project: Genetic Agricultural Optimization (NSGA-II + PROMETHEE II)
# 👨‍💻 Author: Olivier Vertu Ndingaoba
# 🌐 Portfolio: https://ndingaoba-oliviervertu.vercel.app/
# 📅 Date: August 2026
# 📝 Description: Script de test local du moteur (sans API).
#
# Ce code est le moteur back-end d'optimisation spatiale. Il a été conçu, 
# architecturé et développé de A à Z par mes soins pour être intégré 
# comme API Serverless sur ma plateforme interactive.
# ==============================================================================

from src.MapManager import MapManager
from src.FitnessEvaluator import FitnessEvaluator
from src.NSGA2Engine import NSGA2Engine
from src.PrometheeDecision import PrometheeDecision

def run_local_test():
    print("🚀 Démarrage du test local du moteur d'optimisation...\n")

    # 1. Fausses données de test (mini-carte 3x3 pour un test rapide)
    usage_map = [
        ["A", "C", "R"],
        ["C", "C", "C"],
        ["R", "A", "C"]
    ]
    cost_map = [
        [0.0, 150.0, 0.0],
        [200.0, 100.0, 300.0],
        [0.0, 0.0, 250.0]
    ]
    production_map = [
        [0.0, 8.5, 0.0],
        [6.0, 9.2, 5.5],
        [0.0, 0.0, 7.8]
    ]
    budget = 500.0

    # 2. Initialisation des classes
    print("🗺️  Initialisation du MapManager...")
    map_manager = MapManager(usage_map, cost_map, production_map, budget)

    print("⚖️  Initialisation du FitnessEvaluator...")
    evaluator = FitnessEvaluator(map_manager)

    print("🧬 Exécution de NSGA-II (génération de la frontière de Pareto)...")
    # On met de petites valeurs (pop_size=10, generations=5) pour que le test soit instantané
    engine = NSGA2Engine(map_manager, evaluator, pop_size=10, num_generations=5) 
    pareto_front, pareto_fitnesses = engine.run()

    print(f"✅ NSGA-II a trouvé {len(pareto_front)} solutions optimales sur la frontière.\n")

    # 3. Prise de décision PROMETHEE II
    print("🏆 Évaluation PROMETHEE II pour trouver le meilleur compromis...")
    decision_maker = PrometheeDecision(pareto_fitnesses)
    best_index = decision_maker.get_best_solution(weights=[0.2, 0.4, 0.4])

    best_solution_map = pareto_front[best_index]
    best_metrics = pareto_fitnesses[best_index]

    # 4. Affichage des résultats dans le terminal
    print("\n" + "="*50)
    print("🎯 RÉSULTAT FINAL (Meilleure solution Rank 1) :")
    print("="*50)
    print(f"Compacité (C)   : {best_metrics[0]:.4f} (À minimiser)")
    print(f"Proximité (P)   : {best_metrics[1]:.4f} (À minimiser)")
    print(f"Productivité (R): {best_metrics[2]:.4f} (À maximiser)")
    print("\nGrille de la solution (0=Restreint/Ignoré, 2=Ferme existante ou Nouvelle terre achetée) :")
    for row in best_solution_map:
        print(row.tolist())
    print("="*50)
    print("\n✅ Le moteur back-end fonctionne parfaitement !")

if __name__ == "__main__":
    run_local_test()
