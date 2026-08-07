# 🌾 Genetic Agricultural Optimization API (NSGA-II + PROMETHEE II)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Algorithm: NSGA--II](https://img.shields.io/badge/Algorithm-NSGA--II-green.svg)](https://en.wikipedia.org/wiki/NSGA-II)
[![MCDA: PROMETHEE II](https://img.shields.io/badge/MCDA-PROMETHEE%20II-orange.svg)](https://en.wikipedia.org/wiki/PROMETHEE)

> **👨‍💻 Author**: Olivier Vertu Ndingaoba  
> **🌐 Portfolio**: [ndingaoba-oliviervertu.vercel.app](https://ndingaoba-oliviervertu.vercel.app/)  
> **📝 Description**: Moteur d'optimisation spatiale multi-objectif (POO & API Serverless pour plateforme web Vercel).

---

## 📌 Architecture & Présentation

Ce dépôt contient la version **Production / API Serverless** du moteur d'optimisation agricole spatiale. Il combine l'algorithme évolutif multi-objectif **NSGA-II** et la méthode décisionnelle multi-critère **PROMETHEE II**.

### Objectifs d'Optimisation :
1. **Productivité ($R_S$) [Max]** : Rendement agricole total issu des cartes de fertilité des sols.
2. **Proximité ($P_S$) [Min]** : Distance euclidienne moyenne aux parcelles agricoles existantes ($A$).
3. **Compacité ($C_S$) [Min]** : Forme géométrique des blocs ($C_S = \frac{\text{Périmètre}^2}{4 \pi \cdot \text{Surface}}$).
4. **Budget** : Respect d'un plafond budgétaire d'acquisition.

---

## 📂 Structure du Dépôt (POO & API)

```text
genetic_agricultural_optimization/
│
├── api/
│   └── index.py                # Routeur API Serverless (FastAPI) pour Vercel
│
├── src/
│   ├── MapManager.py           # Chargement des cartes & prétraitement des distances
│   ├── FitnessEvaluator.py     # Évaluation vectorisée des 3 objectifs physiques
│   ├── NSGA2Engine.py          # Moteur NSGA-II (Tri non-dominé, Crowding, Mutation)
│   └── PrometheeDecision.py    # Décision multi-critère PROMETHEE II (Classement)
│
├── data/                       # Données géographiques initiales
│   ├── Cost_map.txt
│   ├── Production_map.txt
│   └── Usage_map.txt
│
└── requirements.txt            # Dépendances API (FastAPI, NumPy, Pydantic, Uvicorn)
```

---

## 🚀 Lancement Local de l'API

### 1. Installation des dépendances
```bash
pip install -r requirements.txt
```

### 2. Démarrer le serveur API
```bash
uvicorn api.index:app --reload
```
L'API est accessible à l'adresse : `http://127.0.0.1:8000`  
La documentation interactive Swagger est disponible sur `http://127.0.0.1:8000/docs`.

---

## 🔗 Intégration Front-End (Appel API)

Exemple d'appel depuis l'application web React / Next.js sur `https://ndingaoba-oliviervertu.vercel.app/` :

```javascript
const response = await fetch("https://votre-api-vercel.app/api/optimize", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    budget: 500,
    generations: 30,
    pop_size: 40
  })
});
const data = await response.json();
console.log("Solution Gagnante (Rang 1):", data.best_solution);
```

---

## 📜 Historique & Branches

* **`main`** : Architecture POO & API Serverless moderne pour le web.
* **`old-version-before-website`** : Version procédurale originale avec génération de graphes Matplotlib & GIFs animés.
