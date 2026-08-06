# 🌾 Genetic Agricultural Optimization (NSGA-II + PROMETHEE II)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Algorithm: NSGA--II](https://img.shields.io/badge/Algorithm-NSGA--II-green.svg)](https://en.wikipedia.org/wiki/NSGA-II)
[![MCDA: PROMETHEE II](https://img.shields.io/badge/MCDA-PROMETHEE%20II-orange.svg)](https://en.wikipedia.org/wiki/PROMETHEE)

A high-performance **Multi-Objective Evolutionary Optimization Framework** for spatial agricultural expansion and parcel allocation. 

The framework combines **NSGA-II (Non-dominated Sorting Genetic Algorithm II)** with **PROMETHEE II (Preference Ranking Organization METHod for Enrichment Evaluations)** to find optimal spatial parcel extensions respecting strict land-use and budget constraints.

---

## 📌 Problem Formulation & Objectives

Agricultural extension requires balancing competing spatial and economic goals over complex geographic landscapes without arbitrary scalar weighting:

1. **Productivity ($R_S$)** *(Maximize)*: Maximizes crop yields across selected agricultural cells based on production maps.
2. **Proximity ($P_S$)** *(Minimize)*: Minimizes average Euclidean distance from candidate parcels to existing agricultural infrastructure.
3. **Compactness ($C_S$)** *(Minimize)*: Minimizes the isoperimetric shape quotient ($C_S = \frac{\text{Perimeter}^2}{4 \pi \cdot \text{Area}}$), favoring contiguous, well-grouped parcel blocks (close to 1.0) over fragmented "confetti" shapes.
4. **Budget Constraint**: Enforces a strict budget ceiling ($\sum \text{Cost}(cell) \le B$) across all newly acquired land cells.

---

## 🎯 Core Goal

The goal of this project is to automate spatial land-use decision making for farm extension:
- **Input**: Geographic land usage map, soil productivity map, land acquisition cost map, and budget limit.
- **Process**: NSGA-II evolutionary search identifies the non-dominated Pareto-optimal land configurations balancing yield, proximity, and shape compactness.
- **Output**: Ranked land allocation configurations (via PROMETHEE II) alongside animated GIF visualizations showing spatial parcel evolution.

---

## 🚀 Key Improvements & Architecture

### 1. Multi-Fitness Pareto Dominance (Abandoning Scalar Sums)
Previous scalar implementations summed normalized objectives ($R(S) + \frac{1}{P(S)} + C(S)$), introducing bias and scaling artifacts. The new architecture implements pure **Multi-Objective Pareto Dominance**:
* **Vectorized Evaluation**: `evaluate_individual` returns raw physical objective vectors `(Compactness_C, Proximity_P, Productivity_R)`.
* **Pareto Dominance**: Individual $A$ dominates $B$ ($A \succ B$) iff $A$ is non-worse across all objectives ($C_A \le C_B, P_A \le P_B, R_A \ge R_B$) and strictly superior in at least one objective.


### 2. Native NSGA-II Implementation
* **Fast Non-Dominated Sorting**: Partitions the population into Pareto fronts ($F_1, F_2, \dots, F_k$).
* **Crowding Distance**: Computes boundary density metrics to maintain uniform diversity along the Pareto frontier.
* **Crowded Comparison Tournament**: Selects parents based on Pareto rank first, breaking ties using crowding distance.
* **Environmental Replacement ($2N \rightarrow N$)**: Selects the top $N$ individuals from combined parent and offspring populations.

### 3. Land Preprocessing & Distance Metrics
* **Spatial Preprocessing**: Computes minimum Euclidean distance matrices between candidate cultivable cells ($C$) and existing agricultural parcels ($A$).
* **Connected Component Subgrouping**: Uses Breadth-First Search (BFS) to identify contiguous parcel clusters for spatial analysis.

### 4. PROMETHEE II Preference Ranking
* Evaluates non-dominated solutions on the final Pareto front to provide a decision-maker ranking based on preference preference flows ($\Phi^+, \Phi^-$).

---

## 📂 Repository Structure

```
genetic_agricultural_optimization/
│
├── README.md                           # Comprehensive documentation
└── Projet_RO/                          # Core source code & dataset
    ├── main.py                         # Main execution pipeline
    ├── genetic_algo.py                 # NSGA-II Multi-Objective GA engine
    ├── evaluation.py                   # Objective evaluation & compactness formulas
    ├── mapfunctions.py                 # BFS subgrouping & map parser
    ├── utils.py                        # Spatial preprocessing & decision map loaders
    ├── prometh.py                      # PROMETHEE II MCDA ranking engine
    ├── generator.py                    # Initial heuristic generators
    ├── Cost_map.txt                    # Cell cost grid data
    ├── Production_map.txt              # Cell yield productivity grid data
    ├── Usage_map.txt                   # Land usage classification map (R/C/A)
    ├── pareto.csv                      # Exported Pareto-optimal solutions
    ├── input_maps.png                  # Visualized geographic maps
    └── pareto_frontier.png             # Visualized 3D Pareto frontier & 2D projections
```

---

## ⚡ Quick Start

### 1. Prerequisites
Ensure Python 3.8+ and required libraries are installed:
```bash
pip install numpy scipy matplotlib
```

### 2. Execution
Run the complete multi-objective optimization pipeline:
```bash
cd Projet_RO
python3 main.py
```

---

## 📊 Output Examples

Upon execution, the system outputs:
* **`pareto.csv`**: Contains non-dominated Pareto-optimal solution tuples `(compactness_score, proximity_score, productivity_score)`.
* **`input_maps.png`**: Displays input usage, proximity, productivity, and cost maps.
* **`pareto_frontier.png`**: Visualizes the 3D Pareto front and 2D trade-off projections ranked via PROMETHEE II.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to enhance crossover operators, add new spatial metrics, or extend MCDA algorithms.
