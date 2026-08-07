# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import cdist

# Fix mpl_toolkits namespace import if needed
try:
    import site
    import mpl_toolkits
    site_packages = site.getusersitepackages()
    if site_packages + '/mpl_toolkits' not in mpl_toolkits.__path__:
        mpl_toolkits.__path__.insert(0, site_packages + '/mpl_toolkits')
    from mpl_toolkits.mplot3d import Axes3D
except Exception:
    pass

import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Import project modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mapfunctions import *
from utils import *
from evaluation import *
from genetic_algo import *
from prometh import *

def run_verification(best_solutions=None):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    
    print("=" * 65)
    print("🔍 RUNNING PARETO SOLUTION VERIFICATION & 3D SURFACE AUDIT")
    print("=" * 65)
    
    # 1. Load Data
    Map = create_map(read_file(os.path.join(DATA_DIR, "Usage_map.txt")))
    cost_map = read_ProdCost(os.path.join(DATA_DIR, "Cost_map.txt"), Map)
    productivity_map = read_ProdCost(os.path.join(DATA_DIR, "Production_map.txt"), Map)
    
    # Calculate proximity map
    indices_1 = np.argwhere(Map == 1)
    indices_2 = np.argwhere(Map == 2)
    if len(indices_1) > 0 and len(indices_2) > 0:
        distances = cdist(indices_1, indices_2)
        min_distances = np.min(distances, axis=1)
        proximity_map = np.ones_like(Map, dtype=float) * 9.0
        min_val = np.min(min_distances)
        max_val = np.max(min_distances)
        if max_val > min_val:
            proximity_map[Map == 1] = np.interp(min_distances, (min_val, max_val), (1, 9))
        else:
            proximity_map[Map == 1] = 1.0
    else:
        proximity_map = np.ones_like(Map, dtype=float) * 9.0

    budget_limit = 500.0
    pareto_csv_path = os.path.join(OUTPUT_DIR, "pareto.csv")
    
    if not os.path.exists(pareto_csv_path):
        print(f"❌ Error: {pareto_csv_path} not found. Run src/main.py first.")
        return
        
    df_pareto = pd.read_csv(pareto_csv_path)
    data = df_pareto[['compactness_C', 'proximity_P', 'productivity_R']].values
    
    print(f"\n📊 1. SUMMARY STATISTICS ({len(df_pareto)} Pareto-optimal solutions):")
    print("-" * 65)
    stats_df = pd.DataFrame({
        'Objective': ['Compactness (C, Min)', 'Proximity (P, Min)', 'Productivity (R, Max)'],
        'Min': [df_pareto['compactness_C'].min(), df_pareto['proximity_P'].min(), df_pareto['productivity_R'].min()],
        'Max': [df_pareto['compactness_C'].max(), df_pareto['proximity_P'].max(), df_pareto['productivity_R'].max()],
        'Mean': [df_pareto['compactness_C'].mean(), df_pareto['proximity_P'].mean(), df_pareto['productivity_R'].mean()],
        'Std': [df_pareto['compactness_C'].std(), df_pareto['proximity_P'].std(), df_pareto['productivity_R'].std()]
    })
    print(stats_df.to_string(index=False))

    # 2. Check Pareto Non-Domination
    print("\n🛡️ 2. PARETO NON-DOMINATION CHECK:")
    print("-" * 65)
    dominated_count = 0
    n = len(data)
    for i in range(n):
        for j in range(n):
            if i != j and Pareto_dominates(data[j], data[i]):
                dominated_count += 1
                break
                
    if dominated_count == 0:
        print("✅ SUCCESS: All exported solutions strictly belong to the Non-Dominated Pareto Front!")
    else:
        print(f"⚠️ WARNING: Found {dominated_count} dominated solutions in pareto.csv.")

    # 3. PROMETHEE II Ranking Verification
    print("\n🏆 3. PROMETHEE II DECISION RANKING AUDIT:")
    print("-" * 65)
    weights = np.array([0.2, 0.4, 0.4])
    ranking_indices = promethee(data, weights)
    
    # Calculate net flows for report
    n_data = normalize(data)
    w_data = weigh(n_data, weights)
    p_matrix = positive_flow(w_data)
    n_matrix = negative_flow(w_data)
    net_flows = preference_index(p_matrix, n_matrix)
    
    top_sol_idx = ranking_indices[0]
    print(f"🥇 Best Solution (Rank 1 - Highest Net Flow φ = {net_flows[top_sol_idx]:.4f}):")
    print(f"   Compactness (C): {data[top_sol_idx][0]:.4f} (Isoperimetric Quotient)")
    print(f"   Proximity   (P): {data[top_sol_idx][1]:.4f} (Euclidean Distance)")
    print(f"   Productivity(R): {data[top_sol_idx][2]:.4f} (Crop Yield)")

    # 4. Generate 3D Matplotlib Plot with Surface Mesh
    print("\n🎨 4. GENERATING 3D PARETO SURFACE & SCATTER PLOTS...")
    print("-" * 65)
    fig_3d = plt.figure(figsize=(11, 9))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    # Surface Mesh overlay via trisurf
    try:
        ax_3d.plot_trisurf(
            df_pareto['compactness_C'],
            df_pareto['proximity_P'],
            df_pareto['productivity_R'],
            cmap='viridis',
            alpha=0.3,
            edgecolor='none',
            linewidth=0.2
        )
    except Exception as e:
        print(f"Note on trisurf mesh: {e}")
    
    # Scatter points in 3D
    sc = ax_3d.scatter(
        df_pareto['compactness_C'],
        df_pareto['proximity_P'],
        df_pareto['productivity_R'],
        c=net_flows,
        cmap='viridis',
        s=75,
        edgecolors='k',
        alpha=0.9
    )
    
    # Highlight top PROMETHEE solution
    ax_3d.scatter(
        [data[top_sol_idx][0]],
        [data[top_sol_idx][1]],
        [data[top_sol_idx][2]],
        color='red',
        s=180,
        marker='*',
        label='Solution N°1 (PROMETHEE II - Meilleur compromis)'
    )
    
    ax_3d.set_xlabel('Compacité C (Min)', fontsize=11, labelpad=10)
    ax_3d.set_ylabel('Proximité P (Min)', fontsize=11, labelpad=10)
    ax_3d.set_zlabel('Productivité R (Max)', fontsize=11, labelpad=10)
    ax_3d.set_title('Frontière de Pareto 3D (Surface & Triangulation)', fontsize=13, fontweight='bold', pad=15)
    
    cbar = plt.colorbar(sc, ax=ax_3d, shrink=0.6, pad=0.1)
    cbar.set_label('Flux Net PROMETHEE II (φ)', fontsize=10)
    ax_3d.legend(loc='upper right')
    ax_3d.view_init(elev=25, azim=135)
    
    plot3d_png_path = os.path.join(OUTPUT_DIR, "pareto_frontier_3d.png")
    plt.tight_layout()
    plt.savefig(plot3d_png_path, dpi=300)
    plt.close(fig_3d)
    print(f"✅ Saved static 3D surface plot: {plot3d_png_path}")
    
    # 5. Generate Interactive Plotly 3D HTML with Surface Mesh
    fig_plotly = go.Figure()
    
    # Add 3D Mesh Surface
    fig_plotly.add_trace(go.Mesh3d(
        x=df_pareto['compactness_C'],
        y=df_pareto['proximity_P'],
        z=df_pareto['productivity_R'],
        opacity=0.35,
        colorscale='Viridis',
        intensity=net_flows,
        name='Surface Pareto'
    ))
    
    # Add 3D Scatter Points
    fig_plotly.add_trace(go.Scatter3d(
        x=df_pareto['compactness_C'],
        y=df_pareto['proximity_P'],
        z=df_pareto['productivity_R'],
        mode='markers',
        marker=dict(
            size=7,
            color=net_flows,
            colorscale='Viridis',
            colorbar=dict(title='Flux Net PROMETHEE II (φ)'),
            showscale=True,
            line=dict(width=1, color='DarkSlateGrey')
        ),
        text=[f"Rang: {i+1}<br>C (Compacité): {c:.3f}<br>P (Proximité): {p:.3f}<br>R (Productivité): {r:.3f}<br>Flux Net φ: {nf:.4f}" 
              for i, (c, p, r, nf) in enumerate(zip(data[:, 0], data[:, 1], data[:, 2], net_flows))],
        hoverinfo='text',
        name='Solutions Pareto'
    ))
    
    # Highlight Rank 1 Solution
    fig_plotly.add_trace(go.Scatter3d(
        x=[data[top_sol_idx][0]],
        y=[data[top_sol_idx][1]],
        z=[data[top_sol_idx][2]],
        mode='markers+text',
        marker=dict(size=14, color='red', symbol='diamond'),
        text=["🏆 Solution N°1 Top Compromis"],
        textposition="top center",
        name="Rang 1 PROMETHEE II"
    ))
    
    fig_plotly.update_layout(
        title="Surface et Frontière de Pareto 3D Interactive (NSGA-II + PROMETHEE II)",
        scene=dict(
            xaxis_title="Compacité C (Min)",
            yaxis_title="Proximité P (Min)",
            zaxis_title="Productivité R (Max)"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    plotly_html_path = os.path.join(OUTPUT_DIR, "pareto_3d_interactive.html")
    fig_plotly.write_html(plotly_html_path)
    print(f"✅ Saved interactive 3D HTML surface plot: {plotly_html_path}")

    # 6. Generate Pareto Solutions Tour GIF if solutions are available
    if best_solutions is not None and len(best_solutions) > 0:
        print("\n🎬 6. GENERATING PARETO SOLUTIONS TOUR GIF...")
        print("-" * 65)
        temp_dir = os.path.join(OUTPUT_DIR, "tour_frames")
        os.makedirs(temp_dir, exist_ok=True)
        
        tour_frames = []
        cmap_land, patches_land = get_custom_colormap_and_legend()
        
        # Sample solutions ordered by PROMETHEE II rank
        sample_indices = ranking_indices[::max(1, len(ranking_indices) // 30)]
        if ranking_indices[0] not in sample_indices:
            sample_indices = np.insert(sample_indices, 0, ranking_indices[0])
            
        for step_idx, sol_idx in enumerate(sample_indices):
            sol_map = best_solutions[sol_idx] if sol_idx < len(best_solutions) else best_solutions[0]
            vis_map = get_visualization_map(Map, sol_map)
            
            sol_c, sol_p, sol_r = data[sol_idx]
            sol_phi = net_flows[sol_idx]
            rank_num = np.where(ranking_indices == sol_idx)[0][0] + 1
            
            fig_tour, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1.2, 1]})
            
            # Left: Spatial Land Allocation Map
            ax_left.imshow(vis_map, cmap=cmap_land, vmin=0, vmax=3)
            is_top = (rank_num == 1)
            title_prefix = "★ RANG 1 (MEILLEUR COMPROMIS)" if is_top else f"Solution Pareto #{sol_idx + 1} (Rang {rank_num}/{len(data)})"
            ax_left.set_title(f"{title_prefix}\nAllocation Spatiale des Parcelles", fontsize=11, fontweight='bold', color='darkred' if is_top else 'black')
            ax_left.legend(handles=patches_land, loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8)
            
            # Right: 3D Pareto Position
            ax_right = fig_tour.add_subplot(1, 2, 2, projection='3d')
            ax_right.scatter(data[:, 0], data[:, 1], data[:, 2], c=net_flows, cmap='viridis', s=40, alpha=0.6)
            ax_right.scatter([sol_c], [sol_p], [sol_r], color='red' if is_top else 'orange', s=160, marker='*' if is_top else 'o', label=f'Current: Rank {rank_num}')
            
            ax_right.set_xlabel('Compacité C (Min)', fontsize=9)
            ax_right.set_ylabel('Proximité P (Min)', fontsize=9)
            ax_right.set_zlabel('Productivité R (Max)', fontsize=9)
            ax_right.set_title(f"Score: C={sol_c:.2f} | P={sol_p:.2f} | R={sol_r:.2f}\nFlux Net PROMETHEE φ={sol_phi:.3f}", fontsize=11)
            ax_right.view_init(elev=25, azim=135)
            
            plt.tight_layout()
            frame_path = os.path.join(temp_dir, f"tour_{step_idx:03d}.png")
            plt.savefig(frame_path, dpi=120)
            plt.close(fig_tour)
            tour_frames.append(frame_path)
            
        # Build GIF
        gif_path = os.path.join(OUTPUT_DIR, "pareto_solutions_tour.gif")
        images = [Image.open(f) for f in tour_frames]
        if images:
            images[0].save(gif_path, save_all=True, append_images=images[1:], duration=450, loop=0)
            print(f"✅ GIF Tour créé avec succès : {gif_path}")
            
        # Cleanup temp frames
        for f in tour_frames:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
            
    print("=" * 65)

if __name__ == "__main__":
    run_verification()
