"""
LEVERKUSEN CASE STUDY: Anatomy of an Undefeated Season
======================================================

This script analyzes Bayer 04 Leverkusen's historic 2023/24 Bundesliga campaign
(undefeated, 90 points) through the lens of network cohesion.

Key questions:
1. Who are the hub players that funnel play? (The "Xhaka effect")
2. How does their network structure differ from other teams?
3. Did cohesion vary across the season?

Run from notebooks folder: python 04_leverkusen_case_study.py
"""

import sys
sys.path.insert(0, '../src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from kloppy import impect
from cohesion_metric import (
    analyze_match_from_kloppy, 
    extract_metadata,
    PassingNetworkBuilder,
    CohesionCalculator,
    compute_network_metrics
)

# Output directories
FIGURES_DIR = Path('../figures')
DATA_DIR = Path('../data')
FIGURES_DIR.mkdir(exist_ok=True)


def load_leverkusen_data():
    """Load pre-computed results or process matches."""
    
    results_file = DATA_DIR / 'cohesion_results.csv'
    
    if results_file.exists():
        print("Loading cached results...")
        df = pd.read_csv(results_file)
        lev_df = df[df['team_name'] == 'Bayer 04 Leverkusen'].copy()
        print(f"Found {len(lev_df)} Leverkusen matches")
        return lev_df, df
    else:
        print("ERROR: Run 03_full_analysis.py first to generate cohesion_results.csv")
        return None, None


def find_leverkusen_match_ids(df):
    """Get all match IDs where Leverkusen played."""
    return df[df['team_name'] == 'Bayer 04 Leverkusen']['match_id'].unique().tolist()


def analyze_hub_players(match_ids, n_matches=10):
    """
    Analyze hub players across multiple Leverkusen matches.
    
    Returns aggregated betweenness centrality for identifying key playmakers.
    """
    print(f"\nAnalyzing hub players across {n_matches} matches...")
    
    player_betweenness = defaultdict(list)
    player_degree = defaultdict(list)
    player_appearances = defaultdict(int)
    
    sample_ids = match_ids[:n_matches]
    
    for match_id in tqdm(sample_ids, desc="Processing"):
        try:
            events = impect.load_open_data(match_id=match_id)
            results = analyze_match_from_kloppy(events)
            
            # Find Leverkusen in results
            for team_name, data in results.items():
                if 'Leverkusen' in team_name:
                    G = data['network']
                    metrics = data['metrics']
                    
                    # Store betweenness for each player
                    for player, bc in metrics.get('betweenness', {}).items():
                        player_betweenness[player].append(bc)
                        player_appearances[player] += 1
                    
                    # Store degree for each player
                    for player, deg in metrics.get('degree', {}).items():
                        player_degree[player].append(deg)
                    
                    break
                    
        except Exception as e:
            continue
    
    # Aggregate results
    hub_stats = []
    for player in player_betweenness.keys():
        hub_stats.append({
            'player': player,
            'avg_betweenness': np.mean(player_betweenness[player]),
            'max_betweenness': np.max(player_betweenness[player]),
            'avg_degree': np.mean(player_degree.get(player, [0])),
            'appearances': player_appearances[player]
        })
    
    hub_df = pd.DataFrame(hub_stats)
    hub_df = hub_df.sort_values('avg_betweenness', ascending=False)
    
    return hub_df


def visualize_leverkusen_network(match_id=None):
    """
    Create a visualization of Leverkusen's passing network.
    """
    # If no match_id provided, find a Leverkusen match
    if match_id is None:
        df = pd.read_csv(DATA_DIR / 'cohesion_results.csv')
        match_id = df[df['team_name'] == 'Bayer 04 Leverkusen']['match_id'].iloc[0]
    
    print(f"\nLoading match {match_id} for network visualization...")
    
    events = impect.load_open_data(match_id=match_id)
    player_id_to_name, player_id_to_position, team_id_to_name = extract_metadata(events)
    
    df_events = events.to_df(engine="pandas")
    
    # Find Leverkusen team_id
    lev_team_id = None
    for tid, name in team_id_to_name.items():
        if 'Leverkusen' in name:
            lev_team_id = tid
            break
    
    if lev_team_id is None:
        print("ERROR: Leverkusen not found in this match")
        return
    
    # Build network
    builder = PassingNetworkBuilder(
        df_events, lev_team_id,
        player_id_to_name, player_id_to_position, team_id_to_name
    )
    G = builder.build_network()
    
    # Get opponent name
    opponent = [name for name in team_id_to_name.values() if 'Leverkusen' not in name][0]
    
    # Calculate metrics
    metrics = compute_network_metrics(G)
    positions = builder.get_player_positions()
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Layout: spring layout weighted by pass count
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Node sizes based on total degree
    degrees = dict(G.degree(weight='weight'))
    max_degree = max(degrees.values()) if degrees else 1
    node_sizes = [300 + 1500 * (degrees.get(n, 0) / max_degree) for n in G.nodes()]
    
    # Node colors based on betweenness
    betweenness = nx.betweenness_centrality(G, weight='weight')
    node_colors = [betweenness.get(n, 0) for n in G.nodes()]
    
    # Edge widths based on weight (pass count + pre-shot bonus)
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [0.5 + 4 * (w / max_weight) for w in edge_weights]
    
    # Edge colors based on pre-shot ratio
    edge_colors = [G[u][v].get('pre_shot_ratio', 0) for u, v in G.edges()]
    
    # Draw edges
    edges = nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        edge_color=edge_colors,
        edge_cmap=plt.cm.Reds,
        alpha=0.6,
        arrows=True,
        arrowsize=15,
        connectionstyle="arc3,rad=0.1"
    )
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.YlOrRd,
        alpha=0.9,
        edgecolors='black',
        linewidths=2
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=8,
        font_weight='bold'
    )
    
    # Colorbar for betweenness
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, 
                               norm=plt.Normalize(vmin=0, vmax=max(node_colors) if node_colors else 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, label='Betweenness Centrality')
    
    # Title and annotations
    ax.set_title(f"Bayer 04 Leverkusen Passing Network\nvs {opponent}", 
                fontsize=14, fontweight='bold')
    
    # Add legend for hub player
    top_hub = max(betweenness, key=betweenness.get)
    ax.text(0.02, 0.98, f"Key Hub: {top_hub}\nBetweenness: {betweenness[top_hub]:.3f}",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'leverkusen_network.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'leverkusen_network.png'}")
    
    return G, metrics


def plot_cohesion_trajectory(lev_df):
    """
    Plot Leverkusen's cohesion score across the season.
    """
    print("\nPlotting cohesion trajectory...")
    
    # Sort by match order (using match_id as proxy)
    lev_df = lev_df.sort_values('match_id').reset_index(drop=True)
    lev_df['matchday'] = range(1, len(lev_df) + 1)
    
    # Calculate rolling average
    lev_df['rolling_cohesion'] = lev_df['cohesion_total'].rolling(window=5, min_periods=1).mean()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Color by result
    colors = {'win': '#27ae60', 'draw': '#f39c12', 'loss': '#e74c3c'}
    result_colors = [colors.get(r, 'gray') for r in lev_df['result']]
    
    # Plot individual matches
    ax.scatter(lev_df['matchday'], lev_df['cohesion_total'], 
              c=result_colors, s=100, zorder=3, edgecolors='black', alpha=0.8)
    
    # Plot rolling average
    ax.plot(lev_df['matchday'], lev_df['rolling_cohesion'], 
           'b-', linewidth=2, label='5-match rolling avg', zorder=2)
    
    # Add league average line
    full_df = pd.read_csv(DATA_DIR / 'cohesion_results.csv')
    league_avg = full_df['cohesion_total'].mean()
    ax.axhline(y=league_avg, color='gray', linestyle='--', 
               label=f'League avg: {league_avg:.3f}', alpha=0.7)
    
    # Legend for result colors
    legend_elements = [
        mpatches.Patch(color='#27ae60', label='Win'),
        mpatches.Patch(color='#f39c12', label='Draw'),
        plt.Line2D([0], [0], color='b', linewidth=2, label='5-match rolling avg'),
        plt.Line2D([0], [0], color='gray', linestyle='--', label=f'League avg: {league_avg:.3f}')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    ax.set_xlabel('Matchday', fontsize=12)
    ax.set_ylabel('Cohesion Score', fontsize=12)
    ax.set_title("Bayer 04 Leverkusen: Cohesion Trajectory\nUndefeated Season 2023/24", 
                fontsize=14, fontweight='bold')
    
    # Stats annotation
    stats_text = (f"Season Stats:\n"
                 f"Mean: {lev_df['cohesion_total'].mean():.3f}\n"
                 f"Std: {lev_df['cohesion_total'].std():.3f}\n"
                 f"Min: {lev_df['cohesion_total'].min():.3f}\n"
                 f"Max: {lev_df['cohesion_total'].max():.3f}")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'leverkusen_trajectory.png', dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'leverkusen_trajectory.png'}")


def compare_to_league(lev_df, full_df):
    """
    Compare Leverkusen's metrics to league averages.
    """
    print("\n" + "=" * 60)
    print("LEVERKUSEN vs LEAGUE COMPARISON")
    print("=" * 60)
    
    metrics = ['cohesion_total', 'cohesion_connectivity', 'cohesion_chemistry', 
               'cohesion_balance', 'cohesion_progression', 'density', 'max_betweenness']
    
    comparison = []
    for metric in metrics:
        lev_mean = lev_df[metric].mean()
        league_mean = full_df[metric].mean()
        league_std = full_df[metric].std()
        z_score = (lev_mean - league_mean) / league_std if league_std > 0 else 0
        
        comparison.append({
            'Metric': metric.replace('cohesion_', '').replace('_', ' ').title(),
            'Leverkusen': f"{lev_mean:.3f}",
            'League Avg': f"{league_mean:.3f}",
            'Z-Score': f"{z_score:+.2f}",
            'Significant': '***' if abs(z_score) > 2.58 else '**' if abs(z_score) > 1.96 else '*' if abs(z_score) > 1.64 else ''
        })
    
    comp_df = pd.DataFrame(comparison)
    print(comp_df.to_string(index=False))
    
    return comp_df


def analyze_key_connections(match_ids, n_matches=10):
    """
    Find the most important passing connections for Leverkusen.
    """
    print(f"\nAnalyzing key connections across {n_matches} matches...")
    
    edge_stats = defaultdict(lambda: {'count': 0, 'pre_shot_count': 0, 'total_weight': 0})
    
    sample_ids = match_ids[:n_matches]
    
    for match_id in tqdm(sample_ids, desc="Processing"):
        try:
            events = impect.load_open_data(match_id=match_id)
            results = analyze_match_from_kloppy(events)
            
            for team_name, data in results.items():
                if 'Leverkusen' in team_name:
                    G = data['network']
                    
                    for u, v, d in G.edges(data=True):
                        edge_key = f"{u} → {v}"
                        edge_stats[edge_key]['count'] += d.get('count', 0)
                        edge_stats[edge_key]['pre_shot_count'] += d.get('pre_shot_count', 0)
                        edge_stats[edge_key]['total_weight'] += d.get('weight', 0)
                    
                    break
                    
        except Exception as e:
            continue
    
    # Convert to DataFrame
    connections = []
    for edge, stats in edge_stats.items():
        pre_shot_ratio = stats['pre_shot_count'] / stats['count'] if stats['count'] > 0 else 0
        connections.append({
            'connection': edge,
            'total_passes': stats['count'],
            'pre_shot_passes': stats['pre_shot_count'],
            'pre_shot_ratio': pre_shot_ratio,
            'total_weight': stats['total_weight']
        })
    
    conn_df = pd.DataFrame(connections)
    
    print("\n" + "=" * 60)
    print("TOP 10 PASSING CONNECTIONS (by volume)")
    print("=" * 60)
    print(conn_df.sort_values('total_passes', ascending=False).head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("TOP 10 PRE-SHOT CONNECTIONS (passes leading to shots)")
    print("=" * 60)
    print(conn_df.sort_values('pre_shot_passes', ascending=False).head(10).to_string(index=False))
    
    return conn_df


def main():
    print("=" * 60)
    print("LEVERKUSEN CASE STUDY: UNDEFEATED SEASON ANALYSIS")
    print("=" * 60)
    
    # Load data
    lev_df, full_df = load_leverkusen_data()
    
    if lev_df is None:
        return
    
    # Basic stats
    print("\n" + "=" * 60)
    print("SEASON SUMMARY")
    print("=" * 60)
    print(f"Matches played: {len(lev_df)}")
    print(f"Wins: {len(lev_df[lev_df['result'] == 'win'])}")
    print(f"Draws: {len(lev_df[lev_df['result'] == 'draw'])}")
    print(f"Losses: {len(lev_df[lev_df['result'] == 'loss'])}")
    print(f"Total points: {lev_df['points'].sum()}")
    print(f"Goals for: {lev_df['goals_for'].sum()}")
    print(f"Goals against: {lev_df['goals_against'].sum()}")
    
    # Compare to league
    compare_to_league(lev_df, full_df)
    
    # Plot trajectory
    plot_cohesion_trajectory(lev_df)
    
    # Get match IDs
    match_ids = find_leverkusen_match_ids(full_df)
    
    # Analyze hub players
    print("\n" + "=" * 60)
    print("HUB PLAYER ANALYSIS")
    print("=" * 60)
    hub_df = analyze_hub_players(match_ids, n_matches=15)
    print("\nTop 10 Hub Players (by avg betweenness centrality):")
    print(hub_df.head(10).to_string(index=False))
    
    # Analyze key connections
    conn_df = analyze_key_connections(match_ids, n_matches=15)
    
    # Visualize network
    visualize_leverkusen_network(match_ids[0])
    
    print("\n" + "=" * 60)
    print("CASE STUDY COMPLETE")
    print("=" * 60)
    print("\nKey findings:")
    print("1. Hub players identified - check who has highest betweenness")
    print("2. Cohesion trajectory shows consistency across season")
    print("3. Key passing connections reveal tactical patterns")
    print("\nFigures generated:")
    print("  - figures/leverkusen_trajectory.png")
    print("  - figures/leverkusen_network.png")


if __name__ == "__main__":
    main()
