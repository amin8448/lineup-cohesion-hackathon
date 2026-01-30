"""
DARMSTADT CASE STUDY: Anatomy of a Relegated Season
====================================================

This script analyzes SV Darmstadt 98's 2023/24 Bundesliga campaign
(last place, 21 points) through the lens of network cohesion.

Comparison to Leverkusen (1st, 90 points) to understand:
1. How does network structure differ between best and worst teams?
2. Do relegated teams lack hub players, or just different patterns?
3. What specific network weaknesses predict poor performance?

Run from notebooks folder: python 07_darmstadt_case_study.py
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


def load_team_data(team_name_pattern):
    """Load data for a specific team from cached results."""
    
    results_file = DATA_DIR / 'cohesion_results.csv'
    
    if results_file.exists():
        print(f"Loading cached results...")
        df = pd.read_csv(results_file)
        team_df = df[df['team_name'].str.contains(team_name_pattern, case=False)].copy()
        print(f"Found {len(team_df)} matches for {team_name_pattern}")
        return team_df, df
    else:
        print("ERROR: Run 03_full_analysis.py first to generate cohesion_results.csv")
        return None, None


def find_team_match_ids(df, team_name_pattern):
    """Get all match IDs where team played."""
    return df[df['team_name'].str.contains(team_name_pattern, case=False)]['match_id'].unique().tolist()


def analyze_hub_players(df, team_name_pattern, n_matches=10):
    """
    Analyze hub players across multiple matches.
    """
    match_ids = find_team_match_ids(df, team_name_pattern)
    print(f"\nAnalyzing hub players across {min(n_matches, len(match_ids))} matches...")
    
    player_betweenness = defaultdict(list)
    player_degree = defaultdict(list)
    player_appearances = defaultdict(int)
    
    sample_ids = match_ids[:n_matches]
    
    for match_id in tqdm(sample_ids, desc="Processing"):
        try:
            events = impect.load_open_data(match_id=match_id)
            results = analyze_match_from_kloppy(events)
            
            for team_name, data in results.items():
                if team_name_pattern.lower() in team_name.lower():
                    G = data['network']
                    metrics = data['metrics']
                    
                    for player, bc in metrics.get('betweenness', {}).items():
                        player_betweenness[player].append(bc)
                        player_appearances[player] += 1
                    
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


def visualize_team_network(df, team_name_pattern, match_id=None):
    """
    Create a visualization of team's passing network.
    """
    if match_id is None:
        match_ids = find_team_match_ids(df, team_name_pattern)
        match_id = match_ids[0]
    
    print(f"\nLoading match {match_id} for network visualization...")
    
    events = impect.load_open_data(match_id=match_id)
    player_id_to_name, player_id_to_position, team_id_to_name = extract_metadata(events)
    
    df_events = events.to_df(engine="pandas")
    
    # Find team_id
    team_id = None
    team_full_name = None
    for tid, name in team_id_to_name.items():
        if team_name_pattern.lower() in name.lower():
            team_id = tid
            team_full_name = name
            break
    
    if team_id is None:
        print(f"ERROR: {team_name_pattern} not found in this match")
        return None, None
    
    # Build network
    builder = PassingNetworkBuilder(
        df_events, team_id,
        player_id_to_name, player_id_to_position, team_id_to_name
    )
    G = builder.build_network()
    
    # Get opponent name
    opponent = [name for name in team_id_to_name.values() if team_name_pattern.lower() not in name.lower()][0]
    
    # Calculate metrics
    metrics = compute_network_metrics(G)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 10))
    
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    degrees = dict(G.degree(weight='weight'))
    max_degree = max(degrees.values()) if degrees else 1
    node_sizes = [300 + 1500 * (degrees.get(n, 0) / max_degree) for n in G.nodes()]
    
    betweenness = nx.betweenness_centrality(G, weight='weight')
    node_colors = [betweenness.get(n, 0) for n in G.nodes()]
    
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [0.5 + 4 * (w / max_weight) for w in edge_weights]
    
    edge_colors = [G[u][v].get('pre_shot_ratio', 0) for u, v in G.edges()]
    
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
    
    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.YlOrRd,
        alpha=0.9,
        edgecolors='black',
        linewidths=2
    )
    
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=8,
        font_weight='bold'
    )
    
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, 
                               norm=plt.Normalize(vmin=0, vmax=max(node_colors) if node_colors else 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, label='Betweenness Centrality')
    
    ax.set_title(f"{team_full_name} Passing Network\nvs {opponent}", 
                fontsize=14, fontweight='bold')
    
    if betweenness:
        top_hub = max(betweenness, key=betweenness.get)
        ax.text(0.02, 0.98, f"Key Hub: {top_hub}\nBetweenness: {betweenness[top_hub]:.3f}",
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.axis('off')
    plt.tight_layout()
    
    filename = f'{team_name_pattern.lower()}_network.png'
    plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / filename}")
    
    return G, metrics


def plot_cohesion_trajectory(team_df, team_name, full_df):
    """
    Plot team's cohesion score across the season.
    """
    print("\nPlotting cohesion trajectory...")
    
    team_df = team_df.sort_values('match_id').reset_index(drop=True)
    team_df['matchday'] = range(1, len(team_df) + 1)
    team_df['rolling_cohesion'] = team_df['cohesion_total'].rolling(window=5, min_periods=1).mean()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = {'win': '#27ae60', 'draw': '#f39c12', 'loss': '#e74c3c'}
    result_colors = [colors.get(r, 'gray') for r in team_df['result']]
    
    ax.scatter(team_df['matchday'], team_df['cohesion_total'], 
              c=result_colors, s=100, zorder=3, edgecolors='black', alpha=0.8)
    
    ax.plot(team_df['matchday'], team_df['rolling_cohesion'], 
           'b-', linewidth=2, label='5-match rolling avg', zorder=2)
    
    league_avg = full_df['cohesion_total'].mean()
    ax.axhline(y=league_avg, color='gray', linestyle='--', 
               label=f'League avg: {league_avg:.3f}', alpha=0.7)
    
    legend_elements = [
        mpatches.Patch(color='#27ae60', label='Win'),
        mpatches.Patch(color='#f39c12', label='Draw'),
        mpatches.Patch(color='#e74c3c', label='Loss'),
        plt.Line2D([0], [0], color='b', linewidth=2, label='5-match rolling avg'),
        plt.Line2D([0], [0], color='gray', linestyle='--', label=f'League avg: {league_avg:.3f}')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    ax.set_xlabel('Matchday', fontsize=12)
    ax.set_ylabel('Cohesion Score', fontsize=12)
    ax.set_title(f"{team_name}: Cohesion Trajectory\nBundesliga 2023/24", 
                fontsize=14, fontweight='bold')
    
    stats_text = (f"Season Stats:\n"
                 f"Mean: {team_df['cohesion_total'].mean():.3f}\n"
                 f"Std: {team_df['cohesion_total'].std():.3f}\n"
                 f"Min: {team_df['cohesion_total'].min():.3f}\n"
                 f"Max: {team_df['cohesion_total'].max():.3f}")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    filename = f'{team_name.lower().replace(" ", "_")}_trajectory.png'
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR / filename}")


def compare_teams(team1_df, team2_df, team1_name, team2_name, full_df):
    """
    Create head-to-head comparison between two teams.
    """
    print("\n" + "=" * 70)
    print(f"COMPARISON: {team1_name} vs {team2_name}")
    print("=" * 70)
    
    metrics = ['cohesion_total', 'cohesion_connectivity', 'cohesion_chemistry', 
               'cohesion_balance', 'cohesion_progression', 'density', 'max_betweenness',
               'clustering']
    
    comparison = []
    for metric in metrics:
        if metric in team1_df.columns and metric in team2_df.columns:
            t1_mean = team1_df[metric].mean()
            t2_mean = team2_df[metric].mean()
            league_mean = full_df[metric].mean()
            diff = t1_mean - t2_mean
            
            comparison.append({
                'Metric': metric.replace('cohesion_', '').replace('_', ' ').title(),
                team1_name: f"{t1_mean:.3f}",
                team2_name: f"{t2_mean:.3f}",
                'League Avg': f"{league_mean:.3f}",
                'Difference': f"{diff:+.3f}"
            })
    
    comp_df = pd.DataFrame(comparison)
    print(comp_df.to_string(index=False))
    
    return comp_df


def analyze_key_connections(df, team_name_pattern, n_matches=10):
    """
    Find the most important passing connections for a team.
    """
    match_ids = find_team_match_ids(df, team_name_pattern)
    print(f"\nAnalyzing key connections across {min(n_matches, len(match_ids))} matches...")
    
    edge_stats = defaultdict(lambda: {'count': 0, 'pre_shot_count': 0, 'total_weight': 0})
    
    sample_ids = match_ids[:n_matches]
    
    for match_id in tqdm(sample_ids, desc="Processing"):
        try:
            events = impect.load_open_data(match_id=match_id)
            results = analyze_match_from_kloppy(events)
            
            for team_name, data in results.items():
                if team_name_pattern.lower() in team_name.lower():
                    G = data['network']
                    
                    for u, v, d in G.edges(data=True):
                        edge_key = f"{u} → {v}"
                        edge_stats[edge_key]['count'] += d.get('count', 0)
                        edge_stats[edge_key]['pre_shot_count'] += d.get('pre_shot_count', 0)
                        edge_stats[edge_key]['total_weight'] += d.get('weight', 0)
                    
                    break
                    
        except Exception as e:
            continue
    
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
    
    return conn_df


def create_comparison_visualization(darm_df, lev_df, full_df):
    """
    Create side-by-side visualization comparing Darmstadt and Leverkusen.
    """
    print("\nCreating comparison visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Cohesion components comparison (bar chart)
    ax1 = axes[0, 0]
    components = ['connectivity', 'chemistry', 'balance', 'progression']
    
    darm_values = [darm_df[f'cohesion_{c}'].mean() for c in components]
    lev_values = [lev_df[f'cohesion_{c}'].mean() for c in components]
    
    x = np.arange(len(components))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, darm_values, width, label='Darmstadt (21 pts)', color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x + width/2, lev_values, width, label='Leverkusen (90 pts)', color='#27ae60', alpha=0.8)
    
    ax1.set_ylabel('Score')
    ax1.set_title('Cohesion Components: Last vs First Place')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.title() for c in components])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Total cohesion distribution (box plot)
    ax2 = axes[0, 1]
    data_to_plot = [darm_df['cohesion_total'], lev_df['cohesion_total']]
    bp = ax2.boxplot(data_to_plot, labels=['Darmstadt', 'Leverkusen'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#e74c3c')
    bp['boxes'][1].set_facecolor('#27ae60')
    ax2.axhline(y=full_df['cohesion_total'].mean(), color='gray', linestyle='--', label='League avg')
    ax2.set_ylabel('Cohesion Score')
    ax2.set_title('Cohesion Distribution: Season-wide')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Hub dependence (max betweenness) comparison
    ax3 = axes[1, 0]
    darm_bet = darm_df['max_betweenness'].values
    lev_bet = lev_df['max_betweenness'].values
    
    ax3.hist(darm_bet, bins=15, alpha=0.6, label='Darmstadt', color='#e74c3c')
    ax3.hist(lev_bet, bins=15, alpha=0.6, label='Leverkusen', color='#27ae60')
    ax3.axvline(x=darm_df['max_betweenness'].mean(), color='#e74c3c', linestyle='--', linewidth=2)
    ax3.axvline(x=lev_df['max_betweenness'].mean(), color='#27ae60', linestyle='--', linewidth=2)
    ax3.set_xlabel('Max Betweenness Centrality')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Hub Dependence Distribution')
    ax3.legend()
    
    # 4. Results breakdown
    ax4 = axes[1, 1]
    
    darm_results = darm_df['result'].value_counts()
    lev_results = lev_df['result'].value_counts()
    
    results = ['win', 'draw', 'loss']
    darm_counts = [darm_results.get(r, 0) for r in results]
    lev_counts = [lev_results.get(r, 0) for r in results]
    
    x = np.arange(len(results))
    bars1 = ax4.bar(x - width/2, darm_counts, width, label='Darmstadt', color='#e74c3c', alpha=0.8)
    bars2 = ax4.bar(x + width/2, lev_counts, width, label='Leverkusen', color='#27ae60', alpha=0.8)
    
    ax4.set_ylabel('Count')
    ax4.set_title('Match Results Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Win', 'Draw', 'Loss'])
    ax4.legend()
    
    # Add count labels on bars
    for bar, count in zip(bars1, darm_counts):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(count),
                ha='center', va='bottom', fontsize=10)
    for bar, count in zip(bars2, lev_counts):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(count),
                ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Darmstadt vs Leverkusen: Network Cohesion Analysis\nBundesliga 2023/24', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'darmstadt_vs_leverkusen_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'darmstadt_vs_leverkusen_comparison.png'}")


def main():
    print("=" * 70)
    print("DARMSTADT CASE STUDY: WHY DID THEY GET RELEGATED?")
    print("=" * 70)
    
    # Load Darmstadt data
    darm_df, full_df = load_team_data('Darmstadt')
    
    if darm_df is None:
        return
    
    # Load Leverkusen for comparison
    lev_df, _ = load_team_data('Leverkusen')
    
    # =========================================
    # DARMSTADT SEASON SUMMARY
    # =========================================
    print("\n" + "=" * 70)
    print("DARMSTADT SEASON SUMMARY")
    print("=" * 70)
    print(f"Matches played: {len(darm_df)}")
    print(f"Wins: {len(darm_df[darm_df['result'] == 'win'])}")
    print(f"Draws: {len(darm_df[darm_df['result'] == 'draw'])}")
    print(f"Losses: {len(darm_df[darm_df['result'] == 'loss'])}")
    print(f"Total points: {darm_df['points'].sum()}")
    print(f"Goals for: {darm_df['goals_for'].sum()}")
    print(f"Goals against: {darm_df['goals_against'].sum()}")
    
    # =========================================
    # COMPARE TO LEVERKUSEN
    # =========================================
    compare_teams(darm_df, lev_df, 'Darmstadt', 'Leverkusen', full_df)
    
    # =========================================
    # HUB PLAYER ANALYSIS
    # =========================================
    print("\n" + "=" * 70)
    print("DARMSTADT HUB PLAYER ANALYSIS")
    print("=" * 70)
    darm_hub_df = analyze_hub_players(full_df, 'Darmstadt', n_matches=15)
    print("\nTop 10 Hub Players (by avg betweenness centrality):")
    print(darm_hub_df.head(10).to_string(index=False))
    
    print("\n" + "=" * 70)
    print("LEVERKUSEN HUB PLAYER ANALYSIS (for comparison)")
    print("=" * 70)
    lev_hub_df = analyze_hub_players(full_df, 'Leverkusen', n_matches=15)
    print("\nTop 10 Hub Players (by avg betweenness centrality):")
    print(lev_hub_df.head(10).to_string(index=False))
    
    # =========================================
    # KEY CONNECTIONS ANALYSIS
    # =========================================
    print("\n" + "=" * 70)
    print("DARMSTADT KEY CONNECTIONS")
    print("=" * 70)
    darm_conn_df = analyze_key_connections(full_df, 'Darmstadt', n_matches=15)
    if len(darm_conn_df) > 0:
        print("\nTop 10 Passing Connections (by volume):")
        print(darm_conn_df.sort_values('total_passes', ascending=False).head(10).to_string(index=False))
        
        print("\nTop 10 Pre-Shot Connections:")
        print(darm_conn_df.sort_values('pre_shot_passes', ascending=False).head(10).to_string(index=False))
    
    # =========================================
    # VISUALIZATIONS
    # =========================================
    
    # Cohesion trajectory
    plot_cohesion_trajectory(darm_df, 'SV Darmstadt 98', full_df)
    
    # Network visualization
    visualize_team_network(full_df, 'Darmstadt')
    
    # Comparison visualization
    create_comparison_visualization(darm_df, lev_df, full_df)
    
    # =========================================
    # KEY FINDINGS SUMMARY
    # =========================================
    print("\n" + "=" * 70)
    print("KEY FINDINGS: WHY DARMSTADT FAILED")
    print("=" * 70)
    
    # Calculate differences
    cohesion_diff = lev_df['cohesion_total'].mean() - darm_df['cohesion_total'].mean()
    connectivity_diff = lev_df['cohesion_connectivity'].mean() - darm_df['cohesion_connectivity'].mean()
    hub_diff = lev_df['max_betweenness'].mean() - darm_df['max_betweenness'].mean()
    
    print(f"""
1. LOWER OVERALL COHESION
   - Darmstadt avg: {darm_df['cohesion_total'].mean():.3f}
   - Leverkusen avg: {lev_df['cohesion_total'].mean():.3f}
   - Gap: {cohesion_diff:.3f} ({cohesion_diff/darm_df['cohesion_total'].mean()*100:.1f}% lower)

2. WEAKER CONNECTIVITY
   - Darmstadt: {darm_df['cohesion_connectivity'].mean():.3f}
   - Leverkusen: {lev_df['cohesion_connectivity'].mean():.3f}
   - Less connected passing network = isolated players

3. HUB DEPENDENCE PATTERN
   - Darmstadt max betweenness: {darm_df['max_betweenness'].mean():.3f}
   - Leverkusen max betweenness: {lev_df['max_betweenness'].mean():.3f}
   - Darmstadt lacks a clear "orchestrator" like Xhaka

4. PRE-SHOT PROGRESSION
   - Darmstadt: {darm_df['cohesion_progression'].mean():.3f}
   - Leverkusen: {lev_df['cohesion_progression'].mean():.3f}
   - Fewer passes leading to shots = weaker attack chain
""")
    
    print("\n" + "=" * 70)
    print("FIGURES GENERATED:")
    print("=" * 70)
    print("  - figures/sv_darmstadt_98_trajectory.png")
    print("  - figures/darmstadt_network.png")
    print("  - figures/darmstadt_vs_leverkusen_comparison.png")


if __name__ == "__main__":
    main()