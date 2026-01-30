"""
WHAT-IF LINEUP SIMULATOR: Player Impact Analysis
=================================================

This script answers:
1. "What happens to team cohesion if Player X is removed?"
2. "Which player's absence hurts the team most?"
3. "How replaceable is each player?"

Key concept: For each player, we remove them from the network and 
recalculate cohesion. The DROP in cohesion = player's importance.

Run from notebooks folder: python 08_whatif_simulator.py
"""

import sys
sys.path.insert(0, '../src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


def remove_player_from_network(G: nx.DiGraph, player_name: str) -> nx.DiGraph:
    """
    Remove a player (node) and all their connections from the network.
    
    Returns a new graph without modifying the original.
    """
    G_modified = G.copy()
    if player_name in G_modified.nodes():
        G_modified.remove_node(player_name)
    return G_modified


def calculate_player_impact(G: nx.DiGraph, player_positions: dict = None) -> pd.DataFrame:
    """
    For each player in the network, calculate their impact on cohesion.
    
    Impact = Cohesion(full team) - Cohesion(team without player)
    
    Higher impact = more important player
    """
    calculator = CohesionCalculator(use_optimized=True)
    
    # Baseline: full team cohesion
    baseline_score = calculator.calculate(G, player_positions)
    baseline_total = baseline_score.total
    
    impacts = []
    
    for player in G.nodes():
        # Remove player
        G_without = remove_player_from_network(G, player)
        
        # Recalculate cohesion
        new_score = calculator.calculate(G_without, player_positions)
        
        # Impact = drop in cohesion
        impact = baseline_total - new_score.total
        impact_pct = (impact / baseline_total) * 100 if baseline_total > 0 else 0
        
        # Also track component-level impacts
        connectivity_impact = baseline_score.connectivity - new_score.connectivity
        chemistry_impact = baseline_score.chemistry - new_score.chemistry
        hub_impact = baseline_score.balance - new_score.balance
        progression_impact = baseline_score.progression - new_score.progression
        
        # Get player's network stats
        degree = G.degree(player, weight='weight')
        in_degree = G.in_degree(player, weight='weight')
        out_degree = G.out_degree(player, weight='weight')
        betweenness = nx.betweenness_centrality(G, weight='weight').get(player, 0)
        
        impacts.append({
            'player': player,
            'impact_score': impact,
            'impact_pct': impact_pct,
            'connectivity_impact': connectivity_impact,
            'chemistry_impact': chemistry_impact,
            'hub_impact': hub_impact,
            'progression_impact': progression_impact,
            'total_passes': degree,
            'passes_received': in_degree,
            'passes_made': out_degree,
            'betweenness': betweenness
        })
    
    impact_df = pd.DataFrame(impacts)
    impact_df = impact_df.sort_values('impact_score', ascending=False)
    
    return impact_df, baseline_score


def analyze_team_player_impacts(full_df: pd.DataFrame, team_name_pattern: str, n_matches: int = 10):
    """
    Analyze player impacts across multiple matches for a team.
    """
    match_ids = full_df[full_df['team_name'].str.contains(team_name_pattern, case=False)]['match_id'].unique()
    
    print(f"\nAnalyzing player impacts for {team_name_pattern} across {min(n_matches, len(match_ids))} matches...")
    
    all_impacts = defaultdict(list)
    
    for match_id in tqdm(match_ids[:n_matches], desc="Processing"):
        try:
            events = impect.load_open_data(match_id=match_id)
            results = analyze_match_from_kloppy(events)
            
            for team_name, data in results.items():
                if team_name_pattern.lower() in team_name.lower():
                    G = data['network']
                    positions = data.get('positions', {})
                    
                    impact_df, baseline = calculate_player_impact(G, positions)
                    
                    for _, row in impact_df.iterrows():
                        all_impacts[row['player']].append({
                            'impact_score': row['impact_score'],
                            'impact_pct': row['impact_pct'],
                            'betweenness': row['betweenness'],
                            'total_passes': row['total_passes']
                        })
                    break
                    
        except Exception as e:
            continue
    
    # Aggregate across matches
    aggregated = []
    for player, impacts in all_impacts.items():
        if len(impacts) >= 3:  # Only players in 3+ matches
            aggregated.append({
                'player': player,
                'appearances': len(impacts),
                'avg_impact': np.mean([i['impact_score'] for i in impacts]),
                'max_impact': np.max([i['impact_score'] for i in impacts]),
                'avg_impact_pct': np.mean([i['impact_pct'] for i in impacts]),
                'avg_betweenness': np.mean([i['betweenness'] for i in impacts]),
                'avg_passes': np.mean([i['total_passes'] for i in impacts]),
                'consistency': np.std([i['impact_score'] for i in impacts])  # Lower = more consistent
            })
    
    agg_df = pd.DataFrame(aggregated)
    agg_df = agg_df.sort_values('avg_impact', ascending=False)
    
    return agg_df


def visualize_player_impacts(impact_df: pd.DataFrame, team_name: str, baseline_score):
    """
    Create visualization of player impacts for a single match.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Top 10 by impact
    top_players = impact_df.head(10)
    
    # 1. Impact bar chart
    ax1 = axes[0]
    colors = ['#e74c3c' if x > 0 else '#3498db' for x in top_players['impact_score']]
    bars = ax1.barh(range(len(top_players)), top_players['impact_score'], color=colors, alpha=0.8)
    ax1.set_yticks(range(len(top_players)))
    ax1.set_yticklabels(top_players['player'])
    ax1.invert_yaxis()
    ax1.set_xlabel('Cohesion Impact (drop when removed)')
    ax1.set_title(f'{team_name}: Player Impact Ranking\nBaseline Cohesion: {baseline_score.total:.3f}')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars, top_players['impact_pct'])):
        ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{pct:.1f}%', va='center', fontsize=9)
    
    # 2. Impact vs Betweenness scatter
    ax2 = axes[1]
    scatter = ax2.scatter(impact_df['betweenness'], impact_df['impact_score'], 
                         s=impact_df['total_passes']/2, alpha=0.6, c='#2ecc71', edgecolors='black')
    
    # Label top 5 players
    for _, row in impact_df.head(5).iterrows():
        ax2.annotate(row['player'], (row['betweenness'], row['impact_score']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax2.set_xlabel('Betweenness Centrality')
    ax2.set_ylabel('Cohesion Impact')
    ax2.set_title('Impact vs Hub Position\n(bubble size = pass volume)')
    
    plt.suptitle(f'{team_name}: Who Matters Most?', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filename = f'{team_name.lower().replace(" ", "_")}_player_impacts.png'
    plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / filename}")


def visualize_aggregated_impacts(agg_df: pd.DataFrame, team_name: str):
    """
    Create visualization of season-aggregated player impacts.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    top_15 = agg_df.head(15)
    
    # 1. Average impact ranking
    ax1 = axes[0]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_15)))
    bars = ax1.barh(range(len(top_15)), top_15['avg_impact'], color=colors, alpha=0.8, edgecolor='black')
    ax1.set_yticks(range(len(top_15)))
    ax1.set_yticklabels(top_15['player'])
    ax1.invert_yaxis()
    ax1.set_xlabel('Average Cohesion Impact')
    ax1.set_title(f'{team_name}: Most Important Players\n(Season Average)')
    
    # Add appearance count
    for i, (bar, apps) in enumerate(zip(bars, top_15['appearances'])):
        ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'({apps} games)', va='center', fontsize=8, alpha=0.7)
    
    # 2. Impact vs Consistency
    ax2 = axes[1]
    scatter = ax2.scatter(agg_df['consistency'], agg_df['avg_impact'], 
                         s=agg_df['appearances']*10, alpha=0.6, c=agg_df['avg_betweenness'],
                         cmap='YlOrRd', edgecolors='black')
    
    # Label top players
    for _, row in agg_df.head(5).iterrows():
        ax2.annotate(row['player'], (row['consistency'], row['avg_impact']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Consistency (lower = more reliable impact)')
    ax2.set_ylabel('Average Impact')
    ax2.set_title('Impact vs Reliability\n(color = hub centrality, size = appearances)')
    
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Avg Betweenness')
    
    plt.suptitle(f'{team_name}: Season-Long Player Value', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filename = f'{team_name.lower().replace(" ", "_")}_season_impacts.png'
    plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / filename}")


def create_what_if_report(impact_df: pd.DataFrame, team_name: str, baseline_score):
    """
    Generate a text report of what-if scenarios.
    """
    print("\n" + "=" * 70)
    print(f"WHAT-IF ANALYSIS: {team_name}")
    print("=" * 70)
    print(f"\nBaseline Team Cohesion: {baseline_score.total:.3f}")
    print(f"  - Connectivity:  {baseline_score.connectivity:.3f}")
    print(f"  - Chemistry:     {baseline_score.chemistry:.3f}")
    print(f"  - Hub Dependence:{baseline_score.balance:.3f}")
    print(f"  - Progression:   {baseline_score.progression:.3f}")
    
    print("\n" + "-" * 70)
    print("TOP 5 MOST IMPACTFUL PLAYERS (if removed, cohesion drops by...):")
    print("-" * 70)
    
    for i, row in impact_df.head(5).iterrows():
        print(f"\n{row['player']}:")
        print(f"  Total Impact: -{row['impact_score']:.3f} ({row['impact_pct']:.1f}% drop)")
        print(f"  Passes involved: {row['total_passes']:.0f}")
        print(f"  Betweenness: {row['betweenness']:.3f}")
        
        # Interpretation
        if row['impact_pct'] > 10:
            print(f"  ⚠️  CRITICAL: Team heavily depends on this player")
        elif row['impact_pct'] > 5:
            print(f"  ⚡ IMPORTANT: Significant contributor")
        else:
            print(f"  ✓  MODERATE: Valuable but replaceable")
    
    print("\n" + "-" * 70)
    print("MOST REPLACEABLE PLAYERS (lowest impact):")
    print("-" * 70)
    
    for i, row in impact_df.tail(3).iterrows():
        print(f"  {row['player']}: {row['impact_pct']:.1f}% impact")


def compare_key_players(full_df: pd.DataFrame, team_name: str, player_names: list, n_matches: int = 10):
    """
    Compare specific players' impacts head-to-head.
    """
    print(f"\n" + "=" * 70)
    print(f"HEAD-TO-HEAD: {' vs '.join(player_names)}")
    print("=" * 70)
    
    match_ids = full_df[full_df['team_name'].str.contains(team_name, case=False)]['match_id'].unique()
    
    player_impacts = {name: [] for name in player_names}
    
    for match_id in tqdm(match_ids[:n_matches], desc="Processing"):
        try:
            events = impect.load_open_data(match_id=match_id)
            results = analyze_match_from_kloppy(events)
            
            for t_name, data in results.items():
                if team_name.lower() in t_name.lower():
                    G = data['network']
                    impact_df, baseline = calculate_player_impact(G)
                    
                    for player in player_names:
                        match_row = impact_df[impact_df['player'].str.contains(player, case=False)]
                        if len(match_row) > 0:
                            player_impacts[player].append(match_row.iloc[0]['impact_score'])
                    break
                    
        except Exception as e:
            continue
    
    # Report
    print("\nResults:")
    for player, impacts in player_impacts.items():
        if impacts:
            print(f"\n{player}:")
            print(f"  Matches analyzed: {len(impacts)}")
            print(f"  Average impact: {np.mean(impacts):.4f}")
            print(f"  Max impact: {np.max(impacts):.4f}")
            print(f"  Min impact: {np.min(impacts):.4f}")


def main():
    print("=" * 70)
    print("WHAT-IF LINEUP SIMULATOR")
    print("=" * 70)
    
    # Load data
    results_file = DATA_DIR / 'cohesion_results.csv'
    if not results_file.exists():
        print("ERROR: Run 03_full_analysis.py first")
        return
    
    full_df = pd.read_csv(results_file)
    
    # =========================================
    # LEVERKUSEN ANALYSIS
    # =========================================
    print("\n" + "=" * 70)
    print("LEVERKUSEN: WHO MATTERS MOST?")
    print("=" * 70)
    
    # Single match deep dive
    lev_match_ids = full_df[full_df['team_name'].str.contains('Leverkusen')]['match_id'].unique()
    
    print("\nAnalyzing a sample Leverkusen match...")
    events = impect.load_open_data(match_id=lev_match_ids[0])
    results = analyze_match_from_kloppy(events)
    
    for team_name, data in results.items():
        if 'Leverkusen' in team_name:
            G = data['network']
            positions = data.get('positions', {})
            
            impact_df, baseline = calculate_player_impact(G, positions)
            
            create_what_if_report(impact_df, team_name, baseline)
            visualize_player_impacts(impact_df, team_name, baseline)
            break
    
    # Season aggregated
    print("\n" + "-" * 70)
    print("SEASON-AGGREGATED PLAYER IMPACTS")
    print("-" * 70)
    
    lev_agg = analyze_team_player_impacts(full_df, 'Leverkusen', n_matches=15)
    print("\nTop 10 Most Important Players (season average):")
    print(lev_agg.head(10).to_string(index=False))
    
    visualize_aggregated_impacts(lev_agg, 'Leverkusen')
    
    # =========================================
    # KEY PLAYER COMPARISON
    # =========================================
    print("\n" + "=" * 70)
    print("KEY PLAYER COMPARISON: WIRTZ vs XHAKA")
    print("=" * 70)
    
    compare_key_players(full_df, 'Leverkusen', ['Wirtz', 'Xhaka'], n_matches=15)
    
    # =========================================
    # DARMSTADT COMPARISON
    # =========================================
    print("\n" + "=" * 70)
    print("DARMSTADT: WHO COULD THEY NOT AFFORD TO LOSE?")
    print("=" * 70)
    
    darm_agg = analyze_team_player_impacts(full_df, 'Darmstadt', n_matches=15)
    print("\nTop 10 Most Important Players (season average):")
    print(darm_agg.head(10).to_string(index=False))
    
    visualize_aggregated_impacts(darm_agg, 'Darmstadt')
    
    # =========================================
    # SUMMARY
    # =========================================
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    
    if len(lev_agg) > 0 and len(darm_agg) > 0:
        lev_top_impact = lev_agg.iloc[0]['avg_impact'] if len(lev_agg) > 0 else 0
        darm_top_impact = darm_agg.iloc[0]['avg_impact'] if len(darm_agg) > 0 else 0
        
        print(f"""
1. LEVERKUSEN'S KEY PLAYER: {lev_agg.iloc[0]['player']}
   - Average impact: {lev_agg.iloc[0]['avg_impact']:.4f}
   - Appeared in {lev_agg.iloc[0]['appearances']} matches
   - Removing them drops cohesion by ~{lev_agg.iloc[0]['avg_impact_pct']:.1f}%

2. DARMSTADT'S KEY PLAYER: {darm_agg.iloc[0]['player']}
   - Average impact: {darm_agg.iloc[0]['avg_impact']:.4f}
   - Appeared in {darm_agg.iloc[0]['appearances']} matches
   - Removing them drops cohesion by ~{darm_agg.iloc[0]['avg_impact_pct']:.1f}%

3. COMPARISON:
   - Leverkusen's top player is {lev_top_impact/darm_top_impact:.1f}x more impactful
   - Elite teams have MORE irreplaceable players, not fewer
""")
    
    print("\n" + "=" * 70)
    print("FIGURES GENERATED:")
    print("=" * 70)
    print("  - figures/bayer_04_leverkusen_player_impacts.png")
    print("  - figures/leverkusen_season_impacts.png")
    print("  - figures/darmstadt_season_impacts.png")


if __name__ == "__main__":
    main()