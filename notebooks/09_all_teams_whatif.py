"""
ALL TEAMS WHAT-IF ANALYSIS
==========================
Calculate player impact scores for every team in the Bundesliga.

Output: data/player_impacts.csv
"""

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import sys
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from kloppy import impect

print("=" * 60)
print("ALL TEAMS WHAT-IF ANALYSIS")
print("=" * 60)

# Configuration
DATA_DIR = Path('../data')
OUTPUT_FILE = DATA_DIR / 'player_impacts.csv'

# Component weights (from our validated model)
WEIGHTS = {
    'connectivity': 0.50,
    'chemistry': 0.25,
    'balance': 0.15,
    'progression': 0.10
}


def build_pass_network(events_df, team_id, player_map):
    """Build passing network for a team from events."""
    G = nx.DiGraph()
    
    # Filter to team's successful passes
    team_passes = events_df[
        (events_df['team_id'] == team_id) &
        (events_df['event_type'] == 'PASS') &
        (events_df['result'] == 'COMPLETE')
    ].copy()
    
    if len(team_passes) == 0:
        return G, 0, 0
    
    # Map player IDs to names
    team_passes['player_name'] = team_passes['player_id'].map(player_map)
    team_passes['receiver_name'] = team_passes['receiver_player_id'].map(player_map)
    
    # Count passes between players
    pass_counts = team_passes.groupby(['player_name', 'receiver_name']).size().reset_index(name='weight')
    
    # Build graph
    for _, row in pass_counts.iterrows():
        if pd.notna(row['player_name']) and pd.notna(row['receiver_name']):
            G.add_edge(row['player_name'], row['receiver_name'], weight=row['weight'])
    
    # Count pre-shot passes
    shots = events_df[
        (events_df['team_id'] == team_id) &
        (events_df['event_type'] == 'SHOT')
    ]
    shot_player_ids = shots['player_id'].unique()
    pre_shot = len(team_passes[team_passes['receiver_player_id'].isin(shot_player_ids)])
    
    total_passes = len(team_passes)
    
    return G, pre_shot, total_passes


def calculate_cohesion(G, pre_shot_passes, total_passes):
    """Calculate cohesion score for a network."""
    if G.number_of_nodes() < 3:
        return 0, {}
    
    # 1. Connectivity (density)
    n = G.number_of_nodes()
    max_edges = n * (n - 1)
    density = G.number_of_edges() / max_edges if max_edges > 0 else 0
    connectivity = min(density * 2, 1.0)  # Scale up, cap at 1
    
    # 2. Chemistry (clustering)
    G_undirected = G.to_undirected()
    try:
        clustering = nx.average_clustering(G_undirected, weight='weight')
    except:
        clustering = 0
    chemistry = clustering
    
    # 3. Hub Dependence (betweenness concentration)
    try:
        betweenness = nx.betweenness_centrality(G, weight='weight')
        if betweenness:
            max_betweenness = max(betweenness.values())
            balance = max_betweenness  # Higher = more hub dependent
        else:
            balance = 0
    except:
        balance = 0
    
    # 4. Progression
    progression = pre_shot_passes / total_passes if total_passes > 0 else 0
    progression = min(progression * 10, 1.0)  # Scale up
    
    # Total cohesion
    total = (
        WEIGHTS['connectivity'] * connectivity +
        WEIGHTS['chemistry'] * chemistry +
        WEIGHTS['balance'] * balance +
        WEIGHTS['progression'] * progression
    )
    
    components = {
        'connectivity': connectivity,
        'chemistry': chemistry,
        'balance': balance,
        'progression': progression
    }
    
    return total, components


def calculate_player_impact(G, pre_shot_passes, total_passes, player_name):
    """Calculate impact of removing a player."""
    # Baseline cohesion
    baseline, _ = calculate_cohesion(G, pre_shot_passes, total_passes)
    
    if player_name not in G.nodes():
        return 0
    
    # Remove player and recalculate
    G_modified = G.copy()
    
    # Count passes involving this player
    player_out_passes = sum([d['weight'] for _, _, d in G.out_edges(player_name, data=True)])
    player_in_passes = sum([d['weight'] for _, _, d in G.in_edges(player_name, data=True)])
    player_passes = player_out_passes + player_in_passes
    
    G_modified.remove_node(player_name)
    
    # Adjust pass counts (approximate)
    modified_total = max(total_passes - player_out_passes, 1)
    modified_pre_shot = int(pre_shot_passes * (modified_total / total_passes)) if total_passes > 0 else 0
    
    modified_cohesion, _ = calculate_cohesion(G_modified, modified_pre_shot, modified_total)
    
    # Impact = drop in cohesion
    impact = baseline - modified_cohesion
    return impact


def analyze_match(match_id):
    """Analyze a single match and return player impacts for both teams."""
    try:
        dataset = impect.load_open_data(match_id=match_id)
        events_df = dataset.to_df(engine="pandas")
    except Exception as e:
        return []
    
    # Build player ID to name mapping from metadata
    player_map = {}
    team_map = {}
    
    for team in dataset.metadata.teams:
        team_map[team.team_id] = team.name
        for player in team.players:
            player_map[player.player_id] = player.name
    
    results = []
    
    for team in dataset.metadata.teams:
        team_id = team.team_id
        team_name = team.name
        
        # Build network
        G, pre_shot, total_passes = build_pass_network(events_df, team_id, player_map)
        
        if G.number_of_nodes() < 5:
            continue
        
        # Baseline cohesion
        baseline_cohesion, _ = calculate_cohesion(G, pre_shot, total_passes)
        
        # Calculate impact for each player
        for player_name in G.nodes():
            impact = calculate_player_impact(G, pre_shot, total_passes, player_name)
            
            # Get player pass count
            passes_made = sum([d['weight'] for _, _, d in G.out_edges(player_name, data=True)])
            
            # Betweenness centrality
            try:
                betweenness = nx.betweenness_centrality(G, weight='weight')
                player_betweenness = betweenness.get(player_name, 0)
            except:
                player_betweenness = 0
            
            results.append({
                'match_id': match_id,
                'team_name': team_name,
                'player_name': player_name,
                'impact': impact,
                'impact_pct': impact * 100,
                'passes_made': passes_made,
                'betweenness': player_betweenness,
                'baseline_cohesion': baseline_cohesion
            })
    
    return results


def main():
    # Get all match IDs from our cohesion results
    cohesion_df = pd.read_csv(DATA_DIR / 'cohesion_results.csv')
    match_ids = cohesion_df['match_id'].unique()
    
    print(f"\nAnalyzing {len(match_ids)} matches...")
    print(f"This will take 10-15 minutes...\n")
    
    all_results = []
    
    for i, match_id in enumerate(match_ids):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(match_ids)} matches ({100*(i+1)/len(match_ids):.1f}%)")
        
        match_results = analyze_match(match_id)
        all_results.extend(match_results)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    print(f"\n✓ Analyzed {len(results_df)} player-match combinations")
    
    # Aggregate by player and team (season averages)
    player_summary = results_df.groupby(['team_name', 'player_name']).agg({
        'impact': 'mean',
        'impact_pct': 'mean',
        'passes_made': 'mean',
        'betweenness': 'mean',
        'match_id': 'count'
    }).reset_index()
    
    player_summary.columns = ['team_name', 'player_name', 'avg_impact', 'avg_impact_pct', 
                               'avg_passes', 'avg_betweenness', 'matches_played']
    
    # Filter to players with at least 5 matches
    player_summary = player_summary[player_summary['matches_played'] >= 5]
    
    # Sort by impact within each team
    player_summary = player_summary.sort_values(['team_name', 'avg_impact_pct'], ascending=[True, False])
    
    # Save to CSV
    player_summary.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Saved to {OUTPUT_FILE}")
    
    # Also save raw match-level data
    results_df.to_csv(DATA_DIR / 'player_impacts_raw.csv', index=False)
    print(f"✓ Saved raw data to {DATA_DIR / 'player_impacts_raw.csv'}")
    
    # Show top 3 per team
    print("\n" + "=" * 60)
    print("TOP 3 IMPACTFUL PLAYERS PER TEAM")
    print("=" * 60)
    
    for team in sorted(player_summary['team_name'].unique()):
        team_players = player_summary[player_summary['team_name'] == team].head(3)
        print(f"\n{team}:")
        for _, row in team_players.iterrows():
            print(f"  {row['player_name']}: {row['avg_impact_pct']:.2f}% impact ({int(row['matches_played'])} matches)")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    
    return player_summary


if __name__ == "__main__":
    results = main()