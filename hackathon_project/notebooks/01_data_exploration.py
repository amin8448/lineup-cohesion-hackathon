"""
Phase 1: Data Exploration and Schema Discovery
==============================================

This script loads IMPECT data and explores the schema to understand:
1. What columns are available for passes and shots
2. How to identify pass receivers
3. How to link passes to subsequent shots
4. Basic passing network construction

Run this first to understand the data before building the full analysis.

Usage:
    python 01_data_exploration.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
from tqdm import tqdm

# Import kloppy for loading IMPECT data
from kloppy import impect
from kloppy.domain import EventType

print("=" * 60)
print("PHASE 1: DATA EXPLORATION")
print("=" * 60)

# ============================================================
# STEP 1: Load a sample match
# ============================================================
print("\n[STEP 1] Loading sample match (match_id=122840)...")

try:
    events = impect.load_open_data(match_id=122840)
    print(f"✓ Loaded {len(events)} events")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    print("\nMake sure you have internet access and kloppy>=3.18.0 installed")
    sys.exit(1)

# ============================================================
# STEP 2: Convert to DataFrame and explore schema
# ============================================================
print("\n[STEP 2] Converting to DataFrame...")

df = events.to_df(engine="pandas")
print(f"✓ DataFrame shape: {df.shape}")

print("\n[STEP 2a] All columns:")
for i, col in enumerate(df.columns):
    print(f"  {i+1:2d}. {col}")

print("\n[STEP 2b] Event types present:")
if 'event_type' in df.columns:
    event_counts = df['event_type'].value_counts()
    for event_type, count in event_counts.items():
        print(f"  - {event_type}: {count}")

# ============================================================
# STEP 3: Explore PASS events specifically
# ============================================================
print("\n[STEP 3] Exploring PASS events...")

passes = df[df['event_type'] == 'PASS'].copy()
print(f"✓ Found {len(passes)} passes")

print("\n[STEP 3a] Pass-specific columns (non-null):")
for col in passes.columns:
    non_null = passes[col].notna().sum()
    if non_null > 0:
        sample = passes[col].dropna().iloc[0] if passes[col].notna().any() else None
        print(f"  - {col}: {non_null} non-null, sample: {sample}")

# Check for receiver column
print("\n[STEP 3b] Looking for receiver column...")
receiver_candidates = ['receiver_player_name', 'pass_receiver_name', 'receiver_name', 
                       'receiver_player', 'end_player', 'to_player']
receiver_col = None
for col in receiver_candidates:
    if col in passes.columns:
        non_null = passes[col].notna().sum()
        print(f"  ✓ Found '{col}' with {non_null} non-null values")
        receiver_col = col
        break

if receiver_col is None:
    print("  ✗ No receiver column found. Checking all columns for player-like data...")
    for col in passes.columns:
        if 'player' in col.lower() or 'receiver' in col.lower():
            print(f"    Candidate: {col}")

# ============================================================
# STEP 4: Explore metadata (teams, players, positions)
# ============================================================
print("\n[STEP 4] Exploring metadata...")

# Teams
if 'team_name' in df.columns:
    teams = df['team_name'].unique()
    print(f"✓ Teams: {list(teams)}")

# Get team metadata from kloppy
print("\n[STEP 4a] Team metadata from kloppy:")
for team in events.metadata.teams:
    print(f"\n  Team: {team.name}")
    print(f"    Ground: {team.ground}")
    print(f"    Players ({len(team.players)}):")
    for player in team.players[:5]:  # First 5 players
        pos = player.starting_position if hasattr(player, 'starting_position') else 'N/A'
        print(f"      - {player.name} (#{player.jersey_no}) - Position: {pos}")
    if len(team.players) > 5:
        print(f"      ... and {len(team.players) - 5} more")

# ============================================================
# STEP 5: Explore SHOT events
# ============================================================
print("\n[STEP 5] Exploring SHOT events...")

shots = df[df['event_type'] == 'SHOT'].copy()
print(f"✓ Found {len(shots)} shots")

if len(shots) > 0:
    print("\n[STEP 5a] Shot results:")
    if 'result' in shots.columns:
        print(shots['result'].value_counts())
    
    print("\n[STEP 5b] Sample shot event:")
    sample_shot = shots.iloc[0]
    for col in shots.columns:
        if pd.notna(sample_shot[col]):
            print(f"  - {col}: {sample_shot[col]}")

# ============================================================
# STEP 6: Identify pre-shot passes
# ============================================================
print("\n[STEP 6] Identifying pre-shot passes...")

def identify_pre_shot_passes(df, team_name, n_before=2):
    """
    Find passes that occurred in the buildup to shots.
    
    Returns set of indices that are pre-shot passes.
    """
    # Sort by time
    df_sorted = df.sort_values(['period_id', 'timestamp']).reset_index(drop=True)
    
    # Find shots for this team
    shot_mask = (df_sorted['event_type'] == 'SHOT') & (df_sorted['team_name'] == team_name)
    shot_indices = df_sorted[shot_mask].index.tolist()
    
    pre_shot_indices = set()
    
    for shot_idx in shot_indices:
        passes_found = 0
        current_idx = shot_idx - 1
        
        while current_idx >= 0 and passes_found < n_before:
            event = df_sorted.loc[current_idx]
            
            # Stop if possession changes
            if event['team_name'] != team_name:
                break
            
            # Mark passes
            if event['event_type'] == 'PASS':
                pre_shot_indices.add(current_idx)
                passes_found += 1
            
            current_idx -= 1
    
    return pre_shot_indices, df_sorted

# Test on one team
if 'team_name' in df.columns:
    team = df['team_name'].dropna().iloc[0]
    pre_shot_idx, df_sorted = identify_pre_shot_passes(df, team, n_before=2)
    
    team_passes = df_sorted[(df_sorted['team_name'] == team) & 
                            (df_sorted['event_type'] == 'PASS')]
    
    print(f"✓ Team: {team}")
    print(f"✓ Total passes: {len(team_passes)}")
    print(f"✓ Pre-shot passes (n_before=2): {len(pre_shot_idx)}")
    print(f"✓ Pre-shot ratio: {len(pre_shot_idx)/len(team_passes)*100:.1f}%")

# ============================================================
# STEP 7: Build a basic passing network
# ============================================================
print("\n[STEP 7] Building basic passing network...")

def build_passing_network(df, team_name, receiver_col=None):
    """Build a directed passing network for a team."""
    
    # Get passes for this team
    team_passes = df[(df['team_name'] == team_name) & (df['event_type'] == 'PASS')].copy()
    
    # Try to find receiver column if not specified
    if receiver_col is None:
        for col in ['receiver_player_name', 'pass_receiver_name', 'receiver_name']:
            if col in team_passes.columns and team_passes[col].notna().any():
                receiver_col = col
                break
    
    if receiver_col is None:
        print(f"  ✗ Cannot find receiver column for {team_name}")
        return None
    
    # Build network
    G = nx.DiGraph()
    edge_counts = defaultdict(int)
    
    for _, row in team_passes.iterrows():
        passer = row.get('player_name')
        receiver = row.get(receiver_col)
        
        if pd.notna(passer) and pd.notna(receiver):
            edge_counts[(passer, receiver)] += 1
    
    # Add edges
    for (passer, receiver), count in edge_counts.items():
        G.add_edge(passer, receiver, weight=count)
    
    return G

# Build network for first team
if 'team_name' in df.columns:
    team = df['team_name'].dropna().iloc[0]
    G = build_passing_network(df, team)
    
    if G is not None:
        print(f"✓ Network for {team}:")
        print(f"  - Nodes (players): {G.number_of_nodes()}")
        print(f"  - Edges (pass combinations): {G.number_of_edges()}")
        print(f"  - Density: {nx.density(G):.3f}")
        
        # Top passers (by out-degree)
        out_deg = sorted(G.out_degree(weight='weight'), key=lambda x: x[1], reverse=True)
        print(f"\n  Top 5 passers:")
        for player, weight in out_deg[:5]:
            print(f"    - {player}: {weight:.0f} passes")
        
        # Betweenness centrality
        betweenness = nx.betweenness_centrality(G, weight='weight')
        top_between = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  Top 5 by betweenness centrality:")
        for player, score in top_between[:5]:
            print(f"    - {player}: {score:.3f}")

# ============================================================
# STEP 8: Summary and next steps
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print("""
Key findings from data exploration:
1. Event types available: PASS, SHOT, and others
2. Receiver column identified (check output above)
3. Pre-shot passes can be identified by looking backwards from shots
4. Basic passing network construction works

Next steps (Phase 2):
1. Load all 306 Bundesliga matches
2. Compute weighted networks with pre-shot pass weighting
3. Calculate cohesion scores for each team-match
4. Correlate with match outcomes
5. Deep dive on Leverkusen's undefeated season

Files to create:
- src/cohesion_metric.py: Core metric calculation
- notebooks/02_full_analysis.py: Process all matches
- notebooks/03_validation.py: Statistical validation
- notebooks/04_leverkusen_case_study.py: Deep dive
""")

print("\n[DONE] Phase 1 complete. Check the output above for schema details.")
