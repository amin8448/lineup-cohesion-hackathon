"""
Quick test of the cohesion metric module.
Run from the project root: python notebooks/02_test_cohesion.py
"""

import sys
sys.path.insert(0, '../src')

from kloppy import impect
from cohesion_metric import analyze_match_from_kloppy

print("=" * 60)
print("TESTING COHESION METRIC MODULE")
print("=" * 60)

print("\nLoading sample match (122840)...")
events = impect.load_open_data(match_id=122840)

print("Analyzing match...\n")
results = analyze_match_from_kloppy(events)

for team_name, data in results.items():
    cohesion = data['cohesion']
    metrics = data['metrics']
    
    print(f"{'=' * 50}")
    print(f"TEAM: {team_name}")
    print(f"{'=' * 50}")
    print(f"Goals scored: {data['goals']}")
    print(f"\nNetwork stats:")
    print(f"  Players: {metrics['n_nodes']}")
    print(f"  Pass connections: {metrics['n_edges']}")
    print(f"  Density: {metrics['density']:.3f}")
    print(f"  Top betweenness: {metrics['top_betweenness_player']} ({metrics['max_betweenness']:.3f})")
    print(f"  Top degree: {metrics['top_degree_player']}")
    print(f"\nCohesion score breakdown:")
    print(f"  Connectivity: {cohesion.connectivity:.3f}")
    print(f"  Chemistry:    {cohesion.chemistry:.3f}")
    print(f"  Balance:      {cohesion.balance:.3f}")
    print(f"  Progression:  {cohesion.progression:.3f}")
    print(f"  TOTAL:        {cohesion.total:.3f}")
    print(f"\nPre-shot passes: {data['pre_shot_passes']} / {data['total_passes']} total")
    print()

print("✓ Test complete!")
