"""
Phase 2: Full Season Analysis
=============================

This script processes all 306 Bundesliga 2023/24 matches to:
1. Build weighted passing networks for each team-match
2. Compute cohesion scores
3. Correlate with match outcomes
4. Identify top/bottom teams by cohesion

Usage:
    python 02_full_analysis.py
    
Output:
    - data/cohesion_results.csv: Per-team-match cohesion scores
    - data/team_season_summary.csv: Season-level aggregates
    - figures/cohesion_vs_points.png: Validation plot
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
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import kloppy
from kloppy import impect

# Import our cohesion module
from cohesion_metric import (
    PassingNetworkBuilder, 
    CohesionCalculator, 
    analyze_match,
    compute_network_metrics
)

# Create output directories
DATA_DIR = Path(__file__).parent.parent / 'data'
FIG_DIR = Path(__file__).parent.parent / 'figures'
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("PHASE 2: FULL SEASON ANALYSIS")
print("=" * 60)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================
# STEP 1: Get all match IDs
# ============================================================
print("\n[STEP 1] Getting all match IDs...")

# IMPECT Open Data match IDs for Bundesliga 2023/24
# These are available from the open-data repository
# We'll discover them dynamically

def get_all_match_ids():
    """
    Get all available match IDs from IMPECT open data.
    
    The IMPECT open-data repo has match IDs in the format of 6-digit numbers.
    For Bundesliga 2023/24, these typically range from 122xxx to 124xxx.
    """
    # Known sample match IDs from the hackathon materials
    sample_ids = [122840, 122838, 122976]
    
    # We'll try to load matches systematically
    # Bundesliga 2023/24 has 306 matches (18 teams, 34 matchdays)
    
    # First, try the sample IDs to verify they work
    valid_ids = []
    
    print("  Testing sample match IDs...")
    for mid in sample_ids:
        try:
            events = impect.load_open_data(match_id=mid)
            valid_ids.append(mid)
            print(f"    ✓ {mid} works")
        except Exception as e:
            print(f"    ✗ {mid} failed: {type(e).__name__}")
    
    if not valid_ids:
        print("  No valid matches found. Check network connection.")
        return []
    
    # For a full analysis, we need to discover all match IDs
    # The IMPECT open data stores matches in a specific range
    # Let's scan a reasonable range
    
    print("\n  Scanning for all Bundesliga 2023/24 matches...")
    print("  (This may take a few minutes...)")
    
    # Typical range for Bundesliga 2023/24
    # Based on the samples, matches seem to be in 122000-125000 range
    start_id = 122000
    end_id = 125000
    step = 100  # Check every 100th to find the range first
    
    # Quick scan to find the range
    found_any = False
    first_valid = None
    last_valid = None
    
    for mid in tqdm(range(start_id, end_id, step), desc="  Quick scan"):
        try:
            events = impect.load_open_data(match_id=mid)
            if first_valid is None:
                first_valid = mid
            last_valid = mid
            found_any = True
        except:
            pass
    
    if not found_any:
        print("  Could not find match range. Using sample IDs only.")
        return valid_ids
    
    # Now scan the found range more thoroughly
    print(f"  Found matches in range {first_valid} to {last_valid}")
    print("  Loading all matches in this range...")
    
    all_ids = []
    for mid in tqdm(range(first_valid - 100, last_valid + 100), desc="  Full scan"):
        try:
            events = impect.load_open_data(match_id=mid)
            all_ids.append(mid)
        except:
            pass
    
    print(f"  ✓ Found {len(all_ids)} matches")
    return all_ids


# Alternative: If we can't scan, use a hardcoded list
# This list would need to be populated from IMPECT's open-data repo
KNOWN_MATCH_IDS = None  # Set this if you have the full list

if KNOWN_MATCH_IDS is not None:
    match_ids = KNOWN_MATCH_IDS
    print(f"  Using hardcoded list of {len(match_ids)} matches")
else:
    match_ids = get_all_match_ids()

if len(match_ids) == 0:
    print("\nERROR: No matches found. Please check your network connection.")
    print("Alternatively, provide a list of match IDs in KNOWN_MATCH_IDS variable.")
    sys.exit(1)

print(f"\n✓ Total matches to process: {len(match_ids)}")

# ============================================================
# STEP 2: Process all matches
# ============================================================
print("\n[STEP 2] Processing all matches...")

results = []
errors = []

calculator = CohesionCalculator()

for match_id in tqdm(match_ids, desc="  Processing matches"):
    try:
        # Load match
        events = impect.load_open_data(match_id=match_id)
        df = events.to_df(engine="pandas")
        
        # Get teams
        teams = df['team_name'].dropna().unique().tolist()
        if len(teams) != 2:
            errors.append({'match_id': match_id, 'error': f'Expected 2 teams, got {len(teams)}'})
            continue
        
        home_team = teams[0]  # First team is usually home
        away_team = teams[1]
        
        # Get match result (goals)
        home_goals = len(df[(df['team_name'] == home_team) & 
                           (df['event_type'] == 'SHOT') & 
                           (df['result'] == 'GOAL')])
        away_goals = len(df[(df['team_name'] == away_team) & 
                           (df['event_type'] == 'SHOT') & 
                           (df['result'] == 'GOAL')])
        
        # Process each team
        for team_name in teams:
            # Build network
            builder = PassingNetworkBuilder(df, team_name)
            G = builder.build_network()
            
            # Get positions
            positions = builder.get_player_positions()
            
            # Calculate cohesion
            cohesion = calculator.calculate(G, positions)
            
            # Get network metrics
            metrics = compute_network_metrics(G)
            
            # Determine result for this team
            if team_name == home_team:
                goals_for = home_goals
                goals_against = away_goals
                is_home = True
            else:
                goals_for = away_goals
                goals_against = home_goals
                is_home = False
            
            if goals_for > goals_against:
                result = 'win'
                points = 3
            elif goals_for < goals_against:
                result = 'loss'
                points = 0
            else:
                result = 'draw'
                points = 1
            
            # Store results
            results.append({
                'match_id': match_id,
                'team_name': team_name,
                'is_home': is_home,
                'opponent': away_team if team_name == home_team else home_team,
                'goals_for': goals_for,
                'goals_against': goals_against,
                'goal_diff': goals_for - goals_against,
                'result': result,
                'points': points,
                'cohesion_total': cohesion.total,
                'cohesion_connectivity': cohesion.connectivity,
                'cohesion_chemistry': cohesion.chemistry,
                'cohesion_balance': cohesion.balance,
                'cohesion_progression': cohesion.progression,
                'n_nodes': metrics.get('n_nodes', 0),
                'n_edges': metrics.get('n_edges', 0),
                'density': metrics.get('density', 0),
                'max_betweenness': metrics.get('max_betweenness', 0),
                'top_betweenness_player': metrics.get('top_betweenness_player', ''),
                'pre_shot_passes': len(builder.pre_shot_pass_ids),
                'total_passes': len(builder.team_events[builder.team_events['event_type'] == 'PASS']),
            })
            
    except Exception as e:
        errors.append({'match_id': match_id, 'error': str(e)})

print(f"\n✓ Processed {len(results)} team-matches")
print(f"✗ Errors: {len(errors)}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(DATA_DIR / 'cohesion_results.csv', index=False)
print(f"\n✓ Saved results to {DATA_DIR / 'cohesion_results.csv'}")

if errors:
    errors_df = pd.DataFrame(errors)
    errors_df.to_csv(DATA_DIR / 'processing_errors.csv', index=False)
    print(f"✓ Saved errors to {DATA_DIR / 'processing_errors.csv'}")

# ============================================================
# STEP 3: Aggregate to season level
# ============================================================
print("\n[STEP 3] Aggregating to season level...")

season_summary = results_df.groupby('team_name').agg({
    'points': 'sum',
    'goals_for': 'sum',
    'goals_against': 'sum',
    'cohesion_total': 'mean',
    'cohesion_connectivity': 'mean',
    'cohesion_chemistry': 'mean',
    'cohesion_balance': 'mean',
    'cohesion_progression': 'mean',
    'max_betweenness': 'mean',
    'pre_shot_passes': 'sum',
    'total_passes': 'sum',
    'match_id': 'count',
}).rename(columns={'match_id': 'matches_played'})

season_summary['goal_diff'] = season_summary['goals_for'] - season_summary['goals_against']
season_summary['pre_shot_ratio'] = season_summary['pre_shot_passes'] / season_summary['total_passes']
season_summary = season_summary.sort_values('points', ascending=False)

season_summary.to_csv(DATA_DIR / 'team_season_summary.csv')
print(f"✓ Saved season summary to {DATA_DIR / 'team_season_summary.csv'}")

print("\nSeason standings with cohesion:")
print(season_summary[['points', 'goal_diff', 'cohesion_total', 'cohesion_progression']].head(10))

# ============================================================
# STEP 4: Validation - Correlation with outcomes
# ============================================================
print("\n[STEP 4] Validation: Cohesion vs Points...")

from scipy import stats

# Season-level correlation
corr, pval = stats.pearsonr(season_summary['cohesion_total'], season_summary['points'])
print(f"\n  Season-level correlation (cohesion vs points):")
print(f"    Pearson r = {corr:.3f}, p-value = {pval:.4f}")

# Component correlations
print(f"\n  Component correlations with points:")
for component in ['connectivity', 'chemistry', 'balance', 'progression']:
    col = f'cohesion_{component}'
    r, p = stats.pearsonr(season_summary[col], season_summary['points'])
    print(f"    {component}: r = {r:.3f}, p = {p:.4f}")

# Match-level analysis
print(f"\n  Match-level analysis:")
win_cohesion = results_df[results_df['result'] == 'win']['cohesion_total'].mean()
draw_cohesion = results_df[results_df['result'] == 'draw']['cohesion_total'].mean()
loss_cohesion = results_df[results_df['result'] == 'loss']['cohesion_total'].mean()
print(f"    Win avg cohesion: {win_cohesion:.3f}")
print(f"    Draw avg cohesion: {draw_cohesion:.3f}")
print(f"    Loss avg cohesion: {loss_cohesion:.3f}")

# ANOVA test
from scipy.stats import f_oneway
f_stat, f_pval = f_oneway(
    results_df[results_df['result'] == 'win']['cohesion_total'],
    results_df[results_df['result'] == 'draw']['cohesion_total'],
    results_df[results_df['result'] == 'loss']['cohesion_total']
)
print(f"    One-way ANOVA: F = {f_stat:.3f}, p = {f_pval:.4f}")

# ============================================================
# STEP 5: Visualizations
# ============================================================
print("\n[STEP 5] Creating visualizations...")

# Plot 1: Cohesion vs Points (season level)
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(
    season_summary['cohesion_total'], 
    season_summary['points'],
    s=100, 
    alpha=0.7,
    c=season_summary['goal_diff'],
    cmap='RdYlGn'
)

# Add team labels
for idx, row in season_summary.iterrows():
    ax.annotate(
        idx,  # team name
        (row['cohesion_total'], row['points']),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=8
    )

# Add regression line
z = np.polyfit(season_summary['cohesion_total'], season_summary['points'], 1)
p = np.poly1d(z)
x_line = np.linspace(season_summary['cohesion_total'].min(), season_summary['cohesion_total'].max(), 100)
ax.plot(x_line, p(x_line), 'r--', alpha=0.5, label=f'r = {corr:.3f}')

ax.set_xlabel('Average Cohesion Score', fontsize=12)
ax.set_ylabel('Total Points', fontsize=12)
ax.set_title('Team Cohesion vs League Points\nBundesliga 2023/24', fontsize=14)
ax.legend()
plt.colorbar(scatter, label='Goal Difference')
plt.tight_layout()
plt.savefig(FIG_DIR / 'cohesion_vs_points.png', dpi=150)
plt.close()
print(f"✓ Saved figure: {FIG_DIR / 'cohesion_vs_points.png'}")

# Plot 2: Cohesion components comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
components = ['connectivity', 'chemistry', 'balance', 'progression']

for ax, component in zip(axes.flat, components):
    col = f'cohesion_{component}'
    r, p = stats.pearsonr(season_summary[col], season_summary['points'])
    
    ax.scatter(season_summary[col], season_summary['points'], s=80, alpha=0.7)
    
    # Regression line
    z = np.polyfit(season_summary[col], season_summary['points'], 1)
    poly = np.poly1d(z)
    x_line = np.linspace(season_summary[col].min(), season_summary[col].max(), 100)
    ax.plot(x_line, poly(x_line), 'r--', alpha=0.5)
    
    ax.set_xlabel(f'{component.capitalize()} Score', fontsize=11)
    ax.set_ylabel('Total Points', fontsize=11)
    ax.set_title(f'{component.capitalize()} vs Points (r = {r:.3f})', fontsize=12)

plt.suptitle('Cohesion Components vs League Points', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / 'cohesion_components.png', dpi=150)
plt.close()
print(f"✓ Saved figure: {FIG_DIR / 'cohesion_components.png'}")

# Plot 3: Match outcome boxplot
fig, ax = plt.subplots(figsize=(10, 6))
results_df.boxplot(column='cohesion_total', by='result', ax=ax, 
                   positions=[0, 1, 2], widths=0.6)
ax.set_xticklabels(['Draw', 'Loss', 'Win'])
ax.set_xlabel('Match Result', fontsize=12)
ax.set_ylabel('Cohesion Score', fontsize=12)
ax.set_title('Cohesion Score by Match Outcome', fontsize=14)
plt.suptitle('')  # Remove automatic title
plt.tight_layout()
plt.savefig(FIG_DIR / 'cohesion_by_result.png', dpi=150)
plt.close()
print(f"✓ Saved figure: {FIG_DIR / 'cohesion_by_result.png'}")

# ============================================================
# STEP 6: Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"""
Analysis complete!

Files created:
- {DATA_DIR / 'cohesion_results.csv'}: Per-match cohesion scores
- {DATA_DIR / 'team_season_summary.csv'}: Season-level summary
- {FIG_DIR / 'cohesion_vs_points.png'}: Main validation plot
- {FIG_DIR / 'cohesion_components.png'}: Component analysis
- {FIG_DIR / 'cohesion_by_result.png'}: Match outcome comparison

Key findings:
- Season-level correlation: r = {corr:.3f} (p = {pval:.4f})
- Win avg cohesion: {win_cohesion:.3f}
- Loss avg cohesion: {loss_cohesion:.3f}
- ANOVA F-statistic: {f_stat:.3f} (p = {f_pval:.4f})

Top 5 teams by cohesion:
{season_summary[['cohesion_total', 'points']].head()}

Next steps:
- Run 03_validation.py for detailed statistical tests
- Run 04_leverkusen_case.py for Leverkusen deep dive
""")

print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")