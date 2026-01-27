"""
PHASE 2: FULL ANALYSIS - Process All 306 Bundesliga Matches
============================================================

This script:
1. Discovers all available match IDs from IMPECT
2. Computes cohesion scores for each team-match
3. Correlates with match outcomes
4. Generates validation plots

Run from notebooks folder: python 03_full_analysis.py
"""

import sys
sys.path.insert(0, '../src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from kloppy import impect
from cohesion_metric import analyze_match_from_kloppy, CohesionScore

# Create output directories
DATA_DIR = Path('../data')
FIGURES_DIR = Path('../figures')
DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def discover_match_ids(start=122000, end=125000, sample_size=None):
    """
    Discover valid IMPECT match IDs by testing a range.
    
    Args:
        start: Starting match ID
        end: Ending match ID  
        sample_size: If set, only return this many IDs (for testing)
    
    Returns:
        List of valid match IDs
    """
    print(f"Discovering match IDs in range {start}-{end}...")
    valid_ids = []
    
    for mid in tqdm(range(start, end), desc="Scanning"):
        try:
            # Try to load just metadata (faster)
            events = impect.load_open_data(match_id=mid)
            valid_ids.append(mid)
        except:
            continue
    
    print(f"Found {len(valid_ids)} valid matches")
    
    if sample_size:
        return valid_ids[:sample_size]
    return valid_ids


def get_match_result(team_goals: int, opponent_goals: int) -> str:
    """Determine match result from goals."""
    if team_goals > opponent_goals:
        return 'win'
    elif team_goals < opponent_goals:
        return 'loss'
    else:
        return 'draw'


def get_points(result: str) -> int:
    """Convert result to points."""
    if result == 'win':
        return 3
    elif result == 'draw':
        return 1
    else:
        return 0


def process_all_matches(match_ids: list) -> pd.DataFrame:
    """
    Process all matches and compute cohesion scores.
    
    Returns:
        DataFrame with one row per team-match
    """
    all_results = []
    errors = []
    
    for match_id in tqdm(match_ids, desc="Processing matches"):
        try:
            # Load match
            events = impect.load_open_data(match_id=match_id)
            
            # Analyze both teams
            results = analyze_match_from_kloppy(events)
            
            # Get team names and goals
            teams = list(results.keys())
            if len(teams) != 2:
                continue
                
            team1, team2 = teams
            goals1 = results[team1]['goals']
            goals2 = results[team2]['goals']
            
            # Store results for each team
            for team_name in teams:
                data = results[team_name]
                opponent = team2 if team_name == team1 else team1
                opponent_goals = goals2 if team_name == team1 else goals1
                
                cohesion = data['cohesion']
                metrics = data['metrics']
                
                result = get_match_result(data['goals'], opponent_goals)
                points = get_points(result)
                
                row = {
                    'match_id': match_id,
                    'team_name': team_name,
                    'opponent': opponent,
                    'goals_for': data['goals'],
                    'goals_against': opponent_goals,
                    'result': result,
                    'points': points,
                    'cohesion_total': cohesion.total,
                    'cohesion_connectivity': cohesion.connectivity,
                    'cohesion_chemistry': cohesion.chemistry,
                    'cohesion_balance': cohesion.balance,
                    'cohesion_progression': cohesion.progression,
                    'n_players': metrics.get('n_nodes', 0),
                    'n_connections': metrics.get('n_edges', 0),
                    'density': metrics.get('density', 0),
                    'max_betweenness': metrics.get('max_betweenness', 0),
                    'top_betweenness_player': metrics.get('top_betweenness_player', ''),
                    'pre_shot_passes': data['pre_shot_passes'],
                    'total_passes': data['total_passes'],
                }
                all_results.append(row)
                
        except Exception as e:
            errors.append({'match_id': match_id, 'error': str(e)})
            continue
    
    # Save errors log
    if errors:
        pd.DataFrame(errors).to_csv(DATA_DIR / 'processing_errors.csv', index=False)
        print(f"Logged {len(errors)} errors to data/processing_errors.csv")
    
    return pd.DataFrame(all_results)


def aggregate_season_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to season-level statistics per team."""
    
    season_stats = df.groupby('team_name').agg({
        'points': 'sum',
        'goals_for': 'sum',
        'goals_against': 'sum',
        'cohesion_total': 'mean',
        'cohesion_connectivity': 'mean',
        'cohesion_chemistry': 'mean',
        'cohesion_balance': 'mean',
        'cohesion_progression': 'mean',
        'density': 'mean',
        'max_betweenness': 'mean',
        'match_id': 'count'  # Number of matches
    }).rename(columns={'match_id': 'matches_played'})
    
    season_stats['goal_diff'] = season_stats['goals_for'] - season_stats['goals_against']
    
    return season_stats.sort_values('points', ascending=False)


def validate_season_level(season_df: pd.DataFrame):
    """Validate: Does average cohesion correlate with season points?"""
    
    from scipy import stats
    
    print("\n" + "=" * 60)
    print("VALIDATION: Season-Level Correlation")
    print("=" * 60)
    
    # Pearson correlation: cohesion vs points
    r, p = stats.pearsonr(season_df['cohesion_total'], season_df['points'])
    print(f"\nCohesion vs Points:")
    print(f"  Pearson r = {r:.3f}, p = {p:.4f}")
    
    # Component correlations
    print("\nComponent correlations with points:")
    for component in ['connectivity', 'chemistry', 'balance', 'progression']:
        col = f'cohesion_{component}'
        r, p = stats.pearsonr(season_df[col], season_df['points'])
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        print(f"  {component.capitalize():12s}: r = {r:+.3f}, p = {p:.4f} {sig}")
    
    return r, p


def validate_match_level(df: pd.DataFrame):
    """Validate: Does pre-match cohesion predict outcome?"""
    
    from scipy import stats
    
    print("\n" + "=" * 60)
    print("VALIDATION: Match-Level Analysis")
    print("=" * 60)
    
    # Average cohesion by result
    result_stats = df.groupby('result')['cohesion_total'].agg(['mean', 'std', 'count'])
    print("\nCohesion by match result:")
    print(result_stats.round(3))
    
    # ANOVA: Is there a significant difference?
    wins = df[df['result'] == 'win']['cohesion_total']
    draws = df[df['result'] == 'draw']['cohesion_total']
    losses = df[df['result'] == 'loss']['cohesion_total']
    
    f_stat, p_val = stats.f_oneway(wins, draws, losses)
    print(f"\nOne-way ANOVA: F = {f_stat:.2f}, p = {p_val:.4f}")
    
    if p_val < 0.05:
        print("  → Significant difference in cohesion between win/draw/loss")
    else:
        print("  → No significant difference detected")
    
    return result_stats


def plot_cohesion_vs_points(season_df: pd.DataFrame):
    """Create scatter plot of cohesion vs season points."""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter with color by goal difference
    scatter = ax.scatter(
        season_df['cohesion_total'],
        season_df['points'],
        c=season_df['goal_diff'],
        cmap='RdYlGn',
        s=150,
        alpha=0.8,
        edgecolors='black'
    )
    
    # Add team labels
    for idx, row in season_df.iterrows():
        ax.annotate(
            idx,  # team name is index
            (row['cohesion_total'], row['points']),
            fontsize=8,
            ha='center',
            va='bottom',
            xytext=(0, 5),
            textcoords='offset points'
        )
    
    # Regression line
    from scipy import stats
    slope, intercept, r, p, se = stats.linregress(
        season_df['cohesion_total'], 
        season_df['points']
    )
    x_line = np.linspace(season_df['cohesion_total'].min(), 
                         season_df['cohesion_total'].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r--', alpha=0.7,
            label=f'r = {r:.3f}, p = {p:.3f}')
    
    ax.set_xlabel('Average Cohesion Score', fontsize=12)
    ax.set_ylabel('Season Points', fontsize=12)
    ax.set_title('Team Cohesion vs Season Performance\nBundesliga 2023/24', fontsize=14)
    ax.legend()
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Goal Difference')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cohesion_vs_points.png', dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'cohesion_vs_points.png'}")


def plot_cohesion_components(season_df: pd.DataFrame):
    """Create 2x2 grid of component correlations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    components = ['connectivity', 'chemistry', 'balance', 'progression']
    
    from scipy import stats
    
    for ax, comp in zip(axes.flat, components):
        col = f'cohesion_{comp}'
        
        ax.scatter(season_df[col], season_df['points'], 
                  s=80, alpha=0.7, edgecolors='black')
        
        # Regression
        slope, intercept, r, p, se = stats.linregress(season_df[col], season_df['points'])
        x_line = np.linspace(season_df[col].min(), season_df[col].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, 'r--', alpha=0.7)
        
        ax.set_xlabel(f'{comp.capitalize()} Score', fontsize=10)
        ax.set_ylabel('Season Points', fontsize=10)
        ax.set_title(f'{comp.capitalize()}\nr = {r:.3f}, p = {p:.3f}', fontsize=11)
    
    plt.suptitle('Cohesion Components vs Season Points', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cohesion_components.png', dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'cohesion_components.png'}")


def plot_cohesion_by_result(df: pd.DataFrame):
    """Box plot of cohesion by match result."""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    order = ['loss', 'draw', 'win']
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    
    sns.boxplot(data=df, x='result', y='cohesion_total', 
                order=order, palette=colors, ax=ax)
    
    ax.set_xlabel('Match Result', fontsize=12)
    ax.set_ylabel('Cohesion Score', fontsize=12)
    ax.set_title('Cohesion Score Distribution by Match Result', fontsize=14)
    
    # Add mean markers
    means = df.groupby('result')['cohesion_total'].mean()
    for i, result in enumerate(order):
        ax.scatter(i, means[result], color='black', s=100, zorder=5, marker='D')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cohesion_by_result.png', dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'cohesion_by_result.png'}")


def main():
    print("=" * 60)
    print("PHASE 2: FULL BUNDESLIGA ANALYSIS")
    print("=" * 60)
    
    # Step 1: Discover match IDs
    # For testing, you can set sample_size=10
    # For full run, remove sample_size parameter
    match_ids = discover_match_ids(start=122000, end=125000)
    
    if not match_ids:
        print("ERROR: No matches found. Check network connection.")
        return
    
    print(f"\nProcessing {len(match_ids)} matches...")
    
    # Step 2: Process all matches
    df = process_all_matches(match_ids)
    
    if df.empty:
        print("ERROR: No data processed.")
        return
    
    # Save raw results
    df.to_csv(DATA_DIR / 'cohesion_results.csv', index=False)
    print(f"\nSaved: {DATA_DIR / 'cohesion_results.csv'}")
    print(f"Total team-match records: {len(df)}")
    
    # Step 3: Aggregate to season level
    season_df = aggregate_season_stats(df)
    season_df.to_csv(DATA_DIR / 'team_season_summary.csv')
    print(f"Saved: {DATA_DIR / 'team_season_summary.csv'}")
    
    print("\n" + "=" * 60)
    print("SEASON STANDINGS (by points)")
    print("=" * 60)
    print(season_df[['points', 'goal_diff', 'cohesion_total']].head(10))
    
    # Step 4: Validation
    validate_season_level(season_df)
    validate_match_level(df)
    
    # Step 5: Visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    plot_cohesion_vs_points(season_df)
    plot_cohesion_components(season_df)
    plot_cohesion_by_result(df)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nFiles generated:")
    print(f"  - data/cohesion_results.csv")
    print(f"  - data/team_season_summary.csv")
    print(f"  - figures/cohesion_vs_points.png")
    print(f"  - figures/cohesion_components.png")
    print(f"  - figures/cohesion_by_result.png")


if __name__ == "__main__":
    main()
