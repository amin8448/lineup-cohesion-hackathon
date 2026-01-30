"""
PHASE 3: RE-VALIDATE WITH OPTIMIZED WEIGHTS
============================================

This script re-computes cohesion scores using the empirically optimized weights
and shows the improved correlation with season performance.

Key insight: Balance should be INVERTED (hub-dependent teams perform better)

Run from notebooks folder: python 05_revalidate_optimized.py
"""

import sys
sys.path.insert(0, '../src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

DATA_DIR = Path('../data')
FIGURES_DIR = Path('../figures')


def recompute_cohesion_optimized(df):
    """
    Recompute total cohesion score with optimized weights.
    
    Original (equal weights):
        total = 0.25*connectivity + 0.25*chemistry + 0.25*balance + 0.25*progression
    
    Optimized (based on correlation analysis):
        total = 0.50*connectivity + 0.25*chemistry + 0.15*(1-balance) + 0.10*progression
        
    Note: balance is INVERTED because low balance (high Gini) = hub-dependent = better
    """
    
    df = df.copy()
    
    # Invert balance: original balance = 1 - Gini, so we want Gini = 1 - balance
    df['hub_dependence'] = 1 - df['cohesion_balance']
    
    # New optimized weights
    w_connectivity = 0.50
    w_chemistry = 0.25
    w_hub = 0.15
    w_progression = 0.10
    
    df['cohesion_optimized'] = (
        w_connectivity * df['cohesion_connectivity'] +
        w_chemistry * df['cohesion_chemistry'] +
        w_hub * df['hub_dependence'] +
        w_progression * df['cohesion_progression']
    )
    
    return df


def main():
    print("=" * 60)
    print("RE-VALIDATION WITH OPTIMIZED WEIGHTS")
    print("=" * 60)
    
    # Load original results
    df = pd.read_csv(DATA_DIR / 'cohesion_results.csv')
    print(f"Loaded {len(df)} team-match records")
    
    # Recompute with optimized weights
    df = recompute_cohesion_optimized(df)
    
    # Aggregate to season level
    season_df = df.groupby('team_name').agg({
        'points': 'sum',
        'goals_for': 'sum',
        'goals_against': 'sum',
        'cohesion_total': 'mean',          # Original (equal weights)
        'cohesion_optimized': 'mean',       # New (optimized weights)
        'cohesion_connectivity': 'mean',
        'cohesion_chemistry': 'mean',
        'cohesion_balance': 'mean',
        'hub_dependence': 'mean',
        'cohesion_progression': 'mean',
    })
    season_df['goal_diff'] = season_df['goals_for'] - season_df['goals_against']
    season_df = season_df.sort_values('points', ascending=False)
    
    # Compare correlations
    print("\n" + "=" * 60)
    print("CORRELATION COMPARISON")
    print("=" * 60)
    
    r_original, p_original = stats.pearsonr(season_df['cohesion_total'], season_df['points'])
    r_optimized, p_optimized = stats.pearsonr(season_df['cohesion_optimized'], season_df['points'])
    
    print(f"\nOriginal (equal weights):")
    print(f"  r = {r_original:.3f}, p = {p_original:.4f}")
    
    print(f"\nOptimized weights:")
    print(f"  r = {r_optimized:.3f}, p = {p_optimized:.4f}")
    
    improvement = ((r_optimized - r_original) / abs(r_original)) * 100
    print(f"\nImprovement: {improvement:+.1f}%")
    
    # Component analysis with corrected hub_dependence
    print("\n" + "=" * 60)
    print("COMPONENT CORRELATIONS (with hub_dependence)")
    print("=" * 60)
    
    components = [
        ('Connectivity', 'cohesion_connectivity'),
        ('Chemistry', 'cohesion_chemistry'),
        ('Hub Dependence', 'hub_dependence'),  # Inverted balance
        ('Progression', 'cohesion_progression'),
    ]
    
    for name, col in components:
        r, p = stats.pearsonr(season_df[col], season_df['points'])
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        print(f"  {name:15s}: r = {r:+.3f}, p = {p:.4f} {sig}")
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original
    ax = axes[0]
    ax.scatter(season_df['cohesion_total'], season_df['points'], 
               s=100, alpha=0.7, edgecolors='black')
    slope, intercept, r, p, se = stats.linregress(season_df['cohesion_total'], season_df['points'])
    x_line = np.linspace(season_df['cohesion_total'].min(), season_df['cohesion_total'].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r--', alpha=0.7)
    ax.set_xlabel('Cohesion (Equal Weights)', fontsize=12)
    ax.set_ylabel('Season Points', fontsize=12)
    ax.set_title(f'Original Metric\nr = {r_original:.3f}, p = {p_original:.3f}', fontsize=12)
    
    # Optimized
    ax = axes[1]
    ax.scatter(season_df['cohesion_optimized'], season_df['points'], 
               s=100, alpha=0.7, edgecolors='black', c='green')
    slope, intercept, r, p, se = stats.linregress(season_df['cohesion_optimized'], season_df['points'])
    x_line = np.linspace(season_df['cohesion_optimized'].min(), season_df['cohesion_optimized'].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r--', alpha=0.7)
    ax.set_xlabel('Cohesion (Optimized Weights)', fontsize=12)
    ax.set_ylabel('Season Points', fontsize=12)
    ax.set_title(f'Optimized Metric\nr = {r_optimized:.3f}, p = {p_optimized:.3f}', fontsize=12)
    
    # Add team labels to optimized plot
    for idx, row in season_df.iterrows():
        ax.annotate(idx, (row['cohesion_optimized'], row['points']),
                   fontsize=7, ha='center', va='bottom', xytext=(0, 3),
                   textcoords='offset points')
    
    plt.suptitle('Cohesion Metric: Original vs Optimized Weights', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cohesion_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {FIGURES_DIR / 'cohesion_comparison.png'")
    
    # Match-level validation with optimized metric
    print("\n" + "=" * 60)
    print("MATCH-LEVEL VALIDATION (optimized)")
    print("=" * 60)
    
    result_stats = df.groupby('result')['cohesion_optimized'].agg(['mean', 'std', 'count'])
    print("\nCohesion by match result:")
    print(result_stats.round(3))
    
    wins = df[df['result'] == 'win']['cohesion_optimized']
    draws = df[df['result'] == 'draw']['cohesion_optimized']
    losses = df[df['result'] == 'loss']['cohesion_optimized']
    
    f_stat, p_val = stats.f_oneway(wins, draws, losses)
    print(f"\nOne-way ANOVA: F = {f_stat:.2f}, p = {p_val:.6f}")
    
    # Show top teams with optimized cohesion
    print("\n" + "=" * 60)
    print("TOP 10 TEAMS BY OPTIMIZED COHESION")
    print("=" * 60)
    print(season_df[['points', 'goal_diff', 'cohesion_total', 'cohesion_optimized']].head(10).round(3))
    
    # Save updated results
    df.to_csv(DATA_DIR / 'cohesion_results_optimized.csv', index=False)
    season_df.to_csv(DATA_DIR / 'team_season_summary_optimized.csv')
    print(f"\nSaved updated results with optimized cohesion scores")


if __name__ == "__main__":
    main()
