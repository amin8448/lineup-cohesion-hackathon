"""
HACKATHON SLIDE DECK GENERATOR
==============================

Generates an 8-slide PDF presentation for the Soccer Data Analytics Hackathon.

Prompt A: Starting Eleven Lineup Construction
Team: [Your Name]

Run: python 06_generate_slides.py
Output: ../figures/slides/SoccerHackathon_Slides.pdf
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

# Paths
FIGURES_DIR = Path('../figures')
DATA_DIR = Path('../data')
SLIDES_DIR = FIGURES_DIR / 'slides'
SLIDES_DIR.mkdir(exist_ok=True)

# Color scheme
COLORS = {
    'primary': '#1a5276',      # Dark blue
    'secondary': '#e74c3c',    # Red accent
    'success': '#27ae60',      # Green
    'warning': '#f39c12',      # Orange
    'light': '#ecf0f1',        # Light gray
    'dark': '#2c3e50',         # Dark gray
    'leverkusen': '#e32221',   # Leverkusen red
}


def create_title_slide(fig, ax):
    """Slide 1: Title"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Background
    ax.add_patch(Rectangle((0, 0), 10, 10, facecolor=COLORS['primary'], zorder=0))
    
    # Title
    ax.text(5, 7, 'Lineup Cohesion Score', fontsize=32, fontweight='bold',
            color='white', ha='center', va='center')
    ax.text(5, 6, 'A Network-Based Metric for Optimal Starting Eleven Selection',
            fontsize=16, color='white', ha='center', va='center', style='italic')
    
    # Divider line
    ax.plot([2, 8], [5.2, 5.2], color=COLORS['secondary'], linewidth=3)
    
    # Author info
    ax.text(5, 4.2, 'NEU Sports Analytics Hackathon', fontsize=14, 
            color='white', ha='center', va='center')
    ax.text(5, 3.5, 'Prompt A: Starting Eleven Lineup Construction', fontsize=12,
            color=COLORS['light'], ha='center', va='center')
    ax.text(5, 2.5, 'February 2026', fontsize=12,
            color=COLORS['light'], ha='center', va='center')
    
    # Data source
    ax.text(5, 1.2, 'Data: IMPECT Open Data • Bundesliga 2023/24 • 306 Matches',
            fontsize=10, color=COLORS['light'], ha='center', va='center', alpha=0.8)


def create_problem_slide(fig, ax):
    """Slide 2: Problem & Motivation"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Header
    ax.add_patch(Rectangle((0, 9), 10, 1, facecolor=COLORS['primary'], zorder=0))
    ax.text(0.3, 9.5, 'The Challenge: Optimal Lineup Selection', fontsize=20, 
            fontweight='bold', color='white', va='center')
    
    # Problem statement
    ax.text(0.5, 8.2, 'How do coaches select a starting eleven that maximizes team performance?',
            fontsize=14, color=COLORS['dark'], va='center', style='italic')
    
    # Key questions box
    ax.add_patch(FancyBboxPatch((0.3, 5.5), 4.2, 2.3, boxstyle="round,pad=0.1",
                                facecolor=COLORS['light'], edgecolor=COLORS['primary'], linewidth=2))
    ax.text(2.4, 7.5, 'Key Questions', fontsize=12, fontweight='bold', 
            color=COLORS['primary'], ha='center')
    questions = [
        '• Which players connect well together?',
        '• Who are the critical hub players?',
        '• What passing patterns lead to goals?'
    ]
    for i, q in enumerate(questions):
        ax.text(0.5, 7.0 - i*0.5, q, fontsize=11, color=COLORS['dark'])
    
    # Our approach box
    ax.add_patch(FancyBboxPatch((5.3, 5.5), 4.4, 2.3, boxstyle="round,pad=0.1",
                                facecolor='#e8f6f3', edgecolor=COLORS['success'], linewidth=2))
    ax.text(7.5, 7.5, 'Our Approach', fontsize=12, fontweight='bold',
            color=COLORS['success'], ha='center')
    approach = [
        '• Build player-to-player pass networks',
        '• Weight edges by shot creation',
        '• Quantify lineup "cohesion"'
    ]
    for i, a in enumerate(approach):
        ax.text(5.5, 7.0 - i*0.5, a, fontsize=11, color=COLORS['dark'])
    
    # Bottom insight
    ax.add_patch(FancyBboxPatch((0.3, 1.0), 9.4, 2.0, boxstyle="round,pad=0.1",
                                facecolor='#fef9e7', edgecolor=COLORS['warning'], linewidth=2))
    ax.text(5, 2.6, '💡 Key Insight', fontsize=12, fontweight='bold',
            color=COLORS['warning'], ha='center')
    ax.text(5, 1.9, 'Elite teams are HUB-DEPENDENT: they funnel play through star players',
            fontsize=12, color=COLORS['dark'], ha='center')
    ax.text(5, 1.3, '(Counter to the intuition that "balanced" teams perform better)',
            fontsize=10, color='gray', ha='center', style='italic')


def create_methodology_slide(fig, ax):
    """Slide 3: Methodology - Cohesion Metric"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Header
    ax.add_patch(Rectangle((0, 9), 10, 1, facecolor=COLORS['primary'], zorder=0))
    ax.text(0.3, 9.5, 'Methodology: Lineup Cohesion Score', fontsize=20,
            fontweight='bold', color='white', va='center')
    
    # Formula
    ax.text(5, 8.2, 'Cohesion = 0.50·Connectivity + 0.25·Chemistry + 0.15·HubDep + 0.10·Progression',
            fontsize=11, color=COLORS['dark'], ha='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor=COLORS['light'], edgecolor=COLORS['primary']))
    
    # Component boxes
    components = [
        ('Connectivity (50%)', 'Network density +\navg clustering coefficient', 
         'How interconnected\nare the players?', COLORS['primary']),
        ('Chemistry (25%)', 'Critical position pair\npass frequency', 
         'Midfield→Wing,\nMidfield→Striker links', '#8e44ad'),
        ('Hub Dependence (15%)', 'Gini coefficient of\ndegree distribution', 
         'Star player\nreliance (GOOD)', COLORS['success']),
        ('Progression (10%)', 'Pre-shot pass ratio\n(passes → shots)', 
         'Attacking\neffectiveness', COLORS['secondary']),
    ]
    
    x_positions = [1.25, 3.75, 6.25, 8.75]
    for i, (name, desc, note, color) in enumerate(components):
        x = x_positions[i]
        # Box
        ax.add_patch(FancyBboxPatch((x-1.1, 4.5), 2.2, 3.0, boxstyle="round,pad=0.05",
                                    facecolor='white', edgecolor=color, linewidth=2))
        # Title
        ax.text(x, 7.2, name.split('(')[0].strip(), fontsize=10, fontweight='bold',
                color=color, ha='center')
        ax.text(x, 6.8, f"({name.split('(')[1]}", fontsize=9, color='gray', ha='center')
        # Description
        ax.text(x, 6.0, desc, fontsize=8, color=COLORS['dark'], ha='center', va='center')
        # Note
        ax.text(x, 4.9, note, fontsize=7, color='gray', ha='center', va='center', style='italic')
    
    # Edge weight formula
    ax.text(5, 3.5, 'Edge Weight = PassCount × (1 + PreShotRatio)', fontsize=11,
            color=COLORS['dark'], ha='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#e8f8f5', edgecolor=COLORS['success']))
    
    # Pipeline
    ax.text(0.5, 2.3, 'Pipeline:', fontsize=11, fontweight='bold', color=COLORS['primary'])
    pipeline = 'IMPECT Events → Filter Passes → Build DiGraph → Compute Metrics → Aggregate Season'
    ax.text(0.5, 1.7, pipeline, fontsize=10, color=COLORS['dark'], family='monospace')
    
    # Note on weights
    ax.text(0.5, 0.8, '* Weights optimized empirically: original "Balance" inverted to "Hub Dependence" based on',
            fontsize=8, color='gray')
    ax.text(0.5, 0.4, '  correlation analysis showing elite teams are MORE centralized, not less.',
            fontsize=8, color='gray')


def create_validation_slide(fig, ax):
    """Slide 4: Validation Results"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Header
    ax.add_patch(Rectangle((0, 9), 10, 1, facecolor=COLORS['primary'], zorder=0))
    ax.text(0.3, 9.5, 'Validation: Cohesion Predicts Season Performance', fontsize=20,
            fontweight='bold', color='white', va='center')
    
    # Main result box
    ax.add_patch(FancyBboxPatch((0.3, 6.5), 4.4, 2.2, boxstyle="round,pad=0.1",
                                facecolor='#e8f6f3', edgecolor=COLORS['success'], linewidth=3))
    ax.text(2.5, 8.3, '✓ Season-Level', fontsize=14, fontweight='bold',
            color=COLORS['success'], ha='center')
    ax.text(2.5, 7.6, 'r = 0.728, p < 0.001', fontsize=18, fontweight='bold',
            color=COLORS['dark'], ha='center')
    ax.text(2.5, 7.0, 'Explains 53% of variance\nin season points', fontsize=10,
            color='gray', ha='center')
    
    # Match-level box
    ax.add_patch(FancyBboxPatch((5.3, 6.5), 4.4, 2.2, boxstyle="round,pad=0.1",
                                facecolor='#fef9e7', edgecolor=COLORS['warning'], linewidth=3))
    ax.text(7.5, 8.3, '✓ Match-Level', fontsize=14, fontweight='bold',
            color=COLORS['warning'], ha='center')
    ax.text(7.5, 7.6, 'F = 36.64, p < 0.0001', fontsize=18, fontweight='bold',
            color=COLORS['dark'], ha='center')
    ax.text(7.5, 7.0, 'Significant difference\nwin vs draw vs loss', fontsize=10,
            color='gray', ha='center')
    
    # Component correlations table
    ax.text(5, 5.8, 'Component Correlations with Season Points', fontsize=12,
            fontweight='bold', color=COLORS['primary'], ha='center')
    
    # Table
    table_data = [
        ('Component', 'r', 'p-value', 'Sig'),
        ('Connectivity', '+0.785', '0.0001', '***'),
        ('Hub Dependence', '+0.714', '0.0009', '***'),
        ('Chemistry', '+0.448', '0.0623', '*'),
        ('Progression', '+0.133', '0.5978', ''),
    ]
    
    y_start = 5.3
    for i, row in enumerate(table_data):
        y = y_start - i * 0.45
        bg_color = COLORS['light'] if i == 0 else 'white'
        weight = 'bold' if i == 0 else 'normal'
        
        ax.add_patch(Rectangle((1.5, y-0.2), 7, 0.4, facecolor=bg_color, edgecolor='gray', linewidth=0.5))
        ax.text(3.0, y, row[0], fontsize=10, ha='center', va='center', fontweight=weight)
        ax.text(5.5, y, row[1], fontsize=10, ha='center', va='center', fontweight=weight,
                color=COLORS['success'] if row[1].startswith('+0.7') else COLORS['dark'])
        ax.text(7.0, y, row[2], fontsize=10, ha='center', va='center', fontweight=weight)
        ax.text(8.0, y, row[3], fontsize=10, ha='center', va='center', fontweight=weight,
                color=COLORS['secondary'])
    
    # Key insight
    ax.add_patch(FancyBboxPatch((0.3, 0.8), 9.4, 1.5, boxstyle="round,pad=0.1",
                                facecolor='#fdedec', edgecolor=COLORS['secondary'], linewidth=2))
    ax.text(5, 1.9, '🔑 Key Finding: Connectivity is the strongest predictor', fontsize=11,
            fontweight='bold', color=COLORS['secondary'], ha='center')
    ax.text(5, 1.3, 'Dense, well-clustered passing networks strongly associated with success',
            fontsize=10, color=COLORS['dark'], ha='center')


def create_insight_slide(fig, ax):
    """Slide 5: The Hub Dependence Insight"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Header
    ax.add_patch(Rectangle((0, 9), 10, 1, facecolor=COLORS['secondary'], zorder=0))
    ax.text(0.3, 9.5, 'The "Balance Paradox": Star Players Matter', fontsize=20,
            fontweight='bold', color='white', va='center')
    
    # Before/After comparison
    ax.text(2.5, 8.2, 'Original Assumption', fontsize=14, fontweight='bold',
            color=COLORS['dark'], ha='center')
    ax.add_patch(FancyBboxPatch((0.5, 6.2), 4, 1.7, boxstyle="round,pad=0.1",
                                facecolor='#fadbd8', edgecolor=COLORS['secondary'], linewidth=2))
    ax.text(2.5, 7.5, '"Balanced" teams perform better', fontsize=11, ha='center')
    ax.text(2.5, 7.0, '(Even pass distribution)', fontsize=10, ha='center', color='gray')
    ax.text(2.5, 6.5, 'r = -0.714 with points ❌', fontsize=10, ha='center', 
            color=COLORS['secondary'], fontweight='bold')
    
    ax.text(7.5, 8.2, 'Empirical Reality', fontsize=14, fontweight='bold',
            color=COLORS['dark'], ha='center')
    ax.add_patch(FancyBboxPatch((5.5, 6.2), 4, 1.7, boxstyle="round,pad=0.1",
                                facecolor='#d5f5e3', edgecolor=COLORS['success'], linewidth=2))
    ax.text(7.5, 7.5, 'Hub-dependent teams win more', fontsize=11, ha='center')
    ax.text(7.5, 7.0, '(Star player centralization)', fontsize=10, ha='center', color='gray')
    ax.text(7.5, 6.5, 'r = +0.714 with points ✓', fontsize=10, ha='center',
            color=COLORS['success'], fontweight='bold')
    
    # Arrow
    ax.annotate('', xy=(5.3, 7.0), xytext=(4.7, 7.0),
                arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))
    
    # Examples
    ax.text(5, 5.5, 'Evidence from Top Teams:', fontsize=12, fontweight='bold',
            color=COLORS['primary'], ha='center')
    
    examples = [
        ('Leverkusen (90 pts)', 'Xhaka orchestrates; Wirtz finishes', COLORS['leverkusen']),
        ('Bayern (72 pts)', 'Kimmich as central hub', '#dc052d'),
        ('Stuttgart (75 pts)', 'Clear passing hierarchy', '#e32219'),
    ]
    
    for i, (team, desc, color) in enumerate(examples):
        y = 4.8 - i * 0.7
        ax.add_patch(Rectangle((1.5, y-0.25), 0.3, 0.5, facecolor=color))
        ax.text(2.0, y, team, fontsize=11, fontweight='bold', color=COLORS['dark'], va='center')
        ax.text(5.5, y, desc, fontsize=10, color='gray', va='center')
    
    # Bottom insight
    ax.add_patch(FancyBboxPatch((0.3, 0.8), 9.4, 1.8, boxstyle="round,pad=0.1",
                                facecolor=COLORS['light'], edgecolor=COLORS['primary'], linewidth=2))
    ax.text(5, 2.2, '📊 Implication for Lineup Selection', fontsize=11, fontweight='bold',
            color=COLORS['primary'], ha='center')
    ax.text(5, 1.5, 'Optimize for HUB CONNECTIVITY, not equal distribution.',
            fontsize=11, color=COLORS['dark'], ha='center')
    ax.text(5, 1.0, 'Build lineups around your best playmaker.',
            fontsize=10, color='gray', ha='center', style='italic')


def create_leverkusen_slide(fig, ax):
    """Slide 6: Leverkusen Case Study"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Header
    ax.add_patch(Rectangle((0, 9), 10, 1, facecolor=COLORS['leverkusen'], zorder=0))
    ax.text(0.3, 9.5, 'Case Study: Leverkusen\'s Undefeated Season', fontsize=20,
            fontweight='bold', color='white', va='center')
    
    # Season stats
    stats_box = FancyBboxPatch((0.3, 6.8), 3.2, 1.9, boxstyle="round,pad=0.1",
                               facecolor='white', edgecolor=COLORS['leverkusen'], linewidth=2)
    ax.add_patch(stats_box)
    ax.text(1.9, 8.4, '2023/24 Season', fontsize=11, fontweight='bold',
            color=COLORS['leverkusen'], ha='center')
    ax.text(1.9, 7.9, '28W - 6D - 0L', fontsize=14, fontweight='bold',
            color=COLORS['dark'], ha='center')
    ax.text(1.9, 7.4, '90 points | +63 GD', fontsize=11, color='gray', ha='center')
    ax.text(1.9, 6.95, '87 goals scored', fontsize=10, color='gray', ha='center')
    
    # Cohesion rank
    ax.add_patch(FancyBboxPatch((3.7, 6.8), 2.8, 1.9, boxstyle="round,pad=0.1",
                                facecolor='#e8f6f3', edgecolor=COLORS['success'], linewidth=2))
    ax.text(5.1, 8.4, 'Cohesion Score', fontsize=11, fontweight='bold',
            color=COLORS['success'], ha='center')
    ax.text(5.1, 7.7, '0.560', fontsize=20, fontweight='bold',
            color=COLORS['dark'], ha='center')
    ax.text(5.1, 7.0, 'League avg: 0.548', fontsize=9, color='gray', ha='center')
    
    # Key players box
    ax.add_patch(FancyBboxPatch((6.7, 6.8), 3.0, 1.9, boxstyle="round,pad=0.1",
                                facecolor='#fef9e7', edgecolor=COLORS['warning'], linewidth=2))
    ax.text(8.2, 8.4, 'Hub Players', fontsize=11, fontweight='bold',
            color=COLORS['warning'], ha='center')
    ax.text(8.2, 7.8, '🎯 Granit Xhaka', fontsize=10, color=COLORS['dark'], ha='center')
    ax.text(8.2, 7.4, '(Volume hub: 558 passes)', fontsize=8, color='gray', ha='center')
    ax.text(8.2, 7.0, '⚡ Florian Wirtz', fontsize=10, color=COLORS['dark'], ha='center')
    
    # Key connections table
    ax.text(5, 6.3, 'Top Attacking Connections (Pre-Shot Passes)', fontsize=11,
            fontweight='bold', color=COLORS['primary'], ha='center')
    
    connections = [
        ('Wirtz → Boniface', '22', '23.4%'),
        ('Frimpong → Boniface', '12', '38.7%'),
        ('Xhaka → Wirtz', '11', '6.1%'),
        ('Palacios → Boniface', '9', '15.5%'),
    ]
    
    y_start = 5.8
    ax.text(2.5, y_start, 'Connection', fontsize=9, fontweight='bold', ha='center')
    ax.text(5.5, y_start, 'Pre-Shot', fontsize=9, fontweight='bold', ha='center')
    ax.text(7.5, y_start, 'Conversion', fontsize=9, fontweight='bold', ha='center')
    
    for i, (conn, count, pct) in enumerate(connections):
        y = y_start - 0.45 - i * 0.4
        ax.text(2.5, y, conn, fontsize=9, ha='center')
        ax.text(5.5, y, count, fontsize=9, ha='center', fontweight='bold')
        ax.text(7.5, y, pct, fontsize=9, ha='center', color=COLORS['success'])
    
    # Tactical insight
    ax.add_patch(FancyBboxPatch((0.3, 0.8), 9.4, 2.2, boxstyle="round,pad=0.1",
                                facecolor=COLORS['light'], edgecolor=COLORS['primary'], linewidth=2))
    ax.text(5, 2.6, '🔍 Tactical Pattern: The Leverkusen Chain', fontsize=11,
            fontweight='bold', color=COLORS['primary'], ha='center')
    ax.text(5, 1.9, 'Defense → Xhaka (orchestrate) → Wirtz (create) → Boniface (finish)',
            fontsize=11, color=COLORS['dark'], ha='center', family='monospace')
    ax.text(5, 1.2, 'Wirtz leads ALL pre-shot connections — the attacking fulcrum',
            fontsize=10, color='gray', ha='center', style='italic')


def create_application_slide(fig, ax):
    """Slide 7: Application - Lineup Recommendation"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Header
    ax.add_patch(Rectangle((0, 9), 10, 1, facecolor=COLORS['primary'], zorder=0))
    ax.text(0.3, 9.5, 'Application: Lineup Optimization Framework', fontsize=20,
            fontweight='bold', color='white', va='center')
    
    # Use case boxes
    ax.text(2.5, 8.2, 'For Coaches', fontsize=12, fontweight='bold',
            color=COLORS['primary'], ha='center')
    ax.add_patch(FancyBboxPatch((0.5, 5.8), 4, 2.1, boxstyle="round,pad=0.1",
                                facecolor='white', edgecolor=COLORS['primary'], linewidth=2))
    coach_uses = [
        '• Identify optimal player combinations',
        '• Quantify impact of substitutions',
        '• Plan for injuries: who maintains',
        '  hub connectivity?',
    ]
    for i, use in enumerate(coach_uses):
        ax.text(0.7, 7.5 - i*0.45, use, fontsize=10, color=COLORS['dark'])
    
    ax.text(7.5, 8.2, 'For Analysts', fontsize=12, fontweight='bold',
            color=COLORS['success'], ha='center')
    ax.add_patch(FancyBboxPatch((5.5, 5.8), 4, 2.1, boxstyle="round,pad=0.1",
                                facecolor='white', edgecolor=COLORS['success'], linewidth=2))
    analyst_uses = [
        '• Compare lineup alternatives',
        '• Scout opposition networks',
        '• Evaluate transfer targets\' fit',
        '  with existing passing structure',
    ]
    for i, use in enumerate(analyst_uses):
        ax.text(5.7, 7.5 - i*0.45, use, fontsize=10, color=COLORS['dark'])
    
    # Example: Lineup comparison
    ax.text(5, 5.2, 'Example: Evaluating a Lineup Change', fontsize=11,
            fontweight='bold', color=COLORS['dark'], ha='center')
    
    ax.add_patch(FancyBboxPatch((1, 3.2), 3.5, 1.7, boxstyle="round,pad=0.1",
                                facecolor='#fadbd8', edgecolor=COLORS['secondary'], linewidth=1))
    ax.text(2.75, 4.6, 'Lineup A (Current)', fontsize=10, fontweight='bold', ha='center')
    ax.text(2.75, 4.1, 'Cohesion: 0.52', fontsize=12, ha='center')
    ax.text(2.75, 3.6, 'Connectivity: 0.31', fontsize=9, ha='center', color='gray')
    
    ax.add_patch(FancyBboxPatch((5.5, 3.2), 3.5, 1.7, boxstyle="round,pad=0.1",
                                facecolor='#d5f5e3', edgecolor=COLORS['success'], linewidth=1))
    ax.text(7.25, 4.6, 'Lineup B (Proposed)', fontsize=10, fontweight='bold', ha='center')
    ax.text(7.25, 4.1, 'Cohesion: 0.58 (+12%)', fontsize=12, ha='center',
            color=COLORS['success'])
    ax.text(7.25, 3.6, 'Connectivity: 0.36', fontsize=9, ha='center', color='gray')
    
    ax.annotate('', xy=(5.3, 4.0), xytext=(4.7, 4.0),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2))
    
    # Limitations
    ax.add_patch(FancyBboxPatch((0.3, 0.8), 9.4, 1.8, boxstyle="round,pad=0.1",
                                facecolor='#f9ebea', edgecolor='gray', linewidth=1))
    ax.text(5, 2.2, '⚠️ Limitations', fontsize=10, fontweight='bold', color='gray', ha='center')
    ax.text(5, 1.6, '• Requires historical pass data • Does not capture off-ball movement',
            fontsize=9, color='gray', ha='center')
    ax.text(5, 1.1, '• Single season validation • Opponent-specific effects not modeled',
            fontsize=9, color='gray', ha='center')


def create_conclusion_slide(fig, ax):
    """Slide 8: Conclusions"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Header
    ax.add_patch(Rectangle((0, 9), 10, 1, facecolor=COLORS['primary'], zorder=0))
    ax.text(0.3, 9.5, 'Conclusions & Future Work', fontsize=20,
            fontweight='bold', color='white', va='center')
    
    # Key contributions
    ax.text(5, 8.2, 'Key Contributions', fontsize=14, fontweight='bold',
            color=COLORS['primary'], ha='center')
    
    contributions = [
        ('1. Validated Cohesion Metric', 'r = 0.728 with season points (p < 0.001)'),
        ('2. Hub Dependence Insight', 'Elite teams centralize play through stars'),
        ('3. Pre-Shot Weighting', 'Edges weighted by attacking contribution'),
        ('4. Leverkusen Case Study', 'Xhaka→Wirtz→Boniface chain identified'),
    ]
    
    for i, (title, desc) in enumerate(contributions):
        y = 7.5 - i * 0.9
        ax.add_patch(FancyBboxPatch((0.5, y-0.35), 9, 0.7, boxstyle="round,pad=0.05",
                                    facecolor=COLORS['light'], edgecolor=COLORS['primary'], linewidth=1))
        ax.text(0.7, y, title, fontsize=11, fontweight='bold', color=COLORS['primary'], va='center')
        ax.text(4.5, y, desc, fontsize=10, color=COLORS['dark'], va='center')
    
    # Future work
    ax.text(5, 3.8, 'Future Directions', fontsize=12, fontweight='bold',
            color=COLORS['success'], ha='center')
    
    future = [
        '• Incorporate opponent-specific adjustments',
        '• Add temporal dynamics (fatigue, momentum)',
        '• Extend to multi-season validation',
        '• Build interactive lineup optimizer tool',
    ]
    
    for i, item in enumerate(future):
        ax.text(5, 3.3 - i * 0.4, item, fontsize=10, color=COLORS['dark'], ha='center')
    
    # Thank you / contact
    ax.add_patch(FancyBboxPatch((2, 0.5), 6, 1.2, boxstyle="round,pad=0.1",
                                facecolor=COLORS['primary'], edgecolor=COLORS['primary'], linewidth=0))
    ax.text(5, 1.3, 'Thank You!', fontsize=16, fontweight='bold', color='white', ha='center')
    ax.text(5, 0.8, 'Code: github.com/[your-repo] | Questions?', fontsize=10,
            color=COLORS['light'], ha='center')


def generate_slides():
    """Generate all slides and save to PDF."""
    
    print("Generating slide deck...")
    
    slide_functions = [
        create_title_slide,
        create_problem_slide,
        create_methodology_slide,
        create_validation_slide,
        create_insight_slide,
        create_leverkusen_slide,
        create_application_slide,
        create_conclusion_slide,
    ]
    
    # Create PDF
    pdf_path = SLIDES_DIR / 'SoccerHackathon_Slides.pdf'
    
    with PdfPages(pdf_path) as pdf:
        for i, slide_func in enumerate(slide_functions, 1):
            fig, ax = plt.subplots(figsize=(12, 7))
            slide_func(fig, ax)
            
            # Save individual slide as PNG too
            png_path = SLIDES_DIR / f'slide_{i:02d}.png'
            plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"  Saved: {png_path.name}")
            
            # Add to PDF
            pdf.savefig(fig, bbox_inches='tight', facecolor='white')
            plt.close(fig)
    
    print(f"\n✓ PDF saved: {pdf_path}")
    print(f"  Total slides: {len(slide_functions)}")
    
    return pdf_path


if __name__ == "__main__":
    generate_slides()
    print("\nSlide deck generation complete!")
    print("Review slides in: figures/slides/")
