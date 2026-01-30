import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

DATA_DIR = Path('../data')
FIGURES_DIR = Path('../figures')

# 1. Load Data
df = pd.read_csv(DATA_DIR / 'team_season_summary.csv')

# 2. Re-create the metrics from your paper
# Hub Dependence is 1 - Balance
df['hub_dependence'] = 1 - df['cohesion_balance']

# 3. Calculate Z-Scores (Standardization)
# We must do this because your formula uses Z-scores (Z_c and Z_h)
df['Z_Connectivity'] = (df['cohesion_connectivity'] - df['cohesion_connectivity'].mean()) / df['cohesion_connectivity'].std()
df['Z_Hub'] = (df['hub_dependence'] - df['hub_dependence'].mean()) / df['hub_dependence'].std()

# 4. Calculate the "Optimized Cohesion Score" using your regression weights
# Formula: 11.87 * Z_C + 10.00 * Z_H
df['Optimized_Score'] = (11.87 * df['Z_Connectivity']) + (10.00 * df['Z_Hub'])

# 5. Plot
plt.figure(figsize=(10, 8))
sns.set_style("whitegrid")

# Create scatter plot
scatter = plt.scatter(
    df['Optimized_Score'], 
    df['points'], 
    c=df['goal_diff'], 
    cmap='RdYlGn', 
    s=150, 
    edgecolors='black',
    alpha=0.8
)

# Add team labels
for i, row in df.iterrows():
    plt.text(
        row['Optimized_Score'], 
        row['points'] + 1.5, 
        i, 
        fontsize=9, 
        ha='center', 
        fontweight='bold',
        alpha=0.8
    )

# Add the regression line
slope, intercept, r_value, p_value, std_err = stats.linregress(df['Optimized_Score'], df['points'])
line_x = df['Optimized_Score']
line_y = slope * line_x + intercept
plt.plot(line_x, line_y, color='red', linestyle='--', linewidth=2, label=f'R² = {r_value**2:.3f}')

# Labels and Titles
plt.title(f"The Hub Dependence Paradox: Validating the Metric\n(R² = {r_value**2:.3f})", fontsize=14, fontweight='bold', pad=20)
plt.xlabel("Optimized Cohesion Metric\n(11.87 × Connectivity + 10.00 × Hub Dependence)", fontsize=11)
plt.ylabel("Season Points", fontsize=11)
plt.legend(loc='upper left', frameon=True, fontsize=11)

# Colorbar for Goal Difference
cbar = plt.colorbar(scatter)
cbar.set_label('Goal Difference', rotation=270, labelpad=15)

# Save
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'cohesion_vs_points_FINAL.png', dpi=300)
print(f"Saved correct plot to {FIGURES_DIR / 'cohesion_vs_points_FINAL.png'}")
print(f"New R-squared: {r_value**2:.3f}")