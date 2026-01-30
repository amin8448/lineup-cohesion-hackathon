import sys
sys.path.insert(0, '../src')
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

DATA_DIR = Path('../data')
df = pd.read_csv(DATA_DIR / 'team_season_summary.csv')

# 1. Prepare the Final Features
# We DROP chemistry and progression.
df['hub_dependence'] = 1 - df['cohesion_balance']
features = ['cohesion_connectivity', 'hub_dependence']

X = df[features]
y = df['points']

# 2. Standardize (Z-scores)
X_scaled = (X - X.mean()) / X.std()
X_scaled = sm.add_constant(X_scaled)

# 3. Fit Final Model
model = sm.OLS(y, X_scaled).fit()

print(model.summary())

print("\n" + "="*40)
print("FINAL PUBLICATION COEFFICIENTS")
print("="*40)
print("Use these EXACT numbers in your methodology section:")
for name, coef in model.params.items():
    print(f"{name}: {coef:.4f}")

# 4. Cross-Validation (The "Robustness" Check)
# We train on random 70% of teams, test on 30%
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

print("\n" + "="*40)
print("ROBUSTNESS CHECK (Train/Test Split)")
print("="*40)

# We do 100 random trials to be sure
r2_scores = []
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=i)
    
    # Train
    model_cv = sm.OLS(y_train, X_train).fit()
    
    # Predict
    preds = model_cv.predict(X_test)
    
    # Score
    r2 = r2_score(y_test, preds)
    r2_scores.append(r2)

avg_r2 = sum(r2_scores) / len(r2_scores)
print(f"Average R-squared on unseen test data: {avg_r2:.3f}")
if avg_r2 > 0.5:
    print("PASS: The model generalizes well.")
else:
    print("WARNING: The model might be overfitting.")