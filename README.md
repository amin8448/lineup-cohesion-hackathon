# The Hub Dependence Paradox: Network Centralization as a Predictor of Success in Elite Football

**NEU Sports Analytics Hackathon 2026** — Prompt A (Starting Eleven Lineup Construction)

**Authors:** Mohammad-Amin Nabavi, Shirley Mills  
**Affiliation:** School of Mathematics and Statistics, Carleton University, Ottawa, Canada  
**Contact:** aminnabavi@cmail.carleton.ca
**Live Demo:** [https://lineup-cohesion.streamlit.app](https://lineup-cohesion.streamlit.app)
---

## Key Results

| Metric                         | Value                                |
| ------------------------------ | ------------------------------------ |
| **Model R²**                   | **0.873** (87.3% variance explained) |
| **Cross-validation R²**        | **0.539** (100 random 70/30 splits)  |
| **Connectivity coefficient**   | β = 11.87 (p < 0.001)                |
| **Hub Dependence coefficient** | β = 10.00 (p < 0.001)                |

### The Hub Dependence Paradox

> **Conventional wisdom:** Balanced passing networks (equal distribution) should outperform centralized ones.
>
> **Our finding:** Elite teams benefit from strategic centralization through hub players. Network inequality _positively_ predicts success.

---

## Final Model

```
Expected Points = 46.5 + 11.87 * Z_Connectivity + 10.00 * Z_Hub_Dependence
```

| Component          | Coefficient | p-value | Description                              |
| ------------------ | ----------- | ------- | ---------------------------------------- |
| **Connectivity**   | +11.87      | < 0.001 | Network density + clustering coefficient |
| **Hub Dependence** | +10.00      | < 0.001 | Gini coefficient of degree distribution  |

_Coefficients are unstandardized: a 1-SD increase in each metric corresponds to ~10-12 additional league points._

---

## Methodology

### Data

- **Source:** IMPECT Open Data via kloppy
- **Scope:** Bundesliga 2023/24 season (306 matches, 18 teams, 612 team-match observations)
- **Network construction:** Directed, weighted passing networks per team per match

### Metrics

- **Density:** Proportion of possible edges that exist
- **Clustering Coefficient:** Tendency of players to form connected triads
- **Degree Centralization (Hub Dependence):** Gini coefficient of pass involvement

### Validation

- Stepwise OLS regression eliminated Chemistry (p=0.919) and Progression (p=0.516)
- Final 2-component model validated with 100x cross-validation
- Moderate test R² (0.539) confirms generalization despite small sample (N=18)

---

## Project Structure

```
lineup-cohesion-hackathon/
├── README.md
├── LICENSE                          # MIT License
├── requirements.txt                 # Python dependencies
│
├── data/
│   ├── cohesion_results.csv         # Match-level results (612 observations)
│   ├── team_season_summary.csv      # Season aggregates (18 teams)
│   └── player_impacts.csv           # What-if analysis results
│
├── src/
│   └── cohesion_metric.py           # Core metric implementation
│
├── notebooks/
│   ├── 01_data_exploration.py       # Schema discovery
│   ├── 02_test_cohesion.py          # Quick test on sample match
│   ├── 03_full_analysis.py          # Process all 306 matches
│   ├── 04_leverkusen_case_study.py  # Undefeated season deep dive
│   ├── 05_revalidate_optimized.py   # Re-validation with optimized weights
│   ├── 06_derive_weights.py         # OLS regression for coefficients
│   ├── 07_final_model.py            # Final 2-component model + cross-validation
│   ├── 08_darmstadt_case_study.py   # Relegated team contrast case
│   ├── 09_whatif_simulator.py       # Player removal impact analysis
│   ├── 10_all_teams_whatif.py       # League-wide player impacts
│   ├── 11_plots.py                  # Publication figures
│   └── 12_generate_slides.py        # Slide deck generation
│
├── app/
│   ├── streamlit_app.py             # Interactive dashboard
│   ├── requirements_app.txt         # Streamlit dependencies
│   └── run_app.sh                   # Launch script
│
├── paper/
│   └── soccer_paper_2026.tex        # Publication-ready LaTeX paper
│
├── figures/
│   ├── cohesion_vs_points_FINAL.png # Main validation plot
│   └── leverkusen_network.png       # Case study visualization
│
└── slides/
    └── Nabavi_Hackathon2026.pdf     # Final presentation (8 slides)
```

---

## Quick Start

### 1. Setup Environment

```bash
git clone https://github.com/amin8448/lineup-cohesion-hackathon.git
cd lineup-cohesion-hackathon

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Full Analysis Pipeline

```bash
cd notebooks

# Process all 306 matches
python 03_full_analysis.py

# Derive regression weights
python 06_derive_weights.py

# Final model with cross-validation
python 07_final_model.py

# Generate publication figures
python 11_plots.py
```

### 3. Run Interactive Dashboard

**Live Demo:** [https://lineup-cohesion.streamlit.app](https://lineup-cohesion.streamlit.app)

```bash
cd app
pip install -r requirements_app.txt
streamlit run streamlit_app.py --server.port 8502
```

---

## Case Studies

### Bayer Leverkusen (1st Place - Undefeated)

**Record:** 28W-6D-0L | 90 points | +63 GD

**Network Architecture:** Dual-Hub System

- **Volume Hub (Granit Xhaka):** 95 passes/90min, 92% completion - possession recycler
- **Attack Hub (Florian Wirtz):** Highest betweenness centrality - creative catalyst

**Resilience:** When top player (Palacios) removed, cohesion drops only **1.1%**

### SV Darmstadt 98 (18th Place - Relegated)

**Record:** 4W-7D-23L | 17 points | -50 GD

**Network Architecture:** Fragmented - no consistent hub structure

**Fragility:** When top player (Nurnberger) removed, cohesion drops **7.8%**

### The Insight

Elite teams don't just have better players - they have systems that distribute cohesion across depth. Bayern Munich (champions) has the **lowest** max player impact (0.29%), while struggling teams show dangerous star-dependence.

---

## Interactive Dashboard Features

1. **Overview** - Metric explanation, validation statistics, league rankings
2. **Team Analysis** - Individual team cohesion gauges and season trajectories
3. **Team Comparison** - Radar charts and head-to-head analysis
4. **What-If Simulator** - Player removal impact analysis for all 18 teams
5. **League Insights** - Cohesion vs Points scatter, component correlations

---

## Paper

The full publication-ready paper is available in `paper/soccer_paper_2026.tex`.

**Citation:**

```bibtex
@article{nabavi2026hub,
  title={The Hub Dependence Paradox: Network Centralization as a Predictor of Success in Elite Football},
  author={Nabavi, Mohammad-Amin and Mills, Shirley},
  journal={Northeastern University Sports Analytics Hackathon},
  year={2026}
}
```

---

## Limitations

- **Sample size:** N=18 teams limits statistical power
- **Single season:** Generalizability requires replication across leagues/years
- **Confounders:** Wage expenditure, transfer value not controlled
- **Causality:** Correlational analysis only
- **Data quality:** ~24% of passes lack receiver ID

---

## AI Disclosure

Claude (Anthropic) assisted with code development, data analysis, statistical interpretation, paper writing, and slide formatting. All results were validated and reviewed by the authors.

---

## License

**Code:** MIT License

**Data:** IMPECT Open Data - Non-commercial use only. See [IMPECT License](https://github.com/impect/open-data).

---

## Acknowledgments

- IMPECT for providing open event data
- PySport/kloppy for data standardization tools
- Northeastern University Sports Analytics for hosting the hackathon
