# Network-Based Lineup Cohesion Metric

**NEU Sports Analytics Hackathon 2026**  
**Author:** Mohammad-Amin Nabavi

---

## Research Question

Can we build a **network-based cohesion metric** that predicts match outcomes?

**Hypothesis:** Teams with CM high betweenness + strong CM-Winger weighted connections outperform fragmented midfields.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run data exploration (Phase 1)
cd notebooks
python 01_data_exploration.py

# 3. Run full analysis (Phase 2)
python 02_full_analysis.py
```

---

## Project Structure

```
Soccer analytics Hackathon/
├── data/                     # IMPECT data cache
├── src/
│   └── cohesion_metric.py    # Core metric implementation
├── notebooks/
│   ├── 01_data_exploration.py   # Phase 1: Schema discovery
│   ├── 02_full_analysis.py      # Phase 2: All 306 matches
│   ├── 03_validation.py         # Phase 3: Statistical validation
│   └── 04_leverkusen_case.py    # Phase 4: Case study
├── figures/                  # Output visualizations
├── submission/               # Final slides and deliverables
├── requirements.txt
└── README.md
```

---

## Methodology

### Cohesion Score Components

| Component | Description | Calculation |
|-----------|-------------|-------------|
| **Connectivity** | Network density & clustering | `0.5 × density + 0.5 × avg_clustering` |
| **Chemistry** | Weighted edges for key position pairs | CM↔LW, CM↔RW, CM↔ST connections |
| **Balance** | Inverse Gini of degree distribution | Lower inequality = higher score |
| **Progression** | Pre-shot pass ratio | Passes leading to shots |

**Final Score:** `Cohesion = α×Connectivity + β×Chemistry + γ×Balance + δ×Progression`

### Key Innovation: Pre-Shot Pass Weighting

Unlike traditional passing networks that treat all passes equally, we weight edges by:

```
edge_weight = pass_count × (1 + pre_shot_ratio)
```

Where `pre_shot_ratio` = proportion of passes between two players that occurred in the 2 passes before a shot.

This captures **attacking progression**, not just ball circulation.

---

## Data

- **Source:** IMPECT Open Data via kloppy
- **Scope:** Bundesliga 2023/24 (306 matches, 18 teams)
- **Primary Case Study:** Bayer Leverkusen (undefeated season)

---

## Validation Strategy

1. **Season-level:** Team average cohesion vs final league points
2. **Match-level:** Pre-match cohesion predicts win/draw/loss
3. **Case study:** Leverkusen cohesion trajectory across season

---

## License

MIT License - See LICENSE file

Data: IMPECT Open Data - Non-commercial use only
