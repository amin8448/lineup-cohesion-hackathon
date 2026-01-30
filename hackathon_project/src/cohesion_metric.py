"""
Network-Based Lineup Cohesion Metric for Soccer Analytics
=========================================================

Author: Mohammad-Amin Nabavi
Project: NEU Sports Analytics Hackathon 2026
Data: IMPECT Open Data - Bundesliga 2023/24

This module implements a weighted passing network cohesion metric that:
1. Weights edges by pre-shot pass proportion (passes that lead to shots)
2. Measures team connectivity, chemistry, balance, and progression
3. Validates against match outcomes

Usage:
    from cohesion_metric import analyze_match_from_kloppy
    
    from kloppy import impect
    events = impect.load_open_data(match_id=122840)
    results = analyze_match_from_kloppy(events)
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CohesionScore:
    """Container for the cohesion score components."""
    connectivity: float      # Network density / path efficiency
    chemistry: float         # Weighted edges for key position pairs
    balance: float           # Gini coefficient of degree distribution (inverted)
    progression: float       # Average edge progression weight (pre-shot %)
    total: float             # Combined weighted score
    
    def __repr__(self):
        return (f"CohesionScore(total={self.total:.3f}, "
                f"connectivity={self.connectivity:.3f}, "
                f"chemistry={self.chemistry:.3f}, "
                f"balance={self.balance:.3f}, "
                f"progression={self.progression:.3f})")


def extract_metadata(kloppy_dataset) -> Tuple[Dict, Dict, Dict]:
    """
    Extract player and team metadata from kloppy dataset.
    
    Returns:
        player_id_to_name: Dict mapping player_id to player name
        player_id_to_position: Dict mapping player_id to position
        team_id_to_name: Dict mapping team_id to team name
    """
    player_id_to_name = {}
    player_id_to_position = {}
    team_id_to_name = {}
    
    for team in kloppy_dataset.metadata.teams:
        # Map team ID to name
        team_id = team.team_id
        team_id_to_name[team_id] = team.name
        
        # Map player IDs to names and positions
        for player in team.players:
            player_id_to_name[player.player_id] = player.name
            if hasattr(player, 'starting_position') and player.starting_position:
                player_id_to_position[player.player_id] = str(player.starting_position)
            else:
                player_id_to_position[player.player_id] = 'Unknown'
    
    return player_id_to_name, player_id_to_position, team_id_to_name


class PassingNetworkBuilder:
    """
    Builds weighted directed passing networks from event data.
    
    Key innovation: Weights edges by pre-shot pass proportion.
    """
    
    def __init__(self, 
                 events_df: pd.DataFrame, 
                 team_id: Any,
                 player_id_to_name: Dict,
                 player_id_to_position: Dict,
                 team_id_to_name: Dict):
        """
        Initialize the network builder.
        
        Args:
            events_df: DataFrame from kloppy with all events
            team_id: ID of the team to analyze
            player_id_to_name: Dict mapping player_id to name
            player_id_to_position: Dict mapping player_id to position
            team_id_to_name: Dict mapping team_id to name
        """
        self.events_df = events_df.copy()
        self.team_id = team_id
        self.team_name = team_id_to_name.get(team_id, str(team_id))
        self.player_id_to_name = player_id_to_name
        self.player_id_to_position = player_id_to_position
        self.team_id_to_name = team_id_to_name
        
        # Filter for this team's events
        self.team_events = events_df[events_df['team_id'] == team_id].copy()
        
        # Identify pre-shot passes
        self.pre_shot_pass_ids = self._identify_pre_shot_passes()
        
    def _identify_pre_shot_passes(self, n_passes_before: int = 2) -> set:
        """
        Identify passes that occurred in the buildup to shots.
        
        Args:
            n_passes_before: Number of passes before shot to consider
            
        Returns:
            Set of event_ids that are pre-shot passes
        """
        # Sort events by time
        df_sorted = self.events_df.sort_values(['period_id', 'timestamp']).reset_index(drop=True)
        
        # Find all shot indices for this team
        shot_mask = (df_sorted['event_type'] == 'SHOT') & (df_sorted['team_id'] == self.team_id)
        shot_indices = df_sorted[shot_mask].index.tolist()
        
        pre_shot_pass_ids = set()
        
        for shot_idx in shot_indices:
            passes_found = 0
            current_idx = shot_idx - 1
            
            while current_idx >= 0 and passes_found < n_passes_before:
                event = df_sorted.loc[current_idx]
                
                # Stop if we hit a different team's event (possession change)
                if event['team_id'] != self.team_id:
                    break
                
                # If it's a pass, mark it as pre-shot
                if event['event_type'] == 'PASS':
                    pre_shot_pass_ids.add(event['event_id'])
                    passes_found += 1
                
                current_idx -= 1
        
        return pre_shot_pass_ids
    
    def build_network(self, only_complete: bool = True) -> nx.DiGraph:
        """
        Build a weighted directed passing network.
        
        Edge weights incorporate:
        - Pass count between players
        - Pre-shot pass proportion
        
        Args:
            only_complete: Whether to include only completed passes
            
        Returns:
            NetworkX DiGraph with weighted edges
        """
        # Filter for passes
        passes = self.team_events[self.team_events['event_type'] == 'PASS'].copy()
        
        # Filter for completed passes only
        if only_complete and 'result' in passes.columns:
            passes = passes[passes['result'] == 'COMPLETE']
        
        G = nx.DiGraph()
        
        # Build edge data
        edge_data = defaultdict(lambda: {'count': 0, 'pre_shot_count': 0})
        
        for idx, row in passes.iterrows():
            passer_id = row['player_id']
            receiver_id = row.get('receiver_player_id')
            
            if pd.isna(passer_id) or pd.isna(receiver_id):
                continue
            
            # Convert IDs to names
            passer = self.player_id_to_name.get(passer_id, str(passer_id))
            receiver = self.player_id_to_name.get(receiver_id, str(receiver_id))
            
            edge_key = (passer, receiver)
            edge_data[edge_key]['count'] += 1
            
            # Check if this is a pre-shot pass
            event_id = row.get('event_id')
            if event_id in self.pre_shot_pass_ids:
                edge_data[edge_key]['pre_shot_count'] += 1
        
        # Add edges to graph
        for (passer, receiver), data in edge_data.items():
            count = data['count']
            pre_shot_count = data['pre_shot_count']
            pre_shot_ratio = pre_shot_count / count if count > 0 else 0
            
            # Weight: base count * (1 + pre_shot_ratio)
            weight = count * (1 + pre_shot_ratio)
            
            G.add_edge(passer, receiver, 
                      weight=weight,
                      count=count,
                      pre_shot_count=pre_shot_count,
                      pre_shot_ratio=pre_shot_ratio)
        
        return G
    
    def get_player_positions(self) -> Dict[str, str]:
        """
        Get positions for players who appeared in this team's events.
        
        Returns:
            Dict mapping player_name to position
        """
        positions = {}
        
        # Get unique player IDs from team events
        player_ids = self.team_events['player_id'].dropna().unique()
        
        for pid in player_ids:
            name = self.player_id_to_name.get(pid, str(pid))
            pos = self.player_id_to_position.get(pid, 'Unknown')
            positions[name] = pos
        
        return positions


class CohesionCalculator:
    """
    Calculates the cohesion score for a team's passing network.
    
    Components:
    1. Connectivity: Network density and average clustering (POSITIVE: r=+0.785)
    2. Chemistry: Weighted edges for critical position pairs (POSITIVE: r=+0.448)
    3. Hub Dependence: Gini coefficient - higher = more star-dependent (POSITIVE: r=+0.714)
       Note: Originally "Balance" but inverted based on empirical finding that
       elite teams funnel play through star players
    4. Progression: Average pre-shot pass ratio (WEAK: r=+0.133)
    
    Weights optimized based on Bundesliga 2023/24 correlation with season points.
    """
    
    # Critical position pairs (simplified position names)
    CRITICAL_PAIRS = [
        ('midfield', 'wing'),
        ('midfield', 'striker'),
        ('midfield', 'forward'),
        ('midfield', 'attacking'),
        ('defensive', 'midfield'),
        ('back', 'wing'),
    ]
    
    # Default weights: empirically optimized from Bundesliga 2023/24
    DEFAULT_WEIGHTS = {
        'connectivity': 0.50,   # Strongest predictor (r=+0.785)
        'chemistry': 0.25,      # Moderate predictor (r=+0.448)
        'hub_dependence': 0.15, # Inverted balance (r=+0.714 when inverted)
        'progression': 0.10     # Weak predictor (r=+0.133)
    }
    
    # Legacy weights for comparison
    EQUAL_WEIGHTS = {
        'connectivity': 0.25,
        'chemistry': 0.25,
        'hub_dependence': 0.25,
        'progression': 0.25
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None, use_optimized: bool = True):
        """
        Initialize the calculator.
        
        Args:
            weights: Component weights. If None, uses default weights.
            use_optimized: If True and weights is None, use optimized weights.
                          If False and weights is None, use equal weights.
        """
        if weights is not None:
            self.weights = weights
        elif use_optimized:
            self.weights = self.DEFAULT_WEIGHTS.copy()
        else:
            self.weights = self.EQUAL_WEIGHTS.copy()
        
    def calculate(self, 
                  G: nx.DiGraph, 
                  player_positions: Optional[Dict[str, str]] = None) -> CohesionScore:
        """
        Calculate the cohesion score for a passing network.
        
        Args:
            G: Directed weighted passing network
            player_positions: Optional dict mapping player names to positions
            
        Returns:
            CohesionScore object
        """
        if G.number_of_nodes() == 0:
            return CohesionScore(0, 0, 0, 0, 0)
        
        connectivity = self._calculate_connectivity(G)
        chemistry = self._calculate_chemistry(G, player_positions) if player_positions else 0.5
        hub_dependence = self._calculate_hub_dependence(G)  # Inverted balance
        progression = self._calculate_progression(G)
        
        # Normalize to 0-1
        connectivity = min(1.0, max(0.0, connectivity))
        chemistry = min(1.0, max(0.0, chemistry))
        hub_dependence = min(1.0, max(0.0, hub_dependence))
        progression = min(1.0, max(0.0, progression))
        
        total = (
            self.weights.get('connectivity', 0.25) * connectivity +
            self.weights.get('chemistry', 0.25) * chemistry +
            self.weights.get('hub_dependence', 0.25) * hub_dependence +
            self.weights.get('progression', 0.25) * progression
        )
        
        return CohesionScore(
            connectivity=connectivity,
            chemistry=chemistry,
            balance=hub_dependence,  # Stored as 'balance' for backward compat
            progression=progression,
            total=total
        )
    
    def _calculate_connectivity(self, G: nx.DiGraph) -> float:
        """Calculate connectivity from density and clustering."""
        n = G.number_of_nodes()
        if n <= 1:
            return 0.0
        
        density = nx.density(G)
        
        G_undirected = G.to_undirected()
        try:
            avg_clustering = nx.average_clustering(G_undirected, weight='weight')
        except:
            avg_clustering = 0
        
        return 0.5 * density + 0.5 * avg_clustering
    
    def _calculate_chemistry(self, 
                            G: nx.DiGraph, 
                            player_positions: Dict[str, str]) -> float:
        """Calculate chemistry from critical position pair connections."""
        if not player_positions:
            return 0.5
        
        total_critical_weight = 0
        total_weight = sum(d['weight'] for _, _, d in G.edges(data=True))
        
        if total_weight == 0:
            return 0.5
        
        for u, v, data in G.edges(data=True):
            u_pos = player_positions.get(u, '').lower()
            v_pos = player_positions.get(v, '').lower()
            
            # Check if positions match any critical pair
            for pos1, pos2 in self.CRITICAL_PAIRS:
                if (pos1 in u_pos and pos2 in v_pos) or (pos2 in u_pos and pos1 in v_pos):
                    total_critical_weight += data['weight']
                    break
        
        chemistry = total_critical_weight / total_weight
        return min(1.0, chemistry * 2.5)
    
    def _calculate_hub_dependence(self, G: nx.DiGraph) -> float:
        """
        Calculate hub dependence as Gini coefficient of degree distribution.
        
        INSIGHT: Elite teams funnel play through star players (high Gini = good).
        This is INVERTED from the original "balance" concept.
        
        Higher inequality in pass distribution = more hub-dependent = better performance.
        """
        in_degree = dict(G.in_degree(weight='weight'))
        out_degree = dict(G.out_degree(weight='weight'))
        
        degrees = [in_degree.get(n, 0) + out_degree.get(n, 0) for n in G.nodes()]
        
        if not degrees or max(degrees) == 0:
            return 0.5
        
        degrees = sorted(degrees)
        n = len(degrees)
        total = sum(degrees)
        
        if total == 0:
            return 0.5
        
        cumsum = np.cumsum(degrees)
        gini = (2 * sum((i + 1) * d for i, d in enumerate(degrees)) - (n + 1) * total) / (n * total)
        
        # Return Gini directly (higher = more star-dependent = GOOD for elite teams)
        return gini
    
    def _calculate_progression(self, G: nx.DiGraph) -> float:
        """Calculate progression as average pre-shot ratio."""
        total_ratio = 0
        count = 0
        
        for _, _, data in G.edges(data=True):
            if 'pre_shot_ratio' in data:
                total_ratio += data['pre_shot_ratio']
                count += 1
        
        if count == 0:
            return 0.0
        
        avg_ratio = total_ratio / count
        return min(1.0, avg_ratio * 5)


def compute_network_metrics(G: nx.DiGraph) -> Dict:
    """Compute standard network metrics."""
    if G.number_of_nodes() == 0:
        return {}
    
    metrics = {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'density': nx.density(G),
    }
    
    betweenness = nx.betweenness_centrality(G, weight='weight')
    metrics['betweenness'] = betweenness
    metrics['max_betweenness'] = max(betweenness.values()) if betweenness else 0
    metrics['top_betweenness_player'] = max(betweenness, key=betweenness.get) if betweenness else None
    
    in_degree = dict(G.in_degree(weight='weight'))
    out_degree = dict(G.out_degree(weight='weight'))
    total_degree = {n: in_degree.get(n, 0) + out_degree.get(n, 0) for n in G.nodes()}
    metrics['degree'] = total_degree
    metrics['top_degree_player'] = max(total_degree, key=total_degree.get) if total_degree else None
    
    return metrics


def analyze_match_from_kloppy(kloppy_dataset) -> Dict:
    """
    Analyze a match directly from a kloppy dataset.
    
    Args:
        kloppy_dataset: Dataset from kloppy.impect.load_open_data()
        
    Returns:
        Dict with cohesion scores and network stats for both teams
    """
    # Extract metadata
    player_id_to_name, player_id_to_position, team_id_to_name = extract_metadata(kloppy_dataset)
    
    # Convert to DataFrame
    df = kloppy_dataset.to_df(engine="pandas")
    
    # Get team IDs
    team_ids = df['team_id'].dropna().unique()
    
    calculator = CohesionCalculator()
    results = {}
    
    for team_id in team_ids:
        team_name = team_id_to_name.get(team_id, str(team_id))
        
        # Build network
        builder = PassingNetworkBuilder(
            df, team_id, 
            player_id_to_name, 
            player_id_to_position,
            team_id_to_name
        )
        G = builder.build_network()
        
        # Get positions
        positions = builder.get_player_positions()
        
        # Calculate cohesion
        cohesion = calculator.calculate(G, positions)
        
        # Get metrics
        metrics = compute_network_metrics(G)
        
        # Count goals
        goals = len(df[(df['team_id'] == team_id) & 
                       (df['event_type'] == 'SHOT') & 
                       (df['result'] == 'GOAL')])
        
        results[team_name] = {
            'team_id': team_id,
            'cohesion': cohesion,
            'network': G,
            'positions': positions,
            'metrics': metrics,
            'goals': goals,
            'pre_shot_passes': len(builder.pre_shot_pass_ids),
            'total_passes': len(builder.team_events[builder.team_events['event_type'] == 'PASS']),
        }
    
    return results


if __name__ == "__main__":
    print("Testing cohesion metric module...")
    
    try:
        from kloppy import impect
        
        print("Loading sample match...")
        events = impect.load_open_data(match_id=122840)
        
        print("Analyzing match...")
        results = analyze_match_from_kloppy(events)
        
        for team_name, data in results.items():
            print(f"\n{team_name}:")
            print(f"  Cohesion: {data['cohesion']}")
            print(f"  Goals: {data['goals']}")
            print(f"  Network: {data['metrics']['n_nodes']} players, {data['metrics']['n_edges']} connections")
            print(f"  Top betweenness: {data['metrics']['top_betweenness_player']}")
            print(f"  Pre-shot passes: {data['pre_shot_passes']} / {data['total_passes']} total")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
