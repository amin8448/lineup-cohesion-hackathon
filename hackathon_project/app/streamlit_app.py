"""
LINEUP COHESION ANALYZER - Interactive Streamlit Dashboard
===========================================================

NEU Sports Analytics Hackathon 2026
Author: Mohammad-Amin Nabavi

Run: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, '../src')

# Page config
st.set_page_config(
    page_title="Lineup Cohesion Analyzer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .insight-box {
        background-color: #e8f4ea;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load cached cohesion results."""
    data_path = Path('../data/cohesion_results.csv')
    if data_path.exists():
        return pd.read_csv(data_path)
    else:
        st.error("Data not found. Please run the analysis scripts first.")
        return None


@st.cache_data
def get_team_stats(df, team_name):
    """Get aggregated stats for a team."""
    team_df = df[df['team_name'].str.contains(team_name, case=False)]
    if len(team_df) == 0:
        return None
    
    stats = {
        'matches': len(team_df),
        'wins': len(team_df[team_df['result'] == 'win']),
        'draws': len(team_df[team_df['result'] == 'draw']),
        'losses': len(team_df[team_df['result'] == 'loss']),
        'avg_cohesion': team_df['cohesion_total'].mean(),
        'avg_connectivity': team_df['cohesion_connectivity'].mean(),
        'avg_chemistry': team_df['cohesion_chemistry'].mean(),
        'avg_balance': team_df['cohesion_balance'].mean(),
        'avg_progression': team_df['cohesion_progression'].mean(),
        'goals_for': team_df['goals_for'].sum(),
        'goals_against': team_df['goals_against'].sum(),
    }
    stats['points'] = stats['wins'] * 3 + stats['draws']
    return stats


def create_cohesion_gauge(value, title, max_val=1.0):
    """Create a gauge chart for cohesion metrics."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, max_val], 'tickwidth': 1},
            'bar': {'color': "#1f77b4"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, max_val*0.33], 'color': '#ffcccc'},
                {'range': [max_val*0.33, max_val*0.66], 'color': '#ffffcc'},
                {'range': [max_val*0.66, max_val], 'color': '#ccffcc'}
            ],
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_radar_chart(teams_data):
    """Create radar chart comparing teams."""
    categories = ['Connectivity', 'Chemistry', 'Hub Dependence', 'Progression']
    
    fig = go.Figure()
    
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']
    
    for i, (team_name, data) in enumerate(teams_data.items()):
        values = [
            data['avg_connectivity'],
            data['avg_chemistry'],
            data['avg_balance'],
            data['avg_progression']
        ]
        values.append(values[0])  # Close the polygon
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=team_name,
            line_color=colors[i % len(colors)],
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=400
    )
    return fig


def create_trajectory_plot(df, team_name):
    """Create cohesion trajectory over season."""
    team_df = df[df['team_name'].str.contains(team_name, case=False)].copy()
    team_df = team_df.sort_values('match_id').reset_index(drop=True)
    
    # Create match number (1, 2, 3, ...)
    team_df['match_num'] = range(1, len(team_df) + 1)
    
    # Color by result
    color_map = {'win': '#2ecc71', 'draw': '#f39c12', 'loss': '#e74c3c'}
    team_df['color'] = team_df['result'].map(color_map)
    
    fig = go.Figure()
    
    # Scatter points
    for result in ['win', 'draw', 'loss']:
        result_df = team_df[team_df['result'] == result]
        if len(result_df) > 0:
            fig.add_trace(go.Scatter(
                x=result_df['match_num'],
                y=result_df['cohesion_total'],
                mode='markers',
                name=result.capitalize(),
                marker=dict(size=12, color=color_map[result]),
                hovertemplate='Match %{x}<br>Cohesion: %{y:.3f}<extra></extra>'
            ))
    
    # Rolling average line
    team_df['rolling_avg'] = team_df['cohesion_total'].rolling(5, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=team_df['match_num'],
        y=team_df['rolling_avg'],
        mode='lines',
        name='5-match avg',
        line=dict(color='#3498db', width=3)
    ))
    
    # League average
    league_avg = df['cohesion_total'].mean()
    fig.add_hline(y=league_avg, line_dash="dash", line_color="gray",
                  annotation_text=f"League avg: {league_avg:.3f}")
    
    fig.update_layout(
        title=f"{team_name}: Cohesion Trajectory",
        xaxis_title="Match Number",
        yaxis_title="Cohesion Score",
        height=400,
        showlegend=True
    )
    return fig


def create_network_viz(G, title="Passing Network"):
    """Create interactive network visualization using Plotly."""
    if G is None or G.number_of_nodes() == 0:
        return None
    
    # Get positions using spring layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(edge[2].get('weight', 1))
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines',
        opacity=0.5
    )
    
    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    
    betweenness = nx.betweenness_centrality(G, weight='weight')
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}<br>Betweenness: {betweenness.get(node, 0):.3f}")
        node_size.append(20 + betweenness.get(node, 0) * 100)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[n.split()[-1] for n in G.nodes()],  # Last name only
        textposition="top center",
        hovertext=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            size=node_size,
            color=list(betweenness.values()),
            colorbar=dict(title="Betweenness"),
            line_width=2
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=title,
                       showlegend=False,
                       hovermode='closest',
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=500
                   ))
    return fig


@st.cache_data
def load_player_impacts():
    """Load player impact data."""
    impact_path = Path('../data/player_impacts.csv')
    if impact_path.exists():
        return pd.read_csv(impact_path)
    else:
        return None


def player_impact_simulator(df, team_name):
    """Simulate player removal impact using real calculated data."""
    st.subheader("🔮 What-If Simulator")
    
    st.markdown("""
    **How it works:** We remove each player from the passing network and measure 
    how much cohesion drops. Higher drop = more important player.
    """)
    
    # Load impact data
    impact_df = load_player_impacts()
    
    if impact_df is None:
        st.error("Player impact data not found. Please run 09_all_teams_whatif.py first.")
        return
    
    # Filter to selected team
    team_impacts = impact_df[impact_df['team_name'] == team_name].copy()
    
    if len(team_impacts) == 0:
        st.warning(f"No impact data found for {team_name}")
        return
    
    # Sort by impact
    team_impacts = team_impacts.sort_values('avg_impact_pct', ascending=False)
    
    # Create impact visualization
    fig = px.bar(team_impacts.head(15), 
                 x='avg_impact_pct', 
                 y='player_name', 
                 orientation='h',
                 color='avg_impact_pct', 
                 color_continuous_scale='RdYlGn_r',
                 hover_data=['avg_passes', 'matches_played', 'avg_betweenness'],
                 labels={
                     'avg_impact_pct': 'Impact (%)',
                     'player_name': 'Player',
                     'avg_passes': 'Avg Passes',
                     'matches_played': 'Matches',
                     'avg_betweenness': 'Betweenness'
                 })
    fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Player selector
    player_list = team_impacts['player_name'].tolist()
    selected_player = st.selectbox("Select a player to analyze:", player_list)
    
    if selected_player:
        player_data = team_impacts[team_impacts['player_name'] == selected_player].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Cohesion Drop", f"{player_data['avg_impact_pct']:.2f}%")
        with col2:
            st.metric("Avg Passes/Game", f"{player_data['avg_passes']:.0f}")
        with col3:
            st.metric("Matches Played", f"{player_data['matches_played']:.0f}")
        with col4:
            st.metric("Betweenness", f"{player_data['avg_betweenness']:.3f}")
        
        # Interpretation
        impact = player_data['avg_impact_pct']
        if impact > 2.0:
            st.markdown(f"""
            <div class="warning-box">
            ⚠️ <strong>CRITICAL PLAYER</strong><br>
            {selected_player} is highly important to team cohesion. If injured or unavailable, 
            expect a significant drop in team connectivity. Consider developing backup options.
            </div>
            """, unsafe_allow_html=True)
        elif impact > 1.0:
            st.markdown(f"""
            <div class="insight-box">
            ⚡ <strong>IMPORTANT PLAYER</strong><br>
            {selected_player} contributes meaningfully to team cohesion. 
            Rotation should be managed carefully during congested fixtures.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info(f"✓ {selected_player} is valuable but the team can adapt without them.")
    
    # Team summary stats
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        top_impact = team_impacts['avg_impact_pct'].max()
        st.metric("Max Player Impact", f"{top_impact:.2f}%")
    with col2:
        avg_impact = team_impacts['avg_impact_pct'].mean()
        st.metric("Avg Player Impact", f"{avg_impact:.2f}%")
    with col3:
        critical_count = len(team_impacts[team_impacts['avg_impact_pct'] > 1.0])
        st.metric("Critical Players (>1%)", critical_count)


def main():
    # Header
    st.markdown('<p class="main-header">⚽ Lineup Cohesion Analyzer</p>', unsafe_allow_html=True)
    st.markdown("**NEU Sports Analytics Hackathon 2026** | Bundesliga 2023/24 Analysis")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page:", [
        "🏠 Overview",
        "📊 Team Analysis",
        "⚔️ Team Comparison",
        "🔮 What-If Simulator",
        "📈 League Insights"
    ])
    
    # Get unique teams
    teams = df['team_name'].unique()
    
    # =========================================
    # PAGE: Overview
    # =========================================
    if page == "🏠 Overview":
        st.header("What is Lineup Cohesion?")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Lineup Cohesion Score** measures how well a team's players connect 
            through their passing network. It combines four components:
            
            1. **Connectivity** (50%) — How densely connected is the passing network?
            2. **Chemistry** (25%) — Are critical position pairs (e.g., midfield→attack) linking well?
            3. **Hub Dependence** (15%) — Does the team funnel play through star players?
            4. **Progression** (10%) — What % of passes lead to shots?
            
            ### Key Finding: The Hub Dependence Paradox
            
            > *"Elite teams don't spread the ball around equally — they funnel play 
            through 1-2 star players. Leverkusen runs everything through Xhaka and Wirtz. 
            That's not a weakness — it's their strength."*
            """)
        
        with col2:
            st.markdown("### Validation")
            st.metric("Correlation with Points", "r = 0.728")
            st.metric("Statistical Significance", "p = 0.0006")
            st.metric("Matches Analyzed", "306")
        
        # League overview
        st.header("League Overview")
        
        # Top/Bottom teams
        team_cohesion = df.groupby('team_name')['cohesion_total'].mean().sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Highest Cohesion Teams")
            top_5 = team_cohesion.head(5)
            fig = px.bar(x=top_5.values, y=top_5.index, orientation='h',
                        color=top_5.values, color_continuous_scale='Greens')
            fig.update_layout(height=300, showlegend=False, 
                            xaxis_title="Avg Cohesion", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📉 Lowest Cohesion Teams")
            bottom_5 = team_cohesion.tail(5)
            fig = px.bar(x=bottom_5.values, y=bottom_5.index, orientation='h',
                        color=bottom_5.values, color_continuous_scale='Reds_r')
            fig.update_layout(height=300, showlegend=False,
                            xaxis_title="Avg Cohesion", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
    
    # =========================================
    # PAGE: Team Analysis
    # =========================================
    elif page == "📊 Team Analysis":
        st.header("Team Deep Dive")
        
        selected_team = st.selectbox("Select Team:", sorted(teams))
        
        if selected_team:
            stats = get_team_stats(df, selected_team)
            
            if stats:
                # Key metrics row
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Points", stats['points'])
                with col2:
                    st.metric("Wins", stats['wins'])
                with col3:
                    st.metric("Goals For", stats['goals_for'])
                with col4:
                    st.metric("Goals Against", stats['goals_against'])
                with col5:
                    st.metric("Avg Cohesion", f"{stats['avg_cohesion']:.3f}")
                
                # Cohesion gauges
                st.subheader("Cohesion Components")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.plotly_chart(create_cohesion_gauge(stats['avg_connectivity'], "Connectivity"), 
                                   use_container_width=True)
                with col2:
                    st.plotly_chart(create_cohesion_gauge(stats['avg_chemistry'], "Chemistry"),
                                   use_container_width=True)
                with col3:
                    st.plotly_chart(create_cohesion_gauge(stats['avg_balance'], "Hub Dependence"),
                                   use_container_width=True)
                with col4:
                    st.plotly_chart(create_cohesion_gauge(stats['avg_progression'], "Progression"),
                                   use_container_width=True)
                
                # Trajectory
                st.subheader("Season Trajectory")
                fig = create_trajectory_plot(df, selected_team)
                st.plotly_chart(fig, use_container_width=True)
    
    # =========================================
    # PAGE: Team Comparison
    # =========================================
    elif page == "⚔️ Team Comparison":
        st.header("Head-to-Head Comparison")
        
        col1, col2 = st.columns(2)
        with col1:
            team1 = st.selectbox("Team 1:", sorted(teams), index=list(sorted(teams)).index('Bayer 04 Leverkusen') if 'Bayer 04 Leverkusen' in teams else 0)
        with col2:
            team2 = st.selectbox("Team 2:", sorted(teams), index=list(sorted(teams)).index('SV Darmstadt 98') if 'SV Darmstadt 98' in teams else 1)
        
        if team1 and team2:
            stats1 = get_team_stats(df, team1)
            stats2 = get_team_stats(df, team2)
            
            if stats1 and stats2:
                # Side by side metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(team1)
                    st.metric("Points", stats1['points'])
                    st.metric("Avg Cohesion", f"{stats1['avg_cohesion']:.3f}")
                    st.metric("Record", f"{stats1['wins']}W-{stats1['draws']}D-{stats1['losses']}L")
                
                with col2:
                    st.subheader(team2)
                    st.metric("Points", stats2['points'])
                    st.metric("Avg Cohesion", f"{stats2['avg_cohesion']:.3f}")
                    st.metric("Record", f"{stats2['wins']}W-{stats2['draws']}D-{stats2['losses']}L")
                
                # Radar comparison
                st.subheader("Component Comparison")
                teams_data = {team1: stats1, team2: stats2}
                fig = create_radar_chart(teams_data)
                st.plotly_chart(fig, use_container_width=True)
                
                # Distribution comparison
                st.subheader("Cohesion Distribution")
                team1_df = df[df['team_name'].str.contains(team1.split()[0], case=False)]
                team2_df = df[df['team_name'].str.contains(team2.split()[0], case=False)]
                
                fig = go.Figure()
                fig.add_trace(go.Box(y=team1_df['cohesion_total'], name=team1, marker_color='#2ecc71'))
                fig.add_trace(go.Box(y=team2_df['cohesion_total'], name=team2, marker_color='#e74c3c'))
                fig.update_layout(height=400, yaxis_title="Cohesion Score")
                st.plotly_chart(fig, use_container_width=True)
    
    # =========================================
    # PAGE: What-If Simulator
    # =========================================
    elif page == "🔮 What-If Simulator":
        st.header("Player Impact Simulator")
        
        # Load impact data to get team list
        impact_df = load_player_impacts()
        
        if impact_df is not None:
            team_list = sorted(impact_df['team_name'].unique())
            selected_team = st.selectbox("Select Team:", team_list)
            
            player_impact_simulator(df, selected_team)
        else:
            st.error("Player impact data not found. Please run `python notebooks/09_all_teams_whatif.py` first.")
        
        # League-wide comparison
        st.markdown("---")
        st.subheader("💡 League-Wide Player Impact Comparison")
        
        if impact_df is not None:
            # Top 10 most impactful players in the league
            top_players = impact_df.nlargest(15, 'avg_impact_pct')
            
            fig = px.bar(top_players, 
                        x='avg_impact_pct', 
                        y='player_name',
                        color='team_name',
                        orientation='h',
                        title="Top 15 Most Impactful Players in Bundesliga",
                        labels={'avg_impact_pct': 'Avg Impact (%)', 'player_name': 'Player', 'team_name': 'Team'})
            fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Team dependency comparison
            st.subheader("Team Dependency Analysis")
            
            team_dependency = impact_df.groupby('team_name').agg({
                'avg_impact_pct': ['max', 'mean', 'std']
            }).reset_index()
            team_dependency.columns = ['Team', 'Max Impact', 'Avg Impact', 'Impact Std']
            team_dependency = team_dependency.sort_values('Max Impact', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Most Star-Dependent Teams** (high max impact)")
                fig = px.bar(team_dependency.head(9), x='Max Impact', y='Team', orientation='h',
                            color='Max Impact', color_continuous_scale='Reds')
                fig.update_layout(height=350, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Most Balanced Teams** (low max impact)")
                fig = px.bar(team_dependency.tail(9), x='Max Impact', y='Team', orientation='h',
                            color='Max Impact', color_continuous_scale='Greens_r')
                fig.update_layout(height=350, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="insight-box">
            <strong>Key Insight:</strong> Teams with high individual player impact are more vulnerable 
            to injuries and suspensions. Bayern Munich has the most distributed cohesion (lowest max impact), 
            while mid-table teams tend to be more dependent on specific players.
            </div>
            """, unsafe_allow_html=True)
    
    # =========================================
    # PAGE: League Insights
    # =========================================
    elif page == "📈 League Insights":
        st.header("League-Wide Analysis")
        
        # Cohesion vs Points scatter
        st.subheader("Cohesion vs Final Points")
        
        team_summary = df.groupby('team_name').agg({
            'cohesion_total': 'mean',
            'cohesion_connectivity': 'mean',
            'result': lambda x: (x == 'win').sum() * 3 + (x == 'draw').sum()
        }).reset_index()
        team_summary.columns = ['Team', 'Avg Cohesion', 'Avg Connectivity', 'Points']
        
        fig = px.scatter(team_summary, x='Avg Cohesion', y='Points',
                        text='Team', size='Avg Connectivity',
                        color='Points', color_continuous_scale='RdYlGn',
                        hover_data=['Avg Connectivity'])
        fig.update_traces(textposition='top center', textfont_size=8)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Component correlations
        st.subheader("What Matters Most?")
        
        components = ['cohesion_connectivity', 'cohesion_chemistry', 'cohesion_balance', 'cohesion_progression']
        correlations = []
        
        for comp in components:
            team_comp = df.groupby('team_name').agg({
                comp: 'mean',
                'result': lambda x: (x == 'win').sum() * 3 + (x == 'draw').sum()
            })
            corr = team_comp[comp].corr(team_comp['result'])
            # Clean up display name
            display_name = comp.replace('cohesion_', '').capitalize()
            correlations.append({'Component': display_name, 'Correlation': corr})
        
        corr_df = pd.DataFrame(correlations)
        fig = px.bar(corr_df, x='Component', y='Correlation',
                    color='Correlation', color_continuous_scale='RdYlGn',
                    text='Correlation')
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Methodology
        with st.expander("📚 Methodology"):
            st.markdown("""
            ### Data Source
            - IMPECT Open Data: 306 Bundesliga 2023/24 matches
            - Event-level passing data processed via Kloppy
            
            ### Cohesion Score Formula
            ```
            Total = 0.50 × Connectivity + 0.25 × Chemistry + 0.15 × Hub_Dependence + 0.10 × Progression
            ```
            
            ### Validation
            - Season-level correlation with points: r = 0.728 (p = 0.0006)
            - Match-level ANOVA: F = 15.23 (p < 0.001)
            
            ### GitHub Repository
            [github.com/amin8448/lineup-cohesion-hackathon](https://github.com/amin8448/lineup-cohesion-hackathon)
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("*NEU Sports Analytics Hackathon 2026 | Mohammad-Amin Nabavi*")


if __name__ == "__main__":
    main()
