#!/usr/bin/env python3

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import dash
from dash import dcc, html, Input, Output, callback
import json

from core.simulator import AllianceStrategy, ScoringSimulator, Zone, ParkingLocation
from ..analysis.strategy_analyzer import AdvancedStrategyAnalyzer
from ..analysis.scoring_analyzer import AdvancedScoringAnalyzer
from core.scenario_generator import ScenarioGenerator


class InteractiveVEXVisualizer:
    def __init__(self):
        self.simulator = ScoringSimulator()
        self.strategy_analyzer = AdvancedStrategyAnalyzer(self.simulator)
        self.scoring_analyzer = AdvancedScoringAnalyzer(self.simulator)
        self.generator = ScenarioGenerator(self.simulator)
    
    def create_scoring_timeline_visualization(
        self,
        strategies: List[AllianceStrategy],
        match_duration: int = 120,
        save_html: str = None
    ) -> go.Figure:
        """Create interactive scoring timeline showing how scores evolve over match time"""
        
        fig = go.Figure()
        
        time_points = np.linspace(0, match_duration, 50)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for i, strategy in enumerate(strategies[:5]):  # Limit to 5 for clarity
            # Calculate cumulative scoring over time
            cumulative_scores = []
            
            for t in time_points:
                # Calculate score at time t
                if t <= 15:  # Autonomous period
                    progress = t / 15
                    auto_blocks = sum(strategy.blocks_scored_auto.values())
                    score = int(auto_blocks * progress * 3)
                    if t == 15 and auto_blocks >= 8:  # Assume auto bonus if sufficient blocks
                        score += 10
                else:  # Driver control period
                    # Full auto score + driver progress
                    auto_score = sum(strategy.blocks_scored_auto.values()) * 3
                    auto_bonus = 10 if sum(strategy.blocks_scored_auto.values()) >= 8 else 0
                    
                    driver_progress = (t - 15) / (match_duration - 15)
                    driver_blocks = sum(strategy.blocks_scored_driver.values())
                    driver_score = int(driver_blocks * driver_progress * 3)
                    
                    # Add zone and parking points near end
                    zone_score = len(strategy.zones_controlled) * 10 if t >= match_duration - 30 else 0
                    parking_score = 0
                    if t >= match_duration - 15:  # Parking in last 15 seconds
                        for location in strategy.robots_parked:
                            if location == ParkingLocation.PLATFORM:
                                parking_score += 20
                            elif location == ParkingLocation.ALLIANCE_ZONE:
                                parking_score += 5
                    
                    score = auto_score + auto_bonus + driver_score + zone_score + parking_score
                
                cumulative_scores.append(score)
            
            # Add trace for this strategy
            fig.add_trace(go.Scatter(
                x=time_points,
                y=cumulative_scores,
                mode='lines+markers',
                name=strategy.name,
                line=dict(color=colors[i], width=3),
                marker=dict(size=4),
                hovertemplate=f'<b>{strategy.name}</b><br>' +
                             'Time: %{x:.0f}s<br>' +
                             'Score: %{y:.0f} points<br>' +
                             '<extra></extra>'
            ))
        
        # Add phase boundaries
        fig.add_vline(x=15, line_dash="dash", line_color="gray",
                     annotation_text="Auto End", annotation_position="top")
        fig.add_vline(x=90, line_dash="dash", line_color="orange", 
                     annotation_text="Endgame", annotation_position="top")
        
        # Customize layout
        fig.update_layout(
            title={
                'text': 'VEX U Match Scoring Timeline',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2E86C1'}
            },
            xaxis_title="Time (seconds)",
            yaxis_title="Cumulative Score (points)",
            xaxis=dict(range=[0, match_duration], dtick=15),
            yaxis=dict(range=[0, max([max(scores) for scores in [cumulative_scores] * len(strategies)]) * 1.1]),
            hovermode='x unified',
            template='plotly_white',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            height=600
        )
        
        # Add annotations for key phases
        fig.add_annotation(
            x=7.5, y=fig.layout.yaxis.range[1] * 0.9,
            text="Autonomous<br>Period",
            showarrow=False,
            bgcolor="rgba(255,107,107,0.1)",
            bordercolor="rgba(255,107,107,0.5)",
            font=dict(size=10)
        )
        
        fig.add_annotation(
            x=105, y=fig.layout.yaxis.range[1] * 0.9,
            text="Endgame<br>Push",
            showarrow=False,
            bgcolor="rgba(255,165,0,0.1)",
            bordercolor="rgba(255,165,0,0.5)",
            font=dict(size=10)
        )
        
        if save_html:
            fig.write_html(save_html)
        
        return fig
    
    def create_strategy_comparison_dashboard(
        self,
        strategies: List[AllianceStrategy],
        save_html: str = None
    ) -> go.Figure:
        """Create comprehensive strategy comparison dashboard"""
        
        # Analyze all strategies
        print("Analyzing strategies for comparison...")
        metrics_list = []
        for strategy in strategies:
            metrics = self.strategy_analyzer.analyze_strategy_comprehensive(strategy, 200)
            metrics_list.append(metrics)
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Scores', 'Win Rate Distribution', 
                          'Score Component Breakdown', 'Risk vs Reward Analysis'),
            specs=[[{"type": "bar"}, {"type": "box"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        strategy_names = [m.strategy_name for m in metrics_list]
        colors = px.colors.qualitative.Set2[:len(strategies)]
        
        # 1. Average Scores (Bar Chart)
        fig.add_trace(
            go.Bar(
                x=strategy_names,
                y=[m.avg_score for m in metrics_list],
                name='Avg Score',
                marker_color=colors,
                text=[f'{m.avg_score:.0f}' for m in metrics_list],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Score: %{y:.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Win Rate Box Plot (simulated distribution)
        for i, (name, metrics) in enumerate(zip(strategy_names, metrics_list)):
            # Simulate win rate distribution
            win_rates = np.random.normal(metrics.win_rate, 0.05, 100)
            win_rates = np.clip(win_rates, 0, 1) * 100  # Convert to percentage and clip
            
            fig.add_trace(
                go.Box(
                    y=win_rates,
                    name=name,
                    marker_color=colors[i],
                    showlegend=False,
                    hovertemplate=f'<b>{name}</b><br>Win Rate: %{{y:.1f}}%<extra></extra>'
                ),
                row=1, col=2
            )
        
        # 3. Component Breakdown (Stacked Bar)
        components = ['blocks', 'autonomous', 'zones', 'parking']
        component_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, comp in enumerate(components):
            values = [m.component_breakdown.get(comp, 0) for m in metrics_list]
            fig.add_trace(
                go.Bar(
                    x=strategy_names,
                    y=values,
                    name=comp.title(),
                    marker_color=component_colors[i],
                    hovertemplate=f'<b>%{{x}}</b><br>{comp.title()}: %{{y:.1f}}%<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 4. Risk vs Reward Scatter
        fig.add_trace(
            go.Scatter(
                x=[m.risk_reward_ratio for m in metrics_list],
                y=[m.win_rate * 100 for m in metrics_list],
                mode='markers+text',
                text=strategy_names,
                textposition='top center',
                marker=dict(
                    size=15,
                    color=colors,
                    line=dict(width=2, color='white')
                ),
                name='Strategies',
                showlegend=False,
                hovertemplate='<b>%{text}</b><br>' +
                             'Risk/Reward: %{x:.1f}<br>' +
                             'Win Rate: %{y:.1f}%<br>' +
                             '<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'VEX U Strategy Comparison Dashboard',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': '#2E86C1'}
            },
            template='plotly_white',
            height=800,
            showlegend=True,
            barmode='stack'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Strategy", row=1, col=1)
        fig.update_yaxes(title_text="Points", row=1, col=1)
        
        fig.update_xaxes(title_text="Strategy", row=1, col=2)
        fig.update_yaxes(title_text="Win Rate (%)", row=1, col=2)
        
        fig.update_xaxes(title_text="Strategy", row=2, col=1)
        fig.update_yaxes(title_text="Percentage of Score", row=2, col=1)
        
        fig.update_xaxes(title_text="Risk/Reward Ratio", row=2, col=2)
        fig.update_yaxes(title_text="Win Rate (%)", row=2, col=2)
        
        if save_html:
            fig.write_html(save_html)
        
        return fig
    
    def create_interactive_scenario_explorer(self, save_html: str = None):
        """Create interactive scenario explorer with parameter sliders"""
        
        app = dash.Dash(__name__)
        
        app.layout = html.Div([
            html.H1("VEX U Interactive Scenario Explorer", 
                   style={'textAlign': 'center', 'color': '#2E86C1'}),
            
            html.Div([
                html.Div([
                    html.H3("Robot Parameters"),
                    html.Label("Scoring Rate (blocks/sec):"),
                    dcc.Slider(
                        id='scoring-rate-slider',
                        min=0.1, max=0.8, step=0.05,
                        value=0.3,
                        marks={i/10: str(i/10) for i in range(1, 9)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    
                    html.Label("Defense Effectiveness (%):"),
                    dcc.Slider(
                        id='defense-slider',
                        min=0, max=100, step=5,
                        value=30,
                        marks={i: str(i) for i in range(0, 101, 20)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    
                    html.Label("Robot Capacity (blocks):"),
                    dcc.Slider(
                        id='capacity-slider',
                        min=1, max=8, step=1,
                        value=3,
                        marks={i: str(i) for i in range(1, 9)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    
                    html.Label("Match Time Remaining (sec):"),
                    dcc.Slider(
                        id='time-slider',
                        min=10, max=120, step=10,
                        value=60,
                        marks={i: str(i) for i in range(10, 121, 20)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
                
                html.Div([
                    dcc.Graph(id='scenario-comparison-graph')
                ], style={'width': '65%', 'display': 'inline-block', 'padding': '20px'})
            ]),
            
            html.Div([
                html.H3("Strategy Recommendations", style={'color': '#2E86C1'}),
                html.Div(id='recommendations-output', style={'padding': '10px', 'backgroundColor': '#f8f9fa'})
            ], style={'margin': '20px'})
        ])
        
        @app.callback(
            [Output('scenario-comparison-graph', 'figure'),
             Output('recommendations-output', 'children')],
            [Input('scoring-rate-slider', 'value'),
             Input('defense-slider', 'value'),
             Input('capacity-slider', 'value'),
             Input('time-slider', 'value')]
        )
        def update_scenario_analysis(scoring_rate, defense_eff, capacity, time_remaining):
            # Create strategies with different approaches
            strategies_data = []
            
            # Calculate performance for different strategy types
            strategy_types = ['Aggressive', 'Balanced', 'Defensive', 'Zone Control']
            multipliers = [1.2, 1.0, 0.7, 0.8]  # Scoring multipliers
            
            for strategy_type, multiplier in zip(strategy_types, multipliers):
                # Calculate expected blocks in remaining time
                effective_rate = scoring_rate * multiplier * (1 - defense_eff/100)
                expected_blocks = effective_rate * time_remaining
                expected_score = expected_blocks * 3
                
                # Add strategy-specific bonuses
                if strategy_type == 'Zone Control':
                    expected_score += 20  # Zone control points
                if strategy_type == 'Balanced':
                    expected_score += 25  # Parking points
                
                # Add some realistic variance
                score_variance = expected_score * 0.1
                
                strategies_data.append({
                    'Strategy': strategy_type,
                    'Expected_Score': expected_score,
                    'Min_Score': expected_score - score_variance,
                    'Max_Score': expected_score + score_variance,
                    'Expected_Blocks': expected_blocks
                })
            
            # Create comparison chart
            fig = go.Figure()
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            for i, data in enumerate(strategies_data):
                fig.add_trace(go.Bar(
                    x=[data['Strategy']],
                    y=[data['Expected_Score']],
                    name=data['Strategy'],
                    marker_color=colors[i],
                    error_y=dict(
                        type='data',
                        array=[data['Max_Score'] - data['Expected_Score']],
                        arrayminus=[data['Expected_Score'] - data['Min_Score']],
                        visible=True
                    ),
                    hovertemplate=f'<b>{data["Strategy"]}</b><br>' +
                                 f'Score: {data["Expected_Score"]:.0f}±{(data["Max_Score"]-data["Expected_Score"]):.0f}<br>' +
                                 f'Blocks: {data["Expected_Blocks"]:.1f}<br>' +
                                 '<extra></extra>'
                ))
            
            fig.update_layout(
                title=f'Strategy Performance Comparison<br>Rate: {scoring_rate} b/s, Defense: {defense_eff}%, Time: {time_remaining}s',
                xaxis_title='Strategy Type',
                yaxis_title='Expected Score (Points)',
                template='plotly_white',
                showlegend=False,
                height=400
            )
            
            # Generate recommendations
            best_strategy = max(strategies_data, key=lambda x: x['Expected_Score'])
            worst_strategy = min(strategies_data, key=lambda x: x['Expected_Score'])
            
            recommendations = [
                html.H4(f"🏆 Best Strategy: {best_strategy['Strategy']}", style={'color': '#27AE60'}),
                html.P(f"Expected Score: {best_strategy['Expected_Score']:.0f} points ({best_strategy['Expected_Blocks']:.1f} blocks)"),
                html.H4("📊 Key Insights:", style={'color': '#2E86C1'}),
                html.Ul([
                    html.Li(f"Scoring rate of {scoring_rate} blocks/sec allows {scoring_rate * time_remaining:.1f} total blocks"),
                    html.Li(f"Defense effectiveness of {defense_eff}% reduces scoring by {defense_eff}%"),
                    html.Li(f"Robot capacity of {capacity} blocks affects trip efficiency"),
                    html.Li(f"With {time_remaining}s remaining, focus on {'quick scoring' if time_remaining < 30 else 'consistent performance'}")
                ])
            ]
            
            if defense_eff > 50:
                recommendations.append(html.P("⚠️ High defense detected - consider zone control or parking strategies", 
                                            style={'color': '#E67E22'}))
            
            if time_remaining < 30:
                recommendations.append(html.P("🚨 Endgame phase - prioritize parking and zone control", 
                                            style={'color': '#E74C3C'}))
            
            return fig, recommendations
        
        if save_html:
            app.run_server(debug=False, port=8050)
        
        return app
    
    def create_key_insights_dashboard(
        self,
        analysis_results: Dict,
        save_html: str = None
    ) -> go.Figure:
        """Create comprehensive insights dashboard"""
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Top 5 Winning Strategies', 'Optimal Block Distribution',
                          'Time Allocation Recommendations', 'Risk Assessment',
                          'Win Probability Matrix', 'Performance Trends'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "scatter"}]]
        )
        
        # Sample data for demonstration
        top_strategies = [
            {'name': 'Fast & Furious', 'win_rate': 95, 'score': 270},
            {'name': 'Balanced', 'win_rate': 88, 'score': 240},
            {'name': 'Zone Control', 'win_rate': 82, 'score': 200},
            {'name': 'Endgame Focus', 'win_rate': 78, 'score': 220},
            {'name': 'Defensive', 'win_rate': 65, 'score': 180}
        ]
        
        # 1. Top 5 Strategies
        fig.add_trace(
            go.Bar(
                x=[s['name'] for s in top_strategies],
                y=[s['win_rate'] for s in top_strategies],
                name='Win Rate (%)',
                marker_color='#4ECDC4',
                text=[f"{s['win_rate']}%" for s in top_strategies],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # 2. Optimal Block Distribution
        block_dist = {'Long Goal 1': 25, 'Long Goal 2': 25, 'Center 1': 20, 'Center 2': 18}
        fig.add_trace(
            go.Pie(
                labels=list(block_dist.keys()),
                values=list(block_dist.values()),
                name="Block Distribution",
                marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            ),
            row=1, col=2
        )
        
        # 3. Time Allocation
        time_phases = ['Autonomous', 'Early Driver', 'Mid Driver', 'Endgame']
        time_values = [15, 30, 45, 30]
        fig.add_trace(
            go.Bar(
                x=time_phases,
                y=time_values,
                name='Time (seconds)',
                marker_color='#45B7D1'
            ),
            row=2, col=1
        )
        
        # 4. Risk Assessment
        strategies_risk = [s['name'] for s in top_strategies]
        risk_scores = [85, 70, 60, 75, 45]  # Risk scores
        reward_scores = [s['score'] for s in top_strategies]
        
        fig.add_trace(
            go.Scatter(
                x=risk_scores,
                y=reward_scores,
                mode='markers+text',
                text=strategies_risk,
                textposition='top center',
                marker=dict(size=12, color='#FF6B6B'),
                name='Risk vs Reward'
            ),
            row=2, col=2
        )
        
        # 5. Win Probability Heatmap
        skill_levels = ['Beginner', 'Intermediate', 'Advanced', 'Expert']
        opponents = ['Weak', 'Average', 'Strong']
        win_probs = np.array([[90, 70, 50, 30],
                             [95, 85, 65, 45],
                             [98, 92, 80, 60]])
        
        fig.add_trace(
            go.Heatmap(
                z=win_probs,
                x=skill_levels,
                y=opponents,
                colorscale='RdYlGn',
                text=win_probs,
                texttemplate="%{text}%",
                textfont={"size": 10},
                name="Win Probability"
            ),
            row=3, col=1
        )
        
        # 6. Performance Trends
        match_times = list(range(0, 121, 10))
        cumulative_performance = [i*2 + np.sin(i/10)*10 for i in range(len(match_times))]
        
        fig.add_trace(
            go.Scatter(
                x=match_times,
                y=cumulative_performance,
                mode='lines+markers',
                name='Performance Trend',
                line=dict(color='#96CEB4', width=3)
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'VEX U Push Back - Key Insights Dashboard',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': '#2E86C1'}
            },
            template='plotly_white',
            height=1000,
            showlegend=False
        )
        
        # Update specific subplot layouts
        fig.update_yaxes(title_text="Win Rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="Time (seconds)", row=2, col=1)
        fig.update_xaxes(title_text="Risk Score", row=2, col=2)
        fig.update_yaxes(title_text="Reward Score", row=2, col=2)
        fig.update_xaxes(title_text="Time (seconds)", row=3, col=2)
        fig.update_yaxes(title_text="Performance", row=3, col=2)
        
        if save_html:
            fig.write_html(save_html)
        
        return fig
    
    def create_match_outcome_predictor(
        self,
        current_red_score: int = 100,
        current_blue_score: int = 95,
        time_remaining: int = 30,
        save_html: str = None
    ) -> go.Figure:
        """Create match outcome predictor with win probability and recommendations"""
        
        # Calculate various scenarios
        scoring_rates = np.linspace(0.1, 0.6, 20)  # blocks per second
        scenarios = []
        
        for rate in scoring_rates:
            blocks_possible = rate * time_remaining
            points_possible = blocks_possible * 3
            
            # Add endgame bonuses
            endgame_bonus = 40 if time_remaining >= 15 else 20  # Parking points
            
            final_score = current_red_score + points_possible + endgame_bonus
            
            # Simple win probability calculation
            score_difference = final_score - current_blue_score
            if score_difference > 30:
                win_prob = 95
            elif score_difference > 15:
                win_prob = 80
            elif score_difference > 0:
                win_prob = 65
            elif score_difference > -15:
                win_prob = 35
            elif score_difference > -30:
                win_prob = 20
            else:
                win_prob = 5
            
            scenarios.append({
                'rate': rate,
                'final_score': final_score,
                'win_prob': win_prob,
                'blocks_needed': blocks_possible
            })
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Win Probability vs Scoring Rate',
                'Required Blocks to Win',
                'Score Projection Timeline',
                'Strategy Recommendations'
            ),
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "table"}]]
        )
        
        # 1. Win Probability vs Scoring Rate
        fig.add_trace(
            go.Scatter(
                x=[s['rate'] for s in scenarios],
                y=[s['win_prob'] for s in scenarios],
                mode='lines+markers',
                name='Win Probability (%)',
                line=dict(color='#4ECDC4', width=3),
                fill='tonexty'
            ),
            row=1, col=1
        )
        
        # Add threshold lines
        fig.add_hline(y=50, line_dash="dash", line_color="red", 
                     annotation_text="50% Win Rate", row=1, col=1)
        
        # 2. Required Blocks Bar Chart
        breakeven_blocks = max(0, (current_blue_score - current_red_score + 1) / 3)
        safety_blocks = breakeven_blocks + 5
        
        fig.add_trace(
            go.Bar(
                x=['Breakeven', 'Safe Win', 'Dominant'],
                y=[breakeven_blocks, safety_blocks, safety_blocks + 10],
                marker_color=['#FF6B6B', '#FFEAA7', '#96CEB4'],
                name='Blocks Needed'
            ),
            row=1, col=2
        )
        
        # 3. Score Projection
        time_points = np.linspace(0, time_remaining, 20)
        optimal_rate = 0.4  # Assume optimal scoring rate
        
        red_projections = []
        blue_projections = []  # Assume opponent continues at lower rate
        
        for t in time_points:
            red_proj = current_red_score + (optimal_rate * t * 3)
            blue_proj = current_blue_score + (0.2 * t * 3)  # Slower opponent rate
            
            red_projections.append(red_proj)
            blue_projections.append(blue_proj)
        
        fig.add_trace(
            go.Scatter(
                x=time_remaining - time_points,  # Time remaining
                y=red_projections,
                mode='lines',
                name='Red Alliance Projection',
                line=dict(color='red', width=3)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=time_remaining - time_points,
                y=blue_projections,
                mode='lines',
                name='Blue Alliance Projection',
                line=dict(color='blue', width=3)
            ),
            row=2, col=1
        )
        
        # 4. Recommendations Table
        recommendations = [
            ['Current Situation', f'Red: {current_red_score}, Blue: {current_blue_score}'],
            ['Time Remaining', f'{time_remaining} seconds'],
            ['Blocks to Win', f'{breakeven_blocks:.0f} blocks minimum'],
            ['Required Rate', f'{breakeven_blocks/(time_remaining/60):.1f} blocks/min'],
            ['Recommended Strategy', 'Aggressive scoring if behind' if current_red_score < current_blue_score else 'Balanced approach'],
            ['Priority Actions', 'Focus on parking' if time_remaining < 20 else 'Continue scoring']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'],
                           fill_color='#2E86C1',
                           font=dict(color='white', size=12),
                           align="left"),
                cells=dict(values=[[r[0] for r in recommendations], 
                                 [r[1] for r in recommendations]],
                          fill_color='#f8f9fa',
                          align="left")
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': f'Match Outcome Predictor - {time_remaining}s Remaining',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2E86C1'}
            },
            template='plotly_white',
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Scoring Rate (blocks/sec)", row=1, col=1)
        fig.update_yaxes(title_text="Win Probability (%)", row=1, col=1)
        
        fig.update_yaxes(title_text="Blocks Required", row=1, col=2)
        
        fig.update_xaxes(title_text="Time Remaining (sec)", row=2, col=1)
        fig.update_yaxes(title_text="Score", row=2, col=1)
        
        if save_html:
            fig.write_html(save_html)
        
        return fig
    
    def generate_all_visualizations(self, save_directory: str = "./visualizations/"):
        """Generate all visualization types and save as HTML files"""
        
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        print("Generating comprehensive VEX U visualizations...")
        
        # Create sample strategies
        strategies = [
            AllianceStrategy("Fast & Furious", 
                           {"long_1": 8, "long_2": 8, "center_1": 6, "center_2": 6},
                           {"long_1": 18, "long_2": 18, "center_1": 12, "center_2": 12},
                           [], [ParkingLocation.NONE, ParkingLocation.NONE]),
            AllianceStrategy("Balanced", 
                           {"long_1": 6, "long_2": 6, "center_1": 4, "center_2": 4},
                           {"long_1": 10, "long_2": 10, "center_1": 8, "center_2": 8},
                           [Zone.RED_HOME, Zone.NEUTRAL], 
                           [ParkingLocation.PLATFORM, ParkingLocation.PLATFORM]),
            AllianceStrategy("Zone Control", 
                           {"long_1": 4, "long_2": 4, "center_1": 3, "center_2": 3},
                           {"long_1": 6, "long_2": 6, "center_1": 6, "center_2": 6},
                           [Zone.RED_HOME, Zone.BLUE_HOME, Zone.NEUTRAL],
                           [ParkingLocation.PLATFORM, ParkingLocation.ALLIANCE_ZONE])
        ]
        
        # 1. Scoring Timeline
        print("1. Creating scoring timeline...")
        timeline_fig = self.create_scoring_timeline_visualization(strategies)
        timeline_fig.write_html(os.path.join(save_directory, "scoring_timeline.html"))
        
        # 2. Strategy Comparison Dashboard
        print("2. Creating strategy comparison dashboard...")
        comparison_fig = self.create_strategy_comparison_dashboard(strategies)
        comparison_fig.write_html(os.path.join(save_directory, "strategy_comparison.html"))
        
        # 3. Key Insights Dashboard
        print("3. Creating key insights dashboard...")
        insights_fig = self.create_key_insights_dashboard({})
        insights_fig.write_html(os.path.join(save_directory, "key_insights.html"))
        
        # 4. Match Outcome Predictor
        print("4. Creating match outcome predictor...")
        predictor_fig = self.create_match_outcome_predictor(120, 115, 25)
        predictor_fig.write_html(os.path.join(save_directory, "match_predictor.html"))
        
        print(f"✅ All visualizations saved to {save_directory}")
        print("Files created:")
        print("  • scoring_timeline.html - Interactive match timeline")
        print("  • strategy_comparison.html - Strategy performance dashboard") 
        print("  • key_insights.html - Key insights and recommendations")
        print("  • match_predictor.html - Win probability and outcome prediction")
        
        return True


if __name__ == "__main__":
    # Create visualizer instance
    viz = InteractiveVEXVisualizer()
    
    # Generate all visualizations
    viz.generate_all_visualizations()
    
    # Create interactive scenario explorer (uncomment to run web app)
    # print("\nStarting interactive scenario explorer on http://localhost:8050")
    # app = viz.create_interactive_scenario_explorer()
    
    print("\n🎉 Interactive VEX U visualization system ready!")
    print("Open the generated HTML files in your browser to explore the interactive dashboards.")