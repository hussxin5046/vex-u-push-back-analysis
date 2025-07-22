#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from analysis.strategy_analyzer import StrategyMetrics, MatchupResult

class StrategyVisualizer:
    def __init__(self):
        # Set up plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
    def plot_strategy_performance(self, metrics: List[StrategyMetrics], save_path: str = None):
        """Create comprehensive strategy performance visualization"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('VEX U Strategy Performance Analysis', fontsize=16, fontweight='bold')
        
        # Prepare data
        strategy_names = [m.strategy_name for m in metrics]
        win_rates = [m.win_rate * 100 for m in metrics]
        avg_scores = [m.avg_score for m in metrics]
        risk_reward = [m.risk_reward_ratio for m in metrics]
        consistency = [m.consistency_score for m in metrics]
        
        # 1. Win Rate Comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(strategy_names, win_rates, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        ax1.set_title('Win Rate by Strategy', fontweight='bold')
        ax1.set_ylabel('Win Rate (%)')
        ax1.set_ylim(0, 105)
        
        # Add value labels on bars
        for bar, rate in zip(bars1, win_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Average Score Comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(strategy_names, avg_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        ax2.set_title('Average Score by Strategy', fontweight='bold')
        ax2.set_ylabel('Average Score (Points)')
        
        for bar, score in zip(bars2, avg_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{score:.0f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Risk/Reward Analysis
        ax3 = axes[1, 0]
        scatter = ax3.scatter(risk_reward, win_rates, s=200, alpha=0.7,
                             c=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        
        for i, name in enumerate(strategy_names):
            ax3.annotate(name, (risk_reward[i], win_rates[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax3.set_xlabel('Risk/Reward Ratio')
        ax3.set_ylabel('Win Rate (%)')
        ax3.set_title('Risk vs Reward Analysis', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Component Breakdown (Stacked Bar)
        ax4 = axes[1, 1]
        
        # Prepare component data
        components = ['blocks', 'autonomous', 'zones', 'parking']
        component_data = {comp: [] for comp in components}
        
        for metric in metrics:
            for comp in components:
                component_data[comp].append(metric.component_breakdown.get(comp, 0))
        
        # Create stacked bar chart
        bottom = np.zeros(len(strategy_names))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, (comp, values) in enumerate(component_data.items()):
            ax4.bar(strategy_names, values, bottom=bottom, label=comp.title(), 
                   color=colors[i], alpha=0.8)
            bottom += values
        
        ax4.set_title('Score Component Breakdown', fontweight='bold')
        ax4.set_ylabel('Percentage of Total Score')
        ax4.legend()
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_matchup_heatmap(self, matchup_df: pd.DataFrame, save_path: str = None):
        """Create matchup matrix heatmap"""
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        mask = np.triu(np.ones_like(matchup_df.values, dtype=bool), k=1)
        
        heatmap = sns.heatmap(matchup_df, 
                             annot=True, 
                             fmt='.2f',
                             cmap='RdYlBu_r',
                             center=0.5,
                             square=True,
                             linewidths=0.5,
                             cbar_kws={"shrink": 0.8, "label": "Win Rate"},
                             mask=mask,
                             ax=ax)
        
        ax.set_title('Strategy Matchup Matrix (Win Rates)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Opponent Strategy', fontweight='bold')
        ax.set_ylabel('Your Strategy', fontweight='bold')
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_strategy_radar(self, metrics: List[StrategyMetrics], save_path: str = None):
        """Create radar chart comparing strategies across multiple dimensions"""
        
        # Define categories for radar chart
        categories = ['Win Rate', 'Avg Score', 'Consistency', 'Block Focus', 'Zone Control', 'Parking Efficiency']
        N = len(categories)
        
        # Set up radar chart
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for i, metric in enumerate(metrics[:5]):  # Limit to 5 strategies for clarity
            # Normalize values to 0-1 scale
            values = []
            values.append(metric.win_rate)  # Win Rate (already 0-1)
            values.append(min(metric.avg_score / 300, 1.0))  # Avg Score (normalize to max 300)
            values.append(min(metric.consistency_score / 5, 1.0))  # Consistency (cap at 5)
            values.append(metric.component_breakdown.get('blocks', 0) / 100)  # Block Focus (0-100%)
            values.append(metric.component_breakdown.get('zones', 0) / 30)  # Zone Control (normalize to max 30%)
            values.append(metric.component_breakdown.get('parking', 0) / 30)  # Parking (normalize to max 30%)
            
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=metric.strategy_name, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Customize the chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        plt.title('Strategy Performance Radar Chart', size=16, fontweight='bold', pad=30)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_coordination_comparison(self, coordination_results: Dict[str, List[StrategyMetrics]], save_path: str = None):
        """Compare different coordination strategies"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Two-Robot Coordination Strategy Analysis', fontsize=16, fontweight='bold')
        
        # Prepare data
        coord_names = list(coordination_results.keys())
        coord_data = {
            'win_rates': [],
            'avg_scores': [],
            'consistency': [],
            'risk_reward': []
        }
        
        for coord_name in coord_names:
            metrics_list = coordination_results[coord_name]
            coord_data['win_rates'].append([m.win_rate * 100 for m in metrics_list])
            coord_data['avg_scores'].append([m.avg_score for m in metrics_list])
            coord_data['consistency'].append([m.consistency_score for m in metrics_list])
            coord_data['risk_reward'].append([m.risk_reward_ratio for m in metrics_list])
        
        # 1. Win Rate Box Plot
        ax1 = axes[0, 0]
        box_data = [data for data in coord_data['win_rates']]
        box1 = ax1.boxplot(box_data, labels=coord_names, patch_artist=True)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        for patch, color in zip(box1['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_title('Win Rate Distribution by Coordination Type', fontweight='bold')
        ax1.set_ylabel('Win Rate (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Average Score Box Plot
        ax2 = axes[0, 1]
        box_data = [data for data in coord_data['avg_scores']]
        box2 = ax2.boxplot(box_data, labels=coord_names, patch_artist=True)
        
        for patch, color in zip(box2['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_title('Score Distribution by Coordination Type', fontweight='bold')
        ax2.set_ylabel('Average Score (Points)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Consistency Comparison
        ax3 = axes[1, 0]
        coord_consistency_avg = [np.mean(data) for data in coord_data['consistency']]
        bars3 = ax3.bar(coord_names, coord_consistency_avg, color=colors, alpha=0.8)
        
        for bar, avg in zip(bars3, coord_consistency_avg):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{avg:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax3.set_title('Average Consistency by Coordination Type', fontweight='bold')
        ax3.set_ylabel('Consistency Score')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Risk/Reward Scatter
        ax4 = axes[1, 1]
        
        for i, coord_name in enumerate(coord_names):
            risk_rewards = coord_data['risk_reward'][i]
            win_rates = coord_data['win_rates'][i]
            ax4.scatter(risk_rewards, win_rates, label=coord_name, 
                       color=colors[i], alpha=0.7, s=100)
        
        ax4.set_xlabel('Risk/Reward Ratio')
        ax4.set_ylabel('Win Rate (%)')
        ax4.set_title('Risk vs Reward by Coordination Type', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_strategy_summary_dashboard(self, analysis_results: Dict, save_path: str = None):
        """Create comprehensive dashboard summarizing all analysis"""
        
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('VEX U Push Back - Complete Strategy Analysis Dashboard', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        metrics = analysis_results['metrics']
        matchup_df = analysis_results['matchup_matrix']
        
        # 1. Top Strategies (Top Left)
        ax1 = fig.add_subplot(gs[0, :2])
        top_strategies = sorted(metrics, key=lambda x: x.win_rate, reverse=True)[:5]
        names = [s.strategy_name for s in top_strategies]
        scores = [s.avg_score for s in top_strategies]
        win_rates = [s.win_rate * 100 for s in top_strategies]
        
        x = np.arange(len(names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, scores, width, label='Avg Score', alpha=0.8, color='#4ECDC4')
        bars2 = ax1.bar(x + width/2, [s*3 for s in win_rates], width, label='Win Rate x3', alpha=0.8, color='#FF6B6B')
        
        ax1.set_title('Top 5 Strategies Performance', fontweight='bold')
        ax1.set_ylabel('Points / (Win Rate × 3)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.legend()
        
        # 2. Matchup Heatmap (Top Right)
        ax2 = fig.add_subplot(gs[0, 2:])
        
        # Simplified heatmap for dashboard
        small_df = matchup_df.iloc[:5, :5]  # Top 5 strategies only
        sns.heatmap(small_df, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                   center=0.5, square=True, cbar_kws={'label': 'Win Rate'}, ax=ax2)
        ax2.set_title('Strategy Matchup Matrix (Top 5)', fontweight='bold')
        
        # 3. Component Analysis (Middle Left)
        ax3 = fig.add_subplot(gs[1, :2])
        
        components = ['blocks', 'autonomous', 'zones', 'parking']
        component_means = {comp: np.mean([m.component_breakdown.get(comp, 0) for m in metrics]) 
                          for comp in components}
        
        wedges, texts, autotexts = ax3.pie(component_means.values(), 
                                          labels=components, 
                                          autopct='%1.1f%%',
                                          colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax3.set_title('Average Score Component Distribution', fontweight='bold')
        
        # 4. Risk vs Reward (Middle Right)
        ax4 = fig.add_subplot(gs[1, 2:])
        
        risk_rewards = [m.risk_reward_ratio for m in metrics]
        win_rates_scatter = [m.win_rate * 100 for m in metrics]
        colors_scatter = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(metrics)]
        
        scatter = ax4.scatter(risk_rewards, win_rates_scatter, 
                             c=colors_scatter, s=200, alpha=0.7)
        
        for i, metric in enumerate(metrics):
            ax4.annotate(metric.strategy_name, (risk_rewards[i], win_rates_scatter[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax4.set_xlabel('Risk/Reward Ratio')
        ax4.set_ylabel('Win Rate (%)')
        ax4.set_title('Risk vs Reward Analysis', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance Statistics (Bottom)
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Create statistics table
        stats_data = []
        for metric in top_strategies:
            stats_data.append([
                metric.strategy_name,
                f"{metric.win_rate:.1%}",
                f"{metric.avg_score:.0f}",
                f"{metric.risk_reward_ratio:.1f}",
                f"{metric.consistency_score:.2f}",
                ", ".join(metric.strengths[:2])
            ])
        
        table = ax5.table(cellText=stats_data,
                         colLabels=['Strategy', 'Win Rate', 'Avg Score', 'Risk/Reward', 'Consistency', 'Key Strengths'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0.2, 1, 0.6])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(stats_data) + 1):
            for j in range(6):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4ECDC4')
                    cell.set_text_props(weight='bold')
                else:
                    cell.set_facecolor('#f0f0f0')
        
        ax5.set_title('Strategy Performance Summary', fontweight='bold', pad=50)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    from analysis.strategy_analyzer import AdvancedStrategyAnalyzer
    from core.simulator import ScoringSimulator
    
    print("Creating sample visualizations...")
    
    # Initialize analyzer and run quick analysis
    simulator = ScoringSimulator()
    analyzer = AdvancedStrategyAnalyzer(simulator)
    
    results = analyzer.run_complete_analysis(
        num_monte_carlo=50,  # Reduced for quick demo
        include_coordination=False
    )
    
    # Create visualizer
    viz = StrategyVisualizer()
    
    print("Generating visualizations...")
    
    # 1. Strategy Performance Chart
    viz.plot_strategy_performance(results['metrics'])
    
    # 2. Matchup Heatmap
    viz.plot_matchup_heatmap(results['matchup_matrix'])
    
    # 3. Radar Chart
    viz.plot_strategy_radar(results['metrics'])
    
    # 4. Complete Dashboard
    viz.create_strategy_summary_dashboard(results)
    
    print("✅ Visualization demo complete!")