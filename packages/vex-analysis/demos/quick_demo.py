#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from analysis.strategy_analyzer import AdvancedStrategyAnalyzer
from core.simulator import ScoringSimulator

def quick_strategy_demo():
    """Quick demonstration of strategy analyzer capabilities"""
    print("🎯 VEX U STRATEGY ANALYZER DEMONSTRATION 🎯")
    print("=" * 60)
    
    # Initialize analyzer
    simulator = ScoringSimulator()
    analyzer = AdvancedStrategyAnalyzer(simulator)
    
    # Run quick analysis (reduced simulations for demo)
    print("Running quick analysis with 100 simulations per strategy...")
    
    results = analyzer.run_complete_analysis(
        num_monte_carlo=100,
        include_coordination=False  # Just core strategies for demo
    )
    
    print("\n" + "=" * 60)
    print("📊 STRATEGY ANALYSIS RESULTS")
    print("=" * 60)
    
    # Top strategies
    print("\n🏆 TOP STRATEGIES BY WIN RATE:")
    sorted_metrics = sorted(results['metrics'], key=lambda x: x.win_rate, reverse=True)
    
    for i, metric in enumerate(sorted_metrics, 1):
        print(f"{i}. {metric.strategy_name:<20} | {metric.win_rate:>6.1%} wins | {metric.avg_score:>5.0f} avg pts | Risk/Reward: {metric.risk_reward_ratio:>4.2f}")
    
    # Strategy insights
    print("\n🔍 STRATEGY INSIGHTS:")
    
    best_strategy = sorted_metrics[0]
    print(f"\n🥇 Best Overall: {best_strategy.strategy_name}")
    print(f"   • Win Rate: {best_strategy.win_rate:.1%}")
    print(f"   • Average Score: {best_strategy.avg_score:.0f} points")
    print(f"   • Strengths: {', '.join(best_strategy.strengths[:3])}")
    
    most_consistent = max(sorted_metrics, key=lambda x: x.consistency_score)
    print(f"\n🎯 Most Consistent: {most_consistent.strategy_name}")
    print(f"   • Consistency Score: {most_consistent.consistency_score:.2f}")
    print(f"   • Score Std Dev: ±{most_consistent.score_std:.0f}")
    
    highest_scoring = max(sorted_metrics, key=lambda x: x.avg_score)
    print(f"\n📈 Highest Scoring: {highest_scoring.strategy_name}")
    print(f"   • Average Score: {highest_scoring.avg_score:.0f} points")
    
    # Component analysis
    print("\n📊 SCORING COMPONENT ANALYSIS:")
    print("-" * 40)
    print(f"{'Strategy':<20} {'Blocks':<8} {'Auto':<6} {'Zones':<7} {'Parking':<8}")
    print("-" * 40)
    
    for metric in sorted_metrics:
        blocks = metric.component_breakdown.get('blocks', 0)
        auto = metric.component_breakdown.get('autonomous', 0) 
        zones = metric.component_breakdown.get('zones', 0)
        parking = metric.component_breakdown.get('parking', 0)
        print(f"{metric.strategy_name:<20} {blocks:>6.0f}% {auto:>6.0f}% {zones:>6.0f}% {parking:>7.0f}%")
    
    # Matchup highlights
    print("\n⚔️  KEY MATCHUP INSIGHTS:")
    
    # Find most competitive matchup
    competitive_matchups = []
    for (a, b), result in results['matchup_results'].items():
        win_rate = result.a_wins / result.total_matches
        competitiveness = 1 - abs(win_rate - 0.5) * 2
        competitive_matchups.append(((a, b), competitiveness, win_rate))
    
    most_competitive = max(competitive_matchups, key=lambda x: x[1])
    (team_a, team_b), comp_score, win_rate = most_competitive
    
    print(f"\n🔥 Most Competitive Matchup:")
    print(f"   {team_a} vs {team_b}")
    print(f"   Win Rate: {win_rate:.1%} - {1-win_rate:.1%}")
    print(f"   Competitiveness: {comp_score:.2f}/1.0")
    
    # Find most dominant matchup
    least_competitive = min(competitive_matchups, key=lambda x: x[1])
    (dom_a, dom_b), _, dom_win_rate = least_competitive
    winner = dom_a if dom_win_rate > 0.5 else dom_b
    
    print(f"\n💪 Most Dominant Performance:")
    print(f"   {winner} dominates with {max(dom_win_rate, 1-dom_win_rate):.1%} win rate")
    
    # Dominant strategies
    dominant = results['dominant_strategies']
    if dominant:
        print(f"\n👑 DOMINANT STRATEGIES (>55% overall win rate):")
        for strategy, overall_win_rate in dominant[:3]:
            print(f"   • {strategy}: {overall_win_rate:.1%}")
    
    print("\n" + "=" * 60)
    print("✅ STRATEGY ANALYSIS COMPLETE!")
    print("=" * 60)
    
    # Analysis summary
    summary = results['analysis_summary']
    print(f"\n📈 Analysis Summary:")
    print(f"   • {summary['total_strategies']} strategies analyzed")
    print(f"   • {summary['total_simulations']:,} total simulations")
    print(f"   • {summary['total_matches']:,} head-to-head matches")
    print(f"   • {len(dominant) if dominant else 0} dominant strategies identified")
    
    print("\n🚀 Ready for VEX U competition strategy optimization!")
    
    return results

def main():
    """Main entry point for CLI"""
    results = quick_strategy_demo()
    
    # Show full report snippet
    print("\n" + "=" * 60)
    print("📋 SAMPLE FROM FULL ANALYSIS REPORT:")
    print("=" * 60)
    report_lines = results['report'].split('\n')
    for line in report_lines[:30]:  # Show first 30 lines
        print(line)
    print("...")
    print(f"[Full report contains {len(report_lines)} lines]")
    return results

if __name__ == "__main__":
    main()