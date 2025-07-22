#!/usr/bin/env python3

"""
VEX U Push Back Statistical Analysis Demonstration
Showcase advanced statistical insights and winning edges analysis.
"""

import os
import sys
from pathlib import Path
from typing import List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.simulator import ScoringSimulator, AllianceStrategy, Zone, ParkingLocation
from analysis.strategy_analyzer import AdvancedStrategyAnalyzer
from analysis.statistical_analyzer import StatisticalAnalyzer


def create_comprehensive_strategy_set() -> List[AllianceStrategy]:
    """Create diverse set of strategies for statistical analysis"""
    
    strategies = [
        # High-performance strategies
        AllianceStrategy("Fast & Furious", 
                        {"long_1": 8, "long_2": 8, "center_1": 6, "center_2": 6},
                        {"long_1": 18, "long_2": 18, "center_1": 12, "center_2": 12},
                        [], [ParkingLocation.NONE, ParkingLocation.NONE]),
        
        AllianceStrategy("Balanced Elite", 
                        {"long_1": 6, "long_2": 6, "center_1": 4, "center_2": 4},
                        {"long_1": 10, "long_2": 10, "center_1": 8, "center_2": 8},
                        [Zone.RED_HOME, Zone.NEUTRAL], 
                        [ParkingLocation.PLATFORM, ParkingLocation.PLATFORM]),
        
        # Conservative strategies
        AllianceStrategy("Zone Control", 
                        {"long_1": 4, "long_2": 4, "center_1": 3, "center_2": 3},
                        {"long_1": 6, "long_2": 6, "center_1": 6, "center_2": 6},
                        [Zone.RED_HOME, Zone.BLUE_HOME, Zone.NEUTRAL],
                        [ParkingLocation.PLATFORM, ParkingLocation.ALLIANCE_ZONE]),
        
        AllianceStrategy("Conservative", 
                        {"long_1": 3, "long_2": 3, "center_1": 2, "center_2": 2},
                        {"long_1": 5, "long_2": 5, "center_1": 4, "center_2": 4},
                        [Zone.RED_HOME, Zone.NEUTRAL],
                        [ParkingLocation.PLATFORM, ParkingLocation.PLATFORM]),
        
        # Specialized strategies
        AllianceStrategy("Auto Specialist", 
                        {"long_1": 10, "long_2": 8, "center_1": 6, "center_2": 4},
                        {"long_1": 5, "long_2": 7, "center_1": 5, "center_2": 3},
                        [Zone.RED_HOME],
                        [ParkingLocation.ALLIANCE_ZONE, ParkingLocation.PLATFORM]),
        
        AllianceStrategy("Endgame Focus", 
                        {"long_1": 4, "long_2": 4, "center_1": 2, "center_2": 2},
                        {"long_1": 8, "long_2": 8, "center_1": 6, "center_2": 6},
                        [Zone.RED_HOME, Zone.NEUTRAL],
                        [ParkingLocation.PLATFORM, ParkingLocation.PLATFORM]),
        
        # High-variance strategies
        AllianceStrategy("High Risk/Reward", 
                        {"long_1": 12, "long_2": 10, "center_1": 2, "center_2": 2},
                        {"long_1": 20, "long_2": 15, "center_1": 3, "center_2": 2},
                        [],
                        [ParkingLocation.NONE, ParkingLocation.NONE]),
        
        AllianceStrategy("Minimal Viable", 
                        {"long_1": 2, "long_2": 2, "center_1": 2, "center_2": 2},
                        {"long_1": 4, "long_2": 4, "center_1": 4, "center_2": 4},
                        [Zone.RED_HOME],
                        [ParkingLocation.PLATFORM, ParkingLocation.ALLIANCE_ZONE])
    ]
    
    return strategies


def run_comprehensive_statistical_analysis():
    """Run comprehensive statistical analysis demonstration"""
    
    print("🎯" * 25)
    print("VEX U PUSH BACK - STATISTICAL ANALYSIS DEMONSTRATION")
    print("Finding Winning Edges Through Advanced Statistics")
    print("🎯" * 25)
    
    # Initialize system
    print("\n📝 Initializing Statistical Analysis System...")
    simulator = ScoringSimulator()
    stat_analyzer = StatisticalAnalyzer(simulator)
    
    # Create comprehensive strategy set
    strategies = create_comprehensive_strategy_set()
    print(f"✅ Created {len(strategies)} diverse strategies for analysis")
    
    # 1. SENSITIVITY ANALYSIS
    print("\n" + "="*60)
    print("1. SENSITIVITY ANALYSIS - What Factors Drive Winning?")
    print("="*60)
    
    print("Analyzing which factors have the greatest impact on winning...")
    sensitivity_results = stat_analyzer.perform_sensitivity_analysis(
        strategies[:5],  # Use subset for faster demo
        factors=['total_blocks', 'autonomous_blocks', 'zone_control', 'parking_points'],
        num_simulations=300
    )
    
    print(f"\n🔍 SENSITIVITY ANALYSIS RESULTS:")
    print("-" * 40)
    
    # Sort by impact score
    sorted_factors = sorted(sensitivity_results.items(), 
                          key=lambda x: x[1].impact_score, reverse=True)
    
    for factor_name, result in sorted_factors:
        significance = "✅ SIGNIFICANT" if result.statistical_significance < 0.05 else "⚠️  Not Significant"
        print(f"\n{factor_name.upper().replace('_', ' ')}:")
        print(f"  Impact Score: {result.impact_score:.2f}% (win rate variation)")
        print(f"  Correlation: {result.correlation_strength:.3f}")
        print(f"  Statistical Significance: {significance}")
        print(f"  Optimal Range: {result.optimal_range[0]:.2f} - {result.optimal_range[1]:.2f}")
    
    # 2. MINIMUM VIABLE STRATEGIES
    print("\n" + "="*60)
    print("2. MINIMUM VIABLE STRATEGIES - Lowest Requirements to Win")
    print("="*60)
    
    print("Finding minimum requirements for 50%+ win rate...")
    minimum_viable = stat_analyzer.find_minimum_viable_strategies(
        target_win_rate=0.5,
        num_simulations=400
    )
    
    print(f"\n🎯 MINIMUM VIABLE STRATEGY ANALYSIS:")
    print("-" * 40)
    
    if minimum_viable:
        most_efficient = min(minimum_viable, key=lambda x: x.minimum_blocks_total)
        print(f"\n🏆 MOST EFFICIENT VIABLE STRATEGY:")
        print(f"  Name: {most_efficient.strategy_name}")
        print(f"  Minimum Total Blocks: {most_efficient.minimum_blocks_total}")
        print(f"  Minimum Autonomous: {most_efficient.minimum_auto_blocks} blocks")
        print(f"  Required Zones: {len(most_efficient.required_zones)}")
        print(f"  Parking Required: {len(most_efficient.parking_requirement)} robots")
        print(f"  Win Rate at Minimum: {most_efficient.win_rate_at_minimum:.1%}")
        print(f"  Minimum Scoring Rate: {most_efficient.minimum_scoring_rate:.3f} blocks/sec")
        
        print(f"\n📊 ALL VIABLE STRATEGIES:")
        for mvs in minimum_viable:
            print(f"  • {mvs.strategy_name}: {mvs.minimum_blocks_total} blocks total, {mvs.win_rate_at_minimum:.1%} win rate")
    else:
        print("No minimum viable strategies found at current parameters")
    
    # 3. VARIANCE AND RELIABILITY
    print("\n" + "="*60)
    print("3. VARIANCE & RELIABILITY - Risk Assessment")
    print("="*60)
    
    print("Analyzing strategy consistency and reliability...")
    variance_analyses = stat_analyzer.analyze_variance_and_reliability(
        strategies[:6], num_simulations=500
    )
    
    print(f"\n📊 STRATEGY RISK ASSESSMENT:")
    print("-" * 40)
    
    print(f"{'Rank':<4} {'Strategy':<20} {'Risk Level':<12} {'CV':<8} {'Mean±Std':<15}")
    print("-" * 65)
    
    for analysis in variance_analyses:
        cv_formatted = f"{analysis.coefficient_variation:.3f}"
        score_range = f"{analysis.mean_score:.0f}±{analysis.standard_deviation:.0f}"
        
        print(f"{analysis.consistency_rank:<4} {analysis.strategy_name:<20} "
              f"{analysis.risk_category:<12} {cv_formatted:<8} {score_range:<15}")
    
    # Find best strategies by category
    most_consistent = min(variance_analyses, key=lambda x: x.coefficient_variation)
    highest_scoring = max(variance_analyses, key=lambda x: x.mean_score)
    
    print(f"\n🎯 KEY FINDINGS:")
    print(f"  Most Consistent: {most_consistent.strategy_name} (CV: {most_consistent.coefficient_variation:.3f})")
    print(f"  Highest Scoring: {highest_scoring.strategy_name} ({highest_scoring.mean_score:.0f} avg pts)")
    print(f"  Safest for Qualification: {most_consistent.strategy_name}")
    
    # 4. CONFIDENCE INTERVALS
    print("\n" + "="*60)
    print("4. CONFIDENCE INTERVALS - Statistical Precision")
    print("="*60)
    
    print("Creating confidence intervals for strategy performance...")
    confidence_intervals = stat_analyzer.create_confidence_intervals(
        strategies[:5], num_simulations=500
    )
    
    print(f"\n📈 CONFIDENCE INTERVALS (95% Confidence):")
    print("-" * 50)
    
    for ci in confidence_intervals:
        margin = ci.margin_of_error_95
        print(f"\n{ci.strategy_name}:")
        print(f"  Mean Score: {ci.mean_score:.0f} points")
        print(f"  95% Confidence Interval: ({ci.confidence_95_lower:.0f}, {ci.confidence_95_upper:.0f})")
        print(f"  Margin of Error: ±{margin:.0f} points")
        print(f"  Distribution: {ci.distribution_type}")
        print(f"  Sample Size: {ci.sample_size:,} simulations")
    
    # 5. CORRELATION ANALYSIS
    print("\n" + "="*60)
    print("5. CORRELATION ANALYSIS - Hidden Patterns")
    print("="*60)
    
    print("Discovering correlations between factors and success...")
    correlation_insights = stat_analyzer.perform_correlation_analysis(
        strategies[:6], num_simulations=300
    )
    
    print(f"\n🔗 CORRELATION INSIGHTS:")
    print("-" * 40)
    
    # Show significant correlations
    significant_correlations = [ci for ci in correlation_insights 
                               if ci.significance in ["Highly Significant", "Significant"]]
    
    if significant_correlations:
        print(f"\n✅ STATISTICALLY SIGNIFICANT CORRELATIONS:")
        for insight in significant_correlations[:5]:  # Top 5
            direction = "↗️" if insight.correlation_coefficient > 0 else "↘️"
            print(f"\n{direction} {insight.factor_a.upper().replace('_', ' ')} ↔ {insight.factor_b.upper().replace('_', ' ')}")
            print(f"    Correlation: {insight.correlation_coefficient:.3f}")
            print(f"    Strength: {insight.strength_category}")
            print(f"    Significance: {insight.significance}")
            print(f"    Insight: {insight.interpretation}")
    
    # Show strongest correlations regardless of significance
    strongest_correlations = sorted(correlation_insights, 
                                  key=lambda x: abs(x.correlation_coefficient), reverse=True)[:3]
    
    print(f"\n🔥 STRONGEST CORRELATIONS:")
    for insight in strongest_correlations:
        print(f"  • {insight.factor_a} ↔ {insight.factor_b}: r = {insight.correlation_coefficient:.3f}")
    
    # 6. GENERATE COMPREHENSIVE VISUALIZATIONS
    print("\n" + "="*60)
    print("6. STATISTICAL VISUALIZATIONS")
    print("="*60)
    
    print("Generating comprehensive statistical visualizations...")
    stat_analyzer.generate_statistical_visualizations(
        sensitivity_results,
        variance_analyses,
        confidence_intervals,
        correlation_insights,
        save_dir="./statistical_analysis_demo/"
    )
    
    # 7. COMPREHENSIVE REPORT
    print("\n" + "="*60)
    print("7. COMPREHENSIVE STATISTICAL REPORT")
    print("="*60)
    
    print("Generating detailed statistical analysis report...")
    report = stat_analyzer.create_comprehensive_statistical_report(
        sensitivity_results,
        minimum_viable,
        variance_analyses,
        confidence_intervals,
        correlation_insights
    )
    
    # Save comprehensive report
    os.makedirs("./statistical_analysis_demo/", exist_ok=True)
    with open("./statistical_analysis_demo/comprehensive_statistical_report.txt", "w") as f:
        f.write(report)
    
    # 8. STRATEGIC INSIGHTS SUMMARY
    print("\n" + "🏆"*60)
    print("KEY STRATEGIC INSIGHTS & WINNING EDGES")
    print("🏆"*60)
    
    insights = []
    
    # From sensitivity analysis
    if sensitivity_results:
        most_impactful = max(sensitivity_results.values(), key=lambda x: x.impact_score)
        insights.append(f"🎯 Focus on {most_impactful.factor_name.replace('_', ' ')} - highest impact factor")
        insights.append(f"   Impact: {most_impactful.impact_score:.1f}% variation in win rate")
    
    # From minimum viable
    if minimum_viable:
        most_efficient = min(minimum_viable, key=lambda x: x.minimum_blocks_total)
        insights.append(f"💡 Minimum competitive threshold: {most_efficient.minimum_blocks_total} total blocks")
        insights.append(f"   Teams below this threshold have <50% win rate")
    
    # From variance analysis
    if variance_analyses:
        most_consistent = min(variance_analyses, key=lambda x: x.coefficient_variation)
        highest_risk = max(variance_analyses, key=lambda x: x.coefficient_variation)
        insights.append(f"🛡️  For qualification rounds: Use {most_consistent.strategy_name} (most consistent)")
        insights.append(f"⚡ For elimination rounds: Consider {highest_risk.strategy_name} (high risk/reward)")
    
    # From correlation analysis
    if significant_correlations:
        top_predictor = significant_correlations[0]
        insights.append(f"📈 Best success predictor: {top_predictor.factor_a.replace('_', ' ')}")
        insights.append(f"   Correlation with winning: {top_predictor.correlation_coefficient:.3f}")
    
    print("\n📋 ACTIONABLE INSIGHTS:")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    # Performance metrics summary
    print(f"\n📊 ANALYSIS PERFORMANCE:")
    total_simulations = (
        len(strategies[:5]) * 300 * 4 +  # Sensitivity analysis
        len(minimum_viable) * 400 +      # Minimum viable
        len(strategies[:6]) * 500 +      # Variance analysis  
        len(strategies[:5]) * 500 +      # Confidence intervals
        len(strategies[:6]) * 300        # Correlation analysis
    )
    
    print(f"  • Total Simulations: {total_simulations:,}")
    print(f"  • Strategies Analyzed: {len(strategies)}")
    print(f"  • Statistical Tests: {len(sensitivity_results) + len(correlation_insights)}")
    print(f"  • Visualizations Created: 4 charts")
    print(f"  • Report Length: {len(report.split())} words")
    
    print(f"\n🎉 STATISTICAL ANALYSIS COMPLETE!")
    print(f"📁 Results saved to: ./statistical_analysis_demo/")
    print(f"📊 Visualizations: Check PNG files for charts")
    print(f"📝 Full Report: comprehensive_statistical_report.txt")
    
    return {
        'sensitivity_results': sensitivity_results,
        'minimum_viable': minimum_viable,
        'variance_analyses': variance_analyses,
        'confidence_intervals': confidence_intervals,
        'correlation_insights': correlation_insights,
        'total_simulations': total_simulations
    }


def main():
    """Main entry point for CLI"""
    try:
        results = run_comprehensive_statistical_analysis()
        
        print(f"\n✅ Statistical analysis demonstration completed successfully!")
        print(f"   Key insights discovered and documented")
        print(f"   Ready for competitive strategy optimization")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Statistical analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()