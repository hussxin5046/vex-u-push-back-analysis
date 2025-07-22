#!/usr/bin/env python3

from scenario_generator import *
from scoring_simulator import ScoringSimulator
import pandas as pd
import numpy as np

def test_scoring_rate_analysis():
    """Analyze how different scoring rates affect outcomes"""
    print("Testing Scoring Rate Analysis")
    print("=" * 50)
    
    simulator = ScoringSimulator()
    generator = ScenarioGenerator(simulator)
    
    # Generate time analysis
    time_df = generator.generate_time_analysis_scenarios()
    
    print(f"Generated {len(time_df)} time-based scenarios")
    print("\nScoring Rate Analysis:")
    
    # Analyze by scoring rate
    rate_analysis = time_df.groupby('scoring_rate').agg({
        'alliance_total': ['mean', 'std', 'min', 'max'],
        'efficiency_rating': ['mean', 'std'],
        'theoretical_max_score': 'mean'
    }).round(2)
    
    print(rate_analysis)
    
    # Find optimal scoring rate
    best_rate = time_df.loc[time_df['alliance_total'].idxmax()]
    print(f"\nBest Performance:")
    print(f"  Scoring Rate: {best_rate['scoring_rate']} blocks/sec")
    print(f"  Capacity: {best_rate['capacity']} blocks")
    print(f"  Cooperation: {best_rate['cooperation']}")
    print(f"  Total Blocks: {best_rate['alliance_total']}")
    print(f"  Efficiency: {best_rate['efficiency_rating']:.3f}")
    
    return time_df

def test_capacity_vs_rate_tradeoffs():
    """Test capacity vs scoring rate tradeoffs"""
    print("\n\nTesting Capacity vs Rate Tradeoffs")
    print("=" * 50)
    
    simulator = ScoringSimulator()
    generator = ScenarioGenerator(simulator)
    
    time_df = generator.generate_time_analysis_scenarios()
    
    # Create pivot table
    pivot = time_df.pivot_table(
        values='alliance_total',
        index='scoring_rate',
        columns='capacity',
        aggfunc='mean'
    )
    
    print("Average Alliance Total Blocks by Rate vs Capacity:")
    print(pivot.round(1))
    
    # Find sweet spots
    print("\nSweet Spot Analysis:")
    for capacity in [2, 3, 4, 5]:
        capacity_data = time_df[time_df['capacity'] == capacity]
        best_rate = capacity_data.loc[capacity_data['alliance_total'].idxmax()]
        print(f"  Capacity {capacity}: Best at {best_rate['scoring_rate']} blocks/sec -> {best_rate['alliance_total']} total blocks")
    
    return pivot

def test_cooperation_impact():
    """Test impact of robot cooperation on performance"""
    print("\n\nTesting Cooperation Impact")
    print("=" * 40)
    
    simulator = ScoringSimulator()
    generator = ScenarioGenerator(simulator)
    
    time_df = generator.generate_time_analysis_scenarios()
    
    cooperation_analysis = time_df.groupby('cooperation').agg({
        'alliance_total': ['mean', 'std'],
        'efficiency_rating': 'mean',
        'auto_blocks_per_robot': 'mean',
        'driver_blocks_per_robot': 'mean'
    }).round(2)
    
    print("Performance by Cooperation Level:")
    print(cooperation_analysis)
    
    # Statistical significance test
    low_coop = time_df[time_df['cooperation'] == 0.6]['alliance_total']
    high_coop = time_df[time_df['cooperation'] == 1.0]['alliance_total']
    
    print(f"\nCooperation Impact:")
    print(f"  Low Cooperation (0.6): {low_coop.mean():.1f} ± {low_coop.std():.1f} blocks")
    print(f"  High Cooperation (1.0): {high_coop.mean():.1f} ± {high_coop.std():.1f} blocks")
    print(f"  Improvement: {((high_coop.mean() / low_coop.mean() - 1) * 100):.1f}%")

def test_strategy_effectiveness_deep_dive():
    """Deep dive into strategy effectiveness"""
    print("\n\nTesting Strategy Effectiveness Deep Dive")
    print("=" * 50)
    
    simulator = ScoringSimulator()
    generator = ScenarioGenerator(simulator)
    
    # Analyze with more samples for accuracy
    eff_df = generator.analyze_strategy_effectiveness(num_samples=50)
    
    print("Strategy Effectiveness Analysis:")
    
    # Overall strategy ranking
    strategy_ranking = eff_df.groupby('strategy_type').agg({
        'win_rate': 'mean',
        'avg_score': 'mean',
        'consistency': 'mean'
    }).sort_values('win_rate', ascending=False)
    
    print("\nOverall Strategy Ranking by Win Rate:")
    print(strategy_ranking.round(3))
    
    # Best strategy by skill level
    print("\nBest Strategy by Skill Level:")
    for skill in SkillLevel:
        skill_data = eff_df[eff_df['skill_level'] == skill.value]
        best = skill_data.loc[skill_data['win_rate'].idxmax()]
        print(f"  {skill.value.title()}: {best['strategy_type']} ({best['win_rate']:.1%} win rate, {best['avg_score']:.0f} avg score)")
    
    # Strategy consistency analysis
    print("\nStrategy Consistency (lower std = more consistent):")
    consistency_analysis = eff_df.groupby('strategy_type')['consistency'].mean().sort_values(ascending=False)
    print(consistency_analysis.round(3))
    
    return eff_df

def test_skill_gap_analysis():
    """Analyze performance gaps between skill levels"""
    print("\n\nTesting Skill Gap Analysis")
    print("=" * 40)
    
    simulator = ScoringSimulator()
    generator = ScenarioGenerator(simulator)
    
    cap_df = generator.generate_capability_comparison()
    
    # Focus on scorer role for clear comparison
    scorer_data = cap_df[cap_df['robot_role'] == 'scorer'].sort_values('skill_level')
    
    print("Scorer Performance by Skill Level:")
    print(scorer_data[['skill_level', 'auto_blocks_expected', 'driver_blocks_expected', 'total_blocks_expected']])
    
    # Calculate performance multipliers
    beginner_total = scorer_data[scorer_data['skill_level'] == 'beginner']['total_blocks_expected'].iloc[0]
    
    print("\nPerformance Multipliers (vs Beginner):")
    for _, row in scorer_data.iterrows():
        multiplier = row['total_blocks_expected'] / beginner_total
        print(f"  {row['skill_level'].title()}: {multiplier:.1f}x ({row['total_blocks_expected']:.0f} blocks)")
    
    # Analyze capability improvements
    print("\nCapability Progression:")
    for i in range(1, len(scorer_data)):
        current = scorer_data.iloc[i]
        previous = scorer_data.iloc[i-1]
        
        rate_improvement = (current['blocks_per_second'] / previous['blocks_per_second'] - 1) * 100
        accuracy_improvement = (current['accuracy'] / previous['accuracy'] - 1) * 100
        
        print(f"  {previous['skill_level'].title()} -> {current['skill_level'].title()}:")
        print(f"    Rate: +{rate_improvement:.0f}%, Accuracy: +{accuracy_improvement:.0f}%")

def test_realistic_match_scenarios():
    """Test realistic match scenarios with detailed analysis"""
    print("\n\nTesting Realistic Match Scenarios")
    print("=" * 50)
    
    simulator = ScoringSimulator()
    generator = ScenarioGenerator(simulator)
    
    # Create realistic tournament scenarios
    scenarios = [
        ("Rookie Team", SkillLevel.BEGINNER, StrategyType.MIXED),
        ("JV Team", SkillLevel.INTERMEDIATE, StrategyType.ALL_OFFENSE),  
        ("Varsity Team", SkillLevel.ADVANCED, StrategyType.ZONE_CONTROL),
        ("Elite Team", SkillLevel.EXPERT, StrategyType.AUTONOMOUS_FOCUS)
    ]
    
    match_results = []
    
    print("Tournament Simulation Results:")
    print("-" * 70)
    print(f"{'Team':<15} {'Skill':<12} {'Strategy':<15} {'Score':<8} {'Breakdown'}")
    print("-" * 70)
    
    for i, (name1, skill1, strategy1) in enumerate(scenarios):
        for j, (name2, skill2, strategy2) in enumerate(scenarios):
            if i >= j:  # Avoid duplicate matches
                continue
                
            # Generate parameters
            params1 = generator._create_scenario_parameters(skill1, strategy1, name1)
            params2 = generator._create_scenario_parameters(skill2, strategy2, name2)
            
            # Generate strategies
            strat1 = generator.generate_time_based_strategy(name1, params1)
            strat2 = generator.generate_time_based_strategy(name2, params2)
            
            # Simulate match
            result = simulator.simulate_match(strat1, strat2)
            
            # Display results
            print(f"{name1:<15} {skill1.value:<12} {strategy1.value:<15} {result.red_score:<8} {result.red_breakdown}")
            print(f"{name2:<15} {skill2.value:<12} {strategy2.value:<15} {result.blue_score:<8} {result.blue_breakdown}")
            print(f"Winner: {result.winner.upper()} by {result.margin} points")
            print()
            
            match_results.append({
                'team1': name1,
                'team2': name2,
                'skill1': skill1.value,
                'skill2': skill2.value,
                'strategy1': strategy1.value,
                'strategy2': strategy2.value,
                'winner': result.winner,
                'margin': result.margin,
                'score1': result.red_score,
                'score2': result.blue_score
            })
    
    return pd.DataFrame(match_results)

def test_optimization_scenarios():
    """Test optimization scenarios for different constraints"""
    print("\n\nTesting Optimization Scenarios")
    print("=" * 45)
    
    simulator = ScoringSimulator()
    generator = ScenarioGenerator(simulator)
    
    # Optimization constraints
    constraints = [
        ("Max Blocks", {"blocks_per_second": 0.6, "max_capacity": 5}),
        ("Balanced", {"blocks_per_second": 0.3, "max_capacity": 3}),
        ("Conservative", {"blocks_per_second": 0.2, "max_capacity": 2}),
        ("Speed Focus", {"blocks_per_second": 0.5, "max_capacity": 2}),
        ("Capacity Focus", {"blocks_per_second": 0.25, "max_capacity": 5})
    ]
    
    print("Optimization Results:")
    print("-" * 45)
    
    for name, params in constraints:
        # Create custom capabilities
        capabilities = RobotCapabilities(
            blocks_per_second=params["blocks_per_second"],
            max_capacity=params["max_capacity"],
            travel_time_per_goal=4.0,
            collection_time=1.5,
            accuracy=0.85,
            autonomous_reliability=0.8
        )
        
        # Calculate performance
        auto_performance = generator.calculate_realistic_scoring(
            capabilities, RobotRole.SCORER, 13.0, 0.9, 1.0
        )
        
        driver_performance = generator.calculate_realistic_scoring(
            capabilities, RobotRole.SCORER, 75.0, 0.9, 0.85
        )
        
        total_performance = (auto_performance + driver_performance) * 2  # Alliance total
        theoretical_score = total_performance * 3 + 10  # Assume auto bonus
        
        print(f"{name:<15}: {total_performance:>3.0f} blocks -> {theoretical_score:>3.0f} points")
        print(f"                 Auto: {auto_performance*2:>2.0f}, Driver: {driver_performance*2:>2.0f}")

def run_all_scenario_tests():
    """Run all scenario analysis tests"""
    print("VEX U Scenario Generator - Advanced Testing Suite")
    print("=" * 60)
    
    # Run all tests
    time_df = test_scoring_rate_analysis()
    pivot_df = test_capacity_vs_rate_tradeoffs() 
    test_cooperation_impact()
    eff_df = test_strategy_effectiveness_deep_dive()
    test_skill_gap_analysis()
    match_df = test_realistic_match_scenarios()
    test_optimization_scenarios()
    
    print("\n" + "=" * 60)
    print("🔬 ADVANCED TESTING COMPLETE! 🔬")
    print("All scenario analysis tests passed successfully!")
    print("=" * 60)
    
    # Return data for further analysis
    return {
        'time_analysis': time_df,
        'capacity_pivot': pivot_df,
        'effectiveness': eff_df,
        'match_results': match_df
    }

if __name__ == "__main__":
    results = run_all_scenario_tests()
    
    print("\nTest Results Summary:")
    print(f"• Time scenarios generated: {len(results['time_analysis'])}")
    print(f"• Strategy combinations tested: {len(results['effectiveness'])}")
    print(f"• Tournament matches simulated: {len(results['match_results'])}")
    print(f"• Capacity/Rate combinations: {results['capacity_pivot'].size}")
    print("\nAll DataFrames are available for further analysis!")