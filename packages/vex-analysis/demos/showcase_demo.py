#!/usr/bin/env python3

from scenario_generator import *
from scoring_simulator import ScoringSimulator
import pandas as pd

def showcase_enhanced_generator():
    """Showcase the complete enhanced scenario generator"""
    print("🤖 VEX U PUSH BACK ENHANCED SCENARIO GENERATOR 🤖")
    print("=" * 60)
    print("Complete Analysis & Strategy Optimization Toolkit")
    print("=" * 60)
    
    simulator = ScoringSimulator()
    generator = ScenarioGenerator(simulator)
    
    # 1. Capability Analysis
    print("\n📊 1. ROBOT CAPABILITY ANALYSIS")
    print("-" * 40)
    cap_df = generator.generate_capability_comparison()
    scorer_summary = cap_df[cap_df['robot_role'] == 'scorer'][['skill_level', 'total_blocks_expected']].sort_values('total_blocks_expected')
    print("Scorer Performance by Skill Level:")
    for _, row in scorer_summary.iterrows():
        print(f"  {row['skill_level'].title():<12}: {row['total_blocks_expected']:>2.0f} blocks/match")
    
    # 2. Time-Based Analysis
    print("\n⏱️  2. TIME-BASED PERFORMANCE ANALYSIS") 
    print("-" * 40)
    time_df = generator.generate_time_analysis_scenarios()
    
    # Show optimal configurations
    best_efficiency = time_df.loc[time_df['efficiency_rating'].idxmax()]
    best_total = time_df.loc[time_df['alliance_total'].idxmax()]
    
    print(f"Most Efficient Configuration:")
    print(f"  Rate: {best_efficiency['scoring_rate']} blocks/sec")
    print(f"  Capacity: {best_efficiency['capacity']} blocks")
    print(f"  Efficiency: {best_efficiency['efficiency_rating']:.3f}")
    print(f"  Total Output: {best_efficiency['alliance_total']} blocks")
    
    print(f"\\nHighest Output Configuration:")
    print(f"  Rate: {best_total['scoring_rate']} blocks/sec")
    print(f"  Capacity: {best_total['capacity']} blocks") 
    print(f"  Total Output: {best_total['alliance_total']} blocks")
    
    # 3. Strategy Effectiveness
    print("\n🎯 3. STRATEGY EFFECTIVENESS RANKINGS")
    print("-" * 40)
    eff_df = generator.analyze_strategy_effectiveness(num_samples=25)
    
    strategy_ranking = eff_df.groupby('strategy_type').agg({
        'win_rate': 'mean',
        'avg_score': 'mean'
    }).sort_values('win_rate', ascending=False)
    
    print("Overall Strategy Performance:")
    rank = 1
    for strategy, data in strategy_ranking.iterrows():
        print(f"  {rank}. {strategy.replace('_', ' ').title():<18}: {data['win_rate']:.1%} wins, {data['avg_score']:.0f} avg points")
        rank += 1
    
    # 4. Head-to-Head Tournament
    print("\n🏆 4. TOURNAMENT SIMULATION")
    print("-" * 40)
    
    teams = [
        ("World Champions", SkillLevel.EXPERT, StrategyType.AUTONOMOUS_FOCUS),
        ("Regional Winners", SkillLevel.ADVANCED, StrategyType.ALL_OFFENSE),
        ("State Qualifiers", SkillLevel.INTERMEDIATE, StrategyType.MIXED),
        ("Local Competition", SkillLevel.BEGINNER, StrategyType.ZONE_CONTROL)
    ]
    
    tournament_results = []
    
    for i, (name1, skill1, strat1) in enumerate(teams):
        for j, (name2, skill2, strat2) in enumerate(teams[i+1:], i+1):
            # Generate teams
            params1 = generator._create_scenario_parameters(skill1, strat1, name1)
            params2 = generator._create_scenario_parameters(skill2, strat2, name2)
            
            team1 = generator.generate_time_based_strategy(name1, params1)
            team2 = generator.generate_time_based_strategy(name2, params2)
            
            # Simulate match
            result = simulator.simulate_match(team1, team2)
            
            winner_name = name1 if result.winner == "red" else name2
            loser_name = name2 if result.winner == "red" else name1
            
            print(f"{winner_name} defeats {loser_name}: {max(result.red_score, result.blue_score)}-{min(result.red_score, result.blue_score)}")
            
            tournament_results.append({
                'match': f"{name1} vs {name2}",
                'winner': winner_name,
                'margin': result.margin,
                'score1': result.red_score,
                'score2': result.blue_score
            })
    
    # 5. Optimization Recommendations
    print("\n💡 5. OPTIMIZATION RECOMMENDATIONS")
    print("-" * 40)
    
    print("For Different Skill Levels:")
    
    for skill in SkillLevel:
        skill_data = eff_df[eff_df['skill_level'] == skill.value]
        best_strategy = skill_data.loc[skill_data['win_rate'].idxmax()]
        
        print(f"  {skill.value.title():<12}: Use {best_strategy['strategy_type'].replace('_', ' ').title()}")
        print(f"                Expected: {best_strategy['win_rate']:.0%} win rate, {best_strategy['avg_score']:.0f} points")
    
    # 6. Key Insights
    print("\n🔍 6. KEY STRATEGIC INSIGHTS")
    print("-" * 40)
    
    # Cooperation impact
    cooperation_impact = time_df.groupby('cooperation')['alliance_total'].mean()
    coop_improvement = (cooperation_impact.max() / cooperation_impact.min() - 1) * 100
    
    print(f"• Robot cooperation improves performance by {coop_improvement:.0f}%")
    
    # Skill level impact
    beginner_avg = eff_df[eff_df['skill_level'] == 'beginner']['avg_score'].mean()
    expert_avg = eff_df[eff_df['skill_level'] == 'expert']['avg_score'].mean()
    skill_multiplier = expert_avg / beginner_avg if beginner_avg > 0 else 0
    
    print(f"• Expert teams score {skill_multiplier:.1f}x more than beginners")
    
    # Strategy consistency
    most_consistent = eff_df.loc[eff_df['consistency'].idxmax()]
    print(f"• Most consistent strategy: {most_consistent['strategy_type'].replace('_', ' ').title()}")
    
    # Autonomous importance
    auto_focus_performance = eff_df[eff_df['strategy_type'] == 'autonomous_focus']['win_rate'].mean()
    print(f"• Autonomous focus achieves {auto_focus_performance:.0%} average win rate")
    
    print("\n" + "=" * 60)
    print("📈 PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"✅ Generated {len(time_df)} time-based scenarios")
    print(f"✅ Analyzed {len(eff_df)} strategy combinations")
    print(f"✅ Tested {len(cap_df)} capability profiles")  
    print(f"✅ Simulated {len(tournament_results)} tournament matches")
    print(f"✅ 4 skill levels from Beginner to Expert")
    print(f"✅ 5 strategy types with distinct behaviors")
    print(f"✅ Realistic time constraints and robot physics")
    print(f"✅ Complete DataFrame outputs for analysis")
    
    print("\n🎉 ANALYSIS COMPLETE - Ready for Competition! 🎉")
    
    return {
        'capabilities': cap_df,
        'time_scenarios': time_df,
        'effectiveness': eff_df,
        'tournament': tournament_results
    }

if __name__ == "__main__":
    results = showcase_enhanced_generator()
    
    print("\\n" + "=" * 60)
    print("All results available as DataFrames for further analysis!")
    print("Perfect for strategy optimization and match preparation.")
    print("=" * 60)