#!/usr/bin/env python3

from scenario_generator import *
from scoring_simulator import ScoringSimulator

def test_enhanced_scenario_generator():
    simulator = ScoringSimulator()
    generator = ScenarioGenerator(simulator)
    
    print("Enhanced VEX U Push Back Scenario Generator Test")
    print("=" * 60)
    
    # Test time-based strategy generation
    print("\n1. Time-Based Strategy Generation")
    print("-" * 40)
    
    # Create a realistic advanced team scenario
    advanced_params = ScenarioParameters(
        skill_level=SkillLevel.ADVANCED,
        strategy_type=StrategyType.ALL_OFFENSE,
        robot1_role=RobotRole.SCORER,
        robot2_role=RobotRole.SCORER,
        robot1_capabilities=generator.capability_profiles[SkillLevel.ADVANCED],
        robot2_capabilities=generator.capability_profiles[SkillLevel.ADVANCED],
        field_position="center_field",
        cooperation_efficiency=0.9
    )
    
    advanced_strategy = generator.generate_time_based_strategy("Advanced Team", advanced_params)
    auto_blocks = sum(advanced_strategy.blocks_scored_auto.values())
    driver_blocks = sum(advanced_strategy.blocks_scored_driver.values())
    
    print(f"Advanced All-Offense Strategy:")
    print(f"  Auto blocks: {auto_blocks}")
    print(f"  Driver blocks: {driver_blocks}")
    print(f"  Total blocks: {auto_blocks + driver_blocks}")
    print(f"  Zones controlled: {len(advanced_strategy.zones_controlled)}")
    print(f"  Parking: {[p.value for p in advanced_strategy.robots_parked]}")
    
    # Test beginner strategy
    beginner_params = ScenarioParameters(
        skill_level=SkillLevel.BEGINNER,
        strategy_type=StrategyType.MIXED,
        robot1_role=RobotRole.SCORER,
        robot2_role=RobotRole.SUPPORT,
        robot1_capabilities=generator.capability_profiles[SkillLevel.BEGINNER],
        robot2_capabilities=generator.capability_profiles[SkillLevel.BEGINNER],
        field_position="alliance_side",
        cooperation_efficiency=0.7
    )
    
    beginner_strategy = generator.generate_time_based_strategy("Beginner Team", beginner_params)
    auto_blocks_b = sum(beginner_strategy.blocks_scored_auto.values())
    driver_blocks_b = sum(beginner_strategy.blocks_scored_driver.values())
    
    print(f"\nBeginner Mixed Strategy:")
    print(f"  Auto blocks: {auto_blocks_b}")
    print(f"  Driver blocks: {driver_blocks_b}")
    print(f"  Total blocks: {auto_blocks_b + driver_blocks_b}")
    print(f"  Zones controlled: {len(beginner_strategy.zones_controlled)}")
    print(f"  Parking: {[p.value for p in beginner_strategy.robots_parked]}")
    
    # Simulate match between them
    result = simulator.simulate_match(advanced_strategy, beginner_strategy)
    print(f"\nMatch Result: {result.winner.upper()} wins by {result.margin} points")
    print(f"Advanced Team: {result.red_score} | Beginner Team: {result.blue_score}")
    
    # Generate capability comparison
    print("\n2. Robot Capability Analysis")
    print("-" * 40)
    
    capability_df = generator.generate_capability_comparison()
    print("\nCapability Comparison (sample):")
    print(capability_df[capability_df['robot_role'] == 'scorer'][['skill_level', 'auto_blocks_expected', 'driver_blocks_expected', 'total_blocks_expected']].head(4))
    
    # Generate time analysis
    print("\n3. Time-Based Analysis")
    print("-" * 40)
    
    time_df = generator.generate_time_analysis_scenarios()
    print(f"\nGenerated {len(time_df)} time-based scenarios")
    print("Sample of different scoring rates and their outcomes:")
    sample_rates = time_df[time_df['capacity'] == 4][['scoring_rate', 'cooperation', 'alliance_total', 'efficiency_rating']].head(6)
    print(sample_rates)
    
    # Strategy effectiveness analysis
    print("\n4. Strategy Effectiveness Analysis")
    print("-" * 40)
    
    print("Analyzing strategy effectiveness (this may take a moment...)")
    effectiveness_df = generator.analyze_strategy_effectiveness(num_samples=20)
    
    print("\nStrategy Win Rates by Skill Level:")
    pivot_table = effectiveness_df.pivot_table(
        values='win_rate', 
        index='strategy_type', 
        columns='skill_level', 
        aggfunc='mean'
    )
    print(pivot_table.round(3))
    
    print("\n5. Sample Scenario Matrix (Limited)")
    print("-" * 40)
    
    # Generate a smaller scenario matrix for demonstration
    print("Note: Full scenario matrix generation takes significant time.")
    print("Generating sample scenarios for demonstration...")
    
    sample_scenarios = []
    for red_skill in [SkillLevel.BEGINNER, SkillLevel.ADVANCED]:
        for blue_skill in [SkillLevel.BEGINNER, SkillLevel.ADVANCED]:
            for red_strategy in [StrategyType.ALL_OFFENSE, StrategyType.ZONE_CONTROL]:
                for blue_strategy in [StrategyType.ALL_OFFENSE, StrategyType.ZONE_CONTROL]:
                    red_params = generator._create_scenario_parameters(red_skill, red_strategy, "Red")
                    blue_params = generator._create_scenario_parameters(blue_skill, blue_strategy, "Blue")
                    
                    red_strategy_obj = generator.generate_time_based_strategy("Red", red_params)
                    blue_strategy_obj = generator.generate_time_based_strategy("Blue", blue_params)
                    
                    result = simulator.simulate_match(red_strategy_obj, blue_strategy_obj)
                    
                    sample_scenarios.append({
                        'red_skill': red_skill.value,
                        'blue_skill': blue_skill.value,
                        'red_strategy': red_strategy.value,
                        'blue_strategy': blue_strategy.value,
                        'winner': result.winner,
                        'margin': result.margin,
                        'red_score': result.red_score,
                        'blue_score': result.blue_score
                    })
    
    sample_df = pd.DataFrame(sample_scenarios)
    print(f"\nGenerated {len(sample_df)} sample scenarios:")
    print(sample_df[['red_skill', 'blue_skill', 'red_strategy', 'blue_strategy', 'winner', 'margin']].head(8))
    
    print("\n" + "=" * 60)
    print("Enhanced scenario generator testing complete!")
    print("\nKey Features Demonstrated:")
    print("✓ Time-based realistic scoring calculations")
    print("✓ Robot capability profiles for different skill levels")
    print("✓ Strategy type templates and role assignments")
    print("✓ Comprehensive scenario matrix generation")
    print("✓ DataFrame output for analysis and visualization")
    print("✓ Strategy effectiveness analysis")
    print("\nUse the DataFrames for further analysis and visualization!")
    
    # Return DataFrames for further analysis
    return {
        'capabilities': capability_df,
        'time_analysis': time_df,
        'effectiveness': effectiveness_df,
        'sample_scenarios': sample_df
    }

if __name__ == "__main__":
    results = test_enhanced_scenario_generator()
    
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    # Time analysis summary
    time_df = results['time_analysis']
    print(f"\nTime Analysis: {len(time_df)} scenarios generated")
    print(f"Scoring rate range: {time_df['scoring_rate'].min():.1f} - {time_df['scoring_rate'].max():.1f} blocks/sec")
    print(f"Average alliance total blocks: {time_df['alliance_total'].mean():.1f}")
    print(f"Best efficiency scenario: {time_df.loc[time_df['efficiency_rating'].idxmax(), 'efficiency_rating']:.3f}")
    
    # Capabilities summary
    cap_df = results['capabilities']
    print(f"\nCapability Analysis: {len(cap_df)} combinations tested")
    scorer_caps = cap_df[cap_df['robot_role'] == 'scorer']
    print(f"Scorer performance range: {scorer_caps['total_blocks_expected'].min():.0f} - {scorer_caps['total_blocks_expected'].max():.0f} blocks")
    
    # Effectiveness summary
    eff_df = results['effectiveness']
    best_strategy = eff_df.loc[eff_df['win_rate'].idxmax()]
    print(f"\nEffectiveness Analysis: {len(eff_df)} strategy combinations")
    print(f"Best performing: {best_strategy['strategy_type']} at {best_strategy['skill_level']} level ({best_strategy['win_rate']:.1%} win rate)")
    
    print(f"\nSample scenarios: {len(results['sample_scenarios'])} generated")