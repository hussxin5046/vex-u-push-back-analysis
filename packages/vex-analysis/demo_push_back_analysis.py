#!/usr/bin/env python3
"""
Push Back Strategic Analysis Demo
Showcases the new Push Back-specific strategic analysis capabilities
"""

from vex_analysis.analysis.push_back_strategy_analyzer import (
    PushBackStrategyAnalyzer, PushBackArchetype, PushBackMatchState, 
    PushBackRobotSpecs, AutonomousStrategy, GoalPriority, ParkingTiming
)

def demo_individual_strategic_decisions():
    """Demonstrate each of the 5 key strategic decisions individually"""
    
    analyzer = PushBackStrategyAnalyzer()
    
    print("🎯 PUSH BACK STRATEGIC DECISIONS DEMO")
    print("=" * 60)
    
    # Robot specifications for demo
    robots = [
        PushBackRobotSpecs(1.6, 1.8, 3, 1.3, 0.82, "24_inch"),
        PushBackRobotSpecs(1.8, 2.0, 4, 1.1, 0.85, "15_inch")
    ]
    
    print("\n🤖 ROBOT SPECIFICATIONS:")
    for i, robot in enumerate(robots, 1):
        print(f"  Robot {i}: {robot.size_class}, Speed: {robot.speed}m/s, Capacity: {robot.block_capacity}")
    
    # 1. Block Flow Optimization
    print(f"\n{'='*20} 1. BLOCK FLOW OPTIMIZATION {'='*20}")
    block_flow = analyzer.analyze_block_flow_optimization(robots)
    
    print(f"📊 OPTIMAL BLOCK DISTRIBUTION:")
    for goal, blocks in block_flow.optimal_distribution.items():
        print(f"  • {goal.upper()}: {blocks} blocks")
    
    print(f"\n📈 PERFORMANCE METRICS:")
    print(f"  • Total Expected Points: {block_flow.total_expected_points}")
    print(f"  • Block Points: {block_flow.expected_block_points}")
    print(f"  • Control Points: {block_flow.expected_control_points}")
    print(f"  • Control Efficiency: {block_flow.control_efficiency:.2f} pts/block")
    print(f"  • Flow Rate: {block_flow.flow_rate:.1f} blocks/min")
    print(f"  • Risk Level: {block_flow.risk_level:.1%}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in block_flow.recommendations:
        print(f"  • {rec}")
    
    # 2. Autonomous Strategy Decision
    print(f"\n{'='*20} 2. AUTONOMOUS STRATEGY DECISION {'='*20}")
    auto_decision = analyzer.analyze_autonomous_strategy_decision(robots)
    
    print(f"🏆 RECOMMENDED STRATEGY: {auto_decision.recommended_strategy.value}")
    print(f"\n📊 PROBABILITIES & SCORES:")
    print(f"  • Auto Win Probability: {auto_decision.auto_win_probability:.0%}")
    print(f"  • Bonus Probability: {auto_decision.bonus_probability:.0%}")
    print(f"  • Expected Auto Points: {auto_decision.expected_auto_points:.1f}")
    print(f"  • Positioning Score: {auto_decision.positioning_score:.0%}")
    
    print(f"\n🎯 BLOCK TARGETS:")
    for goal, blocks in auto_decision.block_targets.items():
        print(f"  • {goal.upper()}: {blocks} blocks")
    
    print(f"\n⏱️  TIME ALLOCATION:")
    for activity, time in auto_decision.time_allocation.items():
        print(f"  • {activity}: {time:.1f}s")
    
    print(f"\n⚠️  RISK ASSESSMENT: {auto_decision.risk_assessment}")
    print(f"📝 RATIONALE: {auto_decision.decision_rationale}")
    
    # 3. Goal Priority Analysis
    print(f"\n{'='*20} 3. GOAL PRIORITY STRATEGY {'='*20}")
    goal_priority = analyzer.analyze_goal_priority_strategy(robots, "aggressive", "mid")
    
    print(f"🏆 RECOMMENDED PRIORITY: {goal_priority.recommended_priority.value}")
    print(f"✅ DECISION CONFIDENCE: {goal_priority.decision_confidence:.0%}")
    
    print(f"\n💰 VALUE PER BLOCK:")
    print(f"  • Center Goals: {goal_priority.center_goal_value:.2f} pts/block")
    print(f"  • Long Goals: {goal_priority.long_goal_value:.2f} pts/block")
    
    print(f"\n🎯 OPTIMAL SEQUENCE: {' → '.join(goal_priority.optimal_sequence)}")
    
    print(f"\n🛡️  CONTROL DIFFICULTY:")
    for goal, difficulty in goal_priority.control_difficulty.items():
        print(f"  • {goal.upper()}: {difficulty:.0%}")
    
    print(f"\n⚔️  OPPONENT INTERFERENCE:")
    for goal, interference in goal_priority.opponent_interference.items():
        print(f"  • {goal.upper()}: {interference:.0%}")
    
    # 4. Parking Decision Analysis
    print(f"\n{'='*20} 4. PARKING DECISION TIMING {'='*20}")
    
    # Test different game situations
    situations = [
        ("Leading by 15", PushBackMatchState(30, 85, 70, {"long_1": 8}, {"long_1": 6}, 0, 0, True, "endgame")),
        ("Close game", PushBackMatchState(45, 75, 73, {"long_1": 6}, {"long_1": 7}, 0, 0, True, "driver")),
        ("Trailing by 12", PushBackMatchState(35, 68, 80, {"center_1": 5}, {"center_1": 6}, 0, 0, True, "endgame"))
    ]
    
    for situation_name, match_state in situations:
        print(f"\n📋 SITUATION: {situation_name}")
        parking = analyzer.analyze_parking_decision_timing(match_state, robots)
        
        print(f"  • Recommended Timing: {parking.recommended_timing.value}")
        print(f"  • One Robot Threshold: {parking.one_robot_threshold} pts")
        print(f"  • Two Robot Threshold: {parking.two_robot_threshold} pts") 
        print(f"  • Expected Value: {parking.expected_parking_value:.0f} pts")
        print(f"  • Risk-Benefit Ratio: {parking.risk_benefit_ratio:.2f}")
    
    # 5. Offense-Defense Balance
    print(f"\n{'='*20} 5. OFFENSE-DEFENSE BALANCE {'='*20}")
    
    match_state = PushBackMatchState(
        60, 45, 52, 
        {"long_1": 4, "center_1": 3}, 
        {"long_1": 5, "center_2": 4}, 
        0, 0, True, "driver"
    )
    
    offense_defense = analyzer.analyze_offense_defense_balance(match_state, robots)
    offense_pct, defense_pct = offense_defense.recommended_ratio
    
    print(f"⚖️  RECOMMENDED RATIO: {offense_pct:.0%} Offense / {defense_pct:.0%} Defense")
    
    print(f"\n📊 ROI ANALYSIS:")
    print(f"  • Offensive ROI: {offense_defense.offensive_roi:.1f}")
    print(f"  • Defensive ROI: {offense_defense.defensive_roi:.1f}")
    
    print(f"\n🛡️  CRITICAL ZONES TO DEFEND:")
    for zone in offense_defense.critical_control_zones:
        print(f"  • {zone.upper()}")
    
    print(f"\n⚔️  DISRUPTION TARGETS:")
    for target in offense_defense.disruption_targets:
        print(f"  • {target.upper()}")

def demo_push_back_archetypes():
    """Demonstrate Push Back strategy archetypes"""
    
    analyzer = PushBackStrategyAnalyzer()
    archetypes = analyzer.create_push_back_archetype_strategies()
    
    print(f"\n{'='*20} PUSH BACK STRATEGY ARCHETYPES {'='*20}")
    
    for archetype, strategy in archetypes.items():
        print(f"\n🏗️  {archetype.value.upper().replace('_', ' ')}")
        print(f"   Name: {strategy.name}")
        
        total_auto = sum(strategy.blocks_scored_auto.values())
        total_driver = sum(strategy.blocks_scored_driver.values())
        print(f"   Blocks: {total_auto} Auto + {total_driver} Driver = {total_auto + total_driver} Total")
        
        print(f"   Auto Targets: {strategy.blocks_scored_auto}")
        print(f"   Parking: {len([p for p in strategy.robots_parked if p.value != 'none'])} robots")
        print(f"   Loader Blocks: {strategy.loader_blocks_removed}")
        print(f"   Auto Win Eligible: {'Yes' if total_auto >= 7 and len([k for k,v in strategy.blocks_scored_auto.items() if v > 0]) >= 3 and strategy.loader_blocks_removed >= 3 else 'No'}")

def demo_monte_carlo_comparison():
    """Demonstrate Monte Carlo comparison of archetypes"""
    
    analyzer = PushBackStrategyAnalyzer()
    archetypes = analyzer.create_push_back_archetype_strategies()
    
    print(f"\n{'='*20} MONTE CARLO ARCHETYPE COMPARISON {'='*20}")
    
    # Test subset of archetypes for demo (to keep runtime reasonable)
    test_archetypes = [
        PushBackArchetype.BLOCK_FLOW_MAXIMIZER,
        PushBackArchetype.CONTROL_ZONE_CONTROLLER,
        PushBackArchetype.AUTONOMOUS_SPECIALIST,
        PushBackArchetype.BALANCED_OPTIMIZER
    ]
    
    results = {}
    
    for archetype in test_archetypes:
        print(f"\n🎲 Testing {archetype.value}...")
        strategy = archetypes[archetype]
        mc_result = analyzer.run_push_back_monte_carlo(strategy, num_simulations=200)
        results[archetype] = mc_result
    
    # Display comparison
    print(f"\n📊 ARCHETYPE PERFORMANCE COMPARISON")
    print(f"{'Archetype':<25} {'Win Rate':<10} {'Avg Score':<10} {'Consistency':<12} {'Best vs':<15}")
    print("-" * 75)
    
    for archetype, result in results.items():
        consistency = 1 - (result['score_std'] / result['avg_score']) if result['avg_score'] > 0 else 0
        
        # Find best matchup
        best_matchup = max(result['opponent_matchups'].items(), key=lambda x: x[1]['win_rate'])
        best_vs = f"{best_matchup[0][:12]} ({best_matchup[1]['win_rate']:.0%})"
        
        print(f"{archetype.value:<25} {result['win_rate']:<10.1%} {result['avg_score']:<10.0f} {consistency:<12.1%} {best_vs:<15}")
    
    # Show best overall
    best_archetype = max(results.keys(), key=lambda x: results[x]['win_rate'])
    print(f"\n🏆 BEST PERFORMING ARCHETYPE: {best_archetype.value}")
    print(f"   Win Rate: {results[best_archetype]['win_rate']:.1%}")
    print(f"   Average Score: {results[best_archetype]['avg_score']:.0f}")

def main():
    """Run comprehensive Push Back analysis demo"""
    
    print("🚀 PUSH BACK STRATEGIC ANALYSIS - COMPREHENSIVE DEMO")
    print("=" * 70)
    print("Showcasing Push Back-specific strategic analysis capabilities")
    print("Replacing generic VEX analysis with specialized Push Back decisions")
    
    # Part 1: Individual strategic decisions
    demo_individual_strategic_decisions()
    
    # Part 2: Push Back archetypes
    demo_push_back_archetypes()
    
    # Part 3: Monte Carlo comparison
    demo_monte_carlo_comparison()
    
    print(f"\n{'='*20} DEMO COMPLETE {'='*20}")
    print("✅ All 5 Push Back strategic decisions implemented")
    print("✅ 7 Push Back-specific archetypes created")
    print("✅ Monte Carlo simulation with realistic variations")
    print("✅ Comprehensive strategic recommendations")
    
    print(f"\n🎯 KEY FEATURES:")
    print("• Block Flow Optimization with 88-block constraints")
    print("• Autonomous vs Driver Balance (10pt bonus vs 7pt win)")
    print("• Center vs Long Goal Priority (capacity vs control value)")
    print("• Parking Decision Timing (8pt vs 30pt analysis)")
    print("• Offensive vs Defensive Resource Allocation")
    print("• Push Back-tuned Monte Carlo with robot specs")
    print("• Integrated strategic recommendations")
    
    print(f"\n🔄 REPLACES GENERIC ANALYSIS WITH:")
    print("• Push Back-specific strategy archetypes")
    print("• Hardcoded Push Back scoring rules")
    print("• Realistic robot capabilities and timing")
    print("• VEX U Push Back strategic decision frameworks")

if __name__ == "__main__":
    main()