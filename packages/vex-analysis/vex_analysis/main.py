#!/usr/bin/env python3
"""
VEX U Push Back Strategic Analysis Toolkit - Unified CLI Entry Point

Usage:
    python3 main.py demo                    # Quick strategy demo
    python3 main.py analyze                 # Full strategy analysis
    python3 main.py visualize               # Create interactive visualizations
    python3 main.py report                  # Generate strategic report
    python3 main.py statistical             # Statistical analysis
    python3 main.py ml-train                # Train ML models
    python3 main.py ml-predict              # ML-powered predictions
    python3 main.py ml-optimize             # ML-powered optimizations
    python3 main.py pattern-discovery       # Discover winning patterns
    python3 main.py scenario-evolution      # Evolve optimal scenarios
    python3 main.py test                    # Run system tests
    python3 main.py validate                # Validate system
"""

import sys
import argparse
from pathlib import Path

# Add vex_analysis to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def run_demo():
    """Run quick strategy demonstration"""
    print("🚀 Running VEX U Strategy Demo...")
    
    # Import here to handle path issues
    from core.simulator import ScoringSimulator
    from analysis.strategy_analyzer import AdvancedStrategyAnalyzer
    
    # Initialize analyzer with ML models
    simulator = ScoringSimulator()
    analyzer = AdvancedStrategyAnalyzer(simulator, enable_ml_models=True)
    
    print("🎯 VEX U STRATEGY ANALYZER DEMONSTRATION")
    print("=" * 60)
    print("Running quick analysis with 100 simulations per strategy...")
    
    # Run quick analysis
    results = analyzer.run_complete_analysis(
        num_monte_carlo=100,
        include_coordination=False
    )
    
    print("\n" + "=" * 60)
    print("📊 STRATEGY ANALYSIS RESULTS")
    print("=" * 60)
    
    # Top strategies
    print("\n🏆 TOP STRATEGIES BY WIN RATE:")
    sorted_metrics = sorted(results['metrics'], key=lambda x: x.win_rate, reverse=True)
    
    for i, metric in enumerate(sorted_metrics[:5]):
        print(f"{i+1}. {metric.strategy_name}: {metric.win_rate:.1%} win rate")
        print(f"   Average Score: {metric.avg_score:.0f} points")
        
        # Show ML predictions if available
        if hasattr(metric, 'ml_predicted_score') and metric.ml_predicted_score:
            print(f"   ML Predicted Score: {metric.ml_predicted_score:.0f} points")
        if hasattr(metric, 'ml_coordination_score') and metric.ml_coordination_score:
            print(f"   Coordination Score: {metric.ml_coordination_score:.2f}")
        if hasattr(metric, 'ml_optimization_suggestions') and metric.ml_optimization_suggestions:
            print(f"   Top Suggestion: {metric.ml_optimization_suggestions[0]}")
        print()
    
    print("\n✅ Quick demo completed successfully!")
    return results


def run_analysis():
    """Run comprehensive strategy analysis"""
    print("📊 Running Comprehensive Strategy Analysis...")
    from tests.system_integration import main as system_main
    system_main()


def run_visualizations():
    """Create interactive visualizations"""
    print("📈 Creating Interactive Visualizations...")
    from visualization.interactive import InteractiveVEXVisualizer
    from analysis.strategy_analyzer import AdvancedStrategyAnalyzer
    from core.simulator import ScoringSimulator
    
    # Initialize components
    simulator = ScoringSimulator()
    analyzer = AdvancedStrategyAnalyzer(simulator)
    viz = InteractiveVEXVisualizer()
    
    # Generate sample strategies for visualization
    print("Generating sample strategies...")
    results = analyzer.run_complete_analysis(num_monte_carlo=50, include_coordination=False)
    strategies = analyzer.create_core_strategies()
    
    print("Creating scoring timeline...")
    viz.create_scoring_timeline_visualization(strategies[:3])
    
    print("Creating strategy comparison...")
    viz.create_strategy_comparison_visualization(results['metrics'])
    
    print("Creating insights dashboard...")
    viz.create_insights_dashboard(results)
    
    print("Creating match predictor...")
    viz.create_match_outcome_predictor()
    
    print("✅ Interactive visualizations created in ./visualizations/")


def run_report():
    """Generate strategic report"""
    print("📋 Generating Strategic Report...")
    from reporting.generator import VEXReportGenerator
    
    generator = VEXReportGenerator()
    report_path = generator.generate_comprehensive_report()
    print(f"✅ Strategic report generated: {report_path}")


def run_statistical():
    """Run statistical analysis"""
    print("📊 Running Statistical Analysis...")
    from demos.statistical_demo import main as stats_main
    stats_main()


def run_tests():
    """Run comprehensive tests"""
    print("🧪 Running Comprehensive Tests...")
    from tests.comprehensive_tests import main as test_main
    test_main()


def run_validation():
    """Run system validation"""
    print("✅ Running System Validation...")
    from scripts.validate import main as validate_main
    validate_main()


def run_ml_training():
    """Train ML models with synthetic data"""
    print("🤖 Training ML Models...")
    
    from core.simulator import ScoringSimulator
    from analysis.strategy_analyzer import AdvancedStrategyAnalyzer
    
    # Initialize analyzer
    simulator = ScoringSimulator()
    analyzer = AdvancedStrategyAnalyzer(simulator, enable_ml_models=True)
    
    if not analyzer.enable_ml_models:
        print("❌ ML models not available. Please install required dependencies.")
        return
    
    print("🔧 Training all ML models with 2000 samples...")
    print("This may take 5-10 minutes depending on your system.")
    
    try:
        # Train models
        results = analyzer.train_ml_models(num_samples=2000)
        
        print("\n📈 TRAINING RESULTS:")
        print("=" * 50)
        
        for model_name, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"{model_name.replace('_', ' ').title()}: {status}")
        
        successful_models = sum(results.values())
        total_models = len(results)
        
        print(f"\n🎯 Training Summary: {successful_models}/{total_models} models trained successfully")
        
        if successful_models > 0:
            print("\n✅ Models are now available for enhanced analysis!")
            print("   Try: python3 main.py ml-predict")
        else:
            print("\n⚠️  No models were trained successfully. Check error messages above.")
    
    except Exception as e:
        print(f"❌ Training failed: {e}")


def run_ml_predictions():
    """Run ML-powered strategy predictions"""
    print("🔮 Running ML-Powered Strategy Predictions...")
    
    from core.simulator import ScoringSimulator, AllianceStrategy, Zone, ParkingLocation
    from analysis.strategy_analyzer import AdvancedStrategyAnalyzer
    
    # Initialize analyzer
    simulator = ScoringSimulator()
    analyzer = AdvancedStrategyAnalyzer(simulator, enable_ml_models=True)
    
    if not analyzer.enable_ml_models:
        print("❌ ML models not available. Run 'python3 main.py ml-train' first.")
        return
    
    # Create sample strategies
    test_strategies = [
        AllianceStrategy(
            name="Aggressive Offense",
            blocks_scored_auto={"long_1": 6, "long_2": 6, "center_1": 4, "center_2": 4},
            blocks_scored_driver={"long_1": 15, "long_2": 12, "center_1": 8, "center_2": 6},
            zones_controlled=[],
            robots_parked=[ParkingLocation.NONE, ParkingLocation.ALLIANCE_ZONE]
        ),
        AllianceStrategy(
            name="Balanced Control",
            blocks_scored_auto={"long_1": 4, "long_2": 4, "center_1": 3, "center_2": 3},
            blocks_scored_driver={"long_1": 8, "long_2": 8, "center_1": 6, "center_2": 6},
            zones_controlled=[Zone.RED_HOME, Zone.NEUTRAL],
            robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.PLATFORM]
        ),
        AllianceStrategy(
            name="Zone Dominance",
            blocks_scored_auto={"long_1": 3, "long_2": 3, "center_1": 2, "center_2": 2},
            blocks_scored_driver={"long_1": 6, "long_2": 6, "center_1": 5, "center_2": 5},
            zones_controlled=[Zone.RED_HOME, Zone.BLUE_HOME, Zone.NEUTRAL],
            robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.ALLIANCE_ZONE]
        )
    ]
    
    print("\n🎯 STRATEGY PREDICTIONS")
    print("=" * 60)
    
    for i, strategy in enumerate(test_strategies, 1):
        print(f"\n{i}. {strategy.name}")
        print("-" * 30)
        
        try:
            # Get ML recommendations
            recommendations = analyzer.get_ml_strategy_recommendations(strategy)
            
            if 'error' in recommendations:
                print(f"   ❌ Analysis failed: {recommendations['error']}")
                continue
            
            # Strategy analysis
            if recommendations.get('strategy_analysis'):
                sa = recommendations['strategy_analysis']
                print(f"   🤖 Predicted Strategy: {sa.get('predicted_strategy', 'Unknown')}")
                print(f"   📊 Confidence: {sa.get('confidence', 0):.1%}")
                print(f"   🎭 Robot Roles: {', '.join(sa.get('robot_roles', []))}")
            
            # Score optimization
            if recommendations.get('score_optimization'):
                so = recommendations['score_optimization']
                pred_score = so.get('predicted_score', 0)
                win_prob = so.get('win_probability', 0)
                print(f"   🎯 Predicted Score: {pred_score:.0f} points")
                print(f"   🏆 Win Probability: {win_prob:.1%}")
                print(f"   ⚠️  Risk Level: {so.get('risk_level', 'Unknown')}")
                
                suggestions = so.get('suggestions', [])
                if suggestions:
                    print(f"   💡 Top Suggestion: {suggestions[0]}")
            
            # Coordination plan
            if recommendations.get('coordination_plan'):
                cp = recommendations['coordination_plan']
                print(f"   🤝 Optimal Coordination: {cp.get('optimal_strategy', 'Unknown')}")
                print(f"   ⚡ Synergy Score: {cp.get('synergy_score', 0):.2f}")
                print(f"   🎪 Robot Roles: {cp.get('robot1_role', '?')} + {cp.get('robot2_role', '?')}")
            
            # Overall assessment
            assessment = recommendations.get('overall_assessment', '')
            if assessment:
                print(f"   📋 Assessment: {assessment}")
        
        except Exception as e:
            print(f"   ❌ Prediction failed: {e}")
    
    print("\n✅ ML predictions completed!")


def run_ml_optimization():
    """Run ML-powered strategy optimization"""
    print("⚡ Running ML-Powered Strategy Optimization...")
    
    from core.simulator import ScoringSimulator, AllianceStrategy, Zone, ParkingLocation
    from analysis.strategy_analyzer import AdvancedStrategyAnalyzer
    from ml_models.scoring_optimizer import VEXUScoringOptimizer
    from ml_models.coordination_model import VEXUCoordinationModel
    from ml_models.feature_engineering import create_game_state_from_strategy
    
    # Initialize components
    simulator = ScoringSimulator()
    optimizer = VEXUScoringOptimizer()
    coord_model = VEXUCoordinationModel()
    
    # Try to load models
    opt_loaded = optimizer.load_models()
    coord_loaded = coord_model.load_models()
    
    if not (opt_loaded or coord_loaded):
        print("❌ No trained models found. Run 'python3 main.py ml-train' first.")
        return
    
    # Create a strategy to optimize
    base_strategy = AllianceStrategy(
        name="Strategy to Optimize",
        blocks_scored_auto={"long_1": 4, "long_2": 4, "center_1": 3, "center_2": 3},
        blocks_scored_driver={"long_1": 10, "long_2": 8, "center_1": 6, "center_2": 5},
        zones_controlled=[Zone.RED_HOME],
        robots_parked=[ParkingLocation.ALLIANCE_ZONE, ParkingLocation.NONE]
    )
    
    print(f"\n🎯 OPTIMIZING STRATEGY: {base_strategy.name}")
    print("=" * 60)
    
    try:
        # Create game state
        opponent = AllianceStrategy(
            name="Average Opponent",
            blocks_scored_auto={"long_1": 3, "long_2": 3, "center_1": 2, "center_2": 2},
            blocks_scored_driver={"long_1": 8, "long_2": 8, "center_1": 5, "center_2": 5},
            zones_controlled=[Zone.BLUE_HOME],
            robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.ALLIANCE_ZONE]
        )
        
        game_state = create_game_state_from_strategy(base_strategy, opponent)
        
        # Score optimization
        if opt_loaded:
            print("\n📈 SCORE OPTIMIZATION ANALYSIS")
            print("-" * 40)
            
            result = optimizer.predict_score(game_state, "red")
            
            print(f"🎯 Predicted Score: {result.predicted_score:.1f} points")
            print(f"📊 Confidence Range: {result.confidence_interval[0]:.1f} - {result.confidence_interval[1]:.1f}")
            print(f"🏆 Win Probability: {result.expected_win_probability:.1%}")
            print(f"⚠️  Risk Assessment: {result.risk_assessment}")
            
            print(f"\n💡 TOP OPTIMIZATION SUGGESTIONS:")
            for i, suggestion in enumerate(result.optimization_suggestions[:5], 1):
                print(f"   {i}. {suggestion}")
            
            # Feature importance
            print(f"\n📊 TOP FEATURE CONTRIBUTIONS:")
            sorted_features = sorted(
                result.feature_contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            for feature, contribution in sorted_features[:5]:
                print(f"   {feature}: {contribution:+.2f}")
        
        # Coordination optimization
        if coord_loaded:
            print("\n\n🤝 COORDINATION OPTIMIZATION")
            print("-" * 40)
            
            coord_plan = coord_model.optimize_robot_coordination(game_state, "red")
            
            print(f"⚡ Optimal Strategy: {coord_plan.strategy_type.value}")
            print(f"🎪 Synergy Score: {coord_plan.synergy_score:.3f}")
            print(f"🎯 Expected Score: {coord_plan.expected_total_score:.1f}")
            print(f"⚠️  Risk Level: {coord_plan.risk_level}")
            
            print(f"\n🤖 ROBOT ASSIGNMENTS:")
            print(f"   Robot 1: {coord_plan.robot1_assignment.primary_task.value}")
            print(f"   - Goals: {coord_plan.robot1_assignment.assigned_goals}")
            print(f"   - Expected Contribution: {coord_plan.robot1_assignment.expected_contribution:.1f}")
            
            print(f"\n   Robot 2: {coord_plan.robot2_assignment.primary_task.value}")
            print(f"   - Goals: {coord_plan.robot2_assignment.assigned_goals}")
            print(f"   - Expected Contribution: {coord_plan.robot2_assignment.expected_contribution:.1f}")
        
        print("\n✅ Optimization analysis completed!")
    
    except Exception as e:
        print(f"❌ Optimization failed: {e}")


def run_pattern_discovery():
    """Run ML-powered pattern discovery on strategies"""
    print("🔍 Running ML-Powered Pattern Discovery...")
    
    from core.simulator import ScoringSimulator
    from core.scenario_generator import ScenarioGenerator
    
    # Initialize components
    simulator = ScoringSimulator()
    generator = ScenarioGenerator(simulator, enable_ml=True)
    
    print("\n📊 Analyzing Strategy Effectiveness for Pattern Discovery...")
    
    # Analyze various strategy combinations to build match history
    strategy_df = generator.analyze_strategy_effectiveness(num_samples=30)
    print(f"✅ Analyzed {len(strategy_df)} strategy combinations")
    
    # Display top strategies
    print("\n🏆 TOP PERFORMING STRATEGIES:")
    top_strategies = strategy_df.nlargest(5, 'win_rate')
    for i, (_, row) in enumerate(top_strategies.iterrows(), 1):
        print(f"  {i}. {row['strategy_type']} ({row['skill_level']})")
        print(f"     Win Rate: {row['win_rate']:.1%} | Avg Score: {row['avg_score']:.0f} | Consistency: {row['consistency']:.2f}")
    
    # Discover patterns
    print("\n🧠 Discovering Winning Patterns...")
    patterns = generator.discover_winning_patterns(min_win_rate=0.75)
    
    if patterns:
        print(f"✅ Discovered {len(patterns)} winning patterns with >75% win rate")
        
        # Show pattern insights
        insights = generator.get_pattern_insights()
        print(f"\n💡 PATTERN INSIGHTS:")
        for insight in insights['insights']:
            print(f"  • {insight}")
        
        # Show detailed patterns
        print(f"\n📋 DETAILED PATTERN ANALYSIS:")
        for i, pattern in enumerate(patterns[:3], 1):  # Show top 3
            print(f"\n  Pattern {i}: {pattern.pattern_id}")
            print(f"    Type: {pattern.pattern_type}")
            print(f"    Win Rate: {pattern.win_rate:.1%}")
            print(f"    Frequency: {pattern.frequency:.1%}")
            print(f"    Description: {pattern.description}")
            print(f"    Confidence: {pattern.confidence:.2f}")
    else:
        print("❌ No strong patterns discovered. Try with lower win rate threshold or more data.")
    
    # Answer strategic questions
    print(f"\n❓ STRATEGIC QUESTION ANALYSIS:")
    
    questions = [
        ("high_win_scoring_patterns", "What scoring patterns appear in 80%+ win rate matches?"),
        ("optimal_strategy_switch_timing", "When is optimal time to switch strategy?"),
        ("effective_coordination_patterns", "What coordination patterns are most effective?")
    ]
    
    for q_type, q_text in questions:
        print(f"\n🤔 {q_text}")
        answer = generator.answer_strategic_questions(q_type)
        print(f"💭 {answer}")
    
    # Export results
    print(f"\n💾 Exporting Pattern Discovery Results...")
    try:
        generator.export_ml_analysis("pattern_discovery_results")
        print("✅ Results exported successfully")
    except Exception as e:
        print(f"⚠️  Export failed: {e}")


def run_scenario_evolution():
    """Run genetic algorithm-based scenario evolution"""
    print("🧬 Running Genetic Algorithm-Based Scenario Evolution...")
    
    from core.simulator import ScoringSimulator
    from core.scenario_generator import ScenarioGenerator
    
    # Initialize components
    simulator = ScoringSimulator()
    generator = ScenarioGenerator(simulator, enable_ml=True)
    
    # Define evolution constraints
    constraints = {
        'competitiveness_target': 0.8,  # Target highly competitive matches
        'min_skill_level': 'intermediate',
        'strategy_diversity': True,
        'balanced_matchups': True
    }
    
    print(f"\n⚙️  Evolution Constraints:")
    for key, value in constraints.items():
        print(f"  • {key}: {value}")
    
    # Generate ML-optimized scenarios
    print(f"\n🧬 Evolving Optimal Scenarios...")
    print("This may take 1-2 minutes...")
    
    try:
        ml_scenarios = generator.generate_ml_optimized_scenarios(
            constraints=constraints,
            num_scenarios=5
        )
        
        print(f"✅ Generated {len(ml_scenarios)} evolved scenarios")
        
        # Test and analyze evolved scenarios
        print(f"\n🧪 TESTING EVOLVED SCENARIOS:")
        print("=" * 50)
        
        total_competitiveness = 0
        for i, (red, blue) in enumerate(ml_scenarios):
            result = simulator.simulate_match(red, blue)
            competitiveness = 1 - abs(result.margin) / max(result.red_score, result.blue_score, 1)
            total_competitiveness += competitiveness
            
            print(f"\nScenario {i+1}: {red.name} vs {blue.name}")
            print(f"  Winner: {result.winner.upper()} by {result.margin} points")
            print(f"  Scores: Red {result.red_score} | Blue {result.blue_score}")
            print(f"  Competitiveness: {competitiveness:.1%}")
            
            # Show strategy details
            red_blocks = sum(red.blocks_scored_auto.values()) + sum(red.blocks_scored_driver.values())
            blue_blocks = sum(blue.blocks_scored_auto.values()) + sum(blue.blocks_scored_driver.values())
            print(f"  Red Strategy: {red_blocks} blocks, {len(red.zones_controlled)} zones")
            print(f"  Blue Strategy: {blue_blocks} blocks, {len(blue.zones_controlled)} zones")
        
        avg_competitiveness = total_competitiveness / len(ml_scenarios)
        print(f"\n📊 EVOLUTION RESULTS:")
        print(f"  Average Competitiveness: {avg_competitiveness:.1%}")
        print(f"  Target Competitiveness: {constraints['competitiveness_target']:.1%}")
        
        success_rate = "✅ SUCCESS" if avg_competitiveness >= constraints['competitiveness_target'] else "⚠️  PARTIAL SUCCESS"
        print(f"  Evolution Status: {success_rate}")
        
        # Analyze critical moments
        print(f"\n⏰ CRITICAL MOMENT ANALYSIS:")
        critical_moments = generator.analyze_critical_moments_in_scenarios(ml_scenarios[:3])
        
        total_moments = 0
        for scenario_id, moments in critical_moments.items():
            total_moments += len(moments)
            if moments:
                print(f"\n  {scenario_id}: {len(moments)} critical moments")
                for moment in moments[:2]:  # Show first 2
                    print(f"    • {moment.timestamp:.1f}s: {moment.optimal_choice}")
                    print(f"      Impact: {moment.impact_magnitude:.1f} | Confidence: {moment.confidence:.2f}")
        
        print(f"\n  Total Critical Moments Identified: {total_moments}")
        
    except Exception as e:
        print(f"❌ Scenario evolution failed: {e}")
        print("This may be due to missing ML dependencies. Try installing:")
        print("  pip install ruptures deap scikit-learn")


def main():
    parser = argparse.ArgumentParser(
        description="VEX U Push Back Strategic Analysis Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 main.py demo                    # Quick 30-second demo
  python3 main.py analyze                 # Full 5-minute analysis  
  python3 main.py visualize               # Interactive dashboards
  python3 main.py report                  # Professional report
  python3 main.py statistical             # Statistical insights
  python3 main.py test                    # Validation tests
  python3 main.py validate                # System validation

For detailed usage, see README.md
        """
    )
    
    parser.add_argument(
        'command',
        choices=['demo', 'analyze', 'visualize', 'report', 'statistical', 'ml-train', 'ml-predict', 'ml-optimize', 'pattern-discovery', 'scenario-evolution', 'test', 'validate'],
        help='Command to execute'
    )
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    try:
        if args.command == 'demo':
            run_demo()
        elif args.command == 'analyze':
            run_analysis()
        elif args.command == 'visualize':
            run_visualizations()
        elif args.command == 'report':
            run_report()
        elif args.command == 'statistical':
            run_statistical()
        elif args.command == 'test':
            run_tests()
        elif args.command == 'ml-train':
            run_ml_training()
        elif args.command == 'ml-predict':
            run_ml_predictions()
        elif args.command == 'ml-optimize':
            run_ml_optimization()
        elif args.command == 'pattern-discovery':
            run_pattern_discovery()
        elif args.command == 'scenario-evolution':
            run_scenario_evolution()
        elif args.command == 'validate':
            run_validation()
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()