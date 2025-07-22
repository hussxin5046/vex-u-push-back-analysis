#!/usr/bin/env python3
"""
Demo script for ML-Enhanced Scenario Generator

This demonstrates the key capabilities of the enhanced scenario generator:
1. Pattern Discovery using unsupervised learning
2. Critical Moment Identification with change point detection  
3. Scenario Evolution using genetic algorithms
4. Strategic Question Answering

Usage: python3 demo_ml_scenario_generator.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demo_basic_functionality():
    """Demo basic functionality without requiring all ML dependencies"""
    
    print("🤖 VEX U ML-Enhanced Scenario Generator Demo")
    print("=" * 60)
    
    try:
        from src.core.simulator import ScoringSimulator
        from src.core.scenario_generator import ScenarioGenerator, MLScenarioDiscovery
        
        # Initialize components
        simulator = ScoringSimulator()
        generator = ScenarioGenerator(simulator, enable_ml=True)
        
        print("✅ Core components initialized successfully")
        print(f"   ML capabilities enabled: {generator.ml_discovery.enable_ml}")
        
        # Test basic scenario generation
        print("\n📊 Testing Basic Scenario Generation...")
        
        # Generate traditional scenarios
        traditional_scenarios = generator.generate_competitive_scenarios(3)
        print(f"✅ Generated {len(traditional_scenarios)} traditional scenarios")
        
        # Test each scenario
        for i, (red, blue) in enumerate(traditional_scenarios):
            result = simulator.simulate_match(red, blue)
            competitiveness = 1 - abs(result.margin) / max(result.red_score, result.blue_score, 1)
            print(f"  Scenario {i+1}: {result.winner} wins, competitiveness: {competitiveness:.1%}")
        
        # Test strategy effectiveness analysis
        print("\n🔍 Testing Strategy Effectiveness Analysis...")
        strategy_df = generator.analyze_strategy_effectiveness(num_samples=10)  # Small sample for demo
        print(f"✅ Analyzed {len(strategy_df)} strategy combinations")
        
        if len(strategy_df) > 0:
            top_strategy = strategy_df.nlargest(1, 'win_rate').iloc[0]
            print(f"   Top strategy: {top_strategy['strategy_type']} ({top_strategy['skill_level']})")
            print(f"   Win rate: {top_strategy['win_rate']:.1%}")
        
        # Test pattern discovery (will work even without full ML stack)
        if hasattr(generator, '_match_history'):
            print("\n🧠 Testing Pattern Discovery...")
            patterns = generator.discover_winning_patterns(min_win_rate=0.5)  # Lower threshold for demo
            print(f"✅ Pattern discovery completed: {len(patterns)} patterns found")
            
            if patterns:
                for pattern in patterns[:2]:  # Show first 2
                    print(f"   • {pattern.description} (Win rate: {pattern.win_rate:.1%})")
        
        # Test strategic question answering
        print("\n❓ Testing Strategic Question Answering...")
        questions = [
            "high_win_scoring_patterns",
            "optimal_strategy_switch_timing", 
            "effective_coordination_patterns"
        ]
        
        for question in questions:
            answer = generator.answer_strategic_questions(question)
            print(f"   Q: {question}")
            print(f"   A: {answer.split('.')[0]}...")  # Show first sentence
        
        print("\n✅ Basic Demo Completed Successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the project root directory")
    except Exception as e:
        print(f"❌ Demo error: {e}")

def demo_advanced_ml_features():
    """Demo advanced ML features (requires additional dependencies)"""
    
    print("\n🧬 Advanced ML Features Demo")
    print("-" * 40)
    
    try:
        # Test if advanced ML libraries are available
        import ruptures
        import deap
        from sklearn.cluster import KMeans
        
        print("✅ Advanced ML libraries detected")
        
        from src.core.simulator import ScoringSimulator
        from src.core.scenario_generator import ScenarioGenerator
        
        simulator = ScoringSimulator()
        generator = ScenarioGenerator(simulator, enable_ml=True)
        
        if generator.ml_discovery.enable_ml:
            print("\n⚡ Testing ML-Optimized Scenario Evolution...")
            
            constraints = {
                'competitiveness_target': 0.7,
                'min_skill_level': 'intermediate',
                'strategy_diversity': True
            }
            
            # This will use genetic algorithms
            ml_scenarios = generator.generate_ml_optimized_scenarios(
                constraints=constraints,
                num_scenarios=2  # Small number for demo
            )
            
            print(f"✅ Generated {len(ml_scenarios)} ML-optimized scenarios")
            
            # Test the scenarios
            total_competitiveness = 0
            for i, (red, blue) in enumerate(ml_scenarios):
                result = simulator.simulate_match(red, blue)
                competitiveness = 1 - abs(result.margin) / max(result.red_score, result.blue_score, 1)
                total_competitiveness += competitiveness
                print(f"   Scenario {i+1}: {competitiveness:.1%} competitive")
            
            avg_competitiveness = total_competitiveness / len(ml_scenarios)
            print(f"   Average competitiveness: {avg_competitiveness:.1%}")
            
            # Test critical moment analysis
            print("\n⏰ Testing Critical Moment Analysis...")
            critical_moments = generator.analyze_critical_moments_in_scenarios(ml_scenarios[:1])
            
            total_moments = sum(len(moments) for moments in critical_moments.values())
            print(f"✅ Identified {total_moments} critical moments")
            
            if total_moments > 0:
                for scenario_id, moments in critical_moments.items():
                    if moments:
                        moment = moments[0]  # Show first moment
                        print(f"   {scenario_id}: {moment.timestamp:.1f}s - {moment.optimal_choice}")
        
        print("\n✅ Advanced ML Demo Completed!")
        
    except ImportError as e:
        print("⚠️  Advanced ML libraries not available")
        print(f"   Missing: {e}")
        print("   Install with: pip install ruptures deap scikit-learn")
        print("   This is optional - basic functionality still works!")
    except Exception as e:
        print(f"❌ Advanced demo error: {e}")

def show_available_commands():
    """Show available commands in the main application"""
    
    print("\n📋 Available ML Commands in Main Application:")
    print("-" * 50)
    print("python3 main.py pattern-discovery   # Discover winning patterns")
    print("python3 main.py scenario-evolution  # Evolve optimal scenarios")
    print("python3 main.py ml-train            # Train ML models")
    print("python3 main.py ml-predict          # ML predictions")
    print("python3 main.py ml-optimize         # ML optimization")
    
    print("\n🎯 Strategic Questions the System Can Answer:")
    print("• What scoring patterns appear in 80%+ win rate matches?")
    print("• When is the optimal time to switch from offense to defense?")
    print("• What two-robot coordination patterns are most effective?")
    print("• Which skill level + strategy combinations have highest win rates?")
    print("• What are the critical decision moments in competitive matches?")

if __name__ == "__main__":
    demo_basic_functionality()
    demo_advanced_ml_features()
    show_available_commands()
    
    print("\n🎉 Demo Complete!")
    print("\nNext Steps:")
    print("1. Install additional ML dependencies: pip install -r requirements_ml.txt")
    print("2. Try the new commands: python3 main.py pattern-discovery")
    print("3. Explore scenario evolution: python3 main.py scenario-evolution")
    print("4. Train ML models first: python3 main.py ml-train")