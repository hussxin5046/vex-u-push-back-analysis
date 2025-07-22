#!/usr/bin/env python3

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.scenario_generator import *
from core.simulator import ScoringSimulator

class TestEnhancedScenarioGenerator:
    def __init__(self):
        self.simulator = ScoringSimulator()
        self.generator = ScenarioGenerator(self.simulator)
        
    def test_capability_profiles(self):
        """Test robot capability profiles are properly defined"""
        print("Testing Capability Profiles...")
        
        # Test all skill levels have profiles
        for skill_level in SkillLevel:
            assert skill_level in self.generator.capability_profiles
            profile = self.generator.capability_profiles[skill_level]
            
            # Test profile completeness
            assert hasattr(profile, 'blocks_per_second')
            assert hasattr(profile, 'max_capacity')
            assert hasattr(profile, 'travel_time_per_goal')
            assert hasattr(profile, 'collection_time')
            assert hasattr(profile, 'accuracy')
            assert hasattr(profile, 'autonomous_reliability')
            
            # Test reasonable value ranges
            assert 0.1 <= profile.blocks_per_second <= 1.0
            assert 1 <= profile.max_capacity <= 10
            assert 1.0 <= profile.travel_time_per_goal <= 15.0
            assert 0.5 <= profile.collection_time <= 5.0
            assert 0.5 <= profile.accuracy <= 1.0
            assert 0.5 <= profile.autonomous_reliability <= 1.0
            
        print("✓ All capability profiles valid")
        
        # Test skill progression (higher skill = better performance)
        beginner = self.generator.capability_profiles[SkillLevel.BEGINNER]
        expert = self.generator.capability_profiles[SkillLevel.EXPERT]
        
        assert expert.blocks_per_second > beginner.blocks_per_second
        assert expert.max_capacity >= beginner.max_capacity
        assert expert.travel_time_per_goal < beginner.travel_time_per_goal
        assert expert.collection_time < beginner.collection_time
        assert expert.accuracy > beginner.accuracy
        assert expert.autonomous_reliability > beginner.autonomous_reliability
        
        print("✓ Skill progression validated")
        
    def test_realistic_scoring_calculation(self):
        """Test realistic scoring calculations"""
        print("\\nTesting Realistic Scoring Calculations...")
        
        # Test basic scoring calculation
        capabilities = self.generator.capability_profiles[SkillLevel.INTERMEDIATE]
        
        # Test different roles
        for role in RobotRole:
            auto_score = self.generator.calculate_realistic_scoring(
                capabilities, role, 15.0, 1.0, 1.0
            )
            driver_score = self.generator.calculate_realistic_scoring(
                capabilities, role, 105.0, 1.0, 1.0
            )
            
            # Scores should be non-negative
            assert auto_score >= 0
            assert driver_score >= 0
            
            # Driver period should generally score more (longer time)
            if role != RobotRole.DEFENDER:  # Defenders might score very little
                assert driver_score >= auto_score
                
        print("✓ Role-based scoring validated")
        
        # Test cooperation and interference effects
        base_score = self.generator.calculate_realistic_scoring(
            capabilities, RobotRole.SCORER, 60.0, 1.0, 1.0
        )
        
        low_cooperation = self.generator.calculate_realistic_scoring(
            capabilities, RobotRole.SCORER, 60.0, 0.6, 1.0
        )
        
        high_interference = self.generator.calculate_realistic_scoring(
            capabilities, RobotRole.SCORER, 60.0, 1.0, 0.7
        )
        
        # Lower cooperation and higher interference should reduce scoring
        assert low_cooperation < base_score
        assert high_interference < base_score
        
        print("✓ Cooperation and interference effects validated")
        
    def test_time_based_strategy_generation(self):
        """Test time-based strategy generation"""
        print("\\nTesting Time-Based Strategy Generation...")
        
        # Test all strategy types
        for strategy_type in StrategyType:
            for skill_level in SkillLevel:
                params = ScenarioParameters(
                    skill_level=skill_level,
                    strategy_type=strategy_type,
                    robot1_role=RobotRole.SCORER,
                    robot2_role=RobotRole.SCORER,
                    robot1_capabilities=self.generator.capability_profiles[skill_level],
                    robot2_capabilities=self.generator.capability_profiles[skill_level],
                    field_position="center",
                    cooperation_efficiency=0.8
                )
                
                strategy = self.generator.generate_time_based_strategy("Test", params)
                
                # Validate strategy structure
                assert isinstance(strategy, AllianceStrategy)
                assert strategy.name == "Test"
                assert len(strategy.blocks_scored_auto) == 4
                assert len(strategy.blocks_scored_driver) == 4
                assert len(strategy.robots_parked) == 2
                
                # Check block totals are reasonable
                total_auto = sum(strategy.blocks_scored_auto.values())
                total_driver = sum(strategy.blocks_scored_driver.values())
                
                assert total_auto >= 0
                assert total_driver >= 0
                assert total_auto + total_driver <= 88  # Can't exceed total blocks
                
        print("✓ Time-based strategy generation validated")
        
    def test_strategy_templates(self):
        """Test strategy template generation"""
        print("\\nTesting Strategy Templates...")
        
        templates = self.generator.generate_strategy_templates()
        
        # Test all strategy types have templates
        assert len(templates) == len(StrategyType)
        
        for strategy_type in StrategyType:
            assert strategy_type in templates
            template = templates[strategy_type]
            
            # Validate template structure
            assert isinstance(template, ScenarioParameters)
            assert template.strategy_type == strategy_type
            assert isinstance(template.skill_level, SkillLevel)
            assert isinstance(template.robot1_role, RobotRole)
            assert isinstance(template.robot2_role, RobotRole)
            assert 0.0 <= template.cooperation_efficiency <= 1.0
            
        print("✓ Strategy templates validated")
        
    def test_time_analysis_scenarios(self):
        """Test time analysis scenario generation"""
        print("\\nTesting Time Analysis Scenarios...")
        
        time_df = self.generator.generate_time_analysis_scenarios()
        
        # Validate DataFrame structure
        assert isinstance(time_df, pd.DataFrame)
        assert len(time_df) > 0
        
        required_columns = [
            'scenario_id', 'scoring_rate', 'capacity', 'cooperation',
            'auto_blocks_per_robot', 'driver_blocks_per_robot', 
            'alliance_total', 'efficiency_rating'
        ]
        
        for col in required_columns:
            assert col in time_df.columns
            
        # Validate data ranges
        assert time_df['scoring_rate'].min() >= 0.1
        assert time_df['scoring_rate'].max() <= 0.6
        assert time_df['capacity'].min() >= 2
        assert time_df['capacity'].max() <= 5
        assert time_df['cooperation'].min() >= 0.6
        assert time_df['cooperation'].max() <= 1.0
        assert time_df['alliance_total'].min() >= 0
        assert time_df['efficiency_rating'].min() >= 0
        
        print(f"✓ Time analysis: {len(time_df)} scenarios generated")
        
    def test_capability_comparison(self):
        """Test capability comparison generation"""
        print("\\nTesting Capability Comparison...")
        
        cap_df = self.generator.generate_capability_comparison()
        
        # Validate DataFrame structure
        assert isinstance(cap_df, pd.DataFrame)
        assert len(cap_df) == len(SkillLevel) * len(RobotRole)
        
        required_columns = [
            'skill_level', 'robot_role', 'blocks_per_second', 'accuracy',
            'auto_blocks_expected', 'driver_blocks_expected', 'total_blocks_expected'
        ]
        
        for col in required_columns:
            assert col in cap_df.columns
            
        # Test skill level progression
        scorer_data = cap_df[cap_df['robot_role'] == 'scorer']
        beginner_scorer = scorer_data[scorer_data['skill_level'] == 'beginner'].iloc[0]
        expert_scorer = scorer_data[scorer_data['skill_level'] == 'expert'].iloc[0]
        
        assert expert_scorer['total_blocks_expected'] > beginner_scorer['total_blocks_expected']
        assert expert_scorer['blocks_per_second'] > beginner_scorer['blocks_per_second']
        
        print(f"✓ Capability comparison: {len(cap_df)} combinations tested")
        
    def test_strategy_effectiveness(self):
        """Test strategy effectiveness analysis"""
        print("\\nTesting Strategy Effectiveness Analysis...")
        
        # Use small sample size for testing speed
        eff_df = self.generator.analyze_strategy_effectiveness(num_samples=5)
        
        # Validate DataFrame structure
        assert isinstance(eff_df, pd.DataFrame)
        assert len(eff_df) == len(StrategyType) * len(SkillLevel)
        
        required_columns = [
            'strategy_type', 'skill_level', 'win_rate', 'avg_score', 
            'avg_margin', 'consistency'
        ]
        
        for col in required_columns:
            assert col in eff_df.columns
            
        # Validate data ranges
        assert eff_df['win_rate'].min() >= 0.0
        assert eff_df['win_rate'].max() <= 1.0
        assert eff_df['avg_score'].min() > 0
        
        print(f"✓ Strategy effectiveness: {len(eff_df)} combinations analyzed")
        
    def test_scenario_parameter_creation(self):
        """Test scenario parameter creation"""
        print("\\nTesting Scenario Parameter Creation...")
        
        for skill_level in SkillLevel:
            for strategy_type in StrategyType:
                params = self.generator._create_scenario_parameters(
                    skill_level, strategy_type, "TestAlliance"
                )
                
                # Validate parameter structure
                assert isinstance(params, ScenarioParameters)
                assert params.skill_level == skill_level
                assert params.strategy_type == strategy_type
                assert isinstance(params.robot1_role, RobotRole)
                assert isinstance(params.robot2_role, RobotRole)
                assert 0.0 <= params.cooperation_efficiency <= 1.0
                
        print("✓ Scenario parameter creation validated")
        
    def test_block_distribution_strategies(self):
        """Test block distribution by strategy type"""
        print("\\nTesting Block Distribution Strategies...")
        
        total_blocks = 30
        goals = ["long_1", "long_2", "center_1", "center_2"]
        
        for strategy_type in StrategyType:
            # Test auto distribution
            auto_dist = self.generator._distribute_blocks_by_strategy(
                total_blocks, goals, strategy_type, is_auto=True
            )
            
            # Test driver distribution  
            driver_dist = self.generator._distribute_blocks_by_strategy(
                total_blocks, goals, strategy_type, is_auto=False
            )
            
            # Validate distributions
            assert sum(auto_dist.values()) == total_blocks
            assert sum(driver_dist.values()) == total_blocks
            assert len(auto_dist) == 4
            assert len(driver_dist) == 4
            
            # All values should be non-negative
            for dist in [auto_dist, driver_dist]:
                for blocks in dist.values():
                    assert blocks >= 0
                    
        print("✓ Block distribution strategies validated")
        
    def test_zone_and_parking_determination(self):
        """Test zone control and parking strategy determination"""
        print("\\nTesting Zone Control and Parking Determination...")
        
        for strategy_type in StrategyType:
            for skill_level in SkillLevel:
                params = ScenarioParameters(
                    skill_level=skill_level,
                    strategy_type=strategy_type,
                    robot1_role=RobotRole.SCORER,
                    robot2_role=RobotRole.SCORER,
                    robot1_capabilities=self.generator.capability_profiles[skill_level],
                    robot2_capabilities=self.generator.capability_profiles[skill_level],
                    field_position="test",
                    cooperation_efficiency=0.8
                )
                
                # Test zone control
                zones = self.generator._determine_zone_control(params)
                assert isinstance(zones, list)
                assert len(zones) <= 3  # Can't control more than 3 zones
                for zone in zones:
                    assert isinstance(zone, Zone)
                    
                # Test parking strategy
                parking = self.generator._determine_parking_strategy(params)
                assert isinstance(parking, list)
                assert len(parking) == 2  # Always 2 robots
                for location in parking:
                    assert isinstance(location, ParkingLocation)
                    
        print("✓ Zone control and parking determination validated")
        
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\\nTesting Edge Cases...")
        
        # Test zero time scoring
        capabilities = self.generator.capability_profiles[SkillLevel.BEGINNER]
        zero_score = self.generator.calculate_realistic_scoring(
            capabilities, RobotRole.SCORER, 0.0, 1.0, 1.0
        )
        assert zero_score == 0
        
        # Test zero block distribution
        zero_dist = self.generator._distribute_blocks_by_strategy(
            0, ["long_1", "long_2", "center_1", "center_2"], 
            StrategyType.ALL_OFFENSE, False
        )
        assert sum(zero_dist.values()) == 0
        assert all(blocks == 0 for blocks in zero_dist.values())
        
        print("✓ Edge cases handled correctly")
        
    def test_full_integration(self):
        """Test full integration with match simulation"""
        print("\\nTesting Full Integration...")
        
        # Generate two different strategies
        red_params = ScenarioParameters(
            skill_level=SkillLevel.ADVANCED,
            strategy_type=StrategyType.ALL_OFFENSE,
            robot1_role=RobotRole.SCORER,
            robot2_role=RobotRole.SCORER,
            robot1_capabilities=self.generator.capability_profiles[SkillLevel.ADVANCED],
            robot2_capabilities=self.generator.capability_profiles[SkillLevel.ADVANCED],
            field_position="center",
            cooperation_efficiency=0.9
        )
        
        blue_params = ScenarioParameters(
            skill_level=SkillLevel.INTERMEDIATE,
            strategy_type=StrategyType.ZONE_CONTROL,
            robot1_role=RobotRole.HYBRID,
            robot2_role=RobotRole.DEFENDER,
            robot1_capabilities=self.generator.capability_profiles[SkillLevel.INTERMEDIATE],
            robot2_capabilities=self.generator.capability_profiles[SkillLevel.INTERMEDIATE],
            field_position="defensive",
            cooperation_efficiency=0.8
        )
        
        # Generate strategies
        red_strategy = self.generator.generate_time_based_strategy("Red", red_params)
        blue_strategy = self.generator.generate_time_based_strategy("Blue", blue_params)
        
        # Simulate match
        result = self.simulator.simulate_match(red_strategy, blue_strategy)
        
        # Validate result
        assert hasattr(result, 'winner')
        assert hasattr(result, 'margin')
        assert hasattr(result, 'red_score')
        assert hasattr(result, 'blue_score')
        assert result.red_score > 0
        assert result.blue_score > 0
        
        print("✓ Full integration test passed")
        
    def run_all_tests(self):
        """Run all tests"""
        print("Running Comprehensive Tests on Enhanced Scenario Generator")
        print("=" * 65)
        
        try:
            self.test_capability_profiles()
            self.test_realistic_scoring_calculation()
            self.test_time_based_strategy_generation()
            self.test_strategy_templates()
            self.test_time_analysis_scenarios()
            self.test_capability_comparison()
            self.test_strategy_effectiveness()
            self.test_scenario_parameter_creation()
            self.test_block_distribution_strategies()
            self.test_zone_and_parking_determination()
            self.test_edge_cases()
            self.test_full_integration()
            
            print("\\n" + "=" * 65)
            print("🎉 ALL TESTS PASSED! 🎉")
            print("Enhanced Scenario Generator is fully functional!")
            print("=" * 65)
            
        except AssertionError as e:
            print(f"\\n❌ TEST FAILED: {e}")
            raise
        except Exception as e:
            print(f"\\n💥 UNEXPECTED ERROR: {e}")
            raise

def run_performance_benchmark():
    """Run performance benchmarks"""
    print("\\nRunning Performance Benchmarks...")
    print("-" * 40)
    
    simulator = ScoringSimulator()
    generator = ScenarioGenerator(simulator)
    
    import time
    
    # Benchmark time-based strategy generation
    start_time = time.time()
    for _ in range(100):
        params = ScenarioParameters(
            skill_level=SkillLevel.INTERMEDIATE,
            strategy_type=StrategyType.ALL_OFFENSE,
            robot1_role=RobotRole.SCORER,
            robot2_role=RobotRole.SCORER,
            robot1_capabilities=generator.capability_profiles[SkillLevel.INTERMEDIATE],
            robot2_capabilities=generator.capability_profiles[SkillLevel.INTERMEDIATE],
            field_position="center",
            cooperation_efficiency=0.8
        )
        strategy = generator.generate_time_based_strategy("Test", params)
    
    strategy_time = time.time() - start_time
    print(f"Strategy Generation: {strategy_time:.3f}s for 100 strategies ({strategy_time/100*1000:.2f}ms each)")
    
    # Benchmark scenario matrix generation (small sample)
    start_time = time.time()
    sample_scenarios = []
    for skill in [SkillLevel.BEGINNER, SkillLevel.ADVANCED]:
        for strategy in [StrategyType.ALL_OFFENSE, StrategyType.ZONE_CONTROL]:
            params = generator._create_scenario_parameters(skill, strategy, "Test")
            test_strategy = generator.generate_time_based_strategy("Test", params)
            sample_scenarios.append(test_strategy)
    
    matrix_time = time.time() - start_time
    print(f"Scenario Matrix: {matrix_time:.3f}s for {len(sample_scenarios)} scenarios")
    
    # Benchmark DataFrame generation
    start_time = time.time()
    time_df = generator.generate_time_analysis_scenarios()
    df_time = time.time() - start_time
    print(f"DataFrame Generation: {df_time:.3f}s for {len(time_df)} time scenarios")
    
    print("Performance benchmarks completed!")

def main():
    """Main entry point for CLI"""
    # Run comprehensive tests
    tester = TestEnhancedScenarioGenerator()
    tester.run_all_tests()
    
    # Run performance benchmarks
    run_performance_benchmark()


if __name__ == "__main__":
    main()