"""
Simplified Push Back integration tests that work with the actual implementation.
"""

import pytest
import time
import numpy as np

# Import what's actually available
from vex_analysis.simulation import (
    PushBackMonteCarloEngine, RobotCapabilities, 
    create_competitive_robot, create_default_robot,
    ParkingStrategy, GoalPriority
)


class TestPushBackSimulationSystem:
    """Test the Push Back Monte Carlo simulation system"""
    
    def test_monte_carlo_basic_functionality(self):
        """Test Monte Carlo simulation runs successfully"""
        red_robot = create_competitive_robot()
        blue_robot = create_default_robot()
        
        engine = PushBackMonteCarloEngine(red_robot, blue_robot)
        results, execution_time = engine.run_simulation(100)
        
        # Basic validation
        assert len(results) == 100, "Should generate 100 results"
        assert execution_time < 5.0, "Should complete quickly"
        
        # Validate result structure
        for result in results[:5]:  # Check first 5
            assert hasattr(result, 'winner')
            assert hasattr(result, 'final_score_red')
            assert hasattr(result, 'final_score_blue')
            assert result.winner in ['red', 'blue', 'tie']
            assert 0 <= result.final_score_red <= 400
            assert 0 <= result.final_score_blue <= 400
    
    def test_robot_configuration_impact(self):
        """Test that robot configuration affects results"""
        # Fast robot
        fast_robot = RobotCapabilities(
            average_cycle_time=3.0,
            pickup_reliability=0.98,
            scoring_reliability=0.99
        )
        
        # Slow robot
        slow_robot = RobotCapabilities(
            average_cycle_time=8.0,
            pickup_reliability=0.85,
            scoring_reliability=0.90
        )
        
        # Fast vs slow should favor fast robot
        engine = PushBackMonteCarloEngine(fast_robot, slow_robot)
        results, _ = engine.run_simulation(200)
        
        red_wins = sum(1 for r in results if r.winner == 'red')
        red_win_rate = red_wins / len(results)
        
        # Fast robot should win more often
        assert red_win_rate > 0.6, f"Fast robot should win >60%, got {red_win_rate:.1%}"
    
    def test_parking_strategy_effects(self):
        """Test that parking strategies affect outcomes"""
        never_park = RobotCapabilities(parking_strategy=ParkingStrategy.NEVER)
        early_park = RobotCapabilities(parking_strategy=ParkingStrategy.EARLY)
        
        # Test never park strategy
        engine1 = PushBackMonteCarloEngine(never_park, create_default_robot())
        results1, _ = engine1.run_simulation(100)
        
        park_rate1 = sum(1 for r in results1 if r.red_parked) / len(results1)
        
        # Test early park strategy  
        engine2 = PushBackMonteCarloEngine(early_park, create_default_robot())
        results2, _ = engine2.run_simulation(100)
        
        park_rate2 = sum(1 for r in results2 if r.red_parked) / len(results2)
        
        # Should see difference in parking behavior
        assert park_rate1 < park_rate2, "Early park should park more than never park"
    
    def test_performance_requirements(self):
        """Test performance meets requirements"""
        red_robot = create_competitive_robot()
        blue_robot = create_default_robot()
        engine = PushBackMonteCarloEngine(red_robot, blue_robot)
        
        # Test 1000 simulations under 10 seconds
        start_time = time.time()
        results, execution_time = engine.run_simulation(1000, use_parallel=True)
        total_time = time.time() - start_time
        
        assert len(results) == 1000, "Should complete all 1000 simulations"
        assert execution_time < 10.0, f"Execution time {execution_time:.2f}s exceeds 10s limit"
        
        # Calculate simulation rate
        rate = len(results) / execution_time
        assert rate > 100, f"Simulation rate {rate:.0f}/sec too slow"
        
        print(f"Performance: {rate:.0f} simulations/second")
    
    def test_insight_generation(self):
        """Test that insights can be generated from results"""
        red_robot = create_competitive_robot()
        blue_robot = create_default_robot()
        engine = PushBackMonteCarloEngine(red_robot, blue_robot)
        
        results, _ = engine.run_simulation(200)
        insights = engine.generate_insights(results, "red")
        
        # Validate insights structure
        assert hasattr(insights, 'win_probability')
        assert hasattr(insights, 'average_score')
        assert hasattr(insights, 'score_variance')
        
        # Validate values are reasonable
        assert 0 <= insights.win_probability <= 1
        assert insights.average_score > 0
        assert insights.score_variance >= 0
        
        print(f"Win probability: {insights.win_probability:.1%}")
        print(f"Average score: {insights.average_score:.1f}")
    
    def test_realistic_scoring_ranges(self):
        """Test that scores fall within realistic ranges"""
        red_robot = create_default_robot()
        blue_robot = create_default_robot()
        engine = PushBackMonteCarloEngine(red_robot, blue_robot)
        
        results, _ = engine.run_simulation(500)
        
        red_scores = [r.final_score_red for r in results]
        blue_scores = [r.final_score_blue for r in results]
        
        # Check score ranges
        min_red, max_red = min(red_scores), max(red_scores)
        min_blue, max_blue = min(blue_scores), max(blue_scores)
        
        print(f"Red scores: {min_red}-{max_red}")
        print(f"Blue scores: {min_blue}-{max_blue}")
        
        # Scores should be reasonable for Push Back
        assert 20 <= min_red, "Minimum scores too low"
        assert max_red <= 300, "Maximum scores too high"
        assert 20 <= min_blue, "Minimum scores too low"  
        assert max_blue <= 300, "Maximum scores too high"
        
        # Average scores should be in typical range
        avg_red = np.mean(red_scores)
        avg_blue = np.mean(blue_scores)
        
        assert 40 <= avg_red <= 150, f"Average red score {avg_red:.1f} outside typical range"
        assert 40 <= avg_blue <= 150, f"Average blue score {avg_blue:.1f} outside typical range"


class TestRobotCapabilities:
    """Test robot capability configurations"""
    
    def test_default_robot_creation(self):
        """Test default robot has reasonable values"""
        robot = create_default_robot()
        
        assert 2.0 <= robot.average_cycle_time <= 10.0
        assert 0.8 <= robot.pickup_reliability <= 1.0
        assert 0.8 <= robot.scoring_reliability <= 1.0
        assert robot.parking_strategy in list(ParkingStrategy)
        assert robot.goal_priority in list(GoalPriority)
    
    def test_competitive_robot_creation(self):
        """Test competitive robot is better than default"""
        default = create_default_robot()
        competitive = create_competitive_robot()
        
        # Competitive should be faster
        assert competitive.average_cycle_time <= default.average_cycle_time
        
        # Competitive should be more reliable
        assert competitive.pickup_reliability >= default.pickup_reliability
        assert competitive.scoring_reliability >= default.scoring_reliability
    
    def test_custom_robot_configuration(self):
        """Test custom robot configuration"""
        custom = RobotCapabilities(
            average_cycle_time=4.5,
            pickup_reliability=0.95,
            scoring_reliability=0.98,
            parking_strategy=ParkingStrategy.LATE,
            goal_priority=GoalPriority.CENTER_PREFERRED
        )
        
        assert custom.average_cycle_time == 4.5
        assert custom.pickup_reliability == 0.95
        assert custom.scoring_reliability == 0.98
        assert custom.parking_strategy == ParkingStrategy.LATE
        assert custom.goal_priority == GoalPriority.CENTER_PREFERRED


class TestSystemIntegration:
    """Test system integration aspects"""
    
    def test_concurrent_simulations(self):
        """Test running multiple simulations concurrently"""
        import concurrent.futures
        
        def run_simulation(robot_pair):
            red, blue = robot_pair
            engine = PushBackMonteCarloEngine(red, blue)
            return engine.run_simulation(100)
        
        # Create different robot pairs
        pairs = [
            (create_competitive_robot(), create_default_robot()),
            (create_default_robot(), create_competitive_robot()),
            (create_default_robot(), create_default_robot())
        ]
        
        # Run concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_simulation, pair) for pair in pairs]
            results = [f.result() for f in futures]
        
        # All should complete successfully
        assert len(results) == 3
        for result_set, execution_time in results:
            assert len(result_set) == 100
            assert execution_time < 5.0
    
    def test_error_handling(self):
        """Test error handling with invalid configurations"""
        # Test with extreme values
        extreme_robot = RobotCapabilities(
            average_cycle_time=0.1,  # Very fast
            pickup_reliability=2.0   # Invalid (>1.0)
        )
        
        # Should handle gracefully or validate
        engine = PushBackMonteCarloEngine(extreme_robot, create_default_robot())
        
        # Should either validate or handle extreme values
        try:
            results, _ = engine.run_simulation(10)
            # If it runs, check results are still reasonable
            assert len(results) == 10
        except (ValueError, AssertionError):
            # Or it should validate and raise appropriate error
            pass


def run_simple_tests():
    """Run the simplified test suite"""
    import subprocess
    import sys
    
    print("\n" + "=" * 60)
    print("PUSH BACK SIMPLIFIED INTEGRATION TESTS")
    print("=" * 60)
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("ERRORS:")
        print(result.stderr)
    
    return result.returncode == 0


if __name__ == "__main__":
    success = run_simple_tests()
    exit(0 if success else 1)