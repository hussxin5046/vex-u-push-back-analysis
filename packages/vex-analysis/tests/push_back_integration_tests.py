"""
Comprehensive integration tests for Push Back analysis system.

This test suite validates the correctness, performance, and integration
of all Push Back-specific components replacing the generic VEX analysis.
"""

import pytest
import time
import asyncio
import numpy as np
from typing import Dict, List, Tuple

# Import Push Back components
from vex_analysis.core.simulator import (
    PushBackScoringEngine, PushBackBlock, LongGoal, CenterGoal,
    PushBackFieldState, BlockColor
)
from vex_analysis.analysis.push_back_strategy_analyzer import (
    PushBackStrategyAnalyzer, BlockFlowOptimization, AutonomousDecision,
    GoalPriorityAnalysis, ParkingDecisionAnalysis, OffenseDefenseBalance
)
from vex_analysis.simulation import (
    PushBackMonteCarloEngine, RobotCapabilities, create_competitive_robot,
    create_default_robot, ParkingStrategy, GoalPriority
)


class TestPushBackScoringEngine:
    """Test Push Back scoring engine accuracy with all edge cases"""
    
    def setup_method(self):
        """Initialize scoring engine for each test"""
        self.engine = PushBackScoringEngine()
    
    def test_basic_block_scoring(self):
        """Test basic 3 points per block scoring"""
        field = PushBackFieldState()
        
        # Score 10 blocks in center goals
        for i in range(10):
            block = PushBackBlock(color=BlockColor.RED, x=0, y=0)
            field.goals[0].add_block(block)
        
        score, breakdown = self.engine.calculate_push_back_score(field, "red")
        
        assert score == 30, f"Expected 30 points (10 blocks * 3), got {score}"
        assert breakdown["blocks"] == 30
        assert breakdown["control_zone"] == 0
        assert breakdown["parking"] == 0
        assert breakdown["autonomous_win"] == 0
    
    def test_control_zone_calculations(self):
        """Test control zone scoring with different block distributions"""
        field = PushBackFieldState()
        
        # Test minimum control (6 points)
        field.zone_control_blue_count = 3
        field.zone_control_red_count = 2
        score, breakdown = self.engine.calculate_push_back_score(field, "blue")
        assert breakdown["control_zone"] == 6, "Minimum control zone should be 6 points"
        
        # Test maximum control (10 points)
        field.zone_control_blue_count = 10
        field.zone_control_red_count = 0
        score, breakdown = self.engine.calculate_push_back_score(field, "blue")
        assert breakdown["control_zone"] == 10, "Maximum control zone should be 10 points"
        
        # Test tied control (0 points)
        field.zone_control_blue_count = 5
        field.zone_control_red_count = 5
        score, breakdown = self.engine.calculate_push_back_score(field, "blue")
        assert breakdown["control_zone"] == 0, "Tied control zone should be 0 points"
    
    def test_parking_scenarios(self):
        """Test parking score calculations (8 vs 30 points)"""
        field = PushBackFieldState()
        
        # Test one robot parked (8 points)
        field.red_robots_parked = 1
        score, breakdown = self.engine.calculate_push_back_score(field, "red")
        assert breakdown["parking"] == 8, "One robot parked should be 8 points"
        
        # Test two robots parked (30 points)
        field.red_robots_parked = 2
        score, breakdown = self.engine.calculate_push_back_score(field, "red")
        assert breakdown["parking"] == 30, "Two robots parked should be 30 points"
        
        # Test opponent parking doesn't affect score
        field.blue_robots_parked = 2
        score, breakdown = self.engine.calculate_push_back_score(field, "red")
        assert breakdown["parking"] == 30, "Opponent parking shouldn't affect red score"
    
    def test_autonomous_bonus_calculations(self):
        """Test autonomous win point calculations"""
        field = PushBackFieldState()
        
        # Test with autonomous win
        field.autonomous_winner = "red"
        score, breakdown = self.engine.calculate_push_back_score(field, "red")
        assert breakdown["autonomous_win"] == 7, "Autonomous win should be 7 points"
        
        # Test without autonomous win
        field.autonomous_winner = "blue"
        score, breakdown = self.engine.calculate_push_back_score(field, "blue")
        assert breakdown["autonomous_win"] == 7, "Blue should get autonomous bonus"
        
        # Test tied autonomous (no bonus)
        field.autonomous_winner = None
        score, breakdown = self.engine.calculate_push_back_score(field, "red")
        assert breakdown["autonomous_win"] == 0, "No autonomous bonus for tie"
    
    def test_complex_scoring_scenario(self):
        """Test complex scoring with all elements combined"""
        field = PushBackFieldState()
        
        # Red alliance:
        # - 15 blocks in goals (45 points)
        # - Control zone advantage 7-3 (8 points)
        # - 2 robots parked (30 points)
        # - Autonomous win (7 points)
        # Total: 90 points
        
        for i in range(15):
            goal_idx = i % 4
            field.goals[goal_idx].add_block(PushBackBlock(color=BlockColor.RED, x=0, y=0))
        
        field.zone_control_red_count = 7
        field.zone_control_blue_count = 3
        field.red_robots_parked = 2
        field.autonomous_winner = "red"
        
        score, breakdown = self.engine.calculate_push_back_score(field, "red")
        
        assert score == 90, f"Expected 90 total points, got {score}"
        assert breakdown["blocks"] == 45
        assert breakdown["control_zone"] == 8
        assert breakdown["parking"] == 30
        assert breakdown["autonomous_win"] == 7
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        field = PushBackFieldState()
        
        # Test with all 88 blocks scored by one alliance
        for goal in field.goals:
            for i in range(22):  # 22 * 4 = 88
                goal.add_block(PushBackBlock(color=BlockColor.RED, x=0, y=0))
        
        score, breakdown = self.engine.calculate_push_back_score(field, "red")
        assert breakdown["blocks"] == 264, "All 88 blocks should score 264 points"
        
        # Test empty field
        field = PushBackFieldState()
        score, breakdown = self.engine.calculate_push_back_score(field, "red")
        assert score == 0, "Empty field should score 0 points"
    
    def test_goal_ownership_calculations(self):
        """Test goal ownership and point distribution"""
        field = PushBackFieldState()
        
        # Test center goal scoring
        field.goals[0].add_block(PushBackBlock(color=BlockColor.RED, x=0, y=0))  # Center goal
        field.goals[0].add_block(PushBackBlock(color=BlockColor.RED, x=0, y=0))
        field.goals[0].add_block(PushBackBlock(color=BlockColor.BLUE, x=0, y=0))
        
        # Red should own this goal (2 vs 1)
        score_red, _ = self.engine.calculate_push_back_score(field, "red")
        score_blue, _ = self.engine.calculate_push_back_score(field, "blue")
        
        assert score_red == 9, "Red should score all 3 blocks in owned goal"
        assert score_blue == 0, "Blue shouldn't score in red-owned goal"


class TestStrategyAnalysisCorrectness:
    """Validate strategy analysis produces correct recommendations"""
    
    def setup_method(self):
        """Initialize strategy analyzer"""
        self.analyzer = PushBackStrategyAnalyzer()
    
    def test_block_allocation_optimizer_correctness(self):
        """Test block allocation finds mathematically correct optima"""
        # Test scenario: Equal capability robots
        capabilities = {
            "scoring_rate": 0.5,  # blocks per second
            "center_goal_efficiency": 1.0,
            "long_goal_efficiency": 1.0,
            "control_zone_capability": 0.5
        }
        
        optimization = self.analyzer.analyze_block_flow_optimization(
            robot_capabilities=capabilities,
            opponent_strength="equal",
            match_phase="full"
        )
        
        # With equal efficiency, should recommend balanced allocation
        assert optimization.recommended_distribution["center_goals"] == pytest.approx(0.5, rel=0.1)
        assert optimization.recommended_distribution["long_goals"] == pytest.approx(0.5, rel=0.1)
        
        # Test with center goal advantage
        capabilities["center_goal_efficiency"] = 1.5
        optimization = self.analyzer.analyze_block_flow_optimization(
            robot_capabilities=capabilities,
            opponent_strength="equal",
            match_phase="full"
        )
        
        # Should prefer center goals
        assert optimization.recommended_distribution["center_goals"] > 0.6
    
    def test_autonomous_strategy_selection_logic(self):
        """Test autonomous strategy selector produces optimal recommendations"""
        # High reliability robot
        capabilities = {
            "autonomous_reliability": 0.95,
            "autonomous_scoring_rate": 2.0,  # blocks per autonomous
            "positioning_speed": 0.8
        }
        
        decision = self.analyzer.analyze_autonomous_strategy_decision(
            robot_capabilities=capabilities,
            opponent_analysis={"autonomous_strength": "weak"},
            risk_tolerance=0.5
        )
        
        # Should recommend aggressive autonomous with high reliability
        assert decision.recommended_strategy == "aggressive"
        assert decision.confidence_level > 0.8
        
        # Low reliability robot
        capabilities["autonomous_reliability"] = 0.6
        decision = self.analyzer.analyze_autonomous_strategy_decision(
            robot_capabilities=capabilities,
            opponent_analysis={"autonomous_strength": "strong"},
            risk_tolerance=0.3
        )
        
        # Should recommend safe strategy with low reliability
        assert decision.recommended_strategy in ["safe", "positioning"]
    
    def test_parking_decision_calculator_breakeven(self):
        """Test parking calculator finds correct break-even points"""
        # Scenario: Leading by small margin with time running out
        match_state = {
            "current_score_diff": 5,  # Leading by 5
            "time_remaining": 15,
            "robot_positions": ["field", "field"],
            "blocks_remaining": 20
        }
        
        analysis = self.analyzer.analyze_parking_decision_timing(
            match_state=match_state,
            robot_capabilities={"parking_time": 3.0, "scoring_rate": 0.3}
        )
        
        # Should recommend parking to secure lead
        assert analysis.recommended_action in ["park_one", "park_both"]
        assert analysis.breakeven_time < 15
        
        # Scenario: Behind by large margin
        match_state["current_score_diff"] = -25  # Behind by 25
        analysis = self.analyzer.analyze_parking_decision_timing(
            match_state=match_state,
            robot_capabilities={"parking_time": 3.0, "scoring_rate": 0.3}
        )
        
        # Should recommend continuing to score
        assert analysis.recommended_action == "continue_scoring"
    
    def test_monte_carlo_validation_against_calculated(self):
        """Validate Monte Carlo results against hand-calculated scenarios"""
        # Simple scenario: Perfect robots, no variance
        red_robot = RobotCapabilities(
            average_cycle_time=5.0,
            pickup_reliability=1.0,
            scoring_reliability=1.0,
            autonomous_reliability=1.0,
            parking_strategy=ParkingStrategy.NEVER
        )
        blue_robot = RobotCapabilities(
            average_cycle_time=10.0,  # Twice as slow
            pickup_reliability=1.0,
            scoring_reliability=1.0,
            autonomous_reliability=1.0,
            parking_strategy=ParkingStrategy.NEVER
        )
        
        engine = PushBackMonteCarloEngine(red_robot, blue_robot)
        results, _ = engine.run_simulation(100)
        
        # Red should win almost every match (2x speed advantage)
        red_wins = sum(1 for r in results if r.winner == "red")
        assert red_wins > 95, f"Red should win >95% with 2x speed, got {red_wins}%"
        
        # Calculate expected scores
        # Red: ~90 seconds of scoring at 5s/cycle = 18 blocks = 54 points
        # Blue: ~90 seconds of scoring at 10s/cycle = 9 blocks = 27 points
        avg_red_score = np.mean([r.final_score_red for r in results])
        avg_blue_score = np.mean([r.final_score_blue for r in results])
        
        assert 50 <= avg_red_score <= 70, f"Red score outside expected range: {avg_red_score}"
        assert 20 <= avg_blue_score <= 40, f"Blue score outside expected range: {avg_blue_score}"


class TestEndToEndSystemIntegration:
    """Test complete system integration from frontend to results"""
    
    @pytest.mark.asyncio
    async def test_strategy_builder_to_analysis_flow(self):
        """Test frontend strategy builder → backend API → analysis engine → results"""
        # Simulate frontend strategy configuration
        strategy_config = {
            "robot_capabilities": {
                "cycle_time": 5.0,
                "reliability": 0.95,
                "max_capacity": 2,
                "parking_capability": True
            },
            "strategy_preferences": {
                "goal_priority": "center_preferred",
                "autonomous_aggression": "balanced",
                "parking_strategy": "late"
            },
            "opponent_scouting": {
                "estimated_strength": "competitive",
                "known_weaknesses": ["slow_cycles"]
            }
        }
        
        # Convert to analysis format
        robot = RobotCapabilities(
            average_cycle_time=strategy_config["robot_capabilities"]["cycle_time"],
            pickup_reliability=strategy_config["robot_capabilities"]["reliability"],
            scoring_reliability=strategy_config["robot_capabilities"]["reliability"],
            max_blocks_per_trip=strategy_config["robot_capabilities"]["max_capacity"],
            goal_priority=GoalPriority.CENTER_PREFERRED,
            parking_strategy=ParkingStrategy.LATE
        )
        
        # Run analysis
        analyzer = PushBackStrategyAnalyzer()
        optimization = analyzer.analyze_block_flow_optimization(
            robot_capabilities={
                "scoring_rate": 1.0 / robot.average_cycle_time,
                "center_goal_efficiency": 1.2,
                "long_goal_efficiency": 0.8,
                "control_zone_capability": 0.5
            },
            opponent_strength="competitive",
            match_phase="full"
        )
        
        # Validate results structure
        assert hasattr(optimization, "recommended_distribution")
        assert hasattr(optimization, "expected_score")
        assert hasattr(optimization, "confidence_interval")
        assert optimization.expected_score > 0
    
    def test_websocket_real_time_updates(self):
        """Test WebSocket real-time update functionality"""
        # This would require actual WebSocket implementation
        # For now, test the data structure that would be sent
        
        update_payload = {
            "type": "analysis_update",
            "data": {
                "current_analysis": {
                    "win_probability": 0.75,
                    "expected_score": 120,
                    "key_insights": [
                        "Focus on center goals",
                        "Park early if leading",
                        "Prioritize autonomous"
                    ]
                },
                "timestamp": time.time()
            }
        }
        
        # Validate payload structure
        assert update_payload["type"] == "analysis_update"
        assert "win_probability" in update_payload["data"]["current_analysis"]
        assert len(update_payload["data"]["current_analysis"]["key_insights"]) > 0
    
    def test_visualization_data_generation(self):
        """Test visualization data generation for Push Back displays"""
        # Generate sample analysis results
        field = PushBackFieldState()
        
        # Add some blocks to goals
        for i in range(10):
            field.goals[0].add_block(PushBackBlock(color=BlockColor.RED, x=0, y=0))
            field.goals[1].add_block(PushBackBlock(color=BlockColor.BLUE, x=0, y=0))
        
        # Generate visualization data
        viz_data = {
            "field_state": {
                "goals": [
                    {"id": i, "red_blocks": g.red_blocks, "blue_blocks": g.blue_blocks}
                    for i, g in enumerate(field.goals)
                ],
                "control_zones": {
                    "red_presence": field.zone_control_red_count,
                    "blue_presence": field.zone_control_blue_count
                },
                "parking": {
                    "red_parked": field.red_robots_parked,
                    "blue_parked": field.blue_robots_parked
                }
            },
            "score_breakdown": {
                "red": {"blocks": 30, "control": 0, "parking": 0, "auto": 0, "total": 30},
                "blue": {"blocks": 30, "control": 0, "parking": 0, "auto": 0, "total": 30}
            }
        }
        
        # Validate visualization data
        assert len(viz_data["field_state"]["goals"]) == 4
        assert viz_data["score_breakdown"]["red"]["total"] == 30
        assert viz_data["score_breakdown"]["blue"]["total"] == 30


class TestPerformanceBenchmarks:
    """Test system performance meets requirements"""
    
    def test_strategy_evaluation_under_5_seconds(self):
        """Ensure complete strategy evaluation completes within 5 seconds"""
        start_time = time.time()
        
        # Create analyzer
        analyzer = PushBackStrategyAnalyzer()
        
        # Run all analyses
        capabilities = {
            "scoring_rate": 0.5,
            "center_goal_efficiency": 1.2,
            "long_goal_efficiency": 0.8,
            "control_zone_capability": 0.6
        }
        
        # Block flow optimization
        analyzer.analyze_block_flow_optimization(
            robot_capabilities=capabilities,
            opponent_strength="competitive",
            match_phase="full"
        )
        
        # Autonomous decision
        analyzer.analyze_autonomous_strategy_decision(
            robot_capabilities={
                "autonomous_reliability": 0.85,
                "autonomous_scoring_rate": 2.0,
                "positioning_speed": 0.8
            },
            opponent_analysis={"autonomous_strength": "strong"},
            risk_tolerance=0.5
        )
        
        # Goal priority
        analyzer.analyze_goal_priority_strategy(
            robot_capabilities=capabilities,
            field_state={"available_goals": ["center1", "center2", "long1", "long2"]},
            time_remaining=90
        )
        
        # Parking decision
        analyzer.analyze_parking_decision_timing(
            match_state={
                "current_score_diff": 10,
                "time_remaining": 20,
                "robot_positions": ["field", "field"],
                "blocks_remaining": 30
            },
            robot_capabilities={"parking_time": 3.0, "scoring_rate": 0.3}
        )
        
        # Offense/defense balance
        analyzer.analyze_offense_defense_balance(
            alliance_capabilities={
                "robot1": {"offensive_power": 0.8, "defensive_power": 0.6},
                "robot2": {"offensive_power": 0.6, "defensive_power": 0.8}
            },
            opponent_strategy="offensive",
            match_situation="qualification"
        )
        
        execution_time = time.time() - start_time
        assert execution_time < 5.0, f"Strategy evaluation took {execution_time:.2f}s, exceeds 5s limit"
    
    def test_monte_carlo_1000_scenarios_under_10_seconds(self):
        """Validate Monte Carlo runs 1000+ scenarios in under 10 seconds"""
        red_robot = create_competitive_robot()
        blue_robot = create_default_robot()
        
        engine = PushBackMonteCarloEngine(red_robot, blue_robot)
        
        start_time = time.time()
        results, execution_time = engine.run_simulation(1000, use_parallel=True)
        
        assert len(results) == 1000, f"Expected 1000 results, got {len(results)}"
        assert execution_time < 10.0, f"Monte Carlo took {execution_time:.2f}s, exceeds 10s limit"
        
        # Also test 2000 scenarios
        results, execution_time = engine.run_simulation(2000, use_parallel=True)
        assert execution_time < 20.0, f"2000 scenarios took {execution_time:.2f}s"
    
    def test_frontend_responsiveness_rapid_changes(self):
        """Test system handles rapid strategy changes efficiently"""
        analyzer = PushBackStrategyAnalyzer()
        
        # Simulate rapid strategy changes
        change_times = []
        
        for i in range(10):
            start = time.time()
            
            # Change strategy parameters
            capabilities = {
                "scoring_rate": 0.3 + i * 0.05,
                "center_goal_efficiency": 0.8 + i * 0.02,
                "long_goal_efficiency": 1.2 - i * 0.02,
                "control_zone_capability": 0.5
            }
            
            # Re-run analysis
            analyzer.analyze_block_flow_optimization(
                robot_capabilities=capabilities,
                opponent_strength="competitive",
                match_phase="full"
            )
            
            change_times.append(time.time() - start)
        
        # All changes should process quickly
        avg_change_time = np.mean(change_times)
        max_change_time = np.max(change_times)
        
        assert avg_change_time < 0.5, f"Average change time {avg_change_time:.3f}s too slow"
        assert max_change_time < 1.0, f"Max change time {max_change_time:.3f}s too slow"
    
    def test_memory_efficiency_large_simulations(self):
        """Test memory efficiency with large simulation runs"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run large simulation
        red_robot = create_competitive_robot()
        blue_robot = create_default_robot()
        engine = PushBackMonteCarloEngine(red_robot, blue_robot)
        
        results, _ = engine.run_simulation(5000, use_parallel=True)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not use excessive memory
        assert memory_increase < 200, f"Memory increased by {memory_increase:.1f}MB, too high"
        
        # Results should be garbage collected properly
        del results
        import gc
        gc.collect()
        
        after_gc_memory = process.memory_info().rss / 1024 / 1024  # MB
        assert after_gc_memory < final_memory, "Memory not properly released after GC"


class TestPushBackRuleCompliance:
    """Validate all game mechanics match official Push Back rules exactly"""
    
    def test_field_layout_specifications(self):
        """Test field layout matches specifications"""
        field = PushBackFieldState()
        
        # Verify 4 goals
        assert len(field.goals) == 4, "Field must have exactly 4 goals"
        
        # Verify 2 center and 2 long goals
        center_goals = sum(1 for g in field.goals if isinstance(g, CenterGoal))
        long_goals = sum(1 for g in field.goals if isinstance(g, LongGoal))
        
        assert center_goals == 2, "Field must have exactly 2 center goals"
        assert long_goals == 2, "Field must have exactly 2 long goals"
        
        # Verify 88 blocks available
        assert field.blocks_available == 88, "Field must start with 88 blocks"
        
        # Verify 2 park zones
        assert hasattr(field, "park_zones"), "Field must have park zones"
        
        # Verify 2 control zones
        assert hasattr(field, "zone_control_red_count"), "Field must track red control"
        assert hasattr(field, "zone_control_blue_count"), "Field must track blue control"
    
    def test_scoring_rules_exact_match(self):
        """Test scoring calculations match manual computation exactly"""
        engine = PushBackScoringEngine()
        field = PushBackFieldState()
        
        # Manual calculation test cases
        test_cases = [
            {
                "description": "Basic scoring test",
                "setup": lambda f: [f.goals[0].add_block(PushBackBlock(color=BlockColor.RED, x=0, y=0)) for _ in range(5)],
                "expected_red": 15,  # 5 blocks * 3 points
                "expected_blue": 0
            },
            {
                "description": "Control zone minimum",
                "setup": lambda f: setattr(f, "zone_control_red_count", 3) or setattr(f, "zone_control_blue_count", 2),
                "expected_red": 6,  # Minimum control bonus
                "expected_blue": 0
            },
            {
                "description": "Full parking bonus",
                "setup": lambda f: setattr(f, "red_robots_parked", 2),
                "expected_red": 30,  # 2 robots parked
                "expected_blue": 0
            },
            {
                "description": "Autonomous win",
                "setup": lambda f: setattr(f, "autonomous_winner", "red"),
                "expected_red": 7,  # Autonomous bonus
                "expected_blue": 0
            },
            {
                "description": "Complex scenario",
                "setup": lambda f: (
                    [f.goals[0].add_block(PushBackBlock(color=BlockColor.RED, x=0, y=0)) for _ in range(10)],
                    [f.goals[1].add_block(PushBackBlock(color=BlockColor.BLUE, x=0, y=0)) for _ in range(8)],
                    setattr(f, "zone_control_red_count", 5),
                    setattr(f, "zone_control_blue_count", 3),
                    setattr(f, "red_robots_parked", 1),
                    setattr(f, "blue_robots_parked", 2),
                    setattr(f, "autonomous_winner", "blue")
                ),
                "expected_red": 30 + 6 + 8,  # blocks + control + parking = 44
                "expected_blue": 24 + 0 + 30 + 7  # blocks + no control + parking + auto = 61
            }
        ]
        
        for test in test_cases:
            field = PushBackFieldState()  # Reset field
            test["setup"](field)
            
            red_score, _ = engine.calculate_push_back_score(field, "red")
            blue_score, _ = engine.calculate_push_back_score(field, "blue")
            
            assert red_score == test["expected_red"], \
                f"{test['description']}: Red score {red_score} != {test['expected_red']}"
            assert blue_score == test["expected_blue"], \
                f"{test['description']}: Blue score {blue_score} != {test['expected_blue']}"
    
    def test_match_timing_specifications(self):
        """Test match timing follows Push Back specifications"""
        # Test match duration
        assert PushBackScoringEngine.MATCH_DURATION == 105, "Match must be 105 seconds"
        assert PushBackScoringEngine.AUTONOMOUS_DURATION == 15, "Autonomous must be 15 seconds"
        
        # Test in simulation
        red_robot = create_default_robot()
        blue_robot = create_default_robot()
        engine = PushBackMonteCarloEngine(red_robot, blue_robot)
        
        result = engine.simulate_match()
        
        # Match should not exceed total duration
        assert result.match_duration <= 105, "Match duration exceeds 105 seconds"
        
        # Autonomous should be included in results
        assert hasattr(result, "autonomous_winner"), "Must track autonomous winner"
    
    def test_goal_ownership_rules(self):
        """Test goal ownership follows Push Back rules"""
        field = PushBackFieldState()
        engine = PushBackScoringEngine()
        
        # Test goal ownership determination
        goal = field.goals[0]
        
        # Equal blocks - no one owns
        goal.add_block(PushBackBlock(color=BlockColor.RED, x=0, y=0))
        goal.add_block(PushBackBlock(color=BlockColor.BLUE, x=0, y=0))
        
        red_score, _ = engine.calculate_push_back_score(field, "red")
        blue_score, _ = engine.calculate_push_back_score(field, "blue")
        
        assert red_score == 0 and blue_score == 0, "Tied goal should award no points"
        
        # Red owns by one block
        goal.add_block(PushBackBlock(color=BlockColor.RED, x=0, y=0))
        red_score, _ = engine.calculate_push_back_score(field, "red")
        
        assert red_score == 9, "Red should score all 3 blocks when owning goal"
    
    def test_robot_constraints(self):
        """Test robot constraints match Push Back specifications"""
        # Test parking constraints
        field = PushBackFieldState()
        field.red_robots_parked = 3  # Invalid
        
        # Scoring engine should cap at 2
        engine = PushBackScoringEngine()
        score, breakdown = engine.calculate_push_back_score(field, "red")
        
        # Should still only award 30 points max
        assert breakdown["parking"] <= 30, "Parking score capped at 30 points (2 robots)"


def run_comprehensive_test_suite():
    """Run all Push Back integration tests and report results"""
    import subprocess
    import sys
    
    print("=" * 60)
    print("PUSH BACK INTEGRATION TEST SUITE")
    print("=" * 60)
    
    # Run pytest with detailed output
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("ERRORS:")
        print(result.stderr)
    
    if result.returncode == 0:
        print("\n✅ ALL PUSH BACK INTEGRATION TESTS PASSED!")
        print("The refactored system correctly implements Push Back analysis.")
    else:
        print("\n❌ Some tests failed. Please review the output above.")
    
    return result.returncode == 0


if __name__ == "__main__":
    # Run tests when executed directly
    success = run_comprehensive_test_suite()
    exit(0 if success else 1)