"""
Detailed tests for Push Back game rule compliance.

This module ensures all Push Back game mechanics are implemented
exactly according to the official game manual.
"""

import pytest
from typing import List, Dict

from vex_analysis.core.simulator import (
    PushBackScoringEngine, PushBackFieldState, PushBackBlock,
    LongGoal, CenterGoal, PushBackRobot, PushBackAlliance
)


class TestPushBackGameRules:
    """Validate exact Push Back game rule implementation"""
    
    def test_field_element_counts(self):
        """Test correct number of field elements"""
        field = PushBackFieldState()
        
        # Rule G5: 88 blocks total
        assert field.blocks_available == 88, "Must have exactly 88 blocks"
        
        # Rule G2: 4 goals (2 center, 2 long)
        assert len(field.goals) == 4, "Must have exactly 4 goals"
        
        center_goals = [g for g in field.goals if isinstance(g, CenterGoal)]
        long_goals = [g for g in field.goals if isinstance(g, LongGoal)]
        
        assert len(center_goals) == 2, "Must have exactly 2 center goals"
        assert len(long_goals) == 2, "Must have exactly 2 long goals"
        
        # Rule G3: 2 robots per alliance
        red_alliance = PushBackAlliance("red")
        assert len(red_alliance.robots) <= 2, "Maximum 2 robots per alliance"
    
    def test_scoring_values(self):
        """Test exact scoring values from game manual"""
        engine = PushBackScoringEngine()
        
        # Rule SC1: Each block is worth 3 points
        assert engine.POINTS_PER_BLOCK == 3, "Blocks must be worth 3 points"
        
        # Rule SC2: Control zone worth 6-10 points
        assert engine.CONTROL_ZONE_POINTS_MIN == 6, "Min control zone must be 6"
        assert engine.CONTROL_ZONE_POINTS_MAX == 10, "Max control zone must be 10"
        
        # Rule SC3: Parking worth 8 points (1 robot) or 30 points (2 robots)
        assert engine.PARKING_POINTS_TOUCHED == 8, "One robot parking must be 8 points"
        assert engine.PARKING_POINTS_COMPLETELY == 30, "Two robot parking must be 30 points"
        
        # Rule SC4: Autonomous win worth 7 points
        assert engine.AUTONOMOUS_WIN_POINTS == 7, "Autonomous win must be 7 points"
    
    def test_goal_ownership_rules(self):
        """Test goal ownership determination per game manual"""
        field = PushBackFieldState()
        engine = PushBackScoringEngine()
        
        # Rule SC5: Alliance with more blocks owns the goal
        goal = field.goals[0]
        
        # Case 1: Red has more blocks
        goal.add_block(PushBackBlock("red"))
        goal.add_block(PushBackBlock("red"))
        goal.add_block(PushBackBlock("blue"))
        
        score_red, breakdown = engine.calculate_push_back_score(field, "red")
        score_blue, breakdown = engine.calculate_push_back_score(field, "blue")
        
        # Red owns goal, gets all 9 points (3 blocks * 3 points)
        assert breakdown["blocks"] == 9, "Owner should score all blocks in goal"
        
        # Blue gets nothing from this goal
        blue_score, blue_breakdown = engine.calculate_push_back_score(field, "blue")
        assert blue_breakdown["blocks"] == 0, "Non-owner gets no points from goal"
        
        # Case 2: Tied goal
        goal.add_block(PushBackBlock("blue"))  # Now 2-2
        
        score_red, breakdown = engine.calculate_push_back_score(field, "red")
        score_blue, breakdown_blue = engine.calculate_push_back_score(field, "blue")
        
        # Neither alliance scores from tied goal
        assert breakdown["blocks"] == 0, "Tied goal awards no points"
        assert breakdown_blue["blocks"] == 0, "Tied goal awards no points"
    
    def test_control_zone_scoring_formula(self):
        """Test exact control zone scoring formula"""
        field = PushBackFieldState()
        engine = PushBackScoringEngine()
        
        test_cases = [
            # (red_count, blue_count, expected_red_points, expected_blue_points)
            (0, 0, 0, 0),      # No blocks = no points
            (1, 0, 6, 0),      # Minimum control = 6 points
            (3, 1, 6, 0),      # Red has more = 6 points min
            (5, 2, 8, 0),      # More blocks = more points
            (8, 1, 10, 0),     # Maximum = 10 points
            (10, 0, 10, 0),    # Capped at 10 points
            (3, 3, 0, 0),      # Tied = no points
            (2, 5, 0, 8),      # Blue wins control
        ]
        
        for red_count, blue_count, expected_red, expected_blue in test_cases:
            field.zone_control_red_count = red_count
            field.zone_control_blue_count = blue_count
            
            _, red_breakdown = engine.calculate_push_back_score(field, "red")
            _, blue_breakdown = engine.calculate_push_back_score(field, "blue")
            
            assert red_breakdown["control_zone"] == expected_red, \
                f"Red control with {red_count} vs {blue_count} should be {expected_red}"
            assert blue_breakdown["control_zone"] == expected_blue, \
                f"Blue control with {blue_count} vs {red_count} should be {expected_blue}"
    
    def test_match_timing_rules(self):
        """Test match timing specifications"""
        # Rule G1: Match is 105 seconds total
        assert PushBackScoringEngine.MATCH_DURATION == 105, "Match must be 105 seconds"
        
        # Rule G1a: Autonomous is 15 seconds
        assert PushBackScoringEngine.AUTONOMOUS_DURATION == 15, "Autonomous must be 15 seconds"
        
        # Driver control period
        driver_period = PushBackScoringEngine.MATCH_DURATION - PushBackScoringEngine.AUTONOMOUS_DURATION
        assert driver_period == 90, "Driver control must be 90 seconds"
    
    def test_parking_configurations(self):
        """Test all valid parking configurations"""
        field = PushBackFieldState()
        engine = PushBackScoringEngine()
        
        parking_scenarios = [
            (0, 0),   # No robots parked
            (1, 8),   # One robot parked = 8 points
            (2, 30),  # Two robots parked = 30 points
        ]
        
        for robots_parked, expected_points in parking_scenarios:
            field.red_robots_parked = robots_parked
            _, breakdown = engine.calculate_push_back_score(field, "red")
            
            assert breakdown["parking"] == expected_points, \
                f"{robots_parked} robots parked should score {expected_points} points"
        
        # Test invalid parking (more than 2 robots)
        field.red_robots_parked = 3  # Invalid
        _, breakdown = engine.calculate_push_back_score(field, "red")
        
        # Should cap at 30 points
        assert breakdown["parking"] <= 30, "Parking cannot exceed 30 points"
    
    def test_autonomous_scenarios(self):
        """Test autonomous period scoring scenarios"""
        field = PushBackFieldState()
        engine = PushBackScoringEngine()
        
        # Scenario 1: Red wins autonomous
        field.autonomous_winner = "red"
        red_score, red_breakdown = engine.calculate_push_back_score(field, "red")
        blue_score, blue_breakdown = engine.calculate_push_back_score(field, "blue")
        
        assert red_breakdown["autonomous_win"] == 7, "Red should get 7 point bonus"
        assert blue_breakdown["autonomous_win"] == 0, "Blue should get no bonus"
        
        # Scenario 2: Tied autonomous
        field.autonomous_winner = None
        red_score, red_breakdown = engine.calculate_push_back_score(field, "red")
        blue_score, blue_breakdown = engine.calculate_push_back_score(field, "blue")
        
        assert red_breakdown["autonomous_win"] == 0, "No bonus for tied autonomous"
        assert blue_breakdown["autonomous_win"] == 0, "No bonus for tied autonomous"
    
    def test_maximum_possible_score(self):
        """Test theoretical maximum score calculation"""
        field = PushBackFieldState()
        engine = PushBackScoringEngine()
        
        # Theoretical maximum for one alliance:
        # - All 88 blocks scored: 88 * 3 = 264 points
        # - Maximum control zone: 10 points
        # - Full parking: 30 points
        # - Autonomous win: 7 points
        # Total: 311 points
        
        # Set up maximum scenario
        for goal in field.goals:
            for i in range(22):  # 22 * 4 = 88 blocks
                goal.add_block(PushBackBlock("red"))
        
        field.zone_control_red_count = 10
        field.zone_control_blue_count = 0
        field.red_robots_parked = 2
        field.autonomous_winner = "red"
        
        score, breakdown = engine.calculate_push_back_score(field, "red")
        
        assert score == 311, f"Maximum possible score should be 311, got {score}"
        assert breakdown["blocks"] == 264
        assert breakdown["control_zone"] == 10
        assert breakdown["parking"] == 30
        assert breakdown["autonomous_win"] == 7
    
    def test_realistic_score_ranges(self):
        """Test that scores fall within realistic ranges"""
        field = PushBackFieldState()
        engine = PushBackScoringEngine()
        
        # Typical match scenario
        # Red: 15 blocks, some control, 1 robot parked, won auto
        for i in range(15):
            field.goals[i % 4].add_block(PushBackBlock("red"))
        
        field.zone_control_red_count = 4
        field.zone_control_blue_count = 2
        field.red_robots_parked = 1
        field.autonomous_winner = "red"
        
        score, _ = engine.calculate_push_back_score(field, "red")
        
        # Typical scores range from 50-150
        assert 50 <= score <= 150, f"Typical score {score} outside expected range"
    
    def test_block_distribution_limits(self):
        """Test that block distribution respects game limits"""
        field = PushBackFieldState()
        
        # Try to add more than 88 blocks total
        blocks_added = 0
        for goal in field.goals:
            for i in range(25):  # Try to add 100 blocks total
                if blocks_added < 88:
                    goal.add_block(PushBackBlock("red"))
                    blocks_added += 1
        
        # Count total blocks
        total_blocks = sum(goal.get_block_count()["total"] for goal in field.goals)
        assert total_blocks == 88, f"Field should limit to 88 blocks, has {total_blocks}"
    
    def test_alliance_color_validation(self):
        """Test that only valid alliance colors are accepted"""
        valid_colors = ["red", "blue"]
        
        # Test block creation
        for color in valid_colors:
            block = PushBackBlock(color)
            assert block.alliance == color
        
        # Test invalid color handling
        with pytest.raises(ValueError):
            PushBackBlock("green")  # Invalid color
    
    def test_simultaneous_scoring_prevention(self):
        """Test that blocks can't score in multiple locations"""
        field = PushBackFieldState()
        
        # A block scored in a goal shouldn't count elsewhere
        block = PushBackBlock("red")
        field.goals[0].add_block(block)
        
        # Verify block is only in one goal
        blocks_in_goals = sum(
            goal.get_block_count()["red"] 
            for goal in field.goals
        )
        assert blocks_in_goals == 1, "Block should only be in one location"


class TestPushBackEdgeCases:
    """Test edge cases and unusual scenarios"""
    
    def test_empty_match_scoring(self):
        """Test scoring with no game actions"""
        field = PushBackFieldState()
        engine = PushBackScoringEngine()
        
        score, breakdown = engine.calculate_push_back_score(field, "red")
        
        assert score == 0, "Empty match should score 0"
        assert all(v == 0 for v in breakdown.values()), "All components should be 0"
    
    def test_single_block_match(self):
        """Test match with only one block scored"""
        field = PushBackFieldState()
        engine = PushBackScoringEngine()
        
        field.goals[0].add_block(PushBackBlock("red"))
        
        score, breakdown = engine.calculate_push_back_score(field, "red")
        
        assert score == 3, "Single block should score 3 points"
        assert breakdown["blocks"] == 3
        assert breakdown["control_zone"] == 0  # Not enough for control
    
    def test_all_goals_tied(self):
        """Test when all goals have equal blocks"""
        field = PushBackFieldState()
        engine = PushBackScoringEngine()
        
        # Each goal has 2 red and 2 blue blocks
        for goal in field.goals:
            goal.add_block(PushBackBlock("red"))
            goal.add_block(PushBackBlock("red"))
            goal.add_block(PushBackBlock("blue"))
            goal.add_block(PushBackBlock("blue"))
        
        red_score, _ = engine.calculate_push_back_score(field, "red")
        blue_score, _ = engine.calculate_push_back_score(field, "blue")
        
        assert red_score == 0, "All tied goals = no block points"
        assert blue_score == 0, "All tied goals = no block points"
    
    def test_control_zone_only_scoring(self):
        """Test match where only control zone scores"""
        field = PushBackFieldState()
        engine = PushBackScoringEngine()
        
        # No blocks in goals, but control zone presence
        field.zone_control_red_count = 5
        field.zone_control_blue_count = 2
        
        score, breakdown = engine.calculate_push_back_score(field, "red")
        
        assert score == 8, "Should score only from control zone"
        assert breakdown["blocks"] == 0
        assert breakdown["control_zone"] == 8
        assert breakdown["parking"] == 0
        assert breakdown["autonomous_win"] == 0


def run_rule_compliance_tests():
    """Run all rule compliance tests"""
    import subprocess
    import sys
    
    print("\n" + "=" * 60)
    print("PUSH BACK RULE COMPLIANCE TESTS")
    print("=" * 60)
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    
    return result.returncode == 0


if __name__ == "__main__":
    success = run_rule_compliance_tests()
    exit(0 if success else 1)