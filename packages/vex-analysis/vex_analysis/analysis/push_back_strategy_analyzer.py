#!/usr/bin/env python3
"""
Push Back Strategic Analysis Module
Focuses on the 5 key strategic decisions specific to VEX U Push Back
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import random
import math
from itertools import product
from scipy import optimize

try:
    from ..core.simulator import (
        PushBackScoringEngine, AllianceStrategy, Zone, ParkingLocation, 
        PushBackConstants, BlockColor, PushBackField
    )
    from ..core.scenario_generator import ScenarioGenerator, SkillLevel
except ImportError:
    from core.simulator import (
        PushBackScoringEngine, AllianceStrategy, Zone, ParkingLocation,
        PushBackConstants, BlockColor, PushBackField  
    )
    from core.scenario_generator import ScenarioGenerator, SkillLevel


# Push Back Strategy Archetypes
class PushBackArchetype(Enum):
    BLOCK_FLOW_MAXIMIZER = "block_flow_maximizer"
    CONTROL_ZONE_CONTROLLER = "control_zone_controller"
    AUTONOMOUS_SPECIALIST = "autonomous_specialist"
    DEFENSIVE_DISRUPTOR = "defensive_disruptor"
    PARKING_SPECIALIST = "parking_specialist"
    BALANCED_OPTIMIZER = "balanced_optimizer"
    HIGH_RISK_HIGH_REWARD = "high_risk_high_reward"


# Strategic Decisions
class AutonomousStrategy(Enum):
    AUTO_WIN_FOCUSED = "auto_win_focused"          # Prioritize 7-point auto win
    BONUS_FOCUSED = "bonus_focused"               # Prioritize 10-point bonus
    POSITIONING_FOCUSED = "positioning_focused"    # Position for driver control


class GoalPriority(Enum):
    CENTER_FIRST = "center_first"      # Prioritize Center Goals (higher control value)
    LONG_FIRST = "long_first"         # Prioritize Long Goals (easier control)
    BALANCED_GOALS = "balanced_goals"  # Balanced approach
    OPPORTUNISTIC = "opportunistic"   # Adaptive based on opponent


class ParkingTiming(Enum):
    NEVER_PARK = "never_park"         # Always keep scoring
    EARLY_PARK = "early_park"         # Park at small advantage
    SAFE_PARK = "safe_park"          # Park when comfortable lead
    DESPERATE_PARK = "desperate_park" # Park when behind for points


@dataclass
class BlockFlowOptimization:
    """Results from block flow optimization analysis"""
    total_blocks_available: int
    optimal_distribution: Dict[str, int]  # goal -> blocks
    expected_block_points: int
    expected_control_points: int
    total_expected_points: int
    control_efficiency: float  # control points per block
    flow_rate: float  # blocks per minute
    risk_level: float  # 0-1, higher = more risky
    recommendations: List[str]


@dataclass 
class AutonomousDecision:
    """Analysis of autonomous period strategy decision"""
    recommended_strategy: AutonomousStrategy
    auto_win_probability: float
    bonus_probability: float
    positioning_score: float  # Quality of driver control positioning
    expected_auto_points: float
    risk_assessment: str
    block_targets: Dict[str, int]
    time_allocation: Dict[str, float]  # activity -> seconds
    decision_rationale: str


@dataclass
class GoalPriorityAnalysis:
    """Analysis of goal prioritization strategy"""
    recommended_priority: GoalPriority
    center_goal_value: float  # Expected value per block in center
    long_goal_value: float   # Expected value per block in long
    control_difficulty: Dict[str, float]  # goal -> difficulty (0-1)
    opponent_interference: Dict[str, float]  # goal -> interference risk
    optimal_sequence: List[str]  # goal targeting order
    timing_recommendations: Dict[str, str]  # goal -> timing advice
    decision_confidence: float


@dataclass
class ParkingDecisionAnalysis:
    """Analysis of when to commit robots to parking"""
    recommended_timing: ParkingTiming
    one_robot_threshold: int  # Score difference for 1-robot parking
    two_robot_threshold: int  # Score difference for 2-robot parking
    time_thresholds: Dict[ParkingTiming, int]  # strategy -> seconds remaining
    expected_parking_value: float
    opportunity_cost: float  # Expected points lost from not scoring
    risk_benefit_ratio: float
    situational_recommendations: Dict[str, str]


@dataclass
class OffenseDefenseBalance:
    """Analysis of offensive vs defensive resource allocation"""
    recommended_ratio: Tuple[float, float]  # (offense %, defense %)
    offensive_roi: float  # Return on investment for offense
    defensive_roi: float  # Return on investment for defense
    score_differential_impact: Dict[int, Tuple[float, float]]  # score_diff -> ratios
    critical_control_zones: List[str]  # Most important zones to defend
    disruption_targets: List[str]  # Best opponent zones to disrupt
    timing_strategy: Dict[str, Tuple[float, float]]  # time_period -> ratios


class PushBackRobotSpecs(NamedTuple):
    """Robot specifications for realistic simulation"""
    speed: float  # m/s
    acceleration: float  # m/s²
    block_capacity: int  # blocks carried at once
    scoring_time: float  # seconds per block scored
    accuracy: float  # scoring success rate (0-1)
    size_class: str  # "15_inch" or "24_inch"


@dataclass
class PushBackMatchState:
    """Current match state for strategic decisions"""
    time_remaining: int  # seconds
    red_score: int
    blue_score: int
    red_blocks_in_goals: Dict[str, int]
    blue_blocks_in_goals: Dict[str, int]
    red_robots_parked: int
    blue_robots_parked: int
    autonomous_completed: bool
    phase: str  # "autonomous", "driver", "endgame"


class PushBackStrategyAnalyzer:
    """Advanced strategic analysis specifically for Push Back"""
    
    def __init__(self):
        self.engine = PushBackScoringEngine()
        self.constants = PushBackConstants()
        self.field = PushBackField.create_standard_field()
        
        # Robot specifications for realistic simulation
        self.robot_specs = {
            "aggressive": PushBackRobotSpecs(1.8, 2.0, 4, 1.2, 0.85, "15_inch"),
            "balanced": PushBackRobotSpecs(1.5, 1.8, 3, 1.5, 0.80, "24_inch"), 
            "conservative": PushBackRobotSpecs(1.2, 1.5, 2, 2.0, 0.90, "24_inch")
        }
        
        # Strategic decision cache for performance
        self.decision_cache = {}
    
    def analyze_block_flow_optimization(
        self,
        robot_specs: List[PushBackRobotSpecs],
        opponent_strength: float = 0.7,
        time_constraints: Dict[str, int] = None
    ) -> BlockFlowOptimization:
        """
        1. Block Flow Optimization: Optimal distribution of 88 blocks
        
        Analyzes how to distribute blocks across goals to maximize total points
        considering both block scoring (3 pts) and zone control (6-10 pts).
        """
        if time_constraints is None:
            time_constraints = {"autonomous": 30, "driver": 90}
        
        # Available blocks for this alliance (assume equal split)
        available_blocks = 44  # Half of 88 total blocks
        
        # Goal characteristics
        goals = {
            "long_1": {"capacity": 22, "control_points": 10, "control_threshold": 3, "distance": 1.3},
            "long_2": {"capacity": 22, "control_points": 10, "control_threshold": 3, "distance": 1.3}, 
            "center_1": {"capacity": 8, "control_points": 8, "control_threshold": 3, "distance": 0.8},
            "center_2": {"capacity": 7, "control_points": 6, "control_threshold": 3, "distance": 0.8}
        }
        
        # Calculate robot capabilities
        total_time = sum(time_constraints.values())
        avg_speed = np.mean([spec.speed for spec in robot_specs])
        avg_capacity = np.mean([spec.block_capacity for spec in robot_specs])
        avg_scoring_time = np.mean([spec.scoring_time for spec in robot_specs])
        
        # Estimate scoring rate (blocks per minute)
        scoring_rate = 60 / (avg_scoring_time + 2.0)  # Include travel time
        max_blocks_possible = int(scoring_rate * (total_time / 60) * len(robot_specs))
        
        # Constraint: Can't score more blocks than physically possible
        available_blocks = min(available_blocks, max_blocks_possible)
        
        # Optimization function: maximize total value
        def objective_function(distribution):
            """Calculate total expected value for a block distribution"""
            long_1, long_2, center_1, center_2 = distribution
            
            # Block points (3 per block)
            block_points = sum(distribution) * self.constants.POINTS_PER_BLOCK
            
            # Control points
            control_points = 0
            for i, goal in enumerate(["long_1", "long_2", "center_1", "center_2"]):
                blocks = distribution[i] 
                goal_info = goals[goal]
                
                # Control achieved if we have enough blocks vs opponent
                opponent_blocks = max(0, int(goal_info["capacity"] * opponent_strength * 0.4))
                if blocks >= goal_info["control_threshold"] and blocks > opponent_blocks:
                    control_points += goal_info["control_points"]
            
            return block_points + control_points
        
        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda x: sum(x) - available_blocks},  # Use all blocks
            {"type": "ineq", "fun": lambda x: goals["long_1"]["capacity"] - x[0]},   # Long 1 capacity
            {"type": "ineq", "fun": lambda x: goals["long_2"]["capacity"] - x[1]},   # Long 2 capacity  
            {"type": "ineq", "fun": lambda x: goals["center_1"]["capacity"] - x[2]}, # Center 1 capacity
            {"type": "ineq", "fun": lambda x: goals["center_2"]["capacity"] - x[3]}, # Center 2 capacity
        ]
        
        bounds = [(0, capacity) for capacity in [22, 22, 8, 7]]  # Goal capacities
        
        # Initial guess: proportional to capacity
        total_capacity = sum([22, 22, 8, 7])
        initial_guess = [available_blocks * (cap / total_capacity) for cap in [22, 22, 8, 7]]
        
        # Optimize
        try:
            result = optimize.minimize(
                lambda x: -objective_function(x),  # Minimize negative (maximize positive)
                initial_guess,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimal_dist = [int(round(x)) for x in result.x]
                # Adjust for rounding errors
                diff = available_blocks - sum(optimal_dist)
                if diff != 0:
                    # Add/remove from goal with most capacity remaining
                    remaining_capacity = [goals["long_1"]["capacity"] - optimal_dist[0],
                                        goals["long_2"]["capacity"] - optimal_dist[1], 
                                        goals["center_1"]["capacity"] - optimal_dist[2],
                                        goals["center_2"]["capacity"] - optimal_dist[3]]
                    if diff > 0:
                        idx = remaining_capacity.index(max(remaining_capacity))
                        optimal_dist[idx] += diff
                    else:
                        idx = optimal_dist.index(max(optimal_dist))
                        optimal_dist[idx] += diff
            else:
                # Fallback: balanced distribution
                optimal_dist = [11, 11, 8, 7]  # Use capacities efficiently
                remaining = available_blocks - sum(optimal_dist)
                optimal_dist[0] += remaining  # Add remainder to long_1
                
        except Exception:
            # Fallback distribution
            optimal_dist = [min(available_blocks // 4, 22) for _ in range(4)]
            remaining = available_blocks - sum(optimal_dist)
            optimal_dist[0] += remaining
        
        # Calculate final metrics
        optimal_distribution = {
            "long_1": optimal_dist[0],
            "long_2": optimal_dist[1], 
            "center_1": optimal_dist[2],
            "center_2": optimal_dist[3]
        }
        
        block_points = sum(optimal_dist) * self.constants.POINTS_PER_BLOCK
        control_points = 0
        
        # Count achieved controls
        for i, goal in enumerate(["long_1", "long_2", "center_1", "center_2"]):
            blocks = optimal_dist[i]
            goal_info = goals[goal]
            opponent_blocks = int(goal_info["capacity"] * opponent_strength * 0.4)
            if blocks >= goal_info["control_threshold"] and blocks > opponent_blocks:
                control_points += goal_info["control_points"]
        
        total_points = block_points + control_points
        control_efficiency = control_points / sum(optimal_dist) if sum(optimal_dist) > 0 else 0
        flow_rate = (sum(optimal_dist) * 60) / total_time if total_time > 0 else 0
        
        # Risk assessment
        capacity_utilization = sum(optimal_dist) / sum([22, 22, 8, 7])
        control_dependency = control_points / total_points if total_points > 0 else 0
        risk_level = (capacity_utilization * 0.5 + control_dependency * 0.5)
        
        # Generate recommendations
        recommendations = []
        
        if control_points > block_points:
            recommendations.append("Strategy relies heavily on zone control - ensure defensive capabilities")
        if optimal_dist[2] == 8 and optimal_dist[3] == 7:
            recommendations.append("Maximize center goal utilization for higher control value")
        if max(optimal_dist[:2]) > 15:
            recommendations.append("Focus on one long goal for guaranteed control")
        if sum(optimal_dist) == available_blocks:
            recommendations.append("Utilize all available blocks for maximum efficiency")
        
        return BlockFlowOptimization(
            total_blocks_available=available_blocks,
            optimal_distribution=optimal_distribution,
            expected_block_points=block_points,
            expected_control_points=control_points,
            total_expected_points=total_points,
            control_efficiency=control_efficiency,
            flow_rate=flow_rate,
            risk_level=risk_level,
            recommendations=recommendations
        )
    
    def analyze_autonomous_strategy_decision(
        self,
        robot_specs: List[PushBackRobotSpecs],
        team_skill_level: SkillLevel = SkillLevel.ADVANCED,
        opponent_auto_strength: float = 0.6
    ) -> AutonomousDecision:
        """
        2. Autonomous vs Driver Balance Decision
        
        Determines whether to prioritize:
        - 10-point autonomous bonus (higher score at 15s)
        - 7-point autonomous win (complex requirements) 
        - Positioning for driver control dominance
        """
        
        auto_time = self.constants.AUTONOMOUS_TIME  # 30 seconds
        
        # Calculate robot capabilities in autonomous
        total_robots = len(robot_specs)
        avg_speed = np.mean([spec.speed for spec in robot_specs])
        avg_accuracy = np.mean([spec.accuracy for spec in robot_specs])
        avg_capacity = np.mean([spec.block_capacity for spec in robot_specs])
        
        # Estimate scoring potential
        setup_time = 3.0  # Setup and initial positioning
        effective_time = auto_time - setup_time
        
        # Blocks per trip calculation
        trips_possible = effective_time / (6.0 + 2.0 * avg_capacity)  # 6s travel + 2s per block
        max_auto_blocks = int(trips_possible * avg_capacity * total_robots * avg_accuracy)
        
        # Strategy analysis
        strategies = {}
        
        # 1. Auto Win Focused Strategy
        auto_win_blocks = max(7, max_auto_blocks * 0.7)  # Need ≥7 blocks
        auto_win_goals = 3  # Need ≥3 goals with blocks
        loader_blocks = max(3, int(auto_win_blocks * 0.3))  # Need ≥3 loader blocks
        
        # Auto win feasibility
        auto_win_feasible = (
            max_auto_blocks >= auto_win_blocks and
            auto_time >= (auto_win_blocks * 1.5 + 5)  # Time check
        )
        
        if auto_win_feasible:
            auto_win_prob = min(0.9, avg_accuracy * (max_auto_blocks / auto_win_blocks))
        else:
            auto_win_prob = 0.1
        
        strategies[AutonomousStrategy.AUTO_WIN_FOCUSED] = {
            "blocks": int(auto_win_blocks),
            "expected_points": auto_win_blocks * 3 + 7 * auto_win_prob,  # Blocks + win point
            "probability": auto_win_prob,
            "risk": 0.8,  # High risk due to complex requirements
            "positioning_score": 0.3  # Poor positioning for driver
        }
        
        # 2. Bonus Focused Strategy  
        bonus_blocks = min(max_auto_blocks * 0.5, 15)  # Conservative for reliability
        opponent_auto_blocks = max_auto_blocks * opponent_auto_strength
        
        bonus_prob = 0.7 if bonus_blocks > opponent_auto_blocks + 2 else 0.4
        
        strategies[AutonomousStrategy.BONUS_FOCUSED] = {
            "blocks": int(bonus_blocks),
            "expected_points": bonus_blocks * 3 + 10 * bonus_prob,  # Blocks + bonus
            "probability": bonus_prob,
            "risk": 0.5,  # Medium risk
            "positioning_score": 0.5  # Medium positioning
        }
        
        # 3. Positioning Focused Strategy
        positioning_blocks = min(max_auto_blocks * 0.3, 8)  # Minimal scoring
        
        strategies[AutonomousStrategy.POSITIONING_FOCUSED] = {
            "blocks": int(positioning_blocks),
            "expected_points": positioning_blocks * 3 + 0,  # Just block points
            "probability": 0.9,  # High reliability
            "risk": 0.2,  # Low risk
            "positioning_score": 0.9  # Excellent driver positioning
        }
        
        # Select best strategy based on expected value and risk tolerance
        best_strategy = None
        best_value = -1
        
        for strategy, data in strategies.items():
            # Risk-adjusted expected value
            risk_penalty = data["risk"] * 10  # Penalty for risky strategies
            adjusted_value = data["expected_points"] - risk_penalty
            
            # Bonus for positioning (helps driver control)
            positioning_bonus = data["positioning_score"] * 15  # Driver control worth ~15 pts
            adjusted_value += positioning_bonus
            
            if adjusted_value > best_value:
                best_value = adjusted_value
                best_strategy = strategy
        
        # Generate specific recommendations for chosen strategy
        chosen = strategies[best_strategy]
        
        # Block distribution for chosen strategy
        if best_strategy == AutonomousStrategy.AUTO_WIN_FOCUSED:
            block_targets = {
                "long_1": int(chosen["blocks"] * 0.35),
                "long_2": int(chosen["blocks"] * 0.35), 
                "center_1": int(chosen["blocks"] * 0.15),
                "center_2": int(chosen["blocks"] * 0.15)
            }
            # Ensure 3+ goals have blocks
            for goal in ["center_1", "center_2"]:
                if block_targets[goal] == 0:
                    block_targets[goal] = 1
                    block_targets["long_1"] -= 1
        else:
            # More conservative distribution
            block_targets = {
                "long_1": int(chosen["blocks"] * 0.4),
                "long_2": int(chosen["blocks"] * 0.3),
                "center_1": int(chosen["blocks"] * 0.2), 
                "center_2": int(chosen["blocks"] * 0.1)
            }
        
        # Time allocation
        scoring_time = sum(block_targets.values()) * 1.5  # 1.5s per block average
        time_allocation = {
            "setup_positioning": setup_time,
            "block_scoring": min(scoring_time, effective_time * 0.8),
            "driver_positioning": max(0, effective_time * 0.2),
            "contingency": max(0, auto_time - setup_time - scoring_time)
        }
        
        # Risk assessment
        risk_factors = []
        if chosen["risk"] > 0.7:
            risk_assessment = "HIGH RISK - Complex requirements, high failure potential"
        elif chosen["risk"] > 0.4:
            risk_assessment = "MEDIUM RISK - Balanced approach with moderate complexity"
        else:
            risk_assessment = "LOW RISK - Conservative approach with high reliability"
        
        # Decision rationale
        if best_strategy == AutonomousStrategy.AUTO_WIN_FOCUSED:
            rationale = f"Auto win point (7 pts) + bonus potential maximizes score. {auto_win_prob:.0%} success rate."
        elif best_strategy == AutonomousStrategy.BONUS_FOCUSED:
            rationale = f"Reliable 10-point bonus with {bonus_prob:.0%} probability. Good risk/reward balance."
        else:
            rationale = "Driver control positioning prioritized. Strong foundation for match control."
        
        return AutonomousDecision(
            recommended_strategy=best_strategy,
            auto_win_probability=auto_win_prob if best_strategy == AutonomousStrategy.AUTO_WIN_FOCUSED else 0.0,
            bonus_probability=chosen["probability"] if "bonus" in str(best_strategy) else 0.0,
            positioning_score=chosen["positioning_score"],
            expected_auto_points=chosen["expected_points"],
            risk_assessment=risk_assessment,
            block_targets=block_targets,
            time_allocation=time_allocation,
            decision_rationale=rationale
        )
    
    def analyze_goal_priority_strategy(
        self,
        robot_specs: List[PushBackRobotSpecs],
        opponent_strategy: str = "balanced",
        match_phase: str = "early"
    ) -> GoalPriorityAnalysis:
        """
        3. Center Goal vs Long Goal Priority Decision
        
        Analyzes when to focus on:
        - Center Goals: Higher control values (6-8 pts) but lower capacity (7-8 blocks)
        - Long Goals: Lower control value (10 pts) but higher capacity (22 blocks)
        """
        
        # Goal characteristics
        goals = {
            "center_1": {
                "capacity": 8, "control_points": 8, "control_threshold": 3,
                "distance": 0.8, "traffic": 0.6, "saturation_risk": 0.7
            },
            "center_2": {
                "capacity": 7, "control_points": 6, "control_threshold": 3,
                "distance": 0.8, "traffic": 0.6, "saturation_risk": 0.8
            },
            "long_1": {
                "capacity": 22, "control_points": 10, "control_threshold": 3,
                "distance": 1.3, "traffic": 0.4, "saturation_risk": 0.3
            },
            "long_2": {
                "capacity": 22, "control_points": 10, "control_threshold": 3,
                "distance": 1.3, "traffic": 0.4, "saturation_risk": 0.3
            }
        }
        
        # Robot capabilities
        avg_speed = np.mean([spec.speed for spec in robot_specs])
        avg_capacity = np.mean([spec.block_capacity for spec in robot_specs])
        
        # Calculate value per block for each goal type
        center_value_per_block = []
        long_value_per_block = []
        
        for goal, props in goals.items():
            base_value = self.constants.POINTS_PER_BLOCK  # 3 points
            
            # Control value contribution (amortized over blocks needed for control)
            control_contribution = props["control_points"] / props["control_threshold"]
            
            # Efficiency factors
            distance_penalty = props["distance"] * 0.3  # Further goals take more time
            traffic_penalty = props["traffic"] * 0.2    # Contested goals are harder
            
            value_per_block = base_value + control_contribution - distance_penalty - traffic_penalty
            
            if "center" in goal:
                center_value_per_block.append(value_per_block)
            else:
                long_value_per_block.append(value_per_block)
        
        center_avg_value = np.mean(center_value_per_block)
        long_avg_value = np.mean(long_value_per_block)
        
        # Opponent interference analysis
        opponent_interference = {}
        
        if opponent_strategy == "aggressive":
            interference_multiplier = 1.3
        elif opponent_strategy == "defensive": 
            interference_multiplier = 1.5
        else:  # balanced
            interference_multiplier = 1.0
        
        for goal, props in goals.items():
            base_interference = props["traffic"]
            opponent_interference[goal] = min(0.9, base_interference * interference_multiplier)
        
        # Control difficulty (how hard to achieve/maintain control)
        control_difficulty = {}
        for goal, props in goals.items():
            # Smaller capacity goals harder to control (less margin for error)
            capacity_factor = 1.0 / (props["capacity"] / 22)  # Normalize to long goal capacity
            traffic_factor = props["traffic"]
            difficulty = min(0.95, (capacity_factor * 0.6 + traffic_factor * 0.4))
            control_difficulty[goal] = difficulty
        
        # Phase-specific adjustments
        phase_adjustments = {
            "early": {"center_bonus": 0.1, "long_bonus": 0.0},    # Early game favors centers
            "mid": {"center_bonus": 0.0, "long_bonus": 0.05},     # Mid game balanced
            "late": {"center_bonus": -0.1, "long_bonus": 0.1}     # Late game favors longs
        }
        
        if match_phase in phase_adjustments:
            center_avg_value *= (1 + phase_adjustments[match_phase]["center_bonus"])
            long_avg_value *= (1 + phase_adjustments[match_phase]["long_bonus"])
        
        # Decision logic
        value_difference = center_avg_value - long_avg_value
        
        if abs(value_difference) < 0.5:
            recommended_priority = GoalPriority.BALANCED_GOALS
            decision_confidence = 0.6
        elif center_avg_value > long_avg_value:
            if center_avg_value > long_avg_value * 1.15:
                recommended_priority = GoalPriority.CENTER_FIRST
                decision_confidence = 0.8
            else:
                recommended_priority = GoalPriority.BALANCED_GOALS  
                decision_confidence = 0.7
        else:
            if long_avg_value > center_avg_value * 1.15:
                recommended_priority = GoalPriority.LONG_FIRST
                decision_confidence = 0.8
            else:
                recommended_priority = GoalPriority.BALANCED_GOALS
                decision_confidence = 0.7
        
        # Optimal sequence based on priority
        if recommended_priority == GoalPriority.CENTER_FIRST:
            optimal_sequence = ["center_1", "center_2", "long_1", "long_2"]
        elif recommended_priority == GoalPriority.LONG_FIRST:
            optimal_sequence = ["long_1", "long_2", "center_1", "center_2"]
        else:  # BALANCED or OPPORTUNISTIC
            optimal_sequence = ["center_1", "long_1", "center_2", "long_2"]
        
        # Timing recommendations
        timing_recommendations = {}
        for goal in goals.keys():
            if "center" in goal:
                if match_phase == "early":
                    timing_recommendations[goal] = "Prioritize early - higher value and easier access"
                else:
                    timing_recommendations[goal] = "Fill after establishing long goal control"
            else:
                if recommended_priority == GoalPriority.LONG_FIRST:
                    timing_recommendations[goal] = "Establish control early with 3+ blocks"
                else:
                    timing_recommendations[goal] = "Target after center goals for sustained scoring"
        
        return GoalPriorityAnalysis(
            recommended_priority=recommended_priority,
            center_goal_value=center_avg_value,
            long_goal_value=long_avg_value,
            control_difficulty=control_difficulty,
            opponent_interference=opponent_interference,
            optimal_sequence=optimal_sequence,
            timing_recommendations=timing_recommendations,
            decision_confidence=decision_confidence
        )
    
    def analyze_parking_decision_timing(
        self,
        current_state: PushBackMatchState,
        robot_specs: List[PushBackRobotSpecs]
    ) -> ParkingDecisionAnalysis:
        """
        4. Parking Decision Timing Analysis
        
        Determines when to commit robots to parking based on:
        - 8 points for 1 robot vs 30 points for 2 robots
        - Score differential and time remaining
        - Opportunity cost of continued scoring
        """
        
        # Calculate scoring potential if robots continue playing
        remaining_time = current_state.time_remaining
        score_differential = current_state.red_score - current_state.blue_score
        
        # Robot scoring capability
        avg_scoring_rate = np.mean([60 / (spec.scoring_time + 2.0) for spec in robot_specs])  # blocks/min
        total_robots = len(robot_specs)
        
        # Opportunity cost calculation
        remaining_scoring_potential = (remaining_time / 60) * avg_scoring_rate * total_robots
        opportunity_cost_one_robot = remaining_scoring_potential * 0.5 * self.constants.POINTS_PER_BLOCK
        opportunity_cost_two_robots = remaining_scoring_potential * self.constants.POINTS_PER_BLOCK
        
        # Parking value analysis
        one_robot_value = self.constants.ONE_ROBOT_PARKING  # 8 points
        two_robot_value = self.constants.TWO_ROBOT_PARKING  # 30 points
        
        # Net value calculations
        one_robot_net = one_robot_value - opportunity_cost_one_robot
        two_robot_net = two_robot_value - opportunity_cost_two_robots
        
        # Decision thresholds based on score differential and time
        base_one_robot_threshold = 0   # Park 1 robot when even or ahead
        base_two_robot_threshold = 15  # Park 2 robots when safely ahead
        
        # Time-based adjustments
        if remaining_time <= 15:  # Endgame
            base_one_robot_threshold -= 5
            base_two_robot_threshold -= 10
        elif remaining_time <= 30:  # Late game
            base_one_robot_threshold -= 2
            base_two_robot_threshold -= 5
        
        # Risk assessment based on opponent scoring potential
        opponent_potential = max(10, remaining_time * 0.3)  # Conservative estimate
        
        # Adjust thresholds for risk
        one_robot_threshold = base_one_robot_threshold + int(opponent_potential * 0.3)
        two_robot_threshold = base_two_robot_threshold + int(opponent_potential * 0.2)
        
        # Determine recommended timing
        if score_differential >= two_robot_threshold and two_robot_net > 0:
            recommended_timing = ParkingTiming.SAFE_PARK
        elif score_differential >= one_robot_threshold and one_robot_net > 0:
            recommended_timing = ParkingTiming.EARLY_PARK
        elif score_differential < -10 and remaining_time <= 30:
            # Desperate situation - park for guaranteed points
            if two_robot_net > opportunity_cost_two_robots * 0.3:
                recommended_timing = ParkingTiming.DESPERATE_PARK
            else:
                recommended_timing = ParkingTiming.NEVER_PARK
        else:
            recommended_timing = ParkingTiming.NEVER_PARK
        
        # Time thresholds for different strategies
        time_thresholds = {
            ParkingTiming.EARLY_PARK: max(45, remaining_time + 15),
            ParkingTiming.SAFE_PARK: max(30, remaining_time + 10), 
            ParkingTiming.DESPERATE_PARK: 20,
            ParkingTiming.NEVER_PARK: 0
        }
        
        # Risk-benefit analysis
        if two_robot_net > 0:
            risk_benefit_ratio = two_robot_value / opportunity_cost_two_robots
        elif one_robot_net > 0:
            risk_benefit_ratio = one_robot_value / opportunity_cost_one_robot
        else:
            risk_benefit_ratio = 0.0
        
        # Expected parking value (probability-weighted)
        if recommended_timing == ParkingTiming.SAFE_PARK:
            expected_parking_value = two_robot_value * 0.9  # High confidence
        elif recommended_timing == ParkingTiming.EARLY_PARK:
            expected_parking_value = one_robot_value * 0.8  # Good confidence
        elif recommended_timing == ParkingTiming.DESPERATE_PARK:
            expected_parking_value = two_robot_value * 0.6  # Risky but necessary
        else:
            expected_parking_value = 0.0
        
        # Situational recommendations
        situational_recommendations = {
            "leading_big": f"Park 2 robots at {two_robot_threshold}+ point lead for guaranteed 30 points",
            "leading_small": f"Park 1 robot at {one_robot_threshold}+ point lead for safe 8 points", 
            "tied": "Continue scoring unless <30 seconds remain",
            "trailing": "Only park if <20 seconds and desperate for guaranteed points",
            "endgame": "Prioritize parking if net positive value and secure position"
        }
        
        return ParkingDecisionAnalysis(
            recommended_timing=recommended_timing,
            one_robot_threshold=one_robot_threshold,
            two_robot_threshold=two_robot_threshold,
            time_thresholds=time_thresholds,
            expected_parking_value=expected_parking_value,
            opportunity_cost=opportunity_cost_two_robots,
            risk_benefit_ratio=risk_benefit_ratio,
            situational_recommendations=situational_recommendations
        )
    
    def analyze_offense_defense_balance(
        self,
        current_state: PushBackMatchState,
        robot_specs: List[PushBackRobotSpecs]
    ) -> OffenseDefenseBalance:
        """
        5. Offensive vs Defensive Resource Allocation
        
        Determines optimal balance between:
        - Offensive focus: Block scoring and zone building
        - Defensive focus: Disrupting opponent control and blocking
        """
        
        score_differential = current_state.red_score - current_state.blue_score
        time_remaining = current_state.time_remaining
        
        # Calculate current zone control status
        red_control_zones = []
        blue_control_zones = []
        
        for goal, red_blocks in current_state.red_blocks_in_goals.items():
            blue_blocks = current_state.blue_blocks_in_goals.get(goal, 0)
            
            # Zone control if ≥3 blocks and more than opponent
            if red_blocks >= 3 and red_blocks > blue_blocks:
                red_control_zones.append(goal)
            elif blue_blocks >= 3 and blue_blocks > red_blocks:
                blue_control_zones.append(goal)
        
        # Calculate ROI for offensive and defensive actions
        
        # Offensive ROI: Expected points from continued scoring
        avg_scoring_rate = np.mean([60 / (spec.scoring_time + 2.0) for spec in robot_specs])
        expected_offensive_points = (time_remaining / 60) * avg_scoring_rate * self.constants.POINTS_PER_BLOCK
        
        # Defensive ROI: Expected points denied to opponent
        opponent_scoring_rate = avg_scoring_rate * 0.7  # Assume opponent slightly slower
        expected_defensive_value = (time_remaining / 60) * opponent_scoring_rate * self.constants.POINTS_PER_BLOCK
        
        # Zone control considerations
        offensive_zone_value = 0
        defensive_zone_value = 0
        
        goals_info = {
            "long_1": 10, "long_2": 10, "center_1": 8, "center_2": 6
        }
        
        # Offensive zone value: zones we can gain
        for goal in goals_info:
            if goal not in red_control_zones:
                red_blocks = current_state.red_blocks_in_goals.get(goal, 0)
                blue_blocks = current_state.blue_blocks_in_goals.get(goal, 0)
                blocks_needed = max(1, 3 - red_blocks + blue_blocks)
                
                if blocks_needed <= 5:  # Achievable
                    offensive_zone_value += goals_info[goal] / blocks_needed
        
        # Defensive zone value: zones we can deny
        for goal in blue_control_zones:
            if goal not in red_control_zones:
                blue_blocks = current_state.blue_blocks_in_goals.get(goal, 0)
                disruption_value = goals_info[goal] if blue_blocks <= 6 else goals_info[goal] * 0.5
                defensive_zone_value += disruption_value
        
        # Adjust ROI with zone control
        offensive_roi = expected_offensive_points + offensive_zone_value
        defensive_roi = expected_defensive_value + defensive_zone_value
        
        # Score differential impact on strategy
        score_differential_impact = {}
        
        for diff in range(-30, 31, 5):
            if diff >= 20:  # Leading significantly
                offense_ratio = 0.3  # Focus on defense/parking
                defense_ratio = 0.7
            elif diff >= 10:  # Leading moderately
                offense_ratio = 0.4
                defense_ratio = 0.6
            elif diff >= -5:  # Close game
                offense_ratio = 0.6
                defense_ratio = 0.4
            elif diff >= -15:  # Trailing moderately
                offense_ratio = 0.8  # Must score
                defense_ratio = 0.2
            else:  # Trailing significantly
                offense_ratio = 0.9  # All-out offense
                defense_ratio = 0.1
            
            score_differential_impact[diff] = (offense_ratio, defense_ratio)
        
        # Get recommendation for current differential
        current_ratios = score_differential_impact.get(
            score_differential,
            score_differential_impact[min(score_differential_impact.keys(), 
                                        key=lambda x: abs(x - score_differential))]
        )
        
        # Time-based strategy adjustments
        timing_strategy = {}
        
        if time_remaining > 60:  # Early game
            timing_strategy["early"] = (0.7, 0.3)  # Offensive focus
        elif time_remaining > 30:  # Mid game
            timing_strategy["mid"] = current_ratios
        else:  # Endgame
            if score_differential > 10:
                timing_strategy["endgame"] = (0.2, 0.8)  # Defensive/parking
            elif score_differential < -10:
                timing_strategy["endgame"] = (0.9, 0.1)  # Desperate offense
            else:
                timing_strategy["endgame"] = (0.6, 0.4)  # Balanced endgame
        
        # Critical zones to defend (opponent has control or close to it)
        critical_control_zones = []
        for goal in blue_control_zones:
            critical_control_zones.append(goal)
        
        # Add zones where opponent is close to control
        for goal, blue_blocks in current_state.blue_blocks_in_goals.items():
            if blue_blocks >= 2 and goal not in blue_control_zones:
                red_blocks = current_state.red_blocks_in_goals.get(goal, 0)
                if blue_blocks >= red_blocks:
                    critical_control_zones.append(goal)
        
        # Best disruption targets (high value zones opponent is building)
        disruption_targets = []
        for goal, blue_blocks in current_state.blue_blocks_in_goals.items():
            red_blocks = current_state.red_blocks_in_goals.get(goal, 0)
            if blue_blocks > red_blocks and blue_blocks >= 2:
                value = goals_info.get(goal, 0)
                disruption_targets.append((goal, value))
        
        # Sort by value and take top targets
        disruption_targets.sort(key=lambda x: x[1], reverse=True)
        disruption_targets = [goal for goal, _ in disruption_targets[:2]]
        
        return OffenseDefenseBalance(
            recommended_ratio=current_ratios,
            offensive_roi=offensive_roi,
            defensive_roi=defensive_roi,
            score_differential_impact=score_differential_impact,
            critical_control_zones=list(set(critical_control_zones)),
            disruption_targets=disruption_targets,
            timing_strategy=timing_strategy
        )
    
    def create_push_back_archetype_strategies(self) -> Dict[PushBackArchetype, AllianceStrategy]:
        """
        Create Push Back-specific strategy archetypes
        
        Replaces generic strategies with Push Back-optimized approaches
        """
        
        strategies = {}
        
        # 1. Block Flow Maximizer - Maximize total blocks scored
        strategies[PushBackArchetype.BLOCK_FLOW_MAXIMIZER] = AllianceStrategy(
            name="Block Flow Maximizer",
            blocks_scored_auto={"long_1": 10, "long_2": 8, "center_1": 6, "center_2": 4},  # 28 blocks
            blocks_scored_driver={"long_1": 12, "long_2": 14, "center_1": 2, "center_2": 3},  # 31 blocks
            zones_controlled=[Zone.RED_HOME],
            robots_parked=[ParkingLocation.NONE, ParkingLocation.NONE],  # Never park
            loader_blocks_removed=8,
            park_zone_contact_auto=False
        )
        
        # 2. Control Zone Controller - Maximize zone control points
        strategies[PushBackArchetype.CONTROL_ZONE_CONTROLLER] = AllianceStrategy(
            name="Control Zone Controller", 
            blocks_scored_auto={"long_1": 6, "long_2": 6, "center_1": 5, "center_2": 4},  # 21 blocks
            blocks_scored_driver={"long_1": 8, "long_2": 8, "center_1": 3, "center_2": 3},  # 22 blocks
            zones_controlled=[Zone.RED_HOME, Zone.NEUTRAL],
            robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.PLATFORM],
            loader_blocks_removed=6,
            park_zone_contact_auto=False
        )
        
        # 3. Autonomous Specialist - Maximize autonomous period value
        strategies[PushBackArchetype.AUTONOMOUS_SPECIALIST] = AllianceStrategy(
            name="Autonomous Specialist",
            blocks_scored_auto={"long_1": 12, "long_2": 10, "center_1": 6, "center_2": 5},  # 33 blocks - aggressive auto
            blocks_scored_driver={"long_1": 4, "long_2": 6, "center_1": 1, "center_2": 1},  # 12 blocks - minimal driver
            zones_controlled=[Zone.RED_HOME],
            robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.ALLIANCE_ZONE],
            loader_blocks_removed=10,  # High loader removal
            park_zone_contact_auto=False
        )
        
        # 4. Defensive Disruptor - Focus on disrupting opponent
        strategies[PushBackArchetype.DEFENSIVE_DISRUPTOR] = AllianceStrategy(
            name="Defensive Disruptor",
            blocks_scored_auto={"long_1": 4, "long_2": 4, "center_1": 3, "center_2": 3},  # 14 blocks - conservative auto
            blocks_scored_driver={"long_1": 6, "long_2": 6, "center_1": 5, "center_2": 4},  # 21 blocks - strategic driver
            zones_controlled=[Zone.RED_HOME, Zone.BLUE_HOME, Zone.NEUTRAL],  # All zones
            robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.PLATFORM],
            loader_blocks_removed=3,
            park_zone_contact_auto=False
        )
        
        # 5. Parking Specialist - Optimize parking timing and value
        strategies[PushBackArchetype.PARKING_SPECIALIST] = AllianceStrategy(
            name="Parking Specialist",
            blocks_scored_auto={"long_1": 8, "long_2": 8, "center_1": 4, "center_2": 3},  # 23 blocks
            blocks_scored_driver={"long_1": 6, "long_2": 6, "center_1": 2, "center_2": 2},  # 16 blocks - focus on parking
            zones_controlled=[Zone.RED_HOME],
            robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.PLATFORM],  # Always double park
            loader_blocks_removed=5,
            park_zone_contact_auto=False
        )
        
        # 6. Balanced Optimizer - Optimize across all scoring categories
        strategies[PushBackArchetype.BALANCED_OPTIMIZER] = AllianceStrategy(
            name="Balanced Optimizer",
            blocks_scored_auto={"long_1": 8, "long_2": 7, "center_1": 5, "center_2": 4},  # 24 blocks
            blocks_scored_driver={"long_1": 8, "long_2": 9, "center_1": 3, "center_2": 3},  # 23 blocks
            zones_controlled=[Zone.RED_HOME, Zone.NEUTRAL],
            robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.PLATFORM],
            loader_blocks_removed=6,
            park_zone_contact_auto=False
        )
        
        # 7. High Risk High Reward - Maximum possible points with high variance
        strategies[PushBackArchetype.HIGH_RISK_HIGH_REWARD] = AllianceStrategy(
            name="High Risk High Reward",
            blocks_scored_auto={"long_1": 15, "long_2": 12, "center_1": 8, "center_2": 7},  # 42 blocks - maximum auto
            blocks_scored_driver={"long_1": 7, "long_2": 10, "center_1": 0, "center_2": 0},  # 17 blocks - risky driver
            zones_controlled=[Zone.RED_HOME, Zone.NEUTRAL],
            robots_parked=[ParkingLocation.NONE, ParkingLocation.NONE],  # No parking - all scoring
            loader_blocks_removed=12,  # Maximum loader removal
            park_zone_contact_auto=False
        )
        
        return strategies
    
    def run_push_back_monte_carlo(
        self,
        strategy: AllianceStrategy,
        num_simulations: int = 1000,
        opponent_archetypes: List[PushBackArchetype] = None
    ) -> Dict[str, Any]:
        """
        Push Back-specific Monte Carlo simulation
        
        Uses realistic robot capabilities and Push Back timing constraints
        """
        
        if opponent_archetypes is None:
            opponent_archetypes = list(PushBackArchetype)
        
        # Create opponent strategies
        opponent_strategies = self.create_push_back_archetype_strategies()
        
        results = {
            'wins': 0,
            'losses': 0,
            'ties': 0,
            'scores': [],
            'margins': [],
            'scoring_breakdown': {
                'blocks': [],
                'autonomous_bonus': [],
                'goal_control': [],
                'parking': [],
                'autonomous_win': []
            },
            'opponent_matchups': {},
            'performance_by_phase': {
                'autonomous': [],
                'driver': [],
                'endgame': []
            }
        }
        
        print(f"Running Push Back Monte Carlo for {strategy.name} ({num_simulations} simulations)...")
        
        for i in range(num_simulations):
            if i % 200 == 0:
                print(f"  Simulation {i}/{num_simulations}...")
            
            # Select random opponent archetype
            opponent_archetype = random.choice(opponent_archetypes)
            opponent = opponent_strategies[opponent_archetype]
            
            # Add realistic variations to strategies
            varied_strategy = self._add_realistic_variations(strategy)
            varied_opponent = self._add_realistic_variations(opponent)
            
            # Simulate match with Push Back engine
            match_result = self.engine.simulate_push_back_match(varied_strategy, varied_opponent)
            
            # Record results
            results['scores'].append(match_result.red_score)
            
            if match_result.winner == "red":
                results['wins'] += 1
                results['margins'].append(match_result.margin)
            elif match_result.winner == "blue":
                results['losses'] += 1
                results['margins'].append(-match_result.margin)
            else:
                results['ties'] += 1
                results['margins'].append(0)
            
            # Breakdown analysis
            for component, points in match_result.red_breakdown.items():
                if component in results['scoring_breakdown']:
                    results['scoring_breakdown'][component].append(points)
            
            # Opponent matchup tracking
            opp_name = opponent_archetype.value
            if opp_name not in results['opponent_matchups']:
                results['opponent_matchups'][opp_name] = {'wins': 0, 'total': 0, 'avg_score': []}
            
            results['opponent_matchups'][opp_name]['total'] += 1
            results['opponent_matchups'][opp_name]['avg_score'].append(match_result.red_score)
            
            if match_result.winner == "red":
                results['opponent_matchups'][opp_name]['wins'] += 1
        
        # Calculate final metrics
        total_simulations = results['wins'] + results['losses'] + results['ties']
        results['win_rate'] = results['wins'] / total_simulations
        results['avg_score'] = np.mean(results['scores'])
        results['score_std'] = np.std(results['scores'])
        results['avg_margin'] = np.mean(results['margins'])
        
        # Calculate component percentages
        for component, values in results['scoring_breakdown'].items():
            if values:
                avg_component = np.mean(values)
                results['scoring_breakdown'][component] = {
                    'avg_points': avg_component,
                    'percentage': (avg_component / results['avg_score']) * 100 if results['avg_score'] > 0 else 0,
                    'consistency': 1 - (np.std(values) / avg_component) if avg_component > 0 else 0
                }
        
        # Opponent matchup analysis
        for opponent, data in results['opponent_matchups'].items():
            data['win_rate'] = data['wins'] / data['total']
            data['avg_score'] = np.mean(data['avg_score'])
        
        return results
    
    def _add_realistic_variations(self, strategy: AllianceStrategy) -> AllianceStrategy:
        """Add realistic performance variations to strategy"""
        
        # Performance variation factors
        auto_variation = random.uniform(0.85, 1.15)  # ±15% variation in auto
        driver_variation = random.uniform(0.90, 1.10)  # ±10% variation in driver
        
        # Apply variations while respecting constraints
        varied_auto = {}
        varied_driver = {}
        
        for goal, blocks in strategy.blocks_scored_auto.items():
            varied_blocks = max(0, int(blocks * auto_variation))
            # Respect goal capacity
            if goal in ["long_1", "long_2"]:
                varied_blocks = min(varied_blocks, 22)
            elif goal == "center_1":
                varied_blocks = min(varied_blocks, 8)
            elif goal == "center_2": 
                varied_blocks = min(varied_blocks, 7)
            varied_auto[goal] = varied_blocks
        
        for goal, blocks in strategy.blocks_scored_driver.items():
            varied_blocks = max(0, int(blocks * driver_variation))
            # Respect remaining capacity
            auto_blocks = varied_auto.get(goal, 0)
            if goal in ["long_1", "long_2"]:
                varied_blocks = min(varied_blocks, 22 - auto_blocks)
            elif goal == "center_1":
                varied_blocks = min(varied_blocks, 8 - auto_blocks)
            elif goal == "center_2":
                varied_blocks = min(varied_blocks, 7 - auto_blocks)
            varied_driver[goal] = varied_blocks
        
        return AllianceStrategy(
            name=f"{strategy.name} (Varied)",
            blocks_scored_auto=varied_auto,
            blocks_scored_driver=varied_driver,
            zones_controlled=strategy.zones_controlled,
            robots_parked=strategy.robots_parked,
            loader_blocks_removed=max(0, int(strategy.loader_blocks_removed * auto_variation)),
            park_zone_contact_auto=strategy.park_zone_contact_auto
        )
    
    def run_comprehensive_push_back_analysis(
        self,
        robot_specs: List[PushBackRobotSpecs] = None,
        match_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive Push Back strategic analysis
        
        Analyzes all 5 key strategic decisions and provides recommendations
        """
        
        if robot_specs is None:
            robot_specs = [
                self.robot_specs["balanced"],
                self.robot_specs["balanced"]
            ]
        
        if match_context is None:
            match_context = {
                "current_state": PushBackMatchState(
                    time_remaining=60,
                    red_score=0,
                    blue_score=0,
                    red_blocks_in_goals={"long_1": 0, "long_2": 0, "center_1": 0, "center_2": 0},
                    blue_blocks_in_goals={"long_1": 0, "long_2": 0, "center_1": 0, "center_2": 0},
                    red_robots_parked=0,
                    blue_robots_parked=0,
                    autonomous_completed=False,
                    phase="driver"
                ),
                "opponent_strategy": "balanced",
                "match_phase": "early"
            }
        
        print("🎯 COMPREHENSIVE PUSH BACK STRATEGIC ANALYSIS")
        print("=" * 60)
        
        results = {}
        
        # 1. Block Flow Optimization
        print("\n1. 📦 BLOCK FLOW OPTIMIZATION")
        print("-" * 40)
        block_flow = self.analyze_block_flow_optimization(robot_specs)
        results['block_flow'] = block_flow
        print(f"Optimal Distribution: {block_flow.optimal_distribution}")
        print(f"Expected Total Points: {block_flow.total_expected_points}")
        print(f"Control Efficiency: {block_flow.control_efficiency:.2f} control pts/block")
        print(f"Risk Level: {block_flow.risk_level:.1%}")
        
        # 2. Autonomous Strategy Decision
        print("\n2. 🤖 AUTONOMOUS STRATEGY DECISION")
        print("-" * 40)
        auto_decision = self.analyze_autonomous_strategy_decision(robot_specs)
        results['autonomous_decision'] = auto_decision
        print(f"Recommended Strategy: {auto_decision.recommended_strategy.value}")
        print(f"Expected Auto Points: {auto_decision.expected_auto_points:.1f}")
        print(f"Auto Win Probability: {auto_decision.auto_win_probability:.1%}")
        print(f"Risk Assessment: {auto_decision.risk_assessment}")
        
        # 3. Goal Priority Analysis
        print("\n3. 🎯 GOAL PRIORITY STRATEGY")
        print("-" * 40)
        goal_priority = self.analyze_goal_priority_strategy(
            robot_specs, 
            match_context["opponent_strategy"], 
            match_context["match_phase"]
        )
        results['goal_priority'] = goal_priority
        print(f"Recommended Priority: {goal_priority.recommended_priority.value}")
        print(f"Center Goal Value: {goal_priority.center_goal_value:.2f} pts/block")
        print(f"Long Goal Value: {goal_priority.long_goal_value:.2f} pts/block")
        print(f"Optimal Sequence: {' → '.join(goal_priority.optimal_sequence)}")
        
        # 4. Parking Decision Analysis
        print("\n4. 🅿️  PARKING DECISION TIMING")
        print("-" * 40)
        parking_decision = self.analyze_parking_decision_timing(
            match_context["current_state"], 
            robot_specs
        )
        results['parking_decision'] = parking_decision
        print(f"Recommended Timing: {parking_decision.recommended_timing.value}")
        print(f"One Robot Threshold: {parking_decision.one_robot_threshold} point lead")
        print(f"Two Robot Threshold: {parking_decision.two_robot_threshold} point lead")
        print(f"Risk-Benefit Ratio: {parking_decision.risk_benefit_ratio:.2f}")
        
        # 5. Offense-Defense Balance
        print("\n5. ⚔️  OFFENSE-DEFENSE BALANCE")
        print("-" * 40)
        offense_defense = self.analyze_offense_defense_balance(
            match_context["current_state"],
            robot_specs
        )
        results['offense_defense'] = offense_defense
        offense_pct, defense_pct = offense_defense.recommended_ratio
        print(f"Recommended Ratio: {offense_pct:.0%} Offense / {defense_pct:.0%} Defense")
        print(f"Offensive ROI: {offense_defense.offensive_roi:.1f}")
        print(f"Defensive ROI: {offense_defense.defensive_roi:.1f}")
        print(f"Critical Zones: {offense_defense.critical_control_zones}")
        
        # 6. Generate Push Back Archetype Strategies
        print("\n6. 🏗️  PUSH BACK STRATEGY ARCHETYPES")
        print("-" * 40)
        archetypes = self.create_push_back_archetype_strategies()
        results['archetypes'] = archetypes
        
        print("Available Archetypes:")
        for archetype, strategy in archetypes.items():
            total_auto = sum(strategy.blocks_scored_auto.values())
            total_driver = sum(strategy.blocks_scored_driver.values())
            print(f"  • {archetype.value}: {total_auto}A + {total_driver}D = {total_auto + total_driver} blocks")
        
        # 7. Strategic Recommendations
        print("\n7. 💡 INTEGRATED STRATEGIC RECOMMENDATIONS")
        print("-" * 50)
        recommendations = self._generate_integrated_recommendations(results)
        results['recommendations'] = recommendations
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        # 8. Best Archetype for Current Situation
        best_archetype = self._recommend_best_archetype(results, match_context)
        results['recommended_archetype'] = best_archetype
        
        print(f"\n🏆 RECOMMENDED ARCHETYPE: {best_archetype.value}")
        print(f"Best suited for current situation based on analysis")
        
        return results
    
    def _generate_integrated_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate integrated recommendations from all analyses"""
        recommendations = []
        
        # Block flow recommendations
        block_flow = results['block_flow']
        if block_flow.control_efficiency > 2.0:
            recommendations.append(f"High control efficiency ({block_flow.control_efficiency:.1f}) - prioritize zone control strategy")
        if block_flow.risk_level > 0.7:
            recommendations.append("High risk block distribution - ensure defensive backup plans")
        
        # Autonomous recommendations  
        auto_decision = results['autonomous_decision']
        if auto_decision.auto_win_probability > 0.6:
            recommendations.append(f"Strong auto win potential ({auto_decision.auto_win_probability:.0%}) - commit to autonomous specialist approach")
        elif auto_decision.positioning_score > 0.8:
            recommendations.append("Excellent positioning potential - focus on driver control dominance")
        
        # Goal priority recommendations
        goal_priority = results['goal_priority']
        if goal_priority.center_goal_value > goal_priority.long_goal_value * 1.2:
            recommendations.append("Center goals significantly more valuable - prioritize center-first strategy")
        elif goal_priority.long_goal_value > goal_priority.center_goal_value * 1.2:
            recommendations.append("Long goals more valuable - establish long goal control early")
        
        # Parking recommendations
        parking_decision = results['parking_decision']
        if parking_decision.risk_benefit_ratio > 1.5:
            recommendations.append(f"High parking value (ratio: {parking_decision.risk_benefit_ratio:.1f}) - commit to parking strategy")
        
        # Offense-defense recommendations
        offense_defense = results['offense_defense']
        offense_pct, defense_pct = offense_defense.recommended_ratio
        if offense_pct > 0.8:
            recommendations.append("High offensive focus recommended - maximize scoring opportunities")
        elif defense_pct > 0.6:
            recommendations.append("Defensive strategy recommended - focus on disruption and control")
        
        return recommendations
    
    def _recommend_best_archetype(self, results: Dict[str, Any], match_context: Dict[str, Any]) -> PushBackArchetype:
        """Recommend the best archetype based on comprehensive analysis"""
        
        scores = {}
        
        # Score each archetype based on analysis results
        block_flow = results['block_flow']
        auto_decision = results['autonomous_decision']
        goal_priority = results['goal_priority']
        parking_decision = results['parking_decision']
        offense_defense = results['offense_defense']
        
        # Block Flow Maximizer
        scores[PushBackArchetype.BLOCK_FLOW_MAXIMIZER] = (
            block_flow.flow_rate * 0.4 +
            (1 - block_flow.risk_level) * 0.2 +
            offense_defense.offensive_roi * 0.003 +
            (1 if parking_decision.recommended_timing == ParkingTiming.NEVER_PARK else 0) * 10
        )
        
        # Control Zone Controller
        scores[PushBackArchetype.CONTROL_ZONE_CONTROLLER] = (
            block_flow.control_efficiency * 10 +
            (1 if goal_priority.recommended_priority in [GoalPriority.CENTER_FIRST, GoalPriority.BALANCED_GOALS] else 0) * 15 +
            (1 if parking_decision.recommended_timing in [ParkingTiming.SAFE_PARK, ParkingTiming.EARLY_PARK] else 0) * 10
        )
        
        # Autonomous Specialist  
        scores[PushBackArchetype.AUTONOMOUS_SPECIALIST] = (
            auto_decision.auto_win_probability * 30 +
            auto_decision.expected_auto_points * 0.5 +
            (1 if auto_decision.recommended_strategy == AutonomousStrategy.AUTO_WIN_FOCUSED else 0) * 20
        )
        
        # Defensive Disruptor
        scores[PushBackArchetype.DEFENSIVE_DISRUPTOR] = (
            offense_defense.defensive_roi * 0.005 +
            len(offense_defense.critical_control_zones) * 5 +
            (1 if offense_defense.recommended_ratio[1] > 0.5 else 0) * 15
        )
        
        # Parking Specialist
        scores[PushBackArchetype.PARKING_SPECIALIST] = (
            parking_decision.risk_benefit_ratio * 10 +
            parking_decision.expected_parking_value * 0.5 +
            (1 if parking_decision.recommended_timing in [ParkingTiming.SAFE_PARK, ParkingTiming.EARLY_PARK] else 0) * 15
        )
        
        # Balanced Optimizer
        scores[PushBackArchetype.BALANCED_OPTIMIZER] = (
            (block_flow.total_expected_points / 200) * 20 +
            auto_decision.expected_auto_points * 0.3 +
            goal_priority.decision_confidence * 15 +
            (1 if goal_priority.recommended_priority == GoalPriority.BALANCED_GOALS else 0) * 10
        )
        
        # High Risk High Reward
        scores[PushBackArchetype.HIGH_RISK_HIGH_REWARD] = (
            block_flow.total_expected_points * 0.1 +
            auto_decision.auto_win_probability * 25 +
            block_flow.risk_level * 15 +  # Reward high risk
            (1 if offense_defense.recommended_ratio[0] > 0.8 else 0) * 15
        )
        
        # Return archetype with highest score
        best_archetype = max(scores.keys(), key=lambda x: scores[x])
        return best_archetype


if __name__ == "__main__":
    # Test the new Push Back Strategic Analyzer
    analyzer = PushBackStrategyAnalyzer()
    
    print("🚀 TESTING PUSH BACK STRATEGIC ANALYZER")
    print("=" * 60)
    
    # Test robot specifications
    test_robots = [
        analyzer.robot_specs["aggressive"],
        analyzer.robot_specs["balanced"]
    ]
    
    # Test match context
    test_context = {
        "current_state": PushBackMatchState(
            time_remaining=45,
            red_score=85,
            blue_score=78,
            red_blocks_in_goals={"long_1": 8, "long_2": 6, "center_1": 5, "center_2": 4},
            blue_blocks_in_goals={"long_1": 6, "long_2": 8, "center_1": 4, "center_2": 5},
            red_robots_parked=0,
            blue_robots_parked=0,
            autonomous_completed=True,
            phase="driver"
        ),
        "opponent_strategy": "aggressive",
        "match_phase": "mid"
    }
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_push_back_analysis(test_robots, test_context)
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"📊 Results include {len(results)} analysis categories")
    print(f"🎯 {len(results['recommendations'])} strategic recommendations generated")
    print(f"🏆 Best archetype: {results['recommended_archetype'].value}")
    
    # Test Monte Carlo simulation
    print(f"\n🎲 TESTING MONTE CARLO SIMULATION")
    print("-" * 40)
    
    # Pick the recommended archetype strategy
    archetype_strategies = analyzer.create_push_back_archetype_strategies()
    test_strategy = archetype_strategies[results['recommended_archetype']]
    
    # Run smaller simulation for testing
    mc_results = analyzer.run_push_back_monte_carlo(test_strategy, num_simulations=100)
    
    print(f"Strategy: {test_strategy.name}")
    print(f"Win Rate: {mc_results['win_rate']:.1%}")
    print(f"Average Score: {mc_results['avg_score']:.0f}")
    print(f"Score Consistency: {1 - (mc_results['score_std'] / mc_results['avg_score']):.1%}")
    
    print(f"\n🎯 PUSH BACK STRATEGIC ANALYZER READY!")
    print("All 5 strategic decisions implemented and tested successfully.")