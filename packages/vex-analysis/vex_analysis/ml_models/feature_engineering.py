import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
from collections import defaultdict, deque
import warnings

try:
    from ..core.simulator import Zone, ParkingLocation, AllianceStrategy, MatchResult, GameConstants
except ImportError:
    # Fallback for when running from main.py
    from src.core.simulator import Zone, ParkingLocation, AllianceStrategy, MatchResult, GameConstants


class RobotSize(Enum):
    SMALL = "15_inch"  # 15x15x15 inch robot
    LARGE = "24_inch"  # 24x24x24 inch robot


class MatchPhase(Enum):
    AUTONOMOUS = "autonomous"
    DRIVER_CONTROL = "driver_control"
    ENDGAME = "endgame"


@dataclass
class RobotState:
    robot_id: str
    size: RobotSize
    position: Tuple[float, float]  # (x, y) coordinates on field
    zone: Zone
    has_blocks: int = 0
    task: str = "idle"  # "scoring", "defending", "collecting", "idle"
    last_action_time: float = 0.0


@dataclass
class GameState:
    match_time: float  # seconds elapsed
    phase: MatchPhase
    red_robots: List[RobotState] = field(default_factory=list)
    blue_robots: List[RobotState] = field(default_factory=list)
    red_score: int = 0
    blue_score: int = 0
    red_breakdown: Dict[str, int] = field(default_factory=dict)
    blue_breakdown: Dict[str, int] = field(default_factory=dict)
    blocks_in_goals: Dict[str, int] = field(default_factory=lambda: {
        "red_long_1": 0, "red_long_2": 0, 
        "red_center_upper": 0, "red_center_lower": 0,
        "blue_long_1": 0, "blue_long_2": 0,
        "blue_center_upper": 0, "blue_center_lower": 0
    })
    zones_controlled: Dict[str, Zone] = field(default_factory=dict)
    last_scoring_events: List[Tuple[float, str, int]] = field(default_factory=list)  # (time, alliance, points)


class VEXUFeatureExtractor:
    def __init__(self, window_size: int = 10, max_history: int = 1000):
        self.constants = GameConstants()
        self.window_size = window_size  # seconds for temporal windows
        self.max_history = max_history
        self.score_history = defaultdict(lambda: deque(maxlen=max_history))
        self.feature_cache = {}
        
        # VEX U specific constants
        self.GOAL_CAPACITIES = {
            "long": 22,  # Long goals can hold ~22 blocks
            "center_upper": 8,  # Center upper goal capacity
            "center_lower": 10   # Center lower goal capacity
        }
        
        self.GOAL_POINTS = {
            "long": 3,
            "center_upper": 8,
            "center_lower": 6
        }
        
    def extract_all_features(self, game_state: GameState, alliance: str = "red") -> Dict[str, float]:
        features = {}
        
        # 1. Scoring Features
        features.update(self._extract_scoring_features(game_state, alliance))
        
        # 2. VEX U Specific Features  
        features.update(self._extract_vexu_specific_features(game_state, alliance))
        
        # 3. Temporal Features
        features.update(self._extract_temporal_features(game_state, alliance))
        
        # 4. Strategic Features
        features.update(self._extract_strategic_features(game_state, alliance))
        
        return features
    
    def _extract_scoring_features(self, game_state: GameState, alliance: str) -> Dict[str, float]:
        features = {}
        prefix = f"{alliance}_"
        opponent = "blue" if alliance == "red" else "red"
        
        # Current scores and breakdowns
        current_score = game_state.red_score if alliance == "red" else game_state.blue_score
        opponent_score = game_state.blue_score if alliance == "red" else game_state.red_score
        breakdown = game_state.red_breakdown if alliance == "red" else game_state.blue_breakdown
        
        features[f"{prefix}total_score"] = float(current_score)
        features[f"{prefix}score_differential"] = float(current_score - opponent_score)
        
        # Scoring breakdown features
        features[f"{prefix}blocks_points"] = float(breakdown.get('blocks', 0))
        features[f"{prefix}autonomous_points"] = float(breakdown.get('autonomous', 0))
        features[f"{prefix}zone_control_points"] = float(breakdown.get('zones', 0))
        features[f"{prefix}parking_points"] = float(breakdown.get('parking', 0))
        
        # Goal-specific scoring
        for goal_type in ["long_1", "long_2", "center_upper", "center_lower"]:
            goal_key = f"{alliance}_{goal_type}"
            blocks_in_goal = game_state.blocks_in_goals.get(goal_key, 0)
            features[f"{prefix}{goal_type}_blocks"] = float(blocks_in_goal)
            features[f"{prefix}{goal_type}_points"] = float(blocks_in_goal * 3)
        
        # Zone control status
        zones_controlled = 0
        for zone_name, controlling_alliance in game_state.zones_controlled.items():
            if controlling_alliance.value == alliance:
                zones_controlled += 1
        features[f"{prefix}zones_controlled_count"] = float(zones_controlled)
        
        # Parking status (estimated from robot positions)
        robots = game_state.red_robots if alliance == "red" else game_state.blue_robots
        parked_robots = sum(1 for robot in robots if robot.zone.value == f"{alliance}_home")
        features[f"{prefix}parked_robots"] = float(parked_robots)
        
        return features
    
    def _extract_vexu_specific_features(self, game_state: GameState, alliance: str) -> Dict[str, float]:
        features = {}
        prefix = f"{alliance}_"
        
        robots = game_state.red_robots if alliance == "red" else game_state.blue_robots
        
        if len(robots) < 2:
            # Handle case with fewer than 2 robots
            features[f"{prefix}robot_coordination_distance"] = 0.0
            features[f"{prefix}size_diversity"] = 0.0
            features[f"{prefix}task_allocation_efficiency"] = 0.0
            features[f"{prefix}small_robot_utilization"] = 0.0
            features[f"{prefix}large_robot_utilization"] = 0.0
        else:
            # Two-robot coordination metrics
            robot1, robot2 = robots[0], robots[1]
            distance = np.sqrt((robot1.position[0] - robot2.position[0])**2 + 
                             (robot1.position[1] - robot2.position[1])**2)
            features[f"{prefix}robot_coordination_distance"] = float(distance)
            
            # Size diversity (1.0 if different sizes, 0.0 if same)
            size_diversity = 1.0 if robot1.size != robot2.size else 0.0
            features[f"{prefix}size_diversity"] = size_diversity
            
            # Task allocation efficiency
            unique_tasks = len(set([robot1.task, robot2.task]))
            task_efficiency = float(unique_tasks) / 2.0  # 1.0 if different tasks, 0.5 if same
            features[f"{prefix}task_allocation_efficiency"] = task_efficiency
            
            # Robot size utilization
            small_robots = sum(1 for r in robots if r.size == RobotSize.SMALL)
            large_robots = sum(1 for r in robots if r.size == RobotSize.LARGE)
            features[f"{prefix}small_robot_utilization"] = float(small_robots) / len(robots)
            features[f"{prefix}large_robot_utilization"] = float(large_robots) / len(robots)
        
        # Match phase indicators
        features[f"{prefix}is_autonomous"] = 1.0 if game_state.phase == MatchPhase.AUTONOMOUS else 0.0
        features[f"{prefix}is_driver_control"] = 1.0 if game_state.phase == MatchPhase.DRIVER_CONTROL else 0.0
        features[f"{prefix}is_endgame"] = 1.0 if game_state.phase == MatchPhase.ENDGAME else 0.0
        
        # Match time normalized features
        total_time = 120.0  # 2 minutes total
        features[f"{prefix}match_progress"] = min(game_state.match_time / total_time, 1.0)
        features[f"{prefix}time_remaining"] = max((total_time - game_state.match_time) / total_time, 0.0)
        
        # Autonomous vs driver control time ratios
        if game_state.match_time <= 15.0:
            features[f"{prefix}auto_time_remaining"] = (15.0 - game_state.match_time) / 15.0
        else:
            features[f"{prefix}auto_time_remaining"] = 0.0
        
        return features
    
    def _extract_temporal_features(self, game_state: GameState, alliance: str) -> Dict[str, float]:
        features = {}
        prefix = f"{alliance}_"
        
        current_time = game_state.match_time
        window_start = max(0, current_time - self.window_size)
        
        # Scoring rate in last window
        recent_scoring_events = [
            event for event in game_state.last_scoring_events
            if event[0] >= window_start and event[1] == alliance
        ]
        
        if self.window_size > 0:
            scoring_rate = sum(event[2] for event in recent_scoring_events) / self.window_size
        else:
            scoring_rate = 0.0
        features[f"{prefix}scoring_rate_10s"] = scoring_rate
        
        # Time since last scoring event
        last_score_time = 0.0
        for event in reversed(game_state.last_scoring_events):
            if event[1] == alliance:
                last_score_time = event[0]
                break
        
        time_since_last_score = current_time - last_score_time
        features[f"{prefix}time_since_last_score"] = time_since_last_score
        
        # Score momentum (rate of change in score differential)
        current_differential = (game_state.red_score - game_state.blue_score) if alliance == "red" else (game_state.blue_score - game_state.red_score)
        
        # Store current differential for momentum calculation
        key = f"{alliance}_differential"
        self.score_history[key].append((current_time, current_differential))
        
        # Calculate momentum over last 20 seconds
        momentum_window = 20.0
        momentum_start = current_time - momentum_window
        relevant_history = [(t, diff) for t, diff in self.score_history[key] if t >= momentum_start]
        
        if len(relevant_history) >= 2:
            old_diff = relevant_history[0][1]
            momentum = (current_differential - old_diff) / momentum_window
        else:
            momentum = 0.0
        
        features[f"{prefix}score_momentum"] = momentum
        
        # Zone control duration (simplified - assumes current control has been maintained)
        controlled_zones = sum(1 for zone, controller in game_state.zones_controlled.items() 
                             if controller.value == alliance)
        features[f"{prefix}zone_control_duration"] = float(controlled_zones * current_time)
        
        return features
    
    def _extract_strategic_features(self, game_state: GameState, alliance: str) -> Dict[str, float]:
        features = {}
        prefix = f"{alliance}_"
        
        # Goal saturation levels
        total_saturation = 0.0
        goal_count = 0
        
        for goal_type in ["long_1", "long_2", "center_upper", "center_lower"]:
            goal_key = f"{alliance}_{goal_type}"
            blocks_in_goal = game_state.blocks_in_goals.get(goal_key, 0)
            
            if "long" in goal_type:
                capacity = self.GOAL_CAPACITIES["long"]
            elif "upper" in goal_type:
                capacity = self.GOAL_CAPACITIES["center_upper"]
            else:
                capacity = self.GOAL_CAPACITIES["center_lower"]
            
            saturation = min(float(blocks_in_goal) / capacity, 1.0)
            features[f"{prefix}{goal_type}_saturation"] = saturation
            total_saturation += saturation
            goal_count += 1
        
        features[f"{prefix}average_goal_saturation"] = total_saturation / goal_count if goal_count > 0 else 0.0
        
        # Field position dominance
        robots = game_state.red_robots if alliance == "red" else game_state.blue_robots
        neutral_robots = sum(1 for robot in robots if robot.zone == Zone.NEUTRAL)
        home_robots = sum(1 for robot in robots if robot.zone.value == f"{alliance}_home")
        
        if len(robots) > 0:
            features[f"{prefix}neutral_zone_presence"] = float(neutral_robots) / len(robots)
            features[f"{prefix}home_zone_presence"] = float(home_robots) / len(robots)
        else:
            features[f"{prefix}neutral_zone_presence"] = 0.0
            features[f"{prefix}home_zone_presence"] = 0.0
        
        # Defensive positioning (robots in opponent's zone)
        opponent_zone = f"{'blue' if alliance == 'red' else 'red'}_home"
        defensive_robots = sum(1 for robot in robots if robot.zone.value == opponent_zone)
        
        if len(robots) > 0:
            features[f"{prefix}defensive_positioning"] = float(defensive_robots) / len(robots)
        else:
            features[f"{prefix}defensive_positioning"] = 0.0
        
        # Block collection efficiency (robots carrying blocks)
        robots_with_blocks = sum(robot.has_blocks for robot in robots)
        features[f"{prefix}block_carrying_efficiency"] = float(robots_with_blocks)
        
        # Strategic goal targeting (focusing on high-value goals)
        total_blocks = sum(game_state.blocks_in_goals.get(f"{alliance}_{goal}", 0) 
                          for goal in ["long_1", "long_2", "center_upper", "center_lower"])
        
        if total_blocks > 0:
            high_value_blocks = (game_state.blocks_in_goals.get(f"{alliance}_center_upper", 0) + 
                               game_state.blocks_in_goals.get(f"{alliance}_center_lower", 0))
            features[f"{prefix}high_value_goal_focus"] = float(high_value_blocks) / total_blocks
        else:
            features[f"{prefix}high_value_goal_focus"] = 0.0
        
        return features
    
    def normalize_features(self, features: Dict[str, float], 
                          normalization_params: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, float]:
        if normalization_params is None:
            # Auto-generate normalization parameters
            normalization_params = self._generate_normalization_params()
        
        normalized_features = {}
        
        for feature_name, value in features.items():
            if feature_name in normalization_params:
                min_val, max_val = normalization_params[feature_name]
                if max_val > min_val:
                    normalized_value = (value - min_val) / (max_val - min_val)
                    normalized_value = max(0.0, min(1.0, normalized_value))  # Clamp to [0, 1]
                else:
                    normalized_value = 0.0
            else:
                # Default normalization for unknown features
                normalized_value = max(0.0, min(1.0, value))
            
            normalized_features[feature_name] = normalized_value
        
        return normalized_features
    
    def _generate_normalization_params(self) -> Dict[str, Tuple[float, float]]:
        params = {}
        
        # Score-related features (0 to reasonable max)
        score_features = ["total_score", "blocks_points", "zone_control_points", "parking_points"]
        for alliance in ["red", "blue"]:
            for feature in score_features:
                params[f"{alliance}_{feature}"] = (0.0, 200.0)  # Max reasonable score ~200
        
        # Score differential (-200 to 200)
        for alliance in ["red", "blue"]:
            params[f"{alliance}_score_differential"] = (-200.0, 200.0)
        
        # Block counts (0 to max blocks per goal)
        for alliance in ["red", "blue"]:
            for goal in ["long_1", "long_2"]:
                params[f"{alliance}_{goal}_blocks"] = (0.0, 22.0)
            for goal in ["center_upper", "center_lower"]:
                params[f"{alliance}_{goal}_blocks"] = (0.0, 10.0)
        
        # Ratios and percentages (0 to 1)
        ratio_features = ["saturation", "utilization", "presence", "efficiency", "focus", "progress"]
        for alliance in ["red", "blue"]:
            for feature_name, value in params.items():
                if alliance in feature_name and any(ratio in feature_name for ratio in ratio_features):
                    params[feature_name] = (0.0, 1.0)
        
        # Distance features (0 to field diagonal)
        for alliance in ["red", "blue"]:
            params[f"{alliance}_robot_coordination_distance"] = (0.0, 20.0)  # Field diagonal ~20 feet
        
        # Time features
        for alliance in ["red", "blue"]:
            params[f"{alliance}_time_since_last_score"] = (0.0, 120.0)
            params[f"{alliance}_zone_control_duration"] = (0.0, 120.0)
        
        # Momentum features (-10 to 10 points per second)
        for alliance in ["red", "blue"]:
            params[f"{alliance}_score_momentum"] = (-10.0, 10.0)
        
        return params
    
    def handle_missing_data(self, features: Dict[str, float], 
                          default_values: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        if default_values is None:
            default_values = self._get_default_values()
        
        cleaned_features = {}
        
        for feature_name, value in features.items():
            if np.isnan(value) or np.isinf(value):
                if feature_name in default_values:
                    cleaned_features[feature_name] = default_values[feature_name]
                else:
                    cleaned_features[feature_name] = 0.0
                    warnings.warn(f"Missing data for {feature_name}, using 0.0 as default")
            else:
                cleaned_features[feature_name] = value
        
        return cleaned_features
    
    def _get_default_values(self) -> Dict[str, float]:
        return {
            "total_score": 0.0,
            "score_differential": 0.0,
            "blocks_points": 0.0,
            "autonomous_points": 0.0,
            "zone_control_points": 0.0,
            "parking_points": 0.0,
            "robot_coordination_distance": 10.0,  # Medium distance
            "size_diversity": 0.5,  # Neutral
            "task_allocation_efficiency": 0.5,  # Neutral
            "scoring_rate_10s": 0.0,
            "time_since_last_score": 60.0,  # Medium time
            "score_momentum": 0.0,
            "average_goal_saturation": 0.0,
            "neutral_zone_presence": 0.5,  # Neutral positioning
            "defensive_positioning": 0.0,
        }
    
    def generate_feature_importance(self, feature_data: List[Dict[str, float]], 
                                  target_values: List[float]) -> Dict[str, float]:
        if len(feature_data) == 0 or len(target_values) == 0:
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(feature_data)
        target = pd.Series(target_values)
        
        importance_scores = {}
        
        # Calculate correlation-based importance
        for column in df.columns:
            if df[column].var() > 0:  # Only calculate for features with variance
                correlation = abs(df[column].corr(target))
                if not np.isnan(correlation):
                    importance_scores[column] = correlation
                else:
                    importance_scores[column] = 0.0
            else:
                importance_scores[column] = 0.0
        
        # Normalize importance scores
        max_importance = max(importance_scores.values()) if importance_scores else 1.0
        if max_importance > 0:
            importance_scores = {k: v / max_importance for k, v in importance_scores.items()}
        
        return importance_scores
    
    def extract_match_summary_features(self, match_states: List[GameState], 
                                     alliance: str) -> Dict[str, float]:
        if not match_states:
            return {}
        
        summary_features = {}
        prefix = f"{alliance}_"
        
        # Final scores and outcomes
        final_state = match_states[-1]
        final_score = final_state.red_score if alliance == "red" else final_state.blue_score
        opponent_score = final_state.blue_score if alliance == "red" else final_state.red_score
        
        summary_features[f"{prefix}final_score"] = float(final_score)
        summary_features[f"{prefix}final_differential"] = float(final_score - opponent_score)
        summary_features[f"{prefix}match_won"] = 1.0 if final_score > opponent_score else 0.0
        
        # Performance over time
        scores = []
        for state in match_states:
            score = state.red_score if alliance == "red" else state.blue_score
            scores.append(score)
        
        if len(scores) > 1:
            max_score = max(scores)
            min_score = min(scores)
            score_variance = np.var(scores)
            
            summary_features[f"{prefix}max_score"] = float(max_score)
            summary_features[f"{prefix}score_range"] = float(max_score - min_score)
            summary_features[f"{prefix}score_consistency"] = float(1.0 / (1.0 + score_variance))
        
        return summary_features


# Integration helper functions
def create_game_state_from_strategy(strategy: AllianceStrategy, 
                                  opponent_strategy: AllianceStrategy,
                                  match_time: float = 120.0) -> GameState:
    game_state = GameState(match_time=match_time, phase=MatchPhase.DRIVER_CONTROL)
    
    # Create robot states (simplified)
    game_state.red_robots = [
        RobotState("red_1", RobotSize.LARGE, (2.0, 2.0), Zone.RED_HOME),
        RobotState("red_2", RobotSize.SMALL, (3.0, 2.0), Zone.RED_HOME)
    ]
    
    game_state.blue_robots = [
        RobotState("blue_1", RobotSize.LARGE, (10.0, 10.0), Zone.BLUE_HOME),
        RobotState("blue_2", RobotSize.SMALL, (9.0, 10.0), Zone.BLUE_HOME)
    ]
    
    # Set blocks in goals from strategies
    for goal in ["long_1", "long_2", "center_1", "center_2"]:
        red_blocks = (strategy.blocks_scored_auto.get(goal, 0) + 
                     strategy.blocks_scored_driver.get(goal, 0))
        blue_blocks = (opponent_strategy.blocks_scored_auto.get(goal, 0) + 
                      opponent_strategy.blocks_scored_driver.get(goal, 0))
        
        # Map to correct goal names
        if goal == "center_1":
            game_state.blocks_in_goals["red_center_upper"] = red_blocks
            game_state.blocks_in_goals["blue_center_upper"] = blue_blocks
        elif goal == "center_2":
            game_state.blocks_in_goals["red_center_lower"] = red_blocks  
            game_state.blocks_in_goals["blue_center_lower"] = blue_blocks
        else:
            game_state.blocks_in_goals[f"red_{goal}"] = red_blocks
            game_state.blocks_in_goals[f"blue_{goal}"] = blue_blocks
    
    return game_state