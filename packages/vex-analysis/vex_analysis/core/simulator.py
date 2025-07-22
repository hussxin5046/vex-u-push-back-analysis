import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
from enum import Enum


class Zone(Enum):
    RED_HOME = "red_home"
    BLUE_HOME = "blue_home"
    NEUTRAL = "neutral"


class ParkingLocation(Enum):
    NONE = "none"
    ALLIANCE_ZONE = "alliance_zone"
    PLATFORM = "platform"


@dataclass
class GameConstants:
    TOTAL_BLOCKS: int = 88
    POINTS_PER_BLOCK: int = 3
    TOTAL_GOALS: int = 4
    LONG_GOALS: int = 2
    CENTER_GOALS: int = 2
    AUTONOMOUS_TIME: int = 30  # Updated to 30s for VEX U
    DRIVER_TIME: int = 90     # Updated to 90s for VEX U
    TOTAL_TIME: int = 120
    AUTONOMOUS_BONUS: int = 10
    ROBOTS_PER_ALLIANCE: int = 2
    
    # Parking points (updated for VEX U Push Back)
    ONE_ROBOT_PARKING: int = 8
    TWO_ROBOT_PARKING: int = 30
    
    # Zone control points (updated for VEX U Push Back)
    LONG_GOAL_CONTROL: int = 10
    CENTER_GOAL_UPPER_CONTROL: int = 8
    CENTER_GOAL_LOWER_CONTROL: int = 6
    
    # Goal capacities
    LONG_GOAL_CAPACITY: int = 22
    CENTER_UPPER_CAPACITY: int = 8
    CENTER_LOWER_CAPACITY: int = 10


@dataclass
class AllianceStrategy:
    name: str
    blocks_scored_auto: Dict[str, int]  # goal_name -> blocks
    blocks_scored_driver: Dict[str, int]  # goal_name -> blocks
    zones_controlled: List[Zone]
    robots_parked: List[ParkingLocation]
    wins_auto: bool = False


@dataclass
class MatchResult:
    red_score: int
    blue_score: int
    red_breakdown: Dict[str, int]
    blue_breakdown: Dict[str, int]
    winner: str
    margin: int
    match_features: Optional[Dict[str, float]] = None
    red_features: Optional[Dict[str, float]] = None
    blue_features: Optional[Dict[str, float]] = None


class ScoringSimulator:
    def __init__(self, enable_feature_extraction: bool = True):
        self.constants = GameConstants()
        self.goal_names = ["long_1", "long_2", "center_1", "center_2"]
        self.enable_feature_extraction = enable_feature_extraction
        self.feature_extractor = None
        
        if self.enable_feature_extraction:
            try:
                from ..ml_models.feature_engineering import VEXUFeatureExtractor
                self.feature_extractor = VEXUFeatureExtractor()
            except ImportError:
                print("Warning: Feature extraction disabled - ml_models not available")
                self.enable_feature_extraction = False
    
    def calculate_score(
        self,
        blocks_in_goals: Dict[str, int],
        zones_controlled: List[Zone],
        robots_parked: List[ParkingLocation],
        wins_auto: bool = False
    ) -> Tuple[int, Dict[str, int]]:
        breakdown = {}
        
        # Block scoring
        total_blocks = sum(blocks_in_goals.values())
        block_points = total_blocks * self.constants.POINTS_PER_BLOCK
        breakdown['blocks'] = block_points
        
        # Autonomous bonus
        auto_points = self.constants.AUTONOMOUS_BONUS if wins_auto else 0
        breakdown['autonomous'] = auto_points
        
        # Zone control points (updated for VEX U Push Back)
        zone_points = 0
        for zone in zones_controlled:
            if zone == Zone.NEUTRAL:  # Neutral zone control
                zone_points += self.constants.LONG_GOAL_CONTROL
        breakdown['zones'] = zone_points
        
        # Goal control points (new for VEX U Push Back)
        goal_control_points = 0
        # This would be calculated based on which goals are controlled
        # For now, simplified implementation
        breakdown['goal_control'] = goal_control_points
        
        # Parking points (updated for VEX U Push Back)
        parking_points = 0
        parked_count = sum(1 for loc in robots_parked if loc != ParkingLocation.NONE)
        
        if parked_count == 1:
            parking_points = self.constants.ONE_ROBOT_PARKING
        elif parked_count == 2:
            parking_points = self.constants.TWO_ROBOT_PARKING
        
        breakdown['parking'] = parking_points
        
        total_score = block_points + auto_points + zone_points + goal_control_points + parking_points
        breakdown['total'] = total_score
        
        return total_score, breakdown
    
    def simulate_match(
        self,
        red_strategy: AllianceStrategy,
        blue_strategy: AllianceStrategy,
        extract_features: bool = None
    ) -> MatchResult:
        # Calculate autonomous scores
        red_auto_blocks = sum(red_strategy.blocks_scored_auto.values())
        blue_auto_blocks = sum(blue_strategy.blocks_scored_auto.values())
        
        # Determine autonomous winner
        if red_auto_blocks > blue_auto_blocks:
            red_strategy.wins_auto = True
            blue_strategy.wins_auto = False
        elif blue_auto_blocks > red_auto_blocks:
            red_strategy.wins_auto = False
            blue_strategy.wins_auto = True
        else:
            red_strategy.wins_auto = False
            blue_strategy.wins_auto = False
        
        # Combine autonomous and driver blocks
        red_total_blocks = {}
        blue_total_blocks = {}
        
        for goal in self.goal_names:
            red_total_blocks[goal] = (
                red_strategy.blocks_scored_auto.get(goal, 0) +
                red_strategy.blocks_scored_driver.get(goal, 0)
            )
            blue_total_blocks[goal] = (
                blue_strategy.blocks_scored_auto.get(goal, 0) +
                blue_strategy.blocks_scored_driver.get(goal, 0)
            )
        
        # Calculate final scores
        red_score, red_breakdown = self.calculate_score(
            red_total_blocks,
            red_strategy.zones_controlled,
            red_strategy.robots_parked,
            red_strategy.wins_auto
        )
        
        blue_score, blue_breakdown = self.calculate_score(
            blue_total_blocks,
            blue_strategy.zones_controlled,
            blue_strategy.robots_parked,
            blue_strategy.wins_auto
        )
        
        # Determine winner
        winner, margin = self.get_winner(red_score, blue_score)
        
        # Extract features if enabled
        match_features = None
        red_features = None
        blue_features = None
        
        extract_features = extract_features if extract_features is not None else self.enable_feature_extraction
        
        if extract_features and self.feature_extractor:
            try:
                from ..ml_models.feature_engineering import create_game_state_from_strategy
                game_state = create_game_state_from_strategy(red_strategy, blue_strategy)
                game_state.red_score = red_score
                game_state.blue_score = blue_score
                game_state.red_breakdown = red_breakdown
                game_state.blue_breakdown = blue_breakdown
                
                red_features = self.feature_extractor.extract_all_features(game_state, "red")
                blue_features = self.feature_extractor.extract_all_features(game_state, "blue")
                
                # Combine features for match-level analysis
                match_features = {**red_features, **blue_features}
                match_features['match_winner'] = 1.0 if winner == "red" else (0.0 if winner == "blue" else 0.5)
                match_features['match_margin'] = float(margin)
                
            except Exception as e:
                print(f"Warning: Feature extraction failed - {e}")
        
        return MatchResult(
            red_score=red_score,
            blue_score=blue_score,
            red_breakdown=red_breakdown,
            blue_breakdown=blue_breakdown,
            winner=winner,
            margin=margin,
            match_features=match_features,
            red_features=red_features,
            blue_features=blue_features
        )
    
    def get_winner(self, red_score: int, blue_score: int) -> Tuple[str, int]:
        if red_score > blue_score:
            return "red", red_score - blue_score
        elif blue_score > red_score:
            return "blue", blue_score - red_score
        else:
            return "tie", 0
    
    def validate_strategy(self, strategy: AllianceStrategy) -> bool:
        total_blocks = 0
        for blocks in strategy.blocks_scored_auto.values():
            total_blocks += blocks
        for blocks in strategy.blocks_scored_driver.values():
            total_blocks += blocks
        
        if total_blocks > self.constants.TOTAL_BLOCKS:
            return False
        
        if len(strategy.robots_parked) > self.constants.ROBOTS_PER_ALLIANCE:
            return False
        
        return True
    
    def create_game_state_snapshot(self, 
                                 red_strategy: AllianceStrategy,
                                 blue_strategy: AllianceStrategy,
                                 match_time: float = 120.0) -> Optional['GameState']:
        """Create a GameState snapshot for feature extraction"""
        if not self.enable_feature_extraction or not self.feature_extractor:
            return None
            
        try:
            from ..ml_models.feature_engineering import create_game_state_from_strategy
            return create_game_state_from_strategy(red_strategy, blue_strategy, match_time)
        except ImportError:
            return None
    
    def extract_features_from_strategies(self, 
                                       red_strategy: AllianceStrategy,
                                       blue_strategy: AllianceStrategy,
                                       alliance: str = "red") -> Optional[Dict[str, float]]:
        """Extract features from alliance strategies"""
        game_state = self.create_game_state_snapshot(red_strategy, blue_strategy)
        if game_state and self.feature_extractor:
            return self.feature_extractor.extract_all_features(game_state, alliance)
        return None


if __name__ == "__main__":
    simulator = ScoringSimulator()
    
    # Example match simulation
    red_strategy = AllianceStrategy(
        name="Red Alliance",
        blocks_scored_auto={"long_1": 5, "center_1": 3},
        blocks_scored_driver={"long_1": 10, "long_2": 8, "center_1": 7, "center_2": 5},
        zones_controlled=[Zone.RED_HOME, Zone.NEUTRAL],
        robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.ALLIANCE_ZONE]
    )
    
    blue_strategy = AllianceStrategy(
        name="Blue Alliance",
        blocks_scored_auto={"long_2": 4, "center_2": 4},
        blocks_scored_driver={"long_1": 6, "long_2": 12, "center_1": 5, "center_2": 9},
        zones_controlled=[Zone.BLUE_HOME],
        robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.PLATFORM]
    )
    
    result = simulator.simulate_match(red_strategy, blue_strategy)
    
    print(f"Match Result: {result.winner.upper()} wins by {result.margin} points")
    print(f"Red Score: {result.red_score} - Breakdown: {result.red_breakdown}")
    print(f"Blue Score: {result.blue_score} - Breakdown: {result.blue_breakdown}")