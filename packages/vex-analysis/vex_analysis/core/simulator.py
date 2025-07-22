import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, NamedTuple
from enum import Enum


class Zone(Enum):
    RED_HOME = "red_home"
    BLUE_HOME = "blue_home"
    NEUTRAL = "neutral"


class ParkingLocation(Enum):
    NONE = "none"
    ALLIANCE_ZONE = "alliance_zone"
    PLATFORM = "platform"


class BlockColor(Enum):
    RED = "red"
    BLUE = "blue"


class GoalType(Enum):
    LONG_GOAL = "long_goal"
    CENTER_UPPER = "center_upper"
    CENTER_LOWER = "center_lower"


@dataclass
class PushBackBlock:
    """Push Back specific block with exact properties needed"""
    color: BlockColor
    x: float
    y: float
    scored: bool = False
    goal_id: Optional[str] = None


class Position(NamedTuple):
    x: float
    y: float


@dataclass
class LongGoal:
    """Long Goal with 22 block capacity and 10 point zone control"""
    id: str
    position: Position
    alliance_color: BlockColor
    capacity: int = 22
    zone_control_points: int = 10
    blocks: List[PushBackBlock] = field(default_factory=list)
    
    def is_controlled(self) -> bool:
        """Goal is controlled if alliance has ≥3 blocks and more than opponent"""
        alliance_blocks = sum(1 for block in self.blocks if block.color == self.alliance_color)
        opponent_blocks = len(self.blocks) - alliance_blocks
        return alliance_blocks >= 3 and alliance_blocks > opponent_blocks
    
    def get_scored_blocks(self, color: BlockColor) -> int:
        return sum(1 for block in self.blocks if block.color == color)


@dataclass
class CenterGoal:
    """Center Goal with separate upper (8 blocks, 8 points) and lower (7 blocks, 6 points) sections"""
    id: str
    position: Position
    alliance_color: BlockColor
    upper_capacity: int = 8
    lower_capacity: int = 7
    upper_control_points: int = 8
    lower_control_points: int = 6
    upper_blocks: List[PushBackBlock] = field(default_factory=list)
    lower_blocks: List[PushBackBlock] = field(default_factory=list)
    
    def is_upper_controlled(self) -> bool:
        """Upper section controlled if alliance has ≥3 blocks and more than opponent"""
        alliance_blocks = sum(1 for block in self.upper_blocks if block.color == self.alliance_color)
        opponent_blocks = len(self.upper_blocks) - alliance_blocks
        return alliance_blocks >= 3 and alliance_blocks > opponent_blocks
    
    def is_lower_controlled(self) -> bool:
        """Lower section controlled if alliance has ≥3 blocks and more than opponent"""
        alliance_blocks = sum(1 for block in self.lower_blocks if block.color == self.alliance_color)
        opponent_blocks = len(self.lower_blocks) - alliance_blocks
        return alliance_blocks >= 3 and alliance_blocks > opponent_blocks
    
    def get_upper_blocks(self, color: BlockColor) -> int:
        return sum(1 for block in self.upper_blocks if block.color == color)
    
    def get_lower_blocks(self, color: BlockColor) -> int:
        return sum(1 for block in self.lower_blocks if block.color == color)


@dataclass
class ParkZone:
    """Parking zone for endgame scoring"""
    position: Position
    radius: float = 1.0


@dataclass
class PushBackConstants:
    """Hardcoded Push Back game constants for optimal performance"""
    # Field specifications
    TOTAL_BLOCKS: int = 88
    TOTAL_GOALS: int = 4
    LONG_GOALS: int = 2
    CENTER_GOALS: int = 2
    LOADERS: int = 4
    PARK_ZONES: int = 2
    
    # Timing
    AUTONOMOUS_TIME: int = 30
    DRIVER_TIME: int = 90
    TOTAL_TIME: int = 120
    
    # Scoring constants
    POINTS_PER_BLOCK: int = 3
    AUTONOMOUS_BONUS: int = 10
    ROBOTS_PER_ALLIANCE: int = 2
    
    # Parking points
    ONE_ROBOT_PARKING: int = 8
    TWO_ROBOT_PARKING: int = 30
    
    # Zone control points
    LONG_GOAL_CONTROL: int = 10
    CENTER_UPPER_CONTROL: int = 8
    CENTER_LOWER_CONTROL: int = 6
    
    # Goal capacities
    LONG_GOAL_CAPACITY: int = 22
    CENTER_UPPER_CAPACITY: int = 8
    CENTER_LOWER_CAPACITY: int = 7
    
    # Autonomous win conditions
    AUTO_WIN_MIN_BLOCKS: int = 7
    AUTO_WIN_MIN_GOALS: int = 3
    AUTO_WIN_MIN_LOADER_BLOCKS: int = 3


@dataclass
class PushBackField:
    """Hardcoded Push Back field layout"""
    long_goal_1: LongGoal
    long_goal_2: LongGoal
    center_goal_red: CenterGoal
    center_goal_blue: CenterGoal
    park_zone_red: ParkZone
    park_zone_blue: ParkZone
    blocks: List[PushBackBlock] = field(default_factory=lambda: [])
    
    @classmethod
    def create_standard_field(cls) -> 'PushBackField':
        """Create standard Push Back field with exact specifications"""
        # Long goals at opposite ends
        long_goal_1 = LongGoal(
            id="long_1",
            position=Position(0.0, 6.0),
            alliance_color=BlockColor.RED
        )
        long_goal_2 = LongGoal(
            id="long_2",
            position=Position(12.0, 6.0),
            alliance_color=BlockColor.BLUE
        )
        
        # Center goals in middle
        center_goal_red = CenterGoal(
            id="center_red",
            position=Position(6.0, 3.0),
            alliance_color=BlockColor.RED
        )
        center_goal_blue = CenterGoal(
            id="center_blue",
            position=Position(6.0, 9.0),
            alliance_color=BlockColor.BLUE
        )
        
        # Park zones
        park_zone_red = ParkZone(Position(2.0, 2.0))
        park_zone_blue = ParkZone(Position(10.0, 10.0))
        
        # Create 88 blocks distributed across field
        blocks = []
        for i in range(88):
            # Distribute blocks across field (simplified positioning)
            x = (i % 12) + 0.5
            y = (i // 12) + 0.5
            color = BlockColor.RED if i < 44 else BlockColor.BLUE
            blocks.append(PushBackBlock(color, x, y))
        
        return cls(
            long_goal_1=long_goal_1,
            long_goal_2=long_goal_2,
            center_goal_red=center_goal_red,
            center_goal_blue=center_goal_blue,
            park_zone_red=park_zone_red,
            park_zone_blue=park_zone_blue,
            blocks=blocks
        )


@dataclass
class AllianceStrategy:
    name: str
    blocks_scored_auto: Dict[str, int]  # goal_name -> blocks
    blocks_scored_driver: Dict[str, int]  # goal_name -> blocks
    zones_controlled: List[Zone]
    robots_parked: List[ParkingLocation]
    wins_auto: bool = False
    
    # Additional Push Back specific data
    loader_blocks_removed: int = 0
    park_zone_contact_auto: bool = False


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


class PushBackScoringEngine:
    """Optimized Push Back specific scoring engine"""
    
    def __init__(self, enable_feature_extraction: bool = True):
        self.constants = PushBackConstants()
        self.field = PushBackField.create_standard_field()
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
    
    def calculate_push_back_score(
        self,
        alliance_color: BlockColor,
        blocks_in_goals: Dict[str, int],
        robots_parked: List[ParkingLocation],
        wins_auto: bool = False,
        autonomous_win_eligible: bool = False
    ) -> Tuple[int, Dict[str, int]]:
        """Calculate Push Back score with exact rules implementation"""
        breakdown = {}
        
        # 1. Block scoring - 3 points per scored block
        total_blocks = sum(blocks_in_goals.values())
        block_points = total_blocks * self.constants.POINTS_PER_BLOCK
        breakdown['blocks'] = block_points
        
        # 2. Autonomous bonus - 10 points for higher score at 15 seconds
        auto_bonus = self.constants.AUTONOMOUS_BONUS if wins_auto else 0
        breakdown['autonomous_bonus'] = auto_bonus
        
        # 3. Goal control points - hardcoded Push Back specific
        goal_control_points = 0
        
        # Long Goal 1 control (10 points)
        long_1_ally = blocks_in_goals.get('long_1', 0)
        long_1_opp = 0  # Will be calculated from opponent perspective
        if long_1_ally >= 3:  # Simplified control logic
            goal_control_points += self.constants.LONG_GOAL_CONTROL
        
        # Long Goal 2 control (10 points) 
        long_2_ally = blocks_in_goals.get('long_2', 0)
        if long_2_ally >= 3:
            goal_control_points += self.constants.LONG_GOAL_CONTROL
            
        # Center Goal Upper control (8 points)
        center_upper_ally = blocks_in_goals.get('center_upper', blocks_in_goals.get('center_1', 0))
        if center_upper_ally >= 3:
            goal_control_points += self.constants.CENTER_UPPER_CONTROL
            
        # Center Goal Lower control (6 points)
        center_lower_ally = blocks_in_goals.get('center_lower', blocks_in_goals.get('center_2', 0))
        if center_lower_ally >= 3:
            goal_control_points += self.constants.CENTER_LOWER_CONTROL
        
        breakdown['goal_control'] = goal_control_points
        
        # 4. Parking points
        parking_points = 0
        parked_count = sum(1 for loc in robots_parked if loc != ParkingLocation.NONE)
        
        if parked_count == 1:
            parking_points = self.constants.ONE_ROBOT_PARKING
        elif parked_count == 2:
            parking_points = self.constants.TWO_ROBOT_PARKING
        
        breakdown['parking'] = parking_points
        
        # 5. Autonomous Win Point (7 points)
        auto_win_point = 7 if autonomous_win_eligible else 0
        breakdown['autonomous_win'] = auto_win_point
        
        total_score = (
            block_points + 
            auto_bonus + 
            goal_control_points + 
            parking_points + 
            auto_win_point
        )
        breakdown['total'] = total_score
        
        return total_score, breakdown
    
    def check_autonomous_win_eligibility(
        self,
        strategy: AllianceStrategy
    ) -> bool:
        """Check if alliance qualifies for autonomous win point"""
        auto_blocks = sum(strategy.blocks_scored_auto.values())
        
        # Must score ≥7 blocks
        if auto_blocks < self.constants.AUTO_WIN_MIN_BLOCKS:
            return False
        
        # Must score in ≥3 different goals
        goals_with_blocks = sum(1 for blocks in strategy.blocks_scored_auto.values() if blocks > 0)
        if goals_with_blocks < self.constants.AUTO_WIN_MIN_GOALS:
            return False
        
        # Must remove ≥3 blocks from loaders
        if strategy.loader_blocks_removed < self.constants.AUTO_WIN_MIN_LOADER_BLOCKS:
            return False
        
        # Must not contact park zone during autonomous
        if strategy.park_zone_contact_auto:
            return False
        
        return True
    
    def simulate_push_back_match(
        self,
        red_strategy: AllianceStrategy,
        blue_strategy: AllianceStrategy,
        extract_features: bool = None
    ) -> MatchResult:
        """Simulate Push Back match with exact scoring rules"""
        
        # Calculate autonomous scores at 15 seconds to determine bonus winner
        red_auto_score = sum(red_strategy.blocks_scored_auto.values()) * self.constants.POINTS_PER_BLOCK
        blue_auto_score = sum(blue_strategy.blocks_scored_auto.values()) * self.constants.POINTS_PER_BLOCK
        
        # Determine autonomous bonus winner
        if red_auto_score > blue_auto_score:
            red_strategy.wins_auto = True
            blue_strategy.wins_auto = False
        elif blue_auto_score > red_auto_score:
            red_strategy.wins_auto = False
            blue_strategy.wins_auto = True
        else:
            red_strategy.wins_auto = False
            blue_strategy.wins_auto = False
        
        # Check autonomous win point eligibility
        red_auto_win_eligible = self.check_autonomous_win_eligibility(red_strategy)
        blue_auto_win_eligible = self.check_autonomous_win_eligibility(blue_strategy)
        
        # Combine autonomous and driver blocks for total scoring
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
        
        # Calculate final scores using Push Back rules
        red_score, red_breakdown = self.calculate_push_back_score(
            BlockColor.RED,
            red_total_blocks,
            red_strategy.robots_parked,
            red_strategy.wins_auto,
            red_auto_win_eligible
        )
        
        blue_score, blue_breakdown = self.calculate_push_back_score(
            BlockColor.BLUE,
            blue_total_blocks,
            blue_strategy.robots_parked,
            blue_strategy.wins_auto,
            blue_auto_win_eligible
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
    
    # Keep legacy method for backward compatibility
    def simulate_match(self, red_strategy, blue_strategy, extract_features=None):
        """Legacy method - redirects to Push Back specific implementation"""
        return self.simulate_push_back_match(red_strategy, blue_strategy, extract_features)
    
    def get_winner(self, red_score: int, blue_score: int) -> Tuple[str, int]:
        if red_score > blue_score:
            return "red", red_score - blue_score
        elif blue_score > red_score:
            return "blue", blue_score - red_score
        else:
            return "tie", 0
    
    def validate_push_back_strategy(self, strategy: AllianceStrategy) -> Tuple[bool, List[str]]:
        """Validate strategy against Push Back rules with detailed feedback"""
        errors = []
        
        # Check total blocks don't exceed field capacity
        total_blocks = sum(strategy.blocks_scored_auto.values()) + sum(strategy.blocks_scored_driver.values())
        if total_blocks > 44:  # Each alliance can score max 44 blocks
            errors.append(f"Total blocks ({total_blocks}) exceeds alliance maximum (44)")
        
        # Check goal capacities
        for goal, blocks in {**strategy.blocks_scored_auto, **strategy.blocks_scored_driver}.items():
            auto_blocks = strategy.blocks_scored_auto.get(goal, 0)
            driver_blocks = strategy.blocks_scored_driver.get(goal, 0)
            total_goal_blocks = auto_blocks + driver_blocks
            
            if goal in ['long_1', 'long_2'] and total_goal_blocks > self.constants.LONG_GOAL_CAPACITY:
                errors.append(f"{goal} blocks ({total_goal_blocks}) exceed capacity ({self.constants.LONG_GOAL_CAPACITY})")
            elif goal == 'center_1' and total_goal_blocks > self.constants.CENTER_UPPER_CAPACITY:
                errors.append(f"center_1 blocks ({total_goal_blocks}) exceed upper capacity ({self.constants.CENTER_UPPER_CAPACITY})")
            elif goal == 'center_2' and total_goal_blocks > self.constants.CENTER_LOWER_CAPACITY:
                errors.append(f"center_2 blocks ({total_goal_blocks}) exceed lower capacity ({self.constants.CENTER_LOWER_CAPACITY})")
        
        # Check robot count
        if len(strategy.robots_parked) > self.constants.ROBOTS_PER_ALLIANCE:
            errors.append(f"Too many robots parked ({len(strategy.robots_parked)} > {self.constants.ROBOTS_PER_ALLIANCE})")
        
        # Check autonomous win conditions consistency
        if strategy.loader_blocks_removed < 0:
            errors.append("Loader blocks removed cannot be negative")
        
        return len(errors) == 0, errors
    
    # Keep legacy method for backward compatibility
    def validate_strategy(self, strategy: AllianceStrategy) -> bool:
        """Legacy validation method"""
        valid, _ = self.validate_push_back_strategy(strategy)
        return valid
    
    def get_detailed_score_breakdown(self, strategy: AllianceStrategy, alliance_color: BlockColor) -> Dict[str, any]:
        """Get detailed breakdown of strategy scoring potential"""
        auto_blocks = sum(strategy.blocks_scored_auto.values())
        driver_blocks = sum(strategy.blocks_scored_driver.values())
        total_blocks = auto_blocks + driver_blocks
        
        # Calculate potential scores
        auto_score = auto_blocks * self.constants.POINTS_PER_BLOCK
        total_block_score = total_blocks * self.constants.POINTS_PER_BLOCK
        
        # Parking potential
        parked_count = sum(1 for loc in strategy.robots_parked if loc != ParkingLocation.NONE)
        parking_score = (
            self.constants.ONE_ROBOT_PARKING if parked_count == 1 
            else self.constants.TWO_ROBOT_PARKING if parked_count == 2 
            else 0
        )
        
        # Goal control potential (simplified)
        goal_control_score = 0
        for goal, blocks in {**strategy.blocks_scored_auto, **strategy.blocks_scored_driver}.items():
            total_goal_blocks = strategy.blocks_scored_auto.get(goal, 0) + strategy.blocks_scored_driver.get(goal, 0)
            if total_goal_blocks >= 3:  # Assume control
                if goal in ['long_1', 'long_2']:
                    goal_control_score += self.constants.LONG_GOAL_CONTROL
                elif goal == 'center_1':
                    goal_control_score += self.constants.CENTER_UPPER_CONTROL
                elif goal == 'center_2':
                    goal_control_score += self.constants.CENTER_LOWER_CONTROL
        
        return {
            'auto_blocks': auto_blocks,
            'driver_blocks': driver_blocks,
            'total_blocks': total_blocks,
            'auto_score': auto_score,
            'total_block_score': total_block_score,
            'parking_score': parking_score,
            'goal_control_score': goal_control_score,
            'autonomous_win_eligible': self.check_autonomous_win_eligibility(strategy),
            'max_possible_score': total_block_score + parking_score + goal_control_score + 
                                (self.constants.AUTONOMOUS_BONUS if auto_score > 0 else 0) +
                                (7 if self.check_autonomous_win_eligibility(strategy) else 0)
        }
    
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


# Maintain backward compatibility alias
ScoringSimulator = PushBackScoringEngine
GameConstants = PushBackConstants


if __name__ == "__main__":
    engine = PushBackScoringEngine()
    
    # Example Push Back match simulation
    red_strategy = AllianceStrategy(
        name="Red Alliance",
        blocks_scored_auto={"long_1": 8, "center_1": 4, "center_2": 3, "long_2": 2},
        blocks_scored_driver={"long_1": 6, "long_2": 8, "center_1": 3, "center_2": 4},
        zones_controlled=[Zone.RED_HOME, Zone.NEUTRAL],
        robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.PLATFORM],
        loader_blocks_removed=5,  # Removed 5 blocks from loaders
        park_zone_contact_auto=False  # No park zone contact in auto
    )
    
    blue_strategy = AllianceStrategy(
        name="Blue Alliance",
        blocks_scored_auto={"long_2": 6, "center_2": 3, "center_1": 2, "long_1": 1},
        blocks_scored_driver={"long_1": 4, "long_2": 10, "center_1": 5, "center_2": 2},
        zones_controlled=[Zone.BLUE_HOME],
        robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.ALLIANCE_ZONE],
        loader_blocks_removed=3,  # Removed 3 blocks from loaders
        park_zone_contact_auto=False
    )
    
    # Validate strategies
    red_valid, red_errors = engine.validate_push_back_strategy(red_strategy)
    blue_valid, blue_errors = engine.validate_push_back_strategy(blue_strategy)
    
    if not red_valid:
        print(f"Red strategy validation errors: {red_errors}")
    if not blue_valid:
        print(f"Blue strategy validation errors: {blue_errors}")
    
    if red_valid and blue_valid:
        result = engine.simulate_push_back_match(red_strategy, blue_strategy)
        
        print(f"\n=== PUSH BACK MATCH RESULT ===")
        print(f"Winner: {result.winner.upper()} by {result.margin} points")
        print(f"\nRed Alliance Score: {result.red_score}")
        print(f"Red Breakdown: {result.red_breakdown}")
        print(f"\nBlue Alliance Score: {result.blue_score}")
        print(f"Blue Breakdown: {result.blue_breakdown}")
        
        # Check autonomous win eligibility
        red_auto_eligible = engine.check_autonomous_win_eligibility(red_strategy)
        blue_auto_eligible = engine.check_autonomous_win_eligibility(blue_strategy)
        print(f"\nRed Auto Win Eligible: {red_auto_eligible}")
        print(f"Blue Auto Win Eligible: {blue_auto_eligible}")
    else:
        print("Match simulation aborted due to invalid strategies")