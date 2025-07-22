"""
Push Back specific data models
Simplified models focused on Push Back game mechanics
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from .base import BaseModel

@dataclass
class RobotSpecs:
    """Push Back robot specifications"""
    id: str
    speed: float  # 0.0 - 1.0
    accuracy: float  # 0.0 - 1.0  
    capacity: int  # Number of blocks robot can hold
    autonomous_capability: float = 0.7  # 0.0 - 1.0
    driver_skill: float = 0.8  # 0.0 - 1.0
    reliability: float = 0.9  # 0.0 - 1.0

@dataclass
class PushBackBlock:
    """Individual Push Back block"""
    id: str
    x: float
    y: float
    z: float = 0.0
    alliance: str = "neutral"  # "red", "blue", "neutral"
    goal_id: Optional[str] = None
    zone_id: Optional[str] = None

@dataclass
class PushBackGoal:
    """Push Back goal (Center or Long)"""
    id: str
    x: float
    y: float
    goal_type: str  # "center" or "long"
    alliance: str  # "red" or "blue"
    blocks: List[PushBackBlock] = None
    
    def __post_init__(self):
        if self.blocks is None:
            self.blocks = []

@dataclass
class PushBackControlZone:
    """Push Back control zone"""
    id: str
    x: float
    y: float
    width: float
    height: float
    alliance: str  # "red" or "blue"
    blocks: List[PushBackBlock] = None
    controlled_by: str = "neutral"  # "red", "blue", "neutral"
    
    def __post_init__(self):
        if self.blocks is None:
            self.blocks = []

@dataclass
class PushBackParkZone:
    """Push Back parking zone"""
    id: str
    x: float
    y: float
    width: float
    height: float
    alliance: str  # "red" or "blue"
    robots_parked: int = 0

@dataclass
class PushBackFieldState:
    """Complete Push Back field state"""
    blocks: List[PushBackBlock]
    goals: List[PushBackGoal]
    control_zones: List[PushBackControlZone]
    park_zones: List[PushBackParkZone]
    time_remaining: float = 105.0  # seconds
    match_phase: str = "autonomous"  # "autonomous", "driver", "endgame"
    
    @classmethod
    def create_initial_field(cls) -> 'PushBackFieldState':
        """Create initial Push Back field state"""
        # Create 88 blocks in neutral positions
        blocks = []
        block_id = 0
        for x in range(1, 12):  # 11 columns
            for y in range(1, 12):  # 11 rows
                if block_id < 88:  # Push Back has exactly 88 blocks
                    blocks.append(PushBackBlock(
                        id=f"block_{block_id}",
                        x=float(x),
                        y=float(y),
                        alliance="neutral"
                    ))
                    block_id += 1
        
        # Create 4 goals
        goals = [
            PushBackGoal(id="red_center", x=6.0, y=2.0, goal_type="center", alliance="red"),
            PushBackGoal(id="red_long", x=2.0, y=6.0, goal_type="long", alliance="red"),
            PushBackGoal(id="blue_center", x=6.0, y=10.0, goal_type="center", alliance="blue"),
            PushBackGoal(id="blue_long", x=10.0, y=6.0, goal_type="long", alliance="blue")
        ]
        
        # Create 2 control zones
        control_zones = [
            PushBackControlZone(id="red_zone", x=3.0, y=3.0, width=3.0, height=3.0, alliance="red"),
            PushBackControlZone(id="blue_zone", x=6.0, y=6.0, width=3.0, height=3.0, alliance="blue")
        ]
        
        # Create 2 park zones
        park_zones = [
            PushBackParkZone(id="red_park", x=1.0, y=1.0, width=2.0, height=2.0, alliance="red"),
            PushBackParkZone(id="blue_park", x=9.0, y=9.0, width=2.0, height=2.0, alliance="blue")
        ]
        
        return cls(
            blocks=blocks,
            goals=goals,
            control_zones=control_zones,
            park_zones=park_zones,
            time_remaining=105.0,
            match_phase="autonomous"
        )

@dataclass
class PushBackStrategy:
    """Push Back strategy definition"""
    id: str
    name: str
    archetype: str  # Strategy archetype
    robot_specs: List[RobotSpecs]
    autonomous_strategy: str
    driver_strategy: str
    endgame_strategy: str = "parking_focus"
    priority_sequence: List[str] = None  # Ordered list of priorities
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.priority_sequence is None:
            self.priority_sequence = ["block_collection", "goal_scoring", "control_zones"]
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class PushBackMatchState:
    """Current state during a Push Back match"""
    field_state: PushBackFieldState
    red_score: int = 0
    blue_score: int = 0
    red_breakdown: Dict[str, int] = None
    blue_breakdown: Dict[str, int] = None
    autonomous_completed: bool = False
    auto_win_achieved: str = "none"  # "red", "blue", "none"
    
    def __post_init__(self):
        if self.red_breakdown is None:
            self.red_breakdown = {"blocks": 0, "control_zones": 0, "parking": 0, "autonomous": 0}
        if self.blue_breakdown is None:
            self.blue_breakdown = {"blocks": 0, "control_zones": 0, "parking": 0, "autonomous": 0}

@dataclass
class BlockFlowOptimization:
    """Block flow optimization results"""
    optimal_distribution: Dict[str, int]
    expected_points: float
    risk_level: str  # "low", "medium", "high"
    efficiency_score: float
    recommendations: List[str]
    bottlenecks: List[str] = None
    
    def __post_init__(self):
        if self.bottlenecks is None:
            self.bottlenecks = []

@dataclass
class AutonomousDecision:
    """Autonomous strategy decision analysis"""
    recommended_strategy: str
    auto_win_probability: float
    bonus_probability: float
    expected_points: float
    block_targets: Dict[str, int]
    risk_assessment: str
    time_allocation: Dict[str, float] = None
    
    def __post_init__(self):
        if self.time_allocation is None:
            self.time_allocation = {"collection": 12.0, "scoring": 2.0, "positioning": 1.0}

@dataclass  
class GoalPriorityAnalysis:
    """Goal priority strategy analysis"""
    recommended_priority: str  # "center_first", "long_first", "balanced"
    center_goal_value: float
    long_goal_value: float
    optimal_sequence: List[str]
    decision_confidence: float
    matchup_considerations: Dict[str, str] = None
    
    def __post_init__(self):
        if self.matchup_considerations is None:
            self.matchup_considerations = {}

@dataclass
class ParkingDecisionAnalysis:
    """Parking decision timing analysis"""
    recommended_timing: str
    one_robot_threshold: float  # Score threshold for parking one robot
    two_robot_threshold: float  # Score threshold for parking both robots
    expected_value: float
    risk_benefit_ratio: float
    situational_recommendations: Dict[str, str]

@dataclass
class OffenseDefenseBalance:
    """Offense vs Defense balance analysis"""
    recommended_ratio: Tuple[float, float]  # (offense, defense)
    offensive_roi: float  # Return on investment for offensive actions
    defensive_roi: float  # Return on investment for defensive actions
    critical_zones: List[str]
    disruption_targets: List[str]
    phase_recommendations: Dict[str, str] = None
    
    def __post_init__(self):
        if self.phase_recommendations is None:
            self.phase_recommendations = {
                "autonomous": "offense_focus",
                "driver": "balanced",
                "endgame": "situational"
            }

@dataclass
class PushBackAnalysisResult:
    """Comprehensive Push Back analysis result"""
    analysis_id: str
    strategy: PushBackStrategy
    robot_specs: List[RobotSpecs]
    block_flow_optimization: BlockFlowOptimization
    autonomous_decision: AutonomousDecision
    goal_priority_analysis: GoalPriorityAnalysis
    parking_decision_analysis: ParkingDecisionAnalysis
    offense_defense_balance: OffenseDefenseBalance
    recommended_archetype: str
    overall_score: float
    recommendations: List[str]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "analysis_id": self.analysis_id,
            "strategy": {
                "id": self.strategy.id,
                "name": self.strategy.name,
                "archetype": self.strategy.archetype,
                "autonomous_strategy": self.strategy.autonomous_strategy,
                "driver_strategy": self.strategy.driver_strategy
            },
            "robot_specs": [
                {
                    "id": robot.id,
                    "speed": robot.speed,
                    "accuracy": robot.accuracy,
                    "capacity": robot.capacity
                } for robot in self.robot_specs
            ],
            "block_flow_optimization": {
                "optimal_distribution": self.block_flow_optimization.optimal_distribution,
                "expected_points": self.block_flow_optimization.expected_points,
                "risk_level": self.block_flow_optimization.risk_level,
                "recommendations": self.block_flow_optimization.recommendations
            },
            "autonomous_decision": {
                "recommended_strategy": self.autonomous_decision.recommended_strategy,
                "auto_win_probability": self.autonomous_decision.auto_win_probability,
                "expected_points": self.autonomous_decision.expected_points,
                "risk_assessment": self.autonomous_decision.risk_assessment
            },
            "goal_priority_analysis": {
                "recommended_priority": self.goal_priority_analysis.recommended_priority,
                "center_goal_value": self.goal_priority_analysis.center_goal_value,
                "long_goal_value": self.goal_priority_analysis.long_goal_value,
                "decision_confidence": self.goal_priority_analysis.decision_confidence
            },
            "parking_decision_analysis": {
                "recommended_timing": self.parking_decision_analysis.recommended_timing,
                "one_robot_threshold": self.parking_decision_analysis.one_robot_threshold,
                "two_robot_threshold": self.parking_decision_analysis.two_robot_threshold,
                "expected_value": self.parking_decision_analysis.expected_value
            },
            "offense_defense_balance": {
                "recommended_ratio": self.offense_defense_balance.recommended_ratio,
                "offensive_roi": self.offense_defense_balance.offensive_roi,
                "defensive_roi": self.offense_defense_balance.defensive_roi,
                "critical_zones": self.offense_defense_balance.critical_zones
            },
            "recommended_archetype": self.recommended_archetype,
            "overall_score": self.overall_score,
            "recommendations": self.recommendations,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class MonteCarloSimulationResult:
    """Monte Carlo simulation result"""
    strategy_id: str
    num_simulations: int
    win_rate: float
    avg_score: float
    score_std: float
    scoring_breakdown: Dict[str, float]
    opponent_matchups: Dict[str, float]  # Win rates against different opponent types
    performance_confidence: float  # Statistical confidence in results
    risk_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.risk_metrics is None:
            self.risk_metrics = {
                "score_variance": self.score_std ** 2,
                "worst_case_score": max(0, self.avg_score - 2 * self.score_std),
                "best_case_score": self.avg_score + 2 * self.score_std
            }

# Push Back specific response models extending BaseModel
class PushBackApiResponse(BaseModel):
    """Push Back API response wrapper"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    message: str = ""
    push_back_version: str = "1.0.0"

# Strategy Archetypes Constants
PUSH_BACK_ARCHETYPES = {
    "block_flow_maximizer": {
        "name": "Block Flow Maximizer",
        "description": "Optimizes block collection and transportation efficiency",
        "focus": "block_collection",
        "risk_level": "low",
        "complexity": "medium"
    },
    "control_zone_controller": {
        "name": "Control Zone Controller",
        "description": "Prioritizes control zone domination for consistent points",
        "focus": "control_zones",
        "risk_level": "medium",
        "complexity": "high"
    },
    "goal_rush_specialist": {
        "name": "Goal Rush Specialist",
        "description": "Focuses on rapid goal scoring for maximum points",
        "focus": "goal_scoring",
        "risk_level": "high",
        "complexity": "medium"
    },
    "parking_strategist": {
        "name": "Parking Strategist",
        "description": "Optimizes endgame parking timing and execution",
        "focus": "parking",
        "risk_level": "low",
        "complexity": "low"
    },
    "autonomous_specialist": {
        "name": "Autonomous Specialist", 
        "description": "Maximizes autonomous period effectiveness",
        "focus": "autonomous",
        "risk_level": "medium",
        "complexity": "high"
    },
    "balanced_competitor": {
        "name": "Balanced Competitor",
        "description": "Well-rounded strategy adaptable to various opponents",
        "focus": "balanced",
        "risk_level": "medium",
        "complexity": "medium"
    },
    "defensive_disruptor": {
        "name": "Defensive Disruptor",
        "description": "Focuses on opponent disruption and defensive play",
        "focus": "defense",
        "risk_level": "high",
        "complexity": "high"
    }
}