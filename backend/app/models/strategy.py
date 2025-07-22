"""
Strategy models for alliance strategies and robot configurations
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum

class RobotRole(str, Enum):
    """Robot roles in VEX U competition"""
    OFFENSE = "offense"
    DEFENSE = "defense"
    SUPPORT = "support"
    VERSATILE = "versatile"

class RobotSize(str, Enum):
    """Robot size categories"""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

class StrategyType(str, Enum):
    """Types of alliance strategies"""
    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"
    BALANCED = "balanced"
    OPPORTUNISTIC = "opportunistic"
    CUSTOM = "custom"

class Robot(BaseModel):
    """Individual robot configuration"""
    robot_id: str = Field(description="Unique robot identifier")
    name: str = Field(description="Robot name")
    role: RobotRole = Field(description="Primary role of the robot")
    size: RobotSize = Field(description="Robot size category")
    
    # Performance characteristics
    speed: float = Field(ge=0, le=10, description="Speed rating (0-10)")
    maneuverability: float = Field(ge=0, le=10, description="Maneuverability rating (0-10)")
    scoring_ability: float = Field(ge=0, le=10, description="Scoring ability rating (0-10)")
    defensive_ability: float = Field(ge=0, le=10, description="Defensive ability rating (0-10)")
    reliability: float = Field(ge=0, le=10, description="Reliability rating (0-10)")
    
    # Efficiency metrics
    autonomous_efficiency: float = Field(ge=0, le=1, description="Autonomous period efficiency")
    driver_efficiency: float = Field(ge=0, le=1, description="Driver period efficiency")
    
    # Scoring metrics
    autonomous_score: float = Field(ge=0, description="Average autonomous score")
    driver_score: float = Field(ge=0, description="Average driver control score")
    total_score: float = Field(ge=0, description="Average total score")
    
    # Additional attributes
    attributes: Optional[Dict[str, Any]] = Field(None, description="Additional robot attributes")

class AllianceStrategy(BaseModel):
    """Alliance strategy configuration"""
    strategy_id: str = Field(description="Unique strategy identifier")
    name: str = Field(description="Strategy name")
    strategy_type: StrategyType = Field(description="Type of strategy")
    
    # Robot configuration
    robots: List[Robot] = Field(description="Robots in the alliance", min_items=1, max_items=2)
    
    # Strategy details
    autonomous_strategy: str = Field(description="Autonomous period strategy description")
    driver_strategy: str = Field(description="Driver control strategy description")
    endgame_strategy: str = Field(description="Endgame strategy description")
    
    # Performance predictions
    expected_score: float = Field(ge=0, description="Expected total score")
    win_probability: float = Field(ge=0, le=1, description="Predicted win probability")
    risk_level: str = Field(description="Risk level (low/medium/high)")
    
    # Timing and coordination
    autonomous_duration: int = Field(30, description="Autonomous period duration in seconds")
    driver_duration: int = Field(90, description="Driver control duration in seconds")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Strategy creation time")
    created_by: Optional[str] = Field(None, description="Strategy creator")
    version: int = Field(1, description="Strategy version number")
    
    # Analysis data
    analysis_data: Optional[Dict[str, Any]] = Field(None, description="Supporting analysis data")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class StrategyRequest(BaseModel):
    """Request model for strategy operations"""
    
    # Strategy generation parameters
    strategy_type: Optional[StrategyType] = Field(None, description="Desired strategy type")
    robot_count: int = Field(2, ge=1, le=2, description="Number of robots in alliance")
    
    # Performance requirements
    min_expected_score: Optional[float] = Field(None, ge=0, description="Minimum expected score")
    max_risk_level: Optional[str] = Field(None, description="Maximum acceptable risk level")
    
    # Optimization targets
    optimize_for: Optional[str] = Field("score", description="Optimization target (score/reliability/speed)")
    focus_period: Optional[str] = Field("overall", description="Focus period (autonomous/driver/endgame/overall)")
    
    # Constraints
    robot_constraints: Optional[Dict[str, Any]] = Field(None, description="Robot selection constraints")
    strategy_constraints: Optional[Dict[str, Any]] = Field(None, description="Strategy constraints")
    
    # Custom parameters
    custom_params: Optional[Dict[str, Any]] = Field(None, description="Custom generation parameters")

class StrategyOptimizationRequest(BaseModel):
    """Request model for strategy optimization"""
    strategy_id: str = Field(description="Strategy to optimize")
    
    # Optimization parameters
    optimization_target: str = Field("score", description="What to optimize for")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Optimization constraints")
    max_iterations: int = Field(100, ge=1, le=1000, description="Maximum optimization iterations")
    
    # ML parameters
    use_ml_predictions: bool = Field(True, description="Whether to use ML predictions")
    ml_model_types: Optional[List[str]] = Field(None, description="Specific ML models to use")

class StrategyComparison(BaseModel):
    """Comparison between strategies"""
    strategy_a: str = Field(description="First strategy ID")
    strategy_b: str = Field(description="Second strategy ID")
    
    # Comparison metrics
    score_difference: float = Field(description="Score difference (A - B)")
    win_probability_difference: float = Field(description="Win probability difference")
    risk_difference: str = Field(description="Risk level difference")
    
    # Detailed comparisons
    performance_comparison: Dict[str, float] = Field(description="Performance metric comparisons")
    strengths_a: List[str] = Field(description="Strengths of strategy A")
    strengths_b: List[str] = Field(description="Strengths of strategy B")
    
    # Recommendations
    recommendation: str = Field(description="Which strategy is recommended")
    reasoning: str = Field(description="Reasoning for recommendation")

class StrategyResponse(BaseModel):
    """Response model for strategy operations"""
    strategy: AllianceStrategy = Field(description="Strategy data")
    optimization_info: Optional[Dict[str, Any]] = Field(None, description="Optimization information")
    comparison: Optional[StrategyComparison] = Field(None, description="Strategy comparison data")
    
class StrategyListItem(BaseModel):
    """Strategy item for listing responses"""
    strategy_id: str = Field(description="Strategy identifier")
    name: str = Field(description="Strategy name")
    strategy_type: StrategyType = Field(description="Strategy type")
    expected_score: float = Field(description="Expected score")
    win_probability: float = Field(description="Win probability")
    risk_level: str = Field(description="Risk level")
    created_at: datetime = Field(description="Creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }