"""
Scenario and simulation models for match generation and analysis
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum

class MatchType(str, Enum):
    """Types of matches in VEX U competition"""
    QUALIFICATION = "qualification"
    SEMIFINAL = "semifinal"
    FINAL = "final"
    PRACTICE = "practice"
    CUSTOM = "custom"

class MatchOutcome(str, Enum):
    """Possible match outcomes"""
    RED_WIN = "red_win"
    BLUE_WIN = "blue_win"
    TIE = "tie"
    NO_RESULT = "no_result"

class ScenarioComplexity(str, Enum):
    """Scenario generation complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXTREME = "extreme"

class SimulationParameters(BaseModel):
    """Parameters for match simulation"""
    
    # Basic simulation settings
    match_count: int = Field(100, ge=1, le=10000, description="Number of matches to simulate")
    simulation_seed: Optional[int] = Field(None, description="Random seed for reproducible results")
    
    # Match configuration
    autonomous_duration: int = Field(30, ge=15, le=60, description="Autonomous period duration")
    driver_duration: int = Field(90, ge=60, le=180, description="Driver control duration")
    
    # Variability settings
    performance_variance: float = Field(0.1, ge=0, le=0.5, description="Performance variability factor")
    random_events: bool = Field(True, description="Include random events in simulation")
    
    # Scenario parameters
    scenario_complexity: ScenarioComplexity = Field(ScenarioComplexity.MODERATE, description="Scenario complexity")
    include_endgame: bool = Field(True, description="Include endgame scenarios")
    include_penalties: bool = Field(True, description="Include penalty scenarios")
    
    # Environmental factors
    field_conditions: Optional[str] = Field("standard", description="Field conditions")
    competition_pressure: float = Field(0.5, ge=0, le=1, description="Competition pressure factor")

class MatchScore(BaseModel):
    """Score breakdown for a match alliance"""
    autonomous_score: int = Field(ge=0, description="Autonomous period score")
    driver_score: int = Field(ge=0, description="Driver control score")
    endgame_score: int = Field(ge=0, description="Endgame score")
    penalty_score: int = Field(ge=0, description="Penalty points")
    total_score: int = Field(ge=0, description="Total alliance score")
    
    # Detailed scoring
    scoring_breakdown: Optional[Dict[str, int]] = Field(None, description="Detailed scoring breakdown")

class MatchResult(BaseModel):
    """Complete result of a simulated match"""
    match_id: str = Field(description="Unique match identifier")
    match_number: int = Field(ge=1, description="Match number in sequence")
    match_type: MatchType = Field(description="Type of match")
    
    # Alliance information
    red_alliance: str = Field(description="Red alliance strategy ID")
    blue_alliance: str = Field(description="Blue alliance strategy ID")
    
    # Scores
    red_score: MatchScore = Field(description="Red alliance score breakdown")
    blue_score: MatchScore = Field(description="Blue alliance score breakdown")
    
    # Outcome
    outcome: MatchOutcome = Field(description="Match outcome")
    margin_of_victory: int = Field(description="Score difference")
    
    # Timing and events
    match_duration: float = Field(description="Actual match duration in seconds")
    key_events: List[str] = Field(default_factory=list, description="Key events during match")
    penalties: List[str] = Field(default_factory=list, description="Penalties assessed")
    
    # Performance metrics
    red_performance: Dict[str, float] = Field(description="Red alliance performance metrics")
    blue_performance: Dict[str, float] = Field(description="Blue alliance performance metrics")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Match timestamp")
    simulation_data: Optional[Dict[str, Any]] = Field(None, description="Raw simulation data")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class Match(BaseModel):
    """Match configuration before simulation"""
    match_id: str = Field(description="Unique match identifier")
    red_alliance_strategy: str = Field(description="Red alliance strategy ID")
    blue_alliance_strategy: str = Field(description="Blue alliance strategy ID")
    match_type: MatchType = Field(MatchType.PRACTICE, description="Type of match")
    
    # Optional pre-match data
    field_setup: Optional[Dict[str, Any]] = Field(None, description="Field setup configuration")
    special_conditions: Optional[List[str]] = Field(None, description="Special match conditions")

class ScenarioRequest(BaseModel):
    """Request model for scenario generation"""
    
    # Scenario parameters
    scenario_count: int = Field(10, ge=1, le=1000, description="Number of scenarios to generate")
    complexity_level: ScenarioComplexity = Field(ScenarioComplexity.MODERATE, description="Scenario complexity")
    
    # Strategy parameters
    strategy_pool: Optional[List[str]] = Field(None, description="Pool of strategy IDs to use")
    auto_generate_strategies: bool = Field(True, description="Auto-generate strategies if pool is small")
    
    # Simulation parameters
    simulation_params: SimulationParameters = Field(default_factory=SimulationParameters, description="Simulation parameters")
    
    # Focus areas
    focus_areas: Optional[List[str]] = Field(None, description="Areas to focus scenario generation")
    include_ml_predictions: bool = Field(True, description="Include ML-generated scenarios")
    
    # Evolution parameters
    enable_evolution: bool = Field(False, description="Enable genetic algorithm evolution")
    evolution_generations: Optional[int] = Field(None, ge=1, le=100, description="Number of evolution generations")
    
    # Custom parameters
    custom_conditions: Optional[Dict[str, Any]] = Field(None, description="Custom scenario conditions")

class ScenarioSet(BaseModel):
    """A set of related scenarios"""
    scenario_set_id: str = Field(description="Unique scenario set identifier")
    name: str = Field(description="Scenario set name")
    description: str = Field(description="Scenario set description")
    
    # Scenarios and matches
    scenarios: List[Match] = Field(description="Scenarios in this set")
    total_matches: int = Field(description="Total number of matches")
    
    # Generation parameters
    generation_params: ScenarioRequest = Field(description="Parameters used to generate scenarios")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    # Analysis ready
    ready_for_simulation: bool = Field(True, description="Whether scenarios are ready for simulation")
    estimated_duration: Optional[float] = Field(None, description="Estimated simulation duration")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class SimulationResults(BaseModel):
    """Results from running a set of scenario simulations"""
    scenario_set_id: str = Field(description="Scenario set identifier")
    simulation_id: str = Field(description="Unique simulation run identifier")
    
    # Results
    matches: List[MatchResult] = Field(description="All match results")
    total_matches: int = Field(description="Total number of matches simulated")
    successful_matches: int = Field(description="Number of successfully simulated matches")
    
    # Aggregate statistics
    red_wins: int = Field(description="Number of red alliance wins")
    blue_wins: int = Field(description="Number of blue alliance wins")
    ties: int = Field(description="Number of tied matches")
    
    # Performance metrics
    average_score: float = Field(description="Average total score across all matches")
    score_distribution: Dict[str, float] = Field(description="Score distribution statistics")
    
    # Strategy performance
    strategy_performance: Dict[str, Dict[str, float]] = Field(description="Performance by strategy")
    
    # Timing and metadata
    simulation_duration: float = Field(description="Total simulation time in seconds")
    started_at: datetime = Field(description="Simulation start time")
    completed_at: datetime = Field(description="Simulation completion time")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ScenarioResponse(BaseModel):
    """Response model for scenario operations"""
    scenario_set: ScenarioSet = Field(description="Generated scenario set")
    simulation_results: Optional[SimulationResults] = Field(None, description="Simulation results if executed")
    task_id: Optional[str] = Field(None, description="Background task ID for long-running operations")
    
class ScenarioListItem(BaseModel):
    """Scenario set item for listing responses"""
    scenario_set_id: str = Field(description="Scenario set identifier")
    name: str = Field(description="Scenario set name")
    total_matches: int = Field(description="Total number of matches")
    complexity_level: ScenarioComplexity = Field(description="Complexity level")
    created_at: datetime = Field(description="Creation timestamp")
    ready_for_simulation: bool = Field(description="Ready for simulation")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }