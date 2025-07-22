"""
Analysis models for VEX U scoring analysis operations
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum

class AnalysisType(str, Enum):
    """Types of analysis that can be performed"""
    DEMO = "demo"
    FULL = "full"
    STATISTICAL = "statistical"
    SCORING = "scoring"
    STRATEGY = "strategy"
    ML_PREDICTION = "ml_prediction"

class ComplexityLevel(str, Enum):
    """Complexity levels for analysis"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

class FocusArea(str, Enum):
    """Focus areas for analysis"""
    AUTONOMOUS = "autonomous"
    DRIVER = "driver"
    OVERALL = "overall"
    DEFENSE = "defense"
    OFFENSE = "offense"

class StatisticalMethod(str, Enum):
    """Statistical analysis methods"""
    DESCRIPTIVE = "descriptive"
    INFERENTIAL = "inferential"
    PREDICTIVE = "predictive"
    CORRELATION = "correlation"
    REGRESSION = "regression"

class AnalysisRequest(BaseModel):
    """Request model for analysis operations"""
    analysis_type: AnalysisType = Field(description="Type of analysis to perform")
    
    # General parameters
    strategy_count: Optional[int] = Field(10, ge=1, le=1000, description="Number of strategies to analyze")
    simulation_count: Optional[int] = Field(100, ge=10, le=10000, description="Number of simulations to run")
    complexity: Optional[ComplexityLevel] = Field(ComplexityLevel.INTERMEDIATE, description="Analysis complexity level")
    
    # Scoring analysis parameters
    focus_area: Optional[FocusArea] = Field(None, description="Focus area for scoring analysis")
    time_period: Optional[int] = Field(None, ge=1, le=365, description="Time period in days")
    
    # Statistical analysis parameters
    statistical_method: Optional[StatisticalMethod] = Field(None, description="Statistical method to use")
    sample_size: Optional[int] = Field(None, ge=10, le=10000, description="Sample size for statistical analysis")
    confidence_level: Optional[float] = Field(0.95, ge=0.01, le=0.99, description="Confidence level for statistical tests")
    
    # Strategy analysis parameters
    strategy_name: Optional[str] = Field(None, description="Name for custom strategy")
    include_ml: Optional[bool] = Field(True, description="Whether to include ML predictions")
    
    # Custom parameters
    custom_params: Optional[Dict[str, Any]] = Field(None, description="Custom analysis parameters")

class StatisticalMetrics(BaseModel):
    """Statistical metrics for analysis results"""
    mean: float = Field(description="Mean value")
    median: float = Field(description="Median value")
    std: float = Field(description="Standard deviation")
    variance: float = Field(description="Variance")
    min: float = Field(description="Minimum value")
    max: float = Field(description="Maximum value")
    count: int = Field(description="Number of observations")
    
    # Percentiles
    p25: float = Field(description="25th percentile")
    p50: float = Field(description="50th percentile (median)")
    p75: float = Field(description="75th percentile")
    p90: float = Field(description="90th percentile")
    p95: float = Field(description="95th percentile")
    p99: float = Field(description="99th percentile")
    
    # Additional metrics
    skewness: Optional[float] = Field(None, description="Skewness measure")
    kurtosis: Optional[float] = Field(None, description="Kurtosis measure")

class PerformanceMetrics(BaseModel):
    """Performance metrics for strategies and teams"""
    win_rate: float = Field(ge=0, le=1, description="Win rate (0-1)")
    average_score: float = Field(ge=0, description="Average score")
    score_consistency: StatisticalMetrics = Field(description="Score consistency metrics")
    
    # Efficiency metrics
    autonomous_efficiency: float = Field(ge=0, le=1, description="Autonomous period efficiency")
    driver_efficiency: float = Field(ge=0, le=1, description="Driver period efficiency")
    overall_efficiency: float = Field(ge=0, le=1, description="Overall efficiency")
    
    # Rating metrics
    offensive_rating: float = Field(ge=0, le=100, description="Offensive rating (0-100)")
    defensive_rating: float = Field(ge=0, le=100, description="Defensive rating (0-100)")
    strategic_rating: float = Field(ge=0, le=100, description="Strategic rating (0-100)")
    
    # Comparison metrics
    rank_percentile: Optional[float] = Field(None, ge=0, le=100, description="Rank percentile")
    improvement_rate: Optional[float] = Field(None, description="Improvement rate over time")

class AnalysisInsight(BaseModel):
    """Individual insight from analysis"""
    title: str = Field(description="Insight title")
    description: str = Field(description="Detailed description")
    importance: str = Field(description="Importance level (high/medium/low)")
    category: str = Field(description="Insight category")
    confidence: float = Field(ge=0, le=1, description="Confidence level (0-1)")
    data_points: Optional[Dict[str, Any]] = Field(None, description="Supporting data")

class AnalysisResult(BaseModel):
    """Complete analysis result"""
    analysis_id: str = Field(description="Unique analysis identifier")
    analysis_type: AnalysisType = Field(description="Type of analysis performed")
    title: str = Field(description="Analysis title")
    summary: str = Field(description="Executive summary")
    
    # Metrics and data
    performance_metrics: Optional[PerformanceMetrics] = Field(None, description="Performance metrics")
    statistical_data: Optional[Dict[str, StatisticalMetrics]] = Field(None, description="Statistical data")
    raw_data: Optional[Dict[str, Any]] = Field(None, description="Raw analysis data")
    
    # Insights and recommendations
    insights: List[AnalysisInsight] = Field(default_factory=list, description="Key insights")
    recommendations: List[str] = Field(default_factory=list, description="Strategic recommendations")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Analysis creation time")
    parameters: AnalysisRequest = Field(description="Analysis parameters used")
    duration: Optional[float] = Field(None, description="Analysis duration in seconds")
    
    # Visualization data
    charts: Optional[List[Dict[str, Any]]] = Field(None, description="Chart data for visualization")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class AnalysisResponse(BaseModel):
    """Response model for analysis operations"""
    result: AnalysisResult = Field(description="Analysis result")
    task_id: Optional[str] = Field(None, description="Background task ID if async")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class AnalysisHistoryItem(BaseModel):
    """Historical analysis item for listing"""
    analysis_id: str = Field(description="Analysis identifier")
    analysis_type: AnalysisType = Field(description="Type of analysis")
    title: str = Field(description="Analysis title")
    summary: str = Field(description="Brief summary")
    created_at: datetime = Field(description="Creation timestamp")
    duration: Optional[float] = Field(None, description="Duration in seconds")
    status: str = Field(description="Analysis status")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }