"""
API Models for VEX U Scoring Analysis Platform

This module contains Pydantic models that define the data structures
used by the API, matching the outputs from the Python analysis tools.
"""

from .base import BaseResponse, ErrorResponse, SuccessResponse
from .analysis import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisType,
    AnalysisResult,
    StatisticalMetrics,
    PerformanceMetrics
)
from .strategy import (
    Robot,
    AllianceStrategy,
    StrategyRequest,
    StrategyResponse,
    StrategyOptimizationRequest
)
from .scenario import (
    ScenarioRequest,
    ScenarioResponse,
    Match,
    MatchResult,
    SimulationParameters
)
from .visualization import (
    VisualizationRequest,
    VisualizationResponse,
    ChartData,
    ChartType,
    DashboardData
)
from .report import (
    ReportRequest,
    ReportResponse,
    ReportType,
    ReportSection,
    ReportMetadata
)
from .ml import (
    MLModelRequest,
    MLModelResponse,
    MLModelStatus,
    PredictionRequest,
    OptimizationRequest
)

__all__ = [
    # Base models
    'BaseResponse',
    'ErrorResponse', 
    'SuccessResponse',
    
    # Analysis models
    'AnalysisRequest',
    'AnalysisResponse',
    'AnalysisType',
    'AnalysisResult',
    'StatisticalMetrics',
    'PerformanceMetrics',
    
    # Strategy models
    'Robot',
    'AllianceStrategy',
    'StrategyRequest',
    'StrategyResponse',
    'StrategyOptimizationRequest',
    
    # Scenario models
    'ScenarioRequest',
    'ScenarioResponse',
    'Match',
    'MatchResult',
    'SimulationParameters',
    
    # Visualization models
    'VisualizationRequest',
    'VisualizationResponse',
    'ChartData',
    'ChartType',
    'DashboardData',
    
    # Report models
    'ReportRequest',
    'ReportResponse',
    'ReportType',
    'ReportSection',
    'ReportMetadata',
    
    # ML models
    'MLModelRequest',
    'MLModelResponse',
    'MLModelStatus',
    'PredictionRequest',
    'OptimizationRequest',
]