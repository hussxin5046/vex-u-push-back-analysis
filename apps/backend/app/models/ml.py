"""
Machine Learning models for ML operations and predictions
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum

class MLModelType(str, Enum):
    """Types of ML models available"""
    COORDINATION = "coordination"
    SCORING_OPTIMIZER = "scoring_optimizer"
    STRATEGY_PREDICTOR = "strategy_predictor"
    PATTERN_DISCOVERY = "pattern_discovery"

class ModelStatus(str, Enum):
    """ML model status"""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    TRAINING = "training"
    LOADING = "loading"
    ERROR = "error"

class TrainingStatus(str, Enum):
    """Training job status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class PredictionType(str, Enum):
    """Types of predictions"""
    SCORE_PREDICTION = "score_prediction"
    STRATEGY_OPTIMIZATION = "strategy_optimization"
    PATTERN_MATCHING = "pattern_matching"
    COORDINATION_ANALYSIS = "coordination_analysis"

class MLModelStatus(BaseModel):
    """Status of all ML models"""
    coordination: ModelStatus = Field(description="Coordination model status")
    scoring_optimizer: ModelStatus = Field(description="Scoring optimizer model status")
    strategy_predictor: ModelStatus = Field(description="Strategy predictor model status")
    pattern_discovery: ModelStatus = Field(description="Pattern discovery model status")
    
    # Model information
    model_versions: Dict[str, str] = Field(description="Version of each model")
    last_updated: Dict[str, datetime] = Field(description="Last update time for each model")
    
    # Performance metrics
    model_accuracy: Optional[Dict[str, float]] = Field(None, description="Model accuracy metrics")
    training_data_size: Optional[Dict[str, int]] = Field(None, description="Training data size for each model")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class MLModelRequest(BaseModel):
    """Base request for ML model operations"""
    model_type: MLModelType = Field(description="Type of ML model")
    
class TrainingRequest(MLModelRequest):
    """Request for training ML models"""
    
    # Training data
    training_data: Optional[Dict[str, Any]] = Field(None, description="Training data (if not using default)")
    use_existing_data: bool = Field(True, description="Use existing training data")
    
    # Training parameters
    epochs: Optional[int] = Field(None, ge=1, le=1000, description="Number of training epochs")
    batch_size: Optional[int] = Field(None, ge=1, le=1000, description="Training batch size")
    learning_rate: Optional[float] = Field(None, gt=0, le=1, description="Learning rate")
    
    # Model parameters
    model_params: Optional[Dict[str, Any]] = Field(None, description="Model-specific parameters")
    validation_split: float = Field(0.2, ge=0.1, le=0.5, description="Validation data split")
    
    # Advanced options
    early_stopping: bool = Field(True, description="Use early stopping")
    save_checkpoints: bool = Field(True, description="Save training checkpoints")
    
class PredictionRequest(MLModelRequest):
    """Request for ML predictions"""
    
    # Input data
    input_data: Dict[str, Any] = Field(description="Input data for prediction")
    prediction_type: PredictionType = Field(description="Type of prediction requested")
    
    # Prediction parameters
    confidence_threshold: Optional[float] = Field(None, ge=0, le=1, description="Confidence threshold")
    return_probabilities: bool = Field(False, description="Return prediction probabilities")
    explain_prediction: bool = Field(False, description="Include prediction explanation")
    
    # Context
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for prediction")

class OptimizationRequest(MLModelRequest):
    """Request for strategy/scoring optimization"""
    
    # Optimization target
    optimization_target: str = Field(description="What to optimize (score, efficiency, etc.)")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Optimization constraints")
    
    # Input strategy/scenario
    strategy_data: Optional[Dict[str, Any]] = Field(None, description="Strategy to optimize")
    scenario_data: Optional[Dict[str, Any]] = Field(None, description="Scenario context")
    
    # Optimization parameters
    max_iterations: int = Field(100, ge=1, le=1000, description="Maximum optimization iterations")
    convergence_threshold: float = Field(0.001, gt=0, description="Convergence threshold")
    
    # Advanced options
    use_genetic_algorithm: bool = Field(False, description="Use genetic algorithm optimization")
    population_size: Optional[int] = Field(None, ge=10, le=1000, description="GA population size")

class PatternDiscoveryRequest(MLModelRequest):
    """Request for pattern discovery"""
    
    # Data for analysis
    analysis_data: Dict[str, Any] = Field(description="Data to analyze for patterns")
    discovery_type: str = Field("temporal", description="Type of pattern discovery")
    
    # Discovery parameters
    min_pattern_length: int = Field(3, ge=2, le=20, description="Minimum pattern length")
    max_patterns: int = Field(50, ge=1, le=1000, description="Maximum patterns to discover")
    confidence_threshold: float = Field(0.7, ge=0.1, le=1.0, description="Pattern confidence threshold")
    
    # Advanced options
    include_seasonal: bool = Field(True, description="Include seasonal patterns")
    include_trends: bool = Field(True, description="Include trend patterns")

class ModelMetrics(BaseModel):
    """Model performance metrics"""
    accuracy: Optional[float] = Field(None, description="Model accuracy")
    precision: Optional[float] = Field(None, description="Model precision")
    recall: Optional[float] = Field(None, description="Model recall")
    f1_score: Optional[float] = Field(None, description="F1 score")
    
    # Training metrics
    training_loss: Optional[float] = Field(None, description="Final training loss")
    validation_loss: Optional[float] = Field(None, description="Final validation loss")
    training_time: Optional[float] = Field(None, description="Training time in seconds")
    
    # Additional metrics
    custom_metrics: Optional[Dict[str, float]] = Field(None, description="Model-specific metrics")

class TrainingJob(BaseModel):
    """Training job information"""
    job_id: str = Field(description="Training job identifier")
    model_type: MLModelType = Field(description="Model being trained")
    status: TrainingStatus = Field(description="Training status")
    
    # Progress
    progress: float = Field(ge=0, le=100, description="Training progress percentage")
    current_epoch: Optional[int] = Field(None, description="Current training epoch")
    total_epochs: Optional[int] = Field(None, description="Total training epochs")
    
    # Metrics
    current_metrics: Optional[ModelMetrics] = Field(None, description="Current training metrics")
    best_metrics: Optional[ModelMetrics] = Field(None, description="Best metrics achieved")
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Training start time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    completed_at: Optional[datetime] = Field(None, description="Training completion time")
    
    # Configuration
    training_params: TrainingRequest = Field(description="Training parameters used")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class PredictionResult(BaseModel):
    """Result from ML prediction"""
    prediction_id: str = Field(description="Prediction identifier")
    model_type: MLModelType = Field(description="Model used for prediction")
    prediction_type: PredictionType = Field(description="Type of prediction")
    
    # Results
    prediction: Any = Field(description="Main prediction result")
    confidence: Optional[float] = Field(None, description="Prediction confidence")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities")
    
    # Explanation
    explanation: Optional[Dict[str, Any]] = Field(None, description="Prediction explanation")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")
    
    # Context
    input_data: Dict[str, Any] = Field(description="Input data used")
    model_version: str = Field(description="Model version used")
    
    # Timing
    predicted_at: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")
    processing_time: float = Field(description="Processing time in seconds")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class OptimizationResult(BaseModel):
    """Result from optimization operation"""
    optimization_id: str = Field(description="Optimization identifier")
    model_type: MLModelType = Field(description="Model used for optimization")
    
    # Results
    optimized_strategy: Dict[str, Any] = Field(description="Optimized strategy/configuration")
    improvement_metrics: Dict[str, float] = Field(description="Improvement metrics")
    
    # Optimization details
    iterations: int = Field(description="Number of iterations performed")
    convergence_achieved: bool = Field(description="Whether optimization converged")
    final_score: float = Field(description="Final optimization score")
    
    # Context
    original_strategy: Dict[str, Any] = Field(description="Original strategy/configuration")
    optimization_params: OptimizationRequest = Field(description="Optimization parameters")
    
    # Timing
    optimized_at: datetime = Field(default_factory=datetime.utcnow, description="Optimization timestamp")
    processing_time: float = Field(description="Processing time in seconds")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class PatternDiscoveryResult(BaseModel):
    """Result from pattern discovery"""
    discovery_id: str = Field(description="Discovery identifier")
    
    # Discovered patterns
    patterns: List[Dict[str, Any]] = Field(description="Discovered patterns")
    pattern_count: int = Field(description="Number of patterns found")
    
    # Pattern analysis
    most_significant: List[Dict[str, Any]] = Field(description="Most significant patterns")
    temporal_patterns: Optional[List[Dict[str, Any]]] = Field(None, description="Temporal patterns")
    seasonal_patterns: Optional[List[Dict[str, Any]]] = Field(None, description="Seasonal patterns")
    
    # Discovery details
    discovery_params: PatternDiscoveryRequest = Field(description="Discovery parameters")
    confidence_scores: Dict[str, float] = Field(description="Pattern confidence scores")
    
    # Timing
    discovered_at: datetime = Field(default_factory=datetime.utcnow, description="Discovery timestamp")
    processing_time: float = Field(description="Processing time in seconds")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class MLModelResponse(BaseModel):
    """Response model for ML operations"""
    training_job: Optional[TrainingJob] = Field(None, description="Training job info")
    prediction_result: Optional[PredictionResult] = Field(None, description="Prediction result")
    optimization_result: Optional[OptimizationResult] = Field(None, description="Optimization result")
    pattern_discovery_result: Optional[PatternDiscoveryResult] = Field(None, description="Pattern discovery result")
    task_id: Optional[str] = Field(None, description="Background task ID if async")