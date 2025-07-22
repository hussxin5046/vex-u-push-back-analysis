"""
ML Models API routes for machine learning operations
"""

from flask import Blueprint, request, jsonify, current_app
from app.models.ml import (
    MLModelRequest, MLModelResponse, MLModelStatus, MLModelType,
    TrainingRequest, PredictionRequest, OptimizationRequest, PatternDiscoveryRequest,
    TrainingJob, PredictionResult, OptimizationResult, PatternDiscoveryResult,
    ModelStatus, TrainingStatus, PredictionType
)
from app.models.base import SuccessResponse, ErrorResponse, TaskInfo, StatusEnum
from app.services.vex_analysis_service import VEXAnalysisService
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)

ml_bp = Blueprint('ml', __name__)

def get_vex_service() -> VEXAnalysisService:
    """Get VEX analysis service instance"""
    return VEXAnalysisService(
        vex_path=current_app.config['VEX_ANALYSIS_PATH'],
        python_path=current_app.config['PYTHON_PATH']
    )

@ml_bp.route('/status', methods=['GET'])
def get_ml_model_status():
    """
    Get status of all ML models
    
    GET /api/ml/status
    """
    try:
        # Mock ML model status
        # In a real implementation, this would check actual model availability
        model_status = MLModelStatus(
            coordination=ModelStatus.AVAILABLE,
            scoring_optimizer=ModelStatus.AVAILABLE,
            strategy_predictor=ModelStatus.AVAILABLE,
            pattern_discovery=ModelStatus.AVAILABLE,
            model_versions={
                "coordination": "1.2.0",
                "scoring_optimizer": "2.1.0", 
                "strategy_predictor": "1.5.0",
                "pattern_discovery": "1.0.0"
            },
            last_updated={
                "coordination": datetime.utcnow(),
                "scoring_optimizer": datetime.utcnow(),
                "strategy_predictor": datetime.utcnow(),
                "pattern_discovery": datetime.utcnow()
            },
            model_accuracy={
                "coordination": 0.87,
                "scoring_optimizer": 0.92,
                "strategy_predictor": 0.84,
                "pattern_discovery": 0.79
            },
            training_data_size={
                "coordination": 15000,
                "scoring_optimizer": 25000,
                "strategy_predictor": 18000,
                "pattern_discovery": 12000
            }
        )
        
        response = SuccessResponse(
            message="ML model status retrieved successfully",
            data=model_status.dict()
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error(f"ML model status check failed: {str(e)}")
        error_response = ErrorResponse(
            message="Failed to retrieve ML model status",
            error_code="STATUS_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500

@ml_bp.route('/train', methods=['POST'])
def train_ml_model():
    """
    Train ML model
    
    POST /api/ml/train
    
    Body:
    {
        "model_type": "scoring_optimizer",
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "validation_split": 0.2,
        "early_stopping": true
    }
    """
    try:
        # Parse request data
        data = request.get_json() or {}
        
        # Validate request
        training_request = TrainingRequest(
            model_type=MLModelType(data.get('model_type')),
            epochs=data.get('epochs'),
            batch_size=data.get('batch_size'),
            learning_rate=data.get('learning_rate'),
            validation_split=data.get('validation_split', 0.2),
            early_stopping=data.get('early_stopping', True)
        )
        
        # Start training using VEX service
        vex_service = get_vex_service()
        training_params = {
            "epochs": training_request.epochs,
            "batch_size": training_request.batch_size,
            "learning_rate": training_request.learning_rate,
            "validation_split": training_request.validation_split,
            "early_stopping": training_request.early_stopping
        }
        
        training_result = vex_service.train_ml_model(
            model_type=training_request.model_type.value,
            training_params=training_params
        )
        
        # Create training job
        from app.models.ml import ModelMetrics
        
        training_job = TrainingJob(
            job_id=str(uuid.uuid4()),
            model_type=training_request.model_type,
            status=TrainingStatus.COMPLETED,  # Mock completed training
            progress=100.0,
            current_epoch=training_request.epochs or 100,
            total_epochs=training_request.epochs or 100,
            current_metrics=ModelMetrics(
                accuracy=0.92,
                precision=0.89,
                recall=0.94,
                f1_score=0.91,
                training_loss=0.15,
                validation_loss=0.18,
                training_time=240.5
            ),
            best_metrics=ModelMetrics(
                accuracy=0.92,
                precision=0.89,
                recall=0.94,
                f1_score=0.91,
                training_loss=0.15,
                validation_loss=0.18,
                training_time=240.5
            ),
            completed_at=datetime.utcnow(),
            training_params=training_request
        )
        
        # Create response
        response_data = MLModelResponse(training_job=training_job)
        response = SuccessResponse(
            message="Model training completed successfully",
            data=response_data.dict()
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error(f"ML model training failed: {str(e)}")
        error_response = ErrorResponse(
            message="ML model training failed",
            error_code="TRAINING_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500

@ml_bp.route('/predict', methods=['POST'])
def ml_predict():
    """
    Make ML prediction
    
    POST /api/ml/predict
    
    Body:
    {
        "model_type": "strategy_predictor",
        "prediction_type": "score_prediction",
        "input_data": {
            "strategy": {...},
            "context": {...}
        },
        "return_probabilities": true,
        "explain_prediction": true
    }
    """
    try:
        # Parse request data
        data = request.get_json() or {}
        
        # Validate request
        prediction_request = PredictionRequest(
            model_type=MLModelType(data.get('model_type')),
            prediction_type=PredictionType(data.get('prediction_type')),
            input_data=data.get('input_data', {}),
            return_probabilities=data.get('return_probabilities', False),
            explain_prediction=data.get('explain_prediction', False),
            context=data.get('context')
        )
        
        # Make prediction using VEX service
        vex_service = get_vex_service()
        prediction_result_data = vex_service.ml_predict(
            model_type=prediction_request.model_type.value,
            input_data=prediction_request.input_data
        )
        
        # Create prediction result
        prediction_result = PredictionResult(
            prediction_id=str(uuid.uuid4()),
            model_type=prediction_request.model_type,
            prediction_type=prediction_request.prediction_type,
            prediction=92.5,  # Mock prediction result
            confidence=0.87,
            probabilities={"high_score": 0.75, "medium_score": 0.20, "low_score": 0.05} if prediction_request.return_probabilities else None,
            explanation={
                "key_factors": ["autonomous_efficiency", "driver_coordination", "endgame_strategy"],
                "feature_contributions": {"autonomous_efficiency": 0.45, "driver_coordination": 0.32, "endgame_strategy": 0.23}
            } if prediction_request.explain_prediction else None,
            feature_importance={
                "autonomous_efficiency": 0.45,
                "driver_coordination": 0.32,
                "endgame_strategy": 0.23
            } if prediction_request.explain_prediction else None,
            input_data=prediction_request.input_data,
            model_version="1.5.0",
            processing_time=0.125
        )
        
        # Create response
        response_data = MLModelResponse(prediction_result=prediction_result)
        response = SuccessResponse(
            message="ML prediction completed successfully",
            data=response_data.dict()
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error(f"ML prediction failed: {str(e)}")
        error_response = ErrorResponse(
            message="ML prediction failed",
            error_code="PREDICTION_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500

@ml_bp.route('/optimize', methods=['POST'])
def optimize_scoring():
    """
    Optimize strategy/scoring using ML
    
    POST /api/ml/optimize
    
    Body:
    {
        "model_type": "scoring_optimizer",
        "optimization_target": "total_score",
        "strategy_data": {...},
        "constraints": {...},
        "max_iterations": 100
    }
    """
    try:
        # Parse request data
        data = request.get_json() or {}
        
        # Validate request
        optimization_request = OptimizationRequest(
            model_type=MLModelType(data.get('model_type', 'scoring_optimizer')),
            optimization_target=data.get('optimization_target', 'total_score'),
            strategy_data=data.get('strategy_data'),
            constraints=data.get('constraints'),
            max_iterations=data.get('max_iterations', 100)
        )
        
        # Optimize using VEX service
        vex_service = get_vex_service()
        optimization_result_data = vex_service.optimize_strategy(
            strategy_data=optimization_request.strategy_data or {},
            optimization_params={
                "target": optimization_request.optimization_target,
                "constraints": optimization_request.constraints,
                "max_iterations": optimization_request.max_iterations
            }
        )
        
        # Create optimization result
        optimization_result = OptimizationResult(
            optimization_id=str(uuid.uuid4()),
            model_type=optimization_request.model_type,
            optimized_strategy={
                "autonomous_strategy": "Optimized autonomous strategy for maximum efficiency",
                "driver_strategy": "Enhanced driver control with ML-predicted optimal paths",
                "expected_score": 98.7,
                "efficiency_rating": 0.94
            },
            improvement_metrics={
                "score_improvement": 15.2,
                "efficiency_improvement": 12.8,
                "consistency_improvement": 8.5
            },
            iterations=optimization_request.max_iterations,
            convergence_achieved=True,
            final_score=98.7,
            original_strategy=optimization_request.strategy_data or {},
            optimization_params=optimization_request,
            processing_time=45.2
        )
        
        # Create response
        response_data = MLModelResponse(optimization_result=optimization_result)
        response = SuccessResponse(
            message="Strategy optimization completed successfully",
            data=response_data.dict()
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error(f"ML optimization failed: {str(e)}")
        error_response = ErrorResponse(
            message="ML optimization failed",
            error_code="OPTIMIZATION_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500

@ml_bp.route('/patterns', methods=['POST'])
def discover_patterns():
    """
    Discover patterns in data using ML
    
    POST /api/ml/patterns
    
    Body:
    {
        "model_type": "pattern_discovery",
        "analysis_data": {...},
        "discovery_type": "temporal",
        "min_pattern_length": 3,
        "max_patterns": 50,
        "confidence_threshold": 0.7
    }
    """
    try:
        # Parse request data
        data = request.get_json() or {}
        
        # Validate request
        pattern_request = PatternDiscoveryRequest(
            model_type=MLModelType(data.get('model_type', 'pattern_discovery')),
            analysis_data=data.get('analysis_data', {}),
            discovery_type=data.get('discovery_type', 'temporal'),
            min_pattern_length=data.get('min_pattern_length', 3),
            max_patterns=data.get('max_patterns', 50),
            confidence_threshold=data.get('confidence_threshold', 0.7)
        )
        
        # Mock pattern discovery
        patterns = [
            {
                "pattern_id": str(uuid.uuid4()),
                "type": "scoring_sequence",
                "description": "High-scoring autonomous followed by aggressive driver period",
                "frequency": 0.78,
                "confidence": 0.89,
                "examples": ["match_1", "match_15", "match_23"]
            },
            {
                "pattern_id": str(uuid.uuid4()),
                "type": "coordination_pattern",
                "description": "Synchronized robot movements in endgame",
                "frequency": 0.65,
                "confidence": 0.82,
                "examples": ["match_5", "match_12", "match_31"]
            }
        ]
        
        # Create pattern discovery result
        discovery_result = PatternDiscoveryResult(
            discovery_id=str(uuid.uuid4()),
            patterns=patterns,
            pattern_count=len(patterns),
            most_significant=patterns[:3],  # Top 3 patterns
            temporal_patterns=patterns if pattern_request.discovery_type == "temporal" else None,
            discovery_params=pattern_request,
            confidence_scores={p["pattern_id"]: p["confidence"] for p in patterns},
            processing_time=12.8
        )
        
        # Create response
        response_data = MLModelResponse(pattern_discovery_result=discovery_result)
        response = SuccessResponse(
            message="Pattern discovery completed successfully",
            data=response_data.dict()
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error(f"Pattern discovery failed: {str(e)}")
        error_response = ErrorResponse(
            message="Pattern discovery failed",
            error_code="PATTERN_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500

@ml_bp.route('/models', methods=['GET'])
def list_ml_models():
    """
    List available ML models with their details
    
    GET /api/ml/models
    """
    try:
        models = [
            {
                "model_type": "coordination",
                "name": "Robot Coordination Model",
                "description": "Predicts optimal coordination strategies between alliance robots",
                "version": "1.2.0",
                "status": "available",
                "accuracy": 0.87,
                "last_trained": datetime.utcnow().isoformat()
            },
            {
                "model_type": "scoring_optimizer",
                "name": "Scoring Optimization Model", 
                "description": "Optimizes scoring strategies for maximum points",
                "version": "2.1.0",
                "status": "available",
                "accuracy": 0.92,
                "last_trained": datetime.utcnow().isoformat()
            },
            {
                "model_type": "strategy_predictor",
                "name": "Strategy Prediction Model",
                "description": "Predicts strategy effectiveness and outcomes",
                "version": "1.5.0", 
                "status": "available",
                "accuracy": 0.84,
                "last_trained": datetime.utcnow().isoformat()
            },
            {
                "model_type": "pattern_discovery",
                "name": "Pattern Discovery Model",
                "description": "Discovers patterns and trends in match data",
                "version": "1.0.0",
                "status": "available", 
                "accuracy": 0.79,
                "last_trained": datetime.utcnow().isoformat()
            }
        ]
        
        response = SuccessResponse(
            message="ML models retrieved successfully",
            data={"models": models, "total": len(models)}
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error(f"ML model listing failed: {str(e)}")
        error_response = ErrorResponse(
            message="Failed to list ML models",
            error_code="LISTING_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500

@ml_bp.route('/jobs/<job_id>', methods=['GET'])
def get_training_job(job_id: str):
    """
    Get training job status
    
    GET /api/ml/jobs/{job_id}
    """
    try:
        # Mock job status retrieval
        error_response = ErrorResponse(
            message="Training job not found",
            error_code="NOT_FOUND",
            error_details={"job_id": job_id}
        )
        return jsonify(error_response.dict()), 404
        
    except Exception as e:
        logger.error(f"Failed to get training job {job_id}: {str(e)}")
        error_response = ErrorResponse(
            message="Failed to retrieve training job",
            error_code="RETRIEVAL_ERROR",
            error_details={"error": str(e), "job_id": job_id}
        )
        return jsonify(error_response.dict()), 500