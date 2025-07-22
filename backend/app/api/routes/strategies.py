"""
Strategy API routes for managing alliance strategies
"""

from flask import Blueprint, request, jsonify, current_app
from app.models.strategy import StrategyRequest, StrategyResponse, AllianceStrategy, Robot, RobotRole, StrategyType, StrategyOptimizationRequest
from app.models.base import SuccessResponse, ErrorResponse
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)

strategies_bp = Blueprint('strategies', __name__)

@strategies_bp.route('/', methods=['GET'])
def get_strategies():
    """
    Get available strategies
    
    GET /api/strategies/?limit=20&offset=0&type=aggressive
    """
    try:
        limit = int(request.args.get('limit', 20))
        offset = int(request.args.get('offset', 0))
        strategy_type = request.args.get('type')
        
        # Mock strategy data
        strategies = []
        
        # Create sample strategies
        sample_strategies = [
            {
                "strategy_id": str(uuid.uuid4()),
                "name": "Aggressive Offense",
                "strategy_type": StrategyType.AGGRESSIVE,
                "expected_score": 95.5,
                "win_probability": 0.78,
                "risk_level": "high"
            },
            {
                "strategy_id": str(uuid.uuid4()),
                "name": "Balanced Approach",
                "strategy_type": StrategyType.BALANCED,
                "expected_score": 87.2,
                "win_probability": 0.82,
                "risk_level": "medium"
            },
            {
                "strategy_id": str(uuid.uuid4()),
                "name": "Defensive Control",
                "strategy_type": StrategyType.DEFENSIVE,
                "expected_score": 82.8,
                "win_probability": 0.85,
                "risk_level": "low"
            }
        ]
        
        # Filter by type if specified
        if strategy_type:
            sample_strategies = [s for s in sample_strategies if s["strategy_type"].value == strategy_type]
        
        # Apply pagination
        total = len(sample_strategies)
        strategies = sample_strategies[offset:offset + limit]
        
        response = SuccessResponse(
            message="Strategies retrieved successfully",
            data={
                "strategies": strategies,
                "total": total,
                "limit": limit,
                "offset": offset
            }
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error(f"Strategy listing failed: {str(e)}")
        error_response = ErrorResponse(
            message="Failed to list strategies",
            error_code="LISTING_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500

@strategies_bp.route('/generate', methods=['POST'])
def generate_strategy():
    """
    Generate new strategy
    
    POST /api/strategies/generate
    
    Body:
    {
        "strategy_type": "balanced",
        "robot_count": 2,
        "min_expected_score": 80,
        "optimize_for": "score",
        "focus_period": "overall"
    }
    """
    try:
        # Parse request data
        data = request.get_json() or {}
        
        # Validate request
        strategy_request = StrategyRequest(
            strategy_type=StrategyType(data.get('strategy_type', 'balanced')),
            robot_count=data.get('robot_count', 2),
            min_expected_score=data.get('min_expected_score'),
            optimize_for=data.get('optimize_for', 'score'),
            focus_period=data.get('focus_period', 'overall')
        )
        
        # Generate mock robots
        robots = []
        
        if strategy_request.robot_count >= 1:
            robots.append(Robot(
                robot_id=str(uuid.uuid4()),
                name="Primary Robot",
                role=RobotRole.OFFENSE if strategy_request.strategy_type == StrategyType.AGGRESSIVE else RobotRole.VERSATILE,
                size="medium",
                speed=8.5,
                maneuverability=7.8,
                scoring_ability=9.2,
                defensive_ability=6.5,
                reliability=8.8,
                autonomous_efficiency=0.85,
                driver_efficiency=0.88,
                autonomous_score=35.2,
                driver_score=52.8,
                total_score=88.0
            ))
        
        if strategy_request.robot_count >= 2:
            robots.append(Robot(
                robot_id=str(uuid.uuid4()),
                name="Support Robot",
                role=RobotRole.SUPPORT if strategy_request.strategy_type != StrategyType.AGGRESSIVE else RobotRole.DEFENSE,
                size="small",
                speed=7.2,
                maneuverability=8.9,
                scoring_ability=6.8,
                defensive_ability=8.5,
                reliability=9.1,
                autonomous_efficiency=0.78,
                driver_efficiency=0.82,
                autonomous_score=28.5,
                driver_score=45.3,
                total_score=73.8
            ))
        
        # Calculate strategy metrics
        total_robot_score = sum(robot.total_score for robot in robots)
        avg_efficiency = sum(robot.autonomous_efficiency + robot.driver_efficiency for robot in robots) / (2 * len(robots))
        
        # Create strategy
        strategy = AllianceStrategy(
            strategy_id=str(uuid.uuid4()),
            name=f"{strategy_request.strategy_type.value.title()} Strategy",
            strategy_type=strategy_request.strategy_type,
            robots=robots,
            autonomous_strategy=f"Autonomous strategy optimized for {strategy_request.focus_period}",
            driver_strategy=f"Driver control strategy focusing on {strategy_request.optimize_for}",
            endgame_strategy="Coordinated endgame positioning for maximum points",
            expected_score=total_robot_score * 1.1,  # Team synergy bonus
            win_probability=min(0.95, avg_efficiency * 0.9 + 0.1),
            risk_level="high" if strategy_request.strategy_type == StrategyType.AGGRESSIVE else 
                      "low" if strategy_request.strategy_type == StrategyType.DEFENSIVE else "medium"
        )
        
        # Create response
        response_data = StrategyResponse(strategy=strategy)
        response = SuccessResponse(
            message="Strategy generated successfully",
            data=response_data.dict()
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error(f"Strategy generation failed: {str(e)}")
        error_response = ErrorResponse(
            message="Strategy generation failed",
            error_code="GENERATION_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500

@strategies_bp.route('/optimize', methods=['POST'])
def optimize_strategy():
    """
    Optimize existing strategy
    
    POST /api/strategies/optimize
    
    Body:
    {
        "strategy_id": "strategy-123",
        "optimization_target": "score",
        "constraints": {"max_risk": "medium"},
        "max_iterations": 100,
        "use_ml_predictions": true
    }
    """
    try:
        # Parse request data
        data = request.get_json() or {}
        
        # Validate request
        optimization_request = StrategyOptimizationRequest(
            strategy_id=data.get('strategy_id'),
            optimization_target=data.get('optimization_target', 'score'),
            constraints=data.get('constraints'),
            max_iterations=data.get('max_iterations', 100),
            use_ml_predictions=data.get('use_ml_predictions', True)
        )
        
        if not optimization_request.strategy_id:
            error_response = ErrorResponse(
                message="Strategy ID is required",
                error_code="VALIDATION_ERROR",
                error_details={"field": "strategy_id"}
            )
            return jsonify(error_response.dict()), 400
        
        # Mock optimization process
        # In a real implementation, this would:
        # 1. Load the existing strategy
        # 2. Apply ML optimization algorithms
        # 3. Return the optimized strategy
        
        # Create optimized robot
        optimized_robot = Robot(
            robot_id=str(uuid.uuid4()),
            name="Optimized Primary Robot",
            role=RobotRole.VERSATILE,
            size="medium",
            speed=9.1,  # Improved from optimization
            maneuverability=8.5,  # Improved
            scoring_ability=9.8,  # Improved
            defensive_ability=7.2,  # Improved
            reliability=9.2,  # Improved
            autonomous_efficiency=0.92,  # Optimized
            driver_efficiency=0.95,  # Optimized
            autonomous_score=42.8,  # Higher due to optimization
            driver_score=58.5,  # Higher due to optimization
            total_score=101.3  # Optimized total
        )
        
        # Create optimized strategy
        optimized_strategy = AllianceStrategy(
            strategy_id=str(uuid.uuid4()),
            name="ML-Optimized Strategy",
            strategy_type=StrategyType.BALANCED,
            robots=[optimized_robot],
            autonomous_strategy="ML-optimized autonomous strategy for maximum point efficiency",
            driver_strategy="Optimized driver control strategy with predictive positioning",
            endgame_strategy="AI-enhanced endgame strategy with dynamic adaptation",
            expected_score=105.8,  # Improved through optimization
            win_probability=0.89,  # Higher probability
            risk_level="medium"
        )
        
        # Create optimization info
        optimization_info = {
            "iterations_performed": optimization_request.max_iterations,
            "convergence_achieved": True,
            "improvement_percentage": 15.2,
            "optimization_target": optimization_request.optimization_target,
            "ml_models_used": ["scoring_optimizer", "strategy_predictor"] if optimization_request.use_ml_predictions else []
        }
        
        # Create response
        response_data = StrategyResponse(
            strategy=optimized_strategy,
            optimization_info=optimization_info
        )
        response = SuccessResponse(
            message="Strategy optimization completed successfully",
            data=response_data.dict()
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error(f"Strategy optimization failed: {str(e)}")
        error_response = ErrorResponse(
            message="Strategy optimization failed",
            error_code="OPTIMIZATION_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500

@strategies_bp.route('/<strategy_id>', methods=['GET'])
def get_strategy(strategy_id: str):
    """
    Get specific strategy by ID
    
    GET /api/strategies/{strategy_id}
    """
    try:
        # Mock strategy retrieval
        error_response = ErrorResponse(
            message="Strategy not found",
            error_code="NOT_FOUND",
            error_details={"strategy_id": strategy_id}
        )
        return jsonify(error_response.dict()), 404
        
    except Exception as e:
        logger.error(f"Failed to get strategy {strategy_id}: {str(e)}")
        error_response = ErrorResponse(
            message="Failed to retrieve strategy",
            error_code="RETRIEVAL_ERROR",
            error_details={"error": str(e), "strategy_id": strategy_id}
        )
        return jsonify(error_response.dict()), 500

@strategies_bp.route('/', methods=['POST'])
def save_strategy():
    """
    Save new strategy
    
    POST /api/strategies/
    
    Body: AllianceStrategy object
    """
    try:
        # Parse request data
        data = request.get_json()
        
        if not data:
            error_response = ErrorResponse(
                message="Strategy data is required",
                error_code="VALIDATION_ERROR",
                error_details={"field": "strategy_data"}
            )
            return jsonify(error_response.dict()), 400
        
        # In a real implementation, this would validate and save the strategy
        saved_strategy_id = str(uuid.uuid4())
        
        response = SuccessResponse(
            message="Strategy saved successfully",
            data={
                "strategy_id": saved_strategy_id,
                "saved_at": datetime.utcnow().isoformat()
            }
        )
        
        return jsonify(response.dict()), 201
        
    except Exception as e:
        logger.error(f"Strategy saving failed: {str(e)}")
        error_response = ErrorResponse(
            message="Strategy saving failed",
            error_code="SAVE_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500

@strategies_bp.route('/compare', methods=['POST'])
def compare_strategies():
    """
    Compare multiple strategies
    
    POST /api/strategies/compare
    
    Body:
    {
        "strategy_ids": ["strategy-1", "strategy-2"],
        "comparison_metrics": ["score", "efficiency", "risk"]
    }
    """
    try:
        # Parse request data
        data = request.get_json() or {}
        strategy_ids = data.get('strategy_ids', [])
        comparison_metrics = data.get('comparison_metrics', ['score', 'efficiency', 'risk'])
        
        if len(strategy_ids) < 2:
            error_response = ErrorResponse(
                message="At least 2 strategies required for comparison",
                error_code="VALIDATION_ERROR",
                error_details={"strategy_ids": strategy_ids}
            )
            return jsonify(error_response.dict()), 400
        
        # Mock comparison results
        comparison_results = {
            "comparison_id": str(uuid.uuid4()),
            "strategies_compared": len(strategy_ids),
            "metrics_analyzed": comparison_metrics,
            "winner": strategy_ids[0],  # Mock winner
            "comparison_summary": "Strategy A shows 12% better performance in overall scoring",
            "detailed_comparison": {
                strategy_ids[0]: {"score": 95.2, "efficiency": 0.87, "risk": "medium"},
                strategy_ids[1]: {"score": 84.8, "efficiency": 0.82, "risk": "low"}
            }
        }
        
        response = SuccessResponse(
            message="Strategy comparison completed",
            data=comparison_results
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error(f"Strategy comparison failed: {str(e)}")
        error_response = ErrorResponse(
            message="Strategy comparison failed",
            error_code="COMPARISON_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500