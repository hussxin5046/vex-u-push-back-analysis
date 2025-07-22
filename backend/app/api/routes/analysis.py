"""
Analysis API routes for running various types of VEX U analysis
"""

from flask import Blueprint, request, jsonify, current_app
from app.models.analysis import AnalysisRequest, AnalysisResponse, AnalysisResult, AnalysisType
from app.models.base import SuccessResponse, ErrorResponse
from app.services.vex_analysis_service import VEXAnalysisService
from datetime import datetime
import uuid
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

analysis_bp = Blueprint('analysis', __name__)

def get_vex_service() -> VEXAnalysisService:
    """Get VEX analysis service instance"""
    return VEXAnalysisService(
        vex_path=current_app.config['VEX_ANALYSIS_PATH'],
        python_path=current_app.config['PYTHON_PATH']
    )

@analysis_bp.route('/demo', methods=['POST'])
def run_demo_analysis():
    """
    Run quick demo analysis
    
    POST /api/analysis/demo
    
    Body:
    {
        "strategy_count": 10,
        "complexity": "basic"
    }
    """
    try:
        # Parse request data
        data = request.get_json() or {}
        
        # Validate request
        analysis_request = AnalysisRequest(
            analysis_type=AnalysisType.DEMO,
            strategy_count=data.get('strategy_count', 10),
            complexity=data.get('complexity', 'basic')
        )
        
        # Execute analysis
        vex_service = get_vex_service()
        result_data = vex_service.run_demo_analysis(
            strategy_count=analysis_request.strategy_count
        )
        
        # Create analysis result
        analysis_result = AnalysisResult(
            analysis_id=str(uuid.uuid4()),
            analysis_type=AnalysisType.DEMO,
            title="VEX U Demo Analysis",
            summary=f"Quick analysis of {analysis_request.strategy_count} strategies",
            created_at=datetime.utcnow(),
            parameters=analysis_request,
            raw_data=result_data
        )
        
        # Create response
        response_data = AnalysisResponse(result=analysis_result)
        response = SuccessResponse(
            message="Demo analysis completed successfully",
            data=response_data.dict()
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error(f"Demo analysis failed: {str(e)}")
        error_response = ErrorResponse(
            message="Demo analysis failed",
            error_code="ANALYSIS_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500

@analysis_bp.route('/full', methods=['POST'])
def run_full_analysis():
    """
    Run comprehensive analysis
    
    POST /api/analysis/full
    
    Body:
    {
        "strategy_count": 50,
        "simulation_count": 1000,
        "complexity": "intermediate",
        "focus_area": "overall",
        "include_ml": true
    }
    """
    try:
        # Parse request data
        data = request.get_json() or {}
        
        # Validate request
        analysis_request = AnalysisRequest(
            analysis_type=AnalysisType.FULL,
            strategy_count=data.get('strategy_count', 50),
            simulation_count=data.get('simulation_count', 1000),
            complexity=data.get('complexity', 'intermediate'),
            focus_area=data.get('focus_area'),
            include_ml=data.get('include_ml', True)
        )
        
        # Execute analysis
        vex_service = get_vex_service()
        result_data = vex_service.run_full_analysis(
            strategy_count=analysis_request.strategy_count,
            simulation_count=analysis_request.simulation_count,
            complexity=analysis_request.complexity
        )
        
        # Create analysis result
        analysis_result = AnalysisResult(
            analysis_id=str(uuid.uuid4()),
            analysis_type=AnalysisType.FULL,
            title="VEX U Comprehensive Analysis",
            summary=f"Full analysis of {analysis_request.strategy_count} strategies with {analysis_request.simulation_count} simulations",
            created_at=datetime.utcnow(),
            parameters=analysis_request,
            raw_data=result_data
        )
        
        # Create response
        response_data = AnalysisResponse(result=analysis_result)
        response = SuccessResponse(
            message="Full analysis completed successfully",
            data=response_data.dict()
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error(f"Full analysis failed: {str(e)}")
        error_response = ErrorResponse(
            message="Full analysis failed",
            error_code="ANALYSIS_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500

@analysis_bp.route('/statistical', methods=['POST'])
def run_statistical_analysis():
    """
    Run statistical analysis
    
    POST /api/analysis/statistical
    
    Body:
    {
        "sample_size": 1000,
        "statistical_method": "descriptive",
        "confidence_level": 0.95
    }
    """
    try:
        # Parse request data
        data = request.get_json() or {}
        
        # Validate request
        analysis_request = AnalysisRequest(
            analysis_type=AnalysisType.STATISTICAL,
            sample_size=data.get('sample_size', 1000),
            statistical_method=data.get('statistical_method', 'descriptive'),
            confidence_level=data.get('confidence_level', 0.95)
        )
        
        # Execute analysis
        vex_service = get_vex_service()
        result_data = vex_service.run_statistical_analysis(
            sample_size=analysis_request.sample_size,
            method=analysis_request.statistical_method
        )
        
        # Create analysis result
        analysis_result = AnalysisResult(
            analysis_id=str(uuid.uuid4()),
            analysis_type=AnalysisType.STATISTICAL,
            title="VEX U Statistical Analysis",
            summary=f"Statistical analysis using {analysis_request.statistical_method} method with sample size {analysis_request.sample_size}",
            created_at=datetime.utcnow(),
            parameters=analysis_request,
            raw_data=result_data
        )
        
        # Create response
        response_data = AnalysisResponse(result=analysis_result)
        response = SuccessResponse(
            message="Statistical analysis completed successfully",
            data=response_data.dict()
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error(f"Statistical analysis failed: {str(e)}")
        error_response = ErrorResponse(
            message="Statistical analysis failed",
            error_code="ANALYSIS_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500

@analysis_bp.route('/scoring', methods=['POST'])
def run_scoring_analysis():
    """
    Run scoring analysis
    
    POST /api/analysis/scoring
    
    Body:
    {
        "focus_area": "autonomous",
        "time_period": 30,
        "strategy_count": 25
    }
    """
    try:
        # Parse request data
        data = request.get_json() or {}
        
        # Validate request
        analysis_request = AnalysisRequest(
            analysis_type=AnalysisType.SCORING,
            focus_area=data.get('focus_area', 'overall'),
            time_period=data.get('time_period'),
            strategy_count=data.get('strategy_count', 25)
        )
        
        # For scoring analysis, we'll run a targeted full analysis
        vex_service = get_vex_service()
        result_data = vex_service.run_full_analysis(
            strategy_count=analysis_request.strategy_count,
            simulation_count=500,  # Moderate simulation count for scoring focus
            complexity='intermediate'
        )
        
        # Create analysis result
        analysis_result = AnalysisResult(
            analysis_id=str(uuid.uuid4()),
            analysis_type=AnalysisType.SCORING,
            title="VEX U Scoring Analysis",
            summary=f"Scoring analysis focused on {analysis_request.focus_area} with {analysis_request.strategy_count} strategies",
            created_at=datetime.utcnow(),
            parameters=analysis_request,
            raw_data=result_data
        )
        
        # Create response
        response_data = AnalysisResponse(result=analysis_result)
        response = SuccessResponse(
            message="Scoring analysis completed successfully",
            data=response_data.dict()
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error(f"Scoring analysis failed: {str(e)}")
        error_response = ErrorResponse(
            message="Scoring analysis failed",
            error_code="ANALYSIS_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500

@analysis_bp.route('/strategy', methods=['POST'])
def run_strategy_analysis():
    """
    Run strategy analysis
    
    POST /api/analysis/strategy
    
    Body:
    {
        "strategy_name": "Custom Strategy",
        "complexity": "advanced",
        "include_ml": true
    }
    """
    try:
        # Parse request data
        data = request.get_json() or {}
        
        # Validate request
        analysis_request = AnalysisRequest(
            analysis_type=AnalysisType.STRATEGY,
            strategy_name=data.get('strategy_name', 'Custom Strategy'),
            complexity=data.get('complexity', 'advanced'),
            include_ml=data.get('include_ml', True)
        )
        
        # For strategy analysis, run targeted analysis
        vex_service = get_vex_service()
        result_data = vex_service.run_full_analysis(
            strategy_count=20,  # Focused strategy count
            simulation_count=1000,
            complexity=analysis_request.complexity
        )
        
        # Create analysis result
        analysis_result = AnalysisResult(
            analysis_id=str(uuid.uuid4()),
            analysis_type=AnalysisType.STRATEGY,
            title="VEX U Strategy Analysis",
            summary=f"Strategy analysis for '{analysis_request.strategy_name}' at {analysis_request.complexity} complexity",
            created_at=datetime.utcnow(),
            parameters=analysis_request,
            raw_data=result_data
        )
        
        # Create response
        response_data = AnalysisResponse(result=analysis_result)
        response = SuccessResponse(
            message="Strategy analysis completed successfully",
            data=response_data.dict()
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error(f"Strategy analysis failed: {str(e)}")
        error_response = ErrorResponse(
            message="Strategy analysis failed",
            error_code="ANALYSIS_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500

@analysis_bp.route('/history', methods=['GET'])
def get_analysis_history():
    """
    Get analysis history
    
    GET /api/analysis/history?limit=50&offset=0
    """
    try:
        # For now, return empty history since we don't have persistence
        # In a real implementation, this would query a database
        
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        
        response = SuccessResponse(
            message="Analysis history retrieved successfully",
            data={
                "analyses": [],
                "total": 0,
                "limit": limit,
                "offset": offset
            }
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error(f"Failed to get analysis history: {str(e)}")
        error_response = ErrorResponse(
            message="Failed to retrieve analysis history",
            error_code="HISTORY_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500

@analysis_bp.route('/<analysis_id>', methods=['GET'])
def get_analysis_by_id(analysis_id: str):
    """
    Get specific analysis by ID
    
    GET /api/analysis/{analysis_id}
    """
    try:
        # For now, return not found since we don't have persistence
        # In a real implementation, this would query a database
        
        error_response = ErrorResponse(
            message="Analysis not found",
            error_code="NOT_FOUND",
            error_details={"analysis_id": analysis_id}
        )
        return jsonify(error_response.dict()), 404
        
    except Exception as e:
        logger.error(f"Failed to get analysis {analysis_id}: {str(e)}")
        error_response = ErrorResponse(
            message="Failed to retrieve analysis",
            error_code="RETRIEVAL_ERROR",
            error_details={"error": str(e), "analysis_id": analysis_id}
        )
        return jsonify(error_response.dict()), 500