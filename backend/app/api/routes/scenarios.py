"""
Scenario API routes for generating and managing match scenarios
"""

from flask import Blueprint, request, jsonify, current_app
from app.models.scenario import ScenarioRequest, ScenarioResponse, ScenarioSet, Match, MatchType, SimulationParameters, ScenarioComplexity
from app.models.base import SuccessResponse, ErrorResponse
from app.services.vex_analysis_service import VEXAnalysisService
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)

scenarios_bp = Blueprint('scenarios', __name__)

def get_vex_service() -> VEXAnalysisService:
    """Get VEX analysis service instance"""
    return VEXAnalysisService(
        vex_path=current_app.config['VEX_ANALYSIS_PATH'],
        python_path=current_app.config['PYTHON_PATH']
    )

@scenarios_bp.route('/generate', methods=['POST'])
def generate_scenarios():
    """
    Generate match scenarios
    
    POST /api/scenarios/generate
    
    Body:
    {
        "scenario_count": 50,
        "complexity_level": "moderate",
        "strategy_pool": ["strategy-1", "strategy-2"],
        "auto_generate_strategies": true,
        "simulation_params": {
            "match_count": 100,
            "autonomous_duration": 30,
            "driver_duration": 90
        },
        "enable_evolution": false
    }
    """
    try:
        # Parse request data
        data = request.get_json() or {}
        
        # Validate request
        scenario_request = ScenarioRequest(
            scenario_count=data.get('scenario_count', 10),
            complexity_level=ScenarioComplexity(data.get('complexity_level', 'moderate')),
            strategy_pool=data.get('strategy_pool'),
            auto_generate_strategies=data.get('auto_generate_strategies', True),
            simulation_params=SimulationParameters(**data.get('simulation_params', {})),
            focus_areas=data.get('focus_areas'),
            include_ml_predictions=data.get('include_ml_predictions', True),
            enable_evolution=data.get('enable_evolution', False),
            evolution_generations=data.get('evolution_generations')
        )
        
        # Generate scenarios using VEX service
        vex_service = get_vex_service()
        scenario_params = {
            "scenario_count": scenario_request.scenario_count,
            "complexity": scenario_request.complexity_level.value,
            "strategy_pool": scenario_request.strategy_pool or [],
            "auto_generate": scenario_request.auto_generate_strategies,
            "simulation_params": scenario_request.simulation_params.dict(),
            "include_ml": scenario_request.include_ml_predictions,
            "evolution": scenario_request.enable_evolution
        }
        
        scenario_result = vex_service.generate_scenarios(scenario_params)
        
        # Create mock scenarios
        scenarios = []
        strategy_pool = scenario_request.strategy_pool or [f"strategy-{i}" for i in range(1, 11)]
        
        for i in range(scenario_request.scenario_count):
            # Randomly select strategies from pool
            import random
            red_strategy = random.choice(strategy_pool)
            blue_strategy = random.choice([s for s in strategy_pool if s != red_strategy])
            
            scenario = Match(
                match_id=str(uuid.uuid4()),
                red_alliance_strategy=red_strategy,
                blue_alliance_strategy=blue_strategy,
                match_type=MatchType.PRACTICE,
                field_setup={
                    "complexity": scenario_request.complexity_level.value,
                    "special_conditions": scenario_request.focus_areas or []
                }
            )
            scenarios.append(scenario)
        
        # Create scenario set
        scenario_set = ScenarioSet(
            scenario_set_id=str(uuid.uuid4()),
            name=f"{scenario_request.complexity_level.value.title()} Scenario Set",
            description=f"Generated {scenario_request.scenario_count} scenarios with {scenario_request.complexity_level.value} complexity",
            scenarios=scenarios,
            total_matches=len(scenarios),
            generation_params=scenario_request,
            estimated_duration=len(scenarios) * 2.5  # Estimated 2.5 seconds per scenario
        )
        
        # Create response
        response_data = ScenarioResponse(scenario_set=scenario_set)
        response = SuccessResponse(
            message="Scenarios generated successfully",
            data=response_data.dict()
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error(f"Scenario generation failed: {str(e)}")
        error_response = ErrorResponse(
            message="Scenario generation failed",
            error_code="GENERATION_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500

@scenarios_bp.route('/simulate', methods=['POST'])
def simulate_scenarios():
    """
    Simulate generated scenarios
    
    POST /api/scenarios/simulate
    
    Body:
    {
        "scenario_set_id": "set-123",
        "simulation_params": {
            "match_count": 100,
            "performance_variance": 0.1,
            "random_events": true
        }
    }
    """
    try:
        # Parse request data
        data = request.get_json() or {}
        scenario_set_id = data.get('scenario_set_id')
        simulation_params = SimulationParameters(**data.get('simulation_params', {}))
        
        if not scenario_set_id:
            error_response = ErrorResponse(
                message="Scenario set ID is required",
                error_code="VALIDATION_ERROR",
                error_details={"field": "scenario_set_id"}
            )
            return jsonify(error_response.dict()), 400
        
        # Mock simulation results
        from app.models.scenario import SimulationResults, MatchResult, MatchScore, MatchOutcome
        
        # Create mock match results
        match_results = []
        red_wins = 0
        blue_wins = 0
        ties = 0
        
        for i in range(simulation_params.match_count):
            # Generate random scores
            import random
            
            red_auto = random.randint(15, 45)
            red_driver = random.randint(40, 70)
            red_endgame = random.randint(10, 25)
            red_total = red_auto + red_driver + red_endgame
            
            blue_auto = random.randint(15, 45)
            blue_driver = random.randint(40, 70)
            blue_endgame = random.randint(10, 25)
            blue_total = blue_auto + blue_driver + blue_endgame
            
            # Determine outcome
            if red_total > blue_total:
                outcome = MatchOutcome.RED_WIN
                red_wins += 1
            elif blue_total > red_total:
                outcome = MatchOutcome.BLUE_WIN
                blue_wins += 1
            else:
                outcome = MatchOutcome.TIE
                ties += 1
            
            match_result = MatchResult(
                match_id=str(uuid.uuid4()),
                match_number=i + 1,
                match_type=MatchType.PRACTICE,
                red_alliance=f"red-strategy-{i % 5 + 1}",
                blue_alliance=f"blue-strategy-{i % 5 + 1}",
                red_score=MatchScore(
                    autonomous_score=red_auto,
                    driver_score=red_driver,
                    endgame_score=red_endgame,
                    penalty_score=0,
                    total_score=red_total
                ),
                blue_score=MatchScore(
                    autonomous_score=blue_auto,
                    driver_score=blue_driver,
                    endgame_score=blue_endgame,
                    penalty_score=0,
                    total_score=blue_total
                ),
                outcome=outcome,
                margin_of_victory=abs(red_total - blue_total),
                match_duration=120.0,
                key_events=[f"Event {j}" for j in range(random.randint(1, 4))],
                red_performance={"efficiency": random.uniform(0.7, 0.95)},
                blue_performance={"efficiency": random.uniform(0.7, 0.95)}
            )
            match_results.append(match_result)
        
        # Create simulation results
        simulation_results = SimulationResults(
            scenario_set_id=scenario_set_id,
            simulation_id=str(uuid.uuid4()),
            matches=match_results,
            total_matches=len(match_results),
            successful_matches=len(match_results),
            red_wins=red_wins,
            blue_wins=blue_wins,
            ties=ties,
            average_score=sum(m.red_score.total_score + m.blue_score.total_score for m in match_results) / (2 * len(match_results)),
            score_distribution={"mean": 85.5, "std": 12.3, "min": 45, "max": 125},
            strategy_performance={},
            simulation_duration=len(match_results) * 0.1,  # Mock duration
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow()
        )
        
        response = SuccessResponse(
            message="Scenario simulation completed successfully",
            data=simulation_results.dict()
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error(f"Scenario simulation failed: {str(e)}")
        error_response = ErrorResponse(
            message="Scenario simulation failed",
            error_code="SIMULATION_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500

@scenarios_bp.route('/', methods=['GET'])
def list_scenarios():
    """
    List available scenario sets
    
    GET /api/scenarios/?limit=20&offset=0&complexity=moderate
    """
    try:
        limit = int(request.args.get('limit', 20))
        offset = int(request.args.get('offset', 0))
        complexity = request.args.get('complexity')
        
        # Mock scenario list
        scenarios = []
        
        response = SuccessResponse(
            message="Scenarios retrieved successfully",
            data={
                "scenario_sets": scenarios,
                "total": 0,
                "limit": limit,
                "offset": offset,
                "filters": {"complexity": complexity} if complexity else {}
            }
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error(f"Scenario listing failed: {str(e)}")
        error_response = ErrorResponse(
            message="Failed to list scenarios",
            error_code="LISTING_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500

@scenarios_bp.route('/<scenario_set_id>', methods=['GET'])
def get_scenario_set(scenario_set_id: str):
    """
    Get specific scenario set by ID
    
    GET /api/scenarios/{scenario_set_id}
    """
    try:
        # Mock scenario set retrieval
        error_response = ErrorResponse(
            message="Scenario set not found",
            error_code="NOT_FOUND",
            error_details={"scenario_set_id": scenario_set_id}
        )
        return jsonify(error_response.dict()), 404
        
    except Exception as e:
        logger.error(f"Failed to get scenario set {scenario_set_id}: {str(e)}")
        error_response = ErrorResponse(
            message="Failed to retrieve scenario set",
            error_code="RETRIEVAL_ERROR",
            error_details={"error": str(e), "scenario_set_id": scenario_set_id}
        )
        return jsonify(error_response.dict()), 500

@scenarios_bp.route('/evolve', methods=['POST'])
def evolve_scenarios():
    """
    Evolve scenarios using genetic algorithms
    
    POST /api/scenarios/evolve
    
    Body:
    {
        "base_scenario_set_id": "set-123",
        "evolution_generations": 10,
        "population_size": 50,
        "mutation_rate": 0.1,
        "selection_pressure": 0.7
    }
    """
    try:
        # Parse request data
        data = request.get_json() or {}
        base_scenario_set_id = data.get('base_scenario_set_id')
        evolution_generations = data.get('evolution_generations', 10)
        population_size = data.get('population_size', 50)
        
        if not base_scenario_set_id:
            error_response = ErrorResponse(
                message="Base scenario set ID is required",
                error_code="VALIDATION_ERROR",
                error_details={"field": "base_scenario_set_id"}
            )
            return jsonify(error_response.dict()), 400
        
        # Mock evolution process
        evolution_result = {
            "evolution_id": str(uuid.uuid4()),
            "base_scenario_set_id": base_scenario_set_id,
            "generations_completed": evolution_generations,
            "final_population_size": population_size,
            "best_fitness": 0.94,
            "average_fitness": 0.78,
            "evolution_duration": evolution_generations * 2.5,
            "new_scenario_set_id": str(uuid.uuid4())
        }
        
        response = SuccessResponse(
            message="Scenario evolution completed successfully",
            data=evolution_result
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error(f"Scenario evolution failed: {str(e)}")
        error_response = ErrorResponse(
            message="Scenario evolution failed",
            error_code="EVOLUTION_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500

@scenarios_bp.route('/export', methods=['POST'])
def export_scenarios():
    """
    Export scenario set to file
    
    POST /api/scenarios/export
    
    Body:
    {
        "scenario_set_id": "set-123",
        "format": "json",
        "include_results": true
    }
    """
    try:
        # Parse request data
        data = request.get_json() or {}
        scenario_set_id = data.get('scenario_set_id')
        export_format = data.get('format', 'json')
        include_results = data.get('include_results', False)
        
        if not scenario_set_id:
            error_response = ErrorResponse(
                message="Scenario set ID is required",
                error_code="VALIDATION_ERROR",
                error_details={"field": "scenario_set_id"}
            )
            return jsonify(error_response.dict()), 400
        
        # Mock export process
        export_result = {
            "export_id": str(uuid.uuid4()),
            "scenario_set_id": scenario_set_id,
            "format": export_format,
            "file_size": 2048,  # Mock file size
            "download_url": f"/api/scenarios/{scenario_set_id}/download?format={export_format}",
            "expires_at": datetime.utcnow().isoformat()
        }
        
        response = SuccessResponse(
            message="Scenario export completed",
            data=export_result
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error(f"Scenario export failed: {str(e)}")
        error_response = ErrorResponse(
            message="Scenario export failed",
            error_code="EXPORT_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500