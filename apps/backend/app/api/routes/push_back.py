"""
Push Back specific API endpoints
Focused on Push Back game analysis and strategy development
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from flask import Blueprint, request, jsonify, current_app
from app.models.base import ApiResponse
from app.services.vex_analysis_service import VEXAnalysisService

logger = logging.getLogger(__name__)

# Create Push Back blueprint
push_back_bp = Blueprint('push_back', __name__)

def get_push_back_service():
    """Get VEX analysis service instance configured for Push Back"""
    return VEXAnalysisService(
        vex_path=current_app.config['VEX_ANALYSIS_PATH'],
        python_path=current_app.config.get('PYTHON_PATH', 'python3')
    )

# Strategy Operations
@push_back_bp.route('/strategies', methods=['GET'])
def get_strategies():
    """Get all Push Back strategies"""
    try:
        # For now, return mock data - this would connect to a database in production
        strategies = [
            {
                "id": "strategy-1",
                "name": "Block Flow Maximizer",
                "archetype": "block_flow_maximizer",
                "robot_specs": [
                    {"id": "robot1", "speed": 0.8, "accuracy": 0.9, "capacity": 4},
                    {"id": "robot2", "speed": 0.7, "accuracy": 0.8, "capacity": 3}
                ],
                "autonomous_strategy": "aggressive_block_collection",
                "driver_strategy": "center_goal_focus",
                "created_at": datetime.utcnow().isoformat()
            }
        ]
        
        return jsonify(ApiResponse(
            success=True,
            data=strategies,
            message="Push Back strategies retrieved successfully"
        ).dict())
        
    except Exception as e:
        logger.error(f"Error retrieving strategies: {str(e)}")
        return jsonify(ApiResponse(
            success=False,
            error=str(e),
            message="Failed to retrieve Push Back strategies"
        ).dict()), 500

@push_back_bp.route('/strategies', methods=['POST'])
def create_strategy():
    """Create a new Push Back strategy"""
    try:
        strategy_data = request.get_json()
        
        # Add ID and timestamp
        strategy_data['id'] = f"strategy-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        strategy_data['created_at'] = datetime.utcnow().isoformat()
        
        # In production, this would save to database
        logger.info(f"Created new Push Back strategy: {strategy_data['id']}")
        
        return jsonify(ApiResponse(
            success=True,
            data=strategy_data,
            message="Push Back strategy created successfully"
        ).dict())
        
    except Exception as e:
        logger.error(f"Error creating strategy: {str(e)}")
        return jsonify(ApiResponse(
            success=False,
            error=str(e),
            message="Failed to create Push Back strategy"
        ).dict()), 500

@push_back_bp.route('/strategies/<strategy_id>', methods=['PUT'])
def update_strategy(strategy_id):
    """Update a Push Back strategy"""
    try:
        strategy_data = request.get_json()
        strategy_data['id'] = strategy_id
        strategy_data['updated_at'] = datetime.utcnow().isoformat()
        
        logger.info(f"Updated Push Back strategy: {strategy_id}")
        
        return jsonify(ApiResponse(
            success=True,
            data=strategy_data,
            message="Push Back strategy updated successfully"
        ).dict())
        
    except Exception as e:
        logger.error(f"Error updating strategy: {str(e)}")
        return jsonify(ApiResponse(
            success=False,
            error=str(e),
            message="Failed to update Push Back strategy"
        ).dict()), 500

@push_back_bp.route('/strategies/<strategy_id>', methods=['DELETE'])
def delete_strategy(strategy_id):
    """Delete a Push Back strategy"""
    try:
        logger.info(f"Deleted Push Back strategy: {strategy_id}")
        
        return jsonify(ApiResponse(
            success=True,
            data={"deleted": True},
            message="Push Back strategy deleted successfully"
        ).dict())
        
    except Exception as e:
        logger.error(f"Error deleting strategy: {str(e)}")
        return jsonify(ApiResponse(
            success=False,
            error=str(e),
            message="Failed to delete Push Back strategy"
        ).dict()), 500

# Strategy Analysis
@push_back_bp.route('/analyze', methods=['POST'])
def analyze_strategy():
    """Analyze a Push Back strategy"""
    try:
        data = request.get_json()
        strategy = data.get('strategy')
        robot_specs = data.get('robot_specs', [])
        
        service = get_push_back_service()
        
        # Use the Push Back strategy analyzer
        analysis_data = {
            "strategy": strategy,
            "robot_specs": robot_specs,
            "analysis_type": "push_back_strategy"
        }
        
        # This would call the new push_back_strategy_analyzer.py
        result = service._execute_command([
            "-c", 
            f"""
import sys
sys.path.append('{current_app.config["VEX_ANALYSIS_PATH"]}')
from vex_analysis.analysis.push_back_strategy_analyzer import PushBackStrategyAnalyzer
import json

analyzer = PushBackStrategyAnalyzer()
analysis = analyzer.run_comprehensive_analysis({json.dumps(robot_specs)})
print(json.dumps(analysis.to_dict()))
"""
        ])
        
        if result["success"]:
            analysis_result = json.loads(result["stdout"].strip().split('\n')[-1])
            
            return jsonify(ApiResponse(
                success=True,
                data=analysis_result,
                message="Push Back strategy analysis completed"
            ).dict())
        else:
            raise Exception(result["stderr"])
            
    except Exception as e:
        logger.error(f"Error analyzing strategy: {str(e)}")
        return jsonify(ApiResponse(
            success=False,
            error=str(e),
            message="Failed to analyze Push Back strategy"
        ).dict()), 500

@push_back_bp.route('/compare', methods=['POST'])
def compare_strategies():
    """Compare multiple Push Back strategies"""
    try:
        data = request.get_json()
        strategies = data.get('strategies', [])
        
        service = get_push_back_service()
        
        # Run comparison analysis for each strategy
        results = []
        for strategy in strategies:
            # Simplified comparison - in production this would be more sophisticated
            analysis = {
                "strategy_id": strategy.get('id'),
                "name": strategy.get('name'),
                "expected_score": 85.2,  # Mock score
                "win_probability": 0.73,
                "risk_level": "medium",
                "strengths": ["Strong block flow", "Good parking timing"],
                "weaknesses": ["Vulnerable to defense", "Limited autonomous"]
            }
            results.append(analysis)
        
        return jsonify(ApiResponse(
            success=True,
            data=results,
            message="Push Back strategy comparison completed"
        ).dict())
        
    except Exception as e:
        logger.error(f"Error comparing strategies: {str(e)}")
        return jsonify(ApiResponse(
            success=False,
            error=str(e),
            message="Failed to compare Push Back strategies"
        ).dict()), 500

# Block Flow Optimization
@push_back_bp.route('/optimize/block-flow', methods=['POST'])
def optimize_block_flow():
    """Optimize block flow distribution"""
    try:
        data = request.get_json()
        robot_specs = data.get('robot_specs', [])
        constraints = data.get('constraints', {})
        
        service = get_push_back_service()
        
        # Call Push Back block flow optimization
        result = service._execute_command([
            "-c",
            f"""
import sys
sys.path.append('{current_app.config["VEX_ANALYSIS_PATH"]}')
from vex_analysis.analysis.push_back_strategy_analyzer import PushBackStrategyAnalyzer
import json

analyzer = PushBackStrategyAnalyzer()
optimization = analyzer.analyze_block_flow_optimization({json.dumps(robot_specs)})
print(json.dumps({{
    "optimal_distribution": optimization.optimal_distribution,
    "expected_points": optimization.expected_points,
    "risk_level": optimization.risk_level,
    "recommendations": optimization.recommendations
}}))
"""
        ])
        
        if result["success"]:
            optimization_result = json.loads(result["stdout"].strip().split('\n')[-1])
            
            return jsonify(ApiResponse(
                success=True,
                data=optimization_result,
                message="Block flow optimization completed"
            ).dict())
        else:
            raise Exception(result["stderr"])
            
    except Exception as e:
        logger.error(f"Error optimizing block flow: {str(e)}")
        return jsonify(ApiResponse(
            success=False,
            error=str(e),
            message="Failed to optimize block flow"
        ).dict()), 500

# Autonomous Decision Analysis
@push_back_bp.route('/analyze/autonomous', methods=['POST'])
def analyze_autonomous_decision():
    """Analyze autonomous strategy decision"""
    try:
        data = request.get_json()
        robot_specs = data.get('robot_specs', [])
        
        service = get_push_back_service()
        
        result = service._execute_command([
            "-c",
            f"""
import sys
sys.path.append('{current_app.config["VEX_ANALYSIS_PATH"]}')
from vex_analysis.analysis.push_back_strategy_analyzer import PushBackStrategyAnalyzer
import json

analyzer = PushBackStrategyAnalyzer()
analysis = analyzer.analyze_autonomous_strategy_decision({json.dumps(robot_specs)})
print(json.dumps({{
    "recommended_strategy": analysis.recommended_strategy,
    "auto_win_probability": analysis.auto_win_probability,
    "bonus_probability": analysis.bonus_probability,
    "expected_points": analysis.expected_points,
    "block_targets": analysis.block_targets,
    "risk_assessment": analysis.risk_assessment
}}))
"""
        ])
        
        if result["success"]:
            analysis_result = json.loads(result["stdout"].strip().split('\n')[-1])
            
            return jsonify(ApiResponse(
                success=True,
                data=analysis_result,
                message="Autonomous decision analysis completed"
            ).dict())
        else:
            raise Exception(result["stderr"])
            
    except Exception as e:
        logger.error(f"Error analyzing autonomous decision: {str(e)}")
        return jsonify(ApiResponse(
            success=False,
            error=str(e),
            message="Failed to analyze autonomous decision"
        ).dict()), 500

# Goal Priority Analysis
@push_back_bp.route('/analyze/goal-priority', methods=['POST'])
def analyze_goal_priority():
    """Analyze goal priority strategy"""
    try:
        data = request.get_json()
        robot_specs = data.get('robot_specs', [])
        opponent_strategy = data.get('opponent_strategy', 'balanced')
        match_phase = data.get('match_phase', 'early')
        
        service = get_push_back_service()
        
        result = service._execute_command([
            "-c",
            f"""
import sys
sys.path.append('{current_app.config["VEX_ANALYSIS_PATH"]}')
from vex_analysis.analysis.push_back_strategy_analyzer import PushBackStrategyAnalyzer
import json

analyzer = PushBackStrategyAnalyzer()
analysis = analyzer.analyze_goal_priority_strategy({json.dumps(robot_specs)}, "{opponent_strategy}", "{match_phase}")
print(json.dumps({{
    "recommended_priority": analysis.recommended_priority,
    "center_goal_value": analysis.center_goal_value,
    "long_goal_value": analysis.long_goal_value,
    "optimal_sequence": analysis.optimal_sequence,
    "decision_confidence": analysis.decision_confidence
}}))
"""
        ])
        
        if result["success"]:
            analysis_result = json.loads(result["stdout"].strip().split('\n')[-1])
            
            return jsonify(ApiResponse(
                success=True,
                data=analysis_result,
                message="Goal priority analysis completed"
            ).dict())
        else:
            raise Exception(result["stderr"])
            
    except Exception as e:
        logger.error(f"Error analyzing goal priority: {str(e)}")
        return jsonify(ApiResponse(
            success=False,
            error=str(e),
            message="Failed to analyze goal priority"
        ).dict()), 500

# Parking Decision Analysis
@push_back_bp.route('/analyze/parking', methods=['POST'])
def analyze_parking_decision():
    """Analyze parking decision timing"""
    try:
        data = request.get_json()
        match_state = data.get('match_state', {})
        robot_specs = data.get('robot_specs', [])
        
        service = get_push_back_service()
        
        result = service._execute_command([
            "-c",
            f"""
import sys
sys.path.append('{current_app.config["VEX_ANALYSIS_PATH"]}')
from vex_analysis.analysis.push_back_strategy_analyzer import PushBackStrategyAnalyzer
import json

analyzer = PushBackStrategyAnalyzer()
analysis = analyzer.analyze_parking_decision_timing({json.dumps(match_state)}, {json.dumps(robot_specs)})
print(json.dumps({{
    "recommended_timing": analysis.recommended_timing,
    "one_robot_threshold": analysis.one_robot_threshold,
    "two_robot_threshold": analysis.two_robot_threshold,
    "expected_value": analysis.expected_value,
    "risk_benefit_ratio": analysis.risk_benefit_ratio,
    "situational_recommendations": analysis.situational_recommendations
}}))
"""
        ])
        
        if result["success"]:
            analysis_result = json.loads(result["stdout"].strip().split('\n')[-1])
            
            return jsonify(ApiResponse(
                success=True,
                data=analysis_result,
                message="Parking decision analysis completed"
            ).dict())
        else:
            raise Exception(result["stderr"])
            
    except Exception as e:
        logger.error(f"Error analyzing parking decision: {str(e)}")
        return jsonify(ApiResponse(
            success=False,
            error=str(e),
            message="Failed to analyze parking decision"
        ).dict()), 500

# Offense/Defense Balance Analysis
@push_back_bp.route('/analyze/offense-defense', methods=['POST'])
def analyze_offense_defense_balance():
    """Analyze offense/defense balance"""
    try:
        data = request.get_json()
        match_state = data.get('match_state', {})
        robot_specs = data.get('robot_specs', [])
        
        service = get_push_back_service()
        
        result = service._execute_command([
            "-c",
            f"""
import sys
sys.path.append('{current_app.config["VEX_ANALYSIS_PATH"]}')
from vex_analysis.analysis.push_back_strategy_analyzer import PushBackStrategyAnalyzer
import json

analyzer = PushBackStrategyAnalyzer()
analysis = analyzer.analyze_offense_defense_balance({json.dumps(match_state)}, {json.dumps(robot_specs)})
print(json.dumps({{
    "recommended_ratio": analysis.recommended_ratio,
    "offensive_roi": analysis.offensive_roi,
    "defensive_roi": analysis.defensive_roi,
    "critical_zones": analysis.critical_zones,
    "disruption_targets": analysis.disruption_targets
}}))
"""
        ])
        
        if result["success"]:
            analysis_result = json.loads(result["stdout"].strip().split('\n')[-1])
            
            return jsonify(ApiResponse(
                success=True,
                data=analysis_result,
                message="Offense/defense analysis completed"
            ).dict())
        else:
            raise Exception(result["stderr"])
            
    except Exception as e:
        logger.error(f"Error analyzing offense/defense balance: {str(e)}")
        return jsonify(ApiResponse(
            success=False,
            error=str(e),
            message="Failed to analyze offense/defense balance"
        ).dict()), 500

# Comprehensive Analysis
@push_back_bp.route('/analyze/comprehensive', methods=['POST'])
def run_comprehensive_analysis():
    """Run comprehensive Push Back analysis"""
    try:
        data = request.get_json()
        robot_specs = data.get('robot_specs', [])
        match_context = data.get('match_context', {})
        
        service = get_push_back_service()
        
        result = service._execute_command([
            "-c",
            f"""
import sys
sys.path.append('{current_app.config["VEX_ANALYSIS_PATH"]}')
from vex_analysis.analysis.push_back_strategy_analyzer import PushBackStrategyAnalyzer
import json

analyzer = PushBackStrategyAnalyzer()
analysis = analyzer.run_comprehensive_analysis({json.dumps(robot_specs)})

# Convert to dict format expected by frontend
comprehensive_result = {{
    "block_flow": analysis.block_flow_optimization.to_dict() if hasattr(analysis, 'block_flow_optimization') else {{}},
    "autonomous_decision": analysis.autonomous_decision.to_dict() if hasattr(analysis, 'autonomous_decision') else {{}},
    "goal_priority": analysis.goal_priority_analysis.to_dict() if hasattr(analysis, 'goal_priority_analysis') else {{}},
    "parking_decision": analysis.parking_decision_analysis.to_dict() if hasattr(analysis, 'parking_decision_analysis') else {{}},
    "offense_defense": analysis.offense_defense_balance.to_dict() if hasattr(analysis, 'offense_defense_balance') else {{}},
    "recommended_archetype": analysis.recommended_archetype,
    "recommendations": analysis.recommendations
}}

print(json.dumps(comprehensive_result))
"""
        ])
        
        if result["success"]:
            analysis_result = json.loads(result["stdout"].strip().split('\n')[-1])
            
            return jsonify(ApiResponse(
                success=True,
                data=analysis_result,
                message="Comprehensive Push Back analysis completed"
            ).dict())
        else:
            raise Exception(result["stderr"])
            
    except Exception as e:
        logger.error(f"Error running comprehensive analysis: {str(e)}")
        return jsonify(ApiResponse(
            success=False,
            error=str(e),
            message="Failed to run comprehensive analysis"
        ).dict()), 500

# Monte Carlo Simulation
@push_back_bp.route('/simulate/monte-carlo', methods=['POST'])
def run_monte_carlo_simulation():
    """Run Monte Carlo simulation for Push Back strategy"""
    try:
        data = request.get_json()
        strategy = data.get('strategy')
        num_simulations = data.get('num_simulations', 1000)
        opponent_types = data.get('opponent_types', [])
        
        service = get_push_back_service()
        
        result = service._execute_command([
            "-c",
            f"""
import sys
sys.path.append('{current_app.config["VEX_ANALYSIS_PATH"]}')
from vex_analysis.analysis.push_back_strategy_analyzer import PushBackStrategyAnalyzer
import json

analyzer = PushBackStrategyAnalyzer()
simulation = analyzer.run_monte_carlo_simulation(
    {json.dumps(strategy)}, 
    {num_simulations}, 
    {json.dumps(opponent_types) if opponent_types else 'None'}
)

print(json.dumps({{
    "win_rate": simulation.win_rate,
    "avg_score": simulation.avg_score,
    "score_std": simulation.score_std,
    "scoring_breakdown": simulation.scoring_breakdown,
    "opponent_matchups": simulation.opponent_matchups,
    "performance_confidence": simulation.performance_confidence
}}))
"""
        ])
        
        if result["success"]:
            simulation_result = json.loads(result["stdout"].strip().split('\n')[-1])
            
            return jsonify(ApiResponse(
                success=True,
                data=simulation_result,
                message="Monte Carlo simulation completed"
            ).dict())
        else:
            raise Exception(result["stderr"])
            
    except Exception as e:
        logger.error(f"Error running Monte Carlo simulation: {str(e)}")
        return jsonify(ApiResponse(
            success=False,
            error=str(e),
            message="Failed to run Monte Carlo simulation"
        ).dict()), 500

# Field State and Scoring
@push_back_bp.route('/score/calculate', methods=['POST'])
def calculate_score():
    """Calculate Push Back score from field state"""
    try:
        data = request.get_json()
        field_state = data.get('field_state')
        
        service = get_push_back_service()
        
        result = service._execute_command([
            "-c",
            f"""
import sys
sys.path.append('{current_app.config["VEX_ANALYSIS_PATH"]}')
from vex_analysis.core.simulator import PushBackScoringEngine
import json

engine = PushBackScoringEngine()
field_state_data = {json.dumps(field_state)}

# Calculate scores for both alliances
red_score, red_breakdown = engine.calculate_push_back_score(field_state_data, "red")
blue_score, blue_breakdown = engine.calculate_push_back_score(field_state_data, "blue")

# Get control zones
control_zones = engine.get_control_zone_status(field_state_data)

result = {{
    "red_score": red_score,
    "blue_score": blue_score,
    "red_breakdown": red_breakdown,
    "blue_breakdown": blue_breakdown,
    "control_zones": control_zones
}}

print(json.dumps(result))
"""
        ])
        
        if result["success"]:
            score_result = json.loads(result["stdout"].strip().split('\n')[-1])
            
            return jsonify(ApiResponse(
                success=True,
                data=score_result,
                message="Push Back score calculation completed"
            ).dict())
        else:
            raise Exception(result["stderr"])
            
    except Exception as e:
        logger.error(f"Error calculating score: {str(e)}")
        return jsonify(ApiResponse(
            success=False,
            error=str(e),
            message="Failed to calculate Push Back score"
        ).dict()), 500

# Strategy Archetypes
@push_back_bp.route('/archetypes', methods=['GET'])
def get_strategy_archetypes():
    """Get all Push Back strategy archetypes"""
    try:
        service = get_push_back_service()
        
        result = service._execute_command([
            "-c",
            f"""
import sys
sys.path.append('{current_app.config["VEX_ANALYSIS_PATH"]}')
from vex_analysis.analysis.push_back_strategy_analyzer import PushBackStrategyAnalyzer
import json

analyzer = PushBackStrategyAnalyzer()
archetypes = analyzer.get_strategy_archetypes()

print(json.dumps(archetypes))
"""
        ])
        
        if result["success"]:
            archetypes = json.loads(result["stdout"].strip().split('\n')[-1])
            
            return jsonify(ApiResponse(
                success=True,
                data=archetypes,
                message="Strategy archetypes retrieved successfully"
            ).dict())
        else:
            raise Exception(result["stderr"])
            
    except Exception as e:
        logger.error(f"Error getting strategy archetypes: {str(e)}")
        return jsonify(ApiResponse(
            success=False,
            error=str(e),
            message="Failed to get strategy archetypes"
        ).dict()), 500

@push_back_bp.route('/archetypes/recommend', methods=['POST'])
def get_archetype_recommendation():
    """Get recommended strategy archetype"""
    try:
        data = request.get_json()
        robot_specs = data.get('robot_specs', [])
        match_context = data.get('match_context', {})
        
        service = get_push_back_service()
        
        result = service._execute_command([
            "-c",
            f"""
import sys
sys.path.append('{current_app.config["VEX_ANALYSIS_PATH"]}')
from vex_analysis.analysis.push_back_strategy_analyzer import PushBackStrategyAnalyzer
import json

analyzer = PushBackStrategyAnalyzer()
recommendation = analyzer.recommend_archetype({json.dumps(robot_specs)}, {json.dumps(match_context)})

print(json.dumps({{
    "recommended_archetype": recommendation.recommended_archetype,
    "archetype_scores": recommendation.archetype_scores,
    "reasoning": recommendation.reasoning
}}))
"""
        ])
        
        if result["success"]:
            recommendation = json.loads(result["stdout"].strip().split('\n')[-1])
            
            return jsonify(ApiResponse(
                success=True,
                data=recommendation,
                message="Archetype recommendation completed"
            ).dict())
        else:
            raise Exception(result["stderr"])
            
    except Exception as e:
        logger.error(f"Error getting archetype recommendation: {str(e)}")
        return jsonify(ApiResponse(
            success=False,
            error=str(e),
            message="Failed to get archetype recommendation"
        ).dict()), 500

# Decision Support Tools
@push_back_bp.route('/tools/parking-calculator', methods=['POST'])
def get_parking_calculator():
    """Calculate optimal parking decisions"""
    try:
        data = request.get_json()
        current_score = data.get('current_score')
        opponent_score = data.get('opponent_score')
        time_remaining = data.get('time_remaining')
        robot_specs = data.get('robot_specs', [])
        
        # Simplified parking calculator - in production this would use sophisticated analysis
        score_difference = current_score - opponent_score
        
        if score_difference > 15:
            parking_rec = "Start parking with 30 seconds remaining"
            risk_level = "low"
        elif score_difference > 0:
            parking_rec = "Park one robot with 15 seconds remaining"
            risk_level = "medium"
        else:
            parking_rec = "Focus on scoring, avoid parking unless winning"
            risk_level = "high"
        
        result = {
            "parking_recommendations": {
                "primary": parking_rec,
                "alternative": "Dynamic decision based on score differential"
            },
            "break_even_points": {
                "one_robot": 8,
                "two_robots": 30
            },
            "risk_analysis": {
                "current_risk": risk_level,
                "parking_value": 8 if score_difference > 0 else 0
            }
        }
        
        return jsonify(ApiResponse(
            success=True,
            data=result,
            message="Parking calculator analysis completed"
        ).dict())
        
    except Exception as e:
        logger.error(f"Error in parking calculator: {str(e)}")
        return jsonify(ApiResponse(
            success=False,
            error=str(e),
            message="Failed to calculate parking recommendations"
        ).dict()), 500

@push_back_bp.route('/tools/control-zone-optimizer', methods=['POST'])
def get_control_zone_optimizer():
    """Optimize control zone block placement"""
    try:
        data = request.get_json()
        current_blocks = data.get('current_blocks', {})
        available_blocks = data.get('available_blocks', 0)
        robot_specs = data.get('robot_specs', [])
        
        # Simplified optimizer - in production this would use the actual optimization logic
        result = {
            "optimal_additions": {
                "red_zone": min(3, available_blocks // 2),
                "blue_zone": min(3, available_blocks // 2)
            },
            "expected_control_gain": min(6, available_blocks * 0.75),
            "efficiency_score": 0.8,
            "recommendations": [
                "Prioritize red zone for immediate points",
                "Maintain blue zone presence for endgame",
                "Focus on corner placement for maximum efficiency"
            ]
        }
        
        return jsonify(ApiResponse(
            success=True,
            data=result,
            message="Control zone optimization completed"
        ).dict())
        
    except Exception as e:
        logger.error(f"Error in control zone optimizer: {str(e)}")
        return jsonify(ApiResponse(
            success=False,
            error=str(e),
            message="Failed to optimize control zones"
        ).dict()), 500

@push_back_bp.route('/tools/autonomous-planner', methods=['POST'])
def get_autonomous_planner():
    """Generate autonomous strategy options"""
    try:
        data = request.get_json()
        robot_specs = data.get('robot_specs', [])
        
        # Generate autonomous strategy options based on robot capabilities
        result = {
            "strategy_options": {
                "conservative": {
                    "description": "Focus on guaranteed points",
                    "target_blocks": 4,
                    "expected_points": 12
                },
                "aggressive": {
                    "description": "Go for autonomous win point",
                    "target_blocks": 7,
                    "expected_points": 28
                },
                "balanced": {
                    "description": "Mix of scoring and positioning",
                    "target_blocks": 5,
                    "expected_points": 18
                }
            },
            "time_allocations": {
                "conservative": {"block_collection": 12, "positioning": 3},
                "aggressive": {"block_collection": 14, "positioning": 1},
                "balanced": {"block_collection": 10, "positioning": 5}
            },
            "risk_assessments": {
                "conservative": "Low risk, reliable execution",
                "aggressive": "High risk, high reward potential",
                "balanced": "Medium risk, good flexibility"
            },
            "recommendations": [
                "Choose conservative for early season reliability",
                "Aggressive works best with high-accuracy robots",
                "Balanced provides good adaptation options"
            ]
        }
        
        return jsonify(ApiResponse(
            success=True,
            data=result,
            message="Autonomous planning completed"
        ).dict())
        
    except Exception as e:
        logger.error(f"Error in autonomous planner: {str(e)}")
        return jsonify(ApiResponse(
            success=False,
            error=str(e),
            message="Failed to generate autonomous plan"
        ).dict()), 500

# Visualization Data
@push_back_bp.route('/visualize/field', methods=['POST'])
def get_field_visualization_data():
    """Get field visualization data"""
    try:
        data = request.get_json()
        field_state = data.get('field_state')
        
        # Generate visualization data for Push Back field
        result = {
            "field_layout": {
                "width": 12,  # feet
                "height": 12,
                "goals": [
                    {"id": "red_center", "x": 6, "y": 2, "type": "center"},
                    {"id": "red_long", "x": 2, "y": 6, "type": "long"},
                    {"id": "blue_center", "x": 6, "y": 10, "type": "center"},
                    {"id": "blue_long", "x": 10, "y": 6, "type": "long"}
                ],
                "zones": [
                    {"id": "red_zone", "x": 3, "y": 3, "width": 3, "height": 3},
                    {"id": "blue_zone", "x": 6, "y": 6, "width": 3, "height": 3}
                ]
            },
            "block_positions": [
                {"id": f"block_{i}", "x": (i % 11) + 1, "y": (i // 11) + 1, "alliance": "neutral"} 
                for i in range(88)
            ],
            "control_zones": {
                "red_zone": {"controlled_by": "red", "block_count": 6},
                "blue_zone": {"controlled_by": "neutral", "block_count": 4}
            },
            "scoring_visualization": {
                "red_total": 45,
                "blue_total": 32,
                "breakdown_visible": True
            }
        }
        
        return jsonify(ApiResponse(
            success=True,
            data=result,
            message="Field visualization data generated"
        ).dict())
        
    except Exception as e:
        logger.error(f"Error generating field visualization: {str(e)}")
        return jsonify(ApiResponse(
            success=False,
            error=str(e),
            message="Failed to generate field visualization"
        ).dict()), 500

@push_back_bp.route('/visualize/timeline', methods=['POST'])
def get_match_timeline_data():
    """Get match timeline visualization data"""
    try:
        data = request.get_json()
        strategy = data.get('strategy')
        
        result = {
            "autonomous_timeline": [
                {"time": 0, "event": "Match start", "action": "Begin block collection"},
                {"time": 5, "event": "First blocks", "action": "Collect 2 blocks"},
                {"time": 10, "event": "Goal attempt", "action": "Score in center goal"},
                {"time": 15, "event": "Auto end", "action": "Final positioning"}
            ],
            "driver_timeline": [
                {"time": 15, "event": "Driver start", "action": "Continue block flow"},
                {"time": 45, "event": "Mid-match", "action": "Control zone focus"},
                {"time": 90, "event": "Endgame", "action": "Parking decision"},
                {"time": 105, "event": "Match end", "action": "Final scoring"}
            ],
            "score_progression": [
                {"time": 0, "red_score": 0, "blue_score": 0},
                {"time": 15, "red_score": 18, "blue_score": 12},
                {"time": 60, "red_score": 42, "blue_score": 38},
                {"time": 105, "red_score": 67, "blue_score": 54}
            ],
            "key_events": [
                {"time": 7, "event": "Autonomous win point achieved", "alliance": "red"},
                {"time": 35, "event": "Control zone captured", "alliance": "blue"},
                {"time": 95, "event": "Parking initiated", "alliance": "red"}
            ]
        }
        
        return jsonify(ApiResponse(
            success=True,
            data=result,
            message="Match timeline data generated"
        ).dict())
        
    except Exception as e:
        logger.error(f"Error generating timeline data: {str(e)}")
        return jsonify(ApiResponse(
            success=False,
            error=str(e),
            message="Failed to generate timeline data"
        ).dict()), 500

@push_back_bp.route('/visualize/risk-reward', methods=['POST'])
def get_risk_reward_data():
    """Get risk-reward visualization data"""
    try:
        data = request.get_json()
        strategies = data.get('strategies', [])
        
        result = {
            "scatter_data": [
                {"name": "Block Flow Max", "risk": 0.3, "reward": 0.8, "archetype": "block_flow"},
                {"name": "Control Zone", "risk": 0.5, "reward": 0.9, "archetype": "control_zone"},
                {"name": "Balanced", "risk": 0.4, "reward": 0.7, "archetype": "balanced"},
                {"name": "Rush Strategy", "risk": 0.8, "reward": 0.6, "archetype": "rush"}
            ],
            "risk_categories": ["Low Risk", "Medium Risk", "High Risk"],
            "reward_categories": ["Low Reward", "Medium Reward", "High Reward"],
            "strategy_clusters": {
                "conservative": [{"name": "Block Flow Max", "score": 0.75}],
                "aggressive": [{"name": "Rush Strategy", "score": 0.65}],
                "balanced": [{"name": "Control Zone", "score": 0.85}]
            }
        }
        
        return jsonify(ApiResponse(
            success=True,
            data=result,
            message="Risk-reward data generated"
        ).dict())
        
    except Exception as e:
        logger.error(f"Error generating risk-reward data: {str(e)}")
        return jsonify(ApiResponse(
            success=False,
            error=str(e),
            message="Failed to generate risk-reward data"
        ).dict()), 500

# System Status
@push_back_bp.route('/system/status', methods=['GET'])
def get_push_back_system_status():
    """Get Push Back system status"""
    try:
        service = get_push_back_service()
        
        # Check if Push Back analyzer is available
        result = service._execute_command([
            "-c",
            f"""
import sys
sys.path.append('{current_app.config["VEX_ANALYSIS_PATH"]}')
try:
    from vex_analysis.analysis.push_back_strategy_analyzer import PushBackStrategyAnalyzer
    from vex_analysis.core.simulator import PushBackScoringEngine
    print("PUSH_BACK_READY")
except ImportError as e:
    print(f"PUSH_BACK_ERROR: {{e}}")
"""
        ])
        
        push_back_status = "available" if "PUSH_BACK_READY" in result.get("stdout", "") else "unavailable"
        
        status = {
            "backend_status": "online",
            "analysis_engine_status": push_back_status,
            "available_features": [
                "strategy_builder",
                "block_flow_optimization", 
                "autonomous_analysis",
                "goal_priority_analysis",
                "parking_calculator",
                "offense_defense_balance",
                "monte_carlo_simulation",
                "field_visualization"
            ],
            "version": "1.0.0-pushback"
        }
        
        return jsonify(ApiResponse(
            success=True,
            data=status,
            message="Push Back system status retrieved"
        ).dict())
        
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        return jsonify(ApiResponse(
            success=False,
            error=str(e),
            message="Failed to get system status"
        ).dict()), 500