"""
Visualization API routes for generating charts and dashboards
"""

from flask import Blueprint, request, jsonify, current_app
from app.models.visualization import VisualizationRequest, VisualizationResponse, VisualizationSet, ChartData, ChartType
from app.models.base import SuccessResponse, ErrorResponse
from app.services.vex_analysis_service import VEXAnalysisService
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)

visualization_bp = Blueprint('visualization', __name__)

def get_vex_service() -> VEXAnalysisService:
    """Get VEX analysis service instance"""
    return VEXAnalysisService(
        vex_path=current_app.config['VEX_ANALYSIS_PATH'],
        python_path=current_app.config['PYTHON_PATH']
    )

@visualization_bp.route('/generate', methods=['POST'])
def generate_visualizations():
    """
    Generate visualizations from analysis data
    
    POST /api/visualizations/generate
    
    Body:
    {
        "data_source": "analysis_results",
        "analysis_id": "analysis-123",
        "chart_types": ["line", "bar", "pie"],
        "create_dashboard": true,
        "dashboard_name": "VEX U Performance Dashboard",
        "interactive": true
    }
    """
    try:
        # Parse request data
        data = request.get_json() or {}
        
        # Validate request
        viz_request = VisualizationRequest(
            data_source=data.get('data_source', 'analysis_results'),
            analysis_id=data.get('analysis_id'),
            chart_types=data.get('chart_types', [ChartType.LINE, ChartType.BAR]),
            create_dashboard=data.get('create_dashboard', False),
            dashboard_name=data.get('dashboard_name'),
            interactive=data.get('interactive', True),
            filters=data.get('filters'),
            group_by=data.get('group_by')
        )
        
        # For demonstration, create mock analysis data
        # In a real implementation, this would come from the analysis_id
        mock_analysis_data = {
            "strategies": [
                {"name": "Strategy A", "score": 85, "efficiency": 0.78},
                {"name": "Strategy B", "score": 92, "efficiency": 0.82},
                {"name": "Strategy C", "score": 77, "efficiency": 0.75},
            ],
            "performance_metrics": {
                "average_score": 84.7,
                "win_rate": 0.73,
                "consistency": 0.85
            }
        }
        
        # Generate visualizations using VEX service
        vex_service = get_vex_service()
        viz_result = vex_service.generate_visualization(
            analysis_data=mock_analysis_data,
            chart_types=[ct.value for ct in viz_request.chart_types]
        )
        
        # Create chart data structures
        charts = []
        
        # Create a performance comparison chart
        if ChartType.BAR in viz_request.chart_types:
            charts.append(ChartData(
                chart_id=str(uuid.uuid4()),
                chart_type=ChartType.BAR,
                title="Strategy Performance Comparison",
                data=[
                    {"strategy": "Strategy A", "score": 85},
                    {"strategy": "Strategy B", "score": 92},
                    {"strategy": "Strategy C", "score": 77}
                ],
                x_axis="strategy",
                y_axis="score",
                options={"colors": ["#1976d2", "#dc004e", "#388e3c"]}
            ))
        
        # Create an efficiency trend chart
        if ChartType.LINE in viz_request.chart_types:
            charts.append(ChartData(
                chart_id=str(uuid.uuid4()),
                chart_type=ChartType.LINE,
                title="Efficiency Trend Over Time",
                data=[
                    {"time": "Week 1", "efficiency": 0.75},
                    {"time": "Week 2", "efficiency": 0.78},
                    {"time": "Week 3", "efficiency": 0.82},
                    {"time": "Week 4", "efficiency": 0.85}
                ],
                x_axis="time",
                y_axis="efficiency",
                options={"smooth": True}
            ))
        
        # Create a score distribution pie chart
        if ChartType.PIE in viz_request.chart_types:
            charts.append(ChartData(
                chart_id=str(uuid.uuid4()),
                chart_type=ChartType.PIE,
                title="Score Distribution",
                data=[
                    {"category": "High Score (80+)", "value": 2},
                    {"category": "Medium Score (60-79)", "value": 1},
                    {"category": "Low Score (<60)", "value": 0}
                ],
                options={"labels": True}
            ))
        
        # Create dashboard if requested
        dashboard = None
        if viz_request.create_dashboard:
            from app.models.visualization import DashboardData, DashboardWidget
            
            widgets = []
            for i, chart in enumerate(charts):
                widget = DashboardWidget(
                    widget_id=str(uuid.uuid4()),
                    widget_type="chart",
                    title=chart.title,
                    position={"x": (i % 2) * 6, "y": (i // 2) * 4, "width": 6, "height": 4},
                    chart=chart
                )
                widgets.append(widget)
            
            dashboard = DashboardData(
                dashboard_id=str(uuid.uuid4()),
                name=viz_request.dashboard_name or "VEX U Analysis Dashboard",
                widgets=widgets
            )
        
        # Create visualization set
        viz_set = VisualizationSet(
            visualization_id=str(uuid.uuid4()),
            name=f"Visualization for {viz_request.data_source}",
            data_source=viz_request.data_source,
            charts=charts,
            dashboard=dashboard,
            generation_params=viz_request
        )
        
        # Create response
        response_data = VisualizationResponse(visualization_set=viz_set)
        response = SuccessResponse(
            message="Visualizations generated successfully",
            data=response_data.dict()
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error(f"Visualization generation failed: {str(e)}")
        error_response = ErrorResponse(
            message="Visualization generation failed",
            error_code="VISUALIZATION_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500

@visualization_bp.route('/dashboard', methods=['GET'])
def get_interactive_dashboard():
    """
    Get interactive dashboard data
    
    GET /api/visualizations/dashboard?analysis_id=123&real_time=true
    """
    try:
        analysis_id = request.args.get('analysis_id')
        real_time = request.args.get('real_time', 'false').lower() == 'true'
        
        # Mock dashboard data
        from app.models.visualization import DashboardData, DashboardWidget, ChartData
        
        # Create sample charts
        performance_chart = ChartData(
            chart_id=str(uuid.uuid4()),
            chart_type=ChartType.LINE,
            title="Real-time Performance",
            data=[
                {"time": "10:00", "score": 85},
                {"time": "10:15", "score": 88},
                {"time": "10:30", "score": 92},
                {"time": "10:45", "score": 87}
            ],
            x_axis="time",
            y_axis="score"
        )
        
        efficiency_chart = ChartData(
            chart_id=str(uuid.uuid4()),
            chart_type=ChartType.BAR,
            title="Strategy Efficiency",
            data=[
                {"strategy": "Offensive", "efficiency": 0.85},
                {"strategy": "Defensive", "efficiency": 0.78},
                {"strategy": "Balanced", "efficiency": 0.82}
            ],
            x_axis="strategy",
            y_axis="efficiency"
        )
        
        # Create widgets
        widgets = [
            DashboardWidget(
                widget_id=str(uuid.uuid4()),
                widget_type="chart",
                title="Performance Trend",
                position={"x": 0, "y": 0, "width": 6, "height": 4},
                chart=performance_chart,
                refresh_interval=30 if real_time else None
            ),
            DashboardWidget(
                widget_id=str(uuid.uuid4()),
                widget_type="chart",
                title="Strategy Comparison",
                position={"x": 6, "y": 0, "width": 6, "height": 4},
                chart=efficiency_chart
            ),
            DashboardWidget(
                widget_id=str(uuid.uuid4()),
                widget_type="metric",
                title="Average Score",
                position={"x": 0, "y": 4, "width": 3, "height": 2},
                metric_value=88.5
            ),
            DashboardWidget(
                widget_id=str(uuid.uuid4()),
                widget_type="metric",
                title="Win Rate",
                position={"x": 3, "y": 4, "width": 3, "height": 2},
                metric_value="78%"
            )
        ]
        
        dashboard = DashboardData(
            dashboard_id=str(uuid.uuid4()),
            name="VEX U Live Dashboard",
            description="Real-time analysis dashboard",
            widgets=widgets,
            auto_refresh=real_time,
            refresh_interval=30 if real_time else 300
        )
        
        response = SuccessResponse(
            message="Dashboard data retrieved successfully",
            data=dashboard.dict()
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error(f"Dashboard retrieval failed: {str(e)}")
        error_response = ErrorResponse(
            message="Dashboard retrieval failed",
            error_code="DASHBOARD_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500

@visualization_bp.route('/data', methods=['GET'])
def get_visualization_data():
    """
    Get visualization data for specific type and parameters
    
    GET /api/visualizations/data?type=performance&strategy_id=123&period=7d
    """
    try:
        viz_type = request.args.get('type', 'performance')
        strategy_id = request.args.get('strategy_id')
        period = request.args.get('period', '7d')
        
        # Mock data based on type
        if viz_type == 'performance':
            data = {
                "chart_type": "line",
                "data": [
                    {"date": "2024-01-01", "score": 82},
                    {"date": "2024-01-02", "score": 85},
                    {"date": "2024-01-03", "score": 88},
                    {"date": "2024-01-04", "score": 87},
                    {"date": "2024-01-05", "score": 91}
                ]
            }
        elif viz_type == 'strategy_comparison':
            data = {
                "chart_type": "bar",
                "data": [
                    {"strategy": "Strategy A", "score": 85, "efficiency": 0.78},
                    {"strategy": "Strategy B", "score": 92, "efficiency": 0.82},
                    {"strategy": "Strategy C", "score": 77, "efficiency": 0.75}
                ]
            }
        else:
            data = {"message": f"No data available for type: {viz_type}"}
        
        response = SuccessResponse(
            message="Visualization data retrieved successfully",
            data=data
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error(f"Visualization data retrieval failed: {str(e)}")
        error_response = ErrorResponse(
            message="Visualization data retrieval failed",
            error_code="DATA_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500

@visualization_bp.route('/', methods=['GET'])
def list_visualizations():
    """
    List all available visualizations
    
    GET /api/visualizations/?limit=20&offset=0
    """
    try:
        limit = int(request.args.get('limit', 20))
        offset = int(request.args.get('offset', 0))
        
        # Mock visualization list
        visualizations = []
        
        response = SuccessResponse(
            message="Visualizations retrieved successfully",
            data={
                "visualizations": visualizations,
                "total": 0,
                "limit": limit,
                "offset": offset
            }
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error(f"Visualization listing failed: {str(e)}")
        error_response = ErrorResponse(
            message="Failed to list visualizations",
            error_code="LISTING_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500