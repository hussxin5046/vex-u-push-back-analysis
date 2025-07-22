"""
Visualization models for charts, dashboards, and interactive displays
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum

class ChartType(str, Enum):
    """Types of charts for visualization"""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    HEATMAP = "heatmap"
    RADAR = "radar"
    TREE_MAP = "tree_map"

class ColorScheme(str, Enum):
    """Color schemes for charts"""
    DEFAULT = "default"
    VEX_BLUE_RED = "vex_blue_red"
    PERFORMANCE = "performance"
    CATEGORICAL = "categorical"
    SEQUENTIAL = "sequential"
    DIVERGING = "diverging"

class ChartData(BaseModel):
    """Data structure for individual charts"""
    chart_id: str = Field(description="Unique chart identifier")
    chart_type: ChartType = Field(description="Type of chart")
    title: str = Field(description="Chart title")
    subtitle: Optional[str] = Field(None, description="Chart subtitle")
    
    # Data
    data: List[Dict[str, Any]] = Field(description="Chart data points")
    x_axis: Optional[str] = Field(None, description="X-axis field name")
    y_axis: Optional[str] = Field(None, description="Y-axis field name")
    
    # Styling
    color_scheme: ColorScheme = Field(ColorScheme.DEFAULT, description="Color scheme")
    width: Optional[int] = Field(None, description="Chart width in pixels")
    height: Optional[int] = Field(None, description="Chart height in pixels")
    
    # Configuration
    options: Dict[str, Any] = Field(default_factory=dict, description="Chart-specific options")
    interactive: bool = Field(True, description="Whether chart is interactive")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Chart creation time")
    data_source: Optional[str] = Field(None, description="Source of chart data")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class DashboardWidget(BaseModel):
    """Individual widget on a dashboard"""
    widget_id: str = Field(description="Unique widget identifier")
    widget_type: str = Field(description="Type of widget (chart/metric/text)")
    title: str = Field(description="Widget title")
    
    # Layout
    position: Dict[str, int] = Field(description="Widget position (x, y, width, height)")
    
    # Content
    chart: Optional[ChartData] = Field(None, description="Chart data if chart widget")
    metric_value: Optional[Union[int, float, str]] = Field(None, description="Metric value if metric widget")
    text_content: Optional[str] = Field(None, description="Text content if text widget")
    
    # Configuration
    refresh_interval: Optional[int] = Field(None, description="Auto-refresh interval in seconds")
    clickable: bool = Field(False, description="Whether widget is clickable")
    
class DashboardData(BaseModel):
    """Complete dashboard configuration"""
    dashboard_id: str = Field(description="Unique dashboard identifier")
    name: str = Field(description="Dashboard name")
    description: Optional[str] = Field(None, description="Dashboard description")
    
    # Layout
    layout: str = Field("grid", description="Dashboard layout type")
    grid_size: Dict[str, int] = Field({"cols": 12, "rows": 8}, description="Grid dimensions")
    
    # Widgets
    widgets: List[DashboardWidget] = Field(description="Dashboard widgets")
    
    # Configuration
    auto_refresh: bool = Field(True, description="Auto-refresh dashboard")
    refresh_interval: int = Field(30, description="Refresh interval in seconds")
    theme: str = Field("light", description="Dashboard theme")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Dashboard creation time")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class VisualizationRequest(BaseModel):
    """Request model for visualization generation"""
    
    # Data source
    data_source: str = Field(description="Source of data for visualization")
    analysis_id: Optional[str] = Field(None, description="Analysis ID if visualizing analysis results")
    
    # Chart specifications
    chart_types: List[ChartType] = Field(description="Types of charts to generate")
    focus_metrics: Optional[List[str]] = Field(None, description="Metrics to focus on")
    
    # Dashboard options
    create_dashboard: bool = Field(False, description="Whether to create an interactive dashboard")
    dashboard_name: Optional[str] = Field(None, description="Dashboard name if creating")
    
    # Styling and options
    color_scheme: ColorScheme = Field(ColorScheme.VEX_BLUE_RED, description="Color scheme")
    interactive: bool = Field(True, description="Generate interactive charts")
    
    # Filtering and grouping
    filters: Optional[Dict[str, Any]] = Field(None, description="Data filters")
    group_by: Optional[List[str]] = Field(None, description="Fields to group data by")
    
    # Output options
    export_formats: Optional[List[str]] = Field(None, description="Export formats (png, svg, pdf)")
    include_raw_data: bool = Field(False, description="Include raw data in response")

class VisualizationSet(BaseModel):
    """Set of related visualizations"""
    visualization_id: str = Field(description="Unique visualization set identifier")
    name: str = Field(description="Visualization set name")
    data_source: str = Field(description="Source of visualization data")
    
    # Charts and dashboard
    charts: List[ChartData] = Field(description="Generated charts")
    dashboard: Optional[DashboardData] = Field(None, description="Dashboard if created")
    
    # Export data
    export_urls: Optional[Dict[str, str]] = Field(None, description="URLs for exported files")
    raw_data: Optional[Dict[str, Any]] = Field(None, description="Raw data used")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    generation_params: VisualizationRequest = Field(description="Parameters used for generation")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class VisualizationResponse(BaseModel):
    """Response model for visualization operations"""
    visualization_set: VisualizationSet = Field(description="Generated visualizations")
    task_id: Optional[str] = Field(None, description="Background task ID if async")

class InteractiveVisualization(BaseModel):
    """Interactive visualization with real-time capabilities"""
    visualization_id: str = Field(description="Visualization identifier")
    websocket_endpoint: str = Field(description="WebSocket endpoint for real-time updates")
    
    # Real-time configuration
    real_time_data: bool = Field(description="Whether data updates in real-time")
    update_frequency: int = Field(description="Update frequency in seconds")
    
    # Interactive features
    filters_available: List[str] = Field(description="Available interactive filters")
    drill_down_enabled: bool = Field(description="Whether drill-down is enabled")
    export_enabled: bool = Field(description="Whether export is enabled")
    
class VisualizationListItem(BaseModel):
    """Visualization item for listing responses"""
    visualization_id: str = Field(description="Visualization identifier")
    name: str = Field(description="Visualization name")
    data_source: str = Field(description="Data source")
    chart_count: int = Field(description="Number of charts")
    has_dashboard: bool = Field(description="Whether it includes a dashboard")
    created_at: datetime = Field(description="Creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }