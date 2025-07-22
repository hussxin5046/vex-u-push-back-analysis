"""
Report models for generating and managing strategic reports
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum

class ReportType(str, Enum):
    """Types of reports that can be generated"""
    STRATEGIC = "strategic"
    STATISTICAL = "statistical"
    PERFORMANCE = "performance"
    COMPARATIVE = "comparative"
    ML_INSIGHTS = "ml_insights"
    COMPREHENSIVE = "comprehensive"
    CUSTOM = "custom"

class ReportFormat(str, Enum):
    """Available report output formats"""
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    MARKDOWN = "markdown"
    DOCX = "docx"

class SectionType(str, Enum):
    """Types of report sections"""
    EXECUTIVE_SUMMARY = "executive_summary"
    ANALYSIS_OVERVIEW = "analysis_overview"
    PERFORMANCE_METRICS = "performance_metrics"
    STRATEGIC_INSIGHTS = "strategic_insights"
    RECOMMENDATIONS = "recommendations"
    DATA_VISUALIZATION = "data_visualization"
    TECHNICAL_DETAILS = "technical_details"
    APPENDIX = "appendix"

class ReportSection(BaseModel):
    """Individual section within a report"""
    section_id: str = Field(description="Unique section identifier")
    section_type: SectionType = Field(description="Type of section")
    title: str = Field(description="Section title")
    order: int = Field(description="Section order in report")
    
    # Content
    content: str = Field(description="Section content (HTML/Markdown)")
    summary: Optional[str] = Field(None, description="Section summary")
    
    # Data and visualizations
    charts: Optional[List[str]] = Field(None, description="Chart IDs included in section")
    tables: Optional[List[Dict[str, Any]]] = Field(None, description="Tables included in section")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Key metrics for section")
    
    # Metadata
    importance: str = Field("medium", description="Section importance (high/medium/low)")
    data_sources: Optional[List[str]] = Field(None, description="Data sources used")
    
class ReportMetadata(BaseModel):
    """Report metadata and configuration"""
    report_id: str = Field(description="Unique report identifier")
    title: str = Field(description="Report title")
    subtitle: Optional[str] = Field(None, description="Report subtitle")
    
    # Generation info
    report_type: ReportType = Field(description="Type of report")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Generation timestamp")
    generated_by: Optional[str] = Field(None, description="Report generator")
    
    # Data sources
    analysis_ids: Optional[List[str]] = Field(None, description="Analysis IDs used")
    data_sources: List[str] = Field(description="Data sources used")
    
    # Configuration
    include_visualizations: bool = Field(True, description="Whether to include charts")
    include_raw_data: bool = Field(False, description="Whether to include raw data")
    confidence_level: float = Field(0.95, description="Statistical confidence level")
    
    # Version and tracking
    version: str = Field("1.0", description="Report version")
    template_version: Optional[str] = Field(None, description="Template version used")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ReportRequest(BaseModel):
    """Request model for report generation"""
    
    # Basic report configuration
    report_type: ReportType = Field(description="Type of report to generate")
    title: Optional[str] = Field(None, description="Custom report title")
    
    # Data sources
    analysis_ids: Optional[List[str]] = Field(None, description="Analysis IDs to include")
    strategy_ids: Optional[List[str]] = Field(None, description="Strategy IDs to include")
    scenario_ids: Optional[List[str]] = Field(None, description="Scenario IDs to include")
    
    # Content configuration
    sections: Optional[List[SectionType]] = Field(None, description="Sections to include (default: all)")
    include_executive_summary: bool = Field(True, description="Include executive summary")
    include_recommendations: bool = Field(True, description="Include recommendations")
    include_visualizations: bool = Field(True, description="Include charts and graphs")
    
    # Output configuration
    output_formats: List[ReportFormat] = Field([ReportFormat.HTML], description="Output formats")
    template: Optional[str] = Field(None, description="Custom template to use")
    
    # Filtering and focus
    focus_areas: Optional[List[str]] = Field(None, description="Areas to focus the report on")
    time_period: Optional[Dict[str, str]] = Field(None, description="Time period for data")
    
    # Advanced options
    confidence_level: float = Field(0.95, ge=0.01, le=0.99, description="Statistical confidence level")
    include_technical_details: bool = Field(False, description="Include technical analysis details")
    include_raw_data: bool = Field(False, description="Include raw data appendix")
    
    # Custom parameters
    custom_params: Optional[Dict[str, Any]] = Field(None, description="Custom report parameters")

class ReportData(BaseModel):
    """Complete report data structure"""
    metadata: ReportMetadata = Field(description="Report metadata")
    sections: List[ReportSection] = Field(description="Report sections")
    
    # Executive summary
    executive_summary: str = Field(description="Executive summary")
    key_findings: List[str] = Field(description="Key findings")
    recommendations: List[str] = Field(description="Strategic recommendations")
    
    # Data and analysis
    analysis_results: Optional[Dict[str, Any]] = Field(None, description="Underlying analysis results")
    performance_metrics: Optional[Dict[str, Any]] = Field(None, description="Performance metrics")
    statistical_data: Optional[Dict[str, Any]] = Field(None, description="Statistical data")
    
    # Visualizations
    charts: Optional[List[Dict[str, Any]]] = Field(None, description="Chart data")
    tables: Optional[List[Dict[str, Any]]] = Field(None, description="Table data")
    
    # Generation details
    generation_params: ReportRequest = Field(description="Parameters used for generation")
    generation_duration: Optional[float] = Field(None, description="Generation time in seconds")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ReportOutput(BaseModel):
    """Generated report output"""
    report_id: str = Field(description="Report identifier")
    format: ReportFormat = Field(description="Report format")
    
    # Content
    content: Optional[str] = Field(None, description="Report content (for text formats)")
    file_path: Optional[str] = Field(None, description="File path (for binary formats)")
    download_url: Optional[str] = Field(None, description="Download URL")
    
    # Metadata
    file_size: Optional[int] = Field(None, description="File size in bytes")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Generation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ReportResponse(BaseModel):
    """Response model for report operations"""
    report: ReportData = Field(description="Report data")
    outputs: List[ReportOutput] = Field(description="Generated report outputs")
    task_id: Optional[str] = Field(None, description="Background task ID if async")

class ReportTemplate(BaseModel):
    """Report template configuration"""
    template_id: str = Field(description="Template identifier")
    name: str = Field(description="Template name")
    report_type: ReportType = Field(description="Compatible report type")
    
    # Template structure
    sections: List[SectionType] = Field(description="Default sections")
    styling: Dict[str, Any] = Field(description="Template styling configuration")
    
    # Customization
    customizable_sections: List[str] = Field(description="Sections that can be customized")
    parameters: Dict[str, Any] = Field(description="Template parameters")
    
class ReportListItem(BaseModel):
    """Report item for listing responses"""
    report_id: str = Field(description="Report identifier")
    title: str = Field(description="Report title")
    report_type: ReportType = Field(description="Report type")
    generated_at: datetime = Field(description="Generation timestamp")
    formats_available: List[ReportFormat] = Field(description="Available formats")
    size: Optional[int] = Field(None, description="Total size in bytes")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }