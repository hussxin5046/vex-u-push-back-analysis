"""
Base models for API responses and common data structures
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Generic, TypeVar
from pydantic import BaseModel, Field
from enum import Enum

T = TypeVar('T')

class BaseResponse(BaseModel, Generic[T]):
    """Base response model for all API endpoints"""
    success: bool = Field(description="Whether the request was successful")
    message: Optional[str] = Field(None, description="Human readable message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    data: Optional[T] = Field(None, description="Response data")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class SuccessResponse(BaseResponse[T]):
    """Success response model"""
    success: bool = Field(True, description="Always true for success responses")

class ErrorResponse(BaseResponse[None]):
    """Error response model"""
    success: bool = Field(False, description="Always false for error responses")
    error_code: Optional[str] = Field(None, description="Machine readable error code")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    data: None = Field(None, description="No data in error responses")

class PaginationInfo(BaseModel):
    """Pagination information for list responses"""
    page: int = Field(ge=1, description="Current page number")
    per_page: int = Field(ge=1, le=100, description="Items per page")
    total: int = Field(ge=0, description="Total number of items")
    pages: int = Field(ge=0, description="Total number of pages")
    has_prev: bool = Field(description="Whether there is a previous page")
    has_next: bool = Field(description="Whether there is a next page")

class PaginatedResponse(BaseResponse[List[T]]):
    """Paginated response model"""
    pagination: PaginationInfo = Field(description="Pagination information")

class StatusEnum(str, Enum):
    """Common status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class PriorityEnum(str, Enum):
    """Priority enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TaskInfo(BaseModel):
    """Background task information"""
    task_id: str = Field(description="Unique task identifier")
    status: StatusEnum = Field(description="Current task status")
    progress: float = Field(ge=0, le=100, description="Task progress percentage")
    message: Optional[str] = Field(None, description="Current task message")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result if completed")
    error: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Task creation time")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class FileInfo(BaseModel):
    """File information model"""
    filename: str = Field(description="Original filename")
    size: int = Field(ge=0, description="File size in bytes")
    content_type: str = Field(description="MIME content type")
    upload_time: datetime = Field(default_factory=datetime.utcnow, description="Upload timestamp")
    file_path: Optional[str] = Field(None, description="Server file path")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class HealthStatus(BaseModel):
    """Health check response model"""
    status: str = Field(description="Overall health status")
    version: str = Field(description="API version")
    uptime: float = Field(description="Uptime in seconds")
    checks: Dict[str, Dict[str, Any]] = Field(description="Individual health checks")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }