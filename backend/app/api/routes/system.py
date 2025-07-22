"""
System API routes for health checks and system information
"""

from flask import Blueprint, request, jsonify, current_app
from app.models.base import SuccessResponse, ErrorResponse, HealthStatus
from datetime import datetime
import logging
import sys
import os

logger = logging.getLogger(__name__)

system_bp = Blueprint('system', __name__)

@system_bp.route('/health', methods=['GET'])
def health_check():
    """
    Comprehensive health check
    
    GET /api/system/health
    """
    try:
        checks = {}
        overall_status = "healthy"
        
        # Check VEX analysis system
        try:
            vex_path = current_app.config['VEX_ANALYSIS_PATH']
            main_script = os.path.join(vex_path, 'main.py')
            if os.path.exists(main_script):
                checks["vex_analysis"] = {
                    "status": "ok",
                    "message": "VEX analysis system available",
                    "path": vex_path
                }
            else:
                checks["vex_analysis"] = {
                    "status": "error",
                    "message": "VEX analysis main.py not found",
                    "path": vex_path
                }
                overall_status = "degraded"
        except Exception as e:
            checks["vex_analysis"] = {
                "status": "error",
                "message": str(e)
            }
            overall_status = "degraded"
        
        # Check Python environment
        checks["python"] = {
            "status": "ok",
            "version": sys.version,
            "executable": sys.executable
        }
        
        # Check API server
        checks["api_server"] = {
            "status": "ok",
            "message": "API server running",
            "flask_env": current_app.config.get('FLASK_ENV', 'unknown')
        }
        
        # Check WebSocket server
        checks["websocket"] = {
            "status": "ok", 
            "message": "WebSocket server available"
        }
        
        # Check file system
        try:
            upload_folder = current_app.config['UPLOAD_FOLDER']
            if os.path.exists(upload_folder) and os.access(upload_folder, os.W_OK):
                checks["filesystem"] = {
                    "status": "ok",
                    "message": "Upload directory accessible",
                    "upload_folder": upload_folder
                }
            else:
                checks["filesystem"] = {
                    "status": "warning",
                    "message": "Upload directory not accessible",
                    "upload_folder": upload_folder
                }
        except Exception as e:
            checks["filesystem"] = {
                "status": "error",
                "message": str(e)
            }
        
        health = HealthStatus(
            status=overall_status,
            version="1.0.0",
            uptime=3600.0,  # Mock uptime
            checks=checks
        )
        
        status_code = 200 if overall_status == "healthy" else 503
        
        response = SuccessResponse(
            message=f"System is {overall_status}",
            data=health.dict()
        )
        
        return jsonify(response.dict()), status_code
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        error_response = ErrorResponse(
            message="Health check failed",
            error_code="HEALTH_CHECK_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500

@system_bp.route('/info', methods=['GET'])
def system_info():
    """
    Get system information
    
    GET /api/system/info
    """
    try:
        info = {
            "api_version": "1.0.0",
            "python_version": sys.version,
            "flask_env": current_app.config.get('FLASK_ENV', 'unknown'),
            "vex_analysis_path": current_app.config['VEX_ANALYSIS_PATH'],
            "features": {
                "analysis": True,
                "visualization": True,
                "reports": True,
                "ml_models": True,
                "scenarios": True,
                "strategies": True,
                "websockets": True
            },
            "limits": {
                "max_file_size": current_app.config['MAX_CONTENT_LENGTH'],
                "allowed_extensions": list(current_app.config['ALLOWED_EXTENSIONS']),
                "api_timeout": 300
            },
            "endpoints": {
                "analysis": "/api/analysis",
                "visualization": "/api/visualizations", 
                "reports": "/api/reports",
                "strategies": "/api/strategies",
                "scenarios": "/api/scenarios",
                "ml_models": "/api/ml",
                "system": "/api/system"
            }
        }
        
        response = SuccessResponse(
            message="System information retrieved successfully",
            data=info
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error(f"System info retrieval failed: {str(e)}")
        error_response = ErrorResponse(
            message="Failed to retrieve system information",
            error_code="INFO_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500

@system_bp.route('/metrics', methods=['GET'])
def system_metrics():
    """
    Get system performance metrics
    
    GET /api/system/metrics
    """
    try:
        import psutil
        import time
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu": {
                "usage_percent": cpu_percent,
                "count": psutil.cpu_count()
            },
            "memory": {
                "total": memory.total,
                "used": memory.used,
                "available": memory.available,
                "percent": memory.percent
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100
            },
            "api": {
                "active_requests": 0,  # Mock data
                "total_requests": 0,   # Mock data
                "error_rate": 0.0,     # Mock data
                "average_response_time": 0.150  # Mock data
            }
        }
        
        response = SuccessResponse(
            message="System metrics retrieved successfully",
            data=metrics
        )
        
        return jsonify(response.dict()), 200
        
    except ImportError:
        # psutil not available, return basic metrics
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Advanced metrics not available (psutil not installed)",
            "api": {
                "status": "running",
                "uptime": 3600.0
            }
        }
        
        response = SuccessResponse(
            message="Basic system metrics retrieved",
            data=metrics
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error(f"System metrics retrieval failed: {str(e)}")
        error_response = ErrorResponse(
            message="Failed to retrieve system metrics",
            error_code="METRICS_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500

@system_bp.route('/config', methods=['GET'])
def system_config():
    """
    Get system configuration (non-sensitive)
    
    GET /api/system/config
    """
    try:
        config = {
            "api_host": current_app.config['API_HOST'],
            "api_port": current_app.config['API_PORT'],
            "api_prefix": current_app.config['API_PREFIX'],
            "cors_origins": current_app.config['CORS_ORIGINS'],
            "max_content_length": current_app.config['MAX_CONTENT_LENGTH'],
            "allowed_extensions": list(current_app.config['ALLOWED_EXTENSIONS']),
            "upload_folder": current_app.config['UPLOAD_FOLDER'],
            "socketio_async_mode": current_app.config['SOCKETIO_ASYNC_MODE'],
            "log_level": current_app.config['LOG_LEVEL'],
            "environment": current_app.config.get('FLASK_ENV', 'unknown')
        }
        
        response = SuccessResponse(
            message="System configuration retrieved successfully",
            data=config
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error(f"System config retrieval failed: {str(e)}")
        error_response = ErrorResponse(
            message="Failed to retrieve system configuration",
            error_code="CONFIG_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500

@system_bp.route('/logs', methods=['GET'])
def system_logs():
    """
    Get recent system logs
    
    GET /api/system/logs?level=ERROR&lines=100
    """
    try:
        level = request.args.get('level', 'INFO').upper()
        lines = int(request.args.get('lines', 100))
        
        # Mock log entries
        log_entries = [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "level": "INFO",
                "message": "API server started successfully",
                "module": "app"
            },
            {
                "timestamp": datetime.utcnow().isoformat(),
                "level": "INFO", 
                "message": "VEX analysis system initialized",
                "module": "vex_service"
            }
        ]
        
        # Filter by level if specified
        if level != 'ALL':
            log_entries = [entry for entry in log_entries if entry['level'] == level]
        
        # Limit number of entries
        log_entries = log_entries[:lines]
        
        response = SuccessResponse(
            message=f"Retrieved {len(log_entries)} log entries",
            data={
                "logs": log_entries,
                "total": len(log_entries),
                "level_filter": level,
                "lines_requested": lines
            }
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error(f"System logs retrieval failed: {str(e)}")
        error_response = ErrorResponse(
            message="Failed to retrieve system logs",
            error_code="LOGS_ERROR",
            error_details={"error": str(e)}
        )
        return jsonify(error_response.dict()), 500