"""
Error handlers for the Flask application
"""

from flask import jsonify, request
from werkzeug.exceptions import HTTPException
from marshmallow import ValidationError
from app.models.base import ErrorResponse
import logging

logger = logging.getLogger(__name__)

def register_error_handlers(app):
    """Register error handlers with the Flask app"""
    
    @app.errorhandler(400)
    def bad_request(error):
        """Handle bad request errors"""
        response = ErrorResponse(
            message="Bad request",
            error_code="BAD_REQUEST",
            error_details={"description": str(error.description)}
        )
        return jsonify(response.dict()), 400
    
    @app.errorhandler(401)
    def unauthorized(error):
        """Handle unauthorized errors"""
        response = ErrorResponse(
            message="Unauthorized access",
            error_code="UNAUTHORIZED",
            error_details={"description": "Valid authentication required"}
        )
        return jsonify(response.dict()), 401
    
    @app.errorhandler(403)
    def forbidden(error):
        """Handle forbidden errors"""
        response = ErrorResponse(
            message="Access forbidden",
            error_code="FORBIDDEN",
            error_details={"description": "Insufficient permissions"}
        )
        return jsonify(response.dict()), 403
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle not found errors"""
        response = ErrorResponse(
            message="Resource not found",
            error_code="NOT_FOUND",
            error_details={"path": request.path}
        )
        return jsonify(response.dict()), 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        """Handle method not allowed errors"""
        response = ErrorResponse(
            message="Method not allowed",
            error_code="METHOD_NOT_ALLOWED",
            error_details={
                "method": request.method,
                "path": request.path
            }
        )
        return jsonify(response.dict()), 405
    
    @app.errorhandler(422)
    def unprocessable_entity(error):
        """Handle validation errors"""
        response = ErrorResponse(
            message="Validation failed",
            error_code="VALIDATION_ERROR",
            error_details={"errors": error.description}
        )
        return jsonify(response.dict()), 422
    
    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        """Handle rate limit errors"""
        response = ErrorResponse(
            message="Rate limit exceeded",
            error_code="RATE_LIMIT_EXCEEDED",
            error_details={"description": "Too many requests"}
        )
        return jsonify(response.dict()), 429
    
    @app.errorhandler(500)
    def internal_server_error(error):
        """Handle internal server errors"""
        logger.error(f"Internal server error: {str(error)}")
        response = ErrorResponse(
            message="Internal server error",
            error_code="INTERNAL_ERROR",
            error_details={"description": "An unexpected error occurred"}
        )
        return jsonify(response.dict()), 500
    
    @app.errorhandler(503)
    def service_unavailable(error):
        """Handle service unavailable errors"""
        response = ErrorResponse(
            message="Service temporarily unavailable",
            error_code="SERVICE_UNAVAILABLE",
            error_details={"description": "Please try again later"}
        )
        return jsonify(response.dict()), 503
    
    @app.errorhandler(ValidationError)
    def handle_validation_error(error):
        """Handle Marshmallow validation errors"""
        response = ErrorResponse(
            message="Request validation failed",
            error_code="VALIDATION_ERROR",
            error_details={"field_errors": error.messages}
        )
        return jsonify(response.dict()), 422
    
    @app.errorhandler(HTTPException)
    def handle_http_exception(error):
        """Handle generic HTTP exceptions"""
        response = ErrorResponse(
            message=error.description or "HTTP error occurred",
            error_code=f"HTTP_{error.code}",
            error_details={"status_code": error.code}
        )
        return jsonify(response.dict()), error.code
    
    @app.errorhandler(Exception)
    def handle_generic_exception(error):
        """Handle any unhandled exceptions"""
        logger.error(f"Unhandled exception: {str(error)}", exc_info=True)
        
        # In production, don't expose internal error details
        if app.config.get('DEBUG'):
            error_details = {"exception": str(error), "type": type(error).__name__}
        else:
            error_details = {"description": "An unexpected error occurred"}
        
        response = ErrorResponse(
            message="Internal server error",
            error_code="UNHANDLED_EXCEPTION",
            error_details=error_details
        )
        return jsonify(response.dict()), 500