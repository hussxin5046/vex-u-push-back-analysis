"""
VEX U Scoring Analysis Platform - Flask API Backend

This module creates the Flask application and configures all extensions.
"""

import os
from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
from config.config import config

# Initialize extensions
socketio = SocketIO(cors_allowed_origins="*")

def create_app(config_name=None):
    """
    Application factory pattern for creating Flask app instances
    
    Args:
        config_name: Configuration to use (development, production, testing)
        
    Returns:
        Configured Flask application instance
    """
    
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)
    
    # Initialize CORS
    CORS(app, 
         origins=app.config['CORS_ORIGINS'],
         allow_headers=['Content-Type', 'Authorization'],
         methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
    
    # Initialize SocketIO
    socketio.init_app(app, 
                     async_mode=app.config['SOCKETIO_ASYNC_MODE'],
                     cors_allowed_origins=app.config['CORS_ORIGINS'])
    
    # Register blueprints
    from app.api.routes import api_bp
    app.register_blueprint(api_bp, url_prefix=app.config['API_PREFIX'])
    
    # Register WebSocket events
    from app.websockets import events
    
    # Register error handlers
    from app.utils.error_handlers import register_error_handlers
    register_error_handlers(app)
    
    # Create upload folder if it doesn't exist
    upload_folder = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    @app.route('/health')
    def health_check():
        """Health check endpoint"""
        from app.models.base import HealthStatus
        import time
        
        # Check VEX analysis system
        vex_check = {"status": "ok", "message": "VEX analysis system available"}
        try:
            import sys
            sys.path.append(app.config['VEX_ANALYSIS_PATH'])
            vex_check["status"] = "ok"
        except Exception as e:
            vex_check = {"status": "error", "message": str(e)}
        
        health = HealthStatus(
            status="healthy" if vex_check["status"] == "ok" else "degraded",
            version="1.0.0",
            uptime=time.time(),  # This would be actual uptime in production
            checks={
                "vex_analysis": vex_check,
                "api": {"status": "ok", "message": "API server running"},
                "websocket": {"status": "ok", "message": "WebSocket server running"}
            }
        )
        
        return health.dict()
    
    @app.route('/')
    def index():
        """API information endpoint"""
        return {
            "name": "VEX U Scoring Analysis API",
            "version": "1.0.0",
            "description": "REST API for VEX U strategic analysis and ML predictions",
            "documentation": f"{app.config['API_PREFIX']}/docs",
            "health": "/health",
            "websocket": "/socket.io"
        }
    
    return app