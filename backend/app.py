"""
VEX U Scoring Analysis Platform - Flask API Server
Main application entry point
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app import create_app, socketio

# Create Flask application
app = create_app()

if __name__ == '__main__':
    # Get configuration from environment
    host = app.config.get('API_HOST', '0.0.0.0')
    port = app.config.get('API_PORT', 8000)
    debug = app.config.get('DEBUG', False)
    
    print(f"Starting VEX U Analysis API Server...")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Debug: {debug}")
    print(f"Environment: {app.config.get('FLASK_ENV', 'unknown')}")
    print(f"VEX Analysis Path: {app.config.get('VEX_ANALYSIS_PATH')}")
    print(f"API Documentation: http://{host}:{port}/api/docs")
    print(f"Health Check: http://{host}:{port}/health")
    
    # Run with SocketIO support
    socketio.run(
        app,
        host=host,
        port=port,
        debug=debug,
        allow_unsafe_werkzeug=True  # For development only
    )