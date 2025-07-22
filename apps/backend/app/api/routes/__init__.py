"""
API Routes package - Push Back Focused

This package contains Push Back specific API route definitions.
"""

from flask import Blueprint

# Create the main API blueprint
api_bp = Blueprint('api', __name__)

# Import Push Back specific route module
from . import push_back
# Keep system routes for basic functionality
from . import system

# Register Push Back focused routes
api_bp.register_blueprint(push_back.push_back_bp, url_prefix='/push-back')
api_bp.register_blueprint(system.system_bp, url_prefix='/system')