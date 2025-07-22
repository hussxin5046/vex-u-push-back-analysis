"""
API Routes package

This package contains all the API route definitions organized by functionality.
"""

from flask import Blueprint

# Create the main API blueprint
api_bp = Blueprint('api', __name__)

# Import route modules to register them with the blueprint
from . import analysis
from . import visualization
from . import reports
from . import strategies
from . import scenarios
from . import ml_models
from . import system

# Register all route modules
api_bp.register_blueprint(analysis.analysis_bp, url_prefix='/analysis')
api_bp.register_blueprint(visualization.visualization_bp, url_prefix='/visualizations')
api_bp.register_blueprint(reports.reports_bp, url_prefix='/reports')
api_bp.register_blueprint(strategies.strategies_bp, url_prefix='/strategies')
api_bp.register_blueprint(scenarios.scenarios_bp, url_prefix='/scenarios')
api_bp.register_blueprint(ml_models.ml_bp, url_prefix='/ml')
api_bp.register_blueprint(system.system_bp, url_prefix='/system')