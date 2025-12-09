# src/api/app.py
"""
Flask API for Smart Traffic Management System.

Provides REST endpoints for traffic prediction and management.
"""

import os
import re
import time
import uuid
import logging
from functools import wraps
from typing import Optional

from flask import Flask, request, jsonify, g, abort
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import agents
try:
    from agents.orchestrator import Orchestrator, ValidationError as OrchestratorValidationError
except ImportError:
    # Handle import from different locations
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agents.orchestrator import Orchestrator, ValidationError as OrchestratorValidationError


# =============================================================================
# APP CONFIGURATION
# =============================================================================

def create_app(config: Optional[dict] = None) -> Flask:
    """
    Application factory for creating Flask app.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Load configuration
    app.config.update(
        SECRET_KEY=os.getenv("SECRET_KEY", "dev-key-change-in-production"),
        JSON_SORT_KEYS=False,
        MAX_CONTENT_LENGTH=1 * 1024 * 1024,  # 1MB max request size
    )
    
    if config:
        app.config.update(config)
    
    # Configure CORS
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    CORS(app, resources={
        r"/api/*": {
            "origins": allowed_origins,
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "X-Request-ID"]
        }
    })
    
    # Register error handlers
    register_error_handlers(app)
    
    # Register request hooks
    register_request_hooks(app)
    
    # Register blueprints
    from api.routes import api_bp
    app.register_blueprint(api_bp)
    
    return app


# =============================================================================
# ORCHESTRATOR INITIALIZATION
# =============================================================================

_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    """
    Get or create the orchestrator instance.
    
    Uses lazy initialization for proper error handling.
    
    Returns:
        Orchestrator instance
        
    Raises:
        RuntimeError: If orchestrator cannot be initialized
    """
    global _orchestrator
    
    if _orchestrator is None:
        try:
            logger.info("Initializing orchestrator...")
            _orchestrator = Orchestrator.create_default()
            logger.info("Orchestrator initialized successfully")
        except Exception as e:
            logger.critical(f"Failed to initialize orchestrator: {e}", exc_info=True)
            raise RuntimeError(f"Orchestrator initialization failed: {e}")
    
    return _orchestrator


# =============================================================================
# ERROR HANDLERS
# =============================================================================

def register_error_handlers(app: Flask) -> None:
    """Register error handlers for the application."""
    
    @app.errorhandler(HTTPException)
    def handle_http_exception(e):
        """Handle HTTP exceptions."""
        return jsonify({
            "error": e.name,
            "message": e.description,
            "status_code": e.code
        }), e.code
    
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({
            "error": "Bad Request",
            "message": str(e.description) if hasattr(e, 'description') else str(e),
            "status_code": 400
        }), 400
    
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({
            "error": "Not Found",
            "message": "The requested resource was not found",
            "status_code": 404
        }), 404
    
    @app.errorhandler(429)
    def rate_limit_exceeded(e):
        return jsonify({
            "error": "Too Many Requests",
            "message": "Rate limit exceeded. Please slow down.",
            "status_code": 429
        }), 429
    
    @app.errorhandler(500)
    def internal_error(e):
        logger.error(f"Internal error: {e}", exc_info=True)
        return jsonify({
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "status_code": 500
        }), 500
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        """Catch-all exception handler."""
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        return jsonify({
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "status_code": 500
        }), 500


# =============================================================================
# REQUEST HOOKS
# =============================================================================

def register_request_hooks(app: Flask) -> None:
    """Register before/after request hooks."""
    
    @app.before_request
    def before_request():
        """Execute before each request."""
        g.start_time = time.time()
        g.request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
    
    @app.after_request
    def after_request(response):
        """Execute after each request."""
        # Calculate request duration
        duration = (time.time() - getattr(g, 'start_time', time.time())) * 1000
        
        # Log request (skip health checks for cleaner logs)
        if request.path not in ['/health', '/ready']:
            logger.info(
                f"{request.method} {request.path} "
                f"status={response.status_code} "
                f"duration={duration:.0f}ms "
                f"request_id={getattr(g, 'request_id', 'unknown')}"
            )
        
        # Add headers
        response.headers['X-Request-ID'] = getattr(g, 'request_id', 'unknown')
        response.headers['X-Response-Time'] = f"{duration:.0f}ms"
        
        return response


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_location(location: Optional[str], field_name: str) -> str:
    """
    Validate and sanitize location input.
    
    Args:
        location: Location string to validate
        field_name: Name of field for error messages
        
    Returns:
        Sanitized location string
        
    Raises:
        400 error if validation fails
    """
    if not location:
        abort(400, description=f"{field_name} is required")
    
    if not isinstance(location, str):
        abort(400, description=f"{field_name} must be a string")
    
    location = location.strip()
    
    if len(location) < 2:
        abort(400, description=f"{field_name} must be at least 2 characters")
    
    if len(location) > 500:
        abort(400, description=f"{field_name} too long (max 500 characters)")
    
    # Check for potentially malicious input
    if re.search(r'[<>{}|\\^`\x00-\x1f]', location):
        abort(400, description=f"{field_name} contains invalid characters")
    
    return location


def validate_json_request() -> dict:
    """
    Validate that request has valid JSON body.
    
    Returns:
        Parsed JSON data
        
    Raises:
        400/415 error if invalid
    """
    if not request.is_json:
        abort(415, description="Content-Type must be application/json")
    
    data = request.get_json(silent=True)
    if data is None:
        abort(400, description="Invalid JSON body")
    
    return data


# =============================================================================
# ROUTES
# =============================================================================

# Create app at module level for gunicorn
app = Flask(__name__)

# Configure CORS
CORS(app)

# Register error handlers inline for simple setup
register_error_handlers(app)
register_request_hooks(app)


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    
    Used by load balancers and container orchestrators.
    """
    return jsonify({
        "status": "healthy",
        "service": "traffic-management-api",
        "version": os.getenv("APP_VERSION", "1.0.0"),
        "timestamp": time.time()
    })


@app.route('/ready', methods=['GET'])
def readiness_check():
    """
    Readiness check endpoint.
    
    Verifies all dependencies are available.
    """
    try:
        orchestrator = get_orchestrator()
        health = orchestrator.health_check()
        
        all_healthy = all(health.values())
        status = "ready" if all_healthy else "degraded"
        status_code = 200 if all_healthy else 503
        
        return jsonify({
            "status": status,
            "checks": health
        }), status_code
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return jsonify({
            "status": "not_ready",
            "error": str(e)
        }), 503


@app.route('/api/v1/predict', methods=['POST'])
def predict():
    """
    Traffic prediction endpoint.
    
    Analyzes traffic, weather, events, and routing to provide
    comprehensive traffic predictions and recommendations.
    
    Request Body:
        {
            "origin": "Starting location",
            "destination": "End location",
            "city": "City for events/weather (optional)"
        }
    
    Returns:
        Traffic prediction with recommendations
    """
    # Validate request
    data = validate_json_request()
    
    # Validate and extract fields
    origin = validate_location(data.get("origin"), "origin")
    destination = validate_location(data.get("destination"), "destination")
    
    # City is optional - default to first part of origin
    city = data.get("city")
    if city:
        city = validate_location(city, "city")
    else:
        # Extract city from origin (e.g., "New York, NY" -> "New York")
        city = origin.split(",")[0].strip()
    
    try:
        # Get orchestrator and run prediction
        orchestrator = get_orchestrator()
        result = orchestrator.run(origin, destination, city)
        
        return jsonify(result.to_dict())
        
    except OrchestratorValidationError as e:
        logger.warning(f"Validation error: {e}")
        abort(400, description=str(e))
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        abort(500, description="Prediction service temporarily unavailable")


@app.route('/api/v1/traffic', methods=['POST'])
def get_traffic():
    """
    Get traffic analysis only.
    
    Request Body:
        {
            "origin": "Starting location",
            "destination": "End location"
        }
    """
    data = validate_json_request()
    origin = validate_location(data.get("origin"), "origin")
    destination = validate_location(data.get("destination"), "destination")
    
    try:
        orchestrator = get_orchestrator()
        result = orchestrator.run(
            origin, 
            destination, 
            origin.split(",")[0],
            include_agents=["traffic"]
        )
        
        return jsonify({
            "traffic": result.traffic.to_dict(),
            "execution_time_ms": result.total_execution_time_ms
        })
        
    except Exception as e:
        logger.error(f"Traffic analysis failed: {e}", exc_info=True)
        abort(500, description="Traffic service temporarily unavailable")


@app.route('/api/v1/weather/<city>', methods=['GET'])
def get_weather(city: str):
    """
    Get weather analysis for a city.
    
    Args:
        city: City name
    """
    city = validate_location(city, "city")
    
    try:
        orchestrator = get_orchestrator()
        result = orchestrator.run(
            city, city, city,
            include_agents=["weather"]
        )
        
        response = jsonify({
            "weather": result.weather.to_dict(),
            "execution_time_ms": result.total_execution_time_ms
        })
        
        # Cache weather for 5 minutes
        response.headers['Cache-Control'] = 'public, max-age=300'
        
        return response
        
    except Exception as e:
        logger.error(f"Weather analysis failed: {e}", exc_info=True)
        abort(500, description="Weather service temporarily unavailable")


@app.route('/api/v1/events/<city>', methods=['GET'])
def get_events(city: str):
    """
    Get events analysis for a city.
    
    Args:
        city: City name
    """
    city = validate_location(city, "city")
    
    try:
        orchestrator = get_orchestrator()
        result = orchestrator.run(
            city, city, city,
            include_agents=["events"]
        )
        
        return jsonify({
            "events": result.events.to_dict(),
            "execution_time_ms": result.total_execution_time_ms
        })
        
    except Exception as e:
        logger.error(f"Events analysis failed: {e}", exc_info=True)
        abort(500, description="Events service temporarily unavailable")


@app.route('/api/v1/routes', methods=['POST'])
def get_routes():
    """
    Get alternative routes.
    
    Request Body:
        {
            "origin": "Starting location",
            "destination": "End location"
        }
    """
    data = validate_json_request()
    origin = validate_location(data.get("origin"), "origin")
    destination = validate_location(data.get("destination"), "destination")
    
    try:
        orchestrator = get_orchestrator()
        result = orchestrator.run(
            origin, destination, origin.split(",")[0],
            include_agents=["routing"]
        )
        
        return jsonify({
            "routes": result.routing.to_dict(),
            "execution_time_ms": result.total_execution_time_ms
        })
        
    except Exception as e:
        logger.error(f"Routing analysis failed: {e}", exc_info=True)
        abort(500, description="Routing service temporarily unavailable")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Development server only - use gunicorn in production
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "127.0.0.1")
    
    if debug_mode:
        logger.warning("Running in DEBUG mode - do not use in production!")
    
    # Initialize orchestrator on startup
    try:
        get_orchestrator()
    except Exception as e:
        logger.critical(f"Failed to initialize: {e}")
        exit(1)
    
    app.run(
        host=host,
        port=port,
        debug=debug_mode
    )
