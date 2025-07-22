# VEX U Scoring Analysis Platform - Flask API Backend

A comprehensive Flask API backend that serves as a bridge between the React frontend and the existing Python VEX U analysis tools. This API provides RESTful endpoints for all analysis operations, real-time WebSocket updates, and comprehensive data management.

## 🚀 Features

- **RESTful API** - Complete REST API for all VEX U analysis operations
- **WebSocket Support** - Real-time updates for long-running operations
- **ML Integration** - Direct integration with Python ML models
- **Analysis Operations** - Demo, full, statistical, scoring, and strategy analysis
- **Visualization Generation** - Create charts, dashboards, and interactive visualizations
- **Report Generation** - Generate strategic reports in multiple formats (HTML, PDF, JSON)
- **Strategy Management** - Create, optimize, and compare alliance strategies
- **Scenario Generation** - Generate and simulate match scenarios
- **Error Handling** - Comprehensive error handling and validation
- **API Documentation** - Auto-generated API documentation
- **Health Monitoring** - System health checks and metrics

## 🛠 Technology Stack

- **Flask 3.0** - Web framework
- **Flask-SocketIO** - WebSocket support for real-time updates
- **Flask-CORS** - Cross-origin resource sharing
- **Pydantic** - Data validation and serialization
- **Marshmallow** - Additional schema validation
- **Python 3.8+** - Runtime environment

## 📦 Installation

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Ensure the VEX analysis path is correct in `.env`:
   ```env
   VEX_ANALYSIS_PATH=../vex_u_scoring_analysis
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Flask Configuration
FLASK_APP=app
FLASK_ENV=development
FLASK_DEBUG=True

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
SECRET_KEY=your-secret-key-here

# CORS Configuration
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# VEX Analysis Configuration
VEX_ANALYSIS_PATH=../vex_u_scoring_analysis
PYTHON_PATH=python3

# File Upload Configuration
MAX_CONTENT_LENGTH=16777216  # 16MB
UPLOAD_FOLDER=uploads/
ALLOWED_EXTENSIONS=json,csv,xlsx,txt
```

### Production Configuration

For production deployment, update the following:

```env
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=generate-a-secure-secret-key
API_HOST=0.0.0.0
# Add SSL/TLS configuration
```

## 🚀 Running the Server

### Development Mode

```bash
python app.py
```

### Production Mode

```bash
# Using Gunicorn (recommended)
pip install gunicorn
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:8000 app:app

# Or using the built-in server
FLASK_ENV=production python app.py
```

### With Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:8000", "app:app"]
```

## 📚 API Documentation

Once the server is running, access the API documentation at:

- **Swagger UI**: `http://localhost:8000/api/docs`
- **Health Check**: `http://localhost:8000/health`
- **System Info**: `http://localhost:8000/api/system/info`

## 🛣 API Endpoints

### Analysis Operations
- `POST /api/analysis/demo` - Run quick demo analysis
- `POST /api/analysis/full` - Run comprehensive analysis
- `POST /api/analysis/statistical` - Run statistical analysis
- `POST /api/analysis/scoring` - Run scoring analysis
- `POST /api/analysis/strategy` - Run strategy analysis
- `GET /api/analysis/history` - Get analysis history
- `GET /api/analysis/{id}` - Get specific analysis

### Visualization
- `POST /api/visualizations/generate` - Generate visualizations
- `GET /api/visualizations/dashboard` - Get interactive dashboard
- `GET /api/visualizations/data` - Get visualization data
- `GET /api/visualizations/` - List visualizations

### Reports
- `POST /api/reports/generate` - Generate strategic reports
- `GET /api/reports/` - List reports
- `GET /api/reports/{id}` - Get specific report
- `GET /api/reports/{id}/download` - Download report
- `POST /api/reports/{id}/export` - Export report

### Strategies
- `GET /api/strategies/` - List strategies
- `POST /api/strategies/generate` - Generate new strategy
- `POST /api/strategies/optimize` - Optimize existing strategy
- `POST /api/strategies/compare` - Compare strategies
- `GET /api/strategies/{id}` - Get specific strategy
- `POST /api/strategies/` - Save strategy

### Scenarios
- `POST /api/scenarios/generate` - Generate match scenarios
- `POST /api/scenarios/simulate` - Simulate scenarios
- `POST /api/scenarios/evolve` - Evolve scenarios with genetic algorithms
- `GET /api/scenarios/` - List scenario sets
- `GET /api/scenarios/{id}` - Get specific scenario set
- `POST /api/scenarios/export` - Export scenarios

### ML Models
- `GET /api/ml/status` - Get ML model status
- `POST /api/ml/train` - Train ML models
- `POST /api/ml/predict` - Make ML predictions
- `POST /api/ml/optimize` - Optimize using ML
- `POST /api/ml/patterns` - Discover patterns
- `GET /api/ml/models` - List available models
- `GET /api/ml/jobs/{id}` - Get training job status

### System
- `GET /api/system/health` - Comprehensive health check
- `GET /api/system/info` - System information
- `GET /api/system/metrics` - Performance metrics
- `GET /api/system/config` - System configuration
- `GET /api/system/logs` - System logs

## 🔌 WebSocket Events

### Client → Server Events
- `connect` - Establish connection
- `subscribe` - Subscribe to data streams
- `unsubscribe` - Unsubscribe from streams
- `ping` - Keepalive ping
- `get_status` - Get connection status

### Server → Client Events
- `connected` - Connection established
- `analysis_update` - Analysis progress updates
- `training_update` - ML training progress
- `metrics_update` - System metrics updates
- `visualization_update` - Real-time chart updates
- `notification` - System notifications
- `error` - Error messages

### Subscription Types
- `analysis_progress_{analysis_id}` - Analysis progress
- `ml_training_{job_id}` - ML training progress
- `system_metrics_general` - System metrics
- `visualization_{viz_id}` - Visualization updates

## 🔄 Integration with VEX U Analysis

The API integrates with the existing Python analysis system through:

1. **Command Execution** - Executes Python scripts via subprocess
2. **Data Exchange** - JSON-based data exchange with temporary files
3. **Error Handling** - Captures and translates Python errors to API responses
4. **Progress Monitoring** - Real-time progress updates via WebSocket

### Example Integration Flow

```python
# API receives request
POST /api/analysis/demo

# Service layer calls VEX analysis
vex_service.run_demo_analysis(strategy_count=10)

# Executes: python main.py demo --strategies 10

# Parses output and returns structured response
{
  "success": true,
  "data": {
    "analysis_id": "...",
    "title": "VEX U Demo Analysis",
    "results": {...}
  }
}
```

## 🧪 Testing

### Unit Tests
```bash
pytest tests/
```

### Integration Tests
```bash
pytest tests/integration/
```

### API Testing
```bash
# Test with curl
curl -X POST http://localhost:8000/api/analysis/demo \
  -H "Content-Type: application/json" \
  -d '{"strategy_count": 5}'

# Test WebSocket connection
wscat -c ws://localhost:8000/socket.io/?EIO=4&transport=websocket
```

## 📊 Monitoring and Logging

### Health Checks
- **Basic Health**: `GET /health`
- **Detailed Health**: `GET /api/system/health`
- **Metrics**: `GET /api/system/metrics`

### Logging Configuration
- Development: Console logging with DEBUG level
- Production: File rotation with configurable levels
- JSON structured logging available

### Performance Monitoring
```python
# Enable metrics collection
from prometheus_flask_exporter import PrometheusMetrics
metrics = PrometheusMetrics(app)
```

## 🔐 Security Considerations

### Development
- CORS enabled for `localhost:3000`
- Debug mode enabled
- Detailed error messages

### Production
- Disable debug mode
- Configure proper CORS origins
- Use secure secret keys
- Implement rate limiting
- Add authentication/authorization
- Use HTTPS/WSS

### Recommended Security Headers
```python
# Add security headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response
```

## 🚀 Deployment

### Local Deployment
```bash
python app.py
```

### Docker Deployment
```bash
docker build -t vex-api .
docker run -p 8000:8000 -v $(pwd)/../vex_u_scoring_analysis:/app/vex_analysis vex-api
```

### Production Deployment
```bash
# With Gunicorn + Nginx
gunicorn --worker-class eventlet -w 1 --bind 127.0.0.1:8000 app:app
```

## 🤝 Frontend Integration

The API is designed to work seamlessly with the React frontend:

```typescript
// Frontend API calls
const response = await fetch('/api/analysis/demo', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ strategy_count: 10 })
});

// WebSocket connection
const socket = io('ws://localhost:8000');
socket.emit('subscribe', { 
  type: 'analysis_progress', 
  params: { id: 'analysis-123' } 
});
```

## 📝 Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive error handling
3. Include proper logging
4. Write tests for new endpoints
5. Update API documentation
6. Follow REST conventions

## 🐛 Troubleshooting

### Common Issues

1. **VEX Analysis Path Not Found**
   ```bash
   # Check the path in .env
   VEX_ANALYSIS_PATH=../vex_u_scoring_analysis
   
   # Verify the path exists
   ls ../vex_u_scoring_analysis/main.py
   ```

2. **Port Already in Use**
   ```bash
   # Change port in .env
   API_PORT=8001
   
   # Or kill existing process
   lsof -ti:8000 | xargs kill -9
   ```

3. **CORS Issues**
   ```bash
   # Add frontend URL to CORS_ORIGINS in .env
   CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
   ```

4. **WebSocket Connection Failed**
   - Check firewall settings
   - Verify WebSocket URL format
   - Ensure eventlet is installed

### Debug Mode
```bash
FLASK_DEBUG=True python app.py
```

### Logs Location
- Development: Console output
- Production: `logs/vex_api.log`

## 📄 License

This project is part of the VEX U Scoring Analysis Platform and follows the same licensing terms as the parent project.