# VEX U Analysis Platform

A comprehensive full-stack platform for analyzing, optimizing, and visualizing VEX U Push Back game strategies through advanced simulation, machine learning, and interactive web interfaces.

## 🏗️ Architecture

This monorepo contains three main components:

### 🖥️ Frontend (`apps/frontend/`)
- **React TypeScript** application with Material-UI
- Interactive dashboards and analysis visualizations
- Real-time updates via WebSocket
- Responsive design for desktop and mobile

### 🔧 Backend (`apps/backend/`)
- **Flask API** server with comprehensive REST endpoints
- Real-time WebSocket support for long-running operations
- Direct integration with analysis package
- Auto-generated API documentation

### 📊 Analysis Engine (`packages/vex-analysis/`)
- **Python package** for VEX U strategic analysis
- Monte Carlo simulation engine
- Machine learning models for strategy optimization
- Statistical analysis and pattern discovery

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **Node.js 16+**
- **npm** or **yarn**

### Setup
```bash
# 1. Clone and setup the entire platform
git clone <repository-url>
cd PushBackMLModel

# 2. Run automated setup
chmod +x scripts/setup.sh
./scripts/setup.sh

# 3. Start development servers
./scripts/start-dev.sh
```

### Access the Platform
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/docs

## 📁 Project Structure

```
PushBackMLModel/
├── apps/                              # Applications
│   ├── frontend/                      # React TypeScript frontend
│   │   ├── src/
│   │   │   ├── components/            # Reusable UI components
│   │   │   ├── pages/                 # Application pages
│   │   │   ├── services/              # API service layer
│   │   │   ├── hooks/                 # Custom React hooks
│   │   │   ├── contexts/              # React contexts
│   │   │   └── types/                 # TypeScript type definitions
│   │   ├── public/                    # Static assets
│   │   └── package.json
│   │
│   └── backend/                       # Flask API backend
│       ├── app/
│       │   ├── api/routes/            # API route handlers
│       │   ├── models/                # Data models
│       │   ├── services/              # Business logic
│       │   └── utils/                 # Utilities
│       ├── config/                    # Configuration
│       └── requirements.txt
│
├── packages/                          # Core packages
│   └── vex-analysis/                  # VEX U analysis engine
│       ├── vex_analysis/              # Main Python package
│       │   ├── core/                  # Simulation engine
│       │   ├── analysis/              # Strategy analysis
│       │   ├── ml_models/             # Machine learning
│       │   ├── visualization/         # Charts and graphs
│       │   └── reporting/             # Report generation
│       ├── demos/                     # Demonstration scripts
│       ├── tests/                     # Test suite
│       └── setup.py
│
├── shared/                            # Shared resources
│   ├── types/                         # Shared type definitions
│   ├── configs/                       # Configuration files
│   └── docker/                        # Docker configurations
│
├── scripts/                           # Development scripts
│   ├── setup.sh                       # Project setup
│   └── start-dev.sh                   # Start development servers
│
├── docs/                              # Documentation
├── docker-compose.yml                 # Multi-service orchestration
├── .env.example                       # Environment template
└── README.md                          # This file
```

## 🛠️ Development

### Individual Component Development

#### Frontend Only
```bash
cd apps/frontend
npm start
```

#### Backend Only
```bash
cd apps/backend
source venv/bin/activate
python app.py
```

#### Analysis Package Only
```bash
cd packages/vex-analysis
source venv/bin/activate
python vex_analysis/main.py demo
```

### Docker Development
```bash
# Start all services with Docker
docker-compose up

# Start specific services
docker-compose up frontend backend
```

### Environment Configuration

Copy `.env.example` to `.env` and configure:

```env
# Frontend
REACT_APP_API_URL=http://localhost:8000

# Backend
FLASK_ENV=development
API_PORT=8000
VEX_ANALYSIS_PATH=./packages/vex-analysis

# Analysis Package
PYTHONPATH=./packages/vex-analysis
```

## 📊 Features

### Analysis Capabilities
- **13 distinct strategies** with Monte Carlo simulation
- **Real-time scenario generation** with 400+ realistic match conditions
- **Statistical analysis** with confidence intervals and correlation discovery
- **Machine learning optimization** for strategy selection
- **Interactive visualizations** with web-based dashboards

### Frontend Features
- **Real-time dashboards** with live data updates
- **Strategy comparison tools** with visual analytics
- **Match prediction interface** with probability calculations
- **Report generation** with exportable charts
- **Responsive design** optimized for all devices

### Backend Features
- **RESTful API** with comprehensive endpoints
- **WebSocket support** for real-time updates
- **Background job processing** for long-running analyses
- **File upload handling** for custom data sets
- **Auto-generated documentation** with interactive testing

## 🧪 Testing

### Run All Tests
```bash
# Frontend tests
cd apps/frontend && npm test

# Backend tests
cd apps/backend && python -m pytest

# Analysis package tests
cd packages/vex-analysis && python -m pytest tests/
```

### Integration Testing
```bash
# Full system integration test
cd packages/vex-analysis
python vex_analysis/main.py test
```

## 📚 API Documentation

Once the backend is running, access comprehensive API documentation at:
- **Interactive Docs**: http://localhost:8000/api/docs
- **Health Check**: http://localhost:8000/health

### Key Endpoints
- `POST /api/analysis/demo` - Quick strategy analysis
- `POST /api/analysis/full` - Comprehensive analysis
- `GET /api/strategies/` - List available strategies
- `POST /api/visualizations/generate` - Create visualizations
- `WebSocket: /socket.io` - Real-time updates

## 🚀 Deployment

### Development
```bash
./scripts/start-dev.sh
```

### Production (Docker)
```bash
docker-compose -f docker-compose.prod.yml up
```

### Production (Manual)
```bash
# Build frontend
cd apps/frontend && npm run build

# Start backend with Gunicorn
cd apps/backend && gunicorn --workers 4 --bind 0.0.0.0:8000 app:app
```

## 🤝 Contributing

1. Follow the established project structure
2. Add tests for new functionality
3. Update documentation for API changes
4. Use TypeScript for frontend development
5. Follow Python PEP 8 for backend code

## 📄 License

This project follows the same licensing terms as the original VEX U analysis components.

---

## 🎯 Usage Examples

### Quick Analysis
```bash
# Using the analysis package directly
cd packages/vex-analysis
python vex_analysis/main.py demo

# Via API
curl -X POST http://localhost:8000/api/analysis/demo
```

### Custom Strategy Analysis
```bash
# Generate comprehensive report
cd packages/vex-analysis
python vex_analysis/main.py report

# View results in frontend
open http://localhost:3000/reports
```

### Real-time Visualization
```bash
# Start platform
./scripts/start-dev.sh

# Access interactive dashboards
open http://localhost:3000/dashboard
```

Perfect for teams, coaches, and analysts seeking competitive advantage through data-driven strategic analysis and optimization.