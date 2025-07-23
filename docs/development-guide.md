# Development Guide

This guide provides comprehensive information for developers contributing to or extending the Push Back Analysis System. It covers architecture, development setup, coding standards, and deployment procedures.

## 🏗️ Architecture Overview

The Push Back Analysis System follows a modular architecture optimized for performance and maintainability:

```
VEX U Push Back Analysis System
├── apps/
│   ├── frontend/                 # React TypeScript UI
│   │   ├── src/pages/           # Push Back strategy pages
│   │   ├── src/components/      # Reusable UI components
│   │   └── src/services/        # API integration layer
│   └── backend/                 # Flask API server
│       ├── app/api/routes/      # Push Back API endpoints
│       ├── app/models/          # Data models
│       └── app/services/        # Business logic layer
└── packages/
    └── vex-analysis/            # Core analysis engine
        ├── vex_analysis/
        │   ├── core/           # Push Back scoring engine
        │   ├── analysis/       # Strategy analysis framework
        │   ├── simulation/     # Monte Carlo engine
        │   └── ml_models/      # Machine learning components
        └── tests/              # Comprehensive test suite
```

## 🛠️ Development Environment Setup

### Prerequisites

- **Python 3.8+** with pip
- **Node.js 16+** with npm
- **Git** for version control
- **VS Code** (recommended) with Python and TypeScript extensions

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/hussxin5046/vex-u-push-back-analysis.git
cd vex-u-push-back-analysis

# Set up Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Install the analysis package in development mode
cd packages/vex-analysis
pip install -e .
cd ../..

# Install frontend dependencies
cd apps/frontend
npm install
cd ../..

# Install backend dependencies (if any additional)
cd apps/backend
pip install -r requirements.txt
cd ../..
```

### Environment Configuration

Create environment configuration files:

```bash
# Backend environment (.env)
cat > apps/backend/.env << EOF
FLASK_ENV=development
FLASK_DEBUG=True
DATABASE_URL=sqlite:///pushback_analysis.db
SECRET_KEY=your-secret-key-here
CORS_ORIGINS=http://localhost:3000
LOG_LEVEL=DEBUG
EOF

# Frontend environment (.env.local)
cat > apps/frontend/.env.local << EOF
REACT_APP_API_BASE_URL=http://localhost:5000
REACT_APP_WS_URL=ws://localhost:5000
REACT_APP_ENVIRONMENT=development
EOF
```

### Development Scripts

Add these scripts to your development workflow:

```bash
# Start development servers
./scripts/dev-start.sh         # Starts both frontend and backend
./scripts/backend-dev.sh       # Backend only
./scripts/frontend-dev.sh      # Frontend only

# Run tests
./scripts/test-all.sh          # Complete test suite
./scripts/test-quick.sh        # Quick validation tests
./scripts/test-performance.sh  # Performance benchmarks

# Code quality
./scripts/lint.sh              # Linting and formatting
./scripts/type-check.sh        # TypeScript type checking
```

Create the development scripts:

```bash
mkdir -p scripts

# Backend development script
cat > scripts/backend-dev.sh << 'EOF'
#!/bin/bash
cd apps/backend
export FLASK_ENV=development
export FLASK_DEBUG=True
python app.py
EOF

# Frontend development script
cat > scripts/frontend-dev.sh << 'EOF'
#!/bin/bash
cd apps/frontend
npm start
EOF

# Make scripts executable
chmod +x scripts/*.sh
```

## 🔧 Code Organization

### Python Package Structure

```python
# packages/vex-analysis/vex_analysis/
├── __init__.py                 # Package initialization
├── core/                       # Core Push Back components
│   ├── __init__.py
│   ├── simulator.py           # Push Back scoring engine
│   └── field_elements.py      # Goals, blocks, field state
├── analysis/                   # Strategic analysis
│   ├── __init__.py
│   ├── push_back_strategy_analyzer.py
│   └── decision_support.py
├── simulation/                 # Monte Carlo engine
│   ├── __init__.py
│   ├── push_back_monte_carlo.py
│   ├── push_back_scenarios.py
│   └── push_back_insights.py
└── ml_models/                  # Machine learning (future)
    ├── __init__.py
    └── strategy_predictor.py
```

### Frontend Component Structure

```typescript
// apps/frontend/src/
├── components/                 # Reusable components
│   ├── common/                # Generic UI components
│   ├── analysis/              # Analysis-specific components
│   └── strategy/              # Strategy builder components
├── pages/                     # Route components
│   ├── PushBackDashboard.tsx  # Main dashboard
│   ├── StrategyBuilder.tsx    # Strategy configuration
│   ├── QuickAnalysis.tsx      # Rapid analysis
│   └── DecisionTools.tsx      # Decision support
├── services/                  # API integration
│   ├── pushBackApi.ts         # Push Back API client
│   ├── websocket.ts           # WebSocket connection
│   └── types.ts               # TypeScript definitions
└── utils/                     # Utility functions
    ├── analysis.ts            # Analysis helpers
    └── visualization.ts       # Chart and graph utilities
```

### Backend API Structure

```python
# apps/backend/app/
├── __init__.py
├── api/                       # API endpoints
│   ├── __init__.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── push_back.py       # Push Back analysis endpoints
│   │   └── websocket.py       # WebSocket handlers
├── models/                    # Data models
│   ├── __init__.py
│   ├── push_back.py           # Push Back data structures
│   └── base.py                # Base model classes
├── services/                  # Business logic
│   ├── __init__.py
│   ├── analysis_service.py    # Analysis orchestration
│   └── websocket_service.py   # Real-time updates
└── utils/                     # Utilities
    ├── __init__.py
    ├── validation.py          # Input validation
    └── performance.py         # Performance monitoring
```

## 📝 Coding Standards

### Python Standards

Follow PEP 8 with project-specific guidelines:

```python
# File header template
"""
Module for Push Back [specific functionality].

This module provides [description of functionality] for the Push Back
analysis system, focusing on [key aspects].
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

# Class naming: PascalCase with Push Back prefix for game-specific classes
class PushBackScoringEngine:
    """
    Push Back specific scoring engine.
    
    Calculates scores according to official Push Back rules with
    optimizations for strategic analysis.
    """
    
    def __init__(self):
        # Constants in UPPER_CASE
        self.POINTS_PER_BLOCK = 3
        self.AUTONOMOUS_WIN_POINTS = 7
    
    def calculate_push_back_score(self, field_state: 'PushBackFieldState', 
                                 alliance: str) -> Tuple[int, Dict[str, int]]:
        """
        Calculate Push Back score for alliance.
        
        Args:
            field_state: Current field configuration
            alliance: "red" or "blue"
            
        Returns:
            Tuple of (total_score, score_breakdown)
        """
        # Implementation with clear variable names
        block_points = self._calculate_block_points(field_state, alliance)
        control_points = self._calculate_control_zone_points(field_state, alliance)
        parking_points = self._calculate_parking_points(field_state, alliance)
        autonomous_points = self._calculate_autonomous_points(field_state, alliance)
        
        total_score = block_points + control_points + parking_points + autonomous_points
        
        breakdown = {
            "blocks": block_points,
            "control_zone": control_points,
            "parking": parking_points,
            "autonomous_win": autonomous_points
        }
        
        return total_score, breakdown

# Type hints for all function parameters and return values
def analyze_strategy(robot_capabilities: Dict[str, float],
                    opponent_analysis: Dict[str, str],
                    competition_context: Optional[Dict[str, str]] = None) -> StrategyInsights:
    """Analyze Push Back strategy with comprehensive insights."""
    pass
```

### TypeScript Standards

```typescript
// Type definitions for Push Back analysis
interface PushBackRobotConfig {
  cycleTime: number;              // Seconds per scoring cycle
  pickupReliability: number;      // 0.0 to 1.0
  scoringReliability: number;     // 0.0 to 1.0
  autonomousReliability: number;  // 0.0 to 1.0
  parkingStrategy: ParkingStrategy;
  goalPriority: GoalPriority;
}

interface AnalysisResult {
  winProbability: number;
  expectedScore: number;
  scoreVariance: number;
  confidenceLevel: number;
  executionTime: number;
}

// Component naming: PascalCase with descriptive names
const PushBackStrategyBuilder: React.FC = () => {
  const [robotConfig, setRobotConfig] = useState<PushBackRobotConfig>({
    cycleTime: 5.0,
    pickupReliability: 0.95,
    scoringReliability: 0.98,
    autonomousReliability: 0.90,
    parkingStrategy: ParkingStrategy.LATE,
    goalPriority: GoalPriority.BALANCED
  });

  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async (): Promise<void> => {
    setLoading(true);
    try {
      const result = await pushBackApi.analyzeStrategy({
        robotCapabilities: robotConfig,
        opponentAnalysis: { strength: 'competitive' }
      });
      setAnalysis(result.analysis);
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4">Push Back Strategy Analysis</Typography>
      {/* Component implementation */}
    </Box>
  );
};
```

### API Endpoint Standards

```python
# apps/backend/app/api/routes/push_back.py
from flask import Blueprint, request, jsonify
from marshmallow import Schema, fields, validate
from app.services.analysis_service import PushBackAnalysisService
from app.utils.validation import validate_json
from app.models.push_back import RobotConfigurationModel

push_back_bp = Blueprint('push_back', __name__, url_prefix='/api/push-back')

class AnalyzeStrategySchema(Schema):
    """Schema for strategy analysis request validation."""
    robot_capabilities = fields.Dict(required=True)
    strategy_preferences = fields.Dict(missing={})
    opponent_analysis = fields.Dict(missing={})
    match_context = fields.Dict(missing={})

@push_back_bp.route('/analyze', methods=['POST'])
@validate_json(AnalyzeStrategySchema)
def analyze_strategy():
    """
    Analyze Push Back strategy configuration.
    
    Provides comprehensive strategic analysis including win probability,
    expected score, and strategic recommendations.
    
    Returns:
        JSON response with analysis results and recommendations
    """
    try:
        data = request.get_json()
        
        # Initialize analysis service
        analysis_service = PushBackAnalysisService()
        
        # Perform analysis
        result = analysis_service.analyze_strategy(
            robot_capabilities=data['robot_capabilities'],
            strategy_preferences=data.get('strategy_preferences', {}),
            opponent_analysis=data.get('opponent_analysis', {}),
            match_context=data.get('match_context', {})
        )
        
        return jsonify({
            "success": True,
            "analysis": result.to_dict(),
            "execution_time": result.execution_time
        })
        
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": {
                "code": "INVALID_INPUT",
                "message": str(e)
            }
        }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "Analysis failed"
            }
        }), 500
```

## 🧪 Testing Guidelines

### Test Organization

```python
# tests/test_push_back_[module].py
import pytest
from unittest.mock import Mock, patch
from vex_analysis.simulation import PushBackMonteCarloEngine

class TestPushBackMonteCarloEngine:
    """Test suite for Monte Carlo simulation engine."""
    
    def setup_method(self):
        """Set up test fixtures before each test."""
        self.red_robot = create_competitive_robot()
        self.blue_robot = create_default_robot()
        self.engine = PushBackMonteCarloEngine(self.red_robot, self.blue_robot)
    
    def test_basic_simulation_functionality(self):
        """Test basic simulation runs successfully."""
        results, execution_time = self.engine.run_simulation(100)
        
        assert len(results) == 100
        assert execution_time > 0
        assert all(r.winner in ["red", "blue", "tie"] for r in results)
    
    @pytest.mark.performance
    def test_performance_requirements(self):
        """Test simulation meets performance requirements."""
        results, execution_time = self.engine.run_simulation(1000, use_parallel=True)
        
        assert len(results) == 1000
        assert execution_time < 10.0  # Performance requirement
        
        rate = len(results) / execution_time
        assert rate > 100  # Minimum acceptable rate
    
    @pytest.mark.parametrize("robot_type,expected_win_rate", [
        ("competitive", 0.7),
        ("default", 0.5),
        ("beginner", 0.3)
    ])
    def test_robot_performance_scaling(self, robot_type, expected_win_rate):
        """Test different robot types produce expected win rates."""
        test_robot = globals()[f"create_{robot_type}_robot"]()
        engine = PushBackMonteCarloEngine(test_robot, create_default_robot())
        
        results, _ = engine.run_simulation(500)
        actual_win_rate = sum(1 for r in results if r.winner == "red") / len(results)
        
        assert abs(actual_win_rate - expected_win_rate) < 0.15  # Tolerance
```

### Frontend Testing

```typescript
// apps/frontend/src/components/__tests__/StrategyBuilder.test.tsx
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { PushBackStrategyBuilder } from '../StrategyBuilder';
import * as pushBackApi from '../../services/pushBackApi';

// Mock API calls
jest.mock('../../services/pushBackApi');
const mockPushBackApi = pushBackApi as jest.Mocked<typeof pushBackApi>;

describe('PushBackStrategyBuilder', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders strategy configuration form', () => {
    render(<PushBackStrategyBuilder />);
    
    expect(screen.getByText('Push Back Strategy Builder')).toBeInTheDocument();
    expect(screen.getByLabelText('Cycle Time')).toBeInTheDocument();
    expect(screen.getByLabelText('Pickup Reliability')).toBeInTheDocument();
  });

  test('runs analysis when analyze button clicked', async () => {
    const mockAnalysisResult = {
      analysis: {
        winProbability: 0.75,
        expectedScore: 120.5,
        confidenceLevel: 0.89
      }
    };
    
    mockPushBackApi.analyzeStrategy.mockResolvedValueOnce(mockAnalysisResult);
    
    render(<PushBackStrategyBuilder />);
    
    fireEvent.click(screen.getByText('Run Analysis'));
    
    await waitFor(() => {
      expect(screen.getByText('Win Probability: 75.0%')).toBeInTheDocument();
      expect(screen.getByText('Expected Score: 120.5')).toBeInTheDocument();
    });
    
    expect(pushBackApi.analyzeStrategy).toHaveBeenCalledTimes(1);
  });

  test('handles analysis errors gracefully', async () => {
    mockPushBackApi.analyzeStrategy.mockRejectedValueOnce(new Error('Analysis failed'));
    
    render(<PushBackStrategyBuilder />);
    
    fireEvent.click(screen.getByText('Run Analysis'));
    
    await waitFor(() => {
      expect(screen.getByText('Analysis failed. Please try again.')).toBeInTheDocument();
    });
  });
});
```

## 🚀 Deployment

### Production Build

```bash
# Build frontend for production
cd apps/frontend
npm run build

# Create production distribution
cd ../..
python scripts/build-production.py

# The build process creates:
# - apps/frontend/build/     # Static frontend assets
# - dist/                    # Complete distribution package
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM node:16-alpine AS frontend-build
WORKDIR /app
COPY apps/frontend/package*.json ./
RUN npm ci --only=production
COPY apps/frontend/ ./
RUN npm run build

FROM python:3.9-slim AS backend
WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY apps/backend/ ./backend/
COPY packages/vex-analysis/ ./packages/vex-analysis/
RUN cd packages/vex-analysis && pip install -e .

# Copy frontend build
COPY --from=frontend-build /app/build ./backend/static/

# Set up production environment
ENV FLASK_ENV=production
ENV PYTHONPATH=/app/backend

EXPOSE 5000
CMD ["python", "backend/app.py"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  pushback-analysis:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=sqlite:///data/pushback.db
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - pushback-analysis
    restart: unless-stopped
```

### Environment-Specific Configuration

```python
# apps/backend/config.py
import os
from typing import Type

class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Push Back specific settings
    MONTE_CARLO_DEFAULT_SIZE = 1000
    SIMULATION_TIMEOUT = 30  # seconds
    MAX_CONCURRENT_SIMULATIONS = 5

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///pushback_dev.db'
    CORS_ORIGINS = ['http://localhost:3000']
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '').split(',')
    LOG_LEVEL = 'INFO'
    
    # Production optimizations
    MONTE_CARLO_DEFAULT_SIZE = 2000
    MAX_CONCURRENT_SIMULATIONS = 10

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False
    
    # Faster tests
    MONTE_CARLO_DEFAULT_SIZE = 100

config: Dict[str, Type[Config]] = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
```

## 🔧 Performance Optimization

### Backend Optimization

```python
# Performance monitoring and optimization
import time
import functools
from flask import g
import cProfile
import pstats

def monitor_performance(func):
    """Decorator to monitor function performance."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Log slow operations
        if execution_time > 1.0:
            print(f"SLOW OPERATION: {func.__name__} took {execution_time:.3f}s")
        
        return result
    return wrapper

# Caching for expensive operations
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_strategy_analysis(robot_config_hash: str, opponent_hash: str):
    """Cache strategy analysis results."""
    # Implementation with caching
    pass

# Database query optimization
def optimize_database_queries():
    """Optimize database queries for production."""
    # Use connection pooling
    # Implement query result caching
    # Add database indexes
    pass
```

### Frontend Optimization

```typescript
// Performance optimization techniques
import React, { memo, useMemo, useCallback } from 'react';
import { debounce } from 'lodash';

// Memoize expensive components
const PushBackAnalysisChart = memo(({ data, options }) => {
  const chartData = useMemo(() => {
    return processChartData(data);
  }, [data]);

  return <Chart data={chartData} options={options} />;
});

// Debounce expensive operations
const StrategyAnalyzer: React.FC = () => {
  const [robotConfig, setRobotConfig] = useState<RobotConfig>({});
  
  const debouncedAnalyze = useCallback(
    debounce(async (config: RobotConfig) => {
      const result = await analyzeStrategy(config);
      setAnalysisResult(result);
    }, 500),
    []
  );

  useEffect(() => {
    debouncedAnalyze(robotConfig);
  }, [robotConfig, debouncedAnalyze]);

  return (
    // Component implementation
  );
};

// Code splitting for large components
const LazyMonteCarloAnalysis = React.lazy(() => import('./MonteCarloAnalysis'));

function App() {
  return (
    <Suspense fallback={<div>Loading Monte Carlo Analysis...</div>}>
      <LazyMonteCarloAnalysis />
    </Suspense>
  );
}
```

## 🐛 Debugging and Troubleshooting

### Common Issues and Solutions

```python
# Debug configuration
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Performance debugging
def debug_simulation_performance():
    """Debug Monte Carlo simulation performance issues."""
    import cProfile
    import pstats
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run simulation
    engine = PushBackMonteCarloEngine(create_competitive_robot(), create_default_robot())
    results, _ = engine.run_simulation(1000)
    
    profiler.disable()
    
    # Analyze performance
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions by time

# Memory debugging
def debug_memory_usage():
    """Debug memory usage issues."""
    import tracemalloc
    
    tracemalloc.start()
    
    # Run memory-intensive operation
    engine = PushBackMonteCarloEngine(create_competitive_robot(), create_default_robot())
    results, _ = engine.run_simulation(5000)
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
    
    tracemalloc.stop()
```

### Development Tools

```bash
# Useful development commands
python -m pdb your_script.py           # Python debugger
python -m cProfile your_script.py      # Performance profiling
python -m memory_profiler your_script.py  # Memory profiling

# Frontend debugging
npm run analyze                         # Bundle analysis
npm run lint                           # Linting
npm run type-check                     # TypeScript checking
```

## 🤝 Contributing Guidelines

### Pull Request Process

1. **Fork the repository** and create a feature branch
2. **Follow coding standards** and include comprehensive tests
3. **Update documentation** for any new features
4. **Run the full test suite** and ensure all tests pass
5. **Submit pull request** with clear description

### Code Review Checklist

- [ ] Code follows project style guidelines
- [ ] All tests pass (including new tests for new features)
- [ ] Documentation is updated
- [ ] Performance benchmarks still meet requirements
- [ ] No breaking changes without proper migration path
- [ ] Error handling is comprehensive
- [ ] Security considerations are addressed

### Release Process

```bash
# 1. Update version numbers
python scripts/update-version.py 1.2.0

# 2. Run comprehensive tests
python tests/run_push_back_tests.py

# 3. Build production release
python scripts/build-release.py

# 4. Create release tag
git tag -a v1.2.0 -m "Release version 1.2.0"
git push origin v1.2.0

# 5. Deploy to production
python scripts/deploy-production.py
```

This development guide provides the foundation for contributing to and extending the Push Back Analysis System while maintaining code quality and performance standards.