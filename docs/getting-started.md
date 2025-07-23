# Getting Started

This guide will help you set up and run the VEX U Push Back Analysis System on your local machine.

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher**
- **Node.js 16 or higher**
- **Git** for version control
- **4GB+ RAM** (8GB recommended for large simulations)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/hussxin5046/vex-u-push-back-analysis.git
cd vex-u-push-back-analysis
```

### 2. Set Up Python Environment

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install the VEX analysis package in development mode
cd packages/vex-analysis
pip install -e .
cd ../..
```

### 3. Set Up Frontend Dependencies

```bash
# Install Node.js dependencies
cd apps/frontend
npm install
cd ../..
```

### 4. Verify Installation

```bash
# Test the Monte Carlo engine
python -c "
from vex_analysis.simulation import PushBackMonteCarloEngine, create_default_robot
engine = PushBackMonteCarloEngine(create_default_robot(), create_default_robot())
results, time = engine.run_simulation(100)
print(f'✅ Monte Carlo test: {len(results)} simulations in {time:.3f}s')
"

# Run the test suite
cd packages/vex-analysis
python tests/run_push_back_tests.py
```

Expected output:
```
✅ Monte Carlo test: 100 simulations in 0.009s
✅ ALL TESTS PASSED!
The Push Back analysis system is ready for production.
```

## 🏃‍♂️ Running the System

### Development Mode

Start both the frontend and backend in development mode:

```bash
# Terminal 1: Start the backend
cd apps/backend
python app.py

# Terminal 2: Start the frontend
cd apps/frontend
npm start
```

Access the application:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **API Documentation**: http://localhost:5000/docs

### Production Mode

```bash
# Build the frontend
cd apps/frontend
npm run build

# Start the production server
cd ../backend
python app.py --production
```

## 🎯 First Analysis

Let's run your first Push Back strategic analysis:

### 1. Quick Monte Carlo Simulation

```python
from vex_analysis.simulation import (
    PushBackMonteCarloEngine, 
    create_competitive_robot, 
    create_default_robot
)

# Create robot configurations
competitive = create_competitive_robot()
default = create_default_robot()

# Run simulation comparing the robots
engine = PushBackMonteCarloEngine(competitive, default)
results, execution_time = engine.run_simulation(1000)

# Analyze results
red_wins = sum(1 for r in results if r.winner == "red")
win_rate = red_wins / len(results)

print(f"Competitive Robot Performance:")
print(f"  Win Rate: {win_rate:.1%}")
print(f"  Simulations: {len(results)}")
print(f"  Execution Time: {execution_time:.3f}s")
print(f"  Simulation Rate: {len(results)/execution_time:.0f} sim/sec")
```

### 2. Generate Strategic Insights

```python
# Generate detailed insights
insights = engine.generate_insights(results, "red")

print(f"\nStrategic Insights:")
print(f"  Expected Score: {insights.average_score:.1f}")
print(f"  Score Variance: {insights.score_variance:.1f}")
print(f"  Confidence Level: High")

# Get recommendations
recommendations = insights.improvement_opportunities
for i, rec in enumerate(recommendations[:3], 1):
    print(f"  {i}. {rec}")
```

### 3. Scenario Analysis

```python
from vex_analysis.simulation import PushBackScenarioGenerator

# Generate different competition scenarios
generator = PushBackScenarioGenerator()

# Test against various opponent types
scenarios = [
    "mirror_match",
    "david_vs_goliath", 
    "elimination_pressure",
    "early_season"
]

for scenario_type in scenarios:
    red_robot, blue_robot, metadata = generator.generate_scenario(scenario_type)
    
    engine = PushBackMonteCarloEngine(red_robot, blue_robot)
    results, _ = engine.run_simulation(200)
    
    win_rate = sum(1 for r in results if r.winner == "red") / len(results)
    print(f"{scenario_type.replace('_', ' ').title()}: {win_rate:.1%} win rate")
```

## 🎮 Using the Web Interface

### Strategy Builder

1. **Navigate to Strategy Builder** (http://localhost:3000/strategy-builder)
2. **Configure your robot**:
   - Set cycle times (3-8 seconds typical)
   - Set reliability percentages (90-98% typical)
   - Choose parking strategy
   - Select goal priorities

3. **Analyze performance**:
   - Click "Run Analysis" 
   - View win probability and expected scores
   - Review strategic recommendations

### Quick Analysis Dashboard

1. **Access Quick Analysis** (http://localhost:3000/analysis)
2. **Input match parameters**:
   - Current score differential
   - Time remaining
   - Robot positions
   - Opponent strength estimate

3. **Get real-time recommendations**:
   - Parking timing decisions
   - Goal priority adjustments
   - Risk/reward analysis

## 🔧 Configuration

### Robot Performance Tuning

Create custom robot configurations:

```python
from vex_analysis.simulation import RobotCapabilities, ParkingStrategy, GoalPriority

# Define your robot's actual performance
my_robot = RobotCapabilities(
    average_cycle_time=4.5,        # Seconds per scoring cycle
    pickup_reliability=0.94,       # Success rate picking up blocks
    scoring_reliability=0.97,      # Success rate scoring blocks
    autonomous_reliability=0.88,   # Autonomous routine success rate
    parking_strategy=ParkingStrategy.LATE,
    goal_priority=GoalPriority.CENTER_PREFERRED,
    max_blocks_per_trip=2,         # Blocks carried simultaneously
    control_zone_frequency=0.4     # How often to prioritize control
)
```

### Simulation Parameters

Adjust simulation settings for your needs:

```python
engine = PushBackMonteCarloEngine(red_robot, blue_robot)

# Quick analysis (development)
results, _ = engine.run_simulation(100, use_parallel=False)

# Comprehensive analysis (competition prep)
results, _ = engine.run_simulation(5000, use_parallel=True)

# Real-time analysis (during matches)
results, _ = engine.run_simulation(500, use_parallel=True)
```

## 🐛 Troubleshooting

### Common Issues

**Import Error: Cannot import vex_analysis**
```bash
# Ensure the package is installed in development mode
cd packages/vex-analysis
pip install -e .
```

**Performance Issues**
```bash
# Check Python version (3.8+ required)
python --version

# Verify NumPy installation
python -c "import numpy; print(numpy.__version__)"

# Test with smaller simulation sizes first
python -c "from vex_analysis.simulation import *; print('Import successful')"
```

**Frontend Won't Start**
```bash
# Clear Node.js cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Getting Help

- **GitHub Issues**: https://github.com/hussxin5046/vex-u-push-back-analysis/issues
- **Documentation**: Check the `/docs` folder for detailed guides
- **Test Suite**: Run `python tests/run_push_back_tests.py` to diagnose issues

## ✅ Validation Checklist

Before using the system for competition analysis:

- [ ] All tests pass (`python tests/run_push_back_tests.py`)
- [ ] Monte Carlo achieves 1000+ simulations in <10 seconds
- [ ] Web interface loads and responds quickly
- [ ] Custom robot configurations work as expected
- [ ] Analysis results are reasonable (win rates 20-80%, scores 50-200)

## 🎯 Next Steps

- **[Monte Carlo Engine](monte-carlo-engine.md)** - Deep dive into simulation capabilities
- **[Strategy Analysis](strategy-analysis.md)** - Advanced analysis techniques
- **[API Reference](api-reference.md)** - Integration and automation
- **[Configuration](configuration.md)** - Customization and optimization

You're now ready to use the Push Back Analysis System for strategic development!