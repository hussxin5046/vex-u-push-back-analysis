# Testing Framework

The Push Back Analysis System includes a comprehensive test suite specifically designed to validate Push Back game mechanics, strategic analysis accuracy, and system performance. The testing framework ensures reliability for competitive use.

## 🎯 Test Suite Overview

The testing framework consists of multiple specialized test modules:

- **Core Integration Tests** - System-wide functionality validation
- **Monte Carlo Engine Tests** - Simulation accuracy and performance
- **API Integration Tests** - Endpoint functionality and WebSocket testing
- **Push Back Rule Compliance** - Game mechanic accuracy validation
- **Performance Benchmarks** - Speed and resource usage validation

## 🏃‍♂️ Running Tests

### Quick Test Run

```bash
# Run all tests with the master test runner
cd packages/vex-analysis
python3 tests/run_push_back_tests.py
```

Expected output:
```
======================================================================
                 PUSH BACK ANALYSIS SYSTEM TEST SUITE                 
======================================================================

UNIT TESTS
▶ Monte Carlo Simulation Engine
✅ PASSED - 1.74s
   ============================== 6 passed in 0.69s ===============================

INTEGRATION TESTS  
▶ Core Integration Tests
✅ PASSED - 2.31s
   ============================== 11 passed in 0.35s ===============================

PERFORMANCE BENCHMARKS
▶ Performance Benchmarks
• Strategy evaluation (<5s requirement)... ✅ PASSED
• Monte Carlo 1000 scenarios (<10s requirement)... ✅ PASSED  
• Frontend responsiveness test... ✅ PASSED

======================================================================
                         TEST SUMMARY REPORT                          
======================================================================

Total Tests Run: 3
Passed: 3
Failed: 0
Total Time: 4.05s
Pass Rate: 100.0%

✅ ALL TESTS PASSED!
The Push Back analysis system is ready for production.
```

### Individual Test Categories  

```bash
# Run specific test categories
python3 -m pytest tests/test_push_back_simple.py -v           # Core functionality
python3 -m pytest tests/push_back_integration_tests.py -v    # Full integration  
python3 -m pytest tests/push_back_api_tests.py -v           # API endpoints
python3 -m pytest tests/test_push_back_rules.py -v          # Game rule compliance
```

## 🧪 Core Integration Tests

### Monte Carlo Engine Validation

The core tests validate the Monte Carlo simulation engine performs correctly:

```python
# Test basic simulation functionality
def test_monte_carlo_basic_functionality():
    red_robot = create_competitive_robot()
    blue_robot = create_default_robot()
    
    engine = PushBackMonteCarloEngine(red_robot, blue_robot)
    results, execution_time = engine.run_simulation(100)
    
    # Validate results
    assert len(results) == 100
    assert execution_time < 5.0
    assert all(r.winner in ['red', 'blue', 'tie'] for r in results)
    assert all(0 <= r.final_score_red <= 400 for r in results)
```

### Robot Configuration Impact

Tests verify that different robot configurations produce expected differences:

```python  
def test_robot_configuration_impact():
    # Fast vs slow robot comparison
    fast_robot = RobotCapabilities(average_cycle_time=3.0, pickup_reliability=0.98)
    slow_robot = RobotCapabilities(average_cycle_time=8.0, pickup_reliability=0.85)
    
    engine = PushBackMonteCarloEngine(fast_robot, slow_robot)
    results, _ = engine.run_simulation(200)
    
    red_wins = sum(1 for r in results if r.winner == 'red')
    red_win_rate = red_wins / len(results)
    
    # Fast robot should win significantly more
    assert red_win_rate > 0.6
```

### Strategic Pattern Validation

Tests ensure strategic patterns work as expected:

```python
def test_parking_strategy_effects():
    never_park = RobotCapabilities(parking_strategy=ParkingStrategy.NEVER)
    early_park = RobotCapabilities(parking_strategy=ParkingStrategy.EARLY)
    
    # Test never park strategy
    engine1 = PushBackMonteCarloEngine(never_park, create_default_robot())
    results1, _ = engine1.run_simulation(100)
    park_rate1 = sum(1 for r in results1 if r.red_parked) / len(results1)
    
    # Test early park strategy
    engine2 = PushBackMonteCarloEngine(early_park, create_default_robot())  
    results2, _ = engine2.run_simulation(100)
    park_rate2 = sum(1 for r in results2 if r.red_parked) / len(results2)
    
    # Parking behavior should differ significantly
    assert park_rate1 < park_rate2
```

## ⚡ Performance Benchmarks

### Simulation Speed Requirements

Critical performance tests ensure the system meets competitive requirements:

```python
def test_performance_requirements():
    """Test 1000+ simulations complete in under 10 seconds"""
    red_robot = create_competitive_robot()
    blue_robot = create_default_robot()
    engine = PushBackMonteCarloEngine(red_robot, blue_robot)
    
    # Benchmark 1000 simulations
    start_time = time.time()
    results, execution_time = engine.run_simulation(1000, use_parallel=True)
    
    # Validate performance requirements
    assert len(results) == 1000
    assert execution_time < 10.0  # Target: <10 seconds
    
    rate = len(results) / execution_time
    assert rate > 100  # Target: >100 simulations/second
    
    print(f"Performance: {rate:.0f} simulations/second")
```

Expected results:
```
Performance: 11,042 simulations/second
```

### Strategy Analysis Speed

Tests ensure strategic analysis completes within competitive timeframes:

```python
def test_strategy_evaluation_under_5_seconds():
    """Ensure complete strategy evaluation completes within 5 seconds"""
    analyzer = PushBackStrategyAnalyzer()
    
    start_time = time.time()
    
    # Run all strategic analyses
    analyzer.analyze_block_flow_optimization(...)
    analyzer.analyze_autonomous_strategy_decision(...)
    analyzer.analyze_goal_priority_strategy(...)
    analyzer.analyze_parking_decision_timing(...)
    analyzer.analyze_offense_defense_balance(...)
    
    execution_time = time.time() - start_time
    assert execution_time < 5.0
```

### Memory Efficiency

Tests validate memory usage remains reasonable for large simulations:

```python
def test_memory_efficiency_large_simulations():
    """Test memory usage with large simulation runs"""
    import psutil
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run large simulation
    engine = PushBackMonteCarloEngine(create_competitive_robot(), create_default_robot())
    results, _ = engine.run_simulation(5000, use_parallel=True)
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    # Should not use excessive memory
    assert memory_increase < 200  # Less than 200MB increase
```

## 🎮 Push Back Rule Compliance Tests

### Field Layout Validation

Tests ensure the field setup matches official Push Back specifications:

```python
def test_field_layout_specifications():
    """Test field layout matches specifications"""
    field = PushBackFieldState()
    
    # Verify 4 goals total
    assert len(field.goals) == 4
    
    # Verify 2 center and 2 long goals
    center_goals = sum(1 for g in field.goals if isinstance(g, CenterGoal))
    long_goals = sum(1 for g in field.goals if isinstance(g, LongGoal))
    
    assert center_goals == 2
    assert long_goals == 2
    
    # Verify 88 blocks available
    assert field.blocks_available == 88
```

### Scoring Rules Accuracy

Critical tests validate all scoring calculations match the game manual exactly:

```python
def test_scoring_values():
    """Test exact scoring values from game manual"""
    engine = PushBackScoringEngine()
    
    # Rule SC1: Each block is worth 3 points
    assert engine.POINTS_PER_BLOCK == 3
    
    # Rule SC2: Control zone worth 6-10 points
    assert engine.CONTROL_ZONE_POINTS_MIN == 6
    assert engine.CONTROL_ZONE_POINTS_MAX == 10
    
    # Rule SC3: Parking worth 8 points (1 robot) or 30 points (2 robots)
    assert engine.PARKING_POINTS_TOUCHED == 8
    assert engine.PARKING_POINTS_COMPLETELY == 30
    
    # Rule SC4: Autonomous win worth 7 points
    assert engine.AUTONOMOUS_WIN_POINTS == 7
```

### Manual Calculation Verification

Tests validate scoring calculations against hand-calculated scenarios:

```python
def test_scoring_rules_exact_match():
    """Test scoring calculations match manual computation exactly"""
    engine = PushBackScoringEngine()
    
    test_cases = [
        {
            "description": "Basic scoring: 5 blocks",
            "setup": lambda f: [f.goals[0].add_block(PushBackBlock("red")) for _ in range(5)],
            "expected_score": 15,  # 5 blocks * 3 points
        },
        {
            "description": "Complex scenario: blocks + control + parking + auto",
            "setup": lambda f: (
                [f.goals[0].add_block(PushBackBlock("red")) for _ in range(10)],  # 30 points
                setattr(f, "zone_control_red_count", 5),    # 8 points control
                setattr(f, "red_robots_parked", 1),         # 8 points parking
                setattr(f, "autonomous_winner", "red")      # 7 points auto
            ),
            "expected_score": 53,  # 30 + 8 + 8 + 7
        }
    ]
    
    for test in test_cases:
        field = PushBackFieldState()
        test["setup"](field)
        
        score, breakdown = engine.calculate_push_back_score(field, "red")
        assert score == test["expected_score"], f"{test['description']}: Expected {test['expected_score']}, got {score}"
```

## 🌐 API Integration Tests

### Endpoint Functionality

Tests validate all API endpoints respond correctly:

```python
def test_analyze_strategy_endpoint(client):
    """Test /api/push-back/analyze endpoint"""
    strategy_data = {
        "robot_capabilities": {
            "cycle_time": 5.0,
            "reliability": 0.95,
            "max_capacity": 2
        },
        "strategy_type": "block_flow_maximizer",
        "opponent_strength": "competitive"
    }
    
    response = client.post('/api/push-back/analyze', json=strategy_data)
    
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "win_probability" in data["analysis"]
    assert 0 <= data["analysis"]["win_probability"] <= 1
```

### Monte Carlo API Performance

Tests ensure Monte Carlo API endpoints meet performance requirements:

```python
def test_monte_carlo_endpoint_performance(client):
    """Test Monte Carlo endpoint completes quickly"""
    simulation_request = {
        "red_robot": {"cycle_time": 5.0, "reliability": 0.95},
        "blue_robot": {"cycle_time": 6.0, "reliability": 0.90},
        "num_simulations": 1000
    }
    
    start_time = time.time()
    response = client.post('/api/push-back/monte-carlo', json=simulation_request)
    execution_time = time.time() - start_time
    
    assert response.status_code == 200
    assert execution_time < 5.0  # API should respond quickly
    
    data = response.get_json()
    assert data["results"]["simulations_run"] == 1000
    assert "win_probability" in data["results"]
```

### WebSocket Real-Time Updates

Tests validate WebSocket functionality for real-time analysis:

```python
def test_websocket_real_time_updates(socketio_app):
    """Test WebSocket real-time updates"""
    app, socketio = socketio_app
    client = SocketIOTestClient(app, socketio)
    
    # Subscribe to updates
    client.emit('subscribe_analysis', {'session_id': 'test123'})
    received = client.get_received()
    
    assert len(received) > 0
    assert received[0]['name'] == 'analysis_update'
    assert 'win_probability' in received[0]['args'][0]
    
    # Test strategy change updates
    client.emit('strategy_change', {'parameter': 'cycle_time', 'value': 4.5})
    
    updates = client.get_received()
    assert len(updates) > 0
    assert 'insights' in updates[0]['args'][0]
```

## 🧩 System Integration Tests

### End-to-End Analysis Flow

Tests validate the complete analysis pipeline works correctly:

```python
def test_full_analysis_flow():
    """Test complete flow from input to insights"""
    # 1. Create robot configuration
    robot_config = {
        "cycle_time": 5.0,
        "reliability": 0.92,
        "max_capacity": 2,
        "parking_capability": True
    }
    
    # 2. Run Monte Carlo analysis
    engine = PushBackMonteCarloEngine(
        create_robot_from_config(robot_config),
        create_default_robot()
    )
    results, execution_time = engine.run_simulation(1000)
    
    # 3. Generate insights
    insights = engine.generate_insights(results, "red")
    
    # 4. Validate complete pipeline
    assert len(results) == 1000
    assert execution_time < 10.0
    assert 0 <= insights.win_probability <= 1
    assert insights.average_score > 0
    assert len(insights.improvement_opportunities) > 0
```

### Concurrent Processing

Tests ensure the system handles multiple simultaneous requests:

```python
def test_concurrent_simulations():
    """Test running multiple simulations concurrently"""
    import concurrent.futures
    
    def run_simulation(robot_pair):
        red, blue = robot_pair
        engine = PushBackMonteCarloEngine(red, blue)
        return engine.run_simulation(100)
    
    # Create different robot pairs
    pairs = [
        (create_competitive_robot(), create_default_robot()),
        (create_default_robot(), create_competitive_robot()),
        (create_default_robot(), create_default_robot())
    ]
    
    # Run concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(run_simulation, pair) for pair in pairs]
        results = [f.result() for f in futures]
    
    # All should complete successfully
    assert len(results) == 3
    for result_set, execution_time in results:
        assert len(result_set) == 100
        assert execution_time < 5.0
```

## 📊 Test Data Validation

### Realistic Score Ranges

Tests ensure simulated scores fall within realistic Push Back ranges:

```python
def test_realistic_scoring_ranges():
    """Test that scores fall within realistic ranges"""
    engine = PushBackMonteCarloEngine(create_default_robot(), create_default_robot())
    results, _ = engine.run_simulation(500)
    
    red_scores = [r.final_score_red for r in results]
    blue_scores = [r.final_score_blue for r in results]
    
    # Check score ranges
    min_red, max_red = min(red_scores), max(red_scores)
    min_blue, max_blue = min(blue_scores), max(blue_scores)
    
    # Scores should be reasonable for Push Back
    assert 20 <= min_red <= 250, f"Red score range {min_red}-{max_red} unrealistic"
    assert 20 <= min_blue <= 250, f"Blue score range {min_blue}-{max_blue} unrealistic"
    
    # Average scores should be in typical range
    avg_red = statistics.mean(red_scores)
    avg_blue = statistics.mean(blue_scores)
    
    assert 40 <= avg_red <= 150, f"Average red score {avg_red:.1f} outside typical range"
    assert 40 <= avg_blue <= 150, f"Average blue score {avg_blue:.1f} outside typical range"
```

### Statistical Validation

Tests validate that statistical results are reasonable:

```python
def test_statistical_validity():
    """Test that statistical results are mathematically valid"""
    engine = PushBackMonteCarloEngine(create_competitive_robot(), create_default_robot())
    results, _ = engine.run_simulation(1000)
    
    insights = engine.generate_insights(results, "red")
    
    # Basic statistical validity
    assert 0 <= insights.win_probability <= 1
    assert insights.score_variance >= 0
    assert insights.average_score > 0
    
    # Win probability should correlate with average performance
    red_wins = sum(1 for r in results if r.winner == "red")
    actual_win_rate = red_wins / len(results)
    
    # Should be within reasonable margin of insights calculation
    assert abs(insights.win_probability - actual_win_rate) < 0.02
```

## 🔧 Custom Test Configuration

### Running Specific Test Suites

```bash
# Run only performance tests
python3 -m pytest tests/ -k "performance" -v

# Run only Push Back rule compliance tests  
python3 -m pytest tests/ -k "rule" -v

# Run tests with specific markers
python3 -m pytest tests/ -m "slow" -v      # Long-running tests
python3 -m pytest tests/ -m "integration" -v  # Integration tests only
```

### Test Configuration

Create custom test configurations:

```python
# pytest.ini configuration
[tool:pytest]
markers = 
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    performance: marks tests as performance benchmarks
    api: marks tests as API tests

# Custom test configuration
TEST_CONFIG = {
    "simulation_sizes": [100, 500, 1000],  # Test different simulation sizes
    "performance_thresholds": {
        "monte_carlo_rate": 1000,          # Minimum simulations/second
        "strategy_analysis_time": 5.0,     # Maximum seconds for analysis
        "api_response_time": 0.5           # Maximum API response time
    },
    "validation_ranges": {
        "min_score": 20,                   # Minimum realistic score
        "max_score": 300,                  # Maximum realistic score
        "typical_score_range": (60, 150)   # Typical score range
    }
}
```

### Creating Custom Tests

Add custom tests for specific scenarios:

```python
def test_custom_robot_configuration():
    """Test specific robot configuration for your team"""
    # Define your robot's actual measured performance
    my_robot = RobotCapabilities(
        average_cycle_time=4.8,      # Your measured cycle time
        pickup_reliability=0.93,     # Your measured pickup success
        scoring_reliability=0.96,    # Your measured scoring success
        parking_strategy=ParkingStrategy.LATE
    )
    
    # Test against expected competition
    competition_robot = create_competitive_robot()
    
    engine = PushBackMonteCarloEngine(my_robot, competition_robot)
    results, _ = engine.run_simulation(2000)
    
    insights = engine.generate_insights(results, "red")
    
    # Validate performance meets your expectations
    assert insights.win_probability >= 0.6, f"Win rate {insights.win_probability:.1%} below target"
    assert insights.average_score >= 100, f"Average score {insights.average_score:.1f} below target"
    
    print(f"Your Robot Performance:")
    print(f"  Win Rate: {insights.win_probability:.1%}")
    print(f"  Expected Score: {insights.average_score:.1f}")
    print(f"  Key Recommendations: {insights.improvement_opportunities[:2]}")
```

## 📈 Continuous Integration

### Automated Test Execution

Set up automated testing for continuous validation:

```bash
# GitHub Actions workflow example
name: Push Back Analysis Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        cd packages/vex-analysis && pip install -e .
    
    - name: Run test suite
      run: |
        cd packages/vex-analysis
        python tests/run_push_back_tests.py
    
    - name: Performance benchmark
      run: |
        python -c "
        from vex_analysis.simulation import *
        engine = PushBackMonteCarloEngine(create_competitive_robot(), create_default_robot())
        results, time = engine.run_simulation(1000)
        rate = len(results) / time
        assert rate > 1000, f'Performance {rate:.0f} sim/sec below threshold'
        print(f'✅ Performance: {rate:.0f} simulations/second')
        "
```

The testing framework ensures the Push Back Analysis System maintains accuracy, performance, and reliability for competitive use. All tests are designed to validate the system against real Push Back requirements and provide confidence for strategic decision-making.