"""
Push Back Monte Carlo Simulation Engine

This module provides a high-performance Monte Carlo simulation system specifically
designed for VEX U Push Back strategic analysis. The system features:

- Realistic robot performance modeling
- Push Back-specific game mechanics simulation  
- Fast execution (1000+ simulations in <10 seconds)
- Strategic insight generation
- Comprehensive scenario generation

Key Classes:
- PushBackMonteCarloEngine: Core simulation engine
- RobotCapabilities: Robot performance configuration
- PushBackScenarioGenerator: Scenario creation and variation
- PushBackInsightEngine: Strategic analysis and recommendations

Usage Example:
    from vex_analysis.simulation import (
        PushBackMonteCarloEngine, 
        create_competitive_robot,
        create_default_robot
    )
    
    # Create robots
    red_robot = create_competitive_robot()
    blue_robot = create_default_robot()
    
    # Run simulation
    engine = PushBackMonteCarloEngine(red_robot, blue_robot)
    results, execution_time = engine.run_simulation(1000)
    
    # Generate insights
    insights = engine.generate_insights(results, "red")
    print(f"Win probability: {insights.win_probability:.1%}")
"""

from .push_back_monte_carlo import (
    PushBackMonteCarloEngine,
    RobotCapabilities, 
    SimulationResult,
    StrategyInsights,
    ParkingStrategy,
    GoalPriority, 
    AutonomousStrategy,
    create_default_robot,
    create_competitive_robot,
    create_beginner_robot
)

from .push_back_scenarios import (
    PushBackScenarioGenerator,
    ScenarioConfig,
    TeamSkillLevel,
    MatchType,
    FieldCondition,
    create_scouting_scenarios,
    create_elimination_scenarios
)

from .push_back_insights import (
    PushBackInsightEngine,
    StrategicInsight,
    CompetitiveAnalysis,
    PredictiveModel,
    InsightType,
    ConfidenceLevel,
    format_insights_for_display
)

__all__ = [
    # Core simulation
    'PushBackMonteCarloEngine',
    'RobotCapabilities',
    'SimulationResult', 
    'StrategyInsights',
    
    # Robot configuration enums
    'ParkingStrategy',
    'GoalPriority',
    'AutonomousStrategy',
    
    # Robot factory functions
    'create_default_robot',
    'create_competitive_robot', 
    'create_beginner_robot',
    
    # Scenario generation
    'PushBackScenarioGenerator',
    'ScenarioConfig',
    'TeamSkillLevel',
    'MatchType',
    'FieldCondition',
    'create_scouting_scenarios',
    'create_elimination_scenarios',
    
    # Strategic insights
    'PushBackInsightEngine',
    'StrategicInsight',
    'CompetitiveAnalysis',
    'PredictiveModel',
    'InsightType',
    'ConfidenceLevel',
    'format_insights_for_display'
]

# Version information
__version__ = '1.0.0'
__author__ = 'VEX Analysis Team'
__description__ = 'Push Back Monte Carlo Simulation Engine'