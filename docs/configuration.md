# Configuration and Customization

This guide covers how to configure and customize the Push Back Analysis System for your specific team needs, robot capabilities, and competition requirements.

## 🎯 Overview

The Push Back Analysis System provides extensive configuration options to match your team's actual robot performance and strategic preferences. Proper configuration ensures accurate analysis results that reflect your competitive reality.

## 🤖 Robot Configuration

### Basic Robot Setup

Configure your robot's measured performance characteristics:

```python
from vex_analysis.simulation import RobotCapabilities, ParkingStrategy, GoalPriority

# Define your robot's actual performance
my_robot = RobotCapabilities(
    # Timing Performance (measured from practice)
    min_cycle_time=3.2,           # Fastest observed cycle
    max_cycle_time=7.8,           # Slowest observed cycle
    average_cycle_time=4.9,       # Typical cycle time
    
    # Movement Characteristics
    max_speed=4.2,                # Maximum field traversal speed
    average_speed=2.8,            # Typical movement speed
    
    # Reliability Metrics (measure over 20+ attempts)
    pickup_reliability=0.93,      # Success rate picking up blocks
    scoring_reliability=0.97,     # Success rate scoring blocks
    autonomous_reliability=0.85,  # Autonomous routine success rate
    
    # Strategic Preferences
    parking_strategy=ParkingStrategy.LATE,          # When to park
    goal_priority=GoalPriority.CENTER_PREFERRED,    # Goal targeting preference
    autonomous_strategy=AutonomousStrategy.BALANCED, # Autonomous aggression level
    
    # Physical Capabilities
    max_blocks_per_trip=2,        # Maximum blocks carried simultaneously
    prefers_singles=False,        # Prefers single vs multiple blocks
    
    # Advanced Behavior
    control_zone_frequency=0.35,  # How often to prioritize control zones
    control_zone_duration=4.5     # Seconds spent in control zones
)
```

### Measuring Robot Performance

Use these techniques to accurately measure your robot's capabilities:

#### Cycle Time Measurement

```python
import time

def measure_cycle_times(num_trials=20):
    """Measure robot cycle times during practice."""
    cycle_times = []
    
    print("Measure cycle times - start timing when robot begins pickup")
    print("Stop timing when block is scored and robot returns to neutral")
    
    for i in range(num_trials):
        input(f"Trial {i+1}/{num_trials} - Press Enter to start timing...")
        start_time = time.time()
        
        input("Press Enter when cycle complete...")
        end_time = time.time()
        
        cycle_time = end_time - start_time
        cycle_times.append(cycle_time)
        print(f"  Cycle {i+1}: {cycle_time:.2f}s")
    
    # Calculate statistics
    avg_time = sum(cycle_times) / len(cycle_times)
    min_time = min(cycle_times)
    max_time = max(cycle_times)
    
    print(f"\nCycle Time Analysis:")
    print(f"  Average: {avg_time:.2f}s")
    print(f"  Range: {min_time:.2f}s - {max_time:.2f}s")
    print(f"  Standard Deviation: {statistics.stdev(cycle_times):.2f}s")
    
    return {
        "average_cycle_time": avg_time,
        "min_cycle_time": min_time,
        "max_cycle_time": max_time
    }

# Usage during practice
cycle_stats = measure_cycle_times(15)
```

#### Reliability Measurement

```python
def measure_reliability(task_type="pickup", num_trials=25):
    """Measure task reliability over multiple attempts."""
    successes = 0
    
    print(f"Measuring {task_type} reliability over {num_trials} trials")
    
    for i in range(num_trials):
        success = input(f"Trial {i+1}: Was {task_type} successful? (y/n): ").lower() == 'y'
        if success:
            successes += 1
        
        current_rate = successes / (i + 1)
        print(f"  Current reliability: {current_rate:.1%}")
    
    final_reliability = successes / num_trials
    print(f"\nFinal {task_type} reliability: {final_reliability:.1%}")
    
    return final_reliability

# Measure different aspects
pickup_reliability = measure_reliability("pickup", 25)
scoring_reliability = measure_reliability("scoring", 25)
autonomous_reliability = measure_reliability("autonomous", 15)
```

### Dynamic Robot Configuration

Adjust robot configuration based on match conditions:

```python
def create_adaptive_robot_config(base_robot, match_conditions):
    """Create robot config adapted to specific match conditions."""
    
    adapted_robot = base_robot.copy()
    
    # Adjust for field conditions
    if match_conditions.get("field_condition") == "worn":
        adapted_robot.pickup_reliability *= 0.95
        adapted_robot.average_cycle_time *= 1.05
    
    # Adjust for pressure/elimination matches
    if match_conditions.get("match_type") == "elimination":
        adapted_robot.autonomous_reliability *= 0.92  # Increased pressure
        adapted_robot.parking_strategy = ParkingStrategy.EARLY  # More conservative
    
    # Adjust for opponent strength
    opponent_strength = match_conditions.get("opponent_strength", "average")
    if opponent_strength == "strong":
        adapted_robot.control_zone_frequency *= 0.8  # Focus more on scoring
    elif opponent_strength == "weak":
        adapted_robot.control_zone_frequency *= 1.2  # More zone control
    
    # Time of day adjustments (driver fatigue)
    time_of_day = match_conditions.get("time_of_day")
    if time_of_day == "late":
        adapted_robot.pickup_reliability *= 0.97
        adapted_robot.scoring_reliability *= 0.98
    
    return adapted_robot

# Usage
match_conditions = {
    "field_condition": "worn",
    "match_type": "elimination",
    "opponent_strength": "strong",
    "time_of_day": "afternoon"
}

adapted_robot = create_adaptive_robot_config(my_robot, match_conditions)
```

## ⚙️ Analysis Configuration

### Simulation Parameters

Configure Monte Carlo simulation parameters for different use cases:

```python
# Quick analysis configuration (development/rapid testing)
QUICK_CONFIG = {
    "num_simulations": 200,
    "use_parallel": False,
    "scenario_variation": 0.1,
    "timeout": 10  # seconds
}

# Standard analysis configuration (strategy evaluation)
STANDARD_CONFIG = {
    "num_simulations": 2000,
    "use_parallel": True, 
    "scenario_variation": 0.2,
    "timeout": 30
}

# Comprehensive analysis configuration (competition preparation)
COMPREHENSIVE_CONFIG = {
    "num_simulations": 5000,
    "use_parallel": True,
    "scenario_variation": 0.3,
    "timeout": 60,
    "include_detailed_insights": True
}

# Real-time analysis configuration (during matches)
REALTIME_CONFIG = {
    "num_simulations": 500,
    "use_parallel": True,
    "scenario_variation": 0.15,
    "timeout": 5,
    "prioritize_speed": True
}

# Usage
from vex_analysis.simulation import PushBackMonteCarloEngine

def run_analysis(config_type="standard"):
    """Run analysis with specified configuration."""
    config = {
        "quick": QUICK_CONFIG,
        "standard": STANDARD_CONFIG,
        "comprehensive": COMPREHENSIVE_CONFIG,
        "realtime": REALTIME_CONFIG
    }[config_type]
    
    engine = PushBackMonteCarloEngine(my_robot, opponent_robot)
    results, execution_time = engine.run_simulation(
        num_simulations=config["num_simulations"],
        use_parallel=config["use_parallel"]
    )
    
    print(f"Analysis complete: {len(results)} simulations in {execution_time:.2f}s")
    return results
```

### Strategy Analysis Tuning

Configure the strategy analysis framework for your team's needs:

```python
from vex_analysis.analysis import PushBackStrategyAnalyzer

# Configure analyzer with team-specific parameters
analyzer = PushBackStrategyAnalyzer(
    # Risk tolerance (0.0 = very conservative, 1.0 = very aggressive)
    default_risk_tolerance=0.6,
    
    # Analysis depth
    monte_carlo_samples=2000,
    confidence_threshold=0.8,
    
    # Strategic preferences
    prioritize_consistency=True,
    include_contingency_plans=True,
    
    # Opponent modeling
    default_opponent_strength="competitive",
    opponent_adaptation_factor=0.3
)

# Configure analysis for specific competition levels
def configure_for_competition(competition_level):
    """Configure analysis parameters for competition level."""
    
    configs = {
        "regional": {
            "risk_tolerance": 0.7,
            "opponent_strength": "mixed",
            "monte_carlo_samples": 1500,
            "prioritize_consistency": False
        },
        "state": {
            "risk_tolerance": 0.5,
            "opponent_strength": "strong", 
            "monte_carlo_samples": 3000,
            "prioritize_consistency": True
        },
        "worlds": {
            "risk_tolerance": 0.4,
            "opponent_strength": "elite",
            "monte_carlo_samples": 5000,
            "prioritize_consistency": True
        }
    }
    
    config = configs.get(competition_level, configs["regional"])
    
    return PushBackStrategyAnalyzer(**config)

# Usage
worlds_analyzer = configure_for_competition("worlds")
```

## 🎮 Competition-Specific Configuration

### Tournament Preparation

Configure the system for different tournament phases:

```python
class TournamentConfig:
    """Configuration for different tournament phases."""
    
    def __init__(self, tournament_type, day, match_number):
        self.tournament_type = tournament_type
        self.day = day
        self.match_number = match_number
        
        # Base configuration
        self.robot_config = my_robot.copy()
        self.analysis_config = STANDARD_CONFIG.copy()
        
        # Apply tournament-specific adjustments
        self._apply_tournament_adjustments()
    
    def _apply_tournament_adjustments(self):
        """Apply adjustments based on tournament phase."""
        
        # Day-based adjustments (driver fatigue, robot wear)
        if self.day >= 2:
            self.robot_config.pickup_reliability *= 0.98
            self.robot_config.average_cycle_time *= 1.02
        
        # Match number adjustments
        if self.match_number > 8:  # Late in the day
            self.robot_config.autonomous_reliability *= 0.95
        
        # Tournament type adjustments
        if self.tournament_type == "elimination":
            self.robot_config.parking_strategy = ParkingStrategy.LATE
            self.analysis_config["num_simulations"] = 3000  # More thorough analysis
        
        # Qualification vs elimination strategy
        if self.tournament_type == "qualification":
            self.robot_config.risk_tolerance = 0.7  # More aggressive
        else:
            self.robot_config.risk_tolerance = 0.5  # More conservative

# Usage for different tournament situations
qual_day1_match3 = TournamentConfig("qualification", 1, 3)
elim_day2_semifinal = TournamentConfig("elimination", 2, "semifinal")
```

### Alliance Partner Integration

Configure for alliance partner compatibility:

```python
def configure_for_alliance_partner(partner_robot_estimate, partner_strategy):
    """Configure robot behavior for alliance partner."""
    
    alliance_robot = my_robot.copy()
    
    # Adjust based on partner capabilities
    if partner_robot_estimate.get("strength") == "weak":
        # Take on more responsibility
        alliance_robot.control_zone_frequency *= 1.3
        alliance_robot.parking_strategy = ParkingStrategy.LATE  # Ensure someone parks
    
    elif partner_robot_estimate.get("strength") == "strong":
        # Complement partner's strengths
        if partner_strategy.get("focus") == "scoring":
            alliance_robot.control_zone_frequency *= 1.5  # Focus on control
        elif partner_strategy.get("focus") == "control":
            alliance_robot.control_zone_frequency *= 0.7  # Focus on scoring
    
    # Coordination strategies
    if partner_strategy.get("parking") == "never":
        alliance_robot.parking_strategy = ParkingStrategy.EARLY  # Ensure parking coverage
    
    return alliance_robot

# Example alliance configurations
partner_estimate = {
    "strength": "strong",
    "cycle_time": 4.2,
    "reliability": 0.94,
    "specialization": "center_goals"
}

partner_strategy = {
    "focus": "scoring",
    "parking": "late",
    "autonomous": "aggressive"
}

alliance_configured_robot = configure_for_alliance_partner(partner_estimate, partner_strategy)
```

## 🔧 System Performance Configuration

### Hardware Optimization

Configure the system based on available hardware:

```python
import multiprocessing
import psutil

def auto_configure_performance():
    """Automatically configure performance based on system capabilities."""
    
    # CPU configuration
    cpu_count = multiprocessing.cpu_count()
    available_memory_gb = psutil.virtual_memory().total / (1024**3)
    
    config = {}
    
    # Parallel processing configuration
    if cpu_count >= 8:
        config["max_workers"] = min(cpu_count - 2, 12)
        config["use_parallel_default"] = True
    elif cpu_count >= 4:
        config["max_workers"] = cpu_count - 1
        config["use_parallel_default"] = True
    else:
        config["max_workers"] = 1
        config["use_parallel_default"] = False
    
    # Memory-based simulation size limits
    if available_memory_gb >= 16:
        config["max_simulation_size"] = 10000
        config["default_large_size"] = 5000
    elif available_memory_gb >= 8:
        config["max_simulation_size"] = 5000
        config["default_large_size"] = 2000
    else:
        config["max_simulation_size"] = 2000
        config["default_large_size"] = 1000
    
    # Timeout configuration
    config["default_timeout"] = 30 if cpu_count >= 4 else 60
    config["max_timeout"] = 120
    
    return config

# Apply auto-configuration
performance_config = auto_configure_performance()
```

### API Configuration

Configure API endpoints for different deployment scenarios:

```python
# apps/backend/config.py
class APIConfig:
    """API-specific configuration."""
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE = 100
    BURST_RATE_LIMIT = 10
    
    # Request size limits
    MAX_REQUEST_SIZE = 16 * 1024 * 1024  # 16MB
    MAX_SIMULATION_SIZE = 10000
    
    # Timeout configuration
    DEFAULT_TIMEOUT = 30
    MAX_TIMEOUT = 120
    
    # Cache configuration
    CACHE_TTL = 300  # 5 minutes
    MAX_CACHE_SIZE = 1000
    
    # Monitoring
    ENABLE_PERFORMANCE_LOGGING = True
    LOG_SLOW_REQUESTS = True
    SLOW_REQUEST_THRESHOLD = 2.0  # seconds

class ProductionAPIConfig(APIConfig):
    """Production API configuration."""
    
    RATE_LIMIT_PER_MINUTE = 200
    MAX_SIMULATION_SIZE = 5000
    CACHE_TTL = 600  # 10 minutes
    
class DevelopmentAPIConfig(APIConfig):
    """Development API configuration."""
    
    RATE_LIMIT_PER_MINUTE = 1000  # No limiting in dev
    MAX_SIMULATION_SIZE = 2000
    CACHE_TTL = 60  # 1 minute
    ENABLE_PERFORMANCE_LOGGING = False
```

## 📊 Visualization Configuration

### Chart and Dashboard Settings

Configure visualization components for your preferences:

```typescript
// apps/frontend/src/config/visualization.ts
export interface VisualizationConfig {
  colorScheme: 'red_blue' | 'team_colors' | 'colorblind_friendly';
  chartTheme: 'light' | 'dark' | 'auto';
  animationDuration: number;
  showConfidenceIntervals: boolean;
  defaultTimeRange: number;
  updateFrequency: number;
}

export const DEFAULT_VIZ_CONFIG: VisualizationConfig = {
  colorScheme: 'red_blue',
  chartTheme: 'auto',
  animationDuration: 300,
  showConfidenceIntervals: true,
  defaultTimeRange: 105, // Full match length
  updateFrequency: 1000  // 1 second
};

// Team-specific color schemes
export const COLOR_SCHEMES = {
  red_blue: {
    primary: '#ff4444',
    secondary: '#4444ff',
    background: '#f8f9fa'
  },
  team_colors: {
    primary: '#ffaa00',    // Your team's primary color
    secondary: '#aa00ff',  // Your team's secondary color
    background: '#ffffff'
  },
  colorblind_friendly: {
    primary: '#0173b2',
    secondary: '#de8f05',
    background: '#f0f0f0'
  }
};

// Configure for your team
export function createTeamVizConfig(teamColors: string[]): VisualizationConfig {
  return {
    ...DEFAULT_VIZ_CONFIG,
    colorScheme: 'team_colors'
  };
}
```

### Real-Time Display Configuration

```typescript
// Real-time analysis display configuration
export interface RealTimeConfig {
  updateInterval: number;
  historyLength: number;
  alertThresholds: {
    lowWinProbability: number;
    highScoreVariance: number;
    slowPerformance: number;
  };
  displayPrecision: {
    winProbability: number;
    scores: number;
    timing: number;
  };
}

export const MATCH_DISPLAY_CONFIG: RealTimeConfig = {
  updateInterval: 500,    // 0.5 seconds
  historyLength: 100,     // Keep 100 data points
  alertThresholds: {
    lowWinProbability: 0.3,
    highScoreVariance: 400,
    slowPerformance: 10.0  // seconds
  },
  displayPrecision: {
    winProbability: 1,     // 1 decimal place (72.5%)
    scores: 1,             // 1 decimal place (115.3)
    timing: 1              // 1 decimal place (14.2s)
  }
};

export const STRATEGY_DISPLAY_CONFIG: RealTimeConfig = {
  updateInterval: 2000,   // 2 seconds (less frequent for strategy)
  historyLength: 50,
  alertThresholds: {
    lowWinProbability: 0.4,
    highScoreVariance: 300,
    slowPerformance: 5.0
  },
  displayPrecision: {
    winProbability: 1,
    scores: 0,             // Whole numbers for strategy view
    timing: 0
  }
};
```

## 🔒 Security and Privacy Configuration

### Data Protection Settings

```python
# Security configuration
class SecurityConfig:
    """Security and privacy configuration."""
    
    # Data retention
    ANALYSIS_HISTORY_RETENTION_DAYS = 90
    LOG_RETENTION_DAYS = 30
    CACHE_RETENTION_HOURS = 24
    
    # Privacy settings
    ANONYMIZE_ROBOT_DATA = True
    SHARE_ANONYMOUS_STATS = False  # For system improvement
    STORE_MATCH_VIDEOS = False
    
    # Access control
    REQUIRE_TEAM_AUTHENTICATION = True
    SESSION_TIMEOUT_MINUTES = 60
    MAX_CONCURRENT_SESSIONS = 3
    
    # Data encryption
    ENCRYPT_STORED_CONFIGURATIONS = True
    ENCRYPT_ANALYSIS_RESULTS = True
    
    # Audit logging
    LOG_CONFIGURATION_CHANGES = True
    LOG_ANALYSIS_REQUESTS = True
    LOG_EXPORT_REQUESTS = True

# Team-specific privacy settings
def configure_team_privacy(team_preferences):
    """Configure privacy settings based on team preferences."""
    
    config = SecurityConfig()
    
    if team_preferences.get("share_for_research"):
        config.SHARE_ANONYMOUS_STATS = True
    
    if team_preferences.get("extended_history"):
        config.ANALYSIS_HISTORY_RETENTION_DAYS = 365
    
    if team_preferences.get("public_mode"):
        config.ANONYMIZE_ROBOT_DATA = True
        config.ENCRYPT_STORED_CONFIGURATIONS = False
    
    return config
```

## 📁 Configuration File Management

### Configuration File Structure

```yaml
# config/team_config.yaml
team:
  name: "Team 12345A"
  region: "California"
  competition_level: "worlds"

robot:
  name: "PushBack Champion v2"
  measurements:
    cycle_time:
      average: 4.8
      min: 3.2
      max: 7.1
    reliability:
      pickup: 0.94
      scoring: 0.97
      autonomous: 0.87
  strategy:
    parking: "late"
    goal_priority: "center_preferred"
    autonomous: "balanced"
    risk_tolerance: 0.6

analysis:
  default_simulations: 2000
  use_parallel: true
  confidence_threshold: 0.85
  include_insights: true

competition:
  upcoming_events:
    - name: "Regional Championship"
      date: "2024-02-15"
      expected_opponents: ["strong", "mixed"]
    - name: "State Tournament"
      date: "2024-03-05"
      expected_opponents: ["elite", "strong"]

visualization:
  color_scheme: "team_colors"
  team_colors: ["#ff6600", "#0066ff"] 
  chart_theme: "dark"
  show_confidence_intervals: true
```

### Loading Configuration

```python
import yaml
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class TeamConfiguration:
    """Complete team configuration."""
    team_name: str
    robot_config: RobotCapabilities
    analysis_preferences: Dict
    competition_schedule: List[Dict]
    visualization_settings: Dict

def load_team_config(config_file: str) -> TeamConfiguration:
    """Load team configuration from YAML file."""
    
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Create robot configuration
    robot_measurements = config_data['robot']['measurements']
    robot_strategy = config_data['robot']['strategy']
    
    robot_config = RobotCapabilities(
        average_cycle_time=robot_measurements['cycle_time']['average'],
        min_cycle_time=robot_measurements['cycle_time']['min'],
        max_cycle_time=robot_measurements['cycle_time']['max'],
        pickup_reliability=robot_measurements['reliability']['pickup'],
        scoring_reliability=robot_measurements['reliability']['scoring'],
        autonomous_reliability=robot_measurements['reliability']['autonomous'],
        parking_strategy=ParkingStrategy[robot_strategy['parking'].upper()],
        goal_priority=GoalPriority[robot_strategy['goal_priority'].upper()],
        autonomous_strategy=AutonomousStrategy[robot_strategy['autonomous'].upper()]
    )
    
    return TeamConfiguration(
        team_name=config_data['team']['name'],
        robot_config=robot_config,
        analysis_preferences=config_data['analysis'],
        competition_schedule=config_data['competition']['upcoming_events'],
        visualization_settings=config_data['visualization']
    )

# Usage
team_config = load_team_config('config/team_config.yaml')
```

### Configuration Validation

```python
def validate_configuration(config: TeamConfiguration) -> List[str]:
    """Validate team configuration for common issues."""
    
    issues = []
    
    # Validate robot performance ranges
    robot = config.robot_config
    
    if robot.average_cycle_time < 2.0 or robot.average_cycle_time > 20.0:
        issues.append(f"Average cycle time {robot.average_cycle_time}s outside realistic range (2.0-20.0s)")
    
    if robot.pickup_reliability < 0.5 or robot.pickup_reliability > 1.0:
        issues.append(f"Pickup reliability {robot.pickup_reliability} outside valid range (0.5-1.0)")
    
    if robot.min_cycle_time >= robot.average_cycle_time:
        issues.append("Minimum cycle time should be less than average cycle time")
    
    # Validate analysis settings
    analysis = config.analysis_preferences
    
    if analysis.get('default_simulations', 0) < 100:
        issues.append("Default simulation size should be at least 100 for reliable results")
    
    if analysis.get('confidence_threshold', 0) < 0.7:
        issues.append("Confidence threshold below 0.7 may produce unreliable recommendations")
    
    # Validate competition schedule
    import datetime
    today = datetime.date.today()
    
    for event in config.competition_schedule:
        event_date = datetime.datetime.strptime(event['date'], '%Y-%m-%d').date()
        if event_date < today:
            issues.append(f"Competition '{event['name']}' date is in the past")
    
    return issues

# Usage
validation_issues = validate_configuration(team_config)
if validation_issues:
    print("Configuration Issues:")
    for issue in validation_issues:
        print(f"  • {issue}")
else:
    print("✅ Configuration is valid")
```

This configuration system allows teams to precisely tune the Push Back Analysis System to their specific robot capabilities, strategic preferences, and competitive context for maximum analytical accuracy.