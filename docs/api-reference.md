# API Reference

The Push Back Analysis System provides a comprehensive REST API and WebSocket interface for real-time strategic analysis. All endpoints are optimized for Push Back's specific requirements.

## 🌐 Base Configuration

**Base URL**: `http://localhost:5000/api/push-back`  
**Content-Type**: `application/json`  
**Response Format**: JSON with consistent error handling  
**Rate Limiting**: 100 requests/minute per IP  

## 📊 Core Analysis Endpoints

### POST /analyze

Comprehensive strategy analysis for Push Back robots.

```http
POST /api/push-back/analyze
Content-Type: application/json

{
  "robot_capabilities": {
    "cycle_time": 4.5,
    "pickup_reliability": 0.95,
    "scoring_reliability": 0.98,
    "autonomous_reliability": 0.88,
    "max_capacity": 2,
    "parking_capability": true
  },
  "strategy_preferences": {
    "goal_priority": "center_preferred",
    "parking_strategy": "late",
    "autonomous_aggression": "balanced",
    "risk_tolerance": 0.6
  },
  "opponent_analysis": {
    "estimated_strength": "competitive",
    "known_strategies": ["speed_focused"],
    "reliability_estimate": 0.92
  },
  "match_context": {
    "competition_level": "regional",
    "match_type": "qualification",
    "alliance_partner": "compatible"
  }
}
```

**Response**:
```json
{
  "success": true,
  "analysis": {
    "win_probability": 0.725,
    "expected_score": 115.3,
    "score_variance": 180.5,
    "confidence_level": 0.89,
    "execution_time": 0.245
  },
  "strategic_insights": [
    {
      "category": "block_flow",
      "insight": "Prioritize center goals for 65% efficiency advantage",
      "impact": "high",
      "confidence": 0.92
    },
    {
      "category": "timing",
      "insight": "Park with 13-15 seconds remaining for optimal value",
      "impact": "medium", 
      "confidence": 0.78
    }
  ],
  "recommendations": {
    "primary_strategy": "center_goal_maximizer",
    "block_allocation": {
      "center_goals": 0.65,
      "long_goals": 0.35
    },
    "critical_timings": {
      "parking_decision": 14.2,
      "strategy_pivot": 45.0
    },
    "risk_mitigation": [
      "Practice autonomous consistency",
      "Develop contingency for blocked center goals"
    ]
  }
}
```

### POST /monte-carlo

Run Monte Carlo simulations for statistical analysis.

```http
POST /api/push-back/monte-carlo
Content-Type: application/json

{
  "red_robot": {
    "cycle_time": 4.2,
    "reliability": 0.96,
    "parking_strategy": "late",
    "goal_preference": "center"
  },
  "blue_robot": {
    "cycle_time": 5.1,
    "reliability": 0.91,
    "parking_strategy": "never",
    "goal_preference": "balanced"
  },
  "simulation_config": {
    "num_simulations": 2000,
    "use_parallel": true,
    "scenario_type": "standard"
  }
}
```

**Response**:
```json
{
  "success": true,
  "results": {
    "simulations_run": 2000,
    "execution_time": 0.178,
    "win_probability": 0.731,
    "score_distribution": {
      "red_mean": 118.7,
      "red_std": 13.4,
      "blue_mean": 97.2,
      "blue_std": 15.8
    },
    "match_statistics": {
      "average_margin": 21.5,
      "close_matches": 0.23,
      "dominant_victories": 0.41
    },
    "insights": {
      "key_advantages": [
        "Cycle time advantage provides consistent scoring edge",
        "Center goal preference aligns with robot strengths"
      ],
      "risk_factors": [
        "Vulnerable if center goals become contested"
      ]
    }
  }
}
```

## 🎯 Decision Support Endpoints

### POST /decide/blocks

Optimize block allocation across goals.

```http
POST /api/push-back/decide/blocks
Content-Type: application/json

{
  "robot_capabilities": {
    "center_efficiency": 1.3,
    "long_efficiency": 0.8,
    "scoring_rate": 0.45,
    "travel_speed": 3.2
  },
  "field_state": {
    "time_remaining": 67,
    "available_goals": ["center1", "center2", "long1", "long2"],
    "contested_goals": ["center1"],
    "blocks_remaining": 45
  },
  "match_context": {
    "score_differential": 8,
    "opponent_strategy": "defensive"
  }
}
```

**Response**:
```json
{
  "success": true,
  "optimization": {
    "recommended_distribution": {
      "center_goals": 0.58,
      "long_goals": 0.42
    },
    "primary_target": "center2",
    "secondary_target": "long1",
    "expected_value": 87.3,
    "confidence": 0.84
  },
  "strategic_reasoning": {
    "why_center_focus": "1.3x efficiency advantage outweighs travel time",
    "contingency_plan": "Pivot to long goals if both centers become contested",
    "timing_considerations": "Sufficient time for center goal strategy"
  }
}
```

### POST /decide/autonomous

Select optimal autonomous strategy.

```http
POST /api/push-back/decide/autonomous
Content-Type: application/json

{
  "robot_capabilities": {
    "autonomous_reliability": 0.87,
    "scoring_potential": 2.5,
    "positioning_speed": 0.8,
    "routine_consistency": 0.91
  },
  "competition_context": {
    "opponent_auto_strength": "strong",
    "match_importance": "elimination",
    "field_conditions": "standard"
  },
  "risk_profile": {
    "risk_tolerance": 0.4,
    "backup_plan_available": true
  }
}
```

**Response**:
```json
{
  "success": true,
  "decision": {
    "recommended_strategy": "balanced_aggression",
    "confidence_level": 0.81,
    "expected_points": 6.8,
    "win_point_probability": 0.34
  },
  "analysis": {
    "strategy_comparison": {
      "aggressive": {"points": 8.2, "success_rate": 0.73, "risk": "high"},
      "balanced": {"points": 6.8, "success_rate": 0.87, "risk": "medium"},
      "safe": {"points": 4.9, "success_rate": 0.96, "risk": "low"}
    },
    "recommendation_reasoning": "Balanced approach optimizes expected value while maintaining acceptable risk for elimination match",
    "key_factors": [
      "Strong opponent requires reliable point generation",
      "87% reliability sufficient for moderate aggression",
      "Elimination context favors consistency over maximum risk"
    ]
  }
}
```

### POST /decide/parking

Calculate optimal parking timing.

```http
POST /api/push-back/decide/parking
Content-Type: application/json

{
  "match_state": {
    "current_score_red": 98,
    "current_score_blue": 89,
    "time_remaining": 18.5,
    "robot_positions": ["field_center", "near_goal"],
    "blocks_remaining": 12
  },
  "robot_capabilities": {
    "parking_time": 2.8,
    "scoring_rate": 0.35,
    "movement_speed": 2.9
  },
  "strategic_context": {
    "opponent_parking_intent": "likely",
    "alliance_coordination": "independent"
  }
}
```

**Response**:
```json
{
  "success": true,
  "decision": {
    "recommended_action": "park_one_robot",
    "optimal_timing": 12.3,
    "robot_priority": "near_goal",
    "confidence": 0.89
  },
  "analysis": {
    "breakeven_points": {
      "park_one": 11.4,
      "park_both": 8.7,
      "continue_scoring": "never"
    },
    "probability_scenarios": {
      "maintain_lead_no_park": 0.67,
      "maintain_lead_park_one": 0.84,
      "maintain_lead_park_both": 0.91
    },
    "risk_assessment": {
      "risk_level": "low",
      "primary_risk": "Opponent late scoring surge",
      "mitigation": "Park one robot to secure 8 points while continuing to score"
    }
  },
  "contingency_plans": [
    {
      "condition": "Score differential drops below 5",
      "action": "Continue scoring with both robots"
    },
    {
      "condition": "Time reaches 10 seconds",
      "action": "Park second robot immediately"
    }
  ]
}
```

## ⚡ Real-Time WebSocket API

### Connection Setup

```javascript
const socket = io('ws://localhost:5000/push-back');

socket.on('connect', () => {
  console.log('Connected to Push Back analysis server');
  
  // Subscribe to real-time analysis updates
  socket.emit('subscribe_analysis', {
    session_id: 'team_strategy_session_001',
    analysis_types: ['strategy', 'monte_carlo', 'decisions']
  });
});
```

### Real-Time Strategy Updates

```javascript
// Listen for strategy analysis updates
socket.on('strategy_update', (data) => {
  console.log('Strategy Analysis Update:', data);
  /*
  {
    "session_id": "team_strategy_session_001",
    "timestamp": 1703123456789,
    "analysis": {
      "win_probability": 0.742,
      "expected_score": 121.8,
      "recommended_adjustments": [
        "Increase center goal priority by 10%",
        "Consider earlier parking if leading by 15+"
      ]
    },
    "trigger": "robot_config_change"
  }
  */
});

// Send strategy configuration changes
socket.emit('strategy_change', {
  session_id: 'team_strategy_session_001',
  changes: {
    robot_capabilities: {
      cycle_time: 4.1,  // Improved from 4.5
      reliability: 0.97  // Improved from 0.95
    }
  }
});
```

### Live Match Analysis

```javascript
// Real-time match state analysis
socket.emit('match_state_update', {
  session_id: 'live_match_001',
  match_state: {
    time_remaining: 23.5,
    current_score_diff: 12,
    robot_positions: ['field', 'near_park'],
    field_state: 'blocks_scattered'
  }
});

socket.on('live_recommendation', (data) => {
  console.log('Live Match Recommendation:', data);
  /*
  {
    "urgency": "medium",
    "recommendation": "Park one robot in 8-10 seconds",
    "reasoning": "Leading by 12 points, secure advantage",
    "confidence": 0.91,
    "alternative": "Continue scoring if opponent also parks"
  }
  */
});
```

## 📈 Batch Processing Endpoints

### POST /batch/scenarios

Analyze multiple scenarios simultaneously.

```http
POST /api/push-back/batch/scenarios
Content-Type: application/json

{
  "scenarios": [
    {
      "name": "vs_speed_team",
      "red_robot": {...},
      "blue_robot": {...},
      "simulations": 1000
    },
    {
      "name": "vs_control_team", 
      "red_robot": {...},
      "blue_robot": {...},
      "simulations": 1000
    }
  ],
  "analysis_types": ["monte_carlo", "strategy", "insights"]
}
```

**Response**:
```json
{
  "success": true,
  "batch_id": "batch_20231201_001",
  "status": "completed", 
  "execution_time": 0.892,
  "results": {
    "vs_speed_team": {
      "win_probability": 0.689,
      "expected_score": 108.3,
      "key_strategy": "defensive_counter"
    },
    "vs_control_team": {
      "win_probability": 0.756,
      "expected_score": 119.7,
      "key_strategy": "aggressive_offense"
    }
  },
  "summary": {
    "overall_win_rate": 0.723,
    "consistent_strategies": ["center_goal_focus"],
    "adaptive_needs": ["parking_timing", "goal_switching"]
  }
}
```

## 🛠️ Configuration Endpoints

### GET /config/robots

Get available robot configuration templates.

```http
GET /api/push-back/config/robots
```

**Response**:
```json
{
  "success": true,
  "robot_templates": {
    "beginner": {
      "cycle_time": 8.0,
      "reliability": 0.87,
      "description": "Entry-level robot configuration",
      "typical_score": "60-90 points"
    },
    "competitive": {  
      "cycle_time": 4.2,
      "reliability": 0.96,
      "description": "High-performance competitive robot",
      "typical_score": "100-140 points"
    },
    "speed_focused": {
      "cycle_time": 3.5,
      "reliability": 0.91,
      "description": "Maximum speed, moderate reliability",
      "typical_score": "90-130 points"
    }
  }
}
```

### POST /config/validate

Validate robot configuration for reasonableness.

```http
POST /api/push-back/config/validate
Content-Type: application/json

{
  "robot_config": {
    "cycle_time": 2.0,
    "pickup_reliability": 1.1,
    "scoring_reliability": 0.99
  }
}
```

**Response**:
```json
{
  "success": false,
  "validation": {
    "is_valid": false,
    "errors": [
      {
        "field": "pickup_reliability",
        "message": "Reliability cannot exceed 1.0",
        "suggestion": "Set to maximum 1.0 for perfect reliability"
      }
    ],
    "warnings": [
      {
        "field": "cycle_time", 
        "message": "Cycle time of 2.0s is extremely fast",
        "suggestion": "Typical range is 3.0-8.0 seconds"
      }
    ]
  }
}
```

## 🔍 Analysis History Endpoints

### GET /history/sessions

Get analysis session history.

```http
GET /api/push-back/history/sessions?limit=10&user_id=team_001
```

**Response**:
```json
{
  "success": true,
  "sessions": [
    {
      "session_id": "session_20231201_001",
      "timestamp": "2023-12-01T14:30:00Z",
      "robot_config": "competitive_v2",
      "analysis_count": 15,
      "key_insights": ["Center goal advantage", "Late parking optimal"],
      "win_probability": 0.731
    }
  ],
  "pagination": {
    "current_page": 1,
    "total_pages": 3,
    "total_sessions": 27
  }
}
```

### GET /history/compare

Compare analysis results across sessions.

```http
GET /api/push-back/history/compare?session1=session_001&session2=session_002
```

## 🚨 Error Handling

All endpoints use consistent error response format:

```json
{
  "success": false,
  "error": {
    "code": "INVALID_ROBOT_CONFIG",
    "message": "Robot configuration contains invalid parameters",
    "details": {
      "field": "cycle_time",
      "value": -1.0,
      "constraint": "must be positive"
    },
    "suggestion": "Set cycle_time to a positive value between 2.0 and 15.0 seconds"
  },
  "request_id": "req_20231201_14:30:01_001"
}
```

### Common Error Codes

- `INVALID_ROBOT_CONFIG` - Robot parameters outside valid ranges
- `SIMULATION_TIMEOUT` - Monte Carlo simulation exceeded time limit  
- `INSUFFICIENT_DATA` - Not enough data for reliable analysis
- `RATE_LIMIT_EXCEEDED` - Too many requests per minute
- `INTERNAL_ERROR` - Server-side processing error

## 🔧 Client Examples

### Python Client

```python
import requests
import json

class PushBackAPIClient:
    def __init__(self, base_url="http://localhost:5000/api/push-back"):
        self.base_url = base_url
    
    def analyze_strategy(self, robot_config, opponent_analysis):
        response = requests.post(
            f"{self.base_url}/analyze",
            json={
                "robot_capabilities": robot_config,
                "opponent_analysis": opponent_analysis
            }
        )
        return response.json()
    
    def run_monte_carlo(self, red_robot, blue_robot, num_sims=1000):
        response = requests.post(
            f"{self.base_url}/monte-carlo",
            json={
                "red_robot": red_robot,
                "blue_robot": blue_robot,
                "simulation_config": {"num_simulations": num_sims}
            }
        )
        return response.json()

# Usage
client = PushBackAPIClient()

result = client.analyze_strategy(
    robot_config={"cycle_time": 4.5, "reliability": 0.95},
    opponent_analysis={"strength": "competitive"}
)

print(f"Win Probability: {result['analysis']['win_probability']:.1%}")
```

### JavaScript Client

```javascript
class PushBackAPIClient {
  constructor(baseURL = 'http://localhost:5000/api/push-back') {
    this.baseURL = baseURL;
  }
  
  async analyzeStrategy(robotConfig, opponentAnalysis) {
    const response = await fetch(`${this.baseURL}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        robot_capabilities: robotConfig,
        opponent_analysis: opponentAnalysis
      })
    });
    return response.json();
  }
  
  async optimizeBlocks(robotCapabilities, fieldState) {
    const response = await fetch(`${this.baseURL}/decide/blocks`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        robot_capabilities: robotCapabilities,
        field_state: fieldState
      })
    });
    return response.json();
  }
}

// Usage
const client = new PushBackAPIClient();

const analysis = await client.analyzeStrategy(
  { cycle_time: 4.5, reliability: 0.95 },
  { strength: 'competitive' }
);

console.log(`Expected Score: ${analysis.analysis.expected_score}`);
```

The Push Back API provides comprehensive access to all strategic analysis capabilities with consistent, fast responses optimized for competitive VEX U teams.