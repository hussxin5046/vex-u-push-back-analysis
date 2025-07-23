"""
Push Back-specific scenario generation and analysis.

This module provides realistic scenario generation for Push Back matches,
including various team configurations, field states, and strategic situations
that teams commonly encounter during competition.
"""

import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
import json

from .push_back_monte_carlo import (
    RobotCapabilities, ParkingStrategy, GoalPriority, AutonomousStrategy,
    PushBackMonteCarloEngine, create_default_robot, create_competitive_robot,
    create_beginner_robot
)

class MatchType(Enum):
    """Different types of matches to simulate"""
    PRACTICE = "practice"
    QUALIFICATION = "qualification"
    ELIMINATION = "elimination"
    SCRIMMAGE = "scrimmage"

class TeamSkillLevel(Enum):
    """Team skill level classifications"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    ELITE = "elite"

class FieldCondition(Enum):
    """Field condition variations"""
    PERFECT = "perfect"
    WORN = "worn"
    DAMAGED = "damaged"
    NEW = "new"

@dataclass
class ScenarioConfig:
    """Configuration for scenario generation"""
    match_type: MatchType = MatchType.QUALIFICATION
    red_skill_level: TeamSkillLevel = TeamSkillLevel.INTERMEDIATE
    blue_skill_level: TeamSkillLevel = TeamSkillLevel.INTERMEDIATE
    field_condition: FieldCondition = FieldCondition.PERFECT
    time_of_day: Optional[str] = None  # "morning", "afternoon", "evening"
    driver_fatigue: float = 0.0  # 0.0 to 1.0
    pressure_level: float = 0.5  # 0.0 to 1.0
    
    # Strategic variations
    red_strategy_focus: str = "balanced"
    blue_strategy_focus: str = "balanced"
    
    # Environmental factors
    temperature: float = 72.0  # Fahrenheit
    humidity: float = 0.5  # 0.0 to 1.0

@dataclass
class TeamProfile:
    """Detailed team performance profile"""
    team_number: str
    skill_level: TeamSkillLevel
    robot_capabilities: RobotCapabilities
    strategy_preferences: Dict[str, float]
    consistency_rating: float  # 0.0 to 1.0
    pressure_response: float  # How well they handle pressure
    learning_rate: float  # How quickly they adapt during competition

class PushBackScenarioGenerator:
    """
    Generates realistic Push Back scenarios for strategic analysis.
    
    This class creates diverse match scenarios that teams might encounter,
    including various opponent types, field conditions, and strategic situations.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.random = random.Random(seed)
        self.scenario_templates = self._create_scenario_templates()
        self.team_profiles = self._create_team_profiles()
    
    def _create_scenario_templates(self) -> Dict[str, Dict]:
        """Create templates for common Push Back scenarios"""
        return {
            "mirror_match": {
                "description": "Teams with similar capabilities",
                "red_skill": TeamSkillLevel.INTERMEDIATE,
                "blue_skill": TeamSkillLevel.INTERMEDIATE,
                "strategy_variation": 0.1
            },
            "david_vs_goliath": {
                "description": "Significant skill gap between teams",
                "red_skill": TeamSkillLevel.BEGINNER,
                "blue_skill": TeamSkillLevel.ELITE,
                "strategy_variation": 0.3
            },
            "elimination_pressure": {
                "description": "High-pressure elimination match",
                "match_type": MatchType.ELIMINATION,
                "pressure_level": 0.9,
                "driver_fatigue": 0.7
            },
            "early_season": {
                "description": "Early season with high variability",
                "consistency_modifier": -0.2,
                "strategy_variation": 0.4
            },
            "late_season": {
                "description": "Late season with refined strategies",
                "consistency_modifier": 0.2,
                "strategy_variation": 0.1
            },
            "defensive_focused": {
                "description": "One team focuses on defense",
                "red_strategy": "offensive",
                "blue_strategy": "defensive"
            },
            "speed_vs_precision": {
                "description": "Fast but inconsistent vs slow but precise",
                "red_style": "speed",
                "blue_style": "precision"
            }
        }
    
    def _create_team_profiles(self) -> Dict[str, TeamProfile]:
        """Create diverse team profiles for scenario generation"""
        profiles = {}
        
        # Elite teams
        profiles["elite_1"] = TeamProfile(
            team_number="1234A",
            skill_level=TeamSkillLevel.ELITE,
            robot_capabilities=self._create_elite_robot(),
            strategy_preferences={
                "aggressive_autonomous": 0.9,
                "center_goal_priority": 0.8,
                "late_parking": 0.9,
                "control_zone_focus": 0.7
            },
            consistency_rating=0.95,
            pressure_response=0.9,
            learning_rate=0.8
        )
        
        # Advanced teams
        profiles["advanced_1"] = TeamProfile(
            team_number="5678B",
            skill_level=TeamSkillLevel.ADVANCED,
            robot_capabilities=create_competitive_robot(),
            strategy_preferences={
                "aggressive_autonomous": 0.7,
                "center_goal_priority": 0.6,
                "late_parking": 0.8,
                "control_zone_focus": 0.5
            },
            consistency_rating=0.85,
            pressure_response=0.7,
            learning_rate=0.7
        )
        
        # Intermediate teams
        profiles["intermediate_1"] = TeamProfile(
            team_number="9012C",
            skill_level=TeamSkillLevel.INTERMEDIATE,
            robot_capabilities=create_default_robot(),
            strategy_preferences={
                "aggressive_autonomous": 0.5,
                "center_goal_priority": 0.5,
                "late_parking": 0.6,
                "control_zone_focus": 0.4
            },
            consistency_rating=0.7,
            pressure_response=0.5,
            learning_rate=0.6
        )
        
        # Beginner teams
        profiles["beginner_1"] = TeamProfile(
            team_number="3456D",
            skill_level=TeamSkillLevel.BEGINNER,
            robot_capabilities=create_beginner_robot(),
            strategy_preferences={
                "aggressive_autonomous": 0.3,
                "center_goal_priority": 0.4,
                "late_parking": 0.3,
                "control_zone_focus": 0.2
            },
            consistency_rating=0.5,
            pressure_response=0.3,
            learning_rate=0.8
        )
        
        return profiles
    
    def _create_elite_robot(self) -> RobotCapabilities:
        """Create an elite-level robot configuration"""
        return RobotCapabilities(
            min_cycle_time=2.0,
            max_cycle_time=4.5,
            average_cycle_time=3.0,
            max_speed=6.0,
            average_speed=4.5,
            pickup_reliability=0.99,
            scoring_reliability=0.995,
            autonomous_reliability=0.98,
            parking_strategy=ParkingStrategy.LATE,
            goal_priority=GoalPriority.CENTER_PREFERRED,
            autonomous_strategy=AutonomousStrategy.AGGRESSIVE,
            max_blocks_per_trip=3,
            prefers_singles=False,
            control_zone_frequency=0.6,
            control_zone_duration=3.0
        )
    
    def generate_scenario(self, scenario_type: str = None, 
                         config: Optional[ScenarioConfig] = None) -> Tuple[RobotCapabilities, RobotCapabilities, Dict]:
        """Generate a realistic Push Back scenario"""
        
        if config is None:
            config = ScenarioConfig()
        
        # Select scenario template
        if scenario_type and scenario_type in self.scenario_templates:
            template = self.scenario_templates[scenario_type]
        else:
            template = self.random.choice(list(self.scenario_templates.values()))
        
        # Generate team capabilities based on template and config
        red_robot = self._generate_robot_from_skill(
            config.red_skill_level, template, config, "red"
        )
        blue_robot = self._generate_robot_from_skill(
            config.blue_skill_level, template, config, "blue"
        )
        
        # Apply environmental factors
        self._apply_environmental_factors(red_robot, blue_robot, config)
        
        # Create scenario metadata
        metadata = {
            "scenario_type": scenario_type or "random",
            "template": template,
            "config": config,
            "environmental_factors": self._get_environmental_impact(config),
            "expected_dynamics": self._predict_match_dynamics(red_robot, blue_robot)
        }
        
        return red_robot, blue_robot, metadata
    
    def _generate_robot_from_skill(self, skill_level: TeamSkillLevel, 
                                  template: Dict, config: ScenarioConfig,
                                  alliance: str) -> RobotCapabilities:
        """Generate robot capabilities based on skill level and scenario"""
        
        base_robots = {
            TeamSkillLevel.BEGINNER: create_beginner_robot(),
            TeamSkillLevel.INTERMEDIATE: create_default_robot(),
            TeamSkillLevel.ADVANCED: create_competitive_robot(),
            TeamSkillLevel.ELITE: self._create_elite_robot()
        }
        
        robot = base_robots[skill_level]
        
        # Apply template modifications
        if "consistency_modifier" in template:
            modifier = template["consistency_modifier"]
            robot.pickup_reliability = max(0.5, min(1.0, robot.pickup_reliability + modifier))
            robot.scoring_reliability = max(0.5, min(1.0, robot.scoring_reliability + modifier))
            robot.autonomous_reliability = max(0.3, min(1.0, robot.autonomous_reliability + modifier))
        
        # Apply strategy variations
        if "strategy_variation" in template:
            variation = template["strategy_variation"]
            self._apply_strategy_variation(robot, variation)
        
        # Apply specific strategy focus from template
        strategy_key = f"{alliance}_strategy"
        if strategy_key in template:
            self._apply_strategy_focus(robot, template[strategy_key])
        
        return robot
    
    def _apply_strategy_variation(self, robot: RobotCapabilities, variation: float):
        """Apply random variations to robot strategy"""
        
        # Vary cycle times
        time_variation = self.random.uniform(-variation, variation)
        robot.average_cycle_time *= (1 + time_variation)
        robot.min_cycle_time *= (1 + time_variation * 0.5)
        robot.max_cycle_time *= (1 + time_variation * 1.5)
        
        # Vary reliability
        reliability_variation = self.random.uniform(-variation * 0.1, variation * 0.1)
        robot.pickup_reliability = max(0.5, min(1.0, robot.pickup_reliability + reliability_variation))
        robot.scoring_reliability = max(0.5, min(1.0, robot.scoring_reliability + reliability_variation))
        
        # Randomly adjust strategy preferences
        if self.random.random() < variation:
            strategies = [ParkingStrategy.EARLY, ParkingStrategy.LATE, ParkingStrategy.NEVER]
            robot.parking_strategy = self.random.choice(strategies)
        
        if self.random.random() < variation:
            goals = [GoalPriority.CENTER_PREFERRED, GoalPriority.LONG_PREFERRED, GoalPriority.BALANCED]
            robot.goal_priority = self.random.choice(goals)
    
    def _apply_strategy_focus(self, robot: RobotCapabilities, focus: str):
        """Apply specific strategic focus to robot"""
        
        if focus == "offensive":
            robot.control_zone_frequency *= 0.5
            robot.average_cycle_time *= 0.9
            robot.parking_strategy = ParkingStrategy.NEVER
            
        elif focus == "defensive":
            robot.control_zone_frequency *= 2.0
            robot.parking_strategy = ParkingStrategy.EARLY
            robot.autonomous_strategy = AutonomousStrategy.SAFE
            
        elif focus == "speed":
            robot.average_cycle_time *= 0.8
            robot.pickup_reliability *= 0.9
            robot.scoring_reliability *= 0.95
            
        elif focus == "precision":
            robot.average_cycle_time *= 1.2
            robot.pickup_reliability = min(1.0, robot.pickup_reliability * 1.1)
            robot.scoring_reliability = min(1.0, robot.scoring_reliability * 1.05)
    
    def _apply_environmental_factors(self, red_robot: RobotCapabilities, 
                                   blue_robot: RobotCapabilities, 
                                   config: ScenarioConfig):
        """Apply environmental factors to robot performance"""
        
        # Driver fatigue effect
        if config.driver_fatigue > 0:
            fatigue_penalty = config.driver_fatigue * 0.1
            for robot in [red_robot, blue_robot]:
                robot.average_cycle_time *= (1 + fatigue_penalty)
                robot.pickup_reliability *= (1 - fatigue_penalty * 0.5)
        
        # Pressure effect
        if config.pressure_level > 0.7:
            pressure_penalty = (config.pressure_level - 0.7) * 0.2
            for robot in [red_robot, blue_robot]:
                robot.autonomous_reliability *= (1 - pressure_penalty)
                robot.scoring_reliability *= (1 - pressure_penalty * 0.5)
        
        # Field condition effect
        if config.field_condition == FieldCondition.WORN:
            for robot in [red_robot, blue_robot]:
                robot.pickup_reliability *= 0.95
                robot.max_speed *= 0.9
        elif config.field_condition == FieldCondition.DAMAGED:
            for robot in [red_robot, blue_robot]:
                robot.pickup_reliability *= 0.9
                robot.max_speed *= 0.8
                robot.average_cycle_time *= 1.1
    
    def _get_environmental_impact(self, config: ScenarioConfig) -> Dict:
        """Calculate environmental impact factors"""
        impact = {
            "fatigue_penalty": config.driver_fatigue * 0.1,
            "pressure_penalty": max(0, config.pressure_level - 0.7) * 0.2,
            "field_condition_impact": {
                FieldCondition.PERFECT: 0.0,
                FieldCondition.NEW: 0.02,
                FieldCondition.WORN: -0.05,
                FieldCondition.DAMAGED: -0.15
            }[config.field_condition]
        }
        return impact
    
    def _predict_match_dynamics(self, red_robot: RobotCapabilities, 
                               blue_robot: RobotCapabilities) -> Dict:
        """Predict expected match dynamics"""
        
        # Calculate relative strengths
        red_speed = 1.0 / red_robot.average_cycle_time
        blue_speed = 1.0 / blue_robot.average_cycle_time
        
        red_reliability = (red_robot.pickup_reliability + red_robot.scoring_reliability) / 2
        blue_reliability = (blue_robot.pickup_reliability + blue_robot.scoring_reliability) / 2
        
        dynamics = {
            "expected_closeness": abs(red_speed * red_reliability - blue_speed * blue_reliability),
            "predicted_winner": "red" if red_speed * red_reliability > blue_speed * blue_reliability else "blue",
            "key_factors": [],
            "strategic_interactions": []
        }
        
        # Identify key factors
        if abs(red_robot.autonomous_reliability - blue_robot.autonomous_reliability) > 0.1:
            dynamics["key_factors"].append("autonomous_advantage")
        
        if red_robot.parking_strategy != blue_robot.parking_strategy:
            dynamics["key_factors"].append("parking_strategy_difference")
        
        if red_robot.goal_priority != blue_robot.goal_priority:
            dynamics["strategic_interactions"].append("goal_priority_conflict")
        
        return dynamics
    
    def generate_tournament_scenarios(self, num_scenarios: int = 50) -> List[Tuple]:
        """Generate a diverse set of tournament scenarios"""
        scenarios = []
        
        # Ensure diversity in scenario types
        scenario_types = list(self.scenario_templates.keys())
        
        for i in range(num_scenarios):
            # Cycle through scenario types for diversity
            scenario_type = scenario_types[i % len(scenario_types)]
            
            # Create varied configs
            config = ScenarioConfig(
                match_type=self.random.choice(list(MatchType)),
                red_skill_level=self.random.choice(list(TeamSkillLevel)),
                blue_skill_level=self.random.choice(list(TeamSkillLevel)),
                field_condition=self.random.choice(list(FieldCondition)),
                driver_fatigue=self.random.uniform(0, 0.8),
                pressure_level=self.random.uniform(0.2, 1.0)
            )
            
            scenario = self.generate_scenario(scenario_type, config)
            scenarios.append(scenario)
        
        return scenarios
    
    def analyze_scenario_diversity(self, scenarios: List[Tuple]) -> Dict:
        """Analyze the diversity of generated scenarios"""
        
        skill_combinations = {}
        strategy_combinations = {}
        environmental_factors = {"high_pressure": 0, "high_fatigue": 0, "poor_field": 0}
        
        for red_robot, blue_robot, metadata in scenarios:
            # Track skill combinations
            red_skill = metadata["config"].red_skill_level
            blue_skill = metadata["config"].blue_skill_level
            skill_key = f"{red_skill.value}_vs_{blue_skill.value}"
            skill_combinations[skill_key] = skill_combinations.get(skill_key, 0) + 1
            
            # Track strategy combinations
            strategy_key = f"{red_robot.parking_strategy.value}_{red_robot.goal_priority.value}_vs_{blue_robot.parking_strategy.value}_{blue_robot.goal_priority.value}"
            strategy_combinations[strategy_key] = strategy_combinations.get(strategy_key, 0) + 1
            
            # Track environmental factors
            config = metadata["config"]
            if config.pressure_level > 0.8:
                environmental_factors["high_pressure"] += 1
            if config.driver_fatigue > 0.6:
                environmental_factors["high_fatigue"] += 1
            if config.field_condition in [FieldCondition.WORN, FieldCondition.DAMAGED]:
                environmental_factors["poor_field"] += 1
        
        return {
            "total_scenarios": len(scenarios),
            "skill_diversity": len(skill_combinations),
            "strategy_diversity": len(strategy_combinations),
            "skill_distribution": skill_combinations,
            "environmental_distribution": environmental_factors
        }

def create_scouting_scenarios() -> List[Tuple]:
    """Create scenarios focused on scouting different team archetypes"""
    generator = PushBackScenarioGenerator()
    
    scenarios = []
    
    # Common team archetypes to scout
    archetypes = [
        ("speed_demon", TeamSkillLevel.ADVANCED, "speed"),
        ("precision_master", TeamSkillLevel.ADVANCED, "precision"),
        ("defensive_wall", TeamSkillLevel.INTERMEDIATE, "defensive"),
        ("offensive_juggernaut", TeamSkillLevel.ELITE, "offensive"),
        ("consistent_performer", TeamSkillLevel.INTERMEDIATE, "balanced"),
        ("wildcard", TeamSkillLevel.BEGINNER, "random")
    ]
    
    for name, skill, style in archetypes:
        config = ScenarioConfig(
            red_skill_level=skill,
            blue_skill_level=TeamSkillLevel.INTERMEDIATE,  # Standard opponent
            red_strategy_focus=style
        )
        
        scenario = generator.generate_scenario(name, config)
        scenarios.append(scenario)
    
    return scenarios

def create_elimination_scenarios() -> List[Tuple]:
    """Create high-pressure elimination match scenarios"""
    generator = PushBackScenarioGenerator()
    
    scenarios = []
    
    # Different elimination pressures
    pressure_configs = [
        ("quarterfinals", 0.7, 0.5),
        ("semifinals", 0.8, 0.6),
        ("finals", 0.95, 0.8)
    ]
    
    for round_name, pressure, fatigue in pressure_configs:
        config = ScenarioConfig(
            match_type=MatchType.ELIMINATION,
            pressure_level=pressure,
            driver_fatigue=fatigue,
            red_skill_level=TeamSkillLevel.ADVANCED,
            blue_skill_level=TeamSkillLevel.ADVANCED
        )
        
        scenario = generator.generate_scenario("elimination_pressure", config)
        scenarios.append(scenario)
    
    return scenarios