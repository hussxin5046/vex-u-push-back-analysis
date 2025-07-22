import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from datetime import datetime
import itertools
import warnings

try:
    from .feature_engineering import VEXUFeatureExtractor, GameState, RobotState, MatchPhase, RobotSize
    from ..core.simulator import AllianceStrategy, ScoringSimulator, Zone, ParkingLocation
    from ..core.scenario_generator import ScenarioGenerator, SkillLevel, StrategyType, RobotRole
except ImportError:
    # Fallback for when running from main.py
    from ml_models.feature_engineering import VEXUFeatureExtractor, GameState, RobotState, MatchPhase, RobotSize
    from core.simulator import AllianceStrategy, ScoringSimulator, Zone, ParkingLocation
    from core.scenario_generator import ScenarioGenerator, SkillLevel, StrategyType, RobotRole

warnings.filterwarnings('ignore', category=FutureWarning)


class TaskType(Enum):
    PRIMARY_SCORER = "primary_scorer"
    SECONDARY_SCORER = "secondary_scorer"
    ZONE_CONTROLLER = "zone_controller"
    DEFENDER = "defender"
    SUPPORT = "support"
    LOADER = "loader"
    HYBRID = "hybrid"


class CoordinationStrategy(Enum):
    DIVIDE_AND_CONQUER = "divide_and_conquer"
    DOUBLE_TEAM = "double_team"
    SPECIALIZED_ROLES = "specialized_roles"
    DYNAMIC_SWITCHING = "dynamic_switching"
    SUPPORT_SCORER = "support_scorer"


@dataclass
class RobotAssignment:
    robot_id: str
    primary_task: TaskType
    secondary_task: Optional[TaskType]
    assigned_goals: List[str]
    assigned_zones: List[Zone]
    priority: int  # 1 = highest priority
    expected_contribution: float  # Expected points contribution


@dataclass
class CoordinationPlan:
    strategy_type: CoordinationStrategy
    robot1_assignment: RobotAssignment
    robot2_assignment: RobotAssignment
    synergy_score: float
    expected_total_score: float
    risk_level: str
    coordination_efficiency: float
    task_balance_score: float


@dataclass
class CoordinationMetrics:
    distance_efficiency: float
    task_overlap: float
    goal_coverage: float
    zone_control_effectiveness: float
    role_specialization: float
    communication_requirements: float


class VEXUCoordinationModel:
    def __init__(self, model_name: str = "vex_u_coordination_model"):
        self.model_name = model_name
        self.role_classifier = None
        self.synergy_predictor = None
        self.scaler = StandardScaler()
        self.task_encoder = LabelEncoder()
        self.strategy_encoder = LabelEncoder()
        
        # Feature extractor and simulation components
        self.feature_extractor = VEXUFeatureExtractor()
        self.simulator = ScoringSimulator(enable_feature_extraction=True)
        self.scenario_generator = ScenarioGenerator(self.simulator)
        
        # Model paths
        self.model_dir = "models"
        self.role_classifier_path = os.path.join(self.model_dir, f"{model_name}_role_classifier.pkl")
        self.synergy_predictor_path = os.path.join(self.model_dir, f"{model_name}_synergy_predictor.pkl")
        self.scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.pkl")
        self.encoders_path = os.path.join(self.model_dir, f"{model_name}_encoders.pkl")
        self.config_path = os.path.join(self.model_dir, f"{model_name}_config.json")
        
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Coordination rules and heuristics
        self.coordination_rules = self._initialize_coordination_rules()
        
        # Task compatibility matrix
        self.task_compatibility = self._create_task_compatibility_matrix()
        
    def _initialize_coordination_rules(self) -> Dict[str, Any]:
        """Initialize reinforcement learning-inspired coordination rules"""
        return {
            # Reward values for different coordination patterns
            'rewards': {
                'complementary_tasks': 0.8,  # Different but compatible tasks
                'spatial_efficiency': 0.6,   # Good field positioning
                'goal_specialization': 0.7,  # Each robot focuses on specific goals
                'zone_synergy': 0.5,         # Coordinated zone control
                'role_balance': 0.6,         # Balanced offensive/defensive roles
                'task_overlap_penalty': -0.4 # Penalty for redundant tasks
            },
            
            # State-action values for different scenarios
            'q_values': {
                'high_pressure': {
                    'divide_and_conquer': 0.8,
                    'double_team': 0.3,
                    'specialized_roles': 0.7
                },
                'even_match': {
                    'divide_and_conquer': 0.6,
                    'double_team': 0.6,
                    'specialized_roles': 0.8
                },
                'ahead': {
                    'divide_and_conquer': 0.7,
                    'double_team': 0.4,
                    'specialized_roles': 0.6
                },
                'behind': {
                    'divide_and_conquer': 0.5,
                    'double_team': 0.8,
                    'specialized_roles': 0.4
                }
            }
        }
    
    def _create_task_compatibility_matrix(self) -> np.ndarray:
        """Create compatibility matrix between different tasks"""
        tasks = list(TaskType)
        n_tasks = len(tasks)
        matrix = np.ones((n_tasks, n_tasks))
        
        # Define incompatibilities and synergies
        incompatibilities = [
            (TaskType.PRIMARY_SCORER, TaskType.SECONDARY_SCORER),  # Both can't be primary
            (TaskType.DEFENDER, TaskType.PRIMARY_SCORER),         # Conflict in priorities
        ]
        
        synergies = [
            (TaskType.PRIMARY_SCORER, TaskType.SUPPORT),          # Good combination
            (TaskType.ZONE_CONTROLLER, TaskType.DEFENDER),        # Natural pairing
            (TaskType.LOADER, TaskType.SECONDARY_SCORER),         # Supply chain
        ]
        
        for task1, task2 in incompatibilities:
            i, j = tasks.index(task1), tasks.index(task2)
            matrix[i][j] = matrix[j][i] = 0.3  # Low compatibility
        
        for task1, task2 in synergies:
            i, j = tasks.index(task1), tasks.index(task2)
            matrix[i][j] = matrix[j][i] = 0.9  # High synergy
        
        return matrix
    
    def generate_coordination_training_data(self, num_samples: int = 2000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate training data for coordination models"""
        print(f"Generating {num_samples} coordination training samples...")
        
        features_list = []
        role_labels = []
        synergy_scores = []
        
        for i in range(num_samples):
            if i % 200 == 0:
                print(f"Generated {i}/{num_samples} samples...")
            
            # Generate random scenario
            skill_level = np.random.choice(list(SkillLevel))
            strategy_type = np.random.choice(list(StrategyType))
            
            # Create coordination scenario
            scenario_params = self.scenario_generator._create_scenario_parameters(
                skill_level, strategy_type, f"Alliance_{i}"
            )
            
            # Generate strategy and opponent
            strategy = self.scenario_generator.generate_time_based_strategy(
                f"Alliance_{i}", scenario_params
            )
            opponent = self.scenario_generator.generate_random_strategy("Opponent")
            
            # Simulate match
            result = self.simulator.simulate_match(strategy, opponent, extract_features=True)
            
            if result.red_features is not None:
                # Extract coordination features
                coord_features = self._extract_coordination_features(
                    result.red_features, scenario_params, strategy
                )
                
                # Generate optimal coordination strategy
                optimal_coordination = self._determine_optimal_coordination(
                    scenario_params, result.red_score, result.blue_score
                )
                
                # Calculate synergy score
                synergy = self._calculate_synergy_score(
                    scenario_params, coord_features, result.red_score
                )
                
                features_list.append(coord_features)
                role_labels.append(optimal_coordination.value)
                synergy_scores.append(synergy)
        
        print(f"Successfully generated {len(features_list)} coordination samples")
        
        X = np.array(features_list, dtype=np.float32)
        y_roles = np.array(role_labels)
        y_synergy = np.array(synergy_scores, dtype=np.float32)
        
        return X, y_roles, y_synergy
    
    def _extract_coordination_features(
        self, 
        game_features: Dict[str, float], 
        scenario_params, 
        strategy: AllianceStrategy
    ) -> np.ndarray:
        """Extract features relevant to coordination analysis"""
        features = []
        
        # Robot coordination features
        features.append(game_features.get('red_robot_coordination_distance', 10.0))
        features.append(game_features.get('red_task_allocation_efficiency', 0.5))
        features.append(game_features.get('red_size_diversity', 0.0))
        
        # Scoring distribution features
        total_auto = sum(strategy.blocks_scored_auto.values())
        total_driver = sum(strategy.blocks_scored_driver.values())
        features.append(total_auto)
        features.append(total_driver)
        features.append(total_auto / max(1, total_auto + total_driver))  # Auto ratio
        
        # Goal distribution variance (measure of specialization)
        auto_scores = list(strategy.blocks_scored_auto.values())
        driver_scores = list(strategy.blocks_scored_driver.values())
        features.append(np.var(auto_scores))
        features.append(np.var(driver_scores))
        
        # Zone and parking coordination
        features.append(len(strategy.zones_controlled))
        parking_variety = len(set(strategy.robots_parked))
        features.append(parking_variety)
        
        # Strategic indicators
        features.append(game_features.get('red_neutral_zone_presence', 0.0))
        features.append(game_features.get('red_defensive_positioning', 0.0))
        features.append(game_features.get('red_high_value_goal_focus', 0.0))
        
        # Temporal features
        features.append(game_features.get('red_match_progress', 0.5))
        features.append(game_features.get('red_scoring_rate_10s', 1.0))
        features.append(game_features.get('red_score_momentum', 0.0))
        
        # Role-specific features
        robot1_role_encoding = list(RobotRole).index(scenario_params.robot1_role)
        robot2_role_encoding = list(RobotRole).index(scenario_params.robot2_role)
        features.append(robot1_role_encoding)
        features.append(robot2_role_encoding)
        
        # Skill level
        skill_encoding = list(SkillLevel).index(scenario_params.skill_level)
        features.append(skill_encoding)
        
        return np.array(features, dtype=np.float32)
    
    def _determine_optimal_coordination(
        self, 
        scenario_params, 
        red_score: int, 
        blue_score: int
    ) -> CoordinationStrategy:
        """Determine optimal coordination strategy based on RL principles"""
        
        # Determine match state
        score_diff = red_score - blue_score
        if score_diff > 20:
            match_state = 'ahead'
        elif score_diff < -20:
            match_state = 'behind'
        elif abs(score_diff) <= 10:
            match_state = 'even_match'
        else:
            match_state = 'high_pressure'
        
        # Get Q-values for current state
        q_values = self.coordination_rules['q_values'].get(match_state, {})
        
        # Add skill-based adjustments
        skill_bonus = {
            SkillLevel.BEGINNER: -0.1,
            SkillLevel.INTERMEDIATE: 0.0,
            SkillLevel.ADVANCED: 0.1,
            SkillLevel.EXPERT: 0.2
        }
        
        skill_adj = skill_bonus.get(scenario_params.skill_level, 0.0)
        
        # Adjust Q-values based on robot roles
        adjusted_q_values = {}
        for strategy, value in q_values.items():
            adjusted_value = value + skill_adj
            
            # Role-specific adjustments
            if scenario_params.robot1_role == scenario_params.robot2_role:
                if strategy == 'specialized_roles':
                    adjusted_value -= 0.2  # Penalty for same roles in specialized strategy
                elif strategy == 'double_team':
                    adjusted_value += 0.1  # Bonus for same roles in double team
            
            adjusted_q_values[strategy] = adjusted_value
        
        # Select strategy with highest Q-value
        if adjusted_q_values:
            best_strategy = max(adjusted_q_values.items(), key=lambda x: x[1])[0]
            return CoordinationStrategy(best_strategy)
        
        # Default fallback
        return CoordinationStrategy.SPECIALIZED_ROLES
    
    def _calculate_synergy_score(
        self, 
        scenario_params, 
        coord_features: np.ndarray, 
        actual_score: int
    ) -> float:
        """Calculate synergy score using RL reward function"""
        base_score = 100  # Expected base score
        score_bonus = (actual_score - base_score) / 100.0  # Normalized score bonus
        
        # Extract relevant features
        coordination_distance = coord_features[0]
        task_efficiency = coord_features[1]
        size_diversity = coord_features[2]
        goal_specialization = coord_features[6] + coord_features[7]  # Goal variance
        
        # Calculate reward components
        rewards = self.coordination_rules['rewards']
        
        # Spatial efficiency reward
        spatial_reward = rewards['spatial_efficiency'] if coordination_distance < 12 else -0.2
        
        # Task efficiency reward
        task_reward = rewards['complementary_tasks'] * task_efficiency
        
        # Specialization reward
        specialization_reward = rewards['goal_specialization'] * min(goal_specialization / 5.0, 1.0)
        
        # Size diversity reward
        diversity_reward = rewards['role_balance'] * size_diversity
        
        # Role compatibility reward
        robot1_role = scenario_params.robot1_role
        robot2_role = scenario_params.robot2_role
        task_matrix = self.task_compatibility
        
        # Map roles to task types (simplified)
        role_to_task = {
            RobotRole.SCORER: TaskType.PRIMARY_SCORER,
            RobotRole.DEFENDER: TaskType.DEFENDER,
            RobotRole.SUPPORT: TaskType.SUPPORT,
            RobotRole.HYBRID: TaskType.HYBRID
        }
        
        task1 = role_to_task.get(robot1_role, TaskType.HYBRID)
        task2 = role_to_task.get(robot2_role, TaskType.HYBRID)
        
        task1_idx = list(TaskType).index(task1)
        task2_idx = list(TaskType).index(task2)
        compatibility_reward = task_matrix[task1_idx][task2_idx] * 0.3
        
        # Combine all rewards
        total_synergy = (
            score_bonus + spatial_reward + task_reward + 
            specialization_reward + diversity_reward + compatibility_reward
        )
        
        return max(0.0, min(1.0, total_synergy))  # Bound between 0 and 1
    
    def train_coordination_models(self, X: np.ndarray, y_roles: np.ndarray, y_synergy: np.ndarray):
        """Train role classification and synergy prediction models"""
        print("Training coordination models...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        y_roles_encoded = self.strategy_encoder.fit_transform(y_roles)
        
        # Split data
        X_train, X_test, y_roles_train, y_roles_test = train_test_split(
            X_scaled, y_roles_encoded, test_size=0.2, random_state=42
        )
        
        y_synergy_train, y_synergy_test = train_test_split(
            y_synergy, test_size=0.2, random_state=42
        )[0], train_test_split(
            y_synergy, test_size=0.2, random_state=42
        )[1]
        
        # Train role classifier
        print("Training role classifier...")
        self.role_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.role_classifier.fit(X_train, y_roles_train)
        
        # Train synergy predictor
        print("Training synergy predictor...")
        self.synergy_predictor = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.synergy_predictor.fit(X_train, y_synergy_train)
        
        # Evaluate models
        role_pred = self.role_classifier.predict(X_test)
        role_accuracy = accuracy_score(y_roles_test, role_pred)
        
        synergy_pred = self.synergy_predictor.predict(X_test)
        synergy_mse = np.mean((y_synergy_test - synergy_pred) ** 2)
        synergy_r2 = self.synergy_predictor.score(X_test, y_synergy_test)
        
        print(f"Role Classification Accuracy: {role_accuracy:.3f}")
        print(f"Synergy Prediction MSE: {synergy_mse:.3f}")
        print(f"Synergy Prediction R²: {synergy_r2:.3f}")
        
        # Save models
        self.save_models()
        
        return {
            'role_accuracy': role_accuracy,
            'synergy_mse': synergy_mse,
            'synergy_r2': synergy_r2
        }
    
    def optimize_robot_coordination(
        self, 
        game_state: GameState, 
        alliance: str = "red",
        available_goals: Optional[List[str]] = None
    ) -> CoordinationPlan:
        """Optimize robot coordination using RL-inspired approach"""
        
        if available_goals is None:
            available_goals = ["long_1", "long_2", "center_upper", "center_lower"]
        
        # Extract current features
        features = self.feature_extractor.extract_all_features(game_state, alliance)
        
        # Get robot states
        robots = game_state.red_robots if alliance == "red" else game_state.blue_robots
        
        if len(robots) < 2:
            raise ValueError("Need at least 2 robots for coordination")
        
        robot1, robot2 = robots[0], robots[1]
        
        # Generate all possible coordination strategies
        coordination_options = []
        
        for strategy_type in CoordinationStrategy:
            plan = self._create_coordination_plan(
                strategy_type, robot1, robot2, game_state, features, available_goals
            )
            coordination_options.append(plan)
        
        # Select best coordination plan
        best_plan = max(coordination_options, key=lambda p: p.synergy_score)
        
        return best_plan
    
    def _create_coordination_plan(
        self,
        strategy_type: CoordinationStrategy,
        robot1: RobotState,
        robot2: RobotState,
        game_state: GameState,
        features: Dict[str, float],
        available_goals: List[str]
    ) -> CoordinationPlan:
        """Create specific coordination plan"""
        
        # Determine robot assignments based on strategy type
        if strategy_type == CoordinationStrategy.DIVIDE_AND_CONQUER:
            robot1_assignment = RobotAssignment(
                robot_id=robot1.robot_id,
                primary_task=TaskType.PRIMARY_SCORER,
                secondary_task=None,
                assigned_goals=available_goals[:2],  # First half of goals
                assigned_zones=[Zone.NEUTRAL],
                priority=1,
                expected_contribution=60.0
            )
            
            robot2_assignment = RobotAssignment(
                robot_id=robot2.robot_id,
                primary_task=TaskType.SECONDARY_SCORER,
                secondary_task=TaskType.ZONE_CONTROLLER,
                assigned_goals=available_goals[2:],  # Second half of goals
                assigned_zones=[Zone.RED_HOME],
                priority=2,
                expected_contribution=40.0
            )
        
        elif strategy_type == CoordinationStrategy.DOUBLE_TEAM:
            robot1_assignment = RobotAssignment(
                robot_id=robot1.robot_id,
                primary_task=TaskType.PRIMARY_SCORER,
                secondary_task=TaskType.SUPPORT,
                assigned_goals=available_goals[:2],  # Focus on same goals
                assigned_zones=[],
                priority=1,
                expected_contribution=55.0
            )
            
            robot2_assignment = RobotAssignment(
                robot_id=robot2.robot_id,
                primary_task=TaskType.SUPPORT,
                secondary_task=TaskType.SECONDARY_SCORER,
                assigned_goals=available_goals[:2],  # Same goals as robot1
                assigned_zones=[],
                priority=1,
                expected_contribution=45.0
            )
        
        elif strategy_type == CoordinationStrategy.SPECIALIZED_ROLES:
            # Assign based on robot size
            if robot1.size == RobotSize.LARGE:
                robot1_task = TaskType.PRIMARY_SCORER
                robot2_task = TaskType.ZONE_CONTROLLER
            else:
                robot1_task = TaskType.ZONE_CONTROLLER
                robot2_task = TaskType.PRIMARY_SCORER
            
            robot1_assignment = RobotAssignment(
                robot_id=robot1.robot_id,
                primary_task=robot1_task,
                secondary_task=None,
                assigned_goals=available_goals if robot1_task == TaskType.PRIMARY_SCORER else [],
                assigned_zones=[Zone.NEUTRAL] if robot1_task == TaskType.ZONE_CONTROLLER else [],
                priority=1,
                expected_contribution=70.0 if robot1_task == TaskType.PRIMARY_SCORER else 30.0
            )
            
            robot2_assignment = RobotAssignment(
                robot_id=robot2.robot_id,
                primary_task=robot2_task,
                secondary_task=TaskType.DEFENDER,
                assigned_goals=available_goals if robot2_task == TaskType.PRIMARY_SCORER else available_goals[2:],
                assigned_zones=[Zone.RED_HOME, Zone.BLUE_HOME] if robot2_task == TaskType.ZONE_CONTROLLER else [],
                priority=1,
                expected_contribution=70.0 if robot2_task == TaskType.PRIMARY_SCORER else 30.0
            )
        
        elif strategy_type == CoordinationStrategy.SUPPORT_SCORER:
            robot1_assignment = RobotAssignment(
                robot_id=robot1.robot_id,
                primary_task=TaskType.PRIMARY_SCORER,
                secondary_task=None,
                assigned_goals=available_goals,
                assigned_zones=[],
                priority=1,
                expected_contribution=75.0
            )
            
            robot2_assignment = RobotAssignment(
                robot_id=robot2.robot_id,
                primary_task=TaskType.LOADER,
                secondary_task=TaskType.SUPPORT,
                assigned_goals=[],
                assigned_zones=[Zone.NEUTRAL],
                priority=2,
                expected_contribution=25.0
            )
        
        else:  # DYNAMIC_SWITCHING
            robot1_assignment = RobotAssignment(
                robot_id=robot1.robot_id,
                primary_task=TaskType.HYBRID,
                secondary_task=TaskType.PRIMARY_SCORER,
                assigned_goals=available_goals[:3],
                assigned_zones=[Zone.NEUTRAL],
                priority=1,
                expected_contribution=50.0
            )
            
            robot2_assignment = RobotAssignment(
                robot_id=robot2.robot_id,
                primary_task=TaskType.HYBRID,
                secondary_task=TaskType.ZONE_CONTROLLER,
                assigned_goals=available_goals[1:],  # Overlapping goals
                assigned_zones=[Zone.RED_HOME],
                priority=1,
                expected_contribution=50.0
            )
        
        # Calculate coordination metrics
        metrics = self._calculate_coordination_metrics(
            robot1, robot2, robot1_assignment, robot2_assignment, game_state
        )
        
        # Calculate synergy score
        synergy_score = self._calculate_plan_synergy(
            strategy_type, robot1_assignment, robot2_assignment, metrics, features
        )
        
        # Estimate total expected score
        expected_score = robot1_assignment.expected_contribution + robot2_assignment.expected_contribution
        
        # Assess risk
        risk_level = self._assess_coordination_risk(strategy_type, metrics, features)
        
        return CoordinationPlan(
            strategy_type=strategy_type,
            robot1_assignment=robot1_assignment,
            robot2_assignment=robot2_assignment,
            synergy_score=synergy_score,
            expected_total_score=expected_score,
            risk_level=risk_level,
            coordination_efficiency=metrics.distance_efficiency,
            task_balance_score=abs(robot1_assignment.expected_contribution - robot2_assignment.expected_contribution) / 100.0
        )
    
    def _calculate_coordination_metrics(
        self,
        robot1: RobotState,
        robot2: RobotState,
        assignment1: RobotAssignment,
        assignment2: RobotAssignment,
        game_state: GameState
    ) -> CoordinationMetrics:
        """Calculate detailed coordination metrics"""
        
        # Distance efficiency
        distance = np.sqrt((robot1.position[0] - robot2.position[0])**2 + 
                          (robot1.position[1] - robot2.position[1])**2)
        distance_efficiency = max(0, 1 - distance / 20.0)  # Normalized by field size
        
        # Task overlap
        goal_overlap = len(set(assignment1.assigned_goals) & set(assignment2.assigned_goals))
        zone_overlap = len(set(assignment1.assigned_zones) & set(assignment2.assigned_zones))
        task_overlap = (goal_overlap + zone_overlap) / max(1, len(assignment1.assigned_goals) + len(assignment1.assigned_zones))
        
        # Goal coverage
        all_goals = ["long_1", "long_2", "center_upper", "center_lower"]
        covered_goals = set(assignment1.assigned_goals) | set(assignment2.assigned_goals)
        goal_coverage = len(covered_goals) / len(all_goals)
        
        # Zone control effectiveness
        all_zones = [Zone.RED_HOME, Zone.BLUE_HOME, Zone.NEUTRAL]
        covered_zones = set(assignment1.assigned_zones) | set(assignment2.assigned_zones)
        zone_effectiveness = len(covered_zones) / len(all_zones)
        
        # Role specialization
        primary_tasks = [assignment1.primary_task, assignment2.primary_task]
        role_specialization = 1.0 if len(set(primary_tasks)) == 2 else 0.5
        
        # Communication requirements (inverse of task complexity)
        communication_req = (task_overlap + abs(assignment1.priority - assignment2.priority)) / 2.0
        
        return CoordinationMetrics(
            distance_efficiency=distance_efficiency,
            task_overlap=task_overlap,
            goal_coverage=goal_coverage,
            zone_control_effectiveness=zone_effectiveness,
            role_specialization=role_specialization,
            communication_requirements=communication_req
        )
    
    def _calculate_plan_synergy(
        self,
        strategy_type: CoordinationStrategy,
        assignment1: RobotAssignment,
        assignment2: RobotAssignment,
        metrics: CoordinationMetrics,
        features: Dict[str, float]
    ) -> float:
        """Calculate overall synergy score for coordination plan"""
        
        # Base synergy from strategy type
        strategy_bonuses = {
            CoordinationStrategy.DIVIDE_AND_CONQUER: 0.7,
            CoordinationStrategy.DOUBLE_TEAM: 0.6,
            CoordinationStrategy.SPECIALIZED_ROLES: 0.8,
            CoordinationStrategy.DYNAMIC_SWITCHING: 0.5,
            CoordinationStrategy.SUPPORT_SCORER: 0.6
        }
        
        base_synergy = strategy_bonuses.get(strategy_type, 0.5)
        
        # Metric-based adjustments
        distance_bonus = metrics.distance_efficiency * 0.2
        coverage_bonus = metrics.goal_coverage * 0.3
        specialization_bonus = metrics.role_specialization * 0.2
        overlap_penalty = metrics.task_overlap * -0.1
        
        # Feature-based adjustments
        coordination_feature = features.get('red_robot_coordination_distance', 10.0)
        coordination_bonus = max(0, (15 - coordination_feature) / 15.0) * 0.2
        
        task_efficiency = features.get('red_task_allocation_efficiency', 0.5)
        efficiency_bonus = task_efficiency * 0.1
        
        # Combine all factors
        total_synergy = (
            base_synergy + distance_bonus + coverage_bonus + 
            specialization_bonus + overlap_penalty + 
            coordination_bonus + efficiency_bonus
        )
        
        return max(0.0, min(1.0, total_synergy))
    
    def _assess_coordination_risk(
        self,
        strategy_type: CoordinationStrategy,
        metrics: CoordinationMetrics,
        features: Dict[str, float]
    ) -> str:
        """Assess risk level of coordination plan"""
        
        risk_factors = []
        
        if metrics.distance_efficiency < 0.5:
            risk_factors.append("Poor spatial coordination")
        
        if metrics.task_overlap > 0.5:
            risk_factors.append("High task redundancy")
        
        if metrics.communication_requirements > 0.7:
            risk_factors.append("Complex communication needs")
        
        if features.get('red_score_momentum', 0) < -0.5:
            risk_factors.append("Negative momentum")
        
        if strategy_type == CoordinationStrategy.DYNAMIC_SWITCHING:
            risk_factors.append("Strategy complexity")
        
        if len(risk_factors) == 0:
            return "Low"
        elif len(risk_factors) <= 2:
            return "Medium"
        else:
            return "High"
    
    def calculate_role_synergy(
        self, 
        robot1_role: TaskType, 
        robot2_role: TaskType, 
        field_context: Dict[str, float]
    ) -> float:
        """Calculate synergy between two robot roles"""
        
        # Get compatibility from matrix
        task_list = list(TaskType)
        role1_idx = task_list.index(robot1_role)
        role2_idx = task_list.index(robot2_role)
        base_compatibility = self.task_compatibility[role1_idx][role2_idx]
        
        # Context adjustments
        context_bonus = 0
        
        if field_context.get('field_crowding', 0) > 0.7:
            # High crowding favors specialized roles
            if robot1_role != robot2_role:
                context_bonus += 0.1
        
        if field_context.get('time_pressure', 0) > 0.8:
            # Time pressure favors coordinated roles
            if robot1_role in [TaskType.PRIMARY_SCORER] and robot2_role in [TaskType.SUPPORT, TaskType.LOADER]:
                context_bonus += 0.2
        
        if field_context.get('defensive_pressure', 0) > 0.6:
            # Defensive pressure favors protection roles
            if robot2_role == TaskType.DEFENDER:
                context_bonus += 0.15
        
        return min(1.0, base_compatibility + context_bonus)
    
    def save_models(self):
        """Save coordination models"""
        if self.role_classifier is not None:
            joblib.dump(self.role_classifier, self.role_classifier_path)
        
        if self.synergy_predictor is not None:
            joblib.dump(self.synergy_predictor, self.synergy_predictor_path)
        
        joblib.dump(self.scaler, self.scaler_path)
        
        # Save encoders
        encoders = {
            'task_encoder': self.task_encoder,
            'strategy_encoder': self.strategy_encoder
        }
        joblib.dump(encoders, self.encoders_path)
        
        # Save configuration
        config = {
            'model_name': self.model_name,
            'coordination_rules': self.coordination_rules,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        print(f"Coordination models saved to {self.model_dir}/")
    
    def load_models(self) -> bool:
        """Load pre-trained coordination models"""
        try:
            if os.path.exists(self.role_classifier_path):
                self.role_classifier = joblib.load(self.role_classifier_path)
            
            if os.path.exists(self.synergy_predictor_path):
                self.synergy_predictor = joblib.load(self.synergy_predictor_path)
            
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
            
            if os.path.exists(self.encoders_path):
                encoders = joblib.load(self.encoders_path)
                self.task_encoder = encoders['task_encoder']
                self.strategy_encoder = encoders['strategy_encoder']
            
            print("Coordination models loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading coordination models: {e}")
            return False


def train_coordination_model_pipeline(num_samples: int = 2000) -> VEXUCoordinationModel:
    """Complete training pipeline for coordination model"""
    model = VEXUCoordinationModel()
    
    # Generate training data
    X, y_roles, y_synergy = model.generate_coordination_training_data(num_samples)
    
    # Train models
    performance = model.train_coordination_models(X, y_roles, y_synergy)
    
    print(f"\nCoordination Model Training Results:")
    print(f"Role Classification Accuracy: {performance['role_accuracy']:.3f}")
    print(f"Synergy Prediction R²: {performance['synergy_r2']:.3f}")
    
    return model


if __name__ == "__main__":
    print("VEX U Coordination Model - Training Example")
    print("=" * 50)
    
    # Train the coordination model
    coord_model = train_coordination_model_pipeline(num_samples=500)  # Smaller sample for testing
    
    # Test coordination optimization
    from .feature_engineering import create_game_state_from_strategy
    
    test_strategy = AllianceStrategy(
        name="Test Strategy",
        blocks_scored_auto={"long_1": 5, "long_2": 4, "center_1": 3, "center_2": 3},
        blocks_scored_driver={"long_1": 10, "long_2": 8, "center_1": 6, "center_2": 7},
        zones_controlled=[Zone.RED_HOME, Zone.NEUTRAL],
        robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.PLATFORM]
    )
    
    opponent_strategy = AllianceStrategy(
        name="Opponent",
        blocks_scored_auto={"long_1": 4, "long_2": 3, "center_1": 2, "center_2": 2},
        blocks_scored_driver={"long_1": 8, "long_2": 9, "center_1": 5, "center_2": 6},
        zones_controlled=[Zone.BLUE_HOME],
        robots_parked=[ParkingLocation.ALLIANCE_ZONE, ParkingLocation.PLATFORM]
    )
    
    game_state = create_game_state_from_strategy(test_strategy, opponent_strategy)
    
    # Optimize coordination
    coordination_plan = coord_model.optimize_robot_coordination(game_state, "red")
    
    print(f"\nOptimal Coordination Plan:")
    print(f"Strategy: {coordination_plan.strategy_type.value}")
    print(f"Synergy Score: {coordination_plan.synergy_score:.3f}")
    print(f"Expected Score: {coordination_plan.expected_total_score:.1f}")
    print(f"Risk Level: {coordination_plan.risk_level}")
    
    print(f"\nRobot 1 Assignment:")
    print(f"  Primary Task: {coordination_plan.robot1_assignment.primary_task.value}")
    print(f"  Goals: {coordination_plan.robot1_assignment.assigned_goals}")
    print(f"  Expected Contribution: {coordination_plan.robot1_assignment.expected_contribution:.1f}")
    
    print(f"\nRobot 2 Assignment:")
    print(f"  Primary Task: {coordination_plan.robot2_assignment.primary_task.value}")
    print(f"  Goals: {coordination_plan.robot2_assignment.assigned_goals}")
    print(f"  Expected Contribution: {coordination_plan.robot2_assignment.expected_contribution:.1f}")
    
    # Test role synergy
    synergy = coord_model.calculate_role_synergy(
        TaskType.PRIMARY_SCORER,
        TaskType.SUPPORT,
        {'field_crowding': 0.5, 'time_pressure': 0.7, 'defensive_pressure': 0.3}
    )
    print(f"\nPrimary Scorer + Support Synergy: {synergy:.3f}")