#!/usr/bin/env python3

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import random
from itertools import combinations

try:
    from ..core.simulator import (
        AllianceStrategy, ScoringSimulator, MatchResult, Zone, ParkingLocation
    )
    from ..core.scenario_generator import ScenarioGenerator, SkillLevel, RobotRole, StrategyType
    
    # ML Models integration
    from ..ml_models.strategy_predictor import VEXUStrategyPredictor, StrategyPrediction
    from ..ml_models.scoring_optimizer import VEXUScoringOptimizer, ScoreOptimizationResult
    from ..ml_models.coordination_model import VEXUCoordinationModel, CoordinationPlan
    from ..ml_models.feature_engineering import VEXUFeatureExtractor, create_game_state_from_strategy
    ML_MODELS_AVAILABLE = True
except ImportError:
    # Fallback for when running from main.py
    from core.simulator import (
        AllianceStrategy, ScoringSimulator, MatchResult, Zone, ParkingLocation
    )
    from core.scenario_generator import ScenarioGenerator, SkillLevel, RobotRole, StrategyType
    
    # ML Models integration
    try:
        from ml_models.strategy_predictor import VEXUStrategyPredictor, StrategyPrediction
        from ml_models.scoring_optimizer import VEXUScoringOptimizer, ScoreOptimizationResult
        from ml_models.coordination_model import VEXUCoordinationModel, CoordinationPlan
        from ml_models.feature_engineering import VEXUFeatureExtractor, create_game_state_from_strategy
        ML_MODELS_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: ML models not available - {e}")
        ML_MODELS_AVAILABLE = False


class CoreStrategy(Enum):
    FAST_FURIOUS = "fast_furious"
    ZONE_CONTROL = "zone_control" 
    BALANCED = "balanced"
    DENY_DEFEND = "deny_defend"
    ENDGAME_FOCUS = "endgame_focus"


class CoordinationStrategy(Enum):
    DIVIDE_CONQUER = "divide_conquer"
    DOUBLE_TEAM = "double_team"
    SPECIALIZED_ROLES = "specialized_roles"
    DYNAMIC_SWITCHING = "dynamic_switching"


@dataclass
class StrategyMetrics:
    strategy_name: str
    avg_score: float
    win_rate: float
    avg_margin: float
    score_std: float
    risk_reward_ratio: float
    consistency_score: float
    component_breakdown: Dict[str, float]
    strengths: List[str]
    weaknesses: List[str]
    sample_size: int
    # New ML-enhanced fields
    ml_predicted_score: Optional[float] = None
    ml_coordination_score: Optional[float] = None
    ml_optimization_suggestions: Optional[List[str]] = None
    ml_strategy_confidence: Optional[float] = None


@dataclass
class MatchupResult:
    strategy_a: str
    strategy_b: str
    a_wins: int
    b_wins: int
    ties: int
    avg_margin: float
    a_avg_score: float
    b_avg_score: float
    total_matches: int
    confidence_interval: Tuple[float, float]


class AdvancedStrategyAnalyzer:
    def __init__(self, simulator: ScoringSimulator, enable_ml_models: bool = True):
        self.simulator = simulator
        self.generator = ScenarioGenerator(simulator)
        self.core_strategies = {}
        self.coordination_strategies = {}
        self.matchup_matrix = {}
        
        # ML Models integration
        self.enable_ml_models = enable_ml_models and ML_MODELS_AVAILABLE
        self.strategy_predictor = None
        self.scoring_optimizer = None
        self.coordination_model = None
        self.feature_extractor = None
        
        if self.enable_ml_models:
            try:
                self._initialize_ml_models()
            except Exception as e:
                print(f"Warning: Could not initialize ML models - {e}")
                self.enable_ml_models = False
    
    def _initialize_ml_models(self):
        """Initialize ML models for enhanced analysis"""
        print("Initializing ML models for enhanced strategy analysis...")
        
        self.strategy_predictor = VEXUStrategyPredictor()
        self.scoring_optimizer = VEXUScoringOptimizer()
        self.coordination_model = VEXUCoordinationModel()
        self.feature_extractor = VEXUFeatureExtractor()
        
        # Try to load pre-trained models
        predictor_loaded = self.strategy_predictor.load_model()
        optimizer_loaded = self.scoring_optimizer.load_models()
        coordination_loaded = self.coordination_model.load_models()
        
        if not (predictor_loaded or optimizer_loaded or coordination_loaded):
            print("No pre-trained models found. Models will need to be trained for enhanced analysis.")
        else:
            print("ML models initialized successfully.")
        
    def create_core_strategies(self) -> Dict[CoreStrategy, AllianceStrategy]:
        """Create the 5 core strategic approaches"""
        strategies = {}
        
        # 1. Fast and Furious - Maximum scoring rate
        strategies[CoreStrategy.FAST_FURIOUS] = AllianceStrategy(
            name="Fast and Furious",
            blocks_scored_auto={"long_1": 8, "long_2": 8, "center_1": 6, "center_2": 6},
            blocks_scored_driver={"long_1": 18, "long_2": 18, "center_1": 12, "center_2": 12},
            zones_controlled=[],  # No time for zones
            robots_parked=[ParkingLocation.NONE, ParkingLocation.NONE]  # No time for parking
        )
        
        # 2. Zone Control - Focus on controlling goal zones
        strategies[CoreStrategy.ZONE_CONTROL] = AllianceStrategy(
            name="Zone Control",
            blocks_scored_auto={"long_1": 4, "long_2": 4, "center_1": 3, "center_2": 3},
            blocks_scored_driver={"long_1": 6, "long_2": 6, "center_1": 6, "center_2": 6},
            zones_controlled=[Zone.RED_HOME, Zone.BLUE_HOME, Zone.NEUTRAL],  # All zones
            robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.ALLIANCE_ZONE]
        )
        
        # 3. Balanced - Mix of scoring and zone control
        strategies[CoreStrategy.BALANCED] = AllianceStrategy(
            name="Balanced",
            blocks_scored_auto={"long_1": 6, "long_2": 6, "center_1": 4, "center_2": 4},
            blocks_scored_driver={"long_1": 10, "long_2": 10, "center_1": 8, "center_2": 8},
            zones_controlled=[Zone.RED_HOME, Zone.NEUTRAL],
            robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.PLATFORM]
        )
        
        # 4. Deny and Defend - Block opponent scoring
        strategies[CoreStrategy.DENY_DEFEND] = AllianceStrategy(
            name="Deny and Defend",
            blocks_scored_auto={"long_1": 3, "long_2": 3, "center_1": 2, "center_2": 2},
            blocks_scored_driver={"long_1": 8, "long_2": 8, "center_1": 6, "center_2": 6},
            zones_controlled=[Zone.RED_HOME, Zone.BLUE_HOME],  # Defensive zones
            robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.PLATFORM]
        )
        
        # 5. Endgame Focus - Save energy for parking/final push
        strategies[CoreStrategy.ENDGAME_FOCUS] = AllianceStrategy(
            name="Endgame Focus",
            blocks_scored_auto={"long_1": 5, "long_2": 5, "center_1": 3, "center_2": 3},
            blocks_scored_driver={"long_1": 12, "long_2": 12, "center_1": 8, "center_2": 8},  # Strong finish
            zones_controlled=[Zone.RED_HOME],
            robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.PLATFORM]  # Both park
        )
        
        self.core_strategies = strategies
        return strategies
    
    def create_coordination_variations(
        self, 
        base_strategy: AllianceStrategy,
        coordination_type: CoordinationStrategy
    ) -> AllianceStrategy:
        """Create variations based on coordination strategy"""
        
        # Base totals
        total_auto = sum(base_strategy.blocks_scored_auto.values())
        total_driver = sum(base_strategy.blocks_scored_driver.values())
        
        if coordination_type == CoordinationStrategy.DIVIDE_CONQUER:
            # Split goals - focus on different areas
            auto_dist = {
                "long_1": total_auto // 2, "long_2": 0,
                "center_1": total_auto - total_auto // 2, "center_2": 0
            }
            driver_dist = {
                "long_1": total_driver // 2, "long_2": 0,
                "center_1": total_driver - total_driver // 2, "center_2": 0
            }
            
        elif coordination_type == CoordinationStrategy.DOUBLE_TEAM:
            # Both robots focus on same goals
            auto_dist = {
                "long_1": int(total_auto * 0.6), "long_2": int(total_auto * 0.4),
                "center_1": 0, "center_2": 0
            }
            driver_dist = {
                "long_1": int(total_driver * 0.6), "long_2": int(total_driver * 0.4),
                "center_1": 0, "center_2": 0
            }
            
        elif coordination_type == CoordinationStrategy.SPECIALIZED_ROLES:
            # One scorer, one supporter/defender
            auto_dist = {
                "long_1": int(total_auto * 0.7), "long_2": int(total_auto * 0.3),
                "center_1": 0, "center_2": 0
            }
            driver_dist = {
                "long_1": int(total_driver * 0.8), "long_2": int(total_driver * 0.2),
                "center_1": 0, "center_2": 0
            }
            # More zones for support role
            zones = list(base_strategy.zones_controlled) + [Zone.NEUTRAL]
            zones = list(set(zones))  # Remove duplicates
            
        else:  # DYNAMIC_SWITCHING
            # Adaptive distribution
            auto_dist = {
                "long_1": total_auto // 4, "long_2": total_auto // 4,
                "center_1": total_auto // 4, "center_2": total_auto - 3 * (total_auto // 4)
            }
            driver_dist = {
                "long_1": total_driver // 4, "long_2": total_driver // 4,
                "center_1": total_driver // 4, "center_2": total_driver - 3 * (total_driver // 4)
            }
        
        # Apply coordination-specific zone control
        if coordination_type == CoordinationStrategy.SPECIALIZED_ROLES:
            zones = list(base_strategy.zones_controlled)
            if Zone.NEUTRAL not in zones:
                zones.append(Zone.NEUTRAL)
        else:
            zones = base_strategy.zones_controlled
            
        return AllianceStrategy(
            name=f"{base_strategy.name} ({coordination_type.value.replace('_', ' ').title()})",
            blocks_scored_auto=auto_dist,
            blocks_scored_driver=driver_dist,
            zones_controlled=zones,
            robots_parked=base_strategy.robots_parked
        )
    
    def analyze_strategy_comprehensive(
        self,
        strategy: AllianceStrategy,
        num_opponents: int = 1000,
        opponent_variety: bool = True,
        include_ml_analysis: bool = True
    ) -> StrategyMetrics:
        """Comprehensive strategy analysis with Monte Carlo simulation"""
        
        results = []
        wins = 0
        margins = []
        component_totals = {
            'blocks': 0,
            'autonomous': 0,
            'zones': 0,
            'parking': 0
        }
        
        print(f"Analyzing {strategy.name} with {num_opponents} simulations...")
        
        # Generate diverse opponents if requested
        for i in range(num_opponents):
            if opponent_variety:
                # Mix of skill levels and strategies
                if i < num_opponents * 0.2:  # 20% expert opponents
                    opponent_skill = SkillLevel.EXPERT
                elif i < num_opponents * 0.5:  # 30% advanced
                    opponent_skill = SkillLevel.ADVANCED
                elif i < num_opponents * 0.8:  # 30% intermediate
                    opponent_skill = SkillLevel.INTERMEDIATE
                else:  # 20% beginner
                    opponent_skill = SkillLevel.BEGINNER
                    
                # Random strategy type
                opponent_strat_type = random.choice(list(StrategyType))
                
                # Generate realistic opponent
                opponent_params = self.generator._create_scenario_parameters(
                    opponent_skill, opponent_strat_type, "Opponent"
                )
                opponent = self.generator.generate_time_based_strategy("Opponent", opponent_params)
            else:
                # Use random opponents
                opponent = self.generator.generate_random_strategy("Opponent")
            
            # Simulate match
            result = self.simulator.simulate_match(strategy, opponent)
            
            if result.winner == "red":
                wins += 1
                margins.append(result.margin)
            elif result.winner == "blue":
                margins.append(-result.margin)
            else:
                margins.append(0)
            
            results.append(result.red_score)
            
            # Track component contributions
            for component, points in result.red_breakdown.items():
                if component in component_totals:
                    component_totals[component] += points
        
        # Calculate metrics
        avg_score = np.mean(results)
        score_std = np.std(results)
        win_rate = wins / num_opponents
        avg_margin = np.mean(margins)
        
        # Risk/reward ratio (higher score / lower variance is better)
        risk_reward_ratio = avg_score / (score_std + 1)  # +1 to avoid division by zero
        
        # Consistency score (inverse of coefficient of variation)
        consistency_score = 1 / (score_std / avg_score) if avg_score > 0 else 0
        
        # Component breakdown (as percentage of total average score)
        component_breakdown = {}
        for component, total in component_totals.items():
            component_breakdown[component] = (total / num_opponents) / avg_score * 100
        
        # Identify strengths and weaknesses
        strengths, weaknesses = self._identify_strengths_weaknesses_advanced(
            strategy, component_breakdown, avg_score, win_rate, consistency_score
        )
        
        # ML-enhanced analysis
        ml_predicted_score = None
        ml_coordination_score = None
        ml_optimization_suggestions = None
        ml_strategy_confidence = None
        
        if include_ml_analysis and self.enable_ml_models:
            ml_results = self._perform_ml_analysis(strategy, results[:10])  # Sample for ML analysis
            ml_predicted_score = ml_results.get('predicted_score')
            ml_coordination_score = ml_results.get('coordination_score')
            ml_optimization_suggestions = ml_results.get('optimization_suggestions', [])
            ml_strategy_confidence = ml_results.get('strategy_confidence')
        
        return StrategyMetrics(
            strategy_name=strategy.name,
            avg_score=avg_score,
            win_rate=win_rate,
            avg_margin=avg_margin,
            score_std=score_std,
            risk_reward_ratio=risk_reward_ratio,
            consistency_score=consistency_score,
            component_breakdown=component_breakdown,
            strengths=strengths,
            weaknesses=weaknesses,
            sample_size=num_opponents,
            ml_predicted_score=ml_predicted_score,
            ml_coordination_score=ml_coordination_score,
            ml_optimization_suggestions=ml_optimization_suggestions,
            ml_strategy_confidence=ml_strategy_confidence
        )
    
    def _identify_strengths_weaknesses_advanced(
        self,
        strategy: AllianceStrategy,
        component_breakdown: Dict[str, float],
        avg_score: float,
        win_rate: float,
        consistency_score: float
    ) -> Tuple[List[str], List[str]]:
        """Advanced strength/weakness identification"""
        strengths = []
        weaknesses = []
        
        # Analyze scoring pattern
        total_blocks = sum(strategy.blocks_scored_auto.values()) + \
                      sum(strategy.blocks_scored_driver.values())
        auto_blocks = sum(strategy.blocks_scored_auto.values())
        
        # Strengths
        if avg_score > 150:
            strengths.append("High scoring potential")
        if win_rate > 0.6:
            strengths.append("Strong competitive performance")
        if consistency_score > 1.5:
            strengths.append("Highly consistent results")
        if component_breakdown.get('blocks', 0) > 70:
            strengths.append("Excellent block scoring efficiency")
        if len(strategy.zones_controlled) >= 2:
            strengths.append("Good zone control strategy")
        if auto_blocks > 15:
            strengths.append("Strong autonomous program")
        if sum(1 for p in strategy.robots_parked if p == ParkingLocation.PLATFORM) == 2:
            strengths.append("Maximizes parking points")
        
        # Weaknesses
        if avg_score < 100:
            weaknesses.append("Low scoring output")
        if win_rate < 0.4:
            weaknesses.append("Poor competitive performance")
        if consistency_score < 0.8:
            weaknesses.append("Inconsistent results")
        if component_breakdown.get('blocks', 0) < 50:
            weaknesses.append("Inefficient block scoring")
        if len(strategy.zones_controlled) == 0:
            weaknesses.append("No zone control")
        if auto_blocks < 8:
            weaknesses.append("Weak autonomous performance")
        if all(p == ParkingLocation.NONE for p in strategy.robots_parked):
            weaknesses.append("Missing parking points")
        
        return strengths, weaknesses
    
    def _perform_ml_analysis(self, strategy: AllianceStrategy, sample_results: List[int]) -> Dict[str, Any]:
        """Perform ML-enhanced analysis on strategy"""
        ml_results = {}
        
        try:
            # Create a representative game state for ML analysis
            opponent = self.generator.generate_random_strategy("Sample_Opponent")
            game_state = create_game_state_from_strategy(strategy, opponent)
            
            # Strategy prediction
            if self.strategy_predictor is not None and self.strategy_predictor.model is not None:
                try:
                    prediction = self.strategy_predictor.predict_strategy(game_state, "red")
                    ml_results['strategy_confidence'] = prediction.confidence
                    ml_results['predicted_strategy'] = prediction.predicted_strategy
                except Exception as e:
                    print(f"Strategy prediction failed: {e}")
            
            # Score optimization
            if self.scoring_optimizer is not None and self.scoring_optimizer.ensemble_model is not None:
                try:
                    optimization = self.scoring_optimizer.predict_score(game_state, "red")
                    ml_results['predicted_score'] = optimization.predicted_score
                    ml_results['optimization_suggestions'] = optimization.optimization_suggestions[:3]  # Top 3
                    ml_results['win_probability'] = optimization.expected_win_probability
                except Exception as e:
                    print(f"Score optimization failed: {e}")
            
            # Coordination analysis
            if self.coordination_model is not None:
                try:
                    coordination_plan = self.coordination_model.optimize_robot_coordination(game_state, "red")
                    ml_results['coordination_score'] = coordination_plan.synergy_score
                    ml_results['coordination_strategy'] = coordination_plan.strategy_type.value
                    ml_results['coordination_risk'] = coordination_plan.risk_level
                except Exception as e:
                    print(f"Coordination analysis failed: {e}")
        
        except Exception as e:
            print(f"ML analysis failed: {e}")
        
        return ml_results
    
    def train_ml_models(self, num_samples: int = 3000) -> Dict[str, bool]:
        """Train all ML models with synthetic data"""
        if not self.enable_ml_models:
            print("ML models not available")
            return {'strategy_predictor': False, 'scoring_optimizer': False, 'coordination_model': False}
        
        print(f"Training ML models with {num_samples} samples...")
        results = {}
        
        # Train strategy predictor
        try:
            X, y = self.strategy_predictor.generate_synthetic_training_data(num_samples)
            metrics = self.strategy_predictor.train_model(X, y)
            results['strategy_predictor'] = True
            print(f"Strategy Predictor trained - Accuracy: {metrics.accuracy:.3f}")
        except Exception as e:
            print(f"Failed to train strategy predictor: {e}")
            results['strategy_predictor'] = False
        
        # Train scoring optimizer
        try:
            X, y = self.scoring_optimizer.generate_training_data(num_samples)
            performance = self.scoring_optimizer.train_models(X, y, optimize_hyperparameters=False)
            results['scoring_optimizer'] = True
            print(f"Scoring Optimizer trained - R²: {performance.r2_score:.3f}")
        except Exception as e:
            print(f"Failed to train scoring optimizer: {e}")
            results['scoring_optimizer'] = False
        
        # Train coordination model
        try:
            X, y_roles, y_synergy = self.coordination_model.generate_coordination_training_data(num_samples)
            performance = self.coordination_model.train_coordination_models(X, y_roles, y_synergy)
            results['coordination_model'] = True
            print(f"Coordination Model trained - Role Accuracy: {performance['role_accuracy']:.3f}")
        except Exception as e:
            print(f"Failed to train coordination model: {e}")
            results['coordination_model'] = False
        
        return results
    
    def get_ml_strategy_recommendations(self, strategy: AllianceStrategy) -> Dict[str, Any]:
        """Get comprehensive ML-based strategy recommendations"""
        if not self.enable_ml_models:
            return {'error': 'ML models not available'}
        
        try:
            # Create game state
            opponent = self.generator.generate_random_strategy("Analysis_Opponent")
            game_state = create_game_state_from_strategy(strategy, opponent)
            
            recommendations = {
                'strategy_analysis': {},
                'score_optimization': {},
                'coordination_plan': {},
                'overall_assessment': ''
            }
            
            # Strategy prediction analysis
            if self.strategy_predictor and self.strategy_predictor.model:
                pred = self.strategy_predictor.predict_strategy(game_state, "red")
                recommendations['strategy_analysis'] = {
                    'predicted_strategy': pred.predicted_strategy,
                    'confidence': pred.confidence,
                    'robot_roles': [pred.robot1_role, pred.robot2_role],
                    'adjustments': pred.recommended_adjustments
                }
            
            # Score optimization
            if self.scoring_optimizer and self.scoring_optimizer.ensemble_model:
                opt = self.scoring_optimizer.predict_score(game_state, "red")
                recommendations['score_optimization'] = {
                    'predicted_score': opt.predicted_score,
                    'confidence_interval': opt.confidence_interval,
                    'win_probability': opt.expected_win_probability,
                    'suggestions': opt.optimization_suggestions,
                    'risk_level': opt.risk_assessment
                }
            
            # Coordination optimization
            if self.coordination_model:
                coord = self.coordination_model.optimize_robot_coordination(game_state, "red")
                recommendations['coordination_plan'] = {
                    'optimal_strategy': coord.strategy_type.value,
                    'synergy_score': coord.synergy_score,
                    'expected_score': coord.expected_total_score,
                    'risk_level': coord.risk_level,
                    'robot1_role': coord.robot1_assignment.primary_task.value,
                    'robot2_role': coord.robot2_assignment.primary_task.value
                }
            
            # Overall assessment
            if recommendations['score_optimization']:
                score_pred = recommendations['score_optimization']['predicted_score']
                win_prob = recommendations['score_optimization']['win_probability']
                
                if score_pred > 130 and win_prob > 0.7:
                    assessment = "Strong strategy with high win potential"
                elif score_pred > 100 and win_prob > 0.5:
                    assessment = "Solid strategy with room for optimization"
                else:
                    assessment = "Strategy needs significant improvements"
                
                recommendations['overall_assessment'] = assessment
            
            return recommendations
            
        except Exception as e:
            return {'error': f'Analysis failed: {e}'}
    
    def run_strategy_matchups(
        self,
        strategies: List[AllianceStrategy],
        num_matches_per_pair: int = 1000
    ) -> Dict[Tuple[str, str], MatchupResult]:
        """Run comprehensive matchup analysis between all strategy pairs"""
        
        matchup_results = {}
        total_pairs = len(list(combinations(strategies, 2)))
        current_pair = 0
        
        print(f"Running matchup analysis for {len(strategies)} strategies...")
        print(f"Total matchups: {total_pairs}, Matches per pair: {num_matches_per_pair}")
        
        for strategy_a, strategy_b in combinations(strategies, 2):
            current_pair += 1
            print(f"Matchup {current_pair}/{total_pairs}: {strategy_a.name} vs {strategy_b.name}")
            
            a_wins = 0
            b_wins = 0
            ties = 0
            margins = []
            a_scores = []
            b_scores = []
            
            for _ in range(num_matches_per_pair):
                result = self.simulator.simulate_match(strategy_a, strategy_b)
                
                if result.winner == "red":
                    a_wins += 1
                    margins.append(result.margin)
                elif result.winner == "blue":
                    b_wins += 1
                    margins.append(-result.margin)
                else:
                    ties += 1
                    margins.append(0)
                
                a_scores.append(result.red_score)
                b_scores.append(result.blue_score)
            
            # Calculate confidence interval for win rate
            a_win_rate = a_wins / num_matches_per_pair
            std_error = np.sqrt(a_win_rate * (1 - a_win_rate) / num_matches_per_pair)
            confidence_interval = (
                a_win_rate - 1.96 * std_error,
                a_win_rate + 1.96 * std_error
            )
            
            matchup_result = MatchupResult(
                strategy_a=strategy_a.name,
                strategy_b=strategy_b.name,
                a_wins=a_wins,
                b_wins=b_wins,
                ties=ties,
                avg_margin=np.mean(margins),
                a_avg_score=np.mean(a_scores),
                b_avg_score=np.mean(b_scores),
                total_matches=num_matches_per_pair,
                confidence_interval=confidence_interval
            )
            
            matchup_results[(strategy_a.name, strategy_b.name)] = matchup_result
        
        self.matchup_matrix = matchup_results
        return matchup_results
    
    def create_matchup_matrix_dataframe(
        self,
        matchup_results: Dict[Tuple[str, str], MatchupResult]
    ) -> pd.DataFrame:
        """Create a matchup matrix DataFrame for visualization"""
        
        # Get all strategy names
        strategy_names = set()
        for (a, b) in matchup_results.keys():
            strategy_names.add(a)
            strategy_names.add(b)
        
        strategy_names = sorted(list(strategy_names))
        n = len(strategy_names)
        
        # Create matrix
        matrix = np.zeros((n, n))
        
        for i, strategy_a in enumerate(strategy_names):
            for j, strategy_b in enumerate(strategy_names):
                if i == j:
                    matrix[i][j] = 0.5  # Self matchup
                else:
                    key = (strategy_a, strategy_b) if (strategy_a, strategy_b) in matchup_results else (strategy_b, strategy_a)
                    if key in matchup_results:
                        result = matchup_results[key]
                        if strategy_a == result.strategy_a:
                            matrix[i][j] = result.a_wins / result.total_matches
                        else:
                            matrix[i][j] = result.b_wins / result.total_matches
        
        return pd.DataFrame(matrix, index=strategy_names, columns=strategy_names)
    
    def find_dominant_strategies(
        self,
        matchup_results: Dict[Tuple[str, str], MatchupResult],
        threshold: float = 0.6
    ) -> List[Tuple[str, float]]:
        """Find strategies that dominate others"""
        
        strategy_performance = {}
        
        for (strategy_a, strategy_b), result in matchup_results.items():
            if strategy_a not in strategy_performance:
                strategy_performance[strategy_a] = []
            if strategy_b not in strategy_performance:
                strategy_performance[strategy_b] = []
            
            a_win_rate = result.a_wins / result.total_matches
            b_win_rate = result.b_wins / result.total_matches
            
            strategy_performance[strategy_a].append(a_win_rate)
            strategy_performance[strategy_b].append(b_win_rate)
        
        # Calculate overall win rates
        overall_performance = []
        for strategy, win_rates in strategy_performance.items():
            avg_win_rate = np.mean(win_rates)
            if avg_win_rate >= threshold:
                overall_performance.append((strategy, avg_win_rate))
        
        return sorted(overall_performance, key=lambda x: x[1], reverse=True)
    
    def find_counter_strategies(
        self,
        target_strategy: str,
        matchup_results: Dict[Tuple[str, str], MatchupResult],
        min_advantage: float = 0.1
    ) -> List[Tuple[str, float]]:
        """Find strategies that counter a specific target strategy"""
        
        counters = []
        
        for (strategy_a, strategy_b), result in matchup_results.items():
            target_win_rate = None
            counter_strategy = None
            
            if strategy_a == target_strategy:
                target_win_rate = result.a_wins / result.total_matches
                counter_strategy = strategy_b
                counter_win_rate = result.b_wins / result.total_matches
            elif strategy_b == target_strategy:
                target_win_rate = result.b_wins / result.total_matches
                counter_strategy = strategy_a
                counter_win_rate = result.a_wins / result.total_matches
            
            if target_win_rate is not None and counter_win_rate > (0.5 + min_advantage):
                advantage = counter_win_rate - target_win_rate
                counters.append((counter_strategy, advantage))
        
        return sorted(counters, key=lambda x: x[1], reverse=True)
    
    def generate_comprehensive_report(
        self,
        strategies: List[AllianceStrategy],
        metrics: List[StrategyMetrics],
        matchup_results: Dict[Tuple[str, str], MatchupResult]
    ) -> str:
        """Generate a comprehensive analysis report"""
        
        report = []
        report.append("=" * 80)
        report.append("VEX U PUSH BACK - COMPREHENSIVE STRATEGY ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Executive Summary
        report.append("\\nEXECUTIVE SUMMARY")
        report.append("-" * 50)
        
        # Strategy rankings
        sorted_metrics = sorted(metrics, key=lambda x: x.win_rate, reverse=True)
        
        report.append("\\nStrategy Rankings by Win Rate:")
        for i, metric in enumerate(sorted_metrics, 1):
            report.append(f"{i:2d}. {metric.strategy_name:<25} {metric.win_rate:>6.1%} wins ({metric.avg_score:>5.0f} avg points)")
        
        # Best overall strategy
        best_strategy = sorted_metrics[0]
        report.append(f"\\nBest Overall Strategy: {best_strategy.strategy_name}")
        report.append(f"  - Win Rate: {best_strategy.win_rate:.1%}")
        report.append(f"  - Average Score: {best_strategy.avg_score:.0f} points")
        report.append(f"  - Risk/Reward Ratio: {best_strategy.risk_reward_ratio:.2f}")
        
        # Dominant strategies
        dominant = self.find_dominant_strategies(matchup_results, 0.55)
        if dominant:
            report.append("\\nDominant Strategies (>55% overall win rate):")
            for strategy, win_rate in dominant:
                report.append(f"  - {strategy}: {win_rate:.1%}")
        
        # Detailed Strategy Analysis
        report.append("\\n\\nDETAILED STRATEGY ANALYSIS")
        report.append("=" * 50)
        
        for metric in sorted_metrics:
            report.append(f"\\n{metric.strategy_name.upper()}")
            report.append("-" * len(metric.strategy_name))
            report.append(f"Performance Metrics:")
            report.append(f"  • Win Rate: {metric.win_rate:.1%}")
            report.append(f"  • Average Score: {metric.avg_score:.0f} ± {metric.score_std:.0f}")
            report.append(f"  • Risk/Reward Ratio: {metric.risk_reward_ratio:.2f}")
            report.append(f"  • Consistency Score: {metric.consistency_score:.2f}")
            
            report.append(f"\\nScoring Breakdown:")
            for component, percentage in metric.component_breakdown.items():
                if component != 'total':
                    report.append(f"  • {component.title()}: {percentage:.0f}%")
            
            if metric.strengths:
                report.append(f"\\nStrengths:")
                for strength in metric.strengths:
                    report.append(f"  ✓ {strength}")
            
            if metric.weaknesses:
                report.append(f"\\nWeaknesses:")
                for weakness in metric.weaknesses:
                    report.append(f"  ✗ {weakness}")
            
            # Counter strategies
            counters = self.find_counter_strategies(metric.strategy_name, matchup_results, 0.05)
            if counters:
                report.append(f"\\nVulnerable To:")
                for counter, advantage in counters[:3]:  # Top 3 counters
                    report.append(f"  • {counter} (+{advantage:.1%} advantage)")
        
        # Matchup Matrix Summary
        report.append("\\n\\nMATCHUP ANALYSIS")
        report.append("=" * 50)
        
        # Most competitive matchups (closest to 50/50)
        competitive_matchups = []
        for (a, b), result in matchup_results.items():
            win_rate = result.a_wins / result.total_matches
            competitiveness = 1 - abs(win_rate - 0.5) * 2  # 1.0 = perfectly competitive
            competitive_matchups.append(((a, b), competitiveness, win_rate))
        
        competitive_matchups.sort(key=lambda x: x[1], reverse=True)
        
        report.append("\\nMost Competitive Matchups:")
        for (a, b), comp, win_rate in competitive_matchups[:5]:
            report.append(f"  • {a} vs {b}: {win_rate:.1%} - {1-win_rate:.1%} (Competitiveness: {comp:.2f})")
        
        # Most lopsided matchups
        report.append("\\nMost Lopsided Matchups:")
        lopsided = sorted(competitive_matchups, key=lambda x: x[1])[:5]
        for (a, b), comp, win_rate in lopsided:
            winner = a if win_rate > 0.5 else b
            advantage = max(win_rate, 1-win_rate)
            report.append(f"  • {winner} dominates: {advantage:.1%} win rate")
        
        report.append("\\n" + "=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        return "\\n".join(report)
    
    def run_complete_analysis(
        self,
        num_monte_carlo: int = 1000,
        include_coordination: bool = True
    ) -> Dict:
        """Run complete strategic analysis"""
        
        print("Starting Complete VEX U Strategic Analysis...")
        print("=" * 60)
        
        # 1. Create core strategies
        print("\\n1. Creating Core Strategies...")
        core_strategies = self.create_core_strategies()
        
        all_strategies = list(core_strategies.values())
        
        # 2. Add coordination variations if requested
        if include_coordination:
            print("\\n2. Creating Coordination Variations...")
            coordination_strategies = []
            
            for core_strategy in [CoreStrategy.BALANCED, CoreStrategy.FAST_FURIOUS]:
                base = core_strategies[core_strategy]
                for coord_type in CoordinationStrategy:
                    coord_strategy = self.create_coordination_variations(base, coord_type)
                    coordination_strategies.append(coord_strategy)
            
            all_strategies.extend(coordination_strategies)
            print(f"   Added {len(coordination_strategies)} coordination variations")
        
        print(f"\\nTotal strategies to analyze: {len(all_strategies)}")
        
        # 3. Analyze each strategy individually
        print("\\n3. Analyzing Individual Strategy Performance...")
        all_metrics = []
        
        for i, strategy in enumerate(all_strategies, 1):
            print(f"   {i}/{len(all_strategies)}: {strategy.name}")
            metrics = self.analyze_strategy_comprehensive(
                strategy, 
                num_opponents=num_monte_carlo,
                opponent_variety=True
            )
            all_metrics.append(metrics)
        
        # 4. Run head-to-head matchups
        print("\\n4. Running Head-to-Head Matchups...")
        matchup_results = self.run_strategy_matchups(
            all_strategies, 
            num_matches_per_pair=num_monte_carlo
        )
        
        # 5. Generate comprehensive analysis
        print("\\n5. Generating Analysis Report...")
        report = self.generate_comprehensive_report(
            all_strategies, all_metrics, matchup_results
        )
        
        # Create matchup matrix
        matchup_df = self.create_matchup_matrix_dataframe(matchup_results)
        
        print("\\n✅ Analysis Complete!")
        
        return {
            'strategies': all_strategies,
            'metrics': all_metrics,
            'matchup_results': matchup_results,
            'matchup_matrix': matchup_df,
            'report': report,
            'dominant_strategies': self.find_dominant_strategies(matchup_results),
            'analysis_summary': {
                'total_strategies': len(all_strategies),
                'total_simulations': num_monte_carlo * len(all_strategies),
                'total_matchups': len(matchup_results),
                'total_matches': len(matchup_results) * num_monte_carlo
            }
        }


if __name__ == "__main__":
    from ..core.simulator import ScoringSimulator
    
    # Initialize analyzer
    simulator = ScoringSimulator()
    analyzer = AdvancedStrategyAnalyzer(simulator)
    
    # Run complete analysis
    results = analyzer.run_complete_analysis(
        num_monte_carlo=1000,
        include_coordination=True
    )
    
    # Display results
    print("\\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    
    # Print summary
    summary = results['analysis_summary']
    print(f"Analyzed {summary['total_strategies']} strategies")
    print(f"Ran {summary['total_simulations']:,} individual simulations")
    print(f"Completed {summary['total_matches']:,} head-to-head matches")
    
    # Print top strategies
    print("\\nTop 5 Strategies by Win Rate:")
    sorted_metrics = sorted(results['metrics'], key=lambda x: x.win_rate, reverse=True)
    for i, metric in enumerate(sorted_metrics[:5], 1):
        print(f"{i}. {metric.strategy_name:<30} {metric.win_rate:>6.1%} wins")
    
    # Print dominant strategies
    dominant = results['dominant_strategies']
    if dominant:
        print("\\nDominant Strategies:")
        for strategy, win_rate in dominant:
            print(f"  • {strategy}: {win_rate:.1%}")
    
    # Print matchup matrix preview
    print("\\nMatchup Matrix (Win Rates):")
    print(results['matchup_matrix'].round(2))
    
    print("\\n📊 Full detailed report available in results['report']")
    print("📈 All data available for further analysis and visualization")