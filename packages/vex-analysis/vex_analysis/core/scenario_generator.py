import random
import itertools
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
from collections import defaultdict, deque
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path

# ML and statistical libraries
try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    from scipy import stats
    from scipy.signal import find_peaks
    import ruptures as rpt  # For change point detection
    from deap import base, creator, tools, algorithms  # For genetic algorithms
    import seaborn as sns
    import matplotlib.pyplot as plt
    ML_LIBRARIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some ML libraries not available - {e}")
    print("Install with: pip install scikit-learn scipy ruptures deap seaborn matplotlib")
    ML_LIBRARIES_AVAILABLE = False

try:
    from .simulator import (
        AllianceStrategy, ScoringSimulator, Zone, ParkingLocation, GameConstants
    )
except ImportError:
    # Fallback for when running from main.py
    from core.simulator import (
        AllianceStrategy, ScoringSimulator, Zone, ParkingLocation, GameConstants
    )


class SkillLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class RobotRole(Enum):
    SCORER = "scorer"
    DEFENDER = "defender"
    SUPPORT = "support"
    HYBRID = "hybrid"


class StrategyType(Enum):
    ALL_OFFENSE = "all_offense"
    MIXED = "mixed"
    ZONE_CONTROL = "zone_control"
    DEFENSIVE = "defensive"
    AUTONOMOUS_FOCUS = "autonomous_focus"


@dataclass
class RobotCapabilities:
    blocks_per_second: float  # Scoring rate
    max_capacity: int  # Max blocks held at once
    travel_time_per_goal: float  # Seconds to travel between goals
    collection_time: float  # Time to collect one block
    accuracy: float  # Percentage of successful scoring attempts
    autonomous_reliability: float  # Auto performance consistency
    

@dataclass
class TimeConstraints:
    autonomous_time: int = 15
    driver_time: int = 105
    endgame_time: int = 30  # Last 30 seconds for parking/strategy changes
    setup_time: float = 2.0  # Time needed to start scoring
    interference_factor: float = 0.85  # Reduction due to defense


@dataclass
class MatchEvent:
    """Represents a significant event during a match"""
    timestamp: float  # Time in seconds when event occurred
    event_type: str  # 'score', 'zone_control', 'strategy_switch', 'critical_moment'
    alliance: str  # 'red' or 'blue'
    details: Dict[str, Any]  # Additional event-specific data
    impact_score: float = 0.0  # Quantified impact on match outcome


@dataclass
class StrategyPattern:
    """Represents a discovered strategic pattern"""
    pattern_id: str
    pattern_type: str  # 'scoring', 'coordination', 'timing', 'defensive'
    frequency: float  # How often this pattern appears in winning strategies
    win_rate: float  # Win rate when this pattern is used
    description: str
    key_features: Dict[str, float]
    example_strategies: List[str] = field(default_factory=list)
    confidence: float = 0.0
    

@dataclass
class CriticalMoment:
    """Represents a critical decision point in a match"""
    timestamp: float
    context: Dict[str, Any]  # Game state at this moment
    decision_options: List[str]  # Available strategic choices
    optimal_choice: str  # Recommended action
    impact_magnitude: float  # How much this moment affects outcome
    confidence: float  # Confidence in the recommendation


@dataclass
class ScenarioParameters:
    skill_level: SkillLevel
    strategy_type: StrategyType
    robot1_role: RobotRole
    robot2_role: RobotRole
    robot1_capabilities: RobotCapabilities
    robot2_capabilities: RobotCapabilities
    field_position: str
    cooperation_efficiency: float  # How well robots work together
    

class MLScenarioDiscovery:
    """Machine Learning-powered scenario discovery and pattern analysis"""
    
    def __init__(self, enable_ml: bool = True):
        self.enable_ml = enable_ml and ML_LIBRARIES_AVAILABLE
        
        # ML Models
        self.clustering_model = None
        self.pattern_analyzer = None
        self.scaler = StandardScaler() if self.enable_ml else None
        
        # Data storage
        self.match_history: List[Dict[str, Any]] = []
        self.discovered_patterns: List[StrategyPattern] = []
        self.critical_moments: List[CriticalMoment] = []
        
        # Analysis parameters
        self.pattern_confidence_threshold = 0.7
        self.min_pattern_frequency = 0.1
        self.change_point_min_size = 5
        
        # Genetic algorithm setup
        if self.enable_ml:
            self._setup_genetic_algorithm()
        
        print(f"MLScenarioDiscovery initialized (ML enabled: {self.enable_ml})")
    
    def _setup_genetic_algorithm(self):
        """Initialize genetic algorithm framework for scenario evolution"""
        try:
            # Create fitness and individual classes
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
            
            self.toolbox = base.Toolbox()
            
            # Define genetic operators
            self.toolbox.register("attr_float", random.uniform, 0.0, 1.0)
            self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
                                self.toolbox.attr_float, n=20)  # 20 strategy parameters
            self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
            
            self.toolbox.register("evaluate", self._evaluate_scenario_fitness)
            self.toolbox.register("mate", tools.cxTwoPoint)
            self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
            self.toolbox.register("select", tools.selTournament, tournsize=3)
            
        except Exception as e:
            print(f"Warning: Could not setup genetic algorithm - {e}")
            self.enable_ml = False
    
    def discover_winning_patterns(
        self, 
        match_history: List[Dict[str, Any]], 
        min_win_rate: float = 0.75
    ) -> List[StrategyPattern]:
        """Discover winning patterns from historical match data using unsupervised learning"""
        
        if not self.enable_ml or not match_history:
            print("Warning: ML not available or no match history provided")
            return []
        
        print(f"Analyzing {len(match_history)} matches for winning patterns...")
        
        # Store match history
        self.match_history = match_history
        
        # Extract features from matches
        feature_matrix, match_outcomes = self._extract_match_features(match_history)
        
        if len(feature_matrix) == 0:
            return []
        
        # Perform clustering to identify strategy archetypes
        clusters = self._perform_strategy_clustering(feature_matrix)
        
        # Analyze each cluster for winning patterns
        patterns = []
        for cluster_id in range(len(set(clusters))):
            pattern = self._analyze_cluster_patterns(
                cluster_id, clusters, feature_matrix, match_outcomes, min_win_rate
            )
            if pattern:
                patterns.append(pattern)
        
        # Use sequential pattern mining for temporal patterns
        temporal_patterns = self._discover_temporal_patterns(match_history, min_win_rate)
        patterns.extend(temporal_patterns)
        
        self.discovered_patterns = patterns
        print(f"Discovered {len(patterns)} winning patterns")
        
        return patterns
    
    def _extract_match_features(self, matches: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
        """Extract numerical features from match data for ML analysis"""
        features = []
        outcomes = []
        
        for match in matches:
            try:
                # Basic scoring features
                feature_vector = [
                    match.get('red_score', 0),
                    match.get('blue_score', 0),
                    match.get('red_blocks_total', 0),
                    match.get('blue_blocks_total', 0),
                    match.get('red_blocks_auto', 0),
                    match.get('blue_blocks_auto', 0),
                    match.get('red_zones', 0),
                    match.get('blue_zones', 0),
                    match.get('red_parking_score', 0),
                    match.get('blue_parking_score', 0),
                    match.get('margin', 0),
                    match.get('match_competitiveness', 0.5),
                ]
                
                # Strategy-specific features
                red_strategy = match.get('red_strategy', 'mixed')
                blue_strategy = match.get('blue_strategy', 'mixed')
                
                # One-hot encode strategies
                strategy_types = ['all_offense', 'mixed', 'zone_control', 'defensive', 'autonomous_focus']
                for strategy in strategy_types:
                    feature_vector.append(1.0 if red_strategy == strategy else 0.0)
                    feature_vector.append(1.0 if blue_strategy == strategy else 0.0)
                
                # Skill level features
                skill_levels = ['beginner', 'intermediate', 'advanced', 'expert']
                red_skill = match.get('red_skill', 'intermediate')
                blue_skill = match.get('blue_skill', 'intermediate')
                
                for skill in skill_levels:
                    feature_vector.append(1.0 if red_skill == skill else 0.0)
                    feature_vector.append(1.0 if blue_skill == skill else 0.0)
                
                # Robot role features
                feature_vector.extend([
                    match.get('red_cooperation', 0.8),
                    match.get('blue_cooperation', 0.8),
                ])
                
                features.append(feature_vector)
                outcomes.append(match.get('winner', 'tie'))
                
            except Exception as e:
                print(f"Warning: Error extracting features from match - {e}")
                continue
        
        return np.array(features), outcomes
    
    def _perform_strategy_clustering(self, feature_matrix: np.ndarray) -> List[int]:
        """Perform K-means clustering to identify strategy archetypes"""
        if len(feature_matrix) < 5:
            return [0] * len(feature_matrix)
        
        # Normalize features
        normalized_features = self.scaler.fit_transform(feature_matrix)
        
        # Determine optimal number of clusters
        best_k = 3
        best_score = -1
        
        for k in range(2, min(8, len(feature_matrix) // 2)):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(normalized_features)
                score = silhouette_score(normalized_features, cluster_labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                continue
        
        # Perform final clustering
        self.clustering_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        clusters = self.clustering_model.fit_predict(normalized_features)
        
        print(f"Strategy clustering: {best_k} clusters with silhouette score {best_score:.3f}")
        return clusters.tolist()
    
    def _analyze_cluster_patterns(
        self, 
        cluster_id: int, 
        clusters: List[int], 
        features: np.ndarray, 
        outcomes: List[str],
        min_win_rate: float
    ) -> Optional[StrategyPattern]:
        """Analyze a specific cluster for winning patterns"""
        
        # Get matches in this cluster
        cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
        
        if len(cluster_indices) < 3:
            return None
        
        # Calculate win rate for red alliance in this cluster
        cluster_outcomes = [outcomes[i] for i in cluster_indices]
        red_wins = sum(1 for outcome in cluster_outcomes if outcome == 'red')
        win_rate = red_wins / len(cluster_outcomes)
        
        if win_rate < min_win_rate:
            return None
        
        # Calculate cluster centroid features
        cluster_features = features[cluster_indices]
        centroid = np.mean(cluster_features, axis=0)
        
        # Identify key distinguishing features
        feature_names = [
            'red_score', 'blue_score', 'red_blocks_total', 'blue_blocks_total',
            'red_blocks_auto', 'blue_blocks_auto', 'red_zones', 'blue_zones',
            'red_parking_score', 'blue_parking_score', 'margin', 'competitiveness'
        ]
        
        key_features = {}
        for i, name in enumerate(feature_names[:len(centroid)]):
            if i < len(centroid):
                key_features[name] = float(centroid[i])
        
        # Generate pattern description
        description = self._generate_pattern_description(key_features, win_rate)
        
        return StrategyPattern(
            pattern_id=f"cluster_{cluster_id}",
            pattern_type="strategic_archetype",
            frequency=len(cluster_indices) / len(outcomes),
            win_rate=win_rate,
            description=description,
            key_features=key_features,
            confidence=min(win_rate, len(cluster_indices) / 20.0)  # Confidence based on sample size
        )
    
    def _generate_pattern_description(self, features: Dict[str, float], win_rate: float) -> str:
        """Generate human-readable description of a pattern"""
        desc_parts = []
        
        if features.get('red_blocks_auto', 0) > 15:
            desc_parts.append("strong autonomous performance")
        
        if features.get('red_zones', 0) >= 2:
            desc_parts.append("zone control focus")
        
        if features.get('red_parking_score', 0) > 20:
            desc_parts.append("parking strategy emphasis")
        
        if features.get('margin', 0) > 30:
            desc_parts.append("high-margin victories")
        
        base_desc = f"Strategic pattern with {win_rate:.1%} win rate"
        if desc_parts:
            base_desc += f" featuring {', '.join(desc_parts)}"
        
        return base_desc
    
    def _discover_temporal_patterns(
        self, 
        matches: List[Dict[str, Any]], 
        min_win_rate: float
    ) -> List[StrategyPattern]:
        """Discover temporal patterns using sequential pattern mining"""
        patterns = []
        
        # Analyze autonomous vs driver control performance patterns
        auto_strong_matches = []
        driver_strong_matches = []
        
        for match in matches:
            auto_ratio = match.get('red_blocks_auto', 0) / max(match.get('red_blocks_total', 1), 1)
            driver_ratio = 1 - auto_ratio
            
            if auto_ratio > 0.4 and match.get('winner') == 'red':
                auto_strong_matches.append(match)
            elif driver_ratio > 0.8 and match.get('winner') == 'red':
                driver_strong_matches.append(match)
        
        # Create patterns for temporal strategies
        if len(auto_strong_matches) >= 3:
            auto_win_rate = len(auto_strong_matches) / len([m for m in matches if m.get('red_blocks_auto', 0) / max(m.get('red_blocks_total', 1), 1) > 0.4])
            if auto_win_rate >= min_win_rate:
                patterns.append(StrategyPattern(
                    pattern_id="autonomous_focus_temporal",
                    pattern_type="temporal",
                    frequency=len(auto_strong_matches) / len(matches),
                    win_rate=auto_win_rate,
                    description=f"Autonomous-focused strategy with {auto_win_rate:.1%} win rate",
                    key_features={"autonomous_emphasis": 1.0, "early_game_focus": 1.0},
                    confidence=min(auto_win_rate, len(auto_strong_matches) / 10.0)
                ))
        
        return patterns
    
    def identify_critical_moments(
        self, 
        match_timeline: List[MatchEvent],
        scoring_threshold: float = 10.0
    ) -> List[CriticalMoment]:
        """Identify critical decision points using change point detection"""
        
        if not self.enable_ml or not match_timeline:
            return []
        
        # Extract score differentials over time
        timestamps = [event.timestamp for event in match_timeline]
        score_diffs = []
        
        red_score = 0
        blue_score = 0
        
        for event in match_timeline:
            if event.event_type == 'score':
                if event.alliance == 'red':
                    red_score += event.details.get('points', 0)
                else:
                    blue_score += event.details.get('points', 0)
            
            score_diffs.append(red_score - blue_score)
        
        if len(score_diffs) < self.change_point_min_size:
            return []
        
        # Detect change points in score differential
        try:
            # Use PELT algorithm for change point detection
            algo = rpt.Pelt(model="rbf").fit(np.array(score_diffs))
            change_points = algo.predict(pen=scoring_threshold)
            
            critical_moments = []
            
            for cp_idx in change_points[:-1]:  # Exclude last point (end of match)
                if cp_idx < len(match_timeline):
                    timestamp = timestamps[cp_idx]
                    
                    # Analyze context around this change point
                    context = self._analyze_change_point_context(match_timeline, cp_idx)
                    
                    # Determine optimal strategy choice
                    optimal_choice = self._recommend_strategy_at_moment(context, score_diffs[cp_idx])
                    
                    # Calculate impact magnitude
                    impact = self._calculate_moment_impact(score_diffs, cp_idx)
                    
                    critical_moments.append(CriticalMoment(
                        timestamp=timestamp,
                        context=context,
                        decision_options=["maintain_strategy", "switch_to_defense", "increase_aggression", "focus_zones"],
                        optimal_choice=optimal_choice,
                        impact_magnitude=impact,
                        confidence=min(impact / 20.0, 1.0)  # Normalize impact to confidence
                    ))
            
            return critical_moments
            
        except Exception as e:
            print(f"Warning: Change point detection failed - {e}")
            return []
    
    def _analyze_change_point_context(self, timeline: List[MatchEvent], cp_idx: int) -> Dict[str, Any]:
        """Analyze the context around a change point"""
        # Look at events in a window around the change point
        window_size = 3
        start_idx = max(0, cp_idx - window_size)
        end_idx = min(len(timeline), cp_idx + window_size + 1)
        
        context_events = timeline[start_idx:end_idx]
        
        return {
            'timestamp': timeline[cp_idx].timestamp,
            'recent_events': [{'type': e.event_type, 'alliance': e.alliance, 'details': e.details} for e in context_events],
            'score_momentum': self._calculate_momentum(timeline, cp_idx),
            'match_phase': 'autonomous' if timeline[cp_idx].timestamp <= 15 else 'driver_control'
        }
    
    def _calculate_momentum(self, timeline: List[MatchEvent], idx: int) -> float:
        """Calculate scoring momentum at a given point"""
        window = 5  # Look at last 5 events
        start_idx = max(0, idx - window)
        
        red_points = 0
        blue_points = 0
        
        for i in range(start_idx, idx):
            if i < len(timeline) and timeline[i].event_type == 'score':
                points = timeline[i].details.get('points', 0)
                if timeline[i].alliance == 'red':
                    red_points += points
                else:
                    blue_points += points
        
        return (red_points - blue_points) / max(window, 1)
    
    def _recommend_strategy_at_moment(self, context: Dict[str, Any], score_diff: float) -> str:
        """Recommend optimal strategy choice at a critical moment"""
        
        momentum = context.get('score_momentum', 0)
        
        if score_diff > 20:  # Leading significantly
            if momentum > 0:
                return "maintain_strategy"
            else:
                return "focus_zones"  # Secure lead with zone control
        elif score_diff < -20:  # Trailing significantly
            return "increase_aggression"  # Need to catch up
        elif momentum < -5:  # Losing momentum
            return "switch_to_defense"
        else:
            return "maintain_strategy"
    
    def _calculate_moment_impact(self, score_diffs: List[float], cp_idx: int) -> float:
        """Calculate the impact magnitude of a critical moment"""
        
        if cp_idx == 0 or cp_idx >= len(score_diffs) - 1:
            return 0.0
        
        # Look at score differential change
        before = np.mean(score_diffs[max(0, cp_idx-3):cp_idx])
        after = np.mean(score_diffs[cp_idx:min(len(score_diffs), cp_idx+3)])
        
        return abs(after - before)
    
    def generate_optimal_scenarios(
        self, 
        constraints: Dict[str, Any],
        num_generations: int = 50,
        population_size: int = 100
    ) -> List[Dict[str, Any]]:
        """Generate optimal scenarios using genetic algorithms"""
        
        if not self.enable_ml:
            print("Warning: ML not available for scenario generation")
            return []
        
        print(f"Evolving optimal scenarios over {num_generations} generations...")
        
        # Initialize population
        population = self.toolbox.population(n=population_size)
        
        # Evolution statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Run genetic algorithm
        try:
            population, logbook = algorithms.eaSimple(
                population, self.toolbox,
                cxpb=0.7, mutpb=0.3,
                ngen=num_generations,
                stats=stats, verbose=False
            )
            
            # Convert best individuals to scenario parameters
            best_individuals = tools.selBest(population, k=min(10, len(population)))
            optimal_scenarios = []
            
            for individual in best_individuals:
                scenario = self._individual_to_scenario(individual, constraints)
                scenario['fitness_score'] = individual.fitness.values[0]
                optimal_scenarios.append(scenario)
            
            print(f"Generated {len(optimal_scenarios)} optimal scenarios")
            return optimal_scenarios
            
        except Exception as e:
            print(f"Warning: Genetic algorithm failed - {e}")
            return []
    
    def _evaluate_scenario_fitness(self, individual: List[float]) -> Tuple[float,]:
        """Evaluate fitness of a scenario represented by an individual"""
        
        try:
            # Convert individual to scenario parameters
            scenario_params = {
                'red_skill_level': individual[0],
                'blue_skill_level': individual[1],
                'red_strategy_aggression': individual[2],
                'blue_strategy_aggression': individual[3],
                'red_cooperation': individual[4],
                'blue_cooperation': individual[5],
                'red_auto_focus': individual[6],
                'blue_auto_focus': individual[7],
                'competitiveness': individual[8],
                'duration_factor': individual[9],
            }
            
            # Calculate fitness based on multiple criteria
            fitness_components = []
            
            # 1. Competitiveness (close matches are better)
            competitiveness = 1.0 - abs(scenario_params['red_skill_level'] - scenario_params['blue_skill_level'])
            fitness_components.append(competitiveness * 0.3)
            
            # 2. Strategic diversity
            strategy_diversity = abs(scenario_params['red_strategy_aggression'] - scenario_params['blue_strategy_aggression'])
            fitness_components.append(strategy_diversity * 0.2)
            
            # 3. Cooperation efficiency
            cooperation_score = (scenario_params['red_cooperation'] + scenario_params['blue_cooperation']) / 2
            fitness_components.append(cooperation_score * 0.2)
            
            # 4. Pattern adherence (if patterns discovered)
            pattern_score = self._calculate_pattern_adherence(scenario_params)
            fitness_components.append(pattern_score * 0.3)
            
            return (sum(fitness_components),)
            
        except Exception as e:
            return (0.0,)  # Return low fitness for invalid scenarios
    
    def _calculate_pattern_adherence(self, params: Dict[str, float]) -> float:
        """Calculate how well scenario parameters adhere to discovered winning patterns"""
        
        if not self.discovered_patterns:
            return 0.5  # Neutral score if no patterns discovered
        
        adherence_scores = []
        
        for pattern in self.discovered_patterns:
            if pattern.win_rate > 0.7:  # Only consider high win rate patterns
                score = 0.0
                
                # Check pattern-specific criteria
                if pattern.pattern_type == "temporal" and "autonomous_emphasis" in pattern.key_features:
                    if params.get('red_auto_focus', 0.5) > 0.6:
                        score += pattern.confidence
                
                elif pattern.pattern_type == "strategic_archetype":
                    # Check if scenario matches archetype characteristics
                    if pattern.key_features.get('red_zones', 0) >= 2 and params.get('red_cooperation', 0.5) > 0.8:
                        score += pattern.confidence * 0.8
                
                adherence_scores.append(score)
        
        return np.mean(adherence_scores) if adherence_scores else 0.5
    
    def _individual_to_scenario(self, individual: List[float], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Convert genetic algorithm individual to scenario parameters"""
        
        # Map individual values to meaningful scenario parameters
        skill_levels = ['beginner', 'intermediate', 'advanced', 'expert']
        strategy_types = ['all_offense', 'mixed', 'zone_control', 'defensive', 'autonomous_focus']
        
        red_skill_idx = min(int(individual[0] * len(skill_levels)), len(skill_levels) - 1)
        blue_skill_idx = min(int(individual[1] * len(skill_levels)), len(skill_levels) - 1)
        
        red_strategy_idx = min(int(individual[2] * len(strategy_types)), len(strategy_types) - 1)
        blue_strategy_idx = min(int(individual[3] * len(strategy_types)), len(strategy_types) - 1)
        
        return {
            'red_skill': skill_levels[red_skill_idx],
            'blue_skill': skill_levels[blue_skill_idx],
            'red_strategy': strategy_types[red_strategy_idx],
            'blue_strategy': strategy_types[blue_strategy_idx],
            'red_cooperation': max(0.5, min(1.0, individual[4])),
            'blue_cooperation': max(0.5, min(1.0, individual[5])),
            'red_auto_focus': max(0.0, min(1.0, individual[6])),
            'blue_auto_focus': max(0.0, min(1.0, individual[7])),
            'expected_competitiveness': max(0.0, min(1.0, individual[8])),
            'match_duration_factor': max(0.8, min(1.2, individual[9])),
        }
    
    def export_patterns(self, filename: str) -> None:
        """Export discovered patterns to JSON file"""
        
        export_data = {
            'discovery_timestamp': datetime.now().isoformat(),
            'total_patterns': len(self.discovered_patterns),
            'patterns': []
        }
        
        for pattern in self.discovered_patterns:
            export_data['patterns'].append({
                'pattern_id': pattern.pattern_id,
                'pattern_type': pattern.pattern_type,
                'frequency': pattern.frequency,
                'win_rate': pattern.win_rate,
                'description': pattern.description,
                'key_features': pattern.key_features,
                'confidence': pattern.confidence
            })
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Patterns exported to {filename}")
    
    def visualize_patterns(self, save_path: Optional[str] = None) -> None:
        """Create visualizations of discovered patterns"""
        
        if not self.enable_ml or not self.discovered_patterns:
            print("Warning: No patterns to visualize")
            return
        
        try:
            # Create pattern analysis plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('VEX U Strategy Pattern Analysis', fontsize=16)
            
            # 1. Pattern win rates
            patterns_df = pd.DataFrame([
                {
                    'pattern_id': p.pattern_id,
                    'win_rate': p.win_rate,
                    'frequency': p.frequency,
                    'confidence': p.confidence,
                    'type': p.pattern_type
                }
                for p in self.discovered_patterns
            ])
            
            sns.barplot(data=patterns_df, x='pattern_id', y='win_rate', ax=axes[0,0])
            axes[0,0].set_title('Pattern Win Rates')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # 2. Pattern frequency vs win rate
            sns.scatterplot(data=patterns_df, x='frequency', y='win_rate', 
                          size='confidence', hue='type', ax=axes[0,1])
            axes[0,1].set_title('Frequency vs Win Rate')
            
            # 3. Pattern confidence distribution
            sns.histplot(patterns_df['confidence'], bins=10, ax=axes[1,0])
            axes[1,0].set_title('Pattern Confidence Distribution')
            
            # 4. Pattern types
            type_counts = patterns_df['type'].value_counts()
            axes[1,1].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
            axes[1,1].set_title('Pattern Types Distribution')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Pattern visualization saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            print(f"Warning: Pattern visualization failed - {e}")


class ScenarioGenerator:
    def __init__(self, simulator: ScoringSimulator, enable_ml: bool = True):
        self.simulator = simulator
        self.constants = GameConstants()
        self.time_constraints = TimeConstraints()
        
        # Initialize ML-powered discovery
        self.ml_discovery = MLScenarioDiscovery(enable_ml=enable_ml)
        
        # Define capability profiles for different skill levels
        self.capability_profiles = {
            SkillLevel.BEGINNER: RobotCapabilities(
                blocks_per_second=0.15,
                max_capacity=2,
                travel_time_per_goal=8.0,
                collection_time=3.0,
                accuracy=0.70,
                autonomous_reliability=0.60
            ),
            SkillLevel.INTERMEDIATE: RobotCapabilities(
                blocks_per_second=0.25,
                max_capacity=3,
                travel_time_per_goal=5.0,
                collection_time=2.0,
                accuracy=0.80,
                autonomous_reliability=0.75
            ),
            SkillLevel.ADVANCED: RobotCapabilities(
                blocks_per_second=0.40,
                max_capacity=4,
                travel_time_per_goal=3.5,
                collection_time=1.5,
                accuracy=0.88,
                autonomous_reliability=0.85
            ),
            SkillLevel.EXPERT: RobotCapabilities(
                blocks_per_second=0.55,
                max_capacity=5,
                travel_time_per_goal=2.5,
                collection_time=1.0,
                accuracy=0.95,
                autonomous_reliability=0.95
            )
        }
    
    def calculate_realistic_scoring(
        self,
        capabilities: RobotCapabilities,
        role: RobotRole,
        time_available: float,
        cooperation_factor: float = 1.0,
        interference_factor: float = 1.0
    ) -> int:
        """Calculate realistic blocks scored based on robot capabilities and time"""
        
        # Adjust scoring rate based on role
        role_multipliers = {
            RobotRole.SCORER: 1.0,
            RobotRole.DEFENDER: 0.3,  # Defenders focus on blocking, not scoring
            RobotRole.SUPPORT: 0.6,  # Support robots help with scoring
            RobotRole.HYBRID: 0.8   # Balance between roles
        }
        
        effective_rate = (capabilities.blocks_per_second * 
                         role_multipliers[role] * 
                         cooperation_factor * 
                         interference_factor * 
                         capabilities.accuracy)
        
        # Account for travel and collection time
        cycles_per_second = 1 / (1/effective_rate + capabilities.travel_time_per_goal/60 + capabilities.collection_time)
        
        # Calculate total blocks with capacity constraints
        max_blocks_time = int(time_available * cycles_per_second)
        max_blocks_capacity = int(time_available / (capabilities.collection_time + capabilities.travel_time_per_goal/60)) * capabilities.max_capacity
        
        return min(max_blocks_time, max_blocks_capacity)
    
    def generate_time_based_strategy(
        self,
        alliance_name: str,
        parameters: ScenarioParameters
    ) -> AllianceStrategy:
        """Generate strategy based on realistic time constraints and robot capabilities"""
        
        # Calculate autonomous scoring
        auto_time = self.time_constraints.autonomous_time - self.time_constraints.setup_time
        
        robot1_auto_blocks = self.calculate_realistic_scoring(
            parameters.robot1_capabilities,
            parameters.robot1_role,
            auto_time * parameters.robot1_capabilities.autonomous_reliability,
            parameters.cooperation_efficiency
        )
        
        robot2_auto_blocks = self.calculate_realistic_scoring(
            parameters.robot2_capabilities,
            parameters.robot2_role,
            auto_time * parameters.robot2_capabilities.autonomous_reliability,
            parameters.cooperation_efficiency
        )
        
        total_auto_blocks = min(robot1_auto_blocks + robot2_auto_blocks, 30)  # Realistic auto limit
        
        # Calculate driver control scoring
        driver_time = self.time_constraints.driver_time - self.time_constraints.endgame_time
        
        robot1_driver_blocks = self.calculate_realistic_scoring(
            parameters.robot1_capabilities,
            parameters.robot1_role,
            driver_time,
            parameters.cooperation_efficiency,
            self.time_constraints.interference_factor
        )
        
        robot2_driver_blocks = self.calculate_realistic_scoring(
            parameters.robot2_capabilities,
            parameters.robot2_role,
            driver_time,
            parameters.cooperation_efficiency,
            self.time_constraints.interference_factor
        )
        
        total_driver_blocks = robot1_driver_blocks + robot2_driver_blocks
        
        # Distribute blocks among goals based on strategy
        goals = ["long_1", "long_2", "center_1", "center_2"]
        auto_distribution = self._distribute_blocks_by_strategy(
            total_auto_blocks, goals, parameters.strategy_type, is_auto=True
        )
        driver_distribution = self._distribute_blocks_by_strategy(
            total_driver_blocks, goals, parameters.strategy_type, is_auto=False
        )
        
        # Determine zone control based on strategy
        zones_controlled = self._determine_zone_control(parameters)
        
        # Determine parking based on strategy and capabilities
        robots_parked = self._determine_parking_strategy(parameters)
        
        return AllianceStrategy(
            name=alliance_name,
            blocks_scored_auto=auto_distribution,
            blocks_scored_driver=driver_distribution,
            zones_controlled=zones_controlled,
            robots_parked=robots_parked
        )
    
    def _distribute_blocks_by_strategy(
        self, 
        total_blocks: int, 
        goals: List[str], 
        strategy_type: StrategyType,
        is_auto: bool = False
    ) -> Dict[str, int]:
        """Distribute blocks based on strategy type"""
        
        if total_blocks == 0:
            return {goal: 0 for goal in goals}
        
        distribution = {goal: 0 for goal in goals}
        
        if strategy_type == StrategyType.ALL_OFFENSE:
            # Focus on high-scoring goals (long goals preferred)
            weights = [0.35, 0.35, 0.15, 0.15]  # long_1, long_2, center_1, center_2
        elif strategy_type == StrategyType.ZONE_CONTROL:
            # More conservative, spread across goals
            weights = [0.25, 0.25, 0.25, 0.25]
        elif strategy_type == StrategyType.AUTONOMOUS_FOCUS:
            if is_auto:
                # Focus on easier goals during auto
                weights = [0.4, 0.4, 0.1, 0.1]
            else:
                weights = [0.25, 0.25, 0.25, 0.25]
        else:
            # Default balanced distribution
            weights = [0.3, 0.3, 0.2, 0.2]
        
        # Apply weights
        remaining = total_blocks
        for i, (goal, weight) in enumerate(zip(goals[:-1], weights[:-1])):
            blocks = int(total_blocks * weight)
            blocks = min(blocks, remaining)
            distribution[goal] = blocks
            remaining -= blocks
        
        distribution[goals[-1]] = remaining
        return distribution
    
    def _determine_zone_control(self, parameters: ScenarioParameters) -> List[Zone]:
        """Determine zone control based on strategy and capabilities"""
        zones = []
        
        if parameters.strategy_type == StrategyType.ZONE_CONTROL:
            # Prioritize zone control
            num_zones = 3 if parameters.skill_level in [SkillLevel.ADVANCED, SkillLevel.EXPERT] else 2
            zones = random.sample([Zone.RED_HOME, Zone.BLUE_HOME, Zone.NEUTRAL], num_zones)
        elif parameters.strategy_type == StrategyType.DEFENSIVE:
            # Focus on home zone
            zones = [Zone.RED_HOME]
            if parameters.skill_level in [SkillLevel.ADVANCED, SkillLevel.EXPERT]:
                zones.append(Zone.NEUTRAL)
        elif parameters.strategy_type == StrategyType.ALL_OFFENSE:
            # Minimal zone control
            if random.random() < 0.3:
                zones = [Zone.RED_HOME]
        else:
            # Balanced approach
            if parameters.skill_level == SkillLevel.BEGINNER:
                zones = [Zone.RED_HOME] if random.random() < 0.5 else []
            else:
                zones = random.sample([Zone.RED_HOME, Zone.NEUTRAL], random.randint(1, 2))
        
        return zones
    
    def _determine_parking_strategy(self, parameters: ScenarioParameters) -> List[ParkingLocation]:
        """Determine parking strategy based on robot roles and capabilities"""
        parking = []
        
        for i, (role, capabilities) in enumerate([
            (parameters.robot1_role, parameters.robot1_capabilities),
            (parameters.robot2_role, parameters.robot2_capabilities)
        ]):
            if parameters.strategy_type == StrategyType.ALL_OFFENSE:
                # Scoring robots might not have time to park
                if role == RobotRole.SCORER and capabilities.blocks_per_second > 0.3:
                    parking.append(ParkingLocation.NONE)
                else:
                    parking.append(ParkingLocation.PLATFORM if random.random() < 0.7 else ParkingLocation.ALLIANCE_ZONE)
            elif parameters.strategy_type == StrategyType.DEFENSIVE:
                # Defensive robots more likely to park on platform
                parking.append(ParkingLocation.PLATFORM)
            else:
                # Skill-based parking decision
                if parameters.skill_level == SkillLevel.BEGINNER:
                    parking.append(random.choice([ParkingLocation.ALLIANCE_ZONE, ParkingLocation.NONE]))
                elif parameters.skill_level == SkillLevel.INTERMEDIATE:
                    parking.append(random.choice([ParkingLocation.PLATFORM, ParkingLocation.ALLIANCE_ZONE]))
                else:
                    parking.append(ParkingLocation.PLATFORM)
        
        return parking
    
    def generate_random_strategy(
        self,
        alliance_name: str,
        total_blocks_range: Tuple[int, int] = (20, 50),
        auto_percentage_range: Tuple[float, float] = (0.1, 0.3)
    ) -> AllianceStrategy:
        # Generate total blocks scored
        total_blocks = random.randint(*total_blocks_range)
        
        # Split between autonomous and driver
        auto_percentage = random.uniform(*auto_percentage_range)
        auto_blocks = int(total_blocks * auto_percentage)
        driver_blocks = total_blocks - auto_blocks
        
        # Distribute blocks among goals
        goals = ["long_1", "long_2", "center_1", "center_2"]
        
        auto_distribution = self._distribute_blocks(auto_blocks, goals)
        driver_distribution = self._distribute_blocks(driver_blocks, goals)
        
        # Generate zone control
        available_zones = [Zone.RED_HOME, Zone.BLUE_HOME, Zone.NEUTRAL]
        num_zones = random.randint(0, 3)
        zones_controlled = random.sample(available_zones, num_zones)
        
        # Generate parking
        parking_options = [
            ParkingLocation.NONE,
            ParkingLocation.ALLIANCE_ZONE,
            ParkingLocation.PLATFORM
        ]
        robots_parked = []
        for _ in range(self.constants.ROBOTS_PER_ALLIANCE):
            robots_parked.append(random.choice(parking_options))
        
        return AllianceStrategy(
            name=alliance_name,
            blocks_scored_auto=auto_distribution,
            blocks_scored_driver=driver_distribution,
            zones_controlled=zones_controlled,
            robots_parked=robots_parked
        )
    
    def _distribute_blocks(self, total_blocks: int, goals: List[str]) -> Dict[str, int]:
        if total_blocks == 0:
            return {goal: 0 for goal in goals}
        
        # Random distribution with some bias towards certain goals
        weights = [random.random() for _ in goals]
        total_weight = sum(weights)
        
        distribution = {}
        remaining = total_blocks
        
        for i, goal in enumerate(goals[:-1]):
            if remaining > 0:
                blocks = int(total_blocks * weights[i] / total_weight)
                blocks = min(blocks, remaining)
                distribution[goal] = blocks
                remaining -= blocks
            else:
                distribution[goal] = 0
        
        # Put remaining blocks in last goal
        distribution[goals[-1]] = remaining
        
        return distribution
    
    def generate_strategy_templates(self) -> Dict[StrategyType, ScenarioParameters]:
        """Generate predefined strategy templates"""
        templates = {}
        
        # All Offense Template
        templates[StrategyType.ALL_OFFENSE] = ScenarioParameters(
            skill_level=SkillLevel.ADVANCED,
            strategy_type=StrategyType.ALL_OFFENSE,
            robot1_role=RobotRole.SCORER,
            robot2_role=RobotRole.SCORER,
            robot1_capabilities=self.capability_profiles[SkillLevel.ADVANCED],
            robot2_capabilities=self.capability_profiles[SkillLevel.ADVANCED],
            field_position="center_field",
            cooperation_efficiency=0.9
        )
        
        # Mixed Strategy Template
        templates[StrategyType.MIXED] = ScenarioParameters(
            skill_level=SkillLevel.INTERMEDIATE,
            strategy_type=StrategyType.MIXED,
            robot1_role=RobotRole.SCORER,
            robot2_role=RobotRole.SUPPORT,
            robot1_capabilities=self.capability_profiles[SkillLevel.INTERMEDIATE],
            robot2_capabilities=self.capability_profiles[SkillLevel.INTERMEDIATE],
            field_position="alliance_side",
            cooperation_efficiency=0.8
        )
        
        # Zone Control Template
        templates[StrategyType.ZONE_CONTROL] = ScenarioParameters(
            skill_level=SkillLevel.ADVANCED,
            strategy_type=StrategyType.ZONE_CONTROL,
            robot1_role=RobotRole.HYBRID,
            robot2_role=RobotRole.DEFENDER,
            robot1_capabilities=self.capability_profiles[SkillLevel.ADVANCED],
            robot2_capabilities=self.capability_profiles[SkillLevel.INTERMEDIATE],
            field_position="neutral_zone",
            cooperation_efficiency=0.85
        )
        
        # Defensive Template
        templates[StrategyType.DEFENSIVE] = ScenarioParameters(
            skill_level=SkillLevel.INTERMEDIATE,
            strategy_type=StrategyType.DEFENSIVE,
            robot1_role=RobotRole.DEFENDER,
            robot2_role=RobotRole.SUPPORT,
            robot1_capabilities=self.capability_profiles[SkillLevel.INTERMEDIATE],
            robot2_capabilities=self.capability_profiles[SkillLevel.BEGINNER],
            field_position="home_zone",
            cooperation_efficiency=0.75
        )
        
        # Autonomous Focus Template
        templates[StrategyType.AUTONOMOUS_FOCUS] = ScenarioParameters(
            skill_level=SkillLevel.EXPERT,
            strategy_type=StrategyType.AUTONOMOUS_FOCUS,
            robot1_role=RobotRole.SCORER,
            robot2_role=RobotRole.SCORER,
            robot1_capabilities=self.capability_profiles[SkillLevel.EXPERT],
            robot2_capabilities=self.capability_profiles[SkillLevel.EXPERT],
            field_position="starting_position",
            cooperation_efficiency=0.95
        )
        
        return templates
    
    def generate_scenario_matrix(self) -> pd.DataFrame:
        """Generate comprehensive scenario matrix with different combinations"""
        scenarios = []
        scenario_id = 0
        
        # Generate scenarios for each skill level combination
        for red_skill in SkillLevel:
            for blue_skill in SkillLevel:
                for red_strategy in StrategyType:
                    for blue_strategy in StrategyType:
                        scenario_id += 1
                        
                        # Create red alliance parameters
                        red_params = self._create_scenario_parameters(
                            red_skill, red_strategy, "Red"
                        )
                        
                        # Create blue alliance parameters
                        blue_params = self._create_scenario_parameters(
                            blue_skill, blue_strategy, "Blue"
                        )
                        
                        # Generate strategies
                        red_strategy_obj = self.generate_time_based_strategy("Red", red_params)
                        blue_strategy_obj = self.generate_time_based_strategy("Blue", blue_params)
                        
                        # Simulate match
                        result = self.simulator.simulate_match(red_strategy_obj, blue_strategy_obj)
                        
                        # Calculate additional metrics
                        red_total_blocks = sum(red_strategy_obj.blocks_scored_auto.values()) + sum(red_strategy_obj.blocks_scored_driver.values())
                        blue_total_blocks = sum(blue_strategy_obj.blocks_scored_auto.values()) + sum(blue_strategy_obj.blocks_scored_driver.values())
                        
                        scenarios.append({
                            'scenario_id': scenario_id,
                            'red_skill': red_skill.value,
                            'blue_skill': blue_skill.value,
                            'red_strategy': red_strategy.value,
                            'blue_strategy': blue_strategy.value,
                            'red_robot1_role': red_params.robot1_role.value,
                            'red_robot2_role': red_params.robot2_role.value,
                            'blue_robot1_role': blue_params.robot1_role.value,
                            'blue_robot2_role': blue_params.robot2_role.value,
                            'red_cooperation': red_params.cooperation_efficiency,
                            'blue_cooperation': blue_params.cooperation_efficiency,
                            'red_score': result.red_score,
                            'blue_score': result.blue_score,
                            'winner': result.winner,
                            'margin': result.margin,
                            'red_blocks_total': red_total_blocks,
                            'blue_blocks_total': blue_total_blocks,
                            'red_blocks_auto': sum(red_strategy_obj.blocks_scored_auto.values()),
                            'blue_blocks_auto': sum(blue_strategy_obj.blocks_scored_auto.values()),
                            'red_zones': len(red_strategy_obj.zones_controlled),
                            'blue_zones': len(blue_strategy_obj.zones_controlled),
                            'red_parking_score': result.red_breakdown['parking'],
                            'blue_parking_score': result.blue_breakdown['parking'],
                            'match_competitiveness': 1 - abs(result.margin) / max(result.red_score, result.blue_score),
                            'expected_winner': 'red' if red_skill.value > blue_skill.value else 'blue' if blue_skill.value > red_skill.value else 'tie'
                        })
        
        return pd.DataFrame(scenarios)
    
    def _create_scenario_parameters(
        self, 
        skill_level: SkillLevel, 
        strategy_type: StrategyType, 
        alliance_name: str
    ) -> ScenarioParameters:
        """Create scenario parameters for given skill and strategy combination"""
        
        base_capabilities = self.capability_profiles[skill_level]
        
        # Assign robot roles based on strategy
        if strategy_type == StrategyType.ALL_OFFENSE:
            robot1_role, robot2_role = RobotRole.SCORER, RobotRole.SCORER
            cooperation = 0.9
        elif strategy_type == StrategyType.MIXED:
            robot1_role, robot2_role = RobotRole.SCORER, RobotRole.SUPPORT
            cooperation = 0.8
        elif strategy_type == StrategyType.ZONE_CONTROL:
            robot1_role, robot2_role = RobotRole.HYBRID, RobotRole.DEFENDER
            cooperation = 0.85
        elif strategy_type == StrategyType.DEFENSIVE:
            robot1_role, robot2_role = RobotRole.DEFENDER, RobotRole.SUPPORT
            cooperation = 0.75
        else:  # AUTONOMOUS_FOCUS
            robot1_role, robot2_role = RobotRole.SCORER, RobotRole.SCORER
            cooperation = 0.95
        
        # Vary capabilities slightly between robots
        robot1_capabilities = base_capabilities
        robot2_capabilities = RobotCapabilities(
            blocks_per_second=base_capabilities.blocks_per_second * random.uniform(0.9, 1.1),
            max_capacity=base_capabilities.max_capacity,
            travel_time_per_goal=base_capabilities.travel_time_per_goal * random.uniform(0.9, 1.1),
            collection_time=base_capabilities.collection_time * random.uniform(0.9, 1.1),
            accuracy=base_capabilities.accuracy * random.uniform(0.95, 1.0),
            autonomous_reliability=base_capabilities.autonomous_reliability * random.uniform(0.9, 1.0)
        )
        
        return ScenarioParameters(
            skill_level=skill_level,
            strategy_type=strategy_type,
            robot1_role=robot1_role,
            robot2_role=robot2_role,
            robot1_capabilities=robot1_capabilities,
            robot2_capabilities=robot2_capabilities,
            field_position="variable",
            cooperation_efficiency=cooperation
        )
    
    def generate_competitive_scenarios(self, num_scenarios: int = 10) -> List[Tuple[AllianceStrategy, AllianceStrategy]]:
        scenarios = []
        
        for i in range(num_scenarios):
            # Generate strategies with varying competitiveness
            if i < num_scenarios // 3:
                # Close matches
                block_range = (30, 40)
                red_strategy = self.generate_random_strategy("Red", block_range)
                blue_strategy = self.generate_random_strategy("Blue", block_range)
            elif i < 2 * num_scenarios // 3:
                # Medium difference
                red_strategy = self.generate_random_strategy("Red", (25, 35))
                blue_strategy = self.generate_random_strategy("Blue", (35, 45))
            else:
                # Large difference
                red_strategy = self.generate_random_strategy("Red", (20, 30))
                blue_strategy = self.generate_random_strategy("Blue", (40, 50))
            
            scenarios.append((red_strategy, blue_strategy))
        
        return scenarios
    
    def generate_extreme_scenarios(self) -> List[Tuple[AllianceStrategy, AllianceStrategy]]:
        scenarios = []
        
        # Scenario 1: Maximum blocks vs minimal blocks
        max_blocks_strategy = AllianceStrategy(
            name="Max Blocks",
            blocks_scored_auto={"long_1": 10, "long_2": 10, "center_1": 5, "center_2": 5},
            blocks_scored_driver={"long_1": 15, "long_2": 15, "center_1": 14, "center_2": 14},
            zones_controlled=[],
            robots_parked=[ParkingLocation.NONE, ParkingLocation.NONE]
        )
        
        balanced_strategy = AllianceStrategy(
            name="Balanced",
            blocks_scored_auto={"long_1": 5, "long_2": 5, "center_1": 3, "center_2": 3},
            blocks_scored_driver={"long_1": 8, "long_2": 8, "center_1": 6, "center_2": 6},
            zones_controlled=[Zone.RED_HOME, Zone.NEUTRAL],
            robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.PLATFORM]
        )
        
        scenarios.append((max_blocks_strategy, balanced_strategy))
        
        # Scenario 2: Zone control focus vs block focus
        zone_focus = AllianceStrategy(
            name="Zone Focus",
            blocks_scored_auto={"long_1": 3, "long_2": 3, "center_1": 2, "center_2": 2},
            blocks_scored_driver={"long_1": 5, "long_2": 5, "center_1": 5, "center_2": 5},
            zones_controlled=[Zone.BLUE_HOME, Zone.RED_HOME, Zone.NEUTRAL],
            robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.ALLIANCE_ZONE]
        )
        
        block_focus = AllianceStrategy(
            name="Block Focus",
            blocks_scored_auto={"long_1": 8, "long_2": 8, "center_1": 4, "center_2": 4},
            blocks_scored_driver={"long_1": 12, "long_2": 12, "center_1": 8, "center_2": 8},
            zones_controlled=[],
            robots_parked=[ParkingLocation.ALLIANCE_ZONE, ParkingLocation.ALLIANCE_ZONE]
        )
        
        scenarios.append((zone_focus, block_focus))
        
        # Scenario 3: Parking focus
        parking_focus = AllianceStrategy(
            name="Parking Focus",
            blocks_scored_auto={"long_1": 4, "long_2": 4, "center_1": 2, "center_2": 2},
            blocks_scored_driver={"long_1": 6, "long_2": 6, "center_1": 4, "center_2": 4},
            zones_controlled=[Zone.RED_HOME],
            robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.PLATFORM]
        )
        
        no_parking = AllianceStrategy(
            name="No Parking",
            blocks_scored_auto={"long_1": 6, "long_2": 6, "center_1": 4, "center_2": 4},
            blocks_scored_driver={"long_1": 10, "long_2": 10, "center_1": 8, "center_2": 8},
            zones_controlled=[Zone.BLUE_HOME, Zone.NEUTRAL],
            robots_parked=[ParkingLocation.NONE, ParkingLocation.NONE]
        )
        
        scenarios.append((parking_focus, no_parking))
        
        return scenarios
    
    def generate_autonomous_focused_scenarios(self) -> List[Tuple[AllianceStrategy, AllianceStrategy]]:
        scenarios = []
        
        # Strong autonomous vs weak autonomous
        strong_auto = AllianceStrategy(
            name="Strong Auto",
            blocks_scored_auto={"long_1": 8, "long_2": 8, "center_1": 4, "center_2": 4},
            blocks_scored_driver={"long_1": 5, "long_2": 5, "center_1": 3, "center_2": 3},
            zones_controlled=[Zone.RED_HOME],
            robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.ALLIANCE_ZONE]
        )
        
        weak_auto = AllianceStrategy(
            name="Weak Auto",
            blocks_scored_auto={"long_1": 2, "long_2": 2, "center_1": 1, "center_2": 1},
            blocks_scored_driver={"long_1": 10, "long_2": 10, "center_1": 8, "center_2": 8},
            zones_controlled=[Zone.BLUE_HOME],
            robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.ALLIANCE_ZONE]
        )
        
        scenarios.append((strong_auto, weak_auto))
        
        return scenarios
    
    def generate_time_analysis_scenarios(self) -> pd.DataFrame:
        """Generate scenarios specifically for time-based analysis"""
        time_scenarios = []
        
        # Test different scoring rates
        scoring_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # blocks per second
        capacities = [2, 3, 4, 5]  # max blocks held
        cooperation_levels = [0.6, 0.8, 0.9, 1.0]  # cooperation efficiency
        
        scenario_id = 0
        for rate in scoring_rates:
            for capacity in capacities:
                for cooperation in cooperation_levels:
                    scenario_id += 1
                    
                    # Create custom capabilities
                    capabilities = RobotCapabilities(
                        blocks_per_second=rate,
                        max_capacity=capacity,
                        travel_time_per_goal=5.0,
                        collection_time=2.0,
                        accuracy=0.85,
                        autonomous_reliability=0.8
                    )
                    
                    # Calculate theoretical scoring for both periods
                    auto_blocks = self.calculate_realistic_scoring(
                        capabilities, RobotRole.SCORER, 
                        self.time_constraints.autonomous_time - 2.0,
                        cooperation
                    )
                    
                    driver_blocks = self.calculate_realistic_scoring(
                        capabilities, RobotRole.SCORER,
                        self.time_constraints.driver_time - 30.0,  # Account for endgame
                        cooperation,
                        self.time_constraints.interference_factor
                    )
                    
                    time_scenarios.append({
                        'scenario_id': scenario_id,
                        'scoring_rate': rate,
                        'capacity': capacity,
                        'cooperation': cooperation,
                        'auto_blocks_per_robot': auto_blocks,
                        'driver_blocks_per_robot': driver_blocks,
                        'total_blocks_per_robot': auto_blocks + driver_blocks,
                        'alliance_auto_total': auto_blocks * 2,
                        'alliance_driver_total': driver_blocks * 2,
                        'alliance_total': (auto_blocks + driver_blocks) * 2,
                        'theoretical_max_score': ((auto_blocks + driver_blocks) * 2) * 3 + 10,  # Assume auto win
                        'efficiency_rating': (auto_blocks + driver_blocks) / (rate * 120)  # Efficiency vs theoretical max
                    })
        
        return pd.DataFrame(time_scenarios)
    
    def generate_capability_comparison(self) -> pd.DataFrame:
        """Generate comparison of different robot capability profiles"""
        comparisons = []
        
        for skill_level in SkillLevel:
            capabilities = self.capability_profiles[skill_level]
            
            # Test each robot role
            for role in RobotRole:
                # Calculate performance metrics
                auto_performance = self.calculate_realistic_scoring(
                    capabilities, role, 13.0, 0.8  # 13s effective auto time
                )
                
                driver_performance = self.calculate_realistic_scoring(
                    capabilities, role, 75.0, 0.8, 0.85  # 75s effective driver time
                )
                
                comparisons.append({
                    'skill_level': skill_level.value,
                    'robot_role': role.value,
                    'blocks_per_second': capabilities.blocks_per_second,
                    'max_capacity': capabilities.max_capacity,
                    'travel_time': capabilities.travel_time_per_goal,
                    'collection_time': capabilities.collection_time,
                    'accuracy': capabilities.accuracy,
                    'auto_reliability': capabilities.autonomous_reliability,
                    'auto_blocks_expected': auto_performance,
                    'driver_blocks_expected': driver_performance,
                    'total_blocks_expected': auto_performance + driver_performance,
                    'match_contribution_score': (auto_performance + driver_performance) * 3
                })
        
        return pd.DataFrame(comparisons)
    
    def analyze_strategy_effectiveness(self, num_samples: int = 50) -> pd.DataFrame:
        """Analyze effectiveness of different strategy types with ML enhancement"""
        strategy_analysis = []
        match_history = []
        
        for strategy_type in StrategyType:
            for skill_level in SkillLevel:
                wins = 0
                total_scores = []
                margins = []
                
                # Test strategy against random opponents
                for sample_idx in range(num_samples):
                    # Create strategy parameters
                    params = self._create_scenario_parameters(skill_level, strategy_type, "Test")
                    strategy = self.generate_time_based_strategy("Test", params)
                    
                    # Generate random opponent
                    opponent = self.generate_random_strategy("Opponent", (25, 45))
                    
                    # Simulate match
                    result = self.simulator.simulate_match(strategy, opponent)
                    
                    # Record match for ML analysis
                    match_record = {
                        'scenario_id': f"{strategy_type.value}_{skill_level.value}_{sample_idx}",
                        'red_skill': skill_level.value,
                        'blue_skill': 'intermediate',  # Assume average opponent
                        'red_strategy': strategy_type.value,
                        'blue_strategy': 'mixed',  # Random opponent strategy
                        'red_score': result.red_score,
                        'blue_score': result.blue_score,
                        'winner': result.winner,
                        'margin': result.margin,
                        'red_blocks_total': sum(strategy.blocks_scored_auto.values()) + sum(strategy.blocks_scored_driver.values()),
                        'blue_blocks_total': sum(opponent.blocks_scored_auto.values()) + sum(opponent.blocks_scored_driver.values()),
                        'red_blocks_auto': sum(strategy.blocks_scored_auto.values()),
                        'blue_blocks_auto': sum(opponent.blocks_scored_auto.values()),
                        'red_zones': len(strategy.zones_controlled),
                        'blue_zones': len(opponent.zones_controlled),
                        'red_parking_score': result.red_breakdown.get('parking', 0),
                        'blue_parking_score': result.blue_breakdown.get('parking', 0),
                        'red_cooperation': params.cooperation_efficiency,
                        'blue_cooperation': 0.8,  # Default cooperation
                        'match_competitiveness': 1 - abs(result.margin) / max(result.red_score, result.blue_score, 1)
                    }
                    match_history.append(match_record)
                    
                    if result.winner == "red":
                        wins += 1
                        margins.append(result.margin)
                    else:
                        margins.append(-result.margin)
                    
                    total_scores.append(result.red_score)
                
                strategy_analysis.append({
                    'strategy_type': strategy_type.value,
                    'skill_level': skill_level.value,
                    'win_rate': wins / num_samples,
                    'avg_score': np.mean(total_scores),
                    'score_std': np.std(total_scores),
                    'avg_margin': np.mean(margins),
                    'margin_std': np.std(margins),
                    'consistency': 1 - (np.std(total_scores) / np.mean(total_scores))  # Lower variation = higher consistency
                })
        
        # Store match history for ML analysis
        self._match_history = match_history
        
        return pd.DataFrame(strategy_analysis)
    
    def discover_winning_patterns(self, min_win_rate: float = 0.75) -> List[StrategyPattern]:
        """Discover winning patterns using ML analysis"""
        if not hasattr(self, '_match_history') or not self._match_history:
            print("No match history available. Run analyze_strategy_effectiveness() first.")
            return []
        
        return self.ml_discovery.discover_winning_patterns(self._match_history, min_win_rate)
    
    def generate_ml_optimized_scenarios(
        self, 
        constraints: Optional[Dict[str, Any]] = None,
        num_scenarios: int = 10
    ) -> List[Tuple[AllianceStrategy, AllianceStrategy]]:
        """Generate scenarios optimized using ML-discovered patterns"""
        
        if constraints is None:
            constraints = {
                'competitiveness_target': 0.7,
                'min_skill_level': 'intermediate',
                'strategy_diversity': True
            }
        
        # Use ML to generate optimal scenario parameters
        optimal_params = self.ml_discovery.generate_optimal_scenarios(
            constraints, 
            num_generations=30,
            population_size=50
        )
        
        scenarios = []
        
        for i, params in enumerate(optimal_params[:num_scenarios]):
            try:
                # Convert ML parameters to strategy objects
                red_skill = SkillLevel(params['red_skill'])
                blue_skill = SkillLevel(params['blue_skill'])
                red_strategy_type = StrategyType(params['red_strategy'])
                blue_strategy_type = StrategyType(params['blue_strategy'])
                
                # Create scenario parameters
                red_params = self._create_scenario_parameters(red_skill, red_strategy_type, f"Red_ML_{i}")
                blue_params = self._create_scenario_parameters(blue_skill, blue_strategy_type, f"Blue_ML_{i}")
                
                # Adjust cooperation based on ML recommendations
                red_params.cooperation_efficiency = params['red_cooperation']
                blue_params.cooperation_efficiency = params['blue_cooperation']
                
                # Generate strategies
                red_strategy = self.generate_time_based_strategy(f"Red_ML_Optimized_{i}", red_params)
                blue_strategy = self.generate_time_based_strategy(f"Blue_ML_Optimized_{i}", blue_params)
                
                scenarios.append((red_strategy, blue_strategy))
                
            except Exception as e:
                print(f"Warning: Failed to create ML-optimized scenario {i}: {e}")
                # Fallback to traditional generation
                red_strategy = self.generate_random_strategy(f"Red_Fallback_{i}")
                blue_strategy = self.generate_random_strategy(f"Blue_Fallback_{i}")
                scenarios.append((red_strategy, blue_strategy))
        
        print(f"Generated {len(scenarios)} ML-optimized scenarios")
        return scenarios
    
    def analyze_critical_moments_in_scenarios(
        self, 
        scenarios: List[Tuple[AllianceStrategy, AllianceStrategy]]
    ) -> Dict[str, List[CriticalMoment]]:
        """Analyze critical moments in a set of scenarios"""
        
        scenario_critical_moments = {}
        
        for i, (red_strategy, blue_strategy) in enumerate(scenarios):
            # Simulate match to generate timeline
            result = self.simulator.simulate_match(red_strategy, blue_strategy)
            
            # Create synthetic match timeline based on strategies
            timeline = self._create_match_timeline(red_strategy, blue_strategy, result)
            
            # Identify critical moments
            critical_moments = self.ml_discovery.identify_critical_moments(timeline)
            
            scenario_critical_moments[f"scenario_{i}"] = critical_moments
        
        return scenario_critical_moments
    
    def _create_match_timeline(
        self, 
        red_strategy: AllianceStrategy, 
        blue_strategy: AllianceStrategy, 
        result
    ) -> List[MatchEvent]:
        """Create a synthetic match timeline for critical moment analysis"""
        
        timeline = []
        
        # Autonomous period events (0-15 seconds)
        auto_time_points = np.linspace(1, 15, 8)
        for t in auto_time_points:
            # Red scoring events
            if sum(red_strategy.blocks_scored_auto.values()) > 0:
                timeline.append(MatchEvent(
                    timestamp=t,
                    event_type='score',
                    alliance='red',
                    details={'points': random.randint(3, 9), 'source': 'autonomous'},
                    impact_score=5.0
                ))
            
            # Blue scoring events  
            if sum(blue_strategy.blocks_scored_auto.values()) > 0 and random.random() < 0.7:
                timeline.append(MatchEvent(
                    timestamp=t + 0.5,
                    event_type='score',
                    alliance='blue',
                    details={'points': random.randint(3, 6), 'source': 'autonomous'},
                    impact_score=4.0
                ))
        
        # Driver control period events (15-120 seconds)
        driver_time_points = np.linspace(20, 110, 15)
        for t in driver_time_points:
            # Scoring events based on strategy aggressiveness
            red_blocks_total = sum(red_strategy.blocks_scored_driver.values())
            blue_blocks_total = sum(blue_strategy.blocks_scored_driver.values())
            
            if red_blocks_total > 0:
                timeline.append(MatchEvent(
                    timestamp=t,
                    event_type='score',
                    alliance='red',
                    details={'points': random.randint(3, 12), 'source': 'driver_control'},
                    impact_score=random.uniform(2.0, 8.0)
                ))
            
            if blue_blocks_total > 0:
                timeline.append(MatchEvent(
                    timestamp=t + random.uniform(0.5, 2.0),
                    event_type='score',
                    alliance='blue',
                    details={'points': random.randint(3, 10), 'source': 'driver_control'},
                    impact_score=random.uniform(2.0, 7.0)
                ))
            
            # Zone control events
            if len(red_strategy.zones_controlled) > 0 and random.random() < 0.3:
                timeline.append(MatchEvent(
                    timestamp=t + 1.0,
                    event_type='zone_control',
                    alliance='red',
                    details={'zones': len(red_strategy.zones_controlled)},
                    impact_score=3.0
                ))
        
        # Endgame events (90-120 seconds)
        endgame_time_points = [95, 105, 115]
        for t in endgame_time_points:
            # Parking events
            red_parking_count = sum(1 for parking in red_strategy.robots_parked 
                                  if parking != ParkingLocation.NONE)
            if red_parking_count > 0:
                timeline.append(MatchEvent(
                    timestamp=t,
                    event_type='parking',
                    alliance='red',
                    details={'robots_parked': red_parking_count},
                    impact_score=6.0
                ))
        
        # Sort timeline by timestamp
        timeline.sort(key=lambda x: x.timestamp)
        
        return timeline
    
    def get_pattern_insights(self) -> Dict[str, Any]:
        """Get insights from discovered patterns"""
        
        if not self.ml_discovery.discovered_patterns:
            return {
                'total_patterns': 0,
                'insights': ['No patterns discovered yet. Run pattern discovery first.']
            }
        
        insights = []
        patterns = self.ml_discovery.discovered_patterns
        
        # Analyze pattern types
        pattern_types = {}
        for pattern in patterns:
            pattern_types[pattern.pattern_type] = pattern_types.get(pattern.pattern_type, 0) + 1
        
        insights.append(f"Discovered {len(patterns)} strategic patterns across {len(pattern_types)} categories")
        
        # High win rate patterns
        high_win_patterns = [p for p in patterns if p.win_rate > 0.8]
        if high_win_patterns:
            insights.append(f"Found {len(high_win_patterns)} patterns with >80% win rate")
            for pattern in high_win_patterns[:3]:  # Top 3
                insights.append(f"  • {pattern.description}")
        
        # Most frequent patterns
        frequent_patterns = sorted(patterns, key=lambda x: x.frequency, reverse=True)[:3]
        insights.append("Most common winning patterns:")
        for pattern in frequent_patterns:
            insights.append(f"  • {pattern.description} (appears in {pattern.frequency:.1%} of matches)")
        
        return {
            'total_patterns': len(patterns),
            'pattern_types': pattern_types,
            'high_win_rate_count': len(high_win_patterns),
            'insights': insights,
            'patterns': patterns
        }
    
    def answer_strategic_questions(self, question_type: str) -> str:
        """Answer strategic questions using discovered patterns"""
        
        patterns = self.ml_discovery.discovered_patterns
        
        if not patterns:
            return "No patterns available. Please run pattern discovery first."
        
        if question_type == "high_win_scoring_patterns":
            high_win_patterns = [p for p in patterns if p.win_rate >= 0.8 and 'scoring' in p.pattern_type.lower()]
            if high_win_patterns:
                answer = "Scoring patterns in 80%+ win rate matches:\n"
                for pattern in high_win_patterns:
                    answer += f"• {pattern.description}\n"
                    if 'red_blocks_auto' in pattern.key_features:
                        answer += f"  - Autonomous scoring: {pattern.key_features['red_blocks_auto']:.1f} blocks\n"
                    if 'red_blocks_total' in pattern.key_features:
                        answer += f"  - Total scoring: {pattern.key_features['red_blocks_total']:.1f} blocks\n"
                return answer
            else:
                return "No high-win-rate scoring patterns found in current data."
        
        elif question_type == "optimal_strategy_switch_timing":
            temporal_patterns = [p for p in patterns if p.pattern_type == "temporal"]
            if temporal_patterns:
                answer = "Optimal strategy switch timing insights:\n"
                for pattern in temporal_patterns:
                    if "autonomous" in pattern.description.lower():
                        answer += f"• Strong autonomous performance leads to {pattern.win_rate:.1%} win rate\n"
                    if "driver" in pattern.description.lower():
                        answer += f"• Driver control emphasis shows {pattern.win_rate:.1%} success rate\n"
                answer += "\nRecommendation: Switch from autonomous focus to driver control around 15-30 seconds for optimal results."
                return answer
            else:
                return "No temporal strategy patterns found. More match data needed."
        
        elif question_type == "effective_coordination_patterns":
            coord_patterns = [p for p in patterns if 'coordination' in p.description.lower() or 'cooperation' in str(p.key_features)]
            if coord_patterns:
                answer = "Most effective two-robot coordination patterns:\n"
                for pattern in coord_patterns:
                    answer += f"• {pattern.description} (Win rate: {pattern.win_rate:.1%})\n"
                    if 'red_cooperation' in pattern.key_features:
                        answer += f"  - Cooperation efficiency: {pattern.key_features['red_cooperation']:.2f}\n"
                return answer
            else:
                return "No specific coordination patterns identified. Consider analyzing robot role combinations."
        
        else:
            return f"Question type '{question_type}' not recognized. Available types: 'high_win_scoring_patterns', 'optimal_strategy_switch_timing', 'effective_coordination_patterns'"
    
    def export_ml_analysis(self, base_filename: str) -> None:
        """Export ML analysis results"""
        
        # Export discovered patterns
        if self.ml_discovery.discovered_patterns:
            pattern_file = f"{base_filename}_patterns.json"
            self.ml_discovery.export_patterns(pattern_file)
        
        # Export insights
        insights = self.get_pattern_insights()
        insights_file = f"{base_filename}_insights.json"
        with open(insights_file, 'w') as f:
            json.dump(insights, f, indent=2, default=str)
        
        # Create visualization
        viz_file = f"{base_filename}_patterns.png"
        self.ml_discovery.visualize_patterns(viz_file)
        
        print(f"ML analysis exported to {base_filename}_* files")


if __name__ == "__main__":
    print("🤖 VEX U ML-Enhanced Scenario Generator Demo")
    print("=" * 60)
    
    # Initialize simulator and generator
    from .simulator import ScoringSimulator
    simulator = ScoringSimulator()
    generator = ScenarioGenerator(simulator, enable_ml=True)
    
    # Generate and analyze strategy effectiveness
    print("\n📊 Analyzing Strategy Effectiveness...")
    strategy_df = generator.analyze_strategy_effectiveness(num_samples=20)  # Smaller sample for demo
    print(f"Analyzed {len(strategy_df)} strategy combinations")
    
    # Top performing strategies
    top_strategies = strategy_df.nlargest(3, 'win_rate')
    print("\n🏆 Top Performing Strategies:")
    for _, row in top_strategies.iterrows():
        print(f"  {row['strategy_type']} ({row['skill_level']}): {row['win_rate']:.1%} win rate, {row['avg_score']:.0f} avg score")
    
    # Discover winning patterns
    print("\n🔍 Discovering Winning Patterns...")
    patterns = generator.discover_winning_patterns(min_win_rate=0.7)
    print(f"Discovered {len(patterns)} winning patterns")
    
    if patterns:
        print("\n💡 Pattern Insights:")
        insights = generator.get_pattern_insights()
        for insight in insights['insights'][:5]:
            print(f"  • {insight}")
    
    # Generate ML-optimized scenarios
    print("\n⚡ Generating ML-Optimized Scenarios...")
    ml_scenarios = generator.generate_ml_optimized_scenarios(num_scenarios=3)
    
    print(f"\n🎯 Testing {len(ml_scenarios)} ML-Optimized Scenarios:")
    for i, (red, blue) in enumerate(ml_scenarios):
        result = simulator.simulate_match(red, blue)
        competitiveness = 1 - abs(result.margin) / max(result.red_score, result.blue_score, 1)
        print(f"  Scenario {i+1}: {result.winner.upper()} wins by {result.margin}")
        print(f"    Red: {result.red_score} points | Blue: {result.blue_score} points")
        print(f"    Competitiveness: {competitiveness:.1%}")
    
    # Analyze critical moments
    print("\n⏰ Analyzing Critical Moments...")
    critical_moments = generator.analyze_critical_moments_in_scenarios(ml_scenarios[:2])
    
    for scenario_id, moments in critical_moments.items():
        if moments:
            print(f"\n  {scenario_id}: {len(moments)} critical moments identified")
            for moment in moments[:2]:  # Show first 2
                print(f"    • {moment.timestamp:.1f}s: {moment.optimal_choice} (impact: {moment.impact_magnitude:.1f})")
    
    # Answer strategic questions
    print("\n❓ Strategic Question Analysis:")
    
    questions = [
        ("high_win_scoring_patterns", "What scoring patterns appear in 80%+ win rate matches?"),
        ("optimal_strategy_switch_timing", "When is optimal time to switch strategy?"),
        ("effective_coordination_patterns", "What coordination patterns are most effective?")
    ]
    
    for q_type, q_text in questions:
        print(f"\n🤔 {q_text}")
        answer = generator.answer_strategic_questions(q_type)
        # Show first 2 lines of answer
        answer_lines = answer.split('\n')[:2]
        for line in answer_lines:
            if line.strip():
                print(f"  💭 {line.strip()}")
    
    # Export analysis
    print("\n💾 Exporting ML Analysis...")
    try:
        generator.export_ml_analysis("vex_u_scenario_analysis")
        print("  ✅ Analysis exported successfully")
    except Exception as e:
        print(f"  ⚠️  Export warning: {e}")
    
    print("\n✅ ML-Enhanced Scenario Generator Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("  • Pattern discovery using unsupervised learning")
    print("  • Critical moment identification with change point detection")
    print("  • Genetic algorithm-based scenario optimization")
    print("  • Strategic question answering system")
    print("  • Integration with existing VEX U simulation framework")