"""
Push Back-specific Monte Carlo simulation engine.

This module provides fast, accurate simulation of Push Back matches with realistic
robot performance modeling and strategic insight generation optimized for early
season team strategy development.
"""

import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

class ParkingStrategy(Enum):
    """Parking strategy options"""
    NEVER = "never"
    EARLY = "early"  # Park with 20+ seconds remaining
    LATE = "late"    # Park with 10-15 seconds remaining
    DESPERATE = "desperate"  # Park only if losing badly

class GoalPriority(Enum):
    """Goal targeting priority"""
    CENTER_ONLY = "center_only"
    LONG_ONLY = "long_only"
    CENTER_PREFERRED = "center_preferred"
    LONG_PREFERRED = "long_preferred"
    BALANCED = "balanced"

class AutonomousStrategy(Enum):
    """Autonomous strategy focus"""
    AGGRESSIVE = "aggressive"  # Focus on autonomous win
    SAFE = "safe"  # Focus on consistent points
    HYBRID = "hybrid"  # Balance between both

@dataclass
class RobotCapabilities:
    """Realistic robot performance characteristics"""
    # Cycle times (seconds)
    min_cycle_time: float = 3.0
    max_cycle_time: float = 8.0
    average_cycle_time: float = 5.0
    
    # Movement speeds (field units per second)
    max_speed: float = 4.0
    average_speed: float = 2.5
    
    # Reliability (0.0 to 1.0)
    pickup_reliability: float = 0.95
    scoring_reliability: float = 0.98
    autonomous_reliability: float = 0.90
    
    # Strategy preferences
    parking_strategy: ParkingStrategy = ParkingStrategy.LATE
    goal_priority: GoalPriority = GoalPriority.BALANCED
    autonomous_strategy: AutonomousStrategy = AutonomousStrategy.HYBRID
    
    # Block handling
    max_blocks_per_trip: int = 1
    prefers_singles: bool = True
    
    # Control zone behavior
    control_zone_frequency: float = 0.3  # How often to prioritize control
    control_zone_duration: float = 5.0   # Seconds spent in control zone

@dataclass
class MatchState:
    """Current match state during simulation"""
    time_remaining: float = 105.0
    red_score: int = 0
    blue_score: int = 0
    red_blocks_scored: int = 0
    blue_blocks_scored: int = 0
    blocks_remaining: int = 88
    red_control_zone_time: float = 0.0
    blue_control_zone_time: float = 0.0
    red_parked: bool = False
    blue_parked: bool = False
    autonomous_winner: Optional[str] = None

@dataclass
class SimulationResult:
    """Result of a single match simulation"""
    winner: str  # "red", "blue", or "tie"
    final_score_red: int
    final_score_blue: int
    score_margin: int
    blocks_scored_red: int
    blocks_scored_blue: int
    autonomous_winner: Optional[str]
    red_parked: bool
    blue_parked: bool
    match_duration: float
    critical_moments: List[Dict]  # Key decision points

@dataclass
class StrategyInsights:
    """Strategic insights generated from simulation results"""
    win_probability: float
    average_score: float
    score_variance: float
    critical_timings: Dict[str, float]
    optimal_decisions: Dict[str, str]
    risk_factors: List[str]
    improvement_opportunities: List[str]

class PushBackMonteCarloEngine:
    """
    High-performance Monte Carlo simulation engine for Push Back matches.
    
    Optimized for early season strategy development with focus on:
    - Realistic robot performance modeling
    - Push Back-specific game mechanics
    - Fast execution (1000+ simulations in <10 seconds)
    - Strategic insight generation
    """
    
    def __init__(self, red_robot: RobotCapabilities, blue_robot: RobotCapabilities):
        self.red_robot = red_robot
        self.blue_robot = blue_robot
        self.random = random.Random()
        
        # Push Back field constants
        self.FIELD_SIZE = 12.0  # 12x12 feet
        self.TOTAL_BLOCKS = 88
        self.MATCH_DURATION = 105.0  # seconds
        self.AUTONOMOUS_DURATION = 15.0  # seconds
        
        # Scoring constants
        self.POINTS_PER_BLOCK = 3
        self.CONTROL_ZONE_POINTS_MIN = 6
        self.CONTROL_ZONE_POINTS_MAX = 10
        self.PARKING_POINTS_TOUCHED = 8
        self.PARKING_POINTS_COMPLETELY = 30
        self.AUTONOMOUS_WIN_POINTS = 7
        
        # Performance tracking
        self.simulation_count = 0
        self.last_performance_time = 0.0
    
    def simulate_match(self, match_id: int = 0) -> SimulationResult:
        """Simulate a single Push Back match"""
        self.simulation_count += 1
        
        # Initialize match state
        state = MatchState()
        critical_moments = []
        
        # Simulate autonomous period
        auto_result = self._simulate_autonomous(state)
        if auto_result:
            critical_moments.append({
                "time": 15.0,
                "event": "autonomous_winner",
                "winner": auto_result,
                "impact": self.AUTONOMOUS_WIN_POINTS
            })
        
        # Simulate driver control period
        while state.time_remaining > 0 and state.blocks_remaining > 0:
            # Determine next action for each alliance
            red_action = self._choose_action(self.red_robot, state, "red")
            blue_action = self._choose_action(self.blue_robot, state, "blue")
            
            # Execute actions simultaneously
            time_step = self._execute_actions(state, red_action, blue_action, critical_moments)
            state.time_remaining -= time_step
        
        # Handle end-game parking
        self._handle_parking(state, critical_moments)
        
        # Calculate final scores
        final_red, final_blue = self._calculate_final_scores(state)
        
        # Determine winner
        if final_red > final_blue:
            winner = "red"
        elif final_blue > final_red:
            winner = "blue"
        else:
            winner = "tie"
        
        return SimulationResult(
            winner=winner,
            final_score_red=final_red,
            final_score_blue=final_blue,
            score_margin=abs(final_red - final_blue),
            blocks_scored_red=state.red_blocks_scored,
            blocks_scored_blue=state.blue_blocks_scored,
            autonomous_winner=state.autonomous_winner,
            red_parked=state.red_parked,
            blue_parked=state.blue_parked,
            match_duration=105.0 - state.time_remaining,
            critical_moments=critical_moments
        )
    
    def _simulate_autonomous(self, state: MatchState) -> Optional[str]:
        """Simulate the 15-second autonomous period"""
        red_auto_success = self.random.random() < self.red_robot.autonomous_reliability
        blue_auto_success = self.random.random() < self.blue_robot.autonomous_reliability
        
        # Simple autonomous scoring (1-3 blocks typically)
        if red_auto_success:
            red_auto_blocks = self.random.randint(1, 3)
            state.red_blocks_scored += red_auto_blocks
            state.red_score += red_auto_blocks * self.POINTS_PER_BLOCK
            state.blocks_remaining -= red_auto_blocks
        
        if blue_auto_success:
            blue_auto_blocks = self.random.randint(1, 3)
            state.blue_blocks_scored += blue_auto_blocks
            state.blue_score += blue_auto_blocks * self.POINTS_PER_BLOCK
            state.blocks_remaining -= blue_auto_blocks
        
        # Determine autonomous winner
        if state.red_score > state.blue_score:
            state.autonomous_winner = "red"
            return "red"
        elif state.blue_score > state.red_score:
            state.autonomous_winner = "blue"
            return "blue"
        
        return None
    
    def _choose_action(self, robot: RobotCapabilities, state: MatchState, alliance: str) -> str:
        """Choose the next action for a robot based on strategy and game state"""
        
        # Check if should park
        if self._should_park(robot, state):
            return "park"
        
        # Check if should focus on control zones
        if (self.random.random() < robot.control_zone_frequency and 
            state.time_remaining > 20):
            return "control_zone"
        
        # Default to block scoring
        return "score_blocks"
    
    def _should_park(self, robot: RobotCapabilities, state: MatchState) -> bool:
        """Determine if robot should park based on strategy and game state"""
        if robot.parking_strategy == ParkingStrategy.NEVER:
            return False
        
        score_diff = state.red_score - state.blue_score
        
        if robot.parking_strategy == ParkingStrategy.EARLY:
            return state.time_remaining <= 25
        elif robot.parking_strategy == ParkingStrategy.LATE:
            return state.time_remaining <= 15
        elif robot.parking_strategy == ParkingStrategy.DESPERATE:
            return state.time_remaining <= 10 and abs(score_diff) > 20
        
        return False
    
    def _execute_actions(self, state: MatchState, red_action: str, blue_action: str, 
                        critical_moments: List[Dict]) -> float:
        """Execute actions for both alliances and return time elapsed"""
        
        time_elapsed = 0.0
        
        # Execute red alliance action
        if red_action == "score_blocks":
            cycle_time = self._get_cycle_time(self.red_robot)
            # Attempt to pick up blocks
            if self.random.random() < self.red_robot.pickup_reliability:
                blocks_picked = self.red_robot.max_blocks_per_trip
                # Attempt to score blocks
                if self.random.random() < self.red_robot.scoring_reliability:
                    blocks_scored = blocks_picked
                    state.red_blocks_scored += blocks_scored
                    state.red_score += blocks_scored * self.POINTS_PER_BLOCK
                    state.blocks_remaining = max(0, state.blocks_remaining - blocks_scored)
            time_elapsed = max(time_elapsed, cycle_time)
            
        elif red_action == "control_zone":
            control_time = self.red_robot.control_zone_duration
            state.red_control_zone_time += control_time
            time_elapsed = max(time_elapsed, control_time)
            
        elif red_action == "park":
            state.red_parked = True
            critical_moments.append({
                "time": state.time_remaining,
                "event": "red_park",
                "impact": self.PARKING_POINTS_TOUCHED
            })
        
        # Execute blue alliance action
        if blue_action == "score_blocks":
            cycle_time = self._get_cycle_time(self.blue_robot)
            # Attempt to pick up blocks
            if self.random.random() < self.blue_robot.pickup_reliability:
                blocks_picked = self.blue_robot.max_blocks_per_trip
                # Attempt to score blocks
                if self.random.random() < self.blue_robot.scoring_reliability:
                    blocks_scored = blocks_picked
                    state.blue_blocks_scored += blocks_scored
                    state.blue_score += blocks_scored * self.POINTS_PER_BLOCK
                    state.blocks_remaining = max(0, state.blocks_remaining - blocks_scored)
            time_elapsed = max(time_elapsed, cycle_time)
            
        elif blue_action == "control_zone":
            control_time = self.blue_robot.control_zone_duration
            state.blue_control_zone_time += control_time
            time_elapsed = max(time_elapsed, control_time)
            
        elif blue_action == "park":
            state.blue_parked = True
            critical_moments.append({
                "time": state.time_remaining,
                "event": "blue_park",
                "impact": self.PARKING_POINTS_TOUCHED
            })
        
        return max(time_elapsed, 1.0)  # Minimum 1 second per cycle
    
    def _get_cycle_time(self, robot: RobotCapabilities) -> float:
        """Generate realistic cycle time with variation"""
        # Use normal distribution around average with realistic bounds
        cycle_time = self.random.normalvariate(robot.average_cycle_time, 1.0)
        return max(robot.min_cycle_time, min(robot.max_cycle_time, cycle_time))
    
    def _handle_parking(self, state: MatchState, critical_moments: List[Dict]):
        """Handle end-game parking decisions"""
        # Last chance parking decisions based on strategy
        if (not state.red_parked and state.time_remaining <= 5 and 
            self.red_robot.parking_strategy != ParkingStrategy.NEVER):
            if self.random.random() < 0.7:  # 70% chance of last-second parking
                state.red_parked = True
                critical_moments.append({
                    "time": state.time_remaining,
                    "event": "red_last_second_park",
                    "impact": self.PARKING_POINTS_TOUCHED
                })
        
        if (not state.blue_parked and state.time_remaining <= 5 and
            self.blue_robot.parking_strategy != ParkingStrategy.NEVER):
            if self.random.random() < 0.7:
                state.blue_parked = True
                critical_moments.append({
                    "time": state.time_remaining,
                    "event": "blue_last_second_park",
                    "impact": self.PARKING_POINTS_TOUCHED
                })
    
    def _calculate_final_scores(self, state: MatchState) -> Tuple[int, int]:
        """Calculate final scores including all bonuses"""
        red_final = state.red_score
        blue_final = state.blue_score
        
        # Control zone bonuses (simplified: time-based points)
        red_control_bonus = min(self.CONTROL_ZONE_POINTS_MAX, 
                               int(state.red_control_zone_time / 10) * 2)
        blue_control_bonus = min(self.CONTROL_ZONE_POINTS_MAX,
                                int(state.blue_control_zone_time / 10) * 2)
        
        red_final += red_control_bonus
        blue_final += blue_control_bonus
        
        # Parking bonuses
        if state.red_parked:
            red_final += self.PARKING_POINTS_TOUCHED
        if state.blue_parked:
            blue_final += self.PARKING_POINTS_TOUCHED
        
        # Autonomous win bonus
        if state.autonomous_winner == "red":
            red_final += self.AUTONOMOUS_WIN_POINTS
        elif state.autonomous_winner == "blue":
            blue_final += self.AUTONOMOUS_WIN_POINTS
        
        return red_final, blue_final
    
    def run_simulation(self, num_simulations: int = 1000, 
                      use_parallel: bool = True) -> Tuple[List[SimulationResult], float]:
        """
        Run multiple simulations and return results with performance timing.
        
        Args:
            num_simulations: Number of matches to simulate
            use_parallel: Whether to use parallel processing
            
        Returns:
            Tuple of (simulation results, execution time in seconds)
        """
        start_time = time.time()
        results = []
        
        if use_parallel and num_simulations > 100:
            # Use parallel processing for large simulation runs
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(self.simulate_match, i) 
                          for i in range(num_simulations)]
                
                for future in as_completed(futures):
                    results.append(future.result())
        else:
            # Sequential execution for smaller runs
            for i in range(num_simulations):
                results.append(self.simulate_match(i))
        
        execution_time = time.time() - start_time
        self.last_performance_time = execution_time
        
        return results, execution_time
    
    def generate_insights(self, results: List[SimulationResult], 
                         alliance: str = "red") -> StrategyInsights:
        """Generate strategic insights from simulation results"""
        
        if alliance == "red":
            wins = sum(1 for r in results if r.winner == "red")
            scores = [r.final_score_red for r in results]
            opponent_scores = [r.final_score_blue for r in results]
        else:
            wins = sum(1 for r in results if r.winner == "blue")
            scores = [r.final_score_blue for r in results]
            opponent_scores = [r.final_score_red for r in results]
        
        win_probability = wins / len(results)
        average_score = statistics.mean(scores)
        score_variance = statistics.variance(scores) if len(scores) > 1 else 0.0
        
        # Analyze critical timings
        critical_timings = self._analyze_critical_timings(results)
        
        # Generate optimal decisions
        optimal_decisions = self._generate_optimal_decisions(results, alliance)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(results, alliance)
        
        # Find improvement opportunities
        improvements = self._find_improvements(results, alliance)
        
        return StrategyInsights(
            win_probability=win_probability,
            average_score=average_score,
            score_variance=score_variance,
            critical_timings=critical_timings,
            optimal_decisions=optimal_decisions,
            risk_factors=risk_factors,
            improvement_opportunities=improvements
        )
    
    def _analyze_critical_timings(self, results: List[SimulationResult]) -> Dict[str, float]:
        """Analyze timing of critical match moments"""
        park_times = []
        close_match_times = []
        
        for result in results:
            for moment in result.critical_moments:
                if "park" in moment["event"]:
                    park_times.append(moment["time"])
                if result.score_margin <= 10:
                    close_match_times.append(105 - result.match_duration)
        
        timings = {}
        if park_times:
            timings["optimal_park_time"] = statistics.mean(park_times)
        if close_match_times:
            timings["close_match_average"] = statistics.mean(close_match_times)
        
        return timings
    
    def _generate_optimal_decisions(self, results: List[SimulationResult], 
                                   alliance: str) -> Dict[str, str]:
        """Generate optimal decision recommendations"""
        decisions = {}
        
        # Parking strategy analysis
        park_wins = sum(1 for r in results 
                       if (alliance == "red" and r.red_parked and r.winner == "red") or
                          (alliance == "blue" and r.blue_parked and r.winner == "blue"))
        no_park_wins = sum(1 for r in results 
                          if (alliance == "red" and not r.red_parked and r.winner == "red") or
                             (alliance == "blue" and not r.blue_parked and r.winner == "blue"))
        
        if park_wins > no_park_wins:
            decisions["parking"] = "Always park when possible"
        else:
            decisions["parking"] = "Focus on scoring over parking"
        
        # Autonomous strategy
        auto_wins = sum(1 for r in results if r.autonomous_winner == alliance)
        if auto_wins / len(results) > 0.6:
            decisions["autonomous"] = "Maintain aggressive autonomous strategy"
        else:
            decisions["autonomous"] = "Focus on consistency over aggression"
        
        return decisions
    
    def _identify_risk_factors(self, results: List[SimulationResult], 
                              alliance: str) -> List[str]:
        """Identify potential risk factors in strategy"""
        risks = []
        
        # High variance in scores
        scores = ([r.final_score_red for r in results] if alliance == "red" 
                 else [r.final_score_blue for r in results])
        if statistics.variance(scores) > 400:  # High variance threshold
            risks.append("High score variability - strategy may be inconsistent")
        
        # Low win rate in close matches
        close_matches = [r for r in results if r.score_margin <= 15]
        if close_matches:
            close_wins = sum(1 for r in close_matches if r.winner == alliance)
            if close_wins / len(close_matches) < 0.4:
                risks.append("Poor performance in close matches")
        
        # Autonomous dependency
        auto_wins = sum(1 for r in results if r.autonomous_winner == alliance)
        total_wins = sum(1 for r in results if r.winner == alliance)
        if total_wins > 0 and auto_wins / total_wins > 0.7:
            risks.append("Over-dependent on autonomous success")
        
        return risks
    
    def _find_improvements(self, results: List[SimulationResult], 
                          alliance: str) -> List[str]:
        """Identify improvement opportunities"""
        improvements = []
        
        # Analyze losses for patterns
        losses = [r for r in results if r.winner != alliance and r.winner != "tie"]
        
        if losses:
            # Check if losses are due to parking
            parking_losses = sum(1 for r in losses 
                               if (alliance == "red" and r.blue_parked and not r.red_parked) or
                                  (alliance == "blue" and r.red_parked and not r.blue_parked))
            
            if parking_losses / len(losses) > 0.3:
                improvements.append("Improve parking decision timing")
            
            # Check autonomous performance
            auto_losses = sum(1 for r in losses 
                            if r.autonomous_winner and r.autonomous_winner != alliance)
            if auto_losses / len(losses) > 0.4:
                improvements.append("Focus on autonomous consistency")
            
            # Check for low-scoring losses
            low_scoring_losses = sum(1 for r in losses 
                                   if (alliance == "red" and r.final_score_red < 100) or
                                      (alliance == "blue" and r.final_score_blue < 100))
            if low_scoring_losses / len(losses) > 0.3:
                improvements.append("Optimize cycle times and efficiency")
        
        return improvements

def create_default_robot() -> RobotCapabilities:
    """Create a robot with default Push Back capabilities"""
    return RobotCapabilities()

def create_competitive_robot() -> RobotCapabilities:
    """Create a competitive-level robot for Push Back"""
    return RobotCapabilities(
        min_cycle_time=2.5,
        max_cycle_time=6.5,
        average_cycle_time=4.2,
        max_speed=4.5,
        average_speed=3.0,
        pickup_reliability=0.97,
        scoring_reliability=0.99,
        autonomous_reliability=0.92,
        parking_strategy=ParkingStrategy.LATE,
        goal_priority=GoalPriority.CENTER_PREFERRED,
        autonomous_strategy=AutonomousStrategy.AGGRESSIVE,
        max_blocks_per_trip=2,
        prefers_singles=False,
        control_zone_frequency=0.4,
        control_zone_duration=4.0
    )

def create_beginner_robot() -> RobotCapabilities:
    """Create a beginner-level robot for Push Back"""
    return RobotCapabilities(
        min_cycle_time=4.0,
        max_cycle_time=12.0,
        average_cycle_time=8.0,
        max_speed=2.0,
        average_speed=1.5,
        pickup_reliability=0.85,
        scoring_reliability=0.90,
        autonomous_reliability=0.70,
        parking_strategy=ParkingStrategy.EARLY,
        goal_priority=GoalPriority.BALANCED,
        autonomous_strategy=AutonomousStrategy.SAFE,
        max_blocks_per_trip=1,
        prefers_singles=True,
        control_zone_frequency=0.2,
        control_zone_duration=6.0
    )