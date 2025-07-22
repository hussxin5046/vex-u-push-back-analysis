#!/usr/bin/env python3

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math

from core.simulator import (
    AllianceStrategy, ScoringSimulator, Zone, ParkingLocation, GameConstants
)
from core.scenario_generator import ScenarioGenerator, SkillLevel


class GamePhase(Enum):
    AUTONOMOUS = "autonomous"
    EARLY_DRIVER = "early_driver"
    MID_DRIVER = "mid_driver"
    ENDGAME = "endgame"


@dataclass
class TimeValueAnalysis:
    phase: GamePhase
    seconds_remaining: int
    block_value: float  # Expected points per block at this time
    defense_value: float  # Expected points saved per block defended
    opportunity_cost: float  # Cost of not doing alternative action
    critical_threshold: bool  # Whether this is a critical time period
    recommended_action: str


@dataclass
class GoalPrioritization:
    goal_name: str
    priority_score: float
    expected_blocks_per_minute: float
    zone_control_value: float
    interference_risk: float
    recommended_timing: str


@dataclass
class BreakevenAnalysis:
    current_score: int
    opponent_score: int
    time_remaining: int
    blocks_needed_to_win: int
    safety_margin_blocks: int
    required_scoring_rate: float  # blocks per minute
    win_probability: float
    defensive_threshold: int  # Score difference to switch to defense


@dataclass
class AutonomousOptimization:
    max_theoretical_score: int
    realistic_score_range: Tuple[int, int]
    autonomous_bonus_probability: float
    optimal_starting_position: str
    recommended_block_targets: Dict[str, int]
    time_allocation: Dict[str, float]


@dataclass
class EfficiencyMetrics:
    points_per_second: float
    blocks_per_minute: float
    efficiency_vs_defense: float  # Efficiency reduction when playing defense
    scoring_consistency: float
    peak_efficiency_window: Tuple[int, int]  # Time window for best efficiency


class AdvancedScoringAnalyzer:
    def __init__(self, simulator: ScoringSimulator):
        self.simulator = simulator
        self.generator = ScenarioGenerator(simulator)
        self.constants = GameConstants()
        
        # Time phases (in seconds from match start)
        self.phase_boundaries = {
            GamePhase.AUTONOMOUS: (0, 15),
            GamePhase.EARLY_DRIVER: (15, 45), 
            GamePhase.MID_DRIVER: (45, 90),
            GamePhase.ENDGAME: (90, 120)
        }
    
    def analyze_time_value_optimization(
        self, 
        current_score: int = 0,
        opponent_score: int = 0,
        match_time: int = 60
    ) -> List[TimeValueAnalysis]:
        """Analyze when blocks are worth scoring vs defending"""
        
        analyses = []
        
        # Analyze each game phase
        for phase in GamePhase:
            phase_start, phase_end = self.phase_boundaries[phase]
            
            # Skip if current time is outside this phase
            if match_time < phase_start or match_time > phase_end:
                continue
            
            seconds_remaining = 120 - match_time
            
            # Calculate block value based on time and phase
            base_block_value = 3.0  # Base points per block
            
            # Time multipliers
            if phase == GamePhase.AUTONOMOUS:
                # Autonomous blocks are more valuable (can win bonus)
                block_value = base_block_value * 1.5
                # High opportunity cost if not maximizing auto
                opportunity_cost = 10.0 / max(1, seconds_remaining)
            elif phase == GamePhase.ENDGAME:
                # Endgame blocks less valuable if need to park
                block_value = base_block_value * 0.8
                opportunity_cost = 40.0 / max(1, seconds_remaining)  # Parking value
            else:
                # Driver control phases
                urgency_multiplier = 1 + (120 - seconds_remaining) / 120
                block_value = base_block_value * urgency_multiplier
                opportunity_cost = 5.0 / max(1, seconds_remaining / 30)
            
            # Calculate defense value
            score_difference = current_score - opponent_score
            if score_difference > 0:
                # Leading - defense more valuable
                defense_multiplier = 1 + min(score_difference / 50, 1.0)
            else:
                # Behind - offense more valuable
                defense_multiplier = max(0.5, 1 + score_difference / 100)
            
            defense_value = base_block_value * defense_multiplier
            
            # Determine if critical threshold
            critical_threshold = (
                seconds_remaining < 30 or  # Endgame approaching
                abs(score_difference) < 15 or  # Close game
                phase == GamePhase.AUTONOMOUS  # Auto period critical
            )
            
            # Recommend action
            if block_value > defense_value * 1.2:
                recommended_action = "Focus on scoring"
            elif defense_value > block_value * 1.2:
                recommended_action = "Focus on defense"
            else:
                recommended_action = "Balanced approach"
            
            analysis = TimeValueAnalysis(
                phase=phase,
                seconds_remaining=seconds_remaining,
                block_value=block_value,
                defense_value=defense_value,
                opportunity_cost=opportunity_cost,
                critical_threshold=critical_threshold,
                recommended_action=recommended_action
            )
            
            analyses.append(analysis)
        
        return analyses
    
    def create_goal_prioritization_algorithm(
        self,
        robot_position: str = "center",
        opponent_strategy: str = "balanced",
        time_remaining: int = 60
    ) -> List[GoalPrioritization]:
        """Algorithm to prioritize which goals to target"""
        
        goals = ["long_1", "long_2", "center_1", "center_2"]
        prioritizations = []
        
        # Base characteristics for each goal
        goal_characteristics = {
            "long_1": {"distance_factor": 1.2, "capacity": 25, "zone_value": 8},
            "long_2": {"distance_factor": 1.2, "capacity": 25, "zone_value": 8},
            "center_1": {"distance_factor": 0.8, "capacity": 20, "zone_value": 6},
            "center_2": {"distance_factor": 0.8, "capacity": 20, "zone_value": 6}
        }
        
        for goal in goals:
            chars = goal_characteristics[goal]
            
            # Calculate expected blocks per minute
            base_rate = 15  # Base blocks per minute
            distance_penalty = chars["distance_factor"]
            capacity_bonus = chars["capacity"] / 20  # Normalize to center goal capacity
            
            blocks_per_minute = base_rate / distance_penalty * capacity_bonus
            
            # Adjust for time remaining
            if time_remaining < 30:
                # Endgame - prefer closer goals
                blocks_per_minute *= (2.0 - distance_penalty)
            
            # Zone control value
            zone_control_value = chars["zone_value"]
            if opponent_strategy == "zone_control":
                zone_control_value *= 1.5  # More valuable against zone-focused opponents
            
            # Interference risk
            interference_risk = 0.3  # Base risk
            if goal in ["long_1", "long_2"]:
                interference_risk += 0.2  # Long goals more contested
            if opponent_strategy == "aggressive":
                interference_risk += 0.3
            
            # Calculate overall priority score
            priority_score = (
                blocks_per_minute * 0.4 +
                zone_control_value * 0.3 +
                (1 - interference_risk) * 20 * 0.3
            )
            
            # Recommended timing
            if time_remaining > 60:
                recommended_timing = "Early focus - establish control"
            elif time_remaining > 30:
                recommended_timing = "Mid-game target"
            else:
                recommended_timing = "Endgame priority if close"
            
            prioritization = GoalPrioritization(
                goal_name=goal,
                priority_score=priority_score,
                expected_blocks_per_minute=blocks_per_minute,
                zone_control_value=zone_control_value,
                interference_risk=interference_risk,
                recommended_timing=recommended_timing
            )
            
            prioritizations.append(prioritization)
        
        # Sort by priority score
        prioritizations.sort(key=lambda x: x.priority_score, reverse=True)
        
        return prioritizations
    
    def perform_breakeven_analysis(
        self,
        current_scores: Tuple[int, int],
        time_remaining: int,
        expected_scoring_rate: float = 0.3  # blocks per second
    ) -> BreakevenAnalysis:
        """Analyze how many blocks needed to secure victory"""
        
        red_score, blue_score = current_scores
        score_difference = red_score - blue_score
        
        # Estimate opponent's remaining scoring potential
        opponent_rate = expected_scoring_rate * 0.8  # Assume slightly lower rate
        opponent_potential = int(opponent_rate * time_remaining * 3)  # Convert to points
        
        # Account for endgame points
        endgame_bonus = 0
        if time_remaining <= 30:
            endgame_bonus = 40  # Potential parking points
            opponent_potential += 40  # Opponent can also park
        
        # Calculate blocks needed to win
        target_score = blue_score + opponent_potential + 1  # Beat opponent's potential + 1
        points_needed = max(0, target_score - red_score)
        blocks_needed = math.ceil(points_needed / 3)
        
        # Safety margin (additional blocks for confidence)
        safety_margin_blocks = max(3, int(blocks_needed * 0.3))
        
        # Required scoring rate
        total_blocks_needed = blocks_needed + safety_margin_blocks
        required_rate = total_blocks_needed * 60 / max(1, time_remaining)  # blocks per minute
        
        # Win probability calculation
        current_advantage = score_difference + endgame_bonus
        if current_advantage > 30:
            win_probability = 0.9
        elif current_advantage > 15:
            win_probability = 0.75
        elif current_advantage > 0:
            win_probability = 0.6
        elif current_advantage > -15:
            win_probability = 0.4
        elif current_advantage > -30:
            win_probability = 0.2
        else:
            win_probability = 0.1
        
        # Defensive threshold
        defensive_threshold = 20 + int(time_remaining / 10)
        
        analysis = BreakevenAnalysis(
            current_score=red_score,
            opponent_score=blue_score,
            time_remaining=time_remaining,
            blocks_needed_to_win=blocks_needed,
            safety_margin_blocks=safety_margin_blocks,
            required_scoring_rate=required_rate,
            win_probability=win_probability,
            defensive_threshold=defensive_threshold
        )
        
        return analysis
    
    def optimize_autonomous_period(
        self,
        robot_capabilities: Dict[str, float],
        starting_position: str = "center"
    ) -> AutonomousOptimization:
        """Optimize autonomous period strategy"""
        
        # Extract robot capabilities
        speed = robot_capabilities.get('speed', 1.0)
        accuracy = robot_capabilities.get('accuracy', 0.8)
        capacity = robot_capabilities.get('capacity', 3)
        
        # Calculate maximum theoretical score
        auto_time = 15  # seconds
        perfect_rate = 1.0  # blocks per second (theoretical max)
        max_blocks = min(int(auto_time * perfect_rate), capacity * 3)  # Account for trips
        max_theoretical_score = max_blocks * 3 + 10  # Include auto bonus
        
        # Calculate realistic score range
        realistic_rate = 0.3 * speed * accuracy  # More realistic rate
        realistic_blocks = int(auto_time * realistic_rate)
        realistic_min = max(realistic_blocks - 3, 0)
        realistic_max = min(realistic_blocks + 3, 20)
        realistic_score_range = (realistic_min * 3, realistic_max * 3 + 10)
        
        # Autonomous bonus probability
        if realistic_blocks >= 12:
            bonus_probability = 0.8
        elif realistic_blocks >= 8:
            bonus_probability = 0.6
        elif realistic_blocks >= 5:
            bonus_probability = 0.4
        else:
            bonus_probability = 0.2
        
        # Optimal starting position
        position_analysis = {
            "center": {"travel_time": 2, "goal_access": 4, "flexibility": 5},
            "alliance_side": {"travel_time": 1, "goal_access": 3, "flexibility": 3},
            "neutral_side": {"travel_time": 3, "goal_access": 4, "flexibility": 4}
        }
        
        best_position = max(position_analysis.keys(), 
                          key=lambda x: sum(position_analysis[x].values()))
        
        # Recommended block targets
        if realistic_blocks <= 8:
            # Conservative approach - focus on reliable targets
            targets = {"center_1": 4, "center_2": 4, "long_1": 0, "long_2": 0}
        elif realistic_blocks <= 15:
            # Balanced approach
            targets = {"center_1": 4, "center_2": 4, "long_1": 4, "long_2": 3}
        else:
            # Aggressive approach
            targets = {"center_1": 5, "center_2": 5, "long_1": 5, "long_2": 5}
        
        # Normalize targets to realistic blocks
        total_targets = sum(targets.values())
        if total_targets > realistic_blocks:
            scale_factor = realistic_blocks / total_targets
            targets = {k: int(v * scale_factor) for k, v in targets.items()}
        
        # Time allocation
        setup_time = 2.0
        scoring_time = auto_time - setup_time
        blocks_planned = sum(targets.values())
        
        time_allocation = {
            "setup_and_positioning": setup_time,
            "scoring_blocks": scoring_time * 0.8,
            "positioning_for_driver": scoring_time * 0.2
        }
        
        optimization = AutonomousOptimization(
            max_theoretical_score=max_theoretical_score,
            realistic_score_range=realistic_score_range,
            autonomous_bonus_probability=bonus_probability,
            optimal_starting_position=best_position,
            recommended_block_targets=targets,
            time_allocation=time_allocation
        )
        
        return optimization
    
    def calculate_efficiency_metrics(
        self,
        strategy: AllianceStrategy,
        num_simulations: int = 100
    ) -> EfficiencyMetrics:
        """Calculate detailed efficiency metrics for a strategy"""
        
        total_blocks = sum(strategy.blocks_scored_auto.values()) + sum(strategy.blocks_scored_driver.values())
        total_time = 120  # Match time in seconds
        
        # Points per second
        total_score, _ = self.simulator.calculate_score(
            {**strategy.blocks_scored_auto, **strategy.blocks_scored_driver},
            strategy.zones_controlled,
            strategy.robots_parked,
            True  # Assume auto win for calculation
        )
        
        points_per_second = total_score / total_time
        
        # Blocks per minute
        blocks_per_minute = (total_blocks * 60) / total_time
        
        # Efficiency vs defense simulation
        results_normal = []
        results_defended = []
        
        for _ in range(num_simulations):
            # Normal opponent
            normal_opponent = self.generator.generate_random_strategy("Normal", (25, 35))
            result_normal = self.simulator.simulate_match(strategy, normal_opponent)
            results_normal.append(result_normal.red_score)
            
            # Defensive opponent
            defensive_opponent = AllianceStrategy(
                name="Defensive",
                blocks_scored_auto={"long_1": 2, "long_2": 2, "center_1": 2, "center_2": 2},
                blocks_scored_driver={"long_1": 5, "long_2": 5, "center_1": 5, "center_2": 5},
                zones_controlled=[Zone.RED_HOME, Zone.BLUE_HOME, Zone.NEUTRAL],
                robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.PLATFORM]
            )
            result_defended = self.simulator.simulate_match(strategy, defensive_opponent)
            results_defended.append(result_defended.red_score)
        
        avg_normal = np.mean(results_normal)
        avg_defended = np.mean(results_defended)
        efficiency_vs_defense = avg_defended / avg_normal if avg_normal > 0 else 0
        
        # Scoring consistency
        scoring_consistency = 1 - (np.std(results_normal) / avg_normal) if avg_normal > 0 else 0
        
        # Peak efficiency window (when scoring rate is highest)
        # Assume early-mid driver control is peak efficiency
        peak_efficiency_window = (20, 80)
        
        metrics = EfficiencyMetrics(
            points_per_second=points_per_second,
            blocks_per_minute=blocks_per_minute,
            efficiency_vs_defense=efficiency_vs_defense,
            scoring_consistency=scoring_consistency,
            peak_efficiency_window=peak_efficiency_window
        )
        
        return metrics
    
    def generate_strategic_recommendations(
        self,
        current_situation: Dict,
        analysis_results: Dict
    ) -> List[str]:
        """Generate actionable strategic recommendations"""
        
        recommendations = []
        
        # Time value recommendations
        if 'time_value' in analysis_results:
            time_analyses = analysis_results['time_value']
            for analysis in time_analyses:
                if analysis.critical_threshold:
                    recommendations.append(
                        f"CRITICAL: {analysis.recommended_action} during {analysis.phase.value} "
                        f"(Block value: {analysis.block_value:.1f}, Defense value: {analysis.defense_value:.1f})"
                    )
        
        # Goal prioritization recommendations
        if 'goal_priority' in analysis_results:
            priorities = analysis_results['goal_priority']
            top_goal = priorities[0]
            recommendations.append(
                f"PRIMARY TARGET: {top_goal.goal_name} - Expected {top_goal.expected_blocks_per_minute:.1f} blocks/min"
            )
        
        # Breakeven recommendations
        if 'breakeven' in analysis_results:
            breakeven = analysis_results['breakeven']
            if breakeven.win_probability < 0.3:
                recommendations.append(
                    f"HIGH RISK: Need {breakeven.blocks_needed_to_win} blocks to win. "
                    f"Required rate: {breakeven.required_scoring_rate:.1f} blocks/min"
                )
            elif breakeven.win_probability > 0.7:
                recommendations.append(
                    f"STRONG POSITION: Consider defensive play if lead > {breakeven.defensive_threshold} points"
                )
        
        # Autonomous recommendations
        if 'autonomous' in analysis_results:
            auto_opt = analysis_results['autonomous']
            recommendations.append(
                f"AUTONOMOUS: Target {sum(auto_opt.recommended_block_targets.values())} blocks "
                f"from {auto_opt.optimal_starting_position} position "
                f"(Bonus probability: {auto_opt.autonomous_bonus_probability:.1%})"
            )
        
        # Efficiency recommendations
        if 'efficiency' in analysis_results:
            efficiency = analysis_results['efficiency']
            if efficiency.efficiency_vs_defense < 0.7:
                recommendations.append(
                    f"VULNERABILITY: {(1-efficiency.efficiency_vs_defense)*100:.0f}% efficiency loss under defense"
                )
            
            recommendations.append(
                f"PEAK WINDOW: Maximum efficiency between {efficiency.peak_efficiency_window[0]}s-{efficiency.peak_efficiency_window[1]}s"
            )
        
        return recommendations
    
    def run_comprehensive_analysis(
        self,
        strategy: AllianceStrategy,
        match_context: Dict = None
    ) -> Dict:
        """Run all advanced scoring analyses for a strategy"""
        
        if match_context is None:
            match_context = {
                'current_score': 80,
                'opponent_score': 75,
                'time_remaining': 45,
                'robot_position': 'center',
                'opponent_strategy': 'balanced'
            }
        
        print(f"Running comprehensive analysis for {strategy.name}...")
        
        results = {}
        
        # 1. Time-value optimization
        results['time_value'] = self.analyze_time_value_optimization(
            match_context['current_score'],
            match_context['opponent_score'],
            120 - match_context['time_remaining']
        )
        
        # 2. Goal prioritization
        results['goal_priority'] = self.create_goal_prioritization_algorithm(
            match_context.get('robot_position', 'center'),
            match_context.get('opponent_strategy', 'balanced'),
            match_context['time_remaining']
        )
        
        # 3. Breakeven analysis
        results['breakeven'] = self.perform_breakeven_analysis(
            (match_context['current_score'], match_context['opponent_score']),
            match_context['time_remaining']
        )
        
        # 4. Autonomous optimization
        robot_capabilities = {
            'speed': 1.0,
            'accuracy': 0.8,
            'capacity': 3
        }
        results['autonomous'] = self.optimize_autonomous_period(robot_capabilities)
        
        # 5. Efficiency metrics
        results['efficiency'] = self.calculate_efficiency_metrics(strategy, 50)
        
        # 6. Strategic recommendations
        results['recommendations'] = self.generate_strategic_recommendations(
            match_context, results
        )
        
        return results
    
    def create_detailed_statistics_report(self, analysis_results: Dict) -> str:
        """Create detailed statistics and recommendations report"""
        
        report = []
        report.append("=" * 80)
        report.append("ADVANCED VEX U PUSH BACK SCORING ANALYSIS")
        report.append("=" * 80)
        
        # Time Value Analysis
        report.append("\n📊 TIME-VALUE OPTIMIZATION ANALYSIS")
        report.append("-" * 50)
        
        if 'time_value' in analysis_results:
            for analysis in analysis_results['time_value']:
                report.append(f"\n{analysis.phase.value.upper()} PHASE:")
                report.append(f"  • Block Value: {analysis.block_value:.2f} points")
                report.append(f"  • Defense Value: {analysis.defense_value:.2f} points")
                report.append(f"  • Opportunity Cost: {analysis.opportunity_cost:.2f}")
                report.append(f"  • Critical Period: {'YES' if analysis.critical_threshold else 'NO'}")
                report.append(f"  • Recommendation: {analysis.recommended_action}")
        
        # Goal Prioritization
        report.append("\n\n🎯 GOAL PRIORITIZATION ALGORITHM")
        report.append("-" * 50)
        
        if 'goal_priority' in analysis_results:
            report.append("\nPriority Ranking:")
            for i, goal in enumerate(analysis_results['goal_priority'], 1):
                report.append(f"{i}. {goal.goal_name.upper()}")
                report.append(f"   Priority Score: {goal.priority_score:.1f}")
                report.append(f"   Expected Rate: {goal.expected_blocks_per_minute:.1f} blocks/min")
                report.append(f"   Zone Value: {goal.zone_control_value:.1f}")
                report.append(f"   Interference Risk: {goal.interference_risk:.1%}")
                report.append(f"   Timing: {goal.recommended_timing}")
                report.append("")
        
        # Breakeven Analysis
        report.append("\n⚖️  BREAKEVEN ANALYSIS")
        report.append("-" * 50)
        
        if 'breakeven' in analysis_results:
            breakeven = analysis_results['breakeven']
            report.append(f"\nCurrent Situation:")
            report.append(f"  • Your Score: {breakeven.current_score}")
            report.append(f"  • Opponent Score: {breakeven.opponent_score}")
            report.append(f"  • Score Difference: {breakeven.current_score - breakeven.opponent_score:+d}")
            report.append(f"\nVictory Requirements:")
            report.append(f"  • Blocks Needed to Win: {breakeven.blocks_needed_to_win}")
            report.append(f"  • Safety Margin: +{breakeven.safety_margin_blocks} blocks")
            report.append(f"  • Required Scoring Rate: {breakeven.required_scoring_rate:.1f} blocks/min")
            report.append(f"  • Current Win Probability: {breakeven.win_probability:.1%}")
            report.append(f"  • Defensive Threshold: {breakeven.defensive_threshold} point lead")
        
        # Autonomous Optimization
        report.append("\n\n🤖 AUTONOMOUS PERIOD OPTIMIZATION")
        report.append("-" * 50)
        
        if 'autonomous' in analysis_results:
            auto = analysis_results['autonomous']
            report.append(f"\nScoring Potential:")
            report.append(f"  • Max Theoretical Score: {auto.max_theoretical_score} points")
            report.append(f"  • Realistic Range: {auto.realistic_score_range[0]}-{auto.realistic_score_range[1]} points")
            report.append(f"  • Bonus Win Probability: {auto.autonomous_bonus_probability:.1%}")
            report.append(f"\nStrategy Recommendations:")
            report.append(f"  • Optimal Starting Position: {auto.optimal_starting_position}")
            report.append(f"  • Block Targets:")
            for goal, blocks in auto.recommended_block_targets.items():
                if blocks > 0:
                    report.append(f"    - {goal}: {blocks} blocks")
            report.append(f"  • Time Allocation:")
            for activity, time in auto.time_allocation.items():
                report.append(f"    - {activity}: {time:.1f}s")
        
        # Efficiency Metrics
        report.append("\n\n⚡ SCORING EFFICIENCY METRICS")
        report.append("-" * 50)
        
        if 'efficiency' in analysis_results:
            eff = analysis_results['efficiency']
            report.append(f"\nPerformance Metrics:")
            report.append(f"  • Points per Second: {eff.points_per_second:.2f}")
            report.append(f"  • Blocks per Minute: {eff.blocks_per_minute:.1f}")
            report.append(f"  • Efficiency vs Defense: {eff.efficiency_vs_defense:.1%}")
            report.append(f"  • Scoring Consistency: {eff.scoring_consistency:.1%}")
            report.append(f"  • Peak Efficiency Window: {eff.peak_efficiency_window[0]}s - {eff.peak_efficiency_window[1]}s")
        
        # Strategic Recommendations
        report.append("\n\n💡 STRATEGIC RECOMMENDATIONS")
        report.append("-" * 50)
        
        if 'recommendations' in analysis_results:
            for i, rec in enumerate(analysis_results['recommendations'], 1):
                report.append(f"\n{i}. {rec}")
        
        report.append("\n\n" + "=" * 80)
        report.append("END OF ANALYSIS")
        report.append("=" * 80)
        
        return "\n".join(report)


if __name__ == "__main__":
    from core.simulator import ScoringSimulator
    from analysis.strategy_analyzer import AdvancedStrategyAnalyzer
    
    # Initialize
    simulator = ScoringSimulator()
    analyzer = AdvancedScoringAnalyzer(simulator)
    
    # Test with a sample strategy
    test_strategy = AllianceStrategy(
        name="Test Strategy",
        blocks_scored_auto={"long_1": 6, "long_2": 6, "center_1": 4, "center_2": 4},
        blocks_scored_driver={"long_1": 10, "long_2": 10, "center_1": 8, "center_2": 8},
        zones_controlled=[Zone.RED_HOME, Zone.NEUTRAL],
        robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.PLATFORM]
    )
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis(test_strategy)
    
    # Generate detailed report
    report = analyzer.create_detailed_statistics_report(results)
    
    print(report)
    
    print("\n" + "="*60)
    print("QUICK SUMMARY")
    print("="*60)
    print(f"Strategy: {test_strategy.name}")
    print(f"Efficiency: {results['efficiency'].points_per_second:.2f} points/sec")
    print(f"Peak Window: {results['efficiency'].peak_efficiency_window[0]}-{results['efficiency'].peak_efficiency_window[1]}s")
    print(f"Top Goal: {results['goal_priority'][0].goal_name} ({results['goal_priority'][0].priority_score:.1f} priority)")
    print(f"Auto Target: {sum(results['autonomous'].recommended_block_targets.values())} blocks")
    print(f"Key Recommendations: {len(results['recommendations'])}")