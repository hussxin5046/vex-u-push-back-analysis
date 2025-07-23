"""
Push Back strategy insight generation and analysis.

This module provides advanced analysis capabilities for Push Back simulation results,
generating actionable strategic insights, recommendations, and predictions for
competitive team strategy development.
"""

import statistics
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any
from enum import Enum
import json
from collections import defaultdict, Counter

from .push_back_monte_carlo import (
    SimulationResult, StrategyInsights, RobotCapabilities,
    ParkingStrategy, GoalPriority, AutonomousStrategy
)

class InsightType(Enum):
    """Types of strategic insights"""
    PERFORMANCE = "performance"
    TIMING = "timing"
    STRATEGIC = "strategic"
    COMPETITIVE = "competitive"
    RISK = "risk"
    OPPORTUNITY = "opportunity"

class ConfidenceLevel(Enum):
    """Confidence levels for insights"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class StrategicInsight:
    """Individual strategic insight with detailed analysis"""
    insight_type: InsightType
    title: str
    description: str
    confidence: ConfidenceLevel
    impact_score: float  # 0.0 to 1.0
    supporting_data: Dict[str, Any]
    recommendations: List[str]
    implementation_difficulty: str  # "easy", "medium", "hard"
    time_to_implement: str  # "immediate", "short", "medium", "long"

@dataclass
class CompetitiveAnalysis:
    """Analysis comparing strategies against different opponents"""
    opponent_types: Dict[str, float]  # win rates against different archetypes
    critical_matchups: List[str]  # Difficult opponent types
    advantages: List[str]  # Strategic advantages
    vulnerabilities: List[str]  # Strategic weaknesses
    adaptation_strategies: Dict[str, str]  # How to adapt vs each opponent type

@dataclass
class PredictiveModel:
    """Predictive model for strategy performance"""
    base_win_probability: float
    score_distribution: Dict[str, float]  # score ranges and probabilities
    variance_factors: Dict[str, float]  # What causes high variance
    consistency_metrics: Dict[str, float]
    improvement_trajectory: List[float]  # Expected improvement over time

class PushBackInsightEngine:
    """
    Advanced insight generation engine for Push Back strategic analysis.
    
    Provides deep analysis of simulation results to generate actionable
    strategic recommendations for competitive teams.
    """
    
    def __init__(self):
        self.insight_thresholds = self._define_insight_thresholds()
        self.strategic_patterns = self._define_strategic_patterns()
        self.competitive_archetypes = self._define_competitive_archetypes()
    
    def _define_insight_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Define thresholds for generating insights"""
        return {
            "win_rate": {
                "excellent": 0.8,
                "good": 0.65,
                "average": 0.5,
                "poor": 0.35
            },
            "score_consistency": {
                "very_consistent": 100,
                "consistent": 200,
                "variable": 400,
                "highly_variable": 600
            },
            "margin_of_victory": {
                "dominant": 30,
                "comfortable": 15,
                "close": 5
            },
            "critical_timing": {
                "early_decision": 75,
                "mid_match": 45,
                "late_decision": 15
            }
        }
    
    def _define_strategic_patterns(self) -> Dict[str, Dict]:
        """Define patterns that indicate strategic insights"""
        return {
            "parking_timing": {
                "pattern": "wins correlate with specific parking times",
                "analysis": "parking_time_correlation",
                "threshold": 0.7
            },
            "autonomous_dependency": {
                "pattern": "wins heavily dependent on autonomous success",
                "analysis": "autonomous_win_correlation",
                "threshold": 0.6
            },
            "close_match_performance": {
                "pattern": "performance in close matches differs from average",
                "analysis": "close_match_win_rate",
                "threshold": 0.2
            },
            "consistency_under_pressure": {
                "pattern": "performance varies significantly under pressure",
                "analysis": "pressure_performance_variance",
                "threshold": 0.3
            }
        }
    
    def _define_competitive_archetypes(self) -> Dict[str, Dict]:
        """Define common competitive archetypes for analysis"""
        return {
            "speed_demon": {
                "characteristics": {
                    "average_cycle_time": (2.0, 4.0),
                    "pickup_reliability": (0.85, 0.95),
                    "parking_strategy": ParkingStrategy.NEVER
                },
                "strengths": ["High scoring potential", "Aggressive playstyle"],
                "weaknesses": ["Inconsistent under pressure", "Poor endgame"]
            },
            "consistency_king": {
                "characteristics": {
                    "pickup_reliability": (0.95, 1.0),
                    "scoring_reliability": (0.95, 1.0),
                    "parking_strategy": ParkingStrategy.LATE
                },
                "strengths": ["Reliable performance", "Good under pressure"],
                "weaknesses": ["Lower ceiling", "Vulnerable to speed"]
            },
            "control_master": {
                "characteristics": {
                    "control_zone_frequency": (0.5, 1.0),
                    "goal_priority": GoalPriority.CENTER_PREFERRED,
                    "parking_strategy": ParkingStrategy.EARLY
                },
                "strengths": ["Zone control", "Defensive capability"],
                "weaknesses": ["Lower raw scoring", "Vulnerable to speed"]
            },
            "autonomous_specialist": {
                "characteristics": {
                    "autonomous_reliability": (0.9, 1.0),
                    "autonomous_strategy": AutonomousStrategy.AGGRESSIVE
                },
                "strengths": ["Strong early game", "Consistent autonomous"],
                "weaknesses": ["Driver period weakness", "One-dimensional"]
            }
        }
    
    def generate_comprehensive_insights(self, results: List[SimulationResult], 
                                      robot: RobotCapabilities,
                                      alliance: str = "red") -> List[StrategicInsight]:
        """Generate comprehensive strategic insights from simulation results"""
        
        insights = []
        
        # Performance insights
        insights.extend(self._analyze_performance_patterns(results, robot, alliance))
        
        # Timing insights
        insights.extend(self._analyze_timing_patterns(results, alliance))
        
        # Strategic insights
        insights.extend(self._analyze_strategic_patterns(results, robot, alliance))
        
        # Risk insights
        insights.extend(self._analyze_risk_factors(results, robot, alliance))
        
        # Opportunity insights
        insights.extend(self._analyze_improvement_opportunities(results, robot, alliance))
        
        # Sort by impact score
        insights.sort(key=lambda x: x.impact_score, reverse=True)
        
        return insights
    
    def _analyze_performance_patterns(self, results: List[SimulationResult], 
                                    robot: RobotCapabilities, 
                                    alliance: str) -> List[StrategicInsight]:
        """Analyze performance patterns and generate insights"""
        insights = []
        
        # Extract alliance-specific data
        if alliance == "red":
            wins = sum(1 for r in results if r.winner == "red")
            scores = [r.final_score_red for r in results]
            blocks = [r.blocks_scored_red for r in results]
            parked = [r.red_parked for r in results]
        else:
            wins = sum(1 for r in results if r.winner == "blue")
            scores = [r.final_score_blue for r in results]
            blocks = [r.blocks_scored_blue for r in results]
            parked = [r.blue_parked for r in results]
        
        win_rate = wins / len(results)
        avg_score = statistics.mean(scores)
        score_variance = statistics.variance(scores) if len(scores) > 1 else 0
        
        # Win rate insight
        if win_rate >= self.insight_thresholds["win_rate"]["excellent"]:
            insights.append(StrategicInsight(
                insight_type=InsightType.PERFORMANCE,
                title="Excellent Win Rate Performance",
                description=f"Achieving {win_rate:.1%} win rate indicates a highly competitive strategy.",
                confidence=ConfidenceLevel.HIGH,
                impact_score=0.9,
                supporting_data={
                    "win_rate": win_rate,
                    "average_score": avg_score,
                    "total_matches": len(results)
                },
                recommendations=[
                    "Maintain current strategic approach",
                    "Focus on consistency to preserve win rate",
                    "Consider slight optimizations for score maximization"
                ],
                implementation_difficulty="easy",
                time_to_implement="immediate"
            ))
        elif win_rate <= self.insight_thresholds["win_rate"]["poor"]:
            insights.append(StrategicInsight(
                insight_type=InsightType.PERFORMANCE,
                title="Win Rate Needs Improvement",
                description=f"Current {win_rate:.1%} win rate suggests strategic adjustments needed.",
                confidence=ConfidenceLevel.HIGH,
                impact_score=0.95,
                supporting_data={
                    "win_rate": win_rate,
                    "performance_gap": self.insight_thresholds["win_rate"]["average"] - win_rate
                },
                recommendations=[
                    "Analyze losing patterns for systematic issues",
                    "Consider major strategic pivots",
                    "Focus on fundamental skill improvements"
                ],
                implementation_difficulty="hard",
                time_to_implement="long"
            ))
        
        # Consistency insight
        consistency_level = self._categorize_consistency(score_variance)
        if consistency_level == "highly_variable":
            insights.append(StrategicInsight(
                insight_type=InsightType.PERFORMANCE,
                title="High Score Variability Detected",
                description=f"Score variance of {score_variance:.0f} indicates inconsistent performance.",
                confidence=ConfidenceLevel.HIGH,
                impact_score=0.7,
                supporting_data={
                    "score_variance": score_variance,
                    "score_range": (min(scores), max(scores)),
                    "consistency_category": consistency_level
                },
                recommendations=[
                    "Focus on routine consistency in practice",
                    "Identify and eliminate high-variance actions",
                    "Develop backup strategies for reliability"
                ],
                implementation_difficulty="medium",
                time_to_implement="medium"
            ))
        
        return insights
    
    def _analyze_timing_patterns(self, results: List[SimulationResult], 
                               alliance: str) -> List[StrategicInsight]:
        """Analyze timing-related patterns"""
        insights = []
        
        # Parking timing analysis
        park_times = []
        park_wins = 0
        total_parks = 0
        
        for result in results:
            parked = result.red_parked if alliance == "red" else result.blue_parked
            won = result.winner == alliance
            
            if parked:
                total_parks += 1
                if won:
                    park_wins += 1
                
                # Find parking time from critical moments
                for moment in result.critical_moments:
                    if alliance in moment["event"] and "park" in moment["event"]:
                        park_times.append(moment["time"])
                        break
        
        if park_times and total_parks > 0:
            avg_park_time = statistics.mean(park_times)
            park_win_rate = park_wins / total_parks if total_parks > 0 else 0
            
            # Optimal parking time insight
            if avg_park_time > 20:
                insights.append(StrategicInsight(
                    insight_type=InsightType.TIMING,
                    title="Early Parking Strategy Detected",
                    description=f"Average parking at {avg_park_time:.1f}s with {park_win_rate:.1%} success rate.",
                    confidence=ConfidenceLevel.MEDIUM,
                    impact_score=0.6,
                    supporting_data={
                        "average_park_time": avg_park_time,
                        "park_win_rate": park_win_rate,
                        "sample_size": len(park_times)
                    },
                    recommendations=[
                        "Consider delaying parking for more scoring opportunities",
                        "Analyze if early parking is due to strategy or necessity",
                        "Test later parking timing in practice"
                    ],
                    implementation_difficulty="easy",
                    time_to_implement="short"
                ))
            elif avg_park_time < 10:
                insights.append(StrategicInsight(
                    insight_type=InsightType.TIMING,
                    title="Very Late Parking Strategy",
                    description=f"Average parking at {avg_park_time:.1f}s may be risky.",
                    confidence=ConfidenceLevel.MEDIUM,
                    impact_score=0.5,
                    supporting_data={
                        "average_park_time": avg_park_time,
                        "risk_level": "high"
                    },
                    recommendations=[
                        "Consider earlier parking for safety",
                        "Ensure parking routine is well-practiced",
                        "Have clear decision criteria for parking"
                    ],
                    implementation_difficulty="easy",
                    time_to_implement="short"
                ))
        
        return insights
    
    def _analyze_strategic_patterns(self, results: List[SimulationResult], 
                                  robot: RobotCapabilities,
                                  alliance: str) -> List[StrategicInsight]:
        """Analyze strategic decision patterns"""
        insights = []
        
        # Autonomous dependency analysis
        auto_wins = sum(1 for r in results if r.autonomous_winner == alliance)
        total_wins = sum(1 for r in results if r.winner == alliance)
        
        if total_wins > 0:
            auto_dependency = auto_wins / total_wins
            
            if auto_dependency > 0.7:
                insights.append(StrategicInsight(
                    insight_type=InsightType.STRATEGIC,
                    title="High Autonomous Dependency",
                    description=f"{auto_dependency:.1%} of wins depend on autonomous success.",
                    confidence=ConfidenceLevel.HIGH,
                    impact_score=0.8,
                    supporting_data={
                        "autonomous_dependency": auto_dependency,
                        "autonomous_wins": auto_wins,
                        "total_wins": total_wins
                    },
                    recommendations=[
                        "Strengthen driver period performance",
                        "Develop comeback strategies for lost autonomous",
                        "Practice autonomous consistency"
                    ],
                    implementation_difficulty="medium",
                    time_to_implement="medium"
                ))
        
        # Goal priority analysis
        if robot.goal_priority == GoalPriority.CENTER_ONLY:
            insights.append(StrategicInsight(
                insight_type=InsightType.STRATEGIC,
                title="Center Goal Specialization",
                description="Exclusive focus on center goals may limit scoring opportunities.",
                confidence=ConfidenceLevel.MEDIUM,
                impact_score=0.6,
                supporting_data={
                    "goal_strategy": "center_only",
                    "flexibility": "low"
                },
                recommendations=[
                    "Consider adding long goal capability",
                    "Ensure center goal efficiency is maximized",
                    "Develop contingency for blocked center goals"
                ],
                implementation_difficulty="hard",
                time_to_implement="long"
            ))
        
        return insights
    
    def _analyze_risk_factors(self, results: List[SimulationResult], 
                            robot: RobotCapabilities,
                            alliance: str) -> List[StrategicInsight]:
        """Analyze risk factors in current strategy"""
        insights = []
        
        # Close match performance
        close_matches = [r for r in results if r.score_margin <= 10]
        if close_matches:
            close_wins = sum(1 for r in close_matches if r.winner == alliance)
            close_win_rate = close_wins / len(close_matches)
            overall_win_rate = sum(1 for r in results if r.winner == alliance) / len(results)
            
            if close_win_rate < overall_win_rate - 0.2:
                insights.append(StrategicInsight(
                    insight_type=InsightType.RISK,
                    title="Poor Close Match Performance",
                    description=f"Win rate drops to {close_win_rate:.1%} in close matches vs {overall_win_rate:.1%} overall.",
                    confidence=ConfidenceLevel.HIGH,
                    impact_score=0.85,
                    supporting_data={
                        "close_match_win_rate": close_win_rate,
                        "overall_win_rate": overall_win_rate,
                        "close_match_count": len(close_matches)
                    },
                    recommendations=[
                        "Practice high-pressure situations",
                        "Develop clutch performance routines",
                        "Strengthen mental preparation"
                    ],
                    implementation_difficulty="medium",
                    time_to_implement="medium"
                ))
        
        # Reliability risk
        if robot.pickup_reliability < 0.9 or robot.scoring_reliability < 0.95:
            insights.append(StrategicInsight(
                insight_type=InsightType.RISK,
                title="Reliability Risk Detected",
                description="Low reliability may cause unexpected losses in important matches.",
                confidence=ConfidenceLevel.HIGH,
                impact_score=0.9,
                supporting_data={
                    "pickup_reliability": robot.pickup_reliability,
                    "scoring_reliability": robot.scoring_reliability
                },
                recommendations=[
                    "Focus on mechanical reliability improvements",
                    "Practice consistent routines",
                    "Develop backup procedures for failures"
                ],
                implementation_difficulty="hard",
                time_to_implement="long"
            ))
        
        return insights
    
    def _analyze_improvement_opportunities(self, results: List[SimulationResult], 
                                         robot: RobotCapabilities,
                                         alliance: str) -> List[StrategicInsight]:
        """Identify improvement opportunities"""
        insights = []
        
        # Cycle time optimization opportunity
        if robot.average_cycle_time > 6.0:
            potential_improvement = (robot.average_cycle_time - 4.0) / robot.average_cycle_time
            insights.append(StrategicInsight(
                insight_type=InsightType.OPPORTUNITY,
                title="Cycle Time Optimization Opportunity",
                description=f"Reducing cycle time could improve performance by ~{potential_improvement:.1%}.",
                confidence=ConfidenceLevel.MEDIUM,
                impact_score=0.7,
                supporting_data={
                    "current_cycle_time": robot.average_cycle_time,
                    "target_cycle_time": 4.0,
                    "potential_improvement": potential_improvement
                },
                recommendations=[
                    "Analyze current cycle for inefficiencies",
                    "Practice faster pickup and scoring routines",
                    "Consider mechanical optimizations"
                ],
                implementation_difficulty="medium",
                time_to_implement="medium"
            ))
        
        # Parking optimization
        park_rate = sum(1 for r in results 
                       if (alliance == "red" and r.red_parked) or 
                          (alliance == "blue" and r.blue_parked)) / len(results)
        
        if park_rate < 0.5 and robot.parking_strategy == ParkingStrategy.NEVER:
            insights.append(StrategicInsight(
                insight_type=InsightType.OPPORTUNITY,
                title="Parking Points Opportunity",
                description="Adding parking capability could secure additional points in close matches.",
                confidence=ConfidenceLevel.MEDIUM,
                impact_score=0.6,
                supporting_data={
                    "current_park_rate": park_rate,
                    "potential_points": 8  # Parking points
                },
                recommendations=[
                    "Develop simple parking routine",
                    "Practice parking under time pressure",
                    "Set clear parking decision criteria"
                ],
                implementation_difficulty="easy",
                time_to_implement="short"
            ))
        
        return insights
    
    def _categorize_consistency(self, variance: float) -> str:
        """Categorize performance consistency based on variance"""
        thresholds = self.insight_thresholds["score_consistency"]
        
        if variance <= thresholds["very_consistent"]:
            return "very_consistent"
        elif variance <= thresholds["consistent"]:
            return "consistent"
        elif variance <= thresholds["variable"]:
            return "variable"
        else:
            return "highly_variable"
    
    def generate_competitive_analysis(self, results_vs_opponents: Dict[str, List[SimulationResult]], 
                                    alliance: str = "red") -> CompetitiveAnalysis:
        """Generate competitive analysis against different opponent types"""
        
        opponent_win_rates = {}
        critical_matchups = []
        advantages = []
        vulnerabilities = []
        adaptation_strategies = {}
        
        for opponent_type, results in results_vs_opponents.items():
            wins = sum(1 for r in results if r.winner == alliance)
            win_rate = wins / len(results) if results else 0
            opponent_win_rates[opponent_type] = win_rate
            
            # Identify critical matchups (low win rates)
            if win_rate < 0.4:
                critical_matchups.append(opponent_type)
                # Generate adaptation strategy
                adaptation_strategies[opponent_type] = self._generate_adaptation_strategy(
                    opponent_type, results, alliance
                )
            
            # Identify advantages (high win rates)
            elif win_rate > 0.7:
                advantages.append(f"Strong vs {opponent_type}")
            
            # Identify vulnerabilities
            if win_rate < 0.5:
                vulnerabilities.append(f"Struggles vs {opponent_type}")
        
        return CompetitiveAnalysis(
            opponent_types=opponent_win_rates,
            critical_matchups=critical_matchups,
            advantages=advantages,
            vulnerabilities=vulnerabilities,
            adaptation_strategies=adaptation_strategies
        )
    
    def _generate_adaptation_strategy(self, opponent_type: str, 
                                    results: List[SimulationResult],
                                    alliance: str) -> str:
        """Generate adaptation strategy for difficult matchups"""
        
        # Analyze why this matchup is difficult
        losses = [r for r in results if r.winner != alliance]
        
        if not losses:
            return "Maintain current strategy"
        
        # Common loss patterns
        avg_margin = statistics.mean([r.score_margin for r in losses])
        auto_losses = sum(1 for r in losses if r.autonomous_winner and r.autonomous_winner != alliance)
        
        if auto_losses / len(losses) > 0.6:
            return "Focus on autonomous improvement and early scoring"
        elif avg_margin > 20:
            return "Need significant strategic changes - consider speed or efficiency improvements"
        else:
            return "Minor adjustments needed - focus on consistency and endgame"
    
    def generate_predictive_model(self, results: List[SimulationResult], 
                                alliance: str = "red") -> PredictiveModel:
        """Generate predictive model for future performance"""
        
        # Extract scores
        if alliance == "red":
            scores = [r.final_score_red for r in results]
            wins = sum(1 for r in results if r.winner == "red")
        else:
            scores = [r.final_score_blue for r in results]
            wins = sum(1 for r in results if r.winner == "blue")
        
        base_win_probability = wins / len(results)
        
        # Score distribution
        score_ranges = {
            "0-50": sum(1 for s in scores if s <= 50) / len(scores),
            "51-100": sum(1 for s in scores if 51 <= s <= 100) / len(scores),
            "101-150": sum(1 for s in scores if 101 <= s <= 150) / len(scores),
            "151-200": sum(1 for s in scores if 151 <= s <= 200) / len(scores),
            "200+": sum(1 for s in scores if s > 200) / len(scores)
        }
        
        # Variance factors
        variance_factors = {
            "autonomous_impact": self._calculate_autonomous_variance(results, alliance),
            "parking_impact": self._calculate_parking_variance(results, alliance),
            "close_match_factor": self._calculate_close_match_variance(results, alliance)
        }
        
        # Consistency metrics
        consistency_metrics = {
            "score_std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
            "win_consistency": self._calculate_win_consistency(results, alliance),
            "performance_floor": min(scores),
            "performance_ceiling": max(scores)
        }
        
        # Improvement trajectory (simplified)
        improvement_trajectory = [base_win_probability + i * 0.02 for i in range(10)]
        
        return PredictiveModel(
            base_win_probability=base_win_probability,
            score_distribution=score_ranges,
            variance_factors=variance_factors,
            consistency_metrics=consistency_metrics,
            improvement_trajectory=improvement_trajectory
        )
    
    def _calculate_autonomous_variance(self, results: List[SimulationResult], 
                                     alliance: str) -> float:
        """Calculate how much autonomous affects score variance"""
        auto_wins = [r for r in results if r.autonomous_winner == alliance]
        auto_losses = [r for r in results if r.autonomous_winner and r.autonomous_winner != alliance]
        no_auto = [r for r in results if not r.autonomous_winner]
        
        if alliance == "red":
            auto_win_scores = [r.final_score_red for r in auto_wins]
            auto_loss_scores = [r.final_score_red for r in auto_losses]
            no_auto_scores = [r.final_score_red for r in no_auto]
        else:
            auto_win_scores = [r.final_score_blue for r in auto_wins]
            auto_loss_scores = [r.final_score_blue for r in auto_losses]
            no_auto_scores = [r.final_score_blue for r in no_auto]
        
        # Calculate variance between different autonomous outcomes
        all_groups = [auto_win_scores, auto_loss_scores, no_auto_scores]
        group_means = [statistics.mean(group) if group else 0 for group in all_groups]
        
        if len([m for m in group_means if m > 0]) > 1:
            return statistics.stdev([m for m in group_means if m > 0])
        return 0.0
    
    def _calculate_parking_variance(self, results: List[SimulationResult], 
                                  alliance: str) -> float:
        """Calculate how much parking affects score variance"""
        if alliance == "red":
            parked_scores = [r.final_score_red for r in results if r.red_parked]
            not_parked_scores = [r.final_score_red for r in results if not r.red_parked]
        else:
            parked_scores = [r.final_score_blue for r in results if r.blue_parked]
            not_parked_scores = [r.final_score_blue for r in results if not r.blue_parked]
        
        if parked_scores and not_parked_scores:
            return abs(statistics.mean(parked_scores) - statistics.mean(not_parked_scores))
        return 0.0
    
    def _calculate_close_match_variance(self, results: List[SimulationResult], 
                                      alliance: str) -> float:
        """Calculate performance variance in close matches"""
        close_matches = [r for r in results if r.score_margin <= 10]
        other_matches = [r for r in results if r.score_margin > 10]
        
        if not close_matches or not other_matches:
            return 0.0
        
        close_wins = sum(1 for r in close_matches if r.winner == alliance) / len(close_matches)
        other_wins = sum(1 for r in other_matches if r.winner == alliance) / len(other_matches)
        
        return abs(close_wins - other_wins)
    
    def _calculate_win_consistency(self, results: List[SimulationResult], 
                                 alliance: str) -> float:
        """Calculate how consistent wins are across different scenarios"""
        # Group results by score ranges and calculate win rates
        if alliance == "red":
            scores = [r.final_score_red for r in results]
        else:
            scores = [r.final_score_blue for r in results]
        
        # Create score buckets
        score_buckets = {}
        for i, result in enumerate(results):
            score = scores[i]
            bucket = (score // 25) * 25  # 25-point buckets
            
            if bucket not in score_buckets:
                score_buckets[bucket] = []
            score_buckets[bucket].append(result)
        
        # Calculate win rate variance across buckets
        win_rates = []
        for bucket_results in score_buckets.values():
            if len(bucket_results) >= 3:  # Only consider buckets with sufficient data
                wins = sum(1 for r in bucket_results if r.winner == alliance)
                win_rates.append(wins / len(bucket_results))
        
        if len(win_rates) > 1:
            return 1.0 - statistics.stdev(win_rates)  # Higher is more consistent
        return 1.0

def format_insights_for_display(insights: List[StrategicInsight]) -> str:
    """Format insights for human-readable display"""
    
    if not insights:
        return "No significant insights generated."
    
    output = []
    output.append("=== PUSH BACK STRATEGIC INSIGHTS ===\n")
    
    # Group by insight type
    by_type = defaultdict(list)
    for insight in insights:
        by_type[insight.insight_type].append(insight)
    
    type_order = [InsightType.PERFORMANCE, InsightType.STRATEGIC, InsightType.TIMING, 
                  InsightType.OPPORTUNITY, InsightType.RISK]
    
    for insight_type in type_order:
        if insight_type in by_type:
            output.append(f"\n{insight_type.value.upper()} INSIGHTS:")
            output.append("-" * 40)
            
            for insight in sorted(by_type[insight_type], key=lambda x: x.impact_score, reverse=True):
                output.append(f"\n📊 {insight.title}")
                output.append(f"   {insight.description}")
                output.append(f"   Confidence: {insight.confidence.value.title()}")
                output.append(f"   Impact: {insight.impact_score:.1f}/1.0")
                
                if insight.recommendations:
                    output.append("   Recommendations:")
                    for rec in insight.recommendations[:3]:  # Top 3 recommendations
                        output.append(f"   • {rec}")
                
                output.append(f"   Implementation: {insight.implementation_difficulty} ({insight.time_to_implement})")
                output.append("")
    
    return "\n".join(output)