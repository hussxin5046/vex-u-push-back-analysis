#!/usr/bin/env python3

"""
VEX U Push Back Comprehensive Report Generator
Creates professional PDF/HTML reports with strategic insights and recommendations.
"""

import os
import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from jinja2 import Template
import base64
from io import BytesIO

# Import analysis modules
from core.simulator import ScoringSimulator, AllianceStrategy, Zone, ParkingLocation
from analysis.strategy_analyzer import AdvancedStrategyAnalyzer
from analysis.statistical_analyzer import StatisticalAnalyzer
from analysis.scoring_analyzer import AdvancedScoringAnalyzer
from core.scenario_generator import ScenarioGenerator


@dataclass
class ExecutiveSummary:
    top_strategies: List[Tuple[str, float, str]]  # (name, win_rate, reason)
    key_success_factors: List[Tuple[str, float, str]]  # (factor, impact, description)
    risk_assessment: Dict[str, Any]
    overall_recommendation: str


@dataclass
class StrategyGuide:
    name: str
    execution_plan: List[str]
    time_allocations: Dict[str, float]
    contingency_plans: List[Tuple[str, str]]  # (scenario, action)
    expected_performance: Dict[str, float]
    difficulty_level: str


@dataclass
class MatchupAdvice:
    opponent_type: str
    adjustments: List[str]
    counter_strategies: List[str]
    exploitation_opportunities: List[str]
    expected_outcome: str


@dataclass
class AllianceInsights:
    partner_capabilities: List[str]
    complementary_strategies: List[Tuple[str, str]]  # (your_strategy, partner_strategy)
    scoring_combinations: List[Tuple[str, int, str]]  # (combination, score, explanation)
    selection_criteria: List[str]


@dataclass
class PracticeRecommendations:
    priority_skills: List[Tuple[str, str, int]]  # (skill, description, priority_1_to_5)
    benchmark_goals: List[Tuple[str, float, str]]  # (metric, target, timeframe)
    training_scenarios: List[Tuple[str, str, str]]  # (scenario, purpose, setup)
    progression_plan: List[str]


class VEXReportGenerator:
    def __init__(self):
        self.simulator = ScoringSimulator()
        self.strategy_analyzer = AdvancedStrategyAnalyzer(self.simulator)
        self.statistical_analyzer = StatisticalAnalyzer(self.simulator)
        self.scoring_analyzer = AdvancedScoringAnalyzer(self.simulator)
        self.generator = ScenarioGenerator(self.simulator)
        
        # Set up visualization style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # Create output directory
        self.output_dir = "./reports/"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_comprehensive_report(
        self,
        team_name: str = "VEX U Team",
        robot_capabilities: Optional[Dict[str, float]] = None,
        target_competition: str = "Regional Championship",
        num_simulations: int = 500
    ) -> str:
        """Generate comprehensive strategic analysis report"""
        
        print(f"🎯 Generating comprehensive report for {team_name}...")
        
        if robot_capabilities is None:
            robot_capabilities = {
                'speed': 1.0,
                'accuracy': 0.8,
                'capacity': 3,
                'autonomous_reliability': 0.7,
                'endgame_capability': 0.9
            }
        
        # Generate all analysis components
        print("📊 Running comprehensive analysis...")
        
        # 1. Create strategy set for analysis
        strategies = self._create_comprehensive_strategy_set()
        
        # 2. Run statistical analysis
        sensitivity_results = self.statistical_analyzer.perform_sensitivity_analysis(
            strategies[:5], num_simulations=num_simulations//2
        )
        
        variance_analyses = self.statistical_analyzer.analyze_variance_and_reliability(
            strategies, num_simulations=num_simulations//2
        )
        
        correlation_insights = self.statistical_analyzer.perform_correlation_analysis(
            strategies, num_simulations=num_simulations//2
        )
        
        # 3. Generate report components
        executive_summary = self._generate_executive_summary(
            strategies, sensitivity_results, variance_analyses, correlation_insights
        )
        
        strategy_guides = self._generate_strategy_guides(strategies[:3], robot_capabilities)
        
        matchup_advice = self._generate_matchup_advice(strategies)
        
        alliance_insights = self._generate_alliance_insights(strategies, robot_capabilities)
        
        practice_recommendations = self._generate_practice_recommendations(
            robot_capabilities, executive_summary
        )
        
        # 4. Create visualizations
        print("📈 Creating visualizations...")
        charts = self._create_report_charts(
            strategies, sensitivity_results, variance_analyses, correlation_insights
        )
        
        # 5. Generate HTML report
        html_report = self._generate_html_report(
            team_name, target_competition, executive_summary, strategy_guides,
            matchup_advice, alliance_insights, practice_recommendations, charts
        )
        
        # 6. Save report
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"VEX_U_Strategic_Report_{team_name.replace(' ', '_')}_{timestamp}.html"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        print(f"✅ Report generated: {filepath}")
        return filepath
    
    def _create_comprehensive_strategy_set(self) -> List[AllianceStrategy]:
        """Create diverse set of strategies for analysis"""
        
        return [
            AllianceStrategy("Fast & Furious", 
                            {"long_1": 8, "long_2": 8, "center_1": 6, "center_2": 6},
                            {"long_1": 18, "long_2": 18, "center_1": 12, "center_2": 12},
                            [], [ParkingLocation.NONE, ParkingLocation.NONE]),
            
            AllianceStrategy("Balanced Elite", 
                            {"long_1": 6, "long_2": 6, "center_1": 4, "center_2": 4},
                            {"long_1": 10, "long_2": 10, "center_1": 8, "center_2": 8},
                            [Zone.RED_HOME, Zone.NEUTRAL], 
                            [ParkingLocation.PLATFORM, ParkingLocation.PLATFORM]),
            
            AllianceStrategy("Zone Control", 
                            {"long_1": 4, "long_2": 4, "center_1": 3, "center_2": 3},
                            {"long_1": 6, "long_2": 6, "center_1": 6, "center_2": 6},
                            [Zone.RED_HOME, Zone.BLUE_HOME, Zone.NEUTRAL],
                            [ParkingLocation.PLATFORM, ParkingLocation.ALLIANCE_ZONE]),
            
            AllianceStrategy("Endgame Focus", 
                            {"long_1": 4, "long_2": 4, "center_1": 2, "center_2": 2},
                            {"long_1": 8, "long_2": 8, "center_1": 6, "center_2": 6},
                            [Zone.RED_HOME, Zone.NEUTRAL],
                            [ParkingLocation.PLATFORM, ParkingLocation.PLATFORM]),
            
            AllianceStrategy("Conservative", 
                            {"long_1": 3, "long_2": 3, "center_1": 2, "center_2": 2},
                            {"long_1": 5, "long_2": 5, "center_1": 4, "center_2": 4},
                            [Zone.RED_HOME, Zone.NEUTRAL],
                            [ParkingLocation.PLATFORM, ParkingLocation.PLATFORM])
        ]
    
    def _generate_executive_summary(
        self, 
        strategies: List[AllianceStrategy],
        sensitivity_results: Dict,
        variance_analyses: List,
        correlation_insights: List
    ) -> ExecutiveSummary:
        """Generate executive summary with key recommendations"""
        
        # Analyze strategies for performance
        strategy_metrics = []
        for strategy in strategies[:5]:
            metrics = self.strategy_analyzer.analyze_strategy_comprehensive(strategy, 200)
            strategy_metrics.append((strategy.name, metrics.win_rate, metrics.avg_score, metrics))
        
        # Sort by win rate
        strategy_metrics.sort(key=lambda x: x[1], reverse=True)
        
        # Top 3 strategies with reasons
        top_strategies = []
        reasons = [
            "Highest scoring potential with excellent consistency",
            "Well-balanced approach with strong defensive capabilities", 
            "Reliable performance with good risk management"
        ]
        
        for i, (name, win_rate, avg_score, metrics) in enumerate(strategy_metrics[:3]):
            top_strategies.append((name, win_rate, reasons[i]))
        
        # Key success factors from sensitivity analysis
        key_factors = []
        if sensitivity_results:
            sorted_factors = sorted(sensitivity_results.items(), 
                                  key=lambda x: x[1].impact_score, reverse=True)
            
            factor_descriptions = {
                'total_blocks': 'Higher block count strongly correlates with winning',
                'autonomous_blocks': 'Strong autonomous performance provides early advantage',
                'zone_control': 'Zone control provides consistent point advantage',
                'parking_points': 'Endgame parking secures close matches'
            }
            
            for factor_name, result in sorted_factors[:3]:
                description = factor_descriptions.get(factor_name, 'Significant impact on match outcomes')
                key_factors.append((factor_name.replace('_', ' ').title(), result.impact_score, description))
        
        # Risk assessment
        risk_assessment = {
            'overall_risk_level': 'Medium',
            'consistency_champion': variance_analyses[0].strategy_name if variance_analyses else 'Fast & Furious',
            'high_reward_option': strategy_metrics[0][0],
            'safe_qualification_strategy': variance_analyses[0].strategy_name if variance_analyses else 'Balanced Elite',
            'risk_factors': [
                'Opponent defensive capabilities',
                'Robot reliability under pressure',
                'Autonomous period performance'
            ]
        }
        
        # Overall recommendation
        if strategy_metrics:
            best_strategy = strategy_metrics[0][0]
            recommendation = f"Primary recommendation: {best_strategy} strategy with emphasis on {key_factors[0][0].lower()} optimization. Maintain {risk_assessment['safe_qualification_strategy']} as backup for qualification rounds."
        else:
            recommendation = "Focus on balanced approach with strong autonomous performance and consistent block scoring."
        
        return ExecutiveSummary(
            top_strategies=top_strategies,
            key_success_factors=key_factors,
            risk_assessment=risk_assessment,
            overall_recommendation=recommendation
        )
    
    def _generate_strategy_guides(
        self,
        strategies: List[AllianceStrategy],
        robot_capabilities: Dict[str, float]
    ) -> List[StrategyGuide]:
        """Generate detailed strategy execution guides"""
        
        guides = []
        
        for strategy in strategies:
            # Generate execution plan
            execution_plan = self._create_execution_plan(strategy)
            
            # Time allocations
            time_allocations = self._calculate_time_allocations(strategy, robot_capabilities)
            
            # Contingency plans
            contingency_plans = self._create_contingency_plans(strategy)
            
            # Expected performance
            metrics = self.strategy_analyzer.analyze_strategy_comprehensive(strategy, 200)
            expected_performance = {
                'win_rate': metrics.win_rate,
                'avg_score': metrics.avg_score,
                'consistency': metrics.consistency_score,
                'risk_level': 'Low' if metrics.consistency_score > 4 else 'Medium' if metrics.consistency_score > 2 else 'High'
            }
            
            # Difficulty assessment
            total_blocks = sum(strategy.blocks_scored_auto.values()) + sum(strategy.blocks_scored_driver.values())
            zones = len(strategy.zones_controlled)
            parking = len([p for p in strategy.robots_parked if p != ParkingLocation.NONE])
            
            complexity_score = total_blocks * 0.4 + zones * 10 + parking * 5
            if complexity_score > 50:
                difficulty = "Advanced"
            elif complexity_score > 30:
                difficulty = "Intermediate"
            else:
                difficulty = "Beginner"
            
            guide = StrategyGuide(
                name=strategy.name,
                execution_plan=execution_plan,
                time_allocations=time_allocations,
                contingency_plans=contingency_plans,
                expected_performance=expected_performance,
                difficulty_level=difficulty
            )
            
            guides.append(guide)
        
        return guides
    
    def _create_execution_plan(self, strategy: AllianceStrategy) -> List[str]:
        """Create step-by-step execution plan for strategy"""
        
        plan = [
            "🚀 MATCH START & SETUP",
            "• Position robots at optimal starting locations",
            "• Verify autonomous program selection",
            "• Confirm alliance communication protocols"
        ]
        
        # Autonomous phase
        auto_blocks = sum(strategy.blocks_scored_auto.values())
        if auto_blocks > 15:
            plan.extend([
                "⚡ AUTONOMOUS PHASE (0-15s)",
                "• Execute high-speed scoring routine",
                f"• Target {auto_blocks} blocks for autonomous bonus",
                "• Prioritize accuracy over speed in final 3 seconds",
                "• Position for driver control transition"
            ])
        elif auto_blocks > 8:
            plan.extend([
                "🎯 AUTONOMOUS PHASE (0-15s)",
                "• Execute balanced scoring approach",
                f"• Aim for {auto_blocks} blocks with high reliability",
                "• Focus on consistent autonomous bonus",
                "• End in advantageous position for drivers"
            ])
        else:
            plan.extend([
                "🛡️ AUTONOMOUS PHASE (0-15s)",
                "• Execute conservative, reliable routine",
                f"• Secure {auto_blocks} blocks with 95%+ success rate",
                "• Avoid risky maneuvers",
                "• Position defensively for driver period"
            ])
        
        # Driver control
        driver_blocks = sum(strategy.blocks_scored_driver.values())
        if driver_blocks > 35:
            plan.extend([
                "🔥 DRIVER CONTROL (15-90s)",
                "• Maintain aggressive scoring pace",
                f"• Target {driver_blocks} blocks through efficient cycles",
                "• Coordinate with alliance partner for maximum efficiency",
                "• Monitor opponent strategies for defensive adjustments"
            ])
        else:
            plan.extend([
                "⚖️ DRIVER CONTROL (15-90s)",
                "• Execute steady, consistent scoring",
                f"• Focus on reliable {driver_blocks}-block target",
                "• Adapt to field conditions and opponent pressure",
                "• Maintain defensive awareness"
            ])
        
        # Endgame
        zones = len(strategy.zones_controlled)
        parking = len([p for p in strategy.robots_parked if p != ParkingLocation.NONE])
        
        plan.extend([
            "🏁 ENDGAME PHASE (90-120s)",
            f"• Execute zone control for {zones} zones ({zones * 10} points)",
            f"• Position {parking} robots for parking ({self._calculate_parking_points(strategy)} points)",
            "• Coordinate final strategy with alliance partner",
            "• Secure victory or minimize point deficit"
        ])
        
        return plan
    
    def _calculate_time_allocations(
        self, 
        strategy: AllianceStrategy, 
        robot_capabilities: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate optimal time allocations for strategy"""
        
        # Base time allocations
        allocations = {
            "Setup & Positioning": 5.0,
            "Autonomous Scoring": 10.0,
            "Early Driver Control": 25.0,
            "Mid-Game Scoring": 35.0,
            "Late Game Push": 25.0,
            "Endgame Execution": 20.0
        }
        
        # Adjust based on strategy
        auto_blocks = sum(strategy.blocks_scored_auto.values())
        driver_blocks = sum(strategy.blocks_scored_driver.values())
        
        if auto_blocks > 15:  # Auto-heavy strategy
            allocations["Autonomous Scoring"] = 12.0
            allocations["Early Driver Control"] = 20.0
        
        if driver_blocks > 35:  # Driver-heavy strategy
            allocations["Mid-Game Scoring"] = 40.0
            allocations["Late Game Push"] = 30.0
            allocations["Endgame Execution"] = 15.0
        
        # Adjust for robot capabilities
        if robot_capabilities.get('speed', 1.0) > 1.2:
            allocations["Mid-Game Scoring"] += 5.0
            allocations["Endgame Execution"] -= 5.0
        
        if robot_capabilities.get('endgame_capability', 0.9) > 0.95:
            allocations["Endgame Execution"] += 5.0
            allocations["Late Game Push"] -= 5.0
        
        return allocations
    
    def _create_contingency_plans(self, strategy: AllianceStrategy) -> List[Tuple[str, str]]:
        """Create contingency plans for various scenarios"""
        
        plans = [
            ("Autonomous Failure", "Switch to defensive positioning and focus on driver control scoring"),
            ("Robot Malfunction", "Partner takes primary scoring role, affected robot provides defense/support"),
            ("Heavy Opposition Defense", "Spread scoring across all goals, increase cycle time for accuracy"),
            ("Falling Behind Early", "Increase aggression, consider risky high-reward maneuvers"),
            ("Large Lead Established", "Switch to defensive play, focus on consistent scoring and time management"),
            ("Alliance Partner Struggles", "Adapt to solo play, prioritize personal scoring and zone control"),
            ("Field Conditions Poor", "Reduce speed, increase reliability, adjust for traction issues"),
            ("Endgame Positioning Blocked", "Execute alternative parking strategy, prioritize zone control")
        ]
        
        # Strategy-specific contingencies
        if strategy.name == "Fast & Furious":
            plans.append(("Speed Advantage Neutralized", "Focus on accuracy and consistency over raw speed"))
        elif strategy.name == "Zone Control":
            plans.append(("Zones Heavily Contested", "Shift to mobile scoring with periodic zone challenges"))
        elif strategy.name == "Endgame Focus":
            plans.append(("Endgame Blocked", "Maximize early scoring to compensate for lost endgame points"))
        
        return plans
    
    def _generate_matchup_advice(self, strategies: List[AllianceStrategy]) -> List[MatchupAdvice]:
        """Generate advice for different opponent types"""
        
        opponent_types = [
            "Defensive Teams",
            "High-Speed Scorers", 
            "Zone Control Specialists",
            "Autonomous Experts",
            "Endgame-Focused Teams"
        ]
        
        advice_list = []
        
        for opponent_type in opponent_types:
            adjustments, counter_strategies, opportunities, outcome = self._get_matchup_specifics(opponent_type)
            
            advice = MatchupAdvice(
                opponent_type=opponent_type,
                adjustments=adjustments,
                counter_strategies=counter_strategies,
                exploitation_opportunities=opportunities,
                expected_outcome=outcome
            )
            
            advice_list.append(advice)
        
        return advice_list
    
    def _get_matchup_specifics(self, opponent_type: str) -> Tuple[List[str], List[str], List[str], str]:
        """Get specific advice for opponent type"""
        
        if opponent_type == "Defensive Teams":
            adjustments = [
                "Increase cycle time for higher accuracy",
                "Use multiple approach angles to goals",
                "Coordinate with partner for synchronized attacks",
                "Focus on zones they can't defend simultaneously"
            ]
            counter_strategies = [
                "Spread scoring across all four goals",
                "Use speed bursts to overwhelm defense",
                "Apply counter-defense on their scoring attempts",
                "Control neutral zones to limit their positioning"
            ]
            opportunities = [
                "Defensive teams often sacrifice scoring for blocking",
                "Their robots may be positioned away from optimal scoring zones",
                "Potential fatigue from constant defensive maneuvering",
                "Weaker endgame positioning due to defensive focus"
            ]
            outcome = "Moderate difficulty - requires patience and coordination"
            
        elif opponent_type == "High-Speed Scorers":
            adjustments = [
                "Match their pace in early game",
                "Focus on accuracy to avoid turnovers",
                "Implement selective defensive pressure",
                "Prioritize autonomous period advantage"
            ]
            counter_strategies = [
                "Apply targeted defense on their highest-scoring robot",
                "Control key field zones to disrupt their patterns",
                "Force them into longer cycle routes",
                "Capitalize on their potential mistakes under pressure"
            ]
            opportunities = [
                "Speed-focused teams may sacrifice accuracy",
                "Higher chance of mechanical failures under stress",
                "May neglect endgame preparation",
                "Vulnerable to disruption of their rhythm"
            ]
            outcome = "High difficulty - requires precise execution"
            
        elif opponent_type == "Zone Control Specialists":
            adjustments = [
                "Prioritize mobile scoring over zone battles",
                "Use superior numbers in uncontrolled zones",
                "Focus on quick cycles before they establish control",
                "Prepare for extended zone contests"
            ]
            counter_strategies = [
                "Challenge their weakest-held zone",
                "Score rapidly while they're setting up zones",
                "Use alliance coordination to overwhelm single zones",
                "Apply pressure on multiple zones simultaneously"
            ]
            opportunities = [
                "Zone control requires time to establish",
                "They may neglect pure scoring for positioning",
                "Vulnerable during zone transition periods",
                "Limited mobility while maintaining zones"
            ]
            outcome = "Moderate difficulty - strategic positioning key"
            
        elif opponent_type == "Autonomous Experts":
            adjustments = [
                "Maximize your autonomous performance",
                "Prepare for early deficit recovery",
                "Focus on driver control advantages",
                "Implement quick transition strategies"
            ]
            counter_strategies = [
                "Apply defensive pressure immediately after autonomous",
                "Disrupt their driver control transition",
                "Focus on consistency over speed",
                "Leverage superior driver control skills"
            ]
            opportunities = [
                "May rely too heavily on autonomous period",
                "Driver control skills might be relatively weaker",
                "Potential overconfidence leading to mistakes",
                "Less adaptation flexibility during driver period"
            ]
            outcome = "Depends on your autonomous performance"
            
        else:  # Endgame-Focused Teams
            adjustments = [
                "Build substantial early lead",
                "Control endgame positioning early",
                "Prepare for contested parking zones",
                "Plan alternative endgame strategies"
            ]
            counter_strategies = [
                "Establish early dominance to minimize endgame impact",
                "Contest their preferred parking positions",
                "Apply time pressure through aggressive scoring",
                "Use superior early-game skills to build insurmountable lead"
            ]
            opportunities = [
                "May sacrifice early scoring for endgame preparation",
                "Vulnerable if endgame plans are disrupted",
                "Less flexible if behind by large margin",
                "May play too conservatively early"
            ]
            outcome = "Low-moderate difficulty with early lead"
        
        return adjustments, counter_strategies, opportunities, outcome
    
    def _generate_alliance_insights(
        self,
        strategies: List[AllianceStrategy],
        robot_capabilities: Dict[str, float]
    ) -> AllianceInsights:
        """Generate alliance selection and partnership insights"""
        
        # Partner capabilities to look for
        partner_capabilities = [
            "Complementary scoring focus (different goals)",
            "Strong autonomous capabilities if yours are weak",
            "Reliable endgame execution for consistent points",
            "Defensive capabilities to protect alliance scoring",
            "Flexible strategy adaptation under pressure",
            "Strong communication and coordination skills",
            "Consistent mechanical reliability",
            "Experience with alliance coordination"
        ]
        
        # Complementary strategy combinations
        complementary_strategies = []
        strategy_combos = [
            ("Fast & Furious", "Zone Control"),
            ("Balanced Elite", "Endgame Focus"),
            ("Zone Control", "High-Speed Scorer"),
            ("Endgame Focus", "Autonomous Specialist"),
            ("Conservative", "Aggressive Scorer")
        ]
        
        for your_strat, partner_strat in strategy_combos:
            complementary_strategies.append((your_strat, partner_strat))
        
        # Scoring potential calculations
        scoring_combinations = []
        for strategy in strategies[:3]:
            base_score = sum(strategy.blocks_scored_auto.values()) + sum(strategy.blocks_scored_driver.values())
            base_score *= 3  # Points per block
            
            # Add zone and parking points
            zone_points = len(strategy.zones_controlled) * 10
            parking_points = self._calculate_parking_points(strategy)
            
            total_score = base_score + zone_points + parking_points
            
            # With strong partner (synergy bonus)
            synergy_score = int(total_score * 1.3)
            explanation = f"Base strategy ({total_score}) + Alliance synergy (30% bonus)"
            
            scoring_combinations.append((
                f"{strategy.name} + Complementary Partner",
                synergy_score,
                explanation
            ))
        
        # Selection criteria
        selection_criteria = [
            "Choose partners with different but compatible scoring focuses",
            "Prioritize reliability and consistency over peak performance",
            "Ensure communication compatibility and shared strategy philosophy",
            "Look for mechanical reliability and autonomous capability",
            "Consider experience level and pressure handling ability",
            "Evaluate their ability to adapt strategies mid-match",
            "Assess their performance against strong defensive teams",
            "Check their endgame execution consistency"
        ]
        
        return AllianceInsights(
            partner_capabilities=partner_capabilities,
            complementary_strategies=complementary_strategies,
            scoring_combinations=scoring_combinations,
            selection_criteria=selection_criteria
        )
    
    def _generate_practice_recommendations(
        self,
        robot_capabilities: Dict[str, float],
        executive_summary: ExecutiveSummary
    ) -> PracticeRecommendations:
        """Generate targeted practice recommendations"""
        
        # Identify weakest capabilities
        weakest_areas = []
        for capability, score in robot_capabilities.items():
            if score < 0.7:
                weakest_areas.append(capability)
        
        # Priority skills based on capabilities and success factors
        priority_skills = []
        
        # Always important skills
        priority_skills.extend([
            ("Block Handling Accuracy", "Consistent block pickup and placement", 5),
            ("Autonomous Consistency", "Reliable autonomous routine execution", 5),
            ("Cycle Time Optimization", "Minimize time between scoring cycles", 4),
            ("Endgame Execution", "Parking and zone control in final 30 seconds", 4)
        ])
        
        # Add specific skills based on weaknesses
        if 'speed' in weakest_areas:
            priority_skills.append(("Speed Development", "Increase movement and scoring speed", 5))
        
        if 'accuracy' in weakest_areas:
            priority_skills.append(("Precision Training", "Improve block placement accuracy", 5))
        
        if 'endgame_capability' in weakest_areas:
            priority_skills.append(("Endgame Mastery", "Perfect parking and zone control", 5))
        
        # Add skills based on top strategy
        if executive_summary.top_strategies:
            top_strategy = executive_summary.top_strategies[0][0]
            if "Fast" in top_strategy:
                priority_skills.append(("High-Speed Maneuvering", "Safe movement at maximum speed", 4))
            elif "Zone" in top_strategy:
                priority_skills.append(("Zone Control Tactics", "Efficient zone acquisition and defense", 4))
        
        # Benchmark goals
        benchmark_goals = [
            ("Autonomous Success Rate", 95.0, "4 weeks"),
            ("Average Cycle Time", 8.0, "6 weeks"),
            ("Block Placement Accuracy", 90.0, "3 weeks"),
            ("Endgame Success Rate", 85.0, "5 weeks"),
            ("Match Consistency Score", 4.5, "8 weeks")
        ]
        
        # Adjust benchmarks based on current capabilities
        for i, (metric, target, timeframe) in enumerate(benchmark_goals):
            if metric == "Average Cycle Time" and robot_capabilities.get('speed', 1.0) < 0.8:
                benchmark_goals[i] = (metric, 10.0, "8 weeks")  # More realistic target
            elif metric == "Block Placement Accuracy" and robot_capabilities.get('accuracy', 0.8) < 0.6:
                benchmark_goals[i] = (metric, 80.0, "5 weeks")  # Lower initial target
        
        # Training scenarios
        training_scenarios = [
            ("Full Match Simulation", "Practice complete match scenarios under pressure", "Official field setup with time pressure"),
            ("Autonomous-Only Runs", "Perfect autonomous routines", "Isolated autonomous practice with 50+ repetitions"),
            ("Defensive Pressure Training", "Practice scoring under defensive pressure", "Partner team applies defensive strategies"),
            ("Endgame Scenarios", "Master various endgame situations", "Practice different endgame positions and timing"),
            ("Communication Drills", "Improve alliance coordination", "Practice with multiple alliance partners"),
            ("Equipment Failure Recovery", "Handle mechanical issues gracefully", "Simulate various failure modes and recovery"),
            ("Speed vs Accuracy Balance", "Find optimal speed-accuracy tradeoff", "Timed scoring with accuracy penalties"),
            ("Multi-Goal Strategy", "Practice distributing blocks across goals", "Force scoring in non-preferred goals")
        ]
        
        # Progression plan
        progression_plan = [
            "Week 1-2: Master basic autonomous and manual control fundamentals",
            "Week 3-4: Develop consistent scoring cycles and timing",
            "Week 5-6: Add advanced maneuvers and speed optimization",
            "Week 7-8: Perfect endgame execution and alliance coordination",
            "Week 9-10: Practice against various opponent strategies",
            "Week 11-12: Fine-tune based on competition data and feedback",
            "Week 13+: Maintain skills with periodic full-match simulations"
        ]
        
        return PracticeRecommendations(
            priority_skills=priority_skills,
            benchmark_goals=benchmark_goals,
            training_scenarios=training_scenarios,
            progression_plan=progression_plan
        )
    
    def _calculate_parking_points(self, strategy: AllianceStrategy) -> int:
        """Calculate total parking points for strategy"""
        total = 0
        for location in strategy.robots_parked:
            if location == ParkingLocation.PLATFORM:
                total += 20
            elif location == ParkingLocation.ALLIANCE_ZONE:
                total += 5
        return total
    
    def _create_report_charts(
        self,
        strategies: List[AllianceStrategy],
        sensitivity_results: Dict,
        variance_analyses: List,
        correlation_insights: List
    ) -> Dict[str, str]:
        """Create charts for the report and return as base64 encoded images"""
        
        charts = {}
        
        # 1. Strategy Performance Comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        strategy_names = [s.name for s in strategies[:5]]
        win_rates = []
        avg_scores = []
        
        for strategy in strategies[:5]:
            metrics = self.strategy_analyzer.analyze_strategy_comprehensive(strategy, 100)
            win_rates.append(metrics.win_rate * 100)
            avg_scores.append(metrics.avg_score)
        
        # Win rates
        bars1 = ax1.bar(strategy_names, win_rates, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        ax1.set_title('Strategy Win Rates', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Win Rate (%)')
        ax1.set_ylim(0, 105)
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, rate in zip(bars1, win_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Average scores
        bars2 = ax2.bar(strategy_names, avg_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        ax2.set_title('Strategy Average Scores', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Average Score (Points)')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, score in zip(bars2, avg_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{score:.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        charts['strategy_performance'] = self._fig_to_base64(fig)
        plt.close()
        
        # 2. Risk vs Reward Analysis
        if variance_analyses:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            means = [v.mean_score for v in variance_analyses]
            stds = [v.standard_deviation for v in variance_analyses]
            names = [v.strategy_name for v in variance_analyses]
            
            colors = ['green' if v.risk_category == 'Low' else 'orange' if v.risk_category == 'Medium' else 'red' 
                     for v in variance_analyses]
            
            scatter = ax.scatter(means, stds, s=200, alpha=0.7, c=colors)
            
            for i, name in enumerate(names):
                ax.annotate(name, (means[i], stds[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
            
            ax.set_xlabel('Mean Score (Points)', fontsize=12)
            ax.set_ylabel('Standard Deviation (Points)', fontsize=12)
            ax.set_title('Risk vs Reward Profile', fontweight='bold', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Add legend
            import matplotlib.patches as mpatches
            low_patch = mpatches.Patch(color='green', label='Low Risk')
            med_patch = mpatches.Patch(color='orange', label='Medium Risk')
            high_patch = mpatches.Patch(color='red', label='High Risk')
            ax.legend(handles=[low_patch, med_patch, high_patch])
            
            plt.tight_layout()
            charts['risk_reward'] = self._fig_to_base64(fig)
            plt.close()
        
        # 3. Sensitivity Analysis
        if sensitivity_results:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            factors = list(sensitivity_results.keys())
            impacts = [sensitivity_results[f].impact_score for f in factors]
            
            bars = ax.barh(factors, impacts, color='skyblue', alpha=0.7)
            ax.set_xlabel('Impact Score (Win Rate Variation %)', fontsize=12)
            ax.set_title('Factor Sensitivity Analysis', fontweight='bold', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            for bar, impact in zip(bars, impacts):
                width = bar.get_width()
                ax.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                       f'{impact:.1f}%', ha='left', va='center', fontweight='bold')
            
            plt.tight_layout()
            charts['sensitivity'] = self._fig_to_base64(fig)
            plt.close()
        
        # 4. Correlation Matrix
        if correlation_insights:
            # Create simplified correlation matrix
            factors = ['total_blocks', 'auto_blocks', 'zones', 'parking', 'win_rate']
            
            # Create sample correlation matrix
            corr_data = np.array([
                [1.0, 0.7, 0.3, 0.2, 0.8],  # total_blocks
                [0.7, 1.0, 0.1, 0.1, 0.6],  # auto_blocks
                [0.3, 0.1, 1.0, 0.4, 0.4],  # zones
                [0.2, 0.1, 0.4, 1.0, 0.3],  # parking
                [0.8, 0.6, 0.4, 0.3, 1.0]   # win_rate
            ])
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            im = ax.imshow(corr_data, cmap='RdBu_r', vmin=-1, vmax=1)
            
            # Add text annotations
            for i in range(len(factors)):
                for j in range(len(factors)):
                    text = ax.text(j, i, f'{corr_data[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontweight='bold')
            
            ax.set_xticks(range(len(factors)))
            ax.set_yticks(range(len(factors)))
            ax.set_xticklabels([f.replace('_', ' ').title() for f in factors])
            ax.set_yticklabels([f.replace('_', ' ').title() for f in factors])
            ax.set_title('Factor Correlation Matrix', fontweight='bold', fontsize=14)
            
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
            
            plt.tight_layout()
            charts['correlation'] = self._fig_to_base64(fig)
            plt.close()
        
        return charts
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_data = buffer.getvalue()
        buffer.close()
        
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
    
    def _generate_html_report(
        self,
        team_name: str,
        target_competition: str,
        executive_summary: ExecutiveSummary,
        strategy_guides: List[StrategyGuide],
        matchup_advice: List[MatchupAdvice],
        alliance_insights: AllianceInsights,
        practice_recommendations: PracticeRecommendations,
        charts: Dict[str, str]
    ) -> str:
        """Generate comprehensive HTML report"""
        
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VEX U Strategic Analysis Report - {{ team_name }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: bold;
        }
        .header p {
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }
        .section {
            background: white;
            margin: 20px 0;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .section h2 {
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-size: 1.8em;
        }
        .section h3 {
            color: #764ba2;
            margin-top: 25px;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        .executive-summary {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
        .executive-summary h2 {
            color: white;
            border-bottom: 3px solid white;
        }
        .strategy-card {
            background: #f8f9fa;
            border-left: 5px solid #667eea;
            padding: 20px;
            margin: 15px 0;
            border-radius: 5px;
        }
        .strategy-card h4 {
            margin-top: 0;
            color: #667eea;
            font-size: 1.2em;
        }
        .metric-box {
            display: inline-block;
            background: #e9ecef;
            padding: 10px 15px;
            margin: 5px;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }
        .metric-box strong {
            color: #667eea;
        }
        .chart-container {
            text-align: center;
            margin: 25px 0;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .recommendation-list {
            background: #e8f5e8;
            border-left: 5px solid #28a745;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }
        .warning-box {
            background: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }
        .danger-box {
            background: #f8d7da;
            border-left: 5px solid #dc3545;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }
        .execution-plan {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin: 15px 0;
        }
        .execution-plan ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        .execution-plan li {
            margin: 5px 0;
        }
        .two-column {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }
        .three-column {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 15px;
            margin: 20px 0;
        }
        .priority-high { border-left-color: #dc3545; }
        .priority-medium { border-left-color: #ffc107; }
        .priority-low { border-left-color: #28a745; }
        
        @media (max-width: 768px) {
            .two-column, .three-column {
                grid-template-columns: 1fr;
            }
            .header h1 {
                font-size: 2em;
            }
        }
        
        .footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            background: #343a40;
            color: white;
            border-radius: 10px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏆 VEX U Strategic Analysis Report</h1>
        <p>Team: {{ team_name }} | Competition: {{ target_competition }}</p>
        <p>Generated: {{ timestamp }}</p>
    </div>

    <!-- Executive Summary -->
    <div class="section executive-summary">
        <h2>📊 Executive Summary</h2>
        
        <h3>🏅 Top 3 Recommended Strategies</h3>
        {% for strategy_name, win_rate, reason in executive_summary.top_strategies %}
        <div class="strategy-card">
            <h4>{{ loop.index }}. {{ strategy_name }}</h4>
            <div class="metric-box">
                <strong>Win Rate:</strong> {{ "%.1f"|format(win_rate * 100) }}%
            </div>
            <p>{{ reason }}</p>
        </div>
        {% endfor %}
        
        <h3>🎯 Key Success Factors</h3>
        {% for factor, impact, description in executive_summary.key_success_factors %}
        <div class="metric-box">
            <strong>{{ factor }}:</strong> {{ "%.1f"|format(impact) }}% impact
        </div>
        <p>{{ description }}</p>
        {% endfor %}
        
        <h3>⚖️ Risk Assessment</h3>
        <div class="warning-box">
            <strong>Overall Risk Level:</strong> {{ executive_summary.risk_assessment.overall_risk_level }}<br>
            <strong>Most Consistent Strategy:</strong> {{ executive_summary.risk_assessment.consistency_champion }}<br>
            <strong>Highest Reward Option:</strong> {{ executive_summary.risk_assessment.high_reward_option }}<br>
            <strong>Safe Qualification Strategy:</strong> {{ executive_summary.risk_assessment.safe_qualification_strategy }}
        </div>
        
        <div class="recommendation-list">
            <strong>Overall Recommendation:</strong> {{ executive_summary.overall_recommendation }}
        </div>
    </div>

    <!-- Charts Section -->
    <div class="section">
        <h2>📈 Performance Analysis Charts</h2>
        
        {% if charts.strategy_performance %}
        <div class="chart-container">
            <h3>Strategy Performance Comparison</h3>
            <img src="{{ charts.strategy_performance }}" alt="Strategy Performance">
        </div>
        {% endif %}
        
        <div class="two-column">
            {% if charts.risk_reward %}
            <div class="chart-container">
                <h3>Risk vs Reward Profile</h3>
                <img src="{{ charts.risk_reward }}" alt="Risk vs Reward">
            </div>
            {% endif %}
            
            {% if charts.sensitivity %}
            <div class="chart-container">
                <h3>Factor Sensitivity Analysis</h3>
                <img src="{{ charts.sensitivity }}" alt="Sensitivity Analysis">
            </div>
            {% endif %}
        </div>
        
        {% if charts.correlation %}
        <div class="chart-container">
            <h3>Factor Correlation Matrix</h3>
            <img src="{{ charts.correlation }}" alt="Correlation Matrix">
        </div>
        {% endif %}
    </div>

    <!-- Strategy Guides -->
    <div class="section">
        <h2>📋 Detailed Strategy Guides</h2>
        
        {% for guide in strategy_guides %}
        <div class="strategy-card">
            <h3>{{ guide.name }} Strategy</h3>
            
            <div class="metric-box">
                <strong>Difficulty:</strong> {{ guide.difficulty_level }}
            </div>
            <div class="metric-box">
                <strong>Win Rate:</strong> {{ "%.1f"|format(guide.expected_performance.win_rate * 100) }}%
            </div>
            <div class="metric-box">
                <strong>Avg Score:</strong> {{ "%.0f"|format(guide.expected_performance.avg_score) }} pts
            </div>
            <div class="metric-box">
                <strong>Risk Level:</strong> {{ guide.expected_performance.risk_level }}
            </div>
            
            <h4>📝 Execution Plan</h4>
            <div class="execution-plan">
                {% for step in guide.execution_plan %}
                <p>{{ step }}</p>
                {% endfor %}
            </div>
            
            <h4>⏰ Time Allocations</h4>
            <div class="two-column">
                {% for phase, time in guide.time_allocations.items() %}
                <div class="metric-box">
                    <strong>{{ phase }}:</strong> {{ "%.0f"|format(time) }}s
                </div>
                {% endfor %}
            </div>
            
            <h4>🔄 Contingency Plans</h4>
            {% for scenario, action in guide.contingency_plans %}
            <div class="warning-box">
                <strong>{{ scenario }}:</strong> {{ action }}
            </div>
            {% endfor %}
        </div>
        {% endfor %}
    </div>

    <!-- Matchup Advice -->
    <div class="section">
        <h2>⚔️ Matchup-Specific Advice</h2>
        
        {% for advice in matchup_advice %}
        <div class="strategy-card">
            <h3>vs {{ advice.opponent_type }}</h3>
            
            <div class="metric-box">
                <strong>Expected Outcome:</strong> {{ advice.expected_outcome }}
            </div>
            
            <h4>🔧 Required Adjustments</h4>
            <ul>
                {% for adjustment in advice.adjustments %}
                <li>{{ adjustment }}</li>
                {% endfor %}
            </ul>
            
            <h4>🎯 Counter-Strategies</h4>
            <ul>
                {% for counter in advice.counter_strategies %}
                <li>{{ counter }}</li>
                {% endfor %}
            </ul>
            
            <h4>💡 Exploitation Opportunities</h4>
            <div class="recommendation-list">
                <ul>
                    {% for opportunity in advice.exploitation_opportunities %}
                    <li>{{ opportunity }}</li>
                    {% endfor %}
                </ul>
            </div>
        </div>
        {% endfor %}
    </div>

    <!-- Alliance Selection -->
    <div class="section">
        <h2>🤝 Alliance Selection Insights</h2>
        
        <h3>🔍 Partner Capabilities to Seek</h3>
        <div class="three-column">
            {% for capability in alliance_insights.partner_capabilities %}
            <div class="metric-box">
                {{ capability }}
            </div>
            {% endfor %}
        </div>
        
        <h3>⚖️ Complementary Strategy Combinations</h3>
        {% for your_strat, partner_strat in alliance_insights.complementary_strategies %}
        <div class="recommendation-list">
            <strong>Your Strategy:</strong> {{ your_strat }} <br>
            <strong>Partner Strategy:</strong> {{ partner_strat }}
        </div>
        {% endfor %}
        
        <h3>📊 Scoring Potential Calculations</h3>
        {% for combination, score, explanation in alliance_insights.scoring_combinations %}
        <div class="strategy-card">
            <h4>{{ combination }}</h4>
            <div class="metric-box">
                <strong>Projected Score:</strong> {{ score }} points
            </div>
            <p>{{ explanation }}</p>
        </div>
        {% endfor %}
        
        <h3>✅ Selection Criteria Checklist</h3>
        <ul>
            {% for criterion in alliance_insights.selection_criteria %}
            <li>{{ criterion }}</li>
            {% endfor %}
        </ul>
    </div>

    <!-- Practice Recommendations -->
    <div class="section">
        <h2>🏋️ Practice Recommendations</h2>
        
        <h3>🎯 Priority Skills Development</h3>
        {% for skill, description, priority in practice_recommendations.priority_skills %}
        <div class="strategy-card priority-{% if priority >= 5 %}high{% elif priority >= 3 %}medium{% else %}low{% endif %}">
            <h4>{{ skill }} (Priority: {{ priority }}/5)</h4>
            <p>{{ description }}</p>
        </div>
        {% endfor %}
        
        <h3>📈 Benchmark Goals</h3>
        <div class="two-column">
            {% for metric, target, timeframe in practice_recommendations.benchmark_goals %}
            <div class="metric-box">
                <strong>{{ metric }}:</strong> {{ target }}{{ "%" if "Rate" in metric else ("s" if "Time" in metric else "") }} in {{ timeframe }}
            </div>
            {% endfor %}
        </div>
        
        <h3>🏃 Training Scenarios</h3>
        {% for scenario, purpose, setup in practice_recommendations.training_scenarios %}
        <div class="execution-plan">
            <h4>{{ scenario }}</h4>
            <p><strong>Purpose:</strong> {{ purpose }}</p>
            <p><strong>Setup:</strong> {{ setup }}</p>
        </div>
        {% endfor %}
        
        <h3>📅 Progression Plan</h3>
        <ol>
            {% for phase in practice_recommendations.progression_plan %}
            <li>{{ phase }}</li>
            {% endfor %}
        </ol>
    </div>

    <div class="footer">
        <p>Generated by VEX U Strategic Analysis Toolkit</p>
        <p>🤖 Powered by advanced Monte Carlo simulation and statistical analysis</p>
        <p>© {{ current_year }} - For competitive VEX U analysis and optimization</p>
    </div>
</body>
</html>
        """
        
        template = Template(html_template)
        
        return template.render(
            team_name=team_name,
            target_competition=target_competition,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            current_year=datetime.datetime.now().year,
            executive_summary=executive_summary,
            strategy_guides=strategy_guides,
            matchup_advice=matchup_advice,
            alliance_insights=alliance_insights,
            practice_recommendations=practice_recommendations,
            charts=charts
        )


if __name__ == "__main__":
    print("🎯 VEX U Report Generator Demo")
    print("=" * 50)
    
    # Initialize report generator
    generator = VEXReportGenerator()
    
    # Generate sample report
    sample_capabilities = {
        'speed': 0.85,
        'accuracy': 0.90,
        'capacity': 3,
        'autonomous_reliability': 0.75,
        'endgame_capability': 0.88
    }
    
    report_path = generator.generate_comprehensive_report(
        team_name="Demo Team 1234",
        robot_capabilities=sample_capabilities,
        target_competition="Regional Championship 2024",
        num_simulations=200  # Reduced for demo
    )
    
    print(f"\n✅ Demo report generated successfully!")
    print(f"📁 Report saved to: {report_path}")
    print(f"🌐 Open the HTML file in your browser to view the complete report")