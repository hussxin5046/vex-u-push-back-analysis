#!/usr/bin/env python3

"""
VEX U Push Back Statistical Analysis Module
Advanced statistical analysis to find winning edges and strategic insights.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize_scalar
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

from core.simulator import ScoringSimulator, AllianceStrategy, Zone, ParkingLocation
from analysis.strategy_analyzer import AdvancedStrategyAnalyzer
from core.scenario_generator import ScenarioGenerator, SkillLevel, StrategyType


@dataclass
class SensitivityResult:
    factor_name: str
    impact_score: float
    correlation_strength: float
    statistical_significance: float
    base_value: float
    optimal_range: Tuple[float, float]
    sensitivity_curve: List[Tuple[float, float]]  # (factor_value, win_rate)


@dataclass
class MinimumViableStrategy:
    strategy_name: str
    minimum_scoring_rate: float  # blocks per second
    minimum_blocks_total: int
    minimum_auto_blocks: int
    required_zones: List[Zone]
    parking_requirement: List[ParkingLocation]
    confidence_level: float
    win_rate_at_minimum: float


@dataclass
class VarianceAnalysis:
    strategy_name: str
    mean_score: float
    standard_deviation: float
    coefficient_variation: float  # CV = std/mean
    confidence_interval_95: Tuple[float, float]
    worst_case_5th_percentile: float
    best_case_95th_percentile: float
    consistency_rank: int
    risk_category: str  # "Low", "Medium", "High"


@dataclass
class ConfidenceInterval:
    strategy_name: str
    mean_score: float
    confidence_95_lower: float
    confidence_95_upper: float
    confidence_99_lower: float
    confidence_99_upper: float
    sample_size: int
    margin_of_error_95: float
    distribution_type: str  # "Normal", "Skewed", etc.


@dataclass
class CorrelationInsight:
    factor_a: str
    factor_b: str
    correlation_coefficient: float
    p_value: float
    significance: str  # "Highly Significant", "Significant", "Not Significant"
    interpretation: str
    strength_category: str  # "Strong", "Moderate", "Weak"


class StatisticalAnalyzer:
    def __init__(self, simulator: ScoringSimulator):
        self.simulator = simulator
        self.strategy_analyzer = AdvancedStrategyAnalyzer(simulator)
        self.generator = ScenarioGenerator(simulator)
        
        # Statistical thresholds
        self.significance_levels = {
            'highly_significant': 0.01,
            'significant': 0.05,
            'marginally_significant': 0.10
        }
        
        # Cache for expensive computations
        self._sensitivity_cache = {}
        self._simulation_cache = {}
    
    def perform_sensitivity_analysis(
        self, 
        strategies: List[AllianceStrategy],
        factors: List[str] = None,
        num_simulations: int = 500
    ) -> Dict[str, SensitivityResult]:
        """Analyze sensitivity of winning to various factors"""
        
        if factors is None:
            factors = [
                'total_blocks',
                'autonomous_blocks', 
                'scoring_rate',
                'zone_control',
                'parking_points',
                'block_distribution_balance',
                'autonomous_bonus_probability'
            ]
        
        print("🔍 Performing comprehensive sensitivity analysis...")
        sensitivity_results = {}
        
        for factor in factors:
            print(f"   Analyzing sensitivity to: {factor}")
            
            # Generate factor variation data
            factor_impacts = []
            
            for strategy in strategies:
                # Get base performance
                base_metrics = self.strategy_analyzer.analyze_strategy_comprehensive(
                    strategy, num_simulations
                )
                base_win_rate = base_metrics.win_rate
                
                # Vary the factor and measure impact
                sensitivity_curve = []
                factor_values = self._get_factor_variation_range(factor, strategy)
                
                for factor_value in factor_values:
                    modified_strategy = self._modify_strategy_factor(
                        strategy, factor, factor_value
                    )
                    
                    if modified_strategy:
                        modified_metrics = self.strategy_analyzer.analyze_strategy_comprehensive(
                            modified_strategy, min(num_simulations, 100)  # Faster for variations
                        )
                        win_rate_change = modified_metrics.win_rate - base_win_rate
                        factor_impacts.append((factor_value, win_rate_change))
                        sensitivity_curve.append((factor_value, modified_metrics.win_rate))
                
                # Calculate impact score and correlation
                if factor_impacts:
                    values, changes = zip(*factor_impacts)
                    correlation = abs(np.corrcoef(values, changes)[0, 1]) if len(values) > 1 else 0
                    impact_score = np.std(changes) * 100  # Standard deviation of win rate changes
                    
                    # Statistical significance test
                    if len(changes) > 2:
                        _, p_value = stats.pearsonr(values, changes)
                        statistical_significance = p_value
                    else:
                        statistical_significance = 1.0
                    
                    # Determine optimal range
                    best_idx = np.argmax([abs(change) for _, change in factor_impacts])
                    optimal_value = factor_impacts[best_idx][0]
                    
                    # Estimate optimal range (±20% of best value)
                    optimal_range = (optimal_value * 0.8, optimal_value * 1.2)
                    
                    sensitivity_results[factor] = SensitivityResult(
                        factor_name=factor,
                        impact_score=impact_score,
                        correlation_strength=correlation,
                        statistical_significance=statistical_significance,
                        base_value=self._get_strategy_factor_value(strategy, factor),
                        optimal_range=optimal_range,
                        sensitivity_curve=sensitivity_curve
                    )
        
        return sensitivity_results
    
    def find_minimum_viable_strategies(
        self,
        target_win_rate: float = 0.5,
        confidence_level: float = 0.95,
        num_simulations: int = 1000
    ) -> List[MinimumViableStrategy]:
        """Find minimum requirements for viable strategies"""
        
        print(f"🎯 Finding minimum viable strategies (target: {target_win_rate:.1%} win rate)...")
        
        viable_strategies = []
        
        # Generate a range of strategy configurations
        base_configurations = [
            # Minimal configurations to test
            {"auto_blocks": 4, "driver_blocks": 15, "zones": 0, "parking": 0},
            {"auto_blocks": 6, "driver_blocks": 12, "zones": 1, "parking": 1},
            {"auto_blocks": 8, "driver_blocks": 10, "zones": 0, "parking": 2},
            {"auto_blocks": 5, "driver_blocks": 18, "zones": 2, "parking": 0},
            {"auto_blocks": 3, "driver_blocks": 20, "zones": 1, "parking": 1},
        ]
        
        for i, config in enumerate(base_configurations):
            strategy_name = f"Minimal_Strategy_{i+1}"
            
            # Binary search to find minimum scoring rate
            min_scoring_rate = self._find_minimum_scoring_rate(
                config, target_win_rate, num_simulations
            )
            
            if min_scoring_rate > 0:
                # Create the minimum viable strategy
                min_strategy = self._create_strategy_from_config(
                    strategy_name, config, min_scoring_rate
                )
                
                # Validate with full simulation
                metrics = self.strategy_analyzer.analyze_strategy_comprehensive(
                    min_strategy, num_simulations
                )
                
                if metrics.win_rate >= target_win_rate:
                    # Determine minimum requirements
                    total_blocks = config["auto_blocks"] + config["driver_blocks"]
                    required_zones = self._get_zones_from_count(config["zones"])
                    parking_req = self._get_parking_from_count(config["parking"])
                    
                    viable_strategy = MinimumViableStrategy(
                        strategy_name=strategy_name,
                        minimum_scoring_rate=min_scoring_rate,
                        minimum_blocks_total=total_blocks,
                        minimum_auto_blocks=config["auto_blocks"],
                        required_zones=required_zones,
                        parking_requirement=parking_req,
                        confidence_level=confidence_level,
                        win_rate_at_minimum=metrics.win_rate
                    )
                    
                    viable_strategies.append(viable_strategy)
                    print(f"   ✅ Found viable: {strategy_name} - {min_scoring_rate:.3f} blocks/sec")
        
        # Sort by minimum requirements (lowest first)
        viable_strategies.sort(key=lambda x: x.minimum_blocks_total)
        
        return viable_strategies
    
    def analyze_variance_and_reliability(
        self,
        strategies: List[AllianceStrategy],
        num_simulations: int = 1000
    ) -> List[VarianceAnalysis]:
        """Analyze strategy variance and reliability"""
        
        print("📊 Analyzing strategy variance and reliability...")
        
        variance_analyses = []
        
        for strategy in strategies:
            print(f"   Analyzing variance for: {strategy.name}")
            
            # Run multiple simulations to get score distribution
            scores = []
            for _ in range(num_simulations):
                # Simulate against random opponent
                opponent = self.generator.generate_random_strategy("Random", (20, 35))
                result = self.simulator.simulate_match(strategy, opponent)
                scores.append(result.red_score)
            
            scores = np.array(scores)
            
            # Calculate statistical measures
            mean_score = np.mean(scores)
            std_dev = np.std(scores)
            cv = std_dev / mean_score if mean_score > 0 else float('inf')
            
            # Confidence intervals
            ci_95 = stats.norm.interval(0.95, loc=mean_score, scale=std_dev)
            
            # Percentiles
            p5 = np.percentile(scores, 5)
            p95 = np.percentile(scores, 95)
            
            # Risk categorization
            if cv < 0.1:
                risk_category = "Low"
            elif cv < 0.2:
                risk_category = "Medium"
            else:
                risk_category = "High"
            
            variance_analysis = VarianceAnalysis(
                strategy_name=strategy.name,
                mean_score=mean_score,
                standard_deviation=std_dev,
                coefficient_variation=cv,
                confidence_interval_95=ci_95,
                worst_case_5th_percentile=p5,
                best_case_95th_percentile=p95,
                consistency_rank=0,  # Will be filled after sorting
                risk_category=risk_category
            )
            
            variance_analyses.append(variance_analysis)
        
        # Rank by consistency (lower CV = higher consistency = lower rank number)
        variance_analyses.sort(key=lambda x: x.coefficient_variation)
        for i, analysis in enumerate(variance_analyses):
            analysis.consistency_rank = i + 1
        
        return variance_analyses
    
    def create_confidence_intervals(
        self,
        strategies: List[AllianceStrategy],
        num_simulations: int = 1000
    ) -> List[ConfidenceInterval]:
        """Create detailed confidence intervals for strategy performance"""
        
        print("📈 Creating confidence intervals for strategy performance...")
        
        confidence_intervals = []
        
        for strategy in strategies:
            print(f"   Calculating confidence intervals for: {strategy.name}")
            
            # Collect performance data
            scores = []
            for _ in range(num_simulations):
                opponent = self.generator.generate_random_strategy("Opponent", (25, 35))
                result = self.simulator.simulate_match(strategy, opponent)
                scores.append(result.red_score)
            
            scores = np.array(scores)
            
            # Basic statistics
            mean_score = np.mean(scores)
            std_error = stats.sem(scores)
            
            # Confidence intervals
            ci_95 = stats.t.interval(0.95, len(scores)-1, loc=mean_score, scale=std_error)
            ci_99 = stats.t.interval(0.99, len(scores)-1, loc=mean_score, scale=std_error)
            
            # Margin of error
            margin_95 = abs(ci_95[1] - mean_score)
            
            # Distribution analysis
            _, p_value_normal = stats.normaltest(scores)
            if p_value_normal > 0.05:
                distribution_type = "Normal"
            else:
                # Test for skewness
                skewness = stats.skew(scores)
                if abs(skewness) > 1:
                    distribution_type = "Highly Skewed"
                elif abs(skewness) > 0.5:
                    distribution_type = "Moderately Skewed"
                else:
                    distribution_type = "Slightly Skewed"
            
            confidence_interval = ConfidenceInterval(
                strategy_name=strategy.name,
                mean_score=mean_score,
                confidence_95_lower=ci_95[0],
                confidence_95_upper=ci_95[1],
                confidence_99_lower=ci_99[0],
                confidence_99_upper=ci_99[1],
                sample_size=num_simulations,
                margin_of_error_95=margin_95,
                distribution_type=distribution_type
            )
            
            confidence_intervals.append(confidence_interval)
        
        return confidence_intervals
    
    def perform_correlation_analysis(
        self,
        strategies: List[AllianceStrategy],
        num_simulations: int = 500
    ) -> List[CorrelationInsight]:
        """Analyze correlations between different factors and success"""
        
        print("🔗 Performing correlation analysis to find winning patterns...")
        
        # Collect comprehensive data
        data_points = []
        
        for strategy in strategies:
            print(f"   Collecting data for: {strategy.name}")
            
            # Extract strategy features
            total_blocks = sum(strategy.blocks_scored_auto.values()) + sum(strategy.blocks_scored_driver.values())
            auto_blocks = sum(strategy.blocks_scored_auto.values())
            driver_blocks = sum(strategy.blocks_scored_driver.values())
            zone_count = len(strategy.zones_controlled)
            parking_count = len([p for p in strategy.robots_parked if p != ParkingLocation.NONE])
            
            # Calculate balance metrics
            auto_ratio = auto_blocks / total_blocks if total_blocks > 0 else 0
            long_goal_blocks = strategy.blocks_scored_auto.get("long_1", 0) + strategy.blocks_scored_auto.get("long_2", 0) + \
                             strategy.blocks_scored_driver.get("long_1", 0) + strategy.blocks_scored_driver.get("long_2", 0)
            center_goal_blocks = strategy.blocks_scored_auto.get("center_1", 0) + strategy.blocks_scored_auto.get("center_2", 0) + \
                               strategy.blocks_scored_driver.get("center_1", 0) + strategy.blocks_scored_driver.get("center_2", 0)
            
            goal_balance = min(long_goal_blocks, center_goal_blocks) / max(long_goal_blocks, center_goal_blocks) if max(long_goal_blocks, center_goal_blocks) > 0 else 0
            
            # Performance metrics
            metrics = self.strategy_analyzer.analyze_strategy_comprehensive(strategy, num_simulations)
            
            data_point = {
                'strategy_name': strategy.name,
                'total_blocks': total_blocks,
                'auto_blocks': auto_blocks,
                'driver_blocks': driver_blocks,
                'zone_count': zone_count,
                'parking_count': parking_count,
                'auto_ratio': auto_ratio,
                'goal_balance': goal_balance,
                'win_rate': metrics.win_rate,
                'avg_score': metrics.avg_score,
                'consistency': metrics.consistency_score,
                'risk_reward': metrics.risk_reward_ratio
            }
            
            data_points.append(data_point)
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(data_points)
        
        # Define factor pairs to analyze
        factor_pairs = [
            ('total_blocks', 'win_rate'),
            ('auto_blocks', 'win_rate'),
            ('auto_ratio', 'win_rate'),
            ('zone_count', 'win_rate'),
            ('parking_count', 'win_rate'),
            ('goal_balance', 'win_rate'),
            ('avg_score', 'win_rate'),
            ('total_blocks', 'consistency'),
            ('auto_ratio', 'consistency'),
            ('zone_count', 'avg_score'),
            ('parking_count', 'avg_score')
        ]
        
        correlation_insights = []
        
        for factor_a, factor_b in factor_pairs:
            if factor_a in df.columns and factor_b in df.columns:
                # Calculate correlation
                corr_coef, p_value = stats.pearsonr(df[factor_a], df[factor_b])
                
                # Determine significance
                if p_value < self.significance_levels['highly_significant']:
                    significance = "Highly Significant"
                elif p_value < self.significance_levels['significant']:
                    significance = "Significant"
                elif p_value < self.significance_levels['marginally_significant']:
                    significance = "Marginally Significant"
                else:
                    significance = "Not Significant"
                
                # Determine strength
                abs_corr = abs(corr_coef)
                if abs_corr >= 0.7:
                    strength_category = "Strong"
                elif abs_corr >= 0.4:
                    strength_category = "Moderate"
                elif abs_corr >= 0.2:
                    strength_category = "Weak"
                else:
                    strength_category = "Very Weak"
                
                # Generate interpretation
                direction = "positive" if corr_coef > 0 else "negative"
                interpretation = f"There is a {strength_category.lower()} {direction} relationship between {factor_a} and {factor_b}"
                
                correlation_insight = CorrelationInsight(
                    factor_a=factor_a,
                    factor_b=factor_b,
                    correlation_coefficient=corr_coef,
                    p_value=p_value,
                    significance=significance,
                    interpretation=interpretation,
                    strength_category=strength_category
                )
                
                correlation_insights.append(correlation_insight)
        
        # Sort by correlation strength
        correlation_insights.sort(key=lambda x: abs(x.correlation_coefficient), reverse=True)
        
        return correlation_insights
    
    def generate_statistical_visualizations(
        self,
        sensitivity_results: Dict[str, SensitivityResult],
        variance_analyses: List[VarianceAnalysis],
        confidence_intervals: List[ConfidenceInterval],
        correlation_insights: List[CorrelationInsight],
        save_dir: str = "./statistical_analysis/"
    ):
        """Generate comprehensive statistical visualizations"""
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"📊 Generating statistical visualizations in {save_dir}")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # 1. Sensitivity Analysis Plot
        if sensitivity_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Impact scores
            factors = list(sensitivity_results.keys())
            impacts = [sensitivity_results[f].impact_score for f in factors]
            
            ax1.barh(factors, impacts, color='skyblue', alpha=0.7)
            ax1.set_xlabel('Impact Score (Win Rate Std Dev %)')
            ax1.set_title('Factor Sensitivity Analysis')
            ax1.grid(True, alpha=0.3)
            
            # Correlation strengths
            correlations = [sensitivity_results[f].correlation_strength for f in factors]
            colors = ['red' if c > 0.7 else 'orange' if c > 0.4 else 'green' for c in correlations]
            
            ax2.barh(factors, correlations, color=colors, alpha=0.7)
            ax2.set_xlabel('Correlation Strength')
            ax2.set_title('Factor-Performance Correlations')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, 1)
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/sensitivity_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Variance and Risk Analysis
        if variance_analyses:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Consistency ranking
            strategies = [v.strategy_name for v in variance_analyses]
            cvs = [v.coefficient_variation for v in variance_analyses]
            colors = ['green' if cv < 0.1 else 'orange' if cv < 0.2 else 'red' for cv in cvs]
            
            ax1.barh(strategies, cvs, color=colors, alpha=0.7)
            ax1.set_xlabel('Coefficient of Variation (Risk Level)')
            ax1.set_title('Strategy Risk Assessment')
            ax1.grid(True, alpha=0.3)
            
            # Score distributions
            means = [v.mean_score for v in variance_analyses]
            stds = [v.standard_deviation for v in variance_analyses]
            
            ax2.scatter(means, stds, s=100, alpha=0.7, c=colors)
            for i, strategy in enumerate(strategies):
                ax2.annotate(strategy, (means[i], stds[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax2.set_xlabel('Mean Score')
            ax2.set_ylabel('Standard Deviation')
            ax2.set_title('Risk vs Reward Profile')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/variance_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Confidence Intervals
        if confidence_intervals:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            strategies = [ci.strategy_name for ci in confidence_intervals]
            means = [ci.mean_score for ci in confidence_intervals]
            lower_95 = [ci.confidence_95_lower for ci in confidence_intervals]
            upper_95 = [ci.confidence_95_upper for ci in confidence_intervals]
            
            y_pos = np.arange(len(strategies))
            
            # Plot confidence intervals
            ax.barh(y_pos, means, alpha=0.6, color='skyblue', label='Mean Score')
            ax.errorbar(means, y_pos, 
                       xerr=[[m - l for m, l in zip(means, lower_95)],
                             [u - m for m, u in zip(means, upper_95)]],
                       fmt='o', color='navy', capsize=5, capthick=2, label='95% CI')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(strategies)
            ax.set_xlabel('Score')
            ax.set_title('Strategy Performance with 95% Confidence Intervals')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/confidence_intervals.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Correlation Heatmap
        if correlation_insights:
            # Create correlation matrix
            factors = list(set([ci.factor_a for ci in correlation_insights] + 
                            [ci.factor_b for ci in correlation_insights]))
            
            corr_matrix = np.zeros((len(factors), len(factors)))
            
            for insight in correlation_insights:
                i = factors.index(insight.factor_a)
                j = factors.index(insight.factor_b)
                corr_matrix[i, j] = insight.correlation_coefficient
                corr_matrix[j, i] = insight.correlation_coefficient
            
            # Fill diagonal
            np.fill_diagonal(corr_matrix, 1.0)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            sns.heatmap(corr_matrix, 
                       xticklabels=factors, 
                       yticklabels=factors,
                       annot=True, 
                       fmt='.2f',
                       cmap='RdBu_r',
                       center=0,
                       ax=ax)
            
            ax.set_title('Factor Correlation Matrix')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/correlation_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Statistical visualizations saved to {save_dir}")
    
    def create_comprehensive_statistical_report(
        self,
        sensitivity_results: Dict[str, SensitivityResult],
        minimum_viable: List[MinimumViableStrategy],
        variance_analyses: List[VarianceAnalysis],
        confidence_intervals: List[ConfidenceInterval],
        correlation_insights: List[CorrelationInsight]
    ) -> str:
        """Generate comprehensive statistical analysis report"""
        
        report = []
        report.append("=" * 80)
        report.append("VEX U PUSH BACK - STATISTICAL ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Executive Summary
        report.append("\n📊 EXECUTIVE SUMMARY")
        report.append("-" * 50)
        
        if sensitivity_results:
            most_impactful = max(sensitivity_results.values(), key=lambda x: x.impact_score)
            report.append(f"Most Impactful Factor: {most_impactful.factor_name}")
            report.append(f"Impact Score: {most_impactful.impact_score:.2f}%")
        
        if minimum_viable:
            most_efficient = min(minimum_viable, key=lambda x: x.minimum_blocks_total)
            report.append(f"Most Efficient Strategy: {most_efficient.strategy_name}")
            report.append(f"Minimum Blocks Required: {most_efficient.minimum_blocks_total}")
        
        if variance_analyses:
            most_consistent = min(variance_analyses, key=lambda x: x.coefficient_variation)
            report.append(f"Most Consistent Strategy: {most_consistent.strategy_name}")
            report.append(f"Risk Level: {most_consistent.risk_category}")
        
        # Sensitivity Analysis
        if sensitivity_results:
            report.append("\n\n🔍 SENSITIVITY ANALYSIS")
            report.append("-" * 50)
            
            # Sort by impact score
            sorted_factors = sorted(sensitivity_results.items(), 
                                  key=lambda x: x[1].impact_score, reverse=True)
            
            for factor_name, result in sorted_factors:
                report.append(f"\n{factor_name.upper()}:")
                report.append(f"  Impact Score: {result.impact_score:.2f}%")
                report.append(f"  Correlation: {result.correlation_strength:.3f}")
                report.append(f"  Statistical Significance: {result.statistical_significance:.3f}")
                report.append(f"  Optimal Range: {result.optimal_range[0]:.2f} - {result.optimal_range[1]:.2f}")
                
                if result.statistical_significance < 0.05:
                    report.append("  ✅ STATISTICALLY SIGNIFICANT")
                else:
                    report.append("  ⚠️  Not statistically significant")
        
        # Minimum Viable Strategies
        if minimum_viable:
            report.append("\n\n🎯 MINIMUM VIABLE STRATEGIES")
            report.append("-" * 50)
            
            for mvs in minimum_viable:
                report.append(f"\n{mvs.strategy_name}:")
                report.append(f"  Minimum Scoring Rate: {mvs.minimum_scoring_rate:.3f} blocks/sec")
                report.append(f"  Total Blocks Required: {mvs.minimum_blocks_total}")
                report.append(f"  Autonomous Minimum: {mvs.minimum_auto_blocks}")
                report.append(f"  Zones Required: {len(mvs.required_zones)}")
                report.append(f"  Parking Required: {len(mvs.parking_requirement)} robots")
                report.append(f"  Win Rate at Minimum: {mvs.win_rate_at_minimum:.1%}")
        
        # Variance and Reliability
        if variance_analyses:
            report.append("\n\n📊 VARIANCE AND RELIABILITY ANALYSIS")
            report.append("-" * 50)
            
            report.append("\nConsistency Rankings (1 = Most Consistent):")
            for analysis in variance_analyses:
                report.append(f"{analysis.consistency_rank}. {analysis.strategy_name}")
                report.append(f"   Mean Score: {analysis.mean_score:.0f} ± {analysis.standard_deviation:.0f}")
                report.append(f"   Coefficient of Variation: {analysis.coefficient_variation:.3f}")
                report.append(f"   Risk Category: {analysis.risk_category}")
                report.append(f"   95% CI: ({analysis.confidence_interval_95[0]:.0f}, {analysis.confidence_interval_95[1]:.0f})")
                report.append(f"   Worst Case (5th percentile): {analysis.worst_case_5th_percentile:.0f}")
        
        # Confidence Intervals
        if confidence_intervals:
            report.append("\n\n📈 CONFIDENCE INTERVALS")
            report.append("-" * 50)
            
            for ci in confidence_intervals:
                report.append(f"\n{ci.strategy_name}:")
                report.append(f"  Mean Score: {ci.mean_score:.0f}")
                report.append(f"  95% Confidence Interval: ({ci.confidence_95_lower:.0f}, {ci.confidence_95_upper:.0f})")
                report.append(f"  99% Confidence Interval: ({ci.confidence_99_lower:.0f}, {ci.confidence_99_upper:.0f})")
                report.append(f"  Margin of Error (95%): ±{ci.margin_of_error_95:.0f}")
                report.append(f"  Distribution: {ci.distribution_type}")
                report.append(f"  Sample Size: {ci.sample_size:,}")
        
        # Correlation Analysis
        if correlation_insights:
            report.append("\n\n🔗 CORRELATION ANALYSIS")
            report.append("-" * 50)
            
            # Group by significance
            significant_correlations = [ci for ci in correlation_insights 
                                     if ci.significance in ["Highly Significant", "Significant"]]
            
            if significant_correlations:
                report.append("\nSignificant Correlations:")
                for insight in significant_correlations:
                    report.append(f"\n{insight.factor_a} ↔ {insight.factor_b}:")
                    report.append(f"  Correlation: {insight.correlation_coefficient:.3f}")
                    report.append(f"  Strength: {insight.strength_category}")
                    report.append(f"  Significance: {insight.significance}")
                    report.append(f"  Interpretation: {insight.interpretation}")
        
        # Strategic Recommendations
        report.append("\n\n💡 KEY STRATEGIC INSIGHTS")
        report.append("-" * 50)
        
        insights = []
        
        if sensitivity_results:
            most_impactful = max(sensitivity_results.values(), key=lambda x: x.impact_score)
            insights.append(f"Focus optimization efforts on {most_impactful.factor_name} - highest impact factor")
        
        if minimum_viable:
            most_efficient = min(minimum_viable, key=lambda x: x.minimum_blocks_total)
            insights.append(f"Teams with limited capabilities should target {most_efficient.minimum_blocks_total} total blocks minimum")
        
        if variance_analyses:
            most_consistent = min(variance_analyses, key=lambda x: x.coefficient_variation)
            highest_risk = max(variance_analyses, key=lambda x: x.coefficient_variation)
            insights.append(f"For qualification rounds, use {most_consistent.strategy_name} (most consistent)")
            insights.append(f"For elimination rounds, consider {highest_risk.strategy_name} (high risk/reward)")
        
        if correlation_insights:
            strong_positive = [ci for ci in correlation_insights 
                             if ci.correlation_coefficient > 0.7 and ci.factor_b == 'win_rate']
            if strong_positive:
                top_predictor = strong_positive[0]
                insights.append(f"Strong predictor of success: {top_predictor.factor_a}")
        
        for i, insight in enumerate(insights, 1):
            report.append(f"{i}. {insight}")
        
        report.append("\n\n" + "=" * 80)
        report.append("END OF STATISTICAL ANALYSIS")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    # Helper methods
    def _get_factor_variation_range(self, factor: str, strategy: AllianceStrategy) -> List[float]:
        """Get appropriate variation range for a factor"""
        
        base_value = self._get_strategy_factor_value(strategy, factor)
        
        if factor == 'total_blocks':
            return np.linspace(10, 60, 11)
        elif factor == 'autonomous_blocks':
            return np.linspace(0, 20, 11)
        elif factor == 'scoring_rate':
            return np.linspace(0.1, 0.8, 8)
        elif factor == 'zone_control':
            return [0, 1, 2, 3]
        elif factor == 'parking_points':
            return [0, 5, 20, 40]
        elif factor == 'block_distribution_balance':
            return np.linspace(0.0, 1.0, 11)
        elif factor == 'autonomous_bonus_probability':
            return np.linspace(0.0, 1.0, 11)
        else:
            # Default: ±50% variation
            return np.linspace(base_value * 0.5, base_value * 1.5, 11)
    
    def _get_strategy_factor_value(self, strategy: AllianceStrategy, factor: str) -> float:
        """Extract current factor value from strategy"""
        
        if factor == 'total_blocks':
            return sum(strategy.blocks_scored_auto.values()) + sum(strategy.blocks_scored_driver.values())
        elif factor == 'autonomous_blocks':
            return sum(strategy.blocks_scored_auto.values())
        elif factor == 'zone_control':
            return len(strategy.zones_controlled)
        elif factor == 'parking_points':
            return sum(20 if p == ParkingLocation.PLATFORM else 5 if p == ParkingLocation.ALLIANCE_ZONE else 0 
                      for p in strategy.robots_parked)
        else:
            return 1.0  # Default value
    
    def _modify_strategy_factor(
        self, 
        strategy: AllianceStrategy, 
        factor: str, 
        new_value: float
    ) -> Optional[AllianceStrategy]:
        """Create modified strategy with changed factor"""
        
        try:
            # This is a simplified implementation
            # In practice, you'd need more sophisticated strategy modification
            if factor == 'total_blocks':
                scale_factor = new_value / self._get_strategy_factor_value(strategy, factor)
                if scale_factor > 0:
                    new_auto = {k: int(v * scale_factor * 0.3) 
                               for k, v in strategy.blocks_scored_auto.items()}
                    new_driver = {k: int(v * scale_factor * 0.7) 
                                 for k, v in strategy.blocks_scored_driver.items()}
                    
                    return AllianceStrategy(
                        name=f"{strategy.name}_modified",
                        blocks_scored_auto=new_auto,
                        blocks_scored_driver=new_driver,
                        zones_controlled=strategy.zones_controlled,
                        robots_parked=strategy.robots_parked
                    )
            
            return None
        except:
            return None
    
    def _find_minimum_scoring_rate(
        self, 
        config: Dict, 
        target_win_rate: float, 
        num_simulations: int
    ) -> float:
        """Binary search to find minimum scoring rate for target win rate"""
        
        def test_scoring_rate(rate: float) -> float:
            strategy = self._create_strategy_from_config("Test", config, rate)
            if strategy:
                metrics = self.strategy_analyzer.analyze_strategy_comprehensive(
                    strategy, min(num_simulations, 100)
                )
                return metrics.win_rate
            return 0.0
        
        # Binary search
        low, high = 0.05, 1.0
        tolerance = 0.01
        
        for _ in range(20):  # Max iterations
            mid = (low + high) / 2
            win_rate = test_scoring_rate(mid)
            
            if abs(win_rate - target_win_rate) < tolerance:
                return mid
            elif win_rate < target_win_rate:
                low = mid
            else:
                high = mid
        
        return mid if test_scoring_rate(mid) >= target_win_rate else 0.0
    
    def _create_strategy_from_config(
        self, 
        name: str, 
        config: Dict, 
        scoring_rate: float
    ) -> AllianceStrategy:
        """Create strategy from configuration parameters"""
        
        auto_blocks = config["auto_blocks"]
        driver_blocks = config["driver_blocks"]
        
        # Distribute blocks across goals
        auto_dist = self._distribute_blocks_evenly(auto_blocks)
        driver_dist = self._distribute_blocks_evenly(driver_blocks)
        
        zones = self._get_zones_from_count(config["zones"])
        parking = self._get_parking_from_count(config["parking"])
        
        return AllianceStrategy(
            name=name,
            blocks_scored_auto=auto_dist,
            blocks_scored_driver=driver_dist,
            zones_controlled=zones,
            robots_parked=parking
        )
    
    def _distribute_blocks_evenly(self, total_blocks: int) -> Dict[str, int]:
        """Distribute blocks evenly across goals"""
        goals = ["long_1", "long_2", "center_1", "center_2"]
        per_goal = total_blocks // len(goals)
        remainder = total_blocks % len(goals)
        
        distribution = {goal: per_goal for goal in goals}
        
        # Distribute remainder
        for i in range(remainder):
            distribution[goals[i]] += 1
        
        return distribution
    
    def _get_zones_from_count(self, count: int) -> List[Zone]:
        """Get zone list from count"""
        zones = [Zone.RED_HOME, Zone.NEUTRAL, Zone.BLUE_HOME]
        return zones[:count]
    
    def _get_parking_from_count(self, count: int) -> List[ParkingLocation]:
        """Get parking list from count"""
        if count == 0:
            return [ParkingLocation.NONE, ParkingLocation.NONE]
        elif count == 1:
            return [ParkingLocation.ALLIANCE_ZONE, ParkingLocation.NONE]
        else:
            return [ParkingLocation.PLATFORM, ParkingLocation.PLATFORM]


if __name__ == "__main__":
    from analysis.strategy_analyzer import AdvancedStrategyAnalyzer
    
    print("🎯 VEX U Statistical Analysis Demo")
    print("=" * 50)
    
    # Initialize
    simulator = ScoringSimulator()
    stat_analyzer = StatisticalAnalyzer(simulator)
    
    # Create sample strategies for analysis
    strategies = [
        AllianceStrategy("Fast & Furious", 
                        {"long_1": 8, "long_2": 8, "center_1": 6, "center_2": 6},
                        {"long_1": 18, "long_2": 18, "center_1": 12, "center_2": 12},
                        [], [ParkingLocation.NONE, ParkingLocation.NONE]),
        AllianceStrategy("Balanced", 
                        {"long_1": 6, "long_2": 6, "center_1": 4, "center_2": 4},
                        {"long_1": 10, "long_2": 10, "center_1": 8, "center_2": 8},
                        [Zone.RED_HOME, Zone.NEUTRAL], 
                        [ParkingLocation.PLATFORM, ParkingLocation.PLATFORM]),
        AllianceStrategy("Zone Control", 
                        {"long_1": 4, "long_2": 4, "center_1": 3, "center_2": 3},
                        {"long_1": 6, "long_2": 6, "center_1": 6, "center_2": 6},
                        [Zone.RED_HOME, Zone.BLUE_HOME, Zone.NEUTRAL],
                        [ParkingLocation.PLATFORM, ParkingLocation.ALLIANCE_ZONE])
    ]
    
    # Run analyses (reduced simulations for demo)
    print("Running statistical analyses...")
    
    sensitivity_results = stat_analyzer.perform_sensitivity_analysis(strategies, num_simulations=100)
    print(f"✅ Sensitivity analysis complete: {len(sensitivity_results)} factors analyzed")
    
    minimum_viable = stat_analyzer.find_minimum_viable_strategies(num_simulations=200)
    print(f"✅ Minimum viable strategies: {len(minimum_viable)} found")
    
    variance_analyses = stat_analyzer.analyze_variance_and_reliability(strategies, num_simulations=200)
    print(f"✅ Variance analysis complete: {len(variance_analyses)} strategies analyzed")
    
    confidence_intervals = stat_analyzer.create_confidence_intervals(strategies, num_simulations=200)
    print(f"✅ Confidence intervals: {len(confidence_intervals)} created")
    
    correlation_insights = stat_analyzer.perform_correlation_analysis(strategies, num_simulations=100)
    print(f"✅ Correlation analysis: {len(correlation_insights)} insights found")
    
    # Generate visualizations
    stat_analyzer.generate_statistical_visualizations(
        sensitivity_results, variance_analyses, confidence_intervals, correlation_insights
    )
    
    # Generate comprehensive report
    report = stat_analyzer.create_comprehensive_statistical_report(
        sensitivity_results, minimum_viable, variance_analyses, 
        confidence_intervals, correlation_insights
    )
    
    print("\n📊 STATISTICAL ANALYSIS COMPLETE!")
    print("=" * 50)
    print("Check ./statistical_analysis/ for visualizations")
    print("\nKey findings:")
    
    if sensitivity_results:
        most_impactful = max(sensitivity_results.values(), key=lambda x: x.impact_score)
        print(f"• Most impactful factor: {most_impactful.factor_name}")
    
    if minimum_viable:
        most_efficient = min(minimum_viable, key=lambda x: x.minimum_blocks_total)
        print(f"• Minimum viable blocks: {most_efficient.minimum_blocks_total}")
    
    if variance_analyses:
        most_consistent = min(variance_analyses, key=lambda x: x.coefficient_variation)
        print(f"• Most consistent strategy: {most_consistent.strategy_name}")
    
    # Save report
    with open("./statistical_analysis/comprehensive_report.txt", "w") as f:
        f.write(report)
    print("• Full report saved to: ./statistical_analysis/comprehensive_report.txt")