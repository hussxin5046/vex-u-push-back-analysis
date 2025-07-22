import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import optuna
import shap
import joblib
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from .feature_engineering import VEXUFeatureExtractor, GameState, create_game_state_from_strategy
    from ..core.simulator import AllianceStrategy, ScoringSimulator, Zone, ParkingLocation
    from ..core.scenario_generator import ScenarioGenerator, SkillLevel, StrategyType
except ImportError:
    # Fallback for when running from main.py
    from ml_models.feature_engineering import VEXUFeatureExtractor, GameState, create_game_state_from_strategy
    from core.simulator import AllianceStrategy, ScoringSimulator, Zone, ParkingLocation
    from core.scenario_generator import ScenarioGenerator, SkillLevel, StrategyType

warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class ScoreOptimizationResult:
    predicted_score: float
    confidence_interval: Tuple[float, float]
    feature_contributions: Dict[str, float]
    optimization_suggestions: List[str]
    risk_assessment: str
    expected_win_probability: float


@dataclass
class ModelPerformance:
    rmse: float
    mae: float
    r2_score: float
    cv_score: float
    cv_std: float
    feature_importance: Dict[str, float]
    shap_values: Optional[np.ndarray] = None


class VEXUScoringOptimizer:
    def __init__(self, model_name: str = "vex_u_scoring_optimizer"):
        self.model_name = model_name
        self.xgb_model = None
        self.rf_model = None
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.feature_extractor = VEXUFeatureExtractor()
        self.simulator = ScoringSimulator(enable_feature_extraction=True)
        self.scenario_generator = ScenarioGenerator(self.simulator)
        
        # SHAP explainer
        self.shap_explainer = None
        self.shap_values = None
        
        # Model paths
        self.model_dir = "models"
        self.xgb_path = os.path.join(self.model_dir, f"{model_name}_xgb.json")
        self.rf_path = os.path.join(self.model_dir, f"{model_name}_rf.pkl")
        self.ensemble_path = os.path.join(self.model_dir, f"{model_name}_ensemble.pkl")
        self.scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.pkl")
        self.config_path = os.path.join(self.model_dir, f"{model_name}_config.json")
        
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Feature names for consistency
        self.feature_names = [
            # Scoring features
            'blocks_points', 'autonomous_points', 'zone_control_points', 'parking_points',
            
            # Goal-specific features
            'long_1_blocks', 'long_2_blocks', 'center_upper_blocks', 'center_lower_blocks',
            'long_1_saturation', 'long_2_saturation', 'center_upper_saturation', 'center_lower_saturation',
            
            # VEX U specific features
            'robot_coordination_distance', 'size_diversity', 'task_allocation_efficiency',
            'small_robot_utilization', 'large_robot_utilization', 'match_progress', 'time_remaining',
            
            # Temporal features
            'scoring_rate_10s', 'time_since_last_score', 'score_momentum', 'zone_control_duration',
            
            # Strategic features
            'average_goal_saturation', 'neutral_zone_presence', 'home_zone_presence',
            'defensive_positioning', 'block_carrying_efficiency', 'high_value_goal_focus',
            
            # Additional features
            'zones_controlled_count', 'parked_robots', 'auto_time_remaining',
            'is_autonomous', 'is_driver_control', 'is_endgame'
        ]
    
    def generate_training_data(self, num_samples: int = 3000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate comprehensive training data for scoring prediction"""
        print(f"Generating {num_samples} training samples for scoring optimization...")
        
        features_list = []
        scores_list = []
        
        for i in range(num_samples):
            if i % 300 == 0:
                print(f"Generated {i}/{num_samples} samples...")
            
            # Generate diverse scenarios
            skill_level = np.random.choice(list(SkillLevel))
            strategy_type = np.random.choice(list(StrategyType))
            
            # Create scenario parameters
            scenario_params = self.scenario_generator._create_scenario_parameters(
                skill_level, strategy_type, f"Alliance_{i}"
            )
            
            # Generate strategy
            strategy = self.scenario_generator.generate_time_based_strategy(
                f"Alliance_{i}", scenario_params
            )
            
            # Generate opponent
            opponent = self.scenario_generator.generate_random_strategy("Opponent")
            
            # Simulate match to get actual score
            result = self.simulator.simulate_match(strategy, opponent, extract_features=True)
            
            if result.red_features is not None:
                # Extract and standardize features
                feature_vector = self._extract_feature_vector(result.red_features)
                
                if len(feature_vector) == len(self.feature_names):
                    features_list.append(feature_vector)
                    scores_list.append(result.red_score)
        
        print(f"Successfully generated {len(features_list)} samples")
        
        X = np.array(features_list, dtype=np.float32)
        y = np.array(scores_list, dtype=np.float32)
        
        return X, y
    
    def _extract_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Extract feature vector in consistent order"""
        feature_vector = []
        
        for feature_name in self.feature_names:
            # Map from full feature names to expected names
            red_feature_name = f"red_{feature_name}"
            value = features.get(red_feature_name, 0.0)
            
            # Handle special cases
            if np.isnan(value) or np.isinf(value):
                value = 0.0
            
            feature_vector.append(value)
        
        return np.array(feature_vector, dtype=np.float32)
    
    def optimize_hyperparameters_xgb(self, X: np.ndarray, y: np.ndarray, n_trials: int = 50) -> Dict:
        """Optimize XGBoost hyperparameters using Optuna"""
        print(f"Optimizing XGBoost hyperparameters with {n_trials} trials...")
        
        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'booster': 'gbtree',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
                'random_state': 42
            }
            
            # Cross-validation
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            
            rmse = np.sqrt(mean_squared_error(y_val, predictions))
            return rmse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"Best RMSE: {study.best_value:.3f}")
        print(f"Best parameters: {study.best_params}")
        
        return study.best_params
    
    def optimize_hyperparameters_rf(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Optimize Random Forest hyperparameters using GridSearchCV"""
        print("Optimizing Random Forest hyperparameters...")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 0.3]
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        print(f"Best RF score: {-grid_search.best_score_:.3f}")
        print(f"Best RF parameters: {grid_search.best_params_}")
        
        return grid_search.best_params_
    
    def train_models(self, X: np.ndarray, y: np.ndarray, optimize_hyperparameters: bool = True) -> ModelPerformance:
        """Train XGBoost, Random Forest, and Ensemble models"""
        print("Training VEX U Scoring Optimizer models...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Optimize hyperparameters if requested
        if optimize_hyperparameters:
            best_xgb_params = self.optimize_hyperparameters_xgb(X_train, y_train, n_trials=30)
            best_rf_params = self.optimize_hyperparameters_rf(X_train, y_train)
        else:
            # Default parameters
            best_xgb_params = {
                'n_estimators': 300,
                'max_depth': 6,
                'learning_rate': 0.1,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 1,
                'reg_lambda': 1
            }
            
            best_rf_params = {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt'
            }
        
        # Train XGBoost
        print("Training XGBoost model...")
        self.xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            eval_metric='rmse',
            random_state=42,
            **best_xgb_params
        )
        self.xgb_model.fit(X_train, y_train)
        
        # Train Random Forest
        print("Training Random Forest model...")
        self.rf_model = RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            **best_rf_params
        )
        self.rf_model.fit(X_train, y_train)
        
        # Create ensemble model
        print("Creating ensemble model...")
        self.ensemble_model = VotingRegressor([
            ('xgb', self.xgb_model),
            ('rf', self.rf_model)
        ])
        self.ensemble_model.fit(X_train, y_train)
        
        # Evaluate models
        ensemble_pred = self.ensemble_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        mae = mean_absolute_error(y_test, ensemble_pred)
        r2 = r2_score(y_test, ensemble_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.ensemble_model, X_train, y_train, 
            cv=5, scoring='neg_mean_squared_error'
        )
        cv_rmse = np.sqrt(-cv_scores.mean())
        cv_std = np.sqrt(cv_scores.std())
        
        # Feature importance (using XGBoost)
        feature_importance = dict(zip(
            self.feature_names, 
            self.xgb_model.feature_importances_
        ))
        
        # Initialize SHAP explainer
        print("Initializing SHAP explainer...")
        self.shap_explainer = shap.Explainer(self.xgb_model, X_train[:100])  # Use subset for speed
        self.shap_values = self.shap_explainer(X_test[:50])  # Calculate for subset
        
        # Save models
        self.save_models()
        
        performance = ModelPerformance(
            rmse=rmse,
            mae=mae,
            r2_score=r2,
            cv_score=cv_rmse,
            cv_std=cv_std,
            feature_importance=feature_importance,
            shap_values=self.shap_values.values if self.shap_values is not None else None
        )
        
        print(f"Model training completed:")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  MAE: {mae:.3f}")
        print(f"  R²: {r2:.3f}")
        print(f"  CV RMSE: {cv_rmse:.3f} ± {cv_std:.3f}")
        
        return performance
    
    def predict_score(self, game_state: GameState, alliance: str = "red") -> ScoreOptimizationResult:
        """Predict score and provide optimization insights"""
        if self.ensemble_model is None:
            raise ValueError("Models not trained. Call train_models() first.")
        
        # Extract features
        features = self.feature_extractor.extract_all_features(game_state, alliance)
        feature_vector = self._extract_feature_vector(features)
        feature_vector_scaled = self.scaler.transform([feature_vector])
        
        # Make prediction
        predicted_score = self.ensemble_model.predict(feature_vector_scaled)[0]
        
        # Calculate confidence interval using individual model predictions
        xgb_pred = self.xgb_model.predict(feature_vector_scaled)[0]
        rf_pred = self.rf_model.predict(feature_vector_scaled)[0]
        
        predictions = [xgb_pred, rf_pred]
        pred_std = np.std(predictions)
        confidence_interval = (
            predicted_score - 1.96 * pred_std,
            predicted_score + 1.96 * pred_std
        )
        
        # SHAP feature contributions
        feature_contributions = {}
        if self.shap_explainer is not None:
            try:
                shap_values = self.shap_explainer(feature_vector_scaled)
                for i, feature_name in enumerate(self.feature_names):
                    feature_contributions[feature_name] = float(shap_values.values[0][i])
            except:
                # Fallback to feature importance
                for feature_name, importance in zip(self.feature_names, feature_vector):
                    feature_contributions[feature_name] = float(importance * 0.1)  # Simplified
        
        # Generate optimization suggestions
        suggestions = self._generate_optimization_suggestions(features, feature_contributions)
        
        # Risk assessment
        risk = self._assess_risk(predicted_score, confidence_interval, features)
        
        # Win probability estimation
        win_prob = self._estimate_win_probability(predicted_score, features)
        
        return ScoreOptimizationResult(
            predicted_score=float(predicted_score),
            confidence_interval=confidence_interval,
            feature_contributions=feature_contributions,
            optimization_suggestions=suggestions,
            risk_assessment=risk,
            expected_win_probability=win_prob
        )
    
    def _generate_optimization_suggestions(
        self, 
        features: Dict[str, float], 
        contributions: Dict[str, float]
    ) -> List[str]:
        """Generate actionable optimization suggestions"""
        suggestions = []
        
        # Sort contributions by absolute value
        sorted_contributions = sorted(
            contributions.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        # Top negative contributors (areas for improvement)
        negative_contributors = [
            (name, value) for name, value in sorted_contributions[:10] 
            if value < -0.5
        ]
        
        for feature_name, contribution in negative_contributors:
            if 'blocks' in feature_name:
                suggestions.append(f"Increase {feature_name.replace('_', ' ')} scoring")
            elif 'coordination' in feature_name:
                suggestions.append("Improve robot coordination and positioning")
            elif 'saturation' in feature_name:
                suggestions.append(f"Focus on filling {feature_name.split('_')[0]} goals more effectively")
            elif 'zone' in feature_name:
                suggestions.append("Enhance zone control strategy")
            elif 'momentum' in feature_name:
                suggestions.append("Maintain consistent scoring throughout match")
        
        # Specific feature-based suggestions
        alliance_prefix = "red_"  # Assuming red alliance analysis
        
        if features.get(f'{alliance_prefix}robot_coordination_distance', 0) > 15:
            suggestions.append("Reduce distance between robots for better coordination")
        
        if features.get(f'{alliance_prefix}task_allocation_efficiency', 0) < 0.7:
            suggestions.append("Improve task allocation between robots")
        
        if features.get(f'{alliance_prefix}scoring_rate_10s', 0) < 2:
            suggestions.append("Increase scoring rate through faster cycles")
        
        if features.get(f'{alliance_prefix}average_goal_saturation', 0) < 0.4:
            suggestions.append("Focus on completing goals before moving to new ones")
        
        if features.get(f'{alliance_prefix}defensive_positioning', 0) < 0.1:
            suggestions.append("Consider adding defensive positioning elements")
        
        # Remove duplicates and limit to top suggestions
        unique_suggestions = list(dict.fromkeys(suggestions))
        return unique_suggestions[:5]
    
    def _assess_risk(
        self, 
        predicted_score: float, 
        confidence_interval: Tuple[float, float], 
        features: Dict[str, float]
    ) -> str:
        """Assess risk level of the strategy"""
        uncertainty = confidence_interval[1] - confidence_interval[0]
        
        # Risk factors
        risk_factors = []
        
        if uncertainty > 30:
            risk_factors.append("High prediction uncertainty")
        
        if predicted_score < 80:
            risk_factors.append("Low predicted score")
        
        if features.get('red_score_momentum', 0) < -1:
            risk_factors.append("Negative scoring momentum")
        
        if features.get('red_task_allocation_efficiency', 0) < 0.5:
            risk_factors.append("Poor coordination")
        
        if len(risk_factors) == 0:
            return "Low Risk"
        elif len(risk_factors) <= 2:
            return f"Medium Risk: {', '.join(risk_factors)}"
        else:
            return f"High Risk: {', '.join(risk_factors[:2])}, and others"
    
    def _estimate_win_probability(self, predicted_score: float, features: Dict[str, float]) -> float:
        """Estimate probability of winning based on predicted score and features"""
        # Simplified win probability based on score relative to average
        average_score = 120  # Typical VEX U score
        score_ratio = predicted_score / average_score
        
        # Adjust based on key strategic factors
        coordination_factor = features.get('red_task_allocation_efficiency', 0.5)
        momentum_factor = max(0, features.get('red_score_momentum', 0)) / 2
        consistency_factor = 1 - abs(features.get('red_average_goal_saturation', 0.5) - 0.5)
        
        # Combine factors
        base_prob = min(0.95, score_ratio * 0.5)  # Cap at 95%
        adjusted_prob = base_prob + (coordination_factor * 0.2) + (momentum_factor * 0.1) + (consistency_factor * 0.1)
        
        return max(0.05, min(0.95, adjusted_prob))  # Bound between 5% and 95%
    
    def analyze_feature_importance(self) -> Dict[str, Any]:
        """Comprehensive feature importance analysis"""
        if self.xgb_model is None:
            raise ValueError("Models not trained yet")
        
        # XGBoost feature importance
        xgb_importance = dict(zip(self.feature_names, self.xgb_model.feature_importances_))
        
        # Random Forest feature importance
        rf_importance = dict(zip(self.feature_names, self.rf_model.feature_importances_))
        
        # Combined importance (average)
        combined_importance = {}
        for feature in self.feature_names:
            combined_importance[feature] = (xgb_importance[feature] + rf_importance[feature]) / 2
        
        # Sort by importance
        sorted_importance = sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'xgb_importance': xgb_importance,
            'rf_importance': rf_importance,
            'combined_importance': combined_importance,
            'top_features': sorted_importance[:10],
            'bottom_features': sorted_importance[-10:]
        }
    
    def create_shap_plots(self, save_dir: str = "plots"):
        """Create SHAP visualization plots"""
        if self.shap_values is None:
            print("SHAP values not available. Run training first.")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, feature_names=self.feature_names, show=False)
        plt.title("SHAP Feature Importance Summary")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "shap_summary.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Waterfall plot for first prediction
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(self.shap_values[0], show=False)
        plt.title("SHAP Waterfall Plot - Single Prediction")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "shap_waterfall.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Bar plot
        plt.figure(figsize=(10, 8))
        shap.plots.bar(self.shap_values, show=False)
        plt.title("SHAP Feature Importance Bar Plot")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "shap_bar.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"SHAP plots saved to {save_dir}/")
    
    def save_models(self):
        """Save all trained models and components"""
        if self.xgb_model is not None:
            self.xgb_model.save_model(self.xgb_path)
        
        if self.rf_model is not None:
            joblib.dump(self.rf_model, self.rf_path)
        
        if self.ensemble_model is not None:
            joblib.dump(self.ensemble_model, self.ensemble_path)
        
        joblib.dump(self.scaler, self.scaler_path)
        
        # Save configuration
        config = {
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Models saved to {self.model_dir}/")
    
    def load_models(self) -> bool:
        """Load pre-trained models"""
        try:
            if os.path.exists(self.xgb_path):
                self.xgb_model = xgb.XGBRegressor()
                self.xgb_model.load_model(self.xgb_path)
            
            if os.path.exists(self.rf_path):
                self.rf_model = joblib.load(self.rf_path)
            
            if os.path.exists(self.ensemble_path):
                self.ensemble_model = joblib.load(self.ensemble_path)
            
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
            
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.feature_names = config['feature_names']
            
            # Reinitialize SHAP explainer if XGB model exists
            if self.xgb_model is not None:
                print("Models loaded successfully")
                return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
        
        print("No saved models found")
        return False


def train_scoring_optimizer_pipeline(num_samples: int = 3000, optimize_hyperparams: bool = True) -> VEXUScoringOptimizer:
    """Complete training pipeline for the scoring optimizer"""
    optimizer = VEXUScoringOptimizer()
    
    # Generate training data
    X, y = optimizer.generate_training_data(num_samples)
    
    # Train models
    performance = optimizer.train_models(X, y, optimize_hyperparameters=optimize_hyperparams)
    
    print(f"\nTraining Results:")
    print(f"RMSE: {performance.rmse:.3f}")
    print(f"MAE: {performance.mae:.3f}")
    print(f"R² Score: {performance.r2_score:.3f}")
    print(f"CV RMSE: {performance.cv_score:.3f} ± {performance.cv_std:.3f}")
    
    # Analyze feature importance
    importance_analysis = optimizer.analyze_feature_importance()
    print(f"\nTop 5 Most Important Features:")
    for feature, importance in importance_analysis['top_features'][:5]:
        print(f"  {feature}: {importance:.3f}")
    
    return optimizer


if __name__ == "__main__":
    print("VEX U Scoring Optimizer - Training Example")
    print("=" * 50)
    
    # Train the optimizer
    optimizer = train_scoring_optimizer_pipeline(num_samples=1000, optimize_hyperparams=False)
    
    # Test prediction
    test_strategy = AllianceStrategy(
        name="Test Strategy",
        blocks_scored_auto={"long_1": 5, "long_2": 5, "center_1": 3, "center_2": 3},
        blocks_scored_driver={"long_1": 12, "long_2": 10, "center_1": 8, "center_2": 6},
        zones_controlled=[Zone.RED_HOME, Zone.NEUTRAL],
        robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.PLATFORM]
    )
    
    opponent_strategy = AllianceStrategy(
        name="Opponent",
        blocks_scored_auto={"long_1": 3, "long_2": 4, "center_1": 2, "center_2": 2},
        blocks_scored_driver={"long_1": 8, "long_2": 9, "center_1": 5, "center_2": 6},
        zones_controlled=[Zone.BLUE_HOME],
        robots_parked=[ParkingLocation.ALLIANCE_ZONE, ParkingLocation.PLATFORM]
    )
    
    game_state = create_game_state_from_strategy(test_strategy, opponent_strategy)
    
    # Make prediction
    result = optimizer.predict_score(game_state, "red")
    
    print(f"\nOptimization Results:")
    print(f"Predicted Score: {result.predicted_score:.1f}")
    print(f"Confidence Interval: ({result.confidence_interval[0]:.1f}, {result.confidence_interval[1]:.1f})")
    print(f"Win Probability: {result.expected_win_probability:.1%}")
    print(f"Risk Assessment: {result.risk_assessment}")
    
    print(f"\nTop Optimization Suggestions:")
    for i, suggestion in enumerate(result.optimization_suggestions, 1):
        print(f"  {i}. {suggestion}")
    
    print(f"\nTop Feature Contributions:")
    sorted_contributions = sorted(
        result.feature_contributions.items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )
    for feature, contribution in sorted_contributions[:5]:
        print(f"  {feature}: {contribution:+.2f}")