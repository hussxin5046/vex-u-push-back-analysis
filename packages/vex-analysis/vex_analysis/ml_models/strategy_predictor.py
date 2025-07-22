import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import json
import os
from datetime import datetime

try:
    from .feature_engineering import VEXUFeatureExtractor, GameState, RobotState, MatchPhase
    from ..core.simulator import AllianceStrategy, ScoringSimulator, Zone, ParkingLocation
    from ..core.scenario_generator import ScenarioGenerator, SkillLevel, StrategyType, RobotRole
except ImportError:
    # Fallback for when running from main.py
    from src.ml_models.feature_engineering import VEXUFeatureExtractor, GameState, RobotState, MatchPhase
    from src.core.simulator import AllianceStrategy, ScoringSimulator, Zone, ParkingLocation
    from src.core.scenario_generator import ScenarioGenerator, SkillLevel, StrategyType, RobotRole

warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class StrategyPrediction:
    predicted_strategy: str
    confidence: float
    strategy_probabilities: Dict[str, float]
    robot1_role: str
    robot2_role: str
    expected_score: float
    feature_importance: Dict[str, float]
    recommended_adjustments: List[str]


@dataclass
class TrainingMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    loss: float
    validation_accuracy: float
    training_time: float
    epoch_count: int


class VEXUStrategyPredictor:
    def __init__(self, model_name: str = "vex_u_strategy_predictor"):
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_extractor = VEXUFeatureExtractor()
        self.simulator = ScoringSimulator(enable_feature_extraction=True)
        self.scenario_generator = ScenarioGenerator(self.simulator)
        
        # Strategy types for classification
        self.strategy_types = [
            "all_offense", "mixed", "zone_control", 
            "defensive", "autonomous_focus"
        ]
        
        # Model architecture parameters
        self.input_dim = 60  # Expected number of features
        self.hidden_layers = [128, 64, 32]
        self.dropout_rate = 0.3
        self.l2_reg = 0.001
        
        # Training parameters
        self.batch_size = 32
        self.epochs = 100
        self.learning_rate = 0.001
        self.validation_split = 0.2
        
        # Model paths
        self.model_dir = "models"
        self.model_path = os.path.join(self.model_dir, f"{model_name}.h5")
        self.scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.pkl")
        self.encoder_path = os.path.join(self.model_dir, f"{model_name}_encoder.pkl")
        self.config_path = os.path.join(self.model_dir, f"{model_name}_config.json")
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        
    def _build_neural_network(self, input_dim: int, output_dim: int) -> keras.Model:
        # Input layer
        inputs = layers.Input(shape=(input_dim,), name='features_input')
        
        # Hidden layers
        x = layers.Dense(
            self.hidden_layers[0], 
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name='hidden_1'
        )(inputs)
        x = layers.Dropout(self.dropout_rate, name='dropout_1')(x)
        x = layers.BatchNormalization(name='batch_norm_1')(x)
        
        x = layers.Dense(
            self.hidden_layers[1], 
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name='hidden_2'
        )(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_2')(x)
        x = layers.BatchNormalization(name='batch_norm_2')(x)
        
        x = layers.Dense(
            self.hidden_layers[2], 
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name='hidden_3'
        )(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_3')(x)
        
        # Output layers for multi-task learning
        strategy_output = layers.Dense(output_dim, activation='softmax', name='strategy')(x)
        robot1_role_output = layers.Dense(len(RobotRole), activation='softmax', name='robot1_role')(x)
        robot2_role_output = layers.Dense(len(RobotRole), activation='softmax', name='robot2_role')(x)
        
        # Create multi-output model
        model = models.Model(
            inputs=inputs,
            outputs={
                'strategy': strategy_output,
                'robot1_role': robot1_role_output,
                'robot2_role': robot2_role_output
            }
        )
        
        return model
    
    def generate_synthetic_training_data(self, num_samples: int = 5000) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        print(f"Generating {num_samples} synthetic training samples...")
        
        features_list = []
        strategy_labels = []
        robot1_role_labels = []
        robot2_role_labels = []
        
        for i in range(num_samples):
            if i % 500 == 0:
                print(f"Generated {i}/{num_samples} samples...")
            
            # Generate random scenario parameters
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
            
            # Create game state for feature extraction
            opponent = self.scenario_generator.generate_random_strategy("Opponent")
            game_state = self._create_game_state_from_strategies(strategy, opponent)
            
            # Extract features
            features = self.feature_extractor.extract_all_features(game_state, "red")
            
            # Ensure consistent feature vector length
            feature_vector = self._standardize_feature_vector(features)
            
            if len(feature_vector) == self.input_dim:
                features_list.append(feature_vector)
                strategy_labels.append(strategy_type.value)
                robot1_role_labels.append(scenario_params.robot1_role.value)
                robot2_role_labels.append(scenario_params.robot2_role.value)
        
        print(f"Successfully generated {len(features_list)} samples")
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y = {
            'strategy': np.array(strategy_labels),
            'robot1_role': np.array(robot1_role_labels),
            'robot2_role': np.array(robot2_role_labels)
        }
        
        return X, y
    
    def _standardize_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        expected_features = [
            # Scoring features
            'red_total_score', 'red_score_differential', 'red_blocks_points',
            'red_autonomous_points', 'red_zone_control_points', 'red_parking_points',
            
            # Goal-specific features
            'red_long_1_blocks', 'red_long_2_blocks', 
            'red_center_upper_blocks', 'red_center_lower_blocks',
            'red_long_1_saturation', 'red_long_2_saturation',
            'red_center_upper_saturation', 'red_center_lower_saturation',
            
            # VEX U specific features
            'red_robot_coordination_distance', 'red_size_diversity',
            'red_task_allocation_efficiency', 'red_small_robot_utilization',
            'red_large_robot_utilization', 'red_is_autonomous',
            'red_is_driver_control', 'red_match_progress', 'red_time_remaining',
            
            # Temporal features
            'red_scoring_rate_10s', 'red_time_since_last_score',
            'red_score_momentum', 'red_zone_control_duration',
            
            # Strategic features
            'red_average_goal_saturation', 'red_neutral_zone_presence',
            'red_home_zone_presence', 'red_defensive_positioning',
            'red_block_carrying_efficiency', 'red_high_value_goal_focus',
            
            # Additional derived features
            'red_zones_controlled_count', 'red_parked_robots',
            'red_auto_time_remaining', 'red_is_endgame',
        ]
        
        # Pad to exactly input_dim features if needed
        while len(expected_features) < self.input_dim:
            expected_features.append(f'red_feature_{len(expected_features)}')
        
        expected_features = expected_features[:self.input_dim]
        
        feature_vector = []
        for feature_name in expected_features:
            feature_vector.append(features.get(feature_name, 0.0))
        
        return np.array(feature_vector, dtype=np.float32)
    
    def _create_game_state_from_strategies(
        self, 
        red_strategy: AllianceStrategy, 
        blue_strategy: AllianceStrategy,
        match_time: float = 60.0
    ) -> GameState:
        game_state = GameState(
            match_time=match_time,
            phase=MatchPhase.DRIVER_CONTROL,
            red_score=0,
            blue_score=0
        )
        
        # Create robot states
        game_state.red_robots = [
            RobotState("red_1", "24_inch", (2.0, 2.0), Zone.RED_HOME),
            RobotState("red_2", "15_inch", (3.0, 2.0), Zone.RED_HOME)
        ]
        
        game_state.blue_robots = [
            RobotState("blue_1", "24_inch", (10.0, 10.0), Zone.BLUE_HOME),
            RobotState("blue_2", "15_inch", (9.0, 10.0), Zone.BLUE_HOME)
        ]
        
        # Set blocks based on strategies
        for goal in ["long_1", "long_2", "center_1", "center_2"]:
            red_blocks = (red_strategy.blocks_scored_auto.get(goal, 0) + 
                         red_strategy.blocks_scored_driver.get(goal, 0))
            blue_blocks = (blue_strategy.blocks_scored_auto.get(goal, 0) + 
                          blue_strategy.blocks_scored_driver.get(goal, 0))
            
            # Map to game state format
            if goal == "center_1":
                game_state.blocks_in_goals["red_center_upper"] = red_blocks
                game_state.blocks_in_goals["blue_center_upper"] = blue_blocks
            elif goal == "center_2":
                game_state.blocks_in_goals["red_center_lower"] = red_blocks
                game_state.blocks_in_goals["blue_center_lower"] = blue_blocks
            else:
                game_state.blocks_in_goals[f"red_{goal}"] = red_blocks
                game_state.blocks_in_goals[f"blue_{goal}"] = blue_blocks
        
        return game_state
    
    def train_model(self, X: np.ndarray, y: Dict[str, np.ndarray], 
                   validation_data: Optional[Tuple] = None) -> TrainingMetrics:
        print("Training VEX U Strategy Predictor...")
        start_time = datetime.now()
        
        # Encode labels
        y_strategy_encoded = self.label_encoder.fit_transform(y['strategy'])
        y_strategy_categorical = keras.utils.to_categorical(y_strategy_encoded)
        
        # Encode robot roles
        role_encoder1 = LabelEncoder()
        role_encoder2 = LabelEncoder()
        
        y_robot1_encoded = role_encoder1.fit_transform(y['robot1_role'])
        y_robot2_encoded = role_encoder2.fit_transform(y['robot2_role'])
        
        y_robot1_categorical = keras.utils.to_categorical(y_robot1_encoded)
        y_robot2_categorical = keras.utils.to_categorical(y_robot2_encoded)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        if validation_data is None:
            X_train, X_val, y_strat_train, y_strat_val = train_test_split(
                X_scaled, y_strategy_categorical, 
                test_size=self.validation_split, random_state=42
            )
            
            y_r1_train, y_r1_val = train_test_split(
                y_robot1_categorical, test_size=self.validation_split, random_state=42
            )[0], train_test_split(
                y_robot1_categorical, test_size=self.validation_split, random_state=42
            )[1]
            
            y_r2_train, y_r2_val = train_test_split(
                y_robot2_categorical, test_size=self.validation_split, random_state=42
            )[0], train_test_split(
                y_robot2_categorical, test_size=self.validation_split, random_state=42
            )[1]
        else:
            X_train, X_val = X_scaled, validation_data[0]
            y_strat_train, y_strat_val = y_strategy_categorical, validation_data[1]['strategy']
            y_r1_train, y_r1_val = y_robot1_categorical, validation_data[1]['robot1_role']
            y_r2_train, y_r2_val = y_robot2_categorical, validation_data[1]['robot2_role']
        
        # Build model
        self.model = self._build_neural_network(
            X_train.shape[1], 
            len(np.unique(y['strategy']))  # Use actual number of unique strategies
        )
        
        # Compile model with multiple losses
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss={
                'strategy': 'categorical_crossentropy',
                'robot1_role': 'categorical_crossentropy',
                'robot2_role': 'categorical_crossentropy'
            },
            loss_weights={
                'strategy': 1.0,
                'robot1_role': 0.5,
                'robot2_role': 0.5
            },
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6
            ),
            callbacks.ModelCheckpoint(
                self.model_path,
                monitor='val_strategy_accuracy',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train,
            {
                'strategy': y_strat_train,
                'robot1_role': y_r1_train,
                'robot2_role': y_r2_train
            },
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(
                X_val, 
                {
                    'strategy': y_strat_val,
                    'robot1_role': y_r1_val,
                    'robot2_role': y_r2_val
                }
            ),
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Calculate training metrics
        training_time = (datetime.now() - start_time).total_seconds()
        final_accuracy = max(history.history['strategy_accuracy'])
        final_val_accuracy = max(history.history['val_strategy_accuracy'])
        final_loss = min(history.history['loss'])
        
        # Save model components
        self.save_model()
        
        metrics = TrainingMetrics(
            accuracy=final_accuracy,
            precision=0.0,  # Would need additional calculation
            recall=0.0,     # Would need additional calculation
            f1_score=0.0,   # Would need additional calculation
            loss=final_loss,
            validation_accuracy=final_val_accuracy,
            training_time=training_time,
            epoch_count=len(history.history['loss'])
        )
        
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Final accuracy: {final_accuracy:.3f}")
        print(f"Final validation accuracy: {final_val_accuracy:.3f}")
        
        return metrics
    
    def predict_strategy(self, game_state: GameState, alliance: str = "red") -> StrategyPrediction:
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train_model() or load_model() first.")
        
        # Extract features
        features = self.feature_extractor.extract_all_features(game_state, alliance)
        feature_vector = self._standardize_feature_vector(features)
        feature_vector_scaled = self.scaler.transform([feature_vector])
        
        # Make predictions
        predictions = self.model.predict(feature_vector_scaled, verbose=0)
        
        strategy_probs = predictions['strategy'][0]
        robot1_role_probs = predictions['robot1_role'][0]
        robot2_role_probs = predictions['robot2_role'][0]
        
        # Get predicted classes
        predicted_strategy_idx = np.argmax(strategy_probs)
        predicted_strategy = self.label_encoder.inverse_transform([predicted_strategy_idx])[0]
        confidence = float(strategy_probs[predicted_strategy_idx])
        
        # Get robot roles
        robot1_role = list(RobotRole)[np.argmax(robot1_role_probs)].value
        robot2_role = list(RobotRole)[np.argmax(robot2_role_probs)].value
        
        # Create strategy probabilities dictionary
        strategy_probabilities = {}
        for i, strategy in enumerate(self.strategy_types):
            strategy_probabilities[strategy] = float(strategy_probs[i])
        
        # Calculate feature importance (simplified)
        feature_importance = self._calculate_feature_importance(feature_vector)
        
        # Generate recommendations
        recommendations = self._generate_strategy_recommendations(
            predicted_strategy, confidence, features
        )
        
        # Estimate expected score
        expected_score = self._estimate_expected_score(features)
        
        return StrategyPrediction(
            predicted_strategy=predicted_strategy,
            confidence=confidence,
            strategy_probabilities=strategy_probabilities,
            robot1_role=robot1_role,
            robot2_role=robot2_role,
            expected_score=expected_score,
            feature_importance=feature_importance,
            recommended_adjustments=recommendations
        )
    
    def _calculate_feature_importance(self, feature_vector: np.ndarray) -> Dict[str, float]:
        # Simplified feature importance based on magnitude
        feature_names = [
            'total_score', 'score_differential', 'blocks_points',
            'autonomous_points', 'zone_control_points', 'parking_points',
            'coordination_distance', 'task_efficiency', 'scoring_rate',
            'momentum', 'saturation', 'positioning'
        ]
        
        # Pad feature names to match vector length
        while len(feature_names) < len(feature_vector):
            feature_names.append(f'feature_{len(feature_names)}')
        
        importances = {}
        total_magnitude = np.sum(np.abs(feature_vector))
        
        for i, (name, value) in enumerate(zip(feature_names, feature_vector)):
            if total_magnitude > 0:
                importances[name] = abs(value) / total_magnitude
            else:
                importances[name] = 0.0
        
        return importances
    
    def _generate_strategy_recommendations(
        self, 
        predicted_strategy: str, 
        confidence: float, 
        features: Dict[str, float]
    ) -> List[str]:
        recommendations = []
        
        if confidence < 0.7:
            recommendations.append("Consider hybrid strategy approach due to low prediction confidence")
        
        if features.get('red_scoring_rate_10s', 0) < 1.0:
            recommendations.append("Increase scoring rate through better robot coordination")
        
        if features.get('red_average_goal_saturation', 0) < 0.3:
            recommendations.append("Focus on filling specific goals before moving to others")
        
        if features.get('red_defensive_positioning', 0) < 0.1 and predicted_strategy != 'all_offense':
            recommendations.append("Consider adding defensive positioning")
        
        if features.get('red_zone_control_duration', 0) < 30 and predicted_strategy in ['zone_control', 'mixed']:
            recommendations.append("Improve zone control consistency")
        
        if features.get('red_robot_coordination_distance', 0) > 15:
            recommendations.append("Improve robot coordination - robots are too far apart")
        
        return recommendations
    
    def _estimate_expected_score(self, features: Dict[str, float]) -> float:
        # Simple scoring estimation based on key features
        base_score = features.get('red_blocks_points', 0)
        auto_bonus = features.get('red_autonomous_points', 0)
        zone_points = features.get('red_zone_control_points', 0)
        parking_points = features.get('red_parking_points', 0)
        
        # Add momentum and efficiency bonuses
        momentum_bonus = features.get('red_score_momentum', 0) * 10
        efficiency_bonus = features.get('red_task_allocation_efficiency', 0) * 20
        
        expected_score = base_score + auto_bonus + zone_points + parking_points + momentum_bonus + efficiency_bonus
        return max(0, expected_score)
    
    def evaluate_strategy_effectiveness(
        self, 
        strategy: AllianceStrategy, 
        num_simulations: int = 100
    ) -> Dict[str, float]:
        scores = []
        win_count = 0
        
        for _ in range(num_simulations):
            opponent = self.scenario_generator.generate_random_strategy("Opponent")
            result = self.simulator.simulate_match(strategy, opponent)
            
            scores.append(result.red_score)
            if result.winner == "red":
                win_count += 1
        
        return {
            'average_score': np.mean(scores),
            'score_std': np.std(scores),
            'win_rate': win_count / num_simulations,
            'max_score': max(scores),
            'min_score': min(scores),
            'effectiveness_rating': (np.mean(scores) * win_count / num_simulations) / 100
        }
    
    def real_time_prediction_during_match(
        self, 
        match_states: List[GameState], 
        alliance: str = "red"
    ) -> List[StrategyPrediction]:
        predictions = []
        
        for state in match_states:
            prediction = self.predict_strategy(state, alliance)
            predictions.append(prediction)
        
        return predictions
    
    def save_model(self):
        if self.model is not None:
            self.model.save(self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            joblib.dump(self.label_encoder, self.encoder_path)
            
            # Save configuration
            config = {
                'strategy_types': self.strategy_types,
                'input_dim': self.input_dim,
                'hidden_layers': self.hidden_layers,
                'dropout_rate': self.dropout_rate,
                'model_name': self.model_name
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = keras.models.load_model(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.label_encoder = joblib.load(self.encoder_path)
            
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                self.strategy_types = config['strategy_types']
                self.input_dim = config['input_dim']
            
            print(f"Model loaded from {self.model_path}")
            return True
        else:
            print(f"No saved model found at {self.model_path}")
            return False
    
    def get_model_summary(self) -> str:
        if self.model is None:
            return "Model not initialized"
        
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        return '\n'.join(summary_lines)


def train_strategy_predictor_pipeline(num_samples: int = 5000) -> VEXUStrategyPredictor:
    """Complete training pipeline for the strategy predictor"""
    predictor = VEXUStrategyPredictor()
    
    # Generate training data
    X, y = predictor.generate_synthetic_training_data(num_samples)
    
    # Train model
    metrics = predictor.train_model(X, y)
    
    print(f"\nTraining Results:")
    print(f"Accuracy: {metrics.accuracy:.3f}")
    print(f"Validation Accuracy: {metrics.validation_accuracy:.3f}")
    print(f"Training Time: {metrics.training_time:.2f} seconds")
    print(f"Epochs: {metrics.epoch_count}")
    
    return predictor


if __name__ == "__main__":
    # Example usage
    print("VEX U Strategy Predictor - Training Example")
    print("=" * 50)
    
    # Train the model
    predictor = train_strategy_predictor_pipeline(num_samples=1000)  # Smaller sample for testing
    
    # Test prediction
    from .feature_engineering import create_game_state_from_strategy
    
    # Create test scenario
    test_strategy = AllianceStrategy(
        name="Test Strategy",
        blocks_scored_auto={"long_1": 5, "long_2": 5, "center_1": 3, "center_2": 3},
        blocks_scored_driver={"long_1": 10, "long_2": 8, "center_1": 6, "center_2": 7},
        zones_controlled=[Zone.RED_HOME, Zone.NEUTRAL],
        robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.PLATFORM]
    )
    
    opponent_strategy = AllianceStrategy(
        name="Opponent",
        blocks_scored_auto={"long_1": 4, "long_2": 4, "center_1": 2, "center_2": 2},
        blocks_scored_driver={"long_1": 8, "long_2": 9, "center_1": 5, "center_2": 6},
        zones_controlled=[Zone.BLUE_HOME],
        robots_parked=[ParkingLocation.ALLIANCE_ZONE, ParkingLocation.PLATFORM]
    )
    
    game_state = create_game_state_from_strategy(test_strategy, opponent_strategy)
    
    # Make prediction
    prediction = predictor.predict_strategy(game_state, "red")
    
    print(f"\nPrediction Results:")
    print(f"Predicted Strategy: {prediction.predicted_strategy}")
    print(f"Confidence: {prediction.confidence:.3f}")
    print(f"Robot 1 Role: {prediction.robot1_role}")
    print(f"Robot 2 Role: {prediction.robot2_role}")
    print(f"Expected Score: {prediction.expected_score:.1f}")
    print(f"Top Recommendations:")
    for rec in prediction.recommended_adjustments[:3]:
        print(f"  • {rec}")
    
    print("\nStrategy Probabilities:")
    for strategy, prob in prediction.strategy_probabilities.items():
        print(f"  {strategy}: {prob:.3f}")