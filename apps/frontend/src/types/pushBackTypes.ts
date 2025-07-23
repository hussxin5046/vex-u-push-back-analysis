// Push Back specific TypeScript types

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface RobotSpecs {
  id?: string;
  cycle_time: number; // Seconds per scoring cycle
  pickup_reliability: number; // 0.0 - 1.0
  scoring_reliability: number; // 0.0 - 1.0
  autonomous_reliability: number; // 0.0 - 1.0
  max_capacity: number; // Number of blocks robot can hold
  parking_capability: boolean;
  speed?: number; // 0.0 - 1.0 (legacy)
  accuracy?: number; // 0.0 - 1.0 (legacy)
  driver_skill?: number;
  reliability?: number; // Overall reliability (legacy)
}

export interface PushBackBlock {
  id: string;
  x: number;
  y: number;
  z?: number;
  alliance: 'red' | 'blue' | 'neutral';
  goal_id?: string;
  zone_id?: string;
}

export interface PushBackGoal {
  id: string;
  x: number;
  y: number;
  goal_type: 'center' | 'long';
  alliance: 'red' | 'blue';
  blocks: PushBackBlock[];
}

export interface PushBackControlZone {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  alliance: 'red' | 'blue';
  blocks: PushBackBlock[];
  controlled_by: 'red' | 'blue' | 'neutral';
}

export interface PushBackParkZone {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  alliance: 'red' | 'blue';
  robots_parked: number;
}

export interface PushBackFieldState {
  blocks: PushBackBlock[];
  goals: PushBackGoal[];
  control_zones: PushBackControlZone[];
  park_zones: PushBackParkZone[];
  time_remaining: number;
  match_phase: 'autonomous' | 'driver' | 'endgame';
}

export interface PushBackStrategy {
  id: string;
  name: string;
  description?: string;
  strategy_type?: string;
  archetype?: string;
  robot_specs: RobotSpecs[];
  goal_priorities?: {
    center: number;
    long: number;
  };
  autonomous_strategy: string;
  parking_strategy: string;
  driver_strategy?: string;
  endgame_strategy?: string;
  priority_sequence?: string[];
  created_at: string;
  updated_at: string;
}

export interface PushBackMatchState {
  field_state: PushBackFieldState;
  red_score: number;
  blue_score: number;
  red_breakdown?: Record<string, number>;
  blue_breakdown?: Record<string, number>;
  autonomous_completed: boolean;
  auto_win_achieved: 'red' | 'blue' | 'none';
}

export interface BlockFlowOptimization {
  optimal_distribution: Record<string, number>;
  expected_points: number;
  risk_level: 'low' | 'medium' | 'high';
  efficiency_score: number;
  recommendations: string[];
  bottlenecks?: string[];
}

export interface AutonomousDecision {
  recommended_strategy: string;
  auto_win_probability: number;
  bonus_probability: number;
  expected_points: number;
  block_targets: Record<string, number>;
  risk_assessment: string;
  time_allocation?: Record<string, number>;
}

export interface GoalPriorityAnalysis {
  recommended_priority: string;
  center_goal_value: number;
  long_goal_value: number;
  optimal_sequence: string[];
  decision_confidence: number;
  matchup_considerations?: Record<string, string>;
}

export interface ParkingDecisionAnalysis {
  recommended_timing: string;
  one_robot_threshold: number;
  two_robot_threshold: number;
  expected_value: number;
  risk_benefit_ratio: number;
  situational_recommendations: Record<string, string>;
}

export interface OffenseDefenseBalance {
  recommended_ratio: [number, number];
  offensive_roi: number;
  defensive_roi: number;
  critical_zones: string[];
  disruption_targets: string[];
  phase_recommendations?: Record<string, string>;
}

export interface PushBackAnalysisResult {
  analysis_id: string;
  strategy: PushBackStrategy;
  robot_specs: RobotSpecs[];
  block_flow_optimization: BlockFlowOptimization;
  autonomous_decision: AutonomousDecision;
  goal_priority_analysis: GoalPriorityAnalysis;
  parking_decision_analysis: ParkingDecisionAnalysis;
  offense_defense_balance: OffenseDefenseBalance;
  recommended_archetype: string;
  overall_score: number;
  recommendations: string[];
  created_at: string;
}

export interface MonteCarloSimulationResult {
  strategy_id: string;
  num_simulations: number;
  win_rate: number;
  avg_score: number;
  score_std: number;
  scoring_breakdown: Record<string, number>;
  opponent_matchups: Record<string, number>;
  performance_confidence: number;
  risk_metrics?: Record<string, number>;
}

export interface DecisionAnalysis {
  decision_type: string;
  recommendation: string;
  confidence: number;
  reasoning: string[];
  alternatives: Array<{
    option: string;
    score: number;
    pros: string[];
    cons: string[];
  }>;
}

// Strategy Archetypes
export type StrategyArchetype = 
  | 'block_flow_maximizer'
  | 'control_zone_controller'
  | 'goal_rush_specialist'
  | 'parking_strategist'
  | 'autonomous_specialist'
  | 'balanced_competitor'
  | 'defensive_disruptor';

export interface ArchetypeDefinition {
  id: StrategyArchetype;
  name: string;
  description: string;
  focus: string;
  risk_level: 'low' | 'medium' | 'high';
  complexity: 'low' | 'medium' | 'high';
  strengths: string[];
  weaknesses: string[];
  best_for: string[];
}

// Navigation and UI types
export interface NavigationItem {
  id: string;
  label: string;
  path: string;
  icon?: string;
  children?: NavigationItem[];
}

// WebSocket Event Types
export interface WebSocketEvent {
  type: string;
  timestamp: string;
  data: any;
}

export interface PushBackAnalysisProgress extends WebSocketEvent {
  analysis_id: string;
  stage: string;
  progress: number;
  current_step?: string;
  partial_results?: any;
}

export interface StrategyOptimizationUpdate extends WebSocketEvent {
  strategy_id: string;
  optimization_type: string;
  progress: number;
  current_values?: any;
}

export interface MonteCarloProgress extends WebSocketEvent {
  simulation_id: string;
  completed_simulations: number;
  total_simulations: number;
  progress: number;
  current_stats?: any;
}

export interface FieldStateUpdate extends WebSocketEvent {
  match_id: string;
  field_state: PushBackFieldState;
  scores?: {
    red_score: number;
    blue_score: number;
  };
}

export interface ScoreUpdate extends WebSocketEvent {
  match_id: string;
  red_score: number;
  blue_score: number;
  red_breakdown?: Record<string, number>;
  blue_breakdown?: Record<string, number>;
}

// API Request/Response Types
export interface CreateStrategyRequest {
  name: string;
  archetype: StrategyArchetype;
  robot_specs: RobotSpecs[];
  autonomous_strategy: string;
  driver_strategy: string;
  endgame_strategy?: string;
  priority_sequence?: string[];
}

export interface AnalyzeStrategyRequest {
  strategy: PushBackStrategy;
  robot_specs: RobotSpecs[];
}

export interface OptimizeBlockFlowRequest {
  robot_specs: RobotSpecs[];
  constraints?: any;
}

export interface MonteCarloRequest {
  strategy: PushBackStrategy;
  num_simulations?: number;
  opponent_types?: string[];
}

export interface CalculateScoreRequest {
  field_state: PushBackFieldState;
}

export interface ParkingCalculatorRequest {
  current_score: number;
  opponent_score: number;
  time_remaining: number;
  robot_specs: RobotSpecs[];
}

// System Status Types
export interface PushBackSystemStatus {
  backend_status: string;
  analysis_engine_status: string;
  available_features: string[];
  version: string;
}

// Error Types
export interface PushBackError {
  code: string;
  message: string;
  details?: any;
}

// Visualization Types
export interface FieldVisualizationData {
  field_layout: any;
  block_positions: any[];
  control_zones: Record<string, any>;
  scoring_visualization: any;
}

export interface TimelineData {
  autonomous_timeline: Array<{ time: number; event: string; action: string }>;
  driver_timeline: Array<{ time: number; event: string; action: string }>;
  score_progression: Array<{ time: number; red_score: number; blue_score: number }>;
  key_events: Array<{ time: number; event: string; alliance: string }>;
}

export interface RiskRewardData {
  scatter_data: Array<{
    name: string;
    risk: number;
    reward: number;
    archetype: string;
  }>;
  risk_categories: string[];
  reward_categories: string[];
  strategy_clusters: Record<string, any[]>;
}