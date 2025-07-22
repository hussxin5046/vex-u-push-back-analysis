import axios, { AxiosInstance, AxiosResponse } from 'axios';
import {
  ApiResponse,
  PushBackStrategy,
  PushBackAnalysisResult,
  PushBackMatchState,
  PushBackFieldState,
  RobotSpecs,
  DecisionAnalysis,
} from '../types/pushBackTypes';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

class PushBackApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('authToken');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('Push Back API Error:', error);
        return Promise.reject(error);
      }
    );
  }

  // Strategy Operations
  async createStrategy(strategy: Partial<PushBackStrategy>): Promise<PushBackStrategy> {
    const response = await this.client.post<ApiResponse<PushBackStrategy>>('/api/push-back/strategies', strategy);
    return this.handleResponse(response);
  }

  async updateStrategy(id: string, strategy: Partial<PushBackStrategy>): Promise<PushBackStrategy> {
    const response = await this.client.put<ApiResponse<PushBackStrategy>>(`/api/push-back/strategies/${id}`, strategy);
    return this.handleResponse(response);
  }

  async getStrategies(): Promise<PushBackStrategy[]> {
    const response = await this.client.get<ApiResponse<PushBackStrategy[]>>('/api/push-back/strategies');
    return this.handleResponse(response);
  }

  async deleteStrategy(id: string): Promise<boolean> {
    await this.client.delete(`/api/push-back/strategies/${id}`);
    return true;
  }

  // Strategy Analysis
  async analyzeStrategy(strategy: PushBackStrategy, robotSpecs: RobotSpecs[]): Promise<PushBackAnalysisResult> {
    const response = await this.client.post<ApiResponse<PushBackAnalysisResult>>('/api/push-back/analyze', {
      strategy,
      robot_specs: robotSpecs
    });
    return this.handleResponse(response);
  }

  async compareStrategies(strategies: PushBackStrategy[]): Promise<PushBackAnalysisResult[]> {
    const response = await this.client.post<ApiResponse<PushBackAnalysisResult[]>>('/api/push-back/compare', {
      strategies
    });
    return this.handleResponse(response);
  }

  // Block Flow Optimization
  async optimizeBlockFlow(robotSpecs: RobotSpecs[], constraints?: any): Promise<{
    optimal_distribution: Record<string, number>;
    expected_points: number;
    risk_level: number;
    recommendations: string[];
  }> {
    const response = await this.client.post<ApiResponse<any>>('/api/push-back/optimize/block-flow', {
      robot_specs: robotSpecs,
      constraints
    });
    return this.handleResponse(response);
  }

  // Autonomous Decision Analysis
  async analyzeAutonomousDecision(robotSpecs: RobotSpecs[]): Promise<{
    recommended_strategy: string;
    auto_win_probability: number;
    bonus_probability: number;
    expected_points: number;
    block_targets: Record<string, number>;
    risk_assessment: string;
  }> {
    const response = await this.client.post<ApiResponse<any>>('/api/push-back/analyze/autonomous', {
      robot_specs: robotSpecs
    });
    return this.handleResponse(response);
  }

  // Goal Priority Analysis
  async analyzeGoalPriority(
    robotSpecs: RobotSpecs[], 
    opponentStrategy: string = 'balanced',
    matchPhase: string = 'early'
  ): Promise<{
    recommended_priority: string;
    center_goal_value: number;
    long_goal_value: number;
    optimal_sequence: string[];
    decision_confidence: number;
  }> {
    const response = await this.client.post<ApiResponse<any>>('/api/push-back/analyze/goal-priority', {
      robot_specs: robotSpecs,
      opponent_strategy: opponentStrategy,
      match_phase: matchPhase
    });
    return this.handleResponse(response);
  }

  // Parking Decision Analysis
  async analyzeParkingDecision(matchState: PushBackMatchState, robotSpecs: RobotSpecs[]): Promise<{
    recommended_timing: string;
    one_robot_threshold: number;
    two_robot_threshold: number;
    expected_value: number;
    risk_benefit_ratio: number;
    situational_recommendations: Record<string, string>;
  }> {
    const response = await this.client.post<ApiResponse<any>>('/api/push-back/analyze/parking', {
      match_state: matchState,
      robot_specs: robotSpecs
    });
    return this.handleResponse(response);
  }

  // Offense/Defense Balance Analysis
  async analyzeOffenseDefenseBalance(matchState: PushBackMatchState, robotSpecs: RobotSpecs[]): Promise<{
    recommended_ratio: [number, number];
    offensive_roi: number;
    defensive_roi: number;
    critical_zones: string[];
    disruption_targets: string[];
  }> {
    const response = await this.client.post<ApiResponse<any>>('/api/push-back/analyze/offense-defense', {
      match_state: matchState,
      robot_specs: robotSpecs
    });
    return this.handleResponse(response);
  }

  // Comprehensive Analysis
  async runComprehensiveAnalysis(
    robotSpecs: RobotSpecs[],
    matchContext?: any
  ): Promise<{
    block_flow: any;
    autonomous_decision: any;
    goal_priority: any;
    parking_decision: any;
    offense_defense: any;
    recommended_archetype: string;
    recommendations: string[];
  }> {
    const response = await this.client.post<ApiResponse<any>>('/api/push-back/analyze/comprehensive', {
      robot_specs: robotSpecs,
      match_context: matchContext
    });
    return this.handleResponse(response);
  }

  // Monte Carlo Simulation
  async runMonteCarloSimulation(
    strategy: PushBackStrategy,
    numSimulations: number = 1000,
    opponentTypes?: string[]
  ): Promise<{
    win_rate: number;
    avg_score: number;
    score_std: number;
    scoring_breakdown: Record<string, any>;
    opponent_matchups: Record<string, any>;
    performance_confidence: number;
  }> {
    const response = await this.client.post<ApiResponse<any>>('/api/push-back/simulate/monte-carlo', {
      strategy,
      num_simulations: numSimulations,
      opponent_types: opponentTypes
    });
    return this.handleResponse(response);
  }

  // Field State and Scoring
  async calculateScore(fieldState: PushBackFieldState): Promise<{
    red_score: number;
    blue_score: number;
    red_breakdown: Record<string, number>;
    blue_breakdown: Record<string, number>;
    control_zones: Record<string, string>;
  }> {
    const response = await this.client.post<ApiResponse<any>>('/api/push-back/score/calculate', {
      field_state: fieldState
    });
    return this.handleResponse(response);
  }

  // Strategy Archetypes
  async getStrategyArchetypes(): Promise<Record<string, PushBackStrategy>> {
    const response = await this.client.get<ApiResponse<Record<string, PushBackStrategy>>>('/api/push-back/archetypes');
    return this.handleResponse(response);
  }

  async getArchetypeRecommendation(
    robotSpecs: RobotSpecs[],
    matchContext?: any
  ): Promise<{
    recommended_archetype: string;
    archetype_scores: Record<string, number>;
    reasoning: string[];
  }> {
    const response = await this.client.post<ApiResponse<any>>('/api/push-back/archetypes/recommend', {
      robot_specs: robotSpecs,
      match_context: matchContext
    });
    return this.handleResponse(response);
  }

  // Decision Support Tools
  async getParkingCalculator(
    currentScore: number,
    opponentScore: number,
    timeRemaining: number,
    robotSpecs: RobotSpecs[]
  ): Promise<{
    parking_recommendations: Record<string, any>;
    break_even_points: Record<string, number>;
    risk_analysis: Record<string, number>;
  }> {
    const response = await this.client.post<ApiResponse<any>>('/api/push-back/tools/parking-calculator', {
      current_score: currentScore,
      opponent_score: opponentScore,
      time_remaining: timeRemaining,
      robot_specs: robotSpecs
    });
    return this.handleResponse(response);
  }

  async getControlZoneOptimizer(
    currentBlocks: Record<string, number>,
    availableBlocks: number,
    robotSpecs: RobotSpecs[]
  ): Promise<{
    optimal_additions: Record<string, number>;
    expected_control_gain: number;
    efficiency_score: number;
    recommendations: string[];
  }> {
    const response = await this.client.post<ApiResponse<any>>('/api/push-back/tools/control-zone-optimizer', {
      current_blocks: currentBlocks,
      available_blocks: availableBlocks,
      robot_specs: robotSpecs
    });
    return this.handleResponse(response);
  }

  async getAutonomousPlanner(robotSpecs: RobotSpecs[]): Promise<{
    strategy_options: Record<string, any>;
    time_allocations: Record<string, Record<string, number>>;
    risk_assessments: Record<string, string>;
    recommendations: string[];
  }> {
    const response = await this.client.post<ApiResponse<any>>('/api/push-back/tools/autonomous-planner', {
      robot_specs: robotSpecs
    });
    return this.handleResponse(response);
  }

  // Visualization Data
  async getFieldVisualizationData(fieldState: PushBackFieldState): Promise<{
    field_layout: any;
    block_positions: any[];
    control_zones: Record<string, any>;
    scoring_visualization: any;
  }> {
    const response = await this.client.post<ApiResponse<any>>('/api/push-back/visualize/field', {
      field_state: fieldState
    });
    return this.handleResponse(response);
  }

  async getMatchTimelineData(strategy: PushBackStrategy): Promise<{
    autonomous_timeline: any[];
    driver_timeline: any[];
    score_progression: any[];
    key_events: any[];
  }> {
    const response = await this.client.post<ApiResponse<any>>('/api/push-back/visualize/timeline', {
      strategy
    });
    return this.handleResponse(response);
  }

  async getRiskRewardData(strategies: PushBackStrategy[]): Promise<{
    scatter_data: any[];
    risk_categories: string[];
    reward_categories: string[];
    strategy_clusters: Record<string, any[]>;
  }> {
    const response = await this.client.post<ApiResponse<any>>('/api/push-back/visualize/risk-reward', {
      strategies
    });
    return this.handleResponse(response);
  }

  // System Status
  async healthCheck(): Promise<boolean> {
    try {
      await this.client.get('/health');
      return true;
    } catch {
      return false;
    }
  }

  async getPushBackSystemStatus(): Promise<{
    backend_status: string;
    analysis_engine_status: string;
    available_features: string[];
    version: string;
  }> {
    const response = await this.client.get<ApiResponse<any>>('/api/push-back/system/status');
    return this.handleResponse(response);
  }

  // Utility Methods
  private handleResponse<T>(response: AxiosResponse<ApiResponse<T>>): T {
    const { data } = response;
    
    if (!data.success) {
      throw new Error(data.error || 'Push Back API request failed');
    }
    
    if (data.data === undefined) {
      throw new Error('No data received from Push Back API');
    }
    
    return data.data;
  }
}

// Create and export singleton instance
export const pushBackApiService = new PushBackApiService();
export default pushBackApiService;