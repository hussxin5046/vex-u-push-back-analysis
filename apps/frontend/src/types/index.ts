// VEX U Scoring Analysis Types
export interface Robot {
  id: string;
  name: string;
  role: 'offense' | 'defense' | 'support';
  efficiency: number;
  autonomousScore: number;
  driverScore: number;
}

export interface AllianceStrategy {
  id: string;
  name: string;
  robots: Robot[];
  autonomousStrategy: string;
  driverStrategy: string;
  expectedScore: number;
  winProbability: number;
  riskLevel: 'low' | 'medium' | 'high';
}

export interface Match {
  id: string;
  matchNumber: number;
  redAlliance: AllianceStrategy;
  blueAlliance: AllianceStrategy;
  winner?: 'red' | 'blue' | 'tie';
  finalScore?: {
    red: number;
    blue: number;
  };
  timestamp: string;
}

export interface AnalysisResult {
  id: string;
  type: 'scoring' | 'strategy' | 'statistical' | 'ml_prediction';
  title: string;
  summary: string;
  data: any;
  charts?: ChartData[];
  recommendations?: string[];
  createdAt: string;
}

export interface ChartData {
  id: string;
  type: 'line' | 'bar' | 'pie' | 'scatter' | 'area';
  title: string;
  data: any[];
  xAxis?: string;
  yAxis?: string;
  options?: any;
}

export interface ScenarioGenerationParams {
  numScenarios: number;
  complexityLevel: 'basic' | 'intermediate' | 'advanced';
  includeMLPredictions: boolean;
  focusAreas: string[];
  timeConstraints?: {
    autonomous: number;
    driver: number;
  };
}

export interface MLModelStatus {
  coordination: boolean;
  scoring_optimizer: boolean;
  strategy_predictor: boolean;
  feature_engineering: boolean;
}

export interface SystemStatus {
  backend_connected: boolean;
  ml_models: MLModelStatus;
  last_updated: string;
  version: string;
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp: string;
}

export interface StatisticalMetrics {
  mean: number;
  median: number;
  std: number;
  min: number;
  max: number;
  percentiles: {
    p25: number;
    p75: number;
    p90: number;
    p95: number;
  };
}

export interface PerformanceMetrics {
  winRate: number;
  averageScore: number;
  scoreConsistency: StatisticalMetrics;
  autonomousEfficiency: number;
  driverEfficiency: number;
  defensiveRating: number;
  offensiveRating: number;
}

export interface CompetitionData {
  id: string;
  name: string;
  date: string;
  location: string;
  teams: string[];
  matches: Match[];
  rankings?: TeamRanking[];
}

export interface TeamRanking {
  rank: number;
  team: string;
  wins: number;
  losses: number;
  ties: number;
  totalPoints: number;
  averagePoints: number;
  highestScore: number;
}

export interface ReportData {
  id: string;
  title: string;
  type: 'strategy' | 'statistical' | 'ml_insights' | 'comprehensive';
  summary: string;
  sections: ReportSection[];
  charts: ChartData[];
  recommendations: string[];
  metadata: {
    generated_at: string;
    data_range: string;
    confidence_level: number;
  };
}

export interface ReportSection {
  id: string;
  title: string;
  content: string;
  charts?: string[]; // Chart IDs
  importance: 'high' | 'medium' | 'low';
}

// Navigation and UI Types
export interface NavigationItem {
  id: string;
  label: string;
  path: string;
  icon?: string;
  children?: NavigationItem[];
}

export interface ThemeMode {
  mode: 'light' | 'dark';
}

export interface LoadingState {
  isLoading: boolean;
  operation?: string;
}

export interface NotificationConfig {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message?: string;
  duration?: number;
  actions?: NotificationAction[];
}

export interface NotificationAction {
  label: string;
  action: () => void;
  variant?: 'primary' | 'secondary';
}