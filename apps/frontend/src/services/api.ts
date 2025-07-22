import axios, { AxiosInstance, AxiosResponse } from 'axios';
import {
  ApiResponse,
  AnalysisResult,
  AllianceStrategy,
  Match,
  ScenarioGenerationParams,
  SystemStatus,
  ReportData,
  CompetitionData,
  PerformanceMetrics,
  MLModelStatus,
} from '../types';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

class ApiService {
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
        // Add auth token if available
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
        console.error('API Error:', error);
        return Promise.reject(error);
      }
    );
  }

  // System Status and Health
  async getSystemStatus(): Promise<SystemStatus> {
    const response = await this.client.get<ApiResponse<SystemStatus>>('/api/system/health');
    return this.handleResponse(response);
  }

  async getMLModelStatus(): Promise<MLModelStatus> {
    const response = await this.client.get<ApiResponse<MLModelStatus>>('/api/ml/status');
    return this.handleResponse(response);
  }

  // Analysis Operations
  async runAnalysis(type: string, params?: any): Promise<AnalysisResult> {
    const response = await this.client.post<ApiResponse<AnalysisResult>>(
      `/api/analysis/${type}`,
      params
    );
    return this.handleResponse(response);
  }

  async getAnalysisHistory(): Promise<AnalysisResult[]> {
    const response = await this.client.get<ApiResponse<AnalysisResult[]>>('/api/analysis/history');
    return this.handleResponse(response);
  }

  async getAnalysisById(id: string): Promise<AnalysisResult> {
    const response = await this.client.get<ApiResponse<AnalysisResult>>(`/api/analysis/${id}`);
    return this.handleResponse(response);
  }

  // Strategy Operations
  async generateStrategy(params: any): Promise<AllianceStrategy> {
    const response = await this.client.post<ApiResponse<AllianceStrategy>>(
      '/api/strategies/generate',
      params
    );
    return this.handleResponse(response);
  }

  async optimizeStrategy(strategy: AllianceStrategy): Promise<AllianceStrategy> {
    const response = await this.client.post<ApiResponse<AllianceStrategy>>(
      '/api/strategies/optimize',
      strategy
    );
    return this.handleResponse(response);
  }

  async getStrategies(): Promise<AllianceStrategy[]> {
    const response = await this.client.get<ApiResponse<AllianceStrategy[]>>('/api/strategies/');
    return this.handleResponse(response);
  }

  async saveStrategy(strategy: AllianceStrategy): Promise<AllianceStrategy> {
    const response = await this.client.post<ApiResponse<AllianceStrategy>>(
      '/api/strategies/',
      strategy
    );
    return this.handleResponse(response);
  }

  // Simulation Operations
  async runSimulation(params: any): Promise<Match[]> {
    const response = await this.client.post<ApiResponse<Match[]>>('/api/scenarios/simulate', params);
    return this.handleResponse(response);
  }

  async generateScenarios(params: ScenarioGenerationParams): Promise<Match[]> {
    const response = await this.client.post<ApiResponse<Match[]>>(
      '/api/scenarios/generate',
      params
    );
    return this.handleResponse(response);
  }

  // ML Model Operations
  async trainMLModel(modelType: string, params?: any): Promise<any> {
    const response = await this.client.post<ApiResponse<any>>(
      '/api/ml/train',
      { 
        model_type: modelType,
        ...params 
      }
    );
    return this.handleResponse(response);
  }

  async getMLPrediction(modelType: string, data: any): Promise<any> {
    const response = await this.client.post<ApiResponse<any>>(
      '/api/ml/predict',
      { modelType, data }
    );
    return this.handleResponse(response);
  }

  async optimizeScoring(data: any): Promise<any> {
    const response = await this.client.post<ApiResponse<any>>('/api/ml/optimize', data);
    return this.handleResponse(response);
  }

  async discoverPatterns(data: any): Promise<any> {
    const response = await this.client.post<ApiResponse<any>>('/api/ml/patterns', data);
    return this.handleResponse(response);
  }

  // Reporting Operations
  async generateReport(type: string, params?: any): Promise<ReportData> {
    const response = await this.client.post<ApiResponse<ReportData>>(
      '/api/reports/generate',
      { type, params }
    );
    return this.handleResponse(response);
  }

  async getReports(): Promise<ReportData[]> {
    const response = await this.client.get<ApiResponse<ReportData[]>>('/api/reports');
    return this.handleResponse(response);
  }

  async getReportById(id: string): Promise<ReportData> {
    const response = await this.client.get<ApiResponse<ReportData>>(`/api/reports/${id}`);
    return this.handleResponse(response);
  }

  async exportReport(id: string, format: 'pdf' | 'html' | 'json'): Promise<Blob> {
    const response = await this.client.get(`/api/reports/${id}/export`, {
      params: { format },
      responseType: 'blob',
    });
    return response.data;
  }

  // Data Management
  async uploadCompetitionData(file: File): Promise<CompetitionData> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.client.post<ApiResponse<CompetitionData>>(
      '/api/data/upload',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return this.handleResponse(response);
  }

  async getCompetitionData(): Promise<CompetitionData[]> {
    const response = await this.client.get<ApiResponse<CompetitionData[]>>('/api/data/competitions');
    return this.handleResponse(response);
  }

  async getPerformanceMetrics(teamId?: string): Promise<PerformanceMetrics> {
    const response = await this.client.get<ApiResponse<PerformanceMetrics>>(
      '/api/data/metrics',
      { params: { teamId } }
    );
    return this.handleResponse(response);
  }

  // Visualization Data
  async getVisualizationData(type: string, params?: any): Promise<any> {
    const response = await this.client.get<ApiResponse<any>>('/api/visualizations/data', {
      params: { type, ...params },
    });
    return this.handleResponse(response);
  }

  async getInteractiveDashboardData(): Promise<any> {
    const response = await this.client.get<ApiResponse<any>>('/api/visualizations/dashboard');
    return this.handleResponse(response);
  }

  // Utility Methods
  private handleResponse<T>(response: AxiosResponse<ApiResponse<T>>): T {
    const { data } = response;
    
    if (!data.success) {
      throw new Error(data.error || 'API request failed');
    }
    
    if (data.data === undefined) {
      throw new Error('No data received from API');
    }
    
    return data.data;
  }

  // Health check endpoint
  async healthCheck(): Promise<boolean> {
    try {
      await this.client.get('/health');
      return true;
    } catch {
      return false;
    }
  }
}

// Create and export singleton instance
export const apiService = new ApiService();
export default apiService;