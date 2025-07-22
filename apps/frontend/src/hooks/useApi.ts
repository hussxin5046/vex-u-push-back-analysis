import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import apiService from '../services/api';
import {
  AnalysisResult,
  AllianceStrategy,
  ScenarioGenerationParams,
  SystemStatus,
  ReportData,
  CompetitionData,
  PerformanceMetrics,
  MLModelStatus,
} from '../types';

// Query Keys
export const queryKeys = {
  systemStatus: ['systemStatus'] as const,
  mlModelStatus: ['mlModelStatus'] as const,
  analysisHistory: ['analysisHistory'] as const,
  analysis: (id: string) => ['analysis', id] as const,
  strategies: ['strategies'] as const,
  reports: ['reports'] as const,
  report: (id: string) => ['report', id] as const,
  competitionData: ['competitionData'] as const,
  performanceMetrics: (teamId?: string) => ['performanceMetrics', teamId] as const,
  visualizationData: (type: string, params?: any) => ['visualizationData', type, params] as const,
  dashboardData: ['dashboardData'] as const,
};

// System Status Hooks
export const useSystemStatus = () => {
  return useQuery({
    queryKey: queryKeys.systemStatus,
    queryFn: () => apiService.getSystemStatus(),
    refetchInterval: 30000, // Refresh every 30 seconds
    staleTime: 10000, // Consider fresh for 10 seconds
  });
};

export const useMLModelStatus = () => {
  return useQuery({
    queryKey: queryKeys.mlModelStatus,
    queryFn: async () => {
      try {
        return await apiService.getMLModelStatus();
      } catch (error) {
        console.warn('ML model status fetch failed:', error);
        return null;
      }
    },
    refetchInterval: 60000, // Refresh every minute
    retry: false,
  });
};

// Analysis Hooks
export const useAnalysisHistory = () => {
  return useQuery({
    queryKey: queryKeys.analysisHistory,
    queryFn: async (): Promise<AnalysisResult[]> => {
      try {
        return await apiService.getAnalysisHistory();
      } catch (error) {
        console.warn('Analysis history fetch failed:', error);
        return [];
      }
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
    retry: false,
  });
};

export const useAnalysis = (id: string) => {
  return useQuery({
    queryKey: queryKeys.analysis(id),
    queryFn: () => apiService.getAnalysisById(id),
    enabled: !!id,
  });
};

export const useRunAnalysis = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ type, params }: { type: string; params?: any }) =>
      apiService.runAnalysis(type, params),
    onSuccess: () => {
      // Invalidate analysis history to refetch
      queryClient.invalidateQueries({ queryKey: queryKeys.analysisHistory });
    },
  });
};

// Strategy Hooks
export const useStrategies = () => {
  return useQuery({
    queryKey: queryKeys.strategies,
    queryFn: () => apiService.getStrategies(),
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
};

export const useGenerateStrategy = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (params: any) => apiService.generateStrategy(params),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.strategies });
    },
  });
};

export const useOptimizeStrategy = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (strategy: AllianceStrategy) => apiService.optimizeStrategy(strategy),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.strategies });
    },
  });
};

export const useSaveStrategy = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (strategy: AllianceStrategy) => apiService.saveStrategy(strategy),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.strategies });
    },
  });
};

// Simulation Hooks
export const useRunSimulation = () => {
  return useMutation({
    mutationFn: (params: any) => apiService.runSimulation(params),
  });
};

export const useGenerateScenarios = () => {
  return useMutation({
    mutationFn: (params: ScenarioGenerationParams) => apiService.generateScenarios(params),
  });
};

// ML Model Hooks
export const useTrainMLModel = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ modelType, params }: { modelType: string; params?: any }) =>
      apiService.trainMLModel(modelType, params),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.mlModelStatus });
    },
  });
};

export const useMLPrediction = () => {
  return useMutation({
    mutationFn: ({ modelType, data }: { modelType: string; data: any }) =>
      apiService.getMLPrediction(modelType, data),
  });
};

export const useOptimizeScoring = () => {
  return useMutation({
    mutationFn: (data: any) => apiService.optimizeScoring(data),
  });
};

export const useDiscoverPatterns = () => {
  return useMutation({
    mutationFn: (data: any) => apiService.discoverPatterns(data),
  });
};

// Reporting Hooks
export const useReports = () => {
  return useQuery({
    queryKey: queryKeys.reports,
    queryFn: () => apiService.getReports(),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};

export const useReport = (id: string) => {
  return useQuery({
    queryKey: queryKeys.report(id),
    queryFn: () => apiService.getReportById(id),
    enabled: !!id,
  });
};

export const useGenerateReport = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ type, params }: { type: string; params?: any }) =>
      apiService.generateReport(type, params),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.reports });
    },
  });
};

export const useExportReport = () => {
  return useMutation({
    mutationFn: ({ id, format }: { id: string; format: 'pdf' | 'html' | 'json' }) =>
      apiService.exportReport(id, format),
  });
};

// Data Management Hooks
export const useCompetitionData = () => {
  return useQuery({
    queryKey: queryKeys.competitionData,
    queryFn: () => apiService.getCompetitionData(),
    staleTime: 15 * 60 * 1000, // 15 minutes
  });
};

export const useUploadCompetitionData = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (file: File) => apiService.uploadCompetitionData(file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.competitionData });
    },
  });
};

export const usePerformanceMetrics = (teamId?: string) => {
  return useQuery({
    queryKey: queryKeys.performanceMetrics(teamId),
    queryFn: () => apiService.getPerformanceMetrics(teamId),
    enabled: true,
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
};

// Visualization Hooks
export const useVisualizationData = (type: string, params?: any) => {
  return useQuery({
    queryKey: queryKeys.visualizationData(type, params),
    queryFn: () => apiService.getVisualizationData(type, params),
    enabled: !!type,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};

export const useDashboardData = () => {
  return useQuery({
    queryKey: queryKeys.dashboardData,
    queryFn: () => apiService.getInteractiveDashboardData(),
    refetchInterval: 2 * 60 * 1000, // Refresh every 2 minutes
    staleTime: 60 * 1000, // 1 minute
  });
};

// Health Check Hook
export const useHealthCheck = () => {
  return useQuery({
    queryKey: ['healthCheck'],
    queryFn: () => apiService.healthCheck(),
    refetchInterval: 10000, // Every 10 seconds
    retry: false,
    staleTime: 5000,
  });
};