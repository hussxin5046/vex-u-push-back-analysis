import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  FormControl,
  FormLabel,
  FormGroup,
  FormControlLabel,
  Checkbox,
  Slider,
  TextField,
  LinearProgress,
  Chip,
  Alert,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Grid,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Download as DownloadIcon,
  Assessment as ChartIcon,
  Psychology as StrategyIcon,
} from '@mui/icons-material';
import { useNotification } from '../contexts/NotificationContext';
import { AllianceStrategy, AnalysisResult, ChartData } from '../types';

interface StrategyAnalysisParams {
  strategies: string[];
  simulationCount: number;
  complexityLevel: 'basic' | 'intermediate' | 'advanced';
  timeConstraints: {
    autonomous: number;
    driver: number;
  };
  focusMetrics: string[];
}

interface AnalysisProgress {
  current: number;
  total: number;
  currentStrategy: string;
  stage: string;
}

const StrategyAnalysis: React.FC = () => {
  const [parameters, setParameters] = useState<StrategyAnalysisParams>({
    strategies: [],
    simulationCount: 1000,
    complexityLevel: 'intermediate',
    timeConstraints: {
      autonomous: 15,
      driver: 105,
    },
    focusMetrics: ['win_probability', 'average_score'],
  });

  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState<AnalysisProgress | null>(null);
  const [results, setResults] = useState<AnalysisResult | null>(null);
  const { showNotification } = useNotification();

  const availableStrategies = [
    { id: 'offensive_rush', name: 'Offensive Rush', description: 'High-speed scoring strategy' },
    { id: 'defensive_control', name: 'Defensive Control', description: 'Territory control and blocking' },
    { id: 'balanced_approach', name: 'Balanced Approach', description: 'Mixed offense and defense' },
    { id: 'autonomous_focus', name: 'Autonomous Focus', description: 'Maximize autonomous points' },
    { id: 'endgame_specialist', name: 'Endgame Specialist', description: 'Focus on endgame scoring' },
    { id: 'support_coordination', name: 'Support Coordination', description: 'Alliance coordination strategy' },
  ];

  const focusMetrics = [
    { id: 'win_probability', name: 'Win Probability' },
    { id: 'average_score', name: 'Average Score' },
    { id: 'score_consistency', name: 'Score Consistency' },
    { id: 'autonomous_efficiency', name: 'Autonomous Efficiency' },
    { id: 'driver_efficiency', name: 'Driver Efficiency' },
    { id: 'risk_assessment', name: 'Risk Assessment' },
  ];

  const handleStrategyChange = (strategyId: string, checked: boolean) => {
    setParameters(prev => ({
      ...prev,
      strategies: checked
        ? [...prev.strategies, strategyId]
        : prev.strategies.filter(s => s !== strategyId),
    }));
  };

  const handleMetricChange = (metricId: string, checked: boolean) => {
    setParameters(prev => ({
      ...prev,
      focusMetrics: checked
        ? [...prev.focusMetrics, metricId]
        : prev.focusMetrics.filter(m => m !== metricId),
    }));
  };

  const handleSimulationCountChange = (_: Event, value: number | number[]) => {
    setParameters(prev => ({
      ...prev,
      simulationCount: value as number,
    }));
  };

  const simulateAnalysis = async () => {
    if (parameters.strategies.length === 0) {
      showNotification({
        type: 'warning',
        title: 'No Strategies Selected',
        message: 'Please select at least one strategy to analyze.',
      });
      return;
    }

    setIsRunning(true);
    setProgress({ current: 0, total: parameters.simulationCount, currentStrategy: '', stage: 'Initializing' });

    try {
      // Simulate progressive analysis
      for (let i = 0; i <= parameters.simulationCount; i += Math.floor(parameters.simulationCount / 20)) {
        const strategyIndex = Math.floor((i / parameters.simulationCount) * parameters.strategies.length);
        const currentStrategy = availableStrategies.find(s => s.id === parameters.strategies[strategyIndex])?.name || '';
        
        setProgress({
          current: Math.min(i, parameters.simulationCount),
          total: parameters.simulationCount,
          currentStrategy,
          stage: i < parameters.simulationCount ? 'Running Simulations' : 'Generating Results',
        });

        await new Promise(resolve => setTimeout(resolve, 200));
      }

      // Generate mock results
      const mockResults: AnalysisResult = {
        id: `analysis_${Date.now()}`,
        type: 'strategy',
        title: `Strategy Analysis - ${parameters.strategies.length} Strategies`,
        summary: `Analyzed ${parameters.strategies.length} strategies across ${parameters.simulationCount} simulations`,
        data: {
          strategies: parameters.strategies.map(strategyId => {
            const strategy = availableStrategies.find(s => s.id === strategyId);
            return {
              id: strategyId,
              name: strategy?.name,
              winProbability: Math.random() * 0.6 + 0.2,
              averageScore: Math.floor(Math.random() * 50 + 100),
              consistency: Math.random() * 0.3 + 0.7,
              riskLevel: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)],
            };
          }),
        },
        charts: [
          {
            id: 'win_probability_chart',
            type: 'bar',
            title: 'Win Probability by Strategy',
            data: [],
          },
          {
            id: 'score_distribution_chart',
            type: 'line',
            title: 'Score Distribution',
            data: [],
          },
        ],
        recommendations: [
          'Consider focusing on strategies with higher consistency scores',
          'Balance risk vs. reward based on competition context',
          'Optimize autonomous period performance for selected strategies',
        ],
        createdAt: new Date().toISOString(),
      };

      setResults(mockResults);
      showNotification({
        type: 'success',
        title: 'Analysis Complete',
        message: 'Strategy analysis has been completed successfully.',
      });
    } catch (error) {
      showNotification({
        type: 'error',
        title: 'Analysis Failed',
        message: 'There was an error running the strategy analysis.',
      });
    } finally {
      setIsRunning(false);
      setProgress(null);
    }
  };

  const stopAnalysis = () => {
    setIsRunning(false);
    setProgress(null);
    showNotification({
      type: 'info',
      title: 'Analysis Stopped',
      message: 'Strategy analysis has been stopped.',
    });
  };

  const exportResults = () => {
    if (!results) return;
    
    const dataStr = JSON.stringify(results, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `strategy_analysis_${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 700 }}>
        Strategy Analysis
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Analyze and compare different VEX U strategies through comprehensive simulations.
      </Typography>

      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 2fr' }, gap: 3 }}>
        {/* Configuration Panel */}
        <Box>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                <StrategyIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                Analysis Configuration
              </Typography>

              {/* Strategy Selection */}
              <FormControl component="fieldset" sx={{ mt: 2, width: '100%' }}>
                <FormLabel component="legend">Select Strategies</FormLabel>
                <FormGroup>
                  {availableStrategies.map((strategy) => (
                    <FormControlLabel
                      key={strategy.id}
                      control={
                        <Checkbox
                          checked={parameters.strategies.includes(strategy.id)}
                          onChange={(e) => handleStrategyChange(strategy.id, e.target.checked)}
                        />
                      }
                      label={
                        <Box>
                          <Typography variant="body2">{strategy.name}</Typography>
                          <Typography variant="caption" color="text.secondary">
                            {strategy.description}
                          </Typography>
                        </Box>
                      }
                    />
                  ))}
                </FormGroup>
              </FormControl>

              {/* Simulation Count */}
              <Box sx={{ mt: 3 }}>
                <Typography gutterBottom>
                  Simulation Count: {parameters.simulationCount}
                </Typography>
                <Slider
                  value={parameters.simulationCount}
                  onChange={handleSimulationCountChange}
                  min={100}
                  max={10000}
                  step={100}
                  marks={[
                    { value: 100, label: '100' },
                    { value: 1000, label: '1K' },
                    { value: 5000, label: '5K' },
                    { value: 10000, label: '10K' },
                  ]}
                  disabled={isRunning}
                />
              </Box>

              {/* Focus Metrics */}
              <FormControl component="fieldset" sx={{ mt: 3, width: '100%' }}>
                <FormLabel component="legend">Focus Metrics</FormLabel>
                <FormGroup>
                  {focusMetrics.map((metric) => (
                    <FormControlLabel
                      key={metric.id}
                      control={
                        <Checkbox
                          checked={parameters.focusMetrics.includes(metric.id)}
                          onChange={(e) => handleMetricChange(metric.id, e.target.checked)}
                        />
                      }
                      label={metric.name}
                    />
                  ))}
                </FormGroup>
              </FormControl>

              {/* Action Buttons */}
              <Box sx={{ mt: 3, display: 'flex', gap: 1 }}>
                {!isRunning ? (
                  <Button
                    variant="contained"
                    startIcon={<PlayIcon />}
                    onClick={simulateAnalysis}
                    disabled={parameters.strategies.length === 0}
                    fullWidth
                  >
                    Start Analysis
                  </Button>
                ) : (
                  <Button
                    variant="outlined"
                    startIcon={<StopIcon />}
                    onClick={stopAnalysis}
                    color="error"
                    fullWidth
                  >
                    Stop Analysis
                  </Button>
                )}
              </Box>
            </CardContent>
          </Card>
        </Box>

        {/* Progress and Results Panel */}
        <Box>
          {/* Progress Indicator */}
          {isRunning && progress && (
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Analysis Progress
                </Typography>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    {progress.stage}: {progress.currentStrategy}
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={(progress.current / progress.total) * 100}
                    sx={{ mt: 1 }}
                  />
                  <Typography variant="caption" color="text.secondary">
                    {progress.current} / {progress.total} simulations
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          )}

          {/* Results Display */}
          {results && !isRunning && (
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    <ChartIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                    Analysis Results
                  </Typography>
                  <Button
                    startIcon={<DownloadIcon />}
                    onClick={exportResults}
                    variant="outlined"
                    size="small"
                  >
                    Export
                  </Button>
                </Box>

                <Alert severity="info" sx={{ mb: 3 }}>
                  {results.summary}
                </Alert>

                {/* Strategy Comparison Table */}
                <TableContainer component={Paper} variant="outlined">
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Strategy</TableCell>
                        <TableCell align="right">Win Probability</TableCell>
                        <TableCell align="right">Avg Score</TableCell>
                        <TableCell align="right">Consistency</TableCell>
                        <TableCell align="center">Risk Level</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {results.data.strategies.map((strategy: any) => (
                        <TableRow key={strategy.id}>
                          <TableCell component="th" scope="row">
                            {strategy.name}
                          </TableCell>
                          <TableCell align="right">
                            {(strategy.winProbability * 100).toFixed(1)}%
                          </TableCell>
                          <TableCell align="right">{strategy.averageScore}</TableCell>
                          <TableCell align="right">
                            {(strategy.consistency * 100).toFixed(1)}%
                          </TableCell>
                          <TableCell align="center">
                            <Chip
                              label={strategy.riskLevel}
                              color={
                                strategy.riskLevel === 'low'
                                  ? 'success'
                                  : strategy.riskLevel === 'medium'
                                  ? 'warning'
                                  : 'error'
                              }
                              size="small"
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>

                {/* Recommendations */}
                {results.recommendations && results.recommendations.length > 0 && (
                  <Box sx={{ mt: 3 }}>
                    <Typography variant="h6" gutterBottom>
                      Recommendations
                    </Typography>
                    {results.recommendations.map((rec, index) => (
                      <Alert key={index} severity="info" sx={{ mb: 1 }}>
                        {rec}
                      </Alert>
                    ))}
                  </Box>
                )}
              </CardContent>
            </Card>
          )}

          {/* Empty State */}
          {!results && !isRunning && (
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 8 }}>
                <StrategyIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                <Typography variant="h6" color="text.secondary" gutterBottom>
                  No Analysis Results Yet
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Configure your analysis parameters and start a strategy analysis to see results here.
                </Typography>
              </CardContent>
            </Card>
          )}
        </Box>
      </Box>
    </Box>
  );
};

export default StrategyAnalysis;