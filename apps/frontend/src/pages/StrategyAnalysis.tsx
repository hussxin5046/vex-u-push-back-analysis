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
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Download as DownloadIcon,
  Assessment as ChartIcon,
  Psychology as StrategyIcon,
  Help as HelpIcon,
  Close as CloseIcon,
  Info as InfoIcon,
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
      autonomous: 30,  // VEX U autonomous is 30 seconds
      driver: 90,      // VEX U driver control is 90 seconds
    },
    focusMetrics: ['win_probability', 'average_score'],
  });

  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState<AnalysisProgress | null>(null);
  const [results, setResults] = useState<AnalysisResult | null>(null);
  const [showGuide, setShowGuide] = useState(true);
  const { showNotification } = useNotification();

  const availableStrategies = [
    { 
      id: 'offensive_rush', 
      name: 'Offensive Rush', 
      description: 'Focus on scoring maximum blocks quickly. Best for teams with fast scoring mechanisms.',
      scoringPotential: 'High (150-200+ points)',
      riskLevel: 'Medium'
    },
    { 
      id: 'defensive_control', 
      name: 'Defensive Control', 
      description: 'Control zones and block opponent scoring. Ideal against high-scoring opponents.',
      scoringPotential: 'Medium (100-150 points)',
      riskLevel: 'Low'
    },
    { 
      id: 'balanced_approach', 
      name: 'Balanced Approach', 
      description: 'Mix of scoring and zone control. Adaptable to various opponent strategies.',
      scoringPotential: 'Medium-High (120-180 points)',
      riskLevel: 'Low'
    },
    { 
      id: 'autonomous_focus', 
      name: 'Autonomous Focus', 
      description: 'Maximize autonomous period scoring for early lead. Requires precise programming.',
      scoringPotential: 'High in Auto (40-60 points)',
      riskLevel: 'High'
    },
    { 
      id: 'endgame_specialist', 
      name: 'Platform Parking Focus', 
      description: 'Prioritize platform parking (30 points for 2 robots). Critical in close matches.',
      scoringPotential: 'Guaranteed 30 points',
      riskLevel: 'Medium'
    },
    { 
      id: 'support_coordination', 
      name: 'Alliance Coordination', 
      description: 'One robot scores, one controls zones. Requires excellent teamwork.',
      scoringPotential: 'Very High (180-220+ points)',
      riskLevel: 'Medium'
    },
  ];

  const focusMetrics = [
    { 
      id: 'win_probability', 
      name: 'Win Probability',
      description: 'Calculates the likelihood of winning against average opponents'
    },
    { 
      id: 'average_score', 
      name: 'Average Score',
      description: 'Expected total points including blocks, zones, and parking'
    },
    { 
      id: 'score_consistency', 
      name: 'Score Consistency',
      description: 'How reliable the strategy is across different match conditions'
    },
    { 
      id: 'autonomous_efficiency', 
      name: 'Autonomous Efficiency',
      description: 'Performance during the 30-second autonomous period'
    },
    { 
      id: 'driver_efficiency', 
      name: 'Driver Control Efficiency',
      description: 'Performance during the 90-second driver control period'
    },
    { 
      id: 'risk_assessment', 
      name: 'Risk Assessment',
      description: 'Vulnerability to opponent interference and execution errors'
    },
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
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box>
          <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 700 }}>
            VEX U Scoring Analysis
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Analyze and optimize scoring strategies for VEX U Push Back competitions
          </Typography>
        </Box>
        <Button
          startIcon={<HelpIcon />}
          onClick={() => setShowGuide(true)}
          variant="outlined"
          sx={{ height: 'fit-content' }}
        >
          Show Guide
        </Button>
      </Box>

      {/* User Guide Dialog */}
      {showGuide && (
        <Card sx={{ 
          mb: 3, 
          backgroundColor: (theme) => theme.palette.mode === 'dark' ? 'rgba(33, 150, 243, 0.08)' : '#f0f7ff',
          border: (theme) => `1px solid ${theme.palette.primary.main}`
        }}>
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6" sx={{ fontWeight: 600, color: 'primary.main' }}>
                <InfoIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                VEX U Push Back Scoring Guide
              </Typography>
              <IconButton onClick={() => setShowGuide(false)} size="small">
                <CloseIcon />
              </IconButton>
            </Box>
            
            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(3, 1fr)' }, gap: 3 }}>
              <Box>
                <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                  🎯 How Scoring Works
                </Typography>
                <Typography variant="body2" paragraph>
                  • <strong>Blocks:</strong> 3 points each (88 total blocks)<br/>
                  • <strong>Zone Control:</strong> 6-10 points per goal<br/>
                  • <strong>Parking:</strong> 8 points (1 robot) or 30 points (2 robots)<br/>
                  • <strong>Autonomous Win:</strong> 10 point bonus<br/>
                  • <strong>Match Time:</strong> 30s auto + 90s driver control
                </Typography>
              </Box>
              
              <Box>
                <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                  📊 Analysis Parameters
                </Typography>
                <Typography variant="body2" paragraph>
                  • <strong>Strategies:</strong> Select multiple to compare<br/>
                  • <strong>Simulations:</strong> More = higher accuracy<br/>
                  • <strong>Focus Metrics:</strong> What to optimize for<br/>
                  • Results show win rates, scores, and recommendations
                </Typography>
              </Box>
              
              <Box>
                <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                  💡 Strategy Tips
                </Typography>
                <Typography variant="body2">
                  • Platform parking is worth 10 blocks of scoring<br/>
                  • Zone control prevents opponent scoring<br/>
                  • Autonomous performance sets match tempo<br/>
                  • Balance risk vs. reward based on opponent
                </Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>
      )}

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
                <FormLabel component="legend">
                  Select Strategies to Analyze
                  <Typography variant="caption" color="text.secondary" display="block">
                    Choose multiple strategies to compare their effectiveness
                  </Typography>
                </FormLabel>
                <FormGroup>
                  {availableStrategies.map((strategy) => (
                    <Paper
                      key={strategy.id}
                      variant="outlined"
                      sx={{ 
                        p: 1.5, 
                        mb: 1,
                        backgroundColor: (theme) => parameters.strategies.includes(strategy.id) 
                          ? theme.palette.mode === 'dark' ? 'rgba(33, 150, 243, 0.15)' : '#e3f2fd'
                          : 'transparent',
                        border: (theme) => parameters.strategies.includes(strategy.id) 
                          ? `2px solid ${theme.palette.primary.main}` 
                          : `1px solid ${theme.palette.divider}`,
                        cursor: 'pointer',
                        '&:hover': { 
                          backgroundColor: (theme) => theme.palette.action.hover 
                        }
                      }}
                      onClick={() => handleStrategyChange(strategy.id, !parameters.strategies.includes(strategy.id))}
                    >
                      <FormControlLabel
                        control={
                          <Checkbox
                            checked={parameters.strategies.includes(strategy.id)}
                            onChange={(e) => handleStrategyChange(strategy.id, e.target.checked)}
                          />
                        }
                        label={
                          <Box sx={{ width: '100%' }}>
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                              {strategy.name}
                            </Typography>
                            <Typography variant="caption" color="text.secondary" display="block">
                              {strategy.description}
                            </Typography>
                            <Box sx={{ mt: 0.5, display: 'flex', gap: 1 }}>
                              <Chip 
                                label={strategy.scoringPotential} 
                                size="small" 
                                color="primary" 
                                variant="outlined"
                              />
                              <Chip 
                                label={`Risk: ${strategy.riskLevel}`} 
                                size="small" 
                                color={strategy.riskLevel === 'Low' ? 'success' : strategy.riskLevel === 'Medium' ? 'warning' : 'error'}
                                variant="outlined"
                              />
                            </Box>
                          </Box>
                        }
                        sx={{ width: '100%', m: 0 }}
                      />
                    </Paper>
                  ))}
                </FormGroup>
              </FormControl>

              {/* Simulation Count */}
              <Box sx={{ mt: 3 }}>
                <Typography gutterBottom>
                  Simulation Count: <strong>{parameters.simulationCount.toLocaleString()}</strong>
                  <Tooltip title="More simulations provide more accurate results but take longer to compute">
                    <InfoIcon sx={{ fontSize: 16, ml: 0.5, verticalAlign: 'middle', color: 'text.secondary' }} />
                  </Tooltip>
                </Typography>
                <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                  {parameters.simulationCount < 1000 ? 'Quick analysis' : 
                   parameters.simulationCount < 5000 ? 'Standard analysis' : 'Deep analysis'}
                </Typography>
                <Slider
                  value={parameters.simulationCount}
                  onChange={handleSimulationCountChange}
                  min={100}
                  max={10000}
                  step={100}
                  marks={[
                    { value: 100, label: 'Quick' },
                    { value: 1000, label: 'Standard' },
                    { value: 5000, label: 'Thorough' },
                    { value: 10000, label: 'Deep' },
                  ]}
                  disabled={isRunning}
                />
              </Box>

              {/* Focus Metrics */}
              <FormControl component="fieldset" sx={{ mt: 3, width: '100%' }}>
                <FormLabel component="legend">
                  Analysis Focus Metrics
                  <Typography variant="caption" color="text.secondary" display="block">
                    Select which metrics to prioritize in the analysis
                  </Typography>
                </FormLabel>
                <FormGroup>
                  {focusMetrics.map((metric) => (
                    <Tooltip key={metric.id} title={metric.description} placement="right">
                      <FormControlLabel
                        control={
                          <Checkbox
                            checked={parameters.focusMetrics.includes(metric.id)}
                            onChange={(e) => handleMetricChange(metric.id, e.target.checked)}
                          />
                        }
                        label={
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Typography variant="body2">{metric.name}</Typography>
                            <InfoIcon sx={{ fontSize: 14, ml: 0.5, color: 'text.secondary' }} />
                          </Box>
                        }
                      />
                    </Tooltip>
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
                  Ready to Analyze VEX U Strategies
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Select strategies from the left panel to begin your analysis.
                </Typography>
                <Box sx={{ 
                  mt: 3, 
                  p: 2, 
                  backgroundColor: (theme) => theme.palette.mode === 'dark' 
                    ? theme.palette.background.paper 
                    : theme.palette.grey[100], 
                  borderRadius: 1,
                  border: (theme) => `1px solid ${theme.palette.divider}`
                }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Quick Start:
                  </Typography>
                  <Typography variant="body2" color="text.secondary" align="left">
                    1. Choose 2-3 strategies to compare<br/>
                    2. Set simulation count (1000 is good for start)<br/>
                    3. Select metrics you care about<br/>
                    4. Click "Start Analysis" to begin
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          )}
        </Box>
      </Box>
    </Box>
  );
};

export default StrategyAnalysis;