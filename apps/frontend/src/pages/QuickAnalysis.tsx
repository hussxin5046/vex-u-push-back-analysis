import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Alert,
  CircularProgress,
  LinearProgress,
  Chip,
  Paper,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
} from '@mui/material';
import {
  Analytics as AnalyticsIcon,
  PlayArrow as RunIcon,
  Speed as SpeedIcon,
  GpsFixed as TargetIcon,
  Insights as InsightsIcon,
  ExpandMore as ExpandMoreIcon,
  TrendingUp as TrendingUpIcon,
  Assessment as AssessmentIcon,
  CheckCircle as CheckIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
} from '@mui/icons-material';
import { pushBackApiService } from '../services/pushBackApi';
import type { RobotSpecs } from '../types/pushBackTypes';

interface ComprehensiveAnalysis {
  block_flow: {
    recommended_distribution: { center_goals: number; long_goals: number };
    expected_value: number;
    efficiency: number;
  };
  autonomous_decision: {
    recommended_strategy: string;
    expected_points: number;
    win_point_probability: number;
    risk_assessment: string;
  };
  goal_priority: {
    recommended_priority: string;
    center_goal_value: number;
    long_goal_value: number;
    decision_confidence: number;
  };
  parking_decision: {
    recommended_timing: string;
    expected_value: number;
    risk_benefit_ratio: number;
  };
  offense_defense: {
    recommended_ratio: [number, number];
    offensive_roi: number;
    defensive_roi: number;
  };
  recommended_archetype: string;
  recommendations: string[];
}

interface MonteCarloResult {
  win_rate: number;
  avg_score: number;
  score_std: number;
  performance_confidence: number;
}

const QuickAnalysis: React.FC = () => {
  const [robotSpecs, setRobotSpecs] = useState<RobotSpecs>({
    cycle_time: 5.0,
    pickup_reliability: 0.95,
    scoring_reliability: 0.98,
    autonomous_reliability: 0.88,
    max_capacity: 2,
    parking_capability: true
  });
  
  const [numSimulations, setNumSimulations] = useState(1000);
  const [opponentStrength, setOpponentStrength] = useState('competitive');
  const [analysisResult, setAnalysisResult] = useState<ComprehensiveAnalysis | null>(null);
  const [monteCarloResult, setMonteCarloResult] = useState<MonteCarloResult | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [executionTime, setExecutionTime] = useState<number | null>(null);
  
  const handleRobotSpecChange = (field: keyof RobotSpecs, value: any) => {
    setRobotSpecs(prev => ({ ...prev, [field]: value }));
  };
  
  const runQuickAnalysis = async () => {
    try {
      setAnalyzing(true);
      setError(null);
      setProgress(0);
      const startTime = Date.now();
      
      // Step 1: Run comprehensive analysis (40% progress)
      setProgress(10);
      const comprehensiveAnalysis = await pushBackApiService.runComprehensiveAnalysis(
        [robotSpecs],
        { opponent_strength: opponentStrength }
      );
      setAnalysisResult(comprehensiveAnalysis);
      setProgress(40);
      
      // Step 2: Run Monte Carlo simulation (80% progress)
      const strategy = {
        id: 'quick_analysis',
        name: 'Quick Analysis Strategy',
        description: 'Generated for quick analysis',
        robot_specs: [robotSpecs],
        strategy_type: comprehensiveAnalysis.recommended_archetype || 'balanced',
        goal_priorities: {
          center: comprehensiveAnalysis.block_flow.recommended_distribution.center_goals,
          long: comprehensiveAnalysis.block_flow.recommended_distribution.long_goals
        },
        autonomous_strategy: comprehensiveAnalysis.autonomous_decision.recommended_strategy,
        parking_strategy: comprehensiveAnalysis.parking_decision.recommended_timing,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      };
      
      const monteCarloData = await pushBackApiService.runMonteCarloSimulation(
        strategy,
        numSimulations
      );
      setMonteCarloResult(monteCarloData);
      setProgress(100);
      
      const endTime = Date.now();
      setExecutionTime((endTime - startTime) / 1000);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setAnalyzing(false);
      setProgress(0);
    }
  };
  
  const getRecommendationIcon = (type: 'success' | 'warning' | 'error') => {
    switch (type) {
      case 'success': return <CheckIcon color="success" />;
      case 'warning': return <WarningIcon color="warning" />;
      case 'error': return <ErrorIcon color="error" />;
    }
  };
  
  const getRecommendationType = (confidence: number): 'success' | 'warning' | 'error' => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };
  
  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 700 }}>
        Quick Push Back Analysis
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Get comprehensive strategic insights for your Push Back robot in seconds.
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {analyzing && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
              <CircularProgress size={24} sx={{ mr: 2 }} />
              Running Analysis...
            </Typography>
            <LinearProgress variant="determinate" value={progress} />
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              {progress < 40 && 'Running comprehensive strategic analysis...'}
              {progress >= 40 && progress < 100 && `Running Monte Carlo simulation (${numSimulations} scenarios)...`}
              {progress === 100 && 'Analysis complete!'}
            </Typography>
          </CardContent>
        </Card>
      )}

      <Box sx={{ display: 'flex', gap: 3, flexDirection: { xs: 'column', md: 'row' } }}>
        {/* Configuration Panel */}
        <Box sx={{ flex: { xs: '1 1 100%', md: '0 0 33%' } }}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                <AnalyticsIcon sx={{ mr: 1 }} />
                Quick Setup
              </Typography>
              
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                {/* Robot Specs */}
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Cycle Time: {robotSpecs.cycle_time.toFixed(1)}s
                  </Typography>
                  <Slider
                    value={robotSpecs.cycle_time}
                    min={2.0}
                    max={10.0}
                    step={0.1}
                    onChange={(_, value) => handleRobotSpecChange('cycle_time', value)}
                    valueLabelDisplay="auto"
                    valueLabelFormat={(value) => `${value}s`}
                  />
                </Box>
                
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Overall Reliability: {(robotSpecs.pickup_reliability * 100).toFixed(0)}%
                  </Typography>
                  <Slider
                    value={robotSpecs.pickup_reliability}
                    min={0.7}
                    max={1.0}
                    step={0.01}
                    onChange={(_, value) => {
                      handleRobotSpecChange('pickup_reliability', value);
                      handleRobotSpecChange('scoring_reliability', Math.min(1.0, (value as number) + 0.03));
                    }}
                    valueLabelDisplay="auto"
                    valueLabelFormat={(value) => `${(value * 100).toFixed(0)}%`}
                  />
                </Box>
                
                <FormControl fullWidth>
                  <InputLabel>Opponent Strength</InputLabel>
                  <Select
                    value={opponentStrength}
                    label="Opponent Strength"
                    onChange={(e) => setOpponentStrength(e.target.value)}
                  >
                    <MenuItem value="beginner">Beginner</MenuItem>
                    <MenuItem value="competitive">Competitive</MenuItem>
                    <MenuItem value="elite">Elite</MenuItem>
                  </Select>
                </FormControl>
                
                <TextField
                  label="Monte Carlo Simulations"
                  type="number"
                  value={numSimulations}
                  onChange={(e) => setNumSimulations(parseInt(e.target.value) || 1000)}
                  inputProps={{ min: 100, max: 5000, step: 100 }}
                  helperText="More simulations = higher accuracy"
                />
                
                <Button
                  variant="contained"
                  size="large"
                  startIcon={analyzing ? <CircularProgress size={20} /> : <RunIcon />}
                  onClick={runQuickAnalysis}
                  disabled={analyzing}
                  fullWidth
                >
                  {analyzing ? 'Analyzing...' : 'Run Quick Analysis'}
                </Button>
                
                {executionTime && (
                  <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center' }}>
                    Analysis completed in {executionTime.toFixed(2)}s
                  </Typography>
                )}
              </Box>
            </CardContent>
          </Card>
        </Box>
        
        {/* Results Panel */}
        <Box sx={{ flex: { xs: '1 1 100%', md: '0 0 67%' } }}>
          {monteCarloResult && (
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                  <AssessmentIcon sx={{ mr: 1 }} />
                  Performance Metrics
                </Typography>
                
                <Grid container spacing={2}>
                  <Grid item xs={6} sm={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h4" color="primary.main" sx={{ fontWeight: 700 }}>
                        {(monteCarloResult.win_rate * 100).toFixed(1)}%
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Win Rate
                      </Typography>
                    </Paper>
                  </Box>
                  
                  <Grid item xs={6} sm={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h4" color="success.main" sx={{ fontWeight: 700 }}>
                        {monteCarloResult.avg_score.toFixed(0)}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Avg Score
                      </Typography>
                    </Paper>
                  </Box>
                  
                  <Grid item xs={6} sm={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h5" color="warning.main" sx={{ fontWeight: 600 }}>
                        ±{monteCarloResult.score_std.toFixed(0)}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Score Range
                      </Typography>
                    </Paper>
                  </Box>
                  
                  <Grid item xs={6} sm={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h5" color="info.main" sx={{ fontWeight: 600 }}>
                        {(monteCarloResult.performance_confidence * 100).toFixed(0)}%
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Confidence
                      </Typography>
                    </Paper>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          )}
          
          {analysisResult && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                  <InsightsIcon sx={{ mr: 1 }} />
                  Strategic Analysis
                </Typography>
                
                {/* Key Recommendations */}
                <Alert severity="info" sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Recommended Archetype: <strong>{analysisResult.recommended_archetype}</strong>
                  </Typography>
                  <Typography variant="body2">
                    This strategy optimizes your robot's capabilities for maximum performance.
                  </Typography>
                </Alert>
                
                {/* Detailed Analysis Sections */}
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                        Block Flow Strategy
                      </Typography>
                      <Chip 
                        label={`${(analysisResult.block_flow.efficiency * 100).toFixed(0)}% efficiency`} 
                        size="small" 
                        sx={{ ml: 2 }}
                      />
                    </AccordionSummary>
                    <AccordionDetails>
                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <Typography variant="body2" gutterBottom>
                            <strong>Center Goals:</strong> {(analysisResult.block_flow.recommended_distribution.center_goals * 100).toFixed(0)}%
                          </Typography>
                        </Box>
                        <Grid item xs={6}>
                          <Typography variant="body2" gutterBottom>
                            <strong>Long Goals:</strong> {(analysisResult.block_flow.recommended_distribution.long_goals * 100).toFixed(0)}%
                          </Typography>
                        </Box>
                        <Grid item xs={12}>
                          <Typography variant="body2" color="text.secondary">
                            Expected Value: {analysisResult.block_flow.expected_value.toFixed(1)} points
                          </Typography>
                        </Box>
                      </Box>
                    </AccordionDetails>
                  </Accordion>
                  
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                        Autonomous Strategy
                      </Typography>
                      <Chip 
                        label={analysisResult.autonomous_decision.recommended_strategy} 
                        size="small" 
                        sx={{ ml: 2 }}
                      />
                    </AccordionSummary>
                    <AccordionDetails>
                      <Typography variant="body2" gutterBottom>
                        <strong>Expected Points:</strong> {analysisResult.autonomous_decision.expected_points.toFixed(1)}
                      </Typography>
                      <Typography variant="body2" gutterBottom>
                        <strong>Win Point Probability:</strong> {(analysisResult.autonomous_decision.win_point_probability * 100).toFixed(0)}%
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        <strong>Risk Level:</strong> {analysisResult.autonomous_decision.risk_assessment}
                      </Typography>
                    </AccordionDetails>
                  </Accordion>
                  
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                        Goal Priority & Parking
                      </Typography>
                      <Chip 
                        label={analysisResult.goal_priority.recommended_priority} 
                        size="small" 
                        sx={{ ml: 2 }}
                      />
                    </AccordionSummary>
                    <AccordionDetails>
                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <Typography variant="body2" gutterBottom>
                            <strong>Center Goal Value:</strong> {analysisResult.goal_priority.center_goal_value.toFixed(1)}
                          </Typography>
                          <Typography variant="body2" gutterBottom>
                            <strong>Long Goal Value:</strong> {analysisResult.goal_priority.long_goal_value.toFixed(1)}
                          </Typography>
                        </Box>
                        <Grid item xs={6}>
                          <Typography variant="body2" gutterBottom>
                            <strong>Parking Strategy:</strong> {analysisResult.parking_decision.recommended_timing}
                          </Typography>
                          <Typography variant="body2" gutterBottom>
                            <strong>Parking Value:</strong> {analysisResult.parking_decision.expected_value.toFixed(1)}
                          </Typography>
                        </Box>
                      </Box>
                    </AccordionDetails>
                  </Accordion>
                </Box>
                
                {/* Action Items */}
                {analysisResult.recommendations.length > 0 && (
                  <Box sx={{ mt: 3 }}>
                    <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600 }}>
                      Key Recommendations
                    </Typography>
                    <List dense>
                      {analysisResult.recommendations.map((recommendation, index) => (
                        <ListItem key={index}>
                          <ListItemIcon>
                            {getRecommendationIcon(getRecommendationType(0.8))}
                          </ListItemIcon>
                          <ListItemText primary={recommendation} />
                        </ListItem>
                      ))}
                    </List>
                  </Box>
                )}
              </CardContent>
            </Card>
          )}
          
          {!analysisResult && !analyzing && (
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 6 }}>
                <SpeedIcon sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
                <Typography variant="h5" gutterBottom sx={{ fontWeight: 600 }}>
                  Ready for Analysis
                </Typography>
                <Typography variant="body1" color="text.secondary" paragraph>
                  Configure your robot specifications and click "Run Quick Analysis" to get:
                </Typography>
                <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 2, mt: 3 }}>
                  <Box>
                    <TargetIcon color="primary" sx={{ mb: 1 }} />
                    <Typography variant="body2"><strong>Block Flow Optimization</strong></Typography>
                    <Typography variant="body2" color="text.secondary">Optimal goal distribution</Typography>
                  </Box>
                  <Box>
                    <TrendingUpIcon color="primary" sx={{ mb: 1 }} />
                    <Typography variant="body2"><strong>Performance Prediction</strong></Typography>
                    <Typography variant="body2" color="text.secondary">Win rates & scoring</Typography>
                  </Box>
                  <Box>
                    <InsightsIcon color="primary" sx={{ mb: 1 }} />
                    <Typography variant="body2"><strong>Strategic Insights</strong></Typography>
                    <Typography variant="body2" color="text.secondary">Decision recommendations</Typography>
                  </Box>
                  <Box>
                    <AssessmentIcon color="primary" sx={{ mb: 1 }} />
                    <Typography variant="body2"><strong>Monte Carlo Simulation</strong></Typography>
                    <Typography variant="body2" color="text.secondary">Statistical validation</Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          )}
        </Box>
      </Box>
    </Box>
  );
};

export default QuickAnalysis;