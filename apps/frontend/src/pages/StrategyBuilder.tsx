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
  Chip,
  Divider,
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  Build as BuildIcon,
  Save as SaveIcon,
  PlayArrow as AnalyzeIcon,
  AutoAwesome as ArchetypeIcon,
  Settings as TuneIcon,
  Add as AddIcon,
} from '@mui/icons-material';
import { pushBackApiService } from '../services/pushBackApi';
import type { PushBackStrategy, RobotSpecs } from '../types/pushBackTypes';

interface AnalysisResult {
  win_rate: number;
  avg_score: number;
  score_std: number;
  performance_confidence: number;
  recommended_archetype?: string;
}

const StrategyBuilder: React.FC = () => {
  const [strategy, setStrategy] = useState<Partial<PushBackStrategy>>({
    name: '',
    description: '',
    strategy_type: 'balanced',
    goal_priorities: { center: 0.6, long: 0.4 },
    autonomous_strategy: 'balanced',
    parking_strategy: 'late',
    robot_specs: []
  });
  
  const [robotSpecs, setRobotSpecs] = useState<RobotSpecs>({
    cycle_time: 5.0,
    pickup_reliability: 0.95,
    scoring_reliability: 0.98,
    autonomous_reliability: 0.88,
    max_capacity: 2,
    parking_capability: true
  });
  
  const [archetypes, setArchetypes] = useState<Record<string, PushBackStrategy>>({});
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [archetypeDialogOpen, setArchetypeDialogOpen] = useState(false);
  
  useEffect(() => {
    fetchArchetypes();
  }, []);
  
  const fetchArchetypes = async () => {
    try {
      const archetypeData = await pushBackApiService.getStrategyArchetypes();
      setArchetypes(archetypeData);
    } catch (err) {
      console.error('Failed to fetch archetypes:', err);
    }
  };
  
  const handleRobotSpecChange = (field: keyof RobotSpecs, value: any) => {
    setRobotSpecs(prev => ({ ...prev, [field]: value }));
    // Update strategy robot specs
    setStrategy(prev => ({
      ...prev,
      robot_specs: [{ ...robotSpecs, [field]: value }]
    }));
  };
  
  const handleStrategyChange = (field: keyof PushBackStrategy, value: any) => {
    setStrategy(prev => ({ ...prev, [field]: value }));
  };
  
  const handleGoalPriorityChange = (goalType: 'center' | 'long', value: number) => {
    const otherType = goalType === 'center' ? 'long' : 'center';
    const otherValue = 1 - value;
    
    setStrategy(prev => ({
      ...prev,
      goal_priorities: {
        ...prev.goal_priorities,
        [goalType]: value,
        [otherType]: otherValue
      }
    }));
  };
  
  const loadArchetype = (archetypeName: string) => {
    const archetype = archetypes[archetypeName];
    if (archetype) {
      setStrategy({
        ...archetype,
        id: undefined, // Clear ID for new strategy
        name: `${archetype.name} (Custom)`,
        robot_specs: [robotSpecs] // Keep current robot specs
      });
      setArchetypeDialogOpen(false);
    }
  };
  
  const runAnalysis = async () => {
    if (!strategy.name?.trim()) {
      setError('Please enter a strategy name');
      return;
    }
    
    try {
      setAnalyzing(true);
      setError(null);
      
      const fullStrategy: PushBackStrategy = {
        id: strategy.id || `strategy_${Date.now()}`,
        name: strategy.name,
        description: strategy.description || '',
        robot_specs: [robotSpecs],
        strategy_type: strategy.strategy_type || 'balanced',
        goal_priorities: strategy.goal_priorities || { center: 0.6, long: 0.4 },
        autonomous_strategy: strategy.autonomous_strategy || 'balanced',
        parking_strategy: strategy.parking_strategy || 'late',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      };
      
      // Run Monte Carlo simulation
      const result = await pushBackApiService.runMonteCarloSimulation(fullStrategy, 1000);
      
      // Get archetype recommendation
      try {
        const recommendation = await pushBackApiService.getArchetypeRecommendation([robotSpecs]);
        setAnalysisResult({
          ...result,
          recommended_archetype: recommendation.recommended_archetype
        });
      } catch {
        setAnalysisResult(result);
      }
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setAnalyzing(false);
    }
  };
  
  const saveStrategy = async () => {
    if (!strategy.name?.trim()) {
      setError('Please enter a strategy name');
      return;
    }
    
    try {
      setSaving(true);
      setError(null);
      
      const fullStrategy: PushBackStrategy = {
        id: strategy.id || `strategy_${Date.now()}`,
        name: strategy.name,
        description: strategy.description || '',
        robot_specs: [robotSpecs],
        strategy_type: strategy.strategy_type || 'balanced',
        goal_priorities: strategy.goal_priorities || { center: 0.6, long: 0.4 },
        autonomous_strategy: strategy.autonomous_strategy || 'balanced',
        parking_strategy: strategy.parking_strategy || 'late',
        created_at: strategy.created_at || new Date().toISOString(),
        updated_at: new Date().toISOString()
      };
      
      if (strategy.id) {
        await pushBackApiService.updateStrategy(strategy.id, fullStrategy);
      } else {
        const savedStrategy = await pushBackApiService.createStrategy(fullStrategy);
        setStrategy(savedStrategy);
      }
      
      setError(null);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Save failed');
    } finally {
      setSaving(false);
    }
  };
  
  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 700 }}>
        Push Back Strategy Builder
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Create and analyze your Push Back strategy with real-time simulation and optimization.
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Box sx={{ display: 'flex', gap: 3, flexDirection: { xs: 'column', md: 'row' } }}>
        {/* Strategy Configuration */}
        <Box sx={{ flex: { xs: '1 1 100%', md: '0 0 67%' } }}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                <BuildIcon sx={{ mr: 1 }} />
                Strategy Configuration
              </Typography>
              
              <Box sx={{ display: 'flex', gap: 3, flexDirection: { xs: 'column', md: 'row' } }}>
                {/* Basic Info */}
                <Box>
                  <TextField
                    fullWidth
                    label="Strategy Name"
                    value={strategy.name || ''}
                    onChange={(e) => handleStrategyChange('name', e.target.value)}
                    required
                  />
                </Box>
                
                <Box>
                  <TextField
                    fullWidth
                    label="Description"
                    multiline
                    rows={2}
                    value={strategy.description || ''}
                    onChange={(e) => handleStrategyChange('description', e.target.value)}
                  />
                </Box>
                
                {/* Strategy Type */}
                <Box sx={{ display: 'flex', gap: 2 }}>\n                  <Box sx={{ flex: 1 }}>
                  <FormControl fullWidth>
                    <InputLabel>Strategy Type</InputLabel>
                    <Select
                      value={strategy.strategy_type || 'balanced'}
                      label="Strategy Type"
                      onChange={(e) => handleStrategyChange('strategy_type', e.target.value)}
                    >
                      <MenuItem value="balanced">Balanced</MenuItem>
                      <MenuItem value="aggressive_offense">Aggressive Offense</MenuItem>
                      <MenuItem value="control_zone_dominator">Control Zone Dominator</MenuItem>
                      <MenuItem value="block_flow_maximizer">Block Flow Maximizer</MenuItem>
                      <MenuItem value="defensive_specialist">Defensive Specialist</MenuItem>
                    </Select>
                  </FormControl>
                </Box>
                
                {/* Autonomous Strategy */}
                <Box sx={{ display: 'flex', gap: 2 }}>\n                  <Box sx={{ flex: 1 }}>
                  <FormControl fullWidth>
                    <InputLabel>Autonomous Strategy</InputLabel>
                    <Select
                      value={strategy.autonomous_strategy || 'balanced'}
                      label="Autonomous Strategy"
                      onChange={(e) => handleStrategyChange('autonomous_strategy', e.target.value)}
                    >
                      <MenuItem value="aggressive">Aggressive</MenuItem>
                      <MenuItem value="balanced">Balanced</MenuItem>
                      <MenuItem value="safe">Safe</MenuItem>
                    </Select>
                  </FormControl>
                </Box>
                
                {/* Parking Strategy */}
                <Box sx={{ display: 'flex', gap: 2 }}>\n                  <Box sx={{ flex: 1 }}>
                  <FormControl fullWidth>
                    <InputLabel>Parking Strategy</InputLabel>
                    <Select
                      value={strategy.parking_strategy || 'late'}
                      label="Parking Strategy"
                      onChange={(e) => handleStrategyChange('parking_strategy', e.target.value)}
                    >
                      <MenuItem value="never">Never Park</MenuItem>
                      <MenuItem value="early">Early Park</MenuItem>
                      <MenuItem value="late">Late Park</MenuItem>
                      <MenuItem value="adaptive">Adaptive</MenuItem>
                    </Select>
                  </FormControl>
                </Box>
                
                {/* Goal Priorities */}
                <Box>
                  <Typography variant="subtitle1" gutterBottom>
                    Goal Priorities
                  </Typography>
                  <Box sx={{ px: 2 }}>
                    <Typography variant="body2" gutterBottom>
                      Center Goals: {((strategy.goal_priorities?.center || 0.6) * 100).toFixed(0)}%
                    </Typography>
                    <Slider
                      value={strategy.goal_priorities?.center || 0.6}
                      min={0.1}
                      max={0.9}
                      step={0.05}
                      onChange={(_, value) => handleGoalPriorityChange('center', value as number)}
                      valueLabelDisplay="auto"
                      valueLabelFormat={(value) => `${(value * 100).toFixed(0)}%`}
                    />
                    <Typography variant="body2" color="text.secondary">
                      Long Goals: {((strategy.goal_priorities?.long || 0.4) * 100).toFixed(0)}%
                    </Typography>
                  </Box>
                </Box>
              </Box>
            </CardContent>
          </Card>
          
          {/* Robot Specifications */}
          <Card sx={{ mt: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                <TuneIcon sx={{ mr: 1 }} />
                Robot Specifications
              </Typography>
              
              <Box sx={{ display: 'flex', gap: 3, flexDirection: { xs: 'column', md: 'row' } }}>
                <Box sx={{ display: 'flex', gap: 2 }}>\n                  <Box sx={{ flex: 1 }}>
                  <Typography variant="body2" gutterBottom>
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
                
                <Box sx={{ display: 'flex', gap: 2 }}>\n                  <Box sx={{ flex: 1 }}>
                  <Typography variant="body2" gutterBottom>
                    Pickup Reliability: {(robotSpecs.pickup_reliability * 100).toFixed(0)}%
                  </Typography>
                  <Slider
                    value={robotSpecs.pickup_reliability}
                    min={0.7}
                    max={1.0}
                    step={0.01}
                    onChange={(_, value) => handleRobotSpecChange('pickup_reliability', value)}
                    valueLabelDisplay="auto"
                    valueLabelFormat={(value) => `${(value * 100).toFixed(0)}%`}
                  />
                </Box>
                
                <Box sx={{ display: 'flex', gap: 2 }}>\n                  <Box sx={{ flex: 1 }}>
                  <Typography variant="body2" gutterBottom>
                    Scoring Reliability: {(robotSpecs.scoring_reliability * 100).toFixed(0)}%
                  </Typography>
                  <Slider
                    value={robotSpecs.scoring_reliability}
                    min={0.8}
                    max={1.0}
                    step={0.01}
                    onChange={(_, value) => handleRobotSpecChange('scoring_reliability', value)}
                    valueLabelDisplay="auto"
                    valueLabelFormat={(value) => `${(value * 100).toFixed(0)}%`}
                  />
                </Box>
                
                <Box sx={{ display: 'flex', gap: 2 }}>\n                  <Box sx={{ flex: 1 }}>
                  <Typography variant="body2" gutterBottom>
                    Autonomous Reliability: {(robotSpecs.autonomous_reliability * 100).toFixed(0)}%
                  </Typography>
                  <Slider
                    value={robotSpecs.autonomous_reliability}
                    min={0.6}
                    max={1.0}
                    step={0.01}
                    onChange={(_, value) => handleRobotSpecChange('autonomous_reliability', value)}
                    valueLabelDisplay="auto"
                    valueLabelFormat={(value) => `${(value * 100).toFixed(0)}%`}
                  />
                </Box>
                
                <Box sx={{ display: 'flex', gap: 2 }}>\n                  <Box sx={{ flex: 1 }}>
                  <Typography variant="body2" gutterBottom>
                    Max Capacity: {robotSpecs.max_capacity} blocks
                  </Typography>
                  <Slider
                    value={robotSpecs.max_capacity}
                    min={1}
                    max={4}
                    step={1}
                    onChange={(_, value) => handleRobotSpecChange('max_capacity', value)}
                    valueLabelDisplay="auto"
                    marks
                  />
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Box>
        
        {/* Actions and Results */}
        <Box sx={{ flex: { xs: '1 1 100%', md: '0 0 33%' } }}>
          {/* Quick Actions */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Actions
              </Typography>
              
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Button
                  variant="contained"
                  startIcon={analyzing ? <CircularProgress size={20} /> : <AnalyzeIcon />}
                  onClick={runAnalysis}
                  disabled={analyzing || !strategy.name?.trim()}
                  fullWidth
                >
                  {analyzing ? 'Analyzing...' : 'Run Analysis'}
                </Button>
                
                <Button
                  variant="outlined"
                  startIcon={saving ? <CircularProgress size={20} /> : <SaveIcon />}
                  onClick={saveStrategy}
                  disabled={saving || !strategy.name?.trim()}
                  fullWidth
                >
                  {saving ? 'Saving...' : 'Save Strategy'}
                </Button>
                
                <Button
                  variant="text"
                  startIcon={<ArchetypeIcon />}
                  onClick={() => setArchetypeDialogOpen(true)}
                  fullWidth
                >
                  Load Archetype
                </Button>
              </Box>
            </CardContent>
          </Card>
          
          {/* Analysis Results */}
          {analysisResult && (
            <Card sx={{ mt: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Analysis Results
                </Typography>
                
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <Paper sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="h4" color="primary.main" sx={{ fontWeight: 700 }}>
                      {(analysisResult.win_rate * 100).toFixed(1)}%
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Win Rate
                    </Typography>
                  </Paper>
                  
                  <Paper sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="h5" color="success.main" sx={{ fontWeight: 600 }}>
                      {analysisResult.avg_score.toFixed(1)}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Average Score
                    </Typography>
                  </Paper>
                  
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Score Variation: ±{analysisResult.score_std.toFixed(1)}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Confidence: {(analysisResult.performance_confidence * 100).toFixed(0)}%
                    </Typography>
                  </Box>
                  
                  {analysisResult.recommended_archetype && (
                    <Alert severity="info" sx={{ mt: 2 }}>
                      <strong>Recommended:</strong> {analysisResult.recommended_archetype}
                    </Alert>
                  )}
                </Box>
              </CardContent>
            </Card>
          )}
        </Box>
      </Box>
      
      {/* Archetype Selection Dialog */}
      <Dialog open={archetypeDialogOpen} onClose={() => setArchetypeDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Select Strategy Archetype</DialogTitle>
        <DialogContent>
          <Grid container spacing={2}>
            {Object.entries(archetypes).map(([name, archetype]) => (
              <Grid item xs={12} sm={6} key={name}>
                <Card 
                  sx={{ cursor: 'pointer', '&:hover': { boxShadow: 4 } }}
                  onClick={() => loadArchetype(name)}
                >
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      {archetype.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" paragraph>
                      {archetype.description}
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      <Chip label={archetype.strategy_type} size="small" />
                      <Chip label={archetype.autonomous_strategy} size="small" variant="outlined" />
                      <Chip label={archetype.parking_strategy} size="small" variant="outlined" />
                    </Box>
                  </CardContent>
                </Card>
              </Box>
            ))}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setArchetypeDialogOpen(false)}>Cancel</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default StrategyBuilder;