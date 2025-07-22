import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Alert,
  FormGroup,
  FormControlLabel,
  Checkbox,
  Slider,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Download as DownloadIcon,
  Refresh as RefreshIcon,
  SmartToy as RobotIcon,
  Psychology as ScenarioIcon,
  Visibility as ViewIcon,
} from '@mui/icons-material';
import { useNotification } from '../contexts/NotificationContext';
import { ScenarioGenerationParams } from '../types';

interface GeneratedScenario {
  id: string;
  name: string;
  skillLevel: string;
  strategyType: string;
  expectedScore: number;
  difficulty: number;
  robots: {
    red: { role: string; efficiency: number }[];
    blue: { role: string; efficiency: number }[];
  };
  objectives: string[];
  constraints: string[];
  mlPredictions?: {
    winProbability: number;
    riskAssessment: string;
    recommendations: string[];
  };
}

const ScenarioGenerator: React.FC = () => {
  const [parameters, setParameters] = useState<ScenarioGenerationParams>({
    numScenarios: 5,
    complexityLevel: 'intermediate',
    includeMLPredictions: true,
    focusAreas: ['autonomous', 'driver'],
    timeConstraints: {
      autonomous: 15,
      driver: 105,
    },
  });

  const [scenarios, setScenarios] = useState<GeneratedScenario[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedScenarios, setSelectedScenarios] = useState<string[]>([]);
  const { showNotification } = useNotification();

  const skillLevels = [
    { value: 'beginner', label: 'Beginner', description: 'New teams, basic strategies' },
    { value: 'intermediate', label: 'Intermediate', description: 'Experienced teams, moderate complexity' },
    { value: 'advanced', label: 'Advanced', description: 'Elite teams, complex strategies' },
    { value: 'mixed', label: 'Mixed', description: 'Varying skill levels' },
  ];

  const strategyTypes = [
    { value: 'offensive', label: 'Offensive Focus' },
    { value: 'defensive', label: 'Defensive Focus' },
    { value: 'balanced', label: 'Balanced Approach' },
    { value: 'autonomous_heavy', label: 'Autonomous Heavy' },
    { value: 'endgame_specialist', label: 'Endgame Specialist' },
    { value: 'coordination', label: 'Alliance Coordination' },
  ];

  const focusAreas = [
    { id: 'autonomous', name: 'Autonomous Period' },
    { id: 'driver', name: 'Driver Control' },
    { id: 'endgame', name: 'Endgame' },
    { id: 'defense', name: 'Defensive Play' },
    { id: 'coordination', name: 'Team Coordination' },
    { id: 'efficiency', name: 'Scoring Efficiency' },
  ];

  const handleParameterChange = (field: keyof ScenarioGenerationParams, value: any) => {
    setParameters(prev => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleFocusAreaChange = (areaId: string, checked: boolean) => {
    setParameters(prev => ({
      ...prev,
      focusAreas: checked
        ? [...prev.focusAreas, areaId]
        : prev.focusAreas.filter(area => area !== areaId),
    }));
  };

  const generateScenarios = async () => {
    setIsGenerating(true);
    
    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 2000));

      const newScenarios: GeneratedScenario[] = Array.from(
        { length: parameters.numScenarios },
        (_, index) => ({
          id: `scenario_${Date.now()}_${index}`,
          name: `Scenario ${index + 1}`,
          skillLevel: skillLevels[Math.floor(Math.random() * skillLevels.length)].value,
          strategyType: strategyTypes[Math.floor(Math.random() * strategyTypes.length)].value,
          expectedScore: Math.floor(Math.random() * 100 + 50),
          difficulty: Math.random() * 0.8 + 0.2,
          robots: {
            red: [
              { role: 'offense', efficiency: Math.random() * 0.4 + 0.6 },
              { role: 'defense', efficiency: Math.random() * 0.4 + 0.6 },
            ],
            blue: [
              { role: 'offense', efficiency: Math.random() * 0.4 + 0.6 },
              { role: 'support', efficiency: Math.random() * 0.4 + 0.6 },
            ],
          },
          objectives: [
            'Maximize autonomous scoring',
            'Control strategic game elements',
            'Execute coordinated alliance strategy',
          ].slice(0, Math.floor(Math.random() * 3) + 1),
          constraints: [
            'Limited field positioning',
            'Time pressure scenarios',
            'Interference from opposing alliance',
          ].slice(0, Math.floor(Math.random() * 2) + 1),
          mlPredictions: parameters.includeMLPredictions ? {
            winProbability: Math.random() * 0.6 + 0.2,
            riskAssessment: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)],
            recommendations: [
              'Focus on autonomous programming',
              'Practice alliance coordination',
              'Optimize driver control efficiency',
            ],
          } : undefined,
        })
      );

      setScenarios(newScenarios);
      showNotification({
        type: 'success',
        title: 'Scenarios Generated',
        message: `Successfully generated ${parameters.numScenarios} scenarios.`,
      });
    } catch (error) {
      showNotification({
        type: 'error',
        title: 'Generation Failed',
        message: 'Failed to generate scenarios. Please try again.',
      });
    } finally {
      setIsGenerating(false);
    }
  };

  const handleScenarioSelect = (scenarioId: string, checked: boolean) => {
    setSelectedScenarios(prev =>
      checked
        ? [...prev, scenarioId]
        : prev.filter(id => id !== scenarioId)
    );
  };

  const deleteScenario = (scenarioId: string) => {
    setScenarios(prev => prev.filter(s => s.id !== scenarioId));
    setSelectedScenarios(prev => prev.filter(id => id !== scenarioId));
  };

  const exportScenarios = () => {
    const exportData = selectedScenarios.length > 0
      ? scenarios.filter(s => selectedScenarios.includes(s.id))
      : scenarios;

    const dataStr = JSON.stringify(exportData, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `vex_scenarios_${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const clearScenarios = () => {
    setScenarios([]);
    setSelectedScenarios([]);
  };

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 700 }}>
        Scenario Generator
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Generate custom VEX U scenarios for practice, analysis, and strategic planning.
      </Typography>

      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 2fr' }, gap: 3 }}>
        {/* Configuration Panel */}
        <Box>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                <ScenarioIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                Generation Parameters
              </Typography>

              {/* Number of Scenarios */}
              <Box sx={{ mt: 2 }}>
                <Typography gutterBottom>
                  Number of Scenarios: {parameters.numScenarios}
                </Typography>
                <Slider
                  value={parameters.numScenarios}
                  onChange={(_, value) => handleParameterChange('numScenarios', value)}
                  min={1}
                  max={20}
                  marks={[
                    { value: 1, label: '1' },
                    { value: 5, label: '5' },
                    { value: 10, label: '10' },
                    { value: 20, label: '20' },
                  ]}
                  disabled={isGenerating}
                />
              </Box>

              {/* Complexity Level */}
              <FormControl fullWidth margin="normal">
                <InputLabel>Complexity Level</InputLabel>
                <Select
                  value={parameters.complexityLevel}
                  onChange={(e) => handleParameterChange('complexityLevel', e.target.value)}
                  disabled={isGenerating}
                >
                  {skillLevels.map((level) => (
                    <MenuItem key={level.value} value={level.value}>
                      {level.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {/* Focus Areas */}
              <FormControl component="fieldset" sx={{ mt: 2, width: '100%' }}>
                <Typography variant="subtitle2" gutterBottom>
                  Focus Areas
                </Typography>
                <FormGroup>
                  {focusAreas.map((area) => (
                    <FormControlLabel
                      key={area.id}
                      control={
                        <Checkbox
                          checked={parameters.focusAreas.includes(area.id)}
                          onChange={(e) => handleFocusAreaChange(area.id, e.target.checked)}
                          disabled={isGenerating}
                        />
                      }
                      label={area.name}
                    />
                  ))}
                </FormGroup>
              </FormControl>

              {/* Time Constraints */}
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Time Constraints
                </Typography>
                <TextField
                  label="Autonomous (seconds)"
                  type="number"
                  value={parameters.timeConstraints?.autonomous || 15}
                  onChange={(e) => handleParameterChange('timeConstraints', {
                    ...parameters.timeConstraints,
                    autonomous: parseInt(e.target.value),
                  })}
                  fullWidth
                  margin="dense"
                  disabled={isGenerating}
                />
                <TextField
                  label="Driver Control (seconds)"
                  type="number"
                  value={parameters.timeConstraints?.driver || 105}
                  onChange={(e) => handleParameterChange('timeConstraints', {
                    ...parameters.timeConstraints,
                    driver: parseInt(e.target.value),
                  })}
                  fullWidth
                  margin="dense"
                  disabled={isGenerating}
                />
              </Box>

              {/* ML Predictions Toggle */}
              <FormControlLabel
                control={
                  <Checkbox
                    checked={parameters.includeMLPredictions}
                    onChange={(e) => handleParameterChange('includeMLPredictions', e.target.checked)}
                    disabled={isGenerating}
                  />
                }
                label="Include ML Predictions"
                sx={{ mt: 2 }}
              />

              {/* Generate Button */}
              <Button
                variant="contained"
                startIcon={isGenerating ? <RefreshIcon /> : <AddIcon />}
                onClick={generateScenarios}
                disabled={isGenerating || parameters.focusAreas.length === 0}
                fullWidth
                size="large"
                sx={{ mt: 2 }}
              >
                {isGenerating ? 'Generating...' : 'Generate Scenarios'}
              </Button>
            </CardContent>
          </Card>
        </Box>

        {/* Scenarios Display */}
        <Box>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  Generated Scenarios ({scenarios.length})
                </Typography>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  {scenarios.length > 0 && (
                    <>
                      <Button
                        startIcon={<DownloadIcon />}
                        onClick={exportScenarios}
                        variant="outlined"
                        size="small"
                      >
                        Export ({selectedScenarios.length || scenarios.length})
                      </Button>
                      <Button
                        startIcon={<DeleteIcon />}
                        onClick={clearScenarios}
                        variant="outlined"
                        color="error"
                        size="small"
                      >
                        Clear All
                      </Button>
                    </>
                  )}
                </Box>
              </Box>

              {scenarios.length > 0 ? (
                <TableContainer component={Paper} variant="outlined">
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell padding="checkbox">
                          <Checkbox
                            indeterminate={selectedScenarios.length > 0 && selectedScenarios.length < scenarios.length}
                            checked={scenarios.length > 0 && selectedScenarios.length === scenarios.length}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setSelectedScenarios(scenarios.map(s => s.id));
                              } else {
                                setSelectedScenarios([]);
                              }
                            }}
                          />
                        </TableCell>
                        <TableCell>Scenario</TableCell>
                        <TableCell>Skill Level</TableCell>
                        <TableCell>Strategy Type</TableCell>
                        <TableCell align="right">Expected Score</TableCell>
                        <TableCell align="center">Difficulty</TableCell>
                        <TableCell align="center">Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {scenarios.map((scenario) => (
                        <TableRow key={scenario.id} hover>
                          <TableCell padding="checkbox">
                            <Checkbox
                              checked={selectedScenarios.includes(scenario.id)}
                              onChange={(e) => handleScenarioSelect(scenario.id, e.target.checked)}
                            />
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" sx={{ fontWeight: 500 }}>
                              {scenario.name}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              ID: {scenario.id.slice(-8)}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={scenario.skillLevel}
                              size="small"
                              color="primary"
                              variant="outlined"
                            />
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2">
                              {strategyTypes.find(s => s.value === scenario.strategyType)?.label || scenario.strategyType}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            <Typography variant="body2" sx={{ fontWeight: 500 }}>
                              {scenario.expectedScore}
                            </Typography>
                          </TableCell>
                          <TableCell align="center">
                            <Chip
                              label={`${(scenario.difficulty * 100).toFixed(0)}%`}
                              size="small"
                              color={
                                scenario.difficulty < 0.4
                                  ? 'success'
                                  : scenario.difficulty < 0.7
                                  ? 'warning'
                                  : 'error'
                              }
                            />
                          </TableCell>
                          <TableCell align="center">
                            <IconButton
                              size="small"
                              onClick={() => {
                                showNotification({
                                  type: 'info',
                                  title: 'Scenario Details',
                                  message: 'Detailed scenario view coming soon!',
                                });
                              }}
                            >
                              <ViewIcon />
                            </IconButton>
                            <IconButton
                              size="small"
                              onClick={() => deleteScenario(scenario.id)}
                              color="error"
                            >
                              <DeleteIcon />
                            </IconButton>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              ) : (
                <Box sx={{ textAlign: 'center', py: 8 }}>
                  <RobotIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                  <Typography variant="h6" color="text.secondary" gutterBottom>
                    No Scenarios Generated Yet
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Configure your parameters and click "Generate Scenarios" to create custom VEX U scenarios.
                  </Typography>
                </Box>
              )}

              {scenarios.length > 0 && (
                <Alert severity="info" sx={{ mt: 2 }}>
                  Select scenarios using the checkboxes to export specific ones, or export all scenarios at once.
                  {parameters.includeMLPredictions && ' ML predictions are included in the generated scenarios.'}
                </Alert>
              )}
            </CardContent>
          </Card>
        </Box>
      </Box>
    </Box>
  );
};

export default ScenarioGenerator;