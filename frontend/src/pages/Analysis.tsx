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
  LinearProgress,
  Alert,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Psychology as AnalysisIcon,
  Assessment as ReportIcon,
} from '@mui/icons-material';
import { useRunAnalysis, useAnalysisHistory } from '../hooks/useApi';
import { useNotification } from '../contexts/NotificationContext';

const Analysis: React.FC = () => {
  const [analysisType, setAnalysisType] = useState('scoring');
  const [parameters, setParameters] = useState<Record<string, any>>({});
  
  const runAnalysisMutation = useRunAnalysis();
  const { data: analysisHistory = [], isLoading: historyLoading } = useAnalysisHistory();
  const { showNotification } = useNotification();

  const analysisTypes = [
    { value: 'scoring', label: 'Scoring Analysis', description: 'Analyze scoring patterns and efficiency' },
    { value: 'strategy', label: 'Strategy Analysis', description: 'Evaluate strategic approaches and effectiveness' },
    { value: 'statistical', label: 'Statistical Analysis', description: 'Comprehensive statistical breakdown' },
    { value: 'ml_prediction', label: 'ML Prediction', description: 'Machine learning based predictions' },
  ];

  const handleRunAnalysis = async () => {
    try {
      const result = await runAnalysisMutation.mutateAsync({
        type: analysisType,
        params: parameters,
      });
      
      showNotification({
        type: 'success',
        title: 'Analysis Complete',
        message: `${result.title} has been completed successfully.`,
      });
    } catch (error) {
      showNotification({
        type: 'error',
        title: 'Analysis Failed',
        message: 'There was an error running the analysis. Please try again.',
      });
    }
  };

  const handleParameterChange = (key: string, value: any) => {
    setParameters(prev => ({ ...prev, [key]: value }));
  };

  const renderParameterInputs = () => {
    switch (analysisType) {
      case 'scoring':
        return (
          <>
            <TextField
              label="Time Period (days)"
              type="number"
              value={parameters.timePeriod || ''}
              onChange={(e) => handleParameterChange('timePeriod', parseInt(e.target.value))}
              fullWidth
              margin="normal"
            />
            <FormControl fullWidth margin="normal">
              <InputLabel>Focus Area</InputLabel>
              <Select
                value={parameters.focusArea || ''}
                onChange={(e) => handleParameterChange('focusArea', e.target.value)}
              >
                <MenuItem value="autonomous">Autonomous Period</MenuItem>
                <MenuItem value="driver">Driver Period</MenuItem>
                <MenuItem value="overall">Overall Performance</MenuItem>
              </Select>
            </FormControl>
          </>
        );
      case 'strategy':
        return (
          <>
            <TextField
              label="Strategy Name"
              value={parameters.strategyName || ''}
              onChange={(e) => handleParameterChange('strategyName', e.target.value)}
              fullWidth
              margin="normal"
            />
            <FormControl fullWidth margin="normal">
              <InputLabel>Complexity Level</InputLabel>
              <Select
                value={parameters.complexity || ''}
                onChange={(e) => handleParameterChange('complexity', e.target.value)}
              >
                <MenuItem value="basic">Basic</MenuItem>
                <MenuItem value="intermediate">Intermediate</MenuItem>
                <MenuItem value="advanced">Advanced</MenuItem>
              </Select>
            </FormControl>
          </>
        );
      case 'statistical':
        return (
          <>
            <TextField
              label="Sample Size"
              type="number"
              value={parameters.sampleSize || ''}
              onChange={(e) => handleParameterChange('sampleSize', parseInt(e.target.value))}
              fullWidth
              margin="normal"
            />
            <FormControl fullWidth margin="normal">
              <InputLabel>Statistical Method</InputLabel>
              <Select
                value={parameters.method || ''}
                onChange={(e) => handleParameterChange('method', e.target.value)}
              >
                <MenuItem value="descriptive">Descriptive Statistics</MenuItem>
                <MenuItem value="inferential">Inferential Statistics</MenuItem>
                <MenuItem value="predictive">Predictive Analysis</MenuItem>
              </Select>
            </FormControl>
          </>
        );
      case 'ml_prediction':
        return (
          <>
            <FormControl fullWidth margin="normal">
              <InputLabel>Model Type</InputLabel>
              <Select
                value={parameters.modelType || ''}
                onChange={(e) => handleParameterChange('modelType', e.target.value)}
              >
                <MenuItem value="coordination">Coordination Model</MenuItem>
                <MenuItem value="scoring_optimizer">Scoring Optimizer</MenuItem>
                <MenuItem value="strategy_predictor">Strategy Predictor</MenuItem>
              </Select>
            </FormControl>
            <TextField
              label="Prediction Horizon (matches)"
              type="number"
              value={parameters.horizon || ''}
              onChange={(e) => handleParameterChange('horizon', parseInt(e.target.value))}
              fullWidth
              margin="normal"
            />
          </>
        );
      default:
        return null;
    }
  };

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 700 }}>
        Analysis Center
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Run comprehensive analyses on your VEX U competition data to gain strategic insights.
      </Typography>

      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 3 }}>
        {/* Analysis Configuration */}
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
              Configure Analysis
            </Typography>
            
            <FormControl fullWidth margin="normal">
              <InputLabel>Analysis Type</InputLabel>
              <Select
                value={analysisType}
                onChange={(e) => setAnalysisType(e.target.value)}
              >
                {analysisTypes.map((type) => (
                  <MenuItem key={type.value} value={type.value}>
                    {type.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <Alert severity="info" sx={{ mt: 2, mb: 2 }}>
              {analysisTypes.find(t => t.value === analysisType)?.description}
            </Alert>

            {renderParameterInputs()}

            <Box sx={{ mt: 3 }}>
              <Button
                variant="contained"
                startIcon={<PlayIcon />}
                onClick={handleRunAnalysis}
                disabled={runAnalysisMutation.isPending}
                fullWidth
                size="large"
              >
                {runAnalysisMutation.isPending ? 'Running Analysis...' : 'Run Analysis'}
              </Button>
            </Box>

            {runAnalysisMutation.isPending && (
              <Box sx={{ mt: 2 }}>
                <LinearProgress />
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  This may take a few moments...
                </Typography>
              </Box>
            )}
          </CardContent>
        </Card>

        {/* Analysis History */}
        <Card sx={{ height: 'fit-content' }}>
          <CardContent>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
              Analysis History
            </Typography>
            
            {historyLoading ? (
              <LinearProgress />
            ) : Array.isArray(analysisHistory) && analysisHistory.length > 0 ? (
              <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
                {analysisHistory.map((analysis: any) => (
                  <Box
                    key={analysis.id}
                    sx={{
                      p: 2,
                      mb: 2,
                      border: '1px solid',
                      borderColor: 'divider',
                      borderRadius: 1,
                      '&:hover': {
                        backgroundColor: 'action.hover',
                      },
                    }}
                  >
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', mb: 1 }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 500 }}>
                        {analysis.title}
                      </Typography>
                      <Chip
                        label={analysis.type.replace('_', ' ')}
                        size="small"
                        color="primary"
                        variant="outlined"
                      />
                    </Box>
                    <Typography variant="body2" color="text.secondary" paragraph>
                      {analysis.summary}
                    </Typography>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="caption" color="text.secondary">
                        {new Date(analysis.createdAt).toLocaleString()}
                      </Typography>
                      <Button
                        size="small"
                        startIcon={<ReportIcon />}
                        onClick={() => {
                          showNotification({
                            type: 'info',
                            title: 'Analysis Details',
                            message: 'Analysis details view coming soon!',
                          });
                        }}
                      >
                        View
                      </Button>
                    </Box>
                  </Box>
                ))}
              </Box>
            ) : (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <AnalysisIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                <Typography variant="body1" color="text.secondary">
                  No analyses have been run yet.
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Run your first analysis using the form on the left.
                </Typography>
              </Box>
            )}
          </CardContent>
        </Card>
      </Box>
    </Box>
  );
};

export default Analysis;