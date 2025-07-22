import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  Paper,
  Tabs,
  Tab,
  Chip,
  Alert,
  IconButton,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress,
} from '@mui/material';
import {
  Psychology as MLIcon,
  PlayArrow as TrainIcon,
  TrendingUp as PredictIcon,
  Speed as OptimizeIcon,
  Info as InfoIcon,
  CheckCircle as SuccessIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Refresh as RefreshIcon,
  Help as HelpIcon,
  Close as CloseIcon,
} from '@mui/icons-material';
import { useNotification } from '../contexts/NotificationContext';
import { useMLModelStatus, useTrainMLModel, useMLPrediction } from '../hooks/useApi';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`ml-tabpanel-${index}`}
      aria-labelledby={`ml-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

const MLModels: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [showGuide, setShowGuide] = useState(true);
  const [selectedModel, setSelectedModel] = useState('');
  const [trainingParams, setTrainingParams] = useState({
    epochs: 100,
    batchSize: 32,
    learningRate: 0.001,
  });

  const { showNotification } = useNotification();
  const { data: mlStatus, isLoading: statusLoading, refetch: refetchStatus } = useMLModelStatus();
  const trainModelMutation = useTrainMLModel();
  const mlPredictionMutation = useMLPrediction();

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const availableModels = [
    {
      id: 'scoring_optimizer',
      name: 'Scoring Optimizer',
      description: 'Predicts optimal scoring strategies based on game state',
      status: mlStatus?.scoring_optimizer ? 'trained' : 'not_trained',
      accuracy: 0.85, // Placeholder until API provides actual accuracy
    },
    {
      id: 'coordination',
      name: 'Robot Coordination',
      description: 'Optimizes robot task allocation and teamwork',
      status: mlStatus?.coordination ? 'trained' : 'not_trained',
      accuracy: 0.82,
    },
    {
      id: 'strategy_predictor',
      name: 'Strategy Predictor',
      description: 'Classifies and recommends strategies based on match conditions',
      status: mlStatus?.strategy_predictor ? 'trained' : 'not_trained',
      accuracy: 0.88,
    },
  ];

  const handleTrainModel = async (modelId: string) => {
    try {
      // Frontend model IDs already match backend enum values
      const modelType = modelId;

      await trainModelMutation.mutateAsync({
        modelType: modelType,
        params: {
          epochs: trainingParams.epochs,
          batch_size: trainingParams.batchSize,
          learning_rate: trainingParams.learningRate,
          validation_split: 0.2,
          early_stopping: true,
        },
      });
      
      showNotification({
        type: 'success',
        title: 'Training Started',
        message: `${availableModels.find(m => m.id === modelId)?.name} training has been initiated. This may take several minutes.`,
      });
      
      // Refetch status after a delay
      setTimeout(() => refetchStatus(), 5000);
    } catch (error: any) {
      showNotification({
        type: 'error',
        title: 'Training Failed',
        message: error?.response?.data?.error || 'Failed to start model training. Please try again.',
      });
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'trained':
        return <SuccessIcon color="success" />;
      case 'training':
        return <CircularProgress size={20} />;
      case 'failed':
        return <ErrorIcon color="error" />;
      default:
        return <WarningIcon color="warning" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'trained':
        return 'success';
      case 'training':
        return 'info';
      case 'failed':
        return 'error';
      default:
        return 'warning';
    }
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box>
          <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 700 }}>
            Machine Learning Models
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Train and manage ML models for VEX U strategic analysis
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <IconButton onClick={() => refetchStatus()} disabled={statusLoading}>
            <RefreshIcon />
          </IconButton>
          <Button
            startIcon={<HelpIcon />}
            onClick={() => setShowGuide(true)}
            variant="outlined"
            sx={{ height: 'fit-content' }}
          >
            Show Guide
          </Button>
        </Box>
      </Box>

      {/* ML Guide */}
      {showGuide && (
        <Card sx={{ 
          mb: 3, 
          backgroundColor: (theme) => theme.palette.mode === 'dark' 
            ? 'rgba(156, 39, 176, 0.08)' 
            : '#f3e5f5',
          border: (theme) => `1px solid ${theme.palette.secondary.main}`
        }}>
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6" sx={{ fontWeight: 600, color: 'secondary.main' }}>
                <InfoIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                ML Models Guide
              </Typography>
              <IconButton onClick={() => setShowGuide(false)} size="small">
                <CloseIcon />
              </IconButton>
            </Box>
            
            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(3, 1fr)' }, gap: 3 }}>
              <Box>
                <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                  🧠 Available Models
                </Typography>
                <Typography variant="body2">
                  • <strong>Scoring Optimizer:</strong> Predicts scores and win probability<br/>
                  • <strong>Coordination:</strong> Optimizes robot teamwork<br/>
                  • <strong>Strategy Predictor:</strong> Recommends best strategies<br/>
                  • Each model requires training before use
                </Typography>
              </Box>
              
              <Box>
                <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                  🎯 Model Training
                </Typography>
                <Typography variant="body2">
                  • Training uses synthetic VEX U match data<br/>
                  • Takes 5-10 minutes per model<br/>
                  • Higher epochs = better accuracy<br/>
                  • Models auto-save after training
                </Typography>
              </Box>
              
              <Box>
                <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                  📊 Using Models
                </Typography>
                <Typography variant="body2">
                  • Train models first in the Training tab<br/>
                  • Use Predictions tab for analysis<br/>
                  • Optimization combines all models<br/>
                  • Accuracy improves with more data
                </Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Status Overview */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
            Model Status Overview
          </Typography>
          
          {statusLoading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
              <CircularProgress />
            </Box>
          ) : (
            <List>
              {availableModels.map((model, index) => (
                <React.Fragment key={model.id}>
                  {index > 0 && <Divider />}
                  <ListItem>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          {model.name}
                          <Chip
                            label={model.status.replace('_', ' ')}
                            size="small"
                            color={getStatusColor(model.status) as any}
                            icon={getStatusIcon(model.status)}
                          />
                        </Box>
                      }
                      secondary={
                        <Box>
                          <Typography variant="body2" color="text.secondary">
                            {model.description}
                          </Typography>
                          {model.status === 'trained' && (
                            <Typography variant="caption" color="success.main">
                              Accuracy: {(model.accuracy * 100).toFixed(1)}%
                            </Typography>
                          )}
                        </Box>
                      }
                    />
                    <ListItemSecondaryAction>
                      <Button
                        variant="outlined"
                        size="small"
                        startIcon={<TrainIcon />}
                        onClick={() => handleTrainModel(model.id)}
                        disabled={model.status === 'training' || trainModelMutation.isPending}
                      >
                        {model.status === 'training' ? 'Training...' : 'Train'}
                      </Button>
                    </ListItemSecondaryAction>
                  </ListItem>
                </React.Fragment>
              ))}
            </List>
          )}
        </CardContent>
      </Card>

      {/* Tabs */}
      <Paper sx={{ borderRadius: 2 }}>
        <Tabs
          value={activeTab}
          onChange={handleTabChange}
          indicatorColor="primary"
          textColor="primary"
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          <Tab 
            label="Training" 
            icon={<TrainIcon />} 
            iconPosition="start"
            sx={{ minHeight: 48 }}
          />
          <Tab 
            label="Predictions" 
            icon={<PredictIcon />} 
            iconPosition="start"
            sx={{ minHeight: 48 }}
          />
          <Tab 
            label="Optimization" 
            icon={<OptimizeIcon />} 
            iconPosition="start"
            sx={{ minHeight: 48 }}
          />
        </Tabs>

        <Box sx={{ p: 3 }}>
          {/* Training Tab */}
          <TabPanel value={activeTab} index={0}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                  Train ML Models
                </Typography>
                
                <Alert severity="info" sx={{ mb: 3 }}>
                  Training uses synthetic VEX U match data to learn patterns and strategies. 
                  Each model focuses on a specific aspect of game analysis.
                </Alert>

                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel>Select Model to Train</InputLabel>
                  <Select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                  >
                    {availableModels.map((model) => (
                      <MenuItem key={model.id} value={model.id}>
                        {model.name} - {model.status}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>

                {selectedModel && (
                  <Box sx={{ mt: 3 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Training Parameters
                    </Typography>
                    
                    <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 2, mb: 3 }}>
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          Epochs: {trainingParams.epochs}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          More epochs = better accuracy
                        </Typography>
                      </Box>
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          Batch Size: {trainingParams.batchSize}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Affects training speed
                        </Typography>
                      </Box>
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          Learning Rate: {trainingParams.learningRate}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Model convergence speed
                        </Typography>
                      </Box>
                    </Box>

                    <Button
                      variant="contained"
                      fullWidth
                      size="large"
                      startIcon={<TrainIcon />}
                      onClick={() => handleTrainModel(selectedModel)}
                      disabled={trainModelMutation.isPending}
                    >
                      {trainModelMutation.isPending ? 'Training in Progress...' : 'Start Training'}
                    </Button>
                  </Box>
                )}
              </CardContent>
            </Card>
          </TabPanel>

          {/* Predictions Tab */}
          <TabPanel value={activeTab} index={1}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                  ML-Powered Predictions
                </Typography>
                
                <Alert severity="warning" sx={{ mb: 3 }}>
                  You need to train models before making predictions. 
                  Check the model status above to see which models are available.
                </Alert>

                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <PredictIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                  <Typography variant="h6" color="text.secondary" gutterBottom>
                    Prediction Interface Coming Soon
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Once models are trained, you'll be able to:
                  </Typography>
                  <List sx={{ maxWidth: 400, mx: 'auto', mt: 2 }}>
                    <ListItem>
                      <ListItemText 
                        primary="Predict match outcomes"
                        secondary="Based on team strategies and historical data"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Get strategy recommendations"
                        secondary="AI-suggested strategies for specific opponents"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Analyze win probabilities"
                        secondary="Real-time probability calculations"
                      />
                    </ListItem>
                  </List>
                </Box>
              </CardContent>
            </Card>
          </TabPanel>

          {/* Optimization Tab */}
          <TabPanel value={activeTab} index={2}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                  Strategy Optimization
                </Typography>
                
                <Alert severity="info" sx={{ mb: 3 }}>
                  Optimization combines all trained models to find the best strategies 
                  for your specific match conditions and opponents.
                </Alert>

                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <OptimizeIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                  <Typography variant="h6" color="text.secondary" gutterBottom>
                    Optimization Engine Coming Soon
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Advanced features will include:
                  </Typography>
                  <List sx={{ maxWidth: 400, mx: 'auto', mt: 2 }}>
                    <ListItem>
                      <ListItemText 
                        primary="Multi-objective optimization"
                        secondary="Balance scoring, defense, and risk"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Real-time strategy adaptation"
                        secondary="Adjust strategies during matches"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Opponent analysis"
                        secondary="Counter-strategies based on opponent patterns"
                      />
                    </ListItem>
                  </List>
                </Box>
              </CardContent>
            </Card>
          </TabPanel>
        </Box>
      </Paper>
    </Box>
  );
};

export default MLModels;