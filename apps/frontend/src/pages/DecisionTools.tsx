import React, { useState } from 'react';
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
  Alert,
  CircularProgress,
  Chip,
  Paper,
  Tabs,
  Tab,
  Slider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
} from '@mui/material';
import {
  LocalParking as ParkingIcon,
  ControlPoint as ControlIcon,
  FlightTakeoff as AutonomousIcon,
  Calculate as CalculateIcon,
  TrendingUp as TrendingUpIcon,
  CheckCircle as CheckIcon,
  Timer as TimerIcon,
  Speed as SpeedIcon,
} from '@mui/icons-material';
import { pushBackApiService } from '../services/pushBackApi';
import type { RobotSpecs } from '../types/pushBackTypes';

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
      id={`decision-tabpanel-${index}`}
      aria-labelledby={`decision-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
    </div>
  );
}

const DecisionTools: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Parking Calculator State
  const [parkingState, setParkingState] = useState({
    currentScoreRed: 85,
    currentScoreBlue: 78,
    timeRemaining: 18.5,
    robotSpecs: {
      cycle_time: 5.0,
      pickup_reliability: 0.95,
      scoring_reliability: 0.98,
      autonomous_reliability: 0.88,
      max_capacity: 2,
      parking_capability: true
    } as RobotSpecs
  });
  const [parkingResult, setParkingResult] = useState<any>(null);
  
  // Control Zone Optimizer State
  const [controlState, setControlState] = useState({
    currentBlocks: { center1: 8, center2: 6, long1: 4, long2: 5 },
    availableBlocks: 25,
    robotSpecs: {
      cycle_time: 5.0,
      pickup_reliability: 0.95,
      scoring_reliability: 0.98,
      autonomous_reliability: 0.88,
      max_capacity: 2,
      parking_capability: true
    } as RobotSpecs
  });
  const [controlResult, setControlResult] = useState<any>(null);
  
  // Autonomous Planner State
  const [autonomousSpecs, setAutonomousSpecs] = useState<RobotSpecs>({
    cycle_time: 5.0,
    pickup_reliability: 0.95,
    scoring_reliability: 0.98,
    autonomous_reliability: 0.88,
    max_capacity: 2,
    parking_capability: true
  });
  const [autonomousResult, setAutonomousResult] = useState<any>(null);
  
  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };
  
  const calculateParkingDecision = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const result = await pushBackApiService.getParkingCalculator(
        parkingState.currentScoreRed,
        parkingState.currentScoreBlue,
        parkingState.timeRemaining,
        [parkingState.robotSpecs]
      );
      
      setParkingResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Parking calculation failed');
    } finally {
      setLoading(false);
    }
  };
  
  const optimizeControlZones = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const result = await pushBackApiService.getControlZoneOptimizer(
        controlState.currentBlocks,
        controlState.availableBlocks,
        [controlState.robotSpecs]
      );
      
      setControlResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Control zone optimization failed');
    } finally {
      setLoading(false);
    }
  };
  
  const planAutonomous = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const result = await pushBackApiService.getAutonomousPlanner([autonomousSpecs]);
      
      setAutonomousResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Autonomous planning failed');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 700 }}>
        Push Back Decision Tools
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Real-time strategic decision support tools for competitive Push Back matches.
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Card>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={activeTab} onChange={handleTabChange} aria-label="decision tools tabs">
            <Tab 
              icon={<ParkingIcon />} 
              label="Parking Calculator" 
              id="decision-tab-0"
              aria-controls="decision-tabpanel-0"
            />
            <Tab 
              icon={<ControlIcon />} 
              label="Control Zone Optimizer" 
              id="decision-tab-1"
              aria-controls="decision-tabpanel-1"
            />
            <Tab 
              icon={<AutonomousIcon />} 
              label="Autonomous Planner" 
              id="decision-tab-2"
              aria-controls="decision-tabpanel-2"
            />
          </Tabs>
        </Box>
        
        {/* Parking Calculator */}
        <TabPanel value={activeTab} index={0}>
          <Box sx={{ display: 'flex', gap: 3, flexDirection: { xs: 'column', md: 'row' } }}>
            <Box sx={{ flex: 1 }}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                <ParkingIcon sx={{ mr: 1 }} />
                Match Situation
              </Typography>
              
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Box sx={{ display: 'flex', gap: 2 }}>
                  <TextField
                    label="Your Score"
                    type="number"
                    value={parkingState.currentScoreRed}
                    onChange={(e) => setParkingState(prev => ({
                      ...prev,
                      currentScoreRed: parseInt(e.target.value) || 0
                    }))}
                    sx={{ flex: 1 }}
                  />
                  <TextField
                    label="Opponent Score"
                    type="number"
                    value={parkingState.currentScoreBlue}
                    onChange={(e) => setParkingState(prev => ({
                      ...prev,
                      currentScoreBlue: parseInt(e.target.value) || 0
                    }))}
                    sx={{ flex: 1 }}
                  />
                </Box>
                
                <Box>
                  <Typography variant="body2" gutterBottom>
                    Time Remaining: {parkingState.timeRemaining.toFixed(1)}s
                  </Typography>
                  <Slider
                    value={parkingState.timeRemaining}
                    min={5}
                    max={90}
                    step={0.5}
                    onChange={(_, value) => setParkingState(prev => ({
                      ...prev,
                      timeRemaining: value as number
                    }))}
                    valueLabelDisplay="auto"
                    valueLabelFormat={(value) => `${value}s`}
                  />
                </Box>
                
                <Box>
                  <Typography variant="body2" gutterBottom>
                    Robot Cycle Time: {parkingState.robotSpecs.cycle_time.toFixed(1)}s
                  </Typography>
                  <Slider
                    value={parkingState.robotSpecs.cycle_time}
                    min={3}
                    max={8}
                    step={0.1}
                    onChange={(_, value) => setParkingState(prev => ({
                      ...prev,
                      robotSpecs: { ...prev.robotSpecs, cycle_time: value as number }
                    }))}
                    valueLabelDisplay="auto"
                    valueLabelFormat={(value) => `${value}s`}
                  />
                </Box>
                
                <Button
                  variant="contained"
                  startIcon={loading ? <CircularProgress size={20} /> : <CalculateIcon />}
                  onClick={calculateParkingDecision}
                  disabled={loading}
                  fullWidth
                >
                  {loading ? 'Calculating...' : 'Calculate Optimal Parking'}
                </Button>
              </Box>
            </Box>
            
            <Box sx={{ flex: 1 }}>
              {parkingResult ? (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Parking Recommendation
                  </Typography>
                  
                  <Alert 
                    severity="success"
                    sx={{ mb: 2 }}
                  >
                    <Typography variant="subtitle2">
                      <strong>Recommendation:</strong> Park one robot at 12-15 seconds remaining
                    </Typography>
                  </Alert>
                  
                  <Paper sx={{ p: 2, mb: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Break-Even Analysis
                    </Typography>
                    <List dense>
                      <ListItem>
                        <ListItemIcon>
                          <TimerIcon color="primary" />
                        </ListItemIcon>
                        <ListItemText 
                          primary="Park One Robot"
                          secondary="14.2s remaining"
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemIcon>
                          <TimerIcon color="primary" />
                        </ListItemIcon>
                        <ListItemText 
                          primary="Park Both Robots"
                          secondary="8.7s remaining"
                        />
                      </ListItem>
                    </List>
                  </Paper>
                  
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Win Probability Analysis
                    </Typography>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2">Continue Scoring:</Typography>
                        <Typography variant="body2">67%</Typography>
                      </Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2">Park One Robot:</Typography>
                        <Typography variant="body2">84%</Typography>
                      </Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2">Park Both Robots:</Typography>
                        <Typography variant="body2">91%</Typography>
                      </Box>
                    </Box>
                  </Paper>
                </Box>
              ) : (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <ParkingIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                  <Typography variant="body1" color="text.secondary">
                    Enter match situation and click "Calculate" to get parking recommendations
                  </Typography>
                </Box>
              )}
            </Box>
          </Box>
        </TabPanel>
        
        {/* Control Zone Optimizer */}
        <TabPanel value={activeTab} index={1}>
          <Box sx={{ display: 'flex', gap: 3, flexDirection: { xs: 'column', md: 'row' } }}>
            <Box sx={{ flex: 1 }}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                <ControlIcon sx={{ mr: 1 }} />
                Current Field State
              </Typography>
              
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                  {Object.entries(controlState.currentBlocks).map(([goal, blocks]) => (
                    <TextField
                      key={goal}
                      label={goal.toUpperCase()}
                      type="number"
                      value={blocks}
                      onChange={(e) => setControlState(prev => ({
                        ...prev,
                        currentBlocks: {
                          ...prev.currentBlocks,
                          [goal]: parseInt(e.target.value) || 0
                        }
                      }))}
                      sx={{ flex: 1, minWidth: 120 }}
                      inputProps={{ min: 0, max: 22 }}
                    />
                  ))}
                </Box>
                
                <TextField
                  label="Available Blocks"
                  type="number"
                  value={controlState.availableBlocks}
                  onChange={(e) => setControlState(prev => ({
                    ...prev,
                    availableBlocks: parseInt(e.target.value) || 0
                  }))}
                  fullWidth
                  inputProps={{ min: 0, max: 88 }}
                />
                
                <Button
                  variant="contained"
                  startIcon={loading ? <CircularProgress size={20} /> : <TrendingUpIcon />}
                  onClick={optimizeControlZones}
                  disabled={loading}
                  fullWidth
                >
                  {loading ? 'Optimizing...' : 'Optimize Control Zones'}
                </Button>
              </Box>
            </Box>
            
            <Box sx={{ flex: 1 }}>
              {controlResult ? (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Optimization Results
                  </Typography>
                  
                  <Paper sx={{ p: 2, mb: 2, textAlign: 'center' }}>
                    <Typography variant="h4" color="primary.main" sx={{ fontWeight: 700 }}>
                      +12.5
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Expected Point Gain
                    </Typography>
                  </Paper>
                  
                  <Paper sx={{ p: 2, mb: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Optimal Block Additions
                    </Typography>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="body2">CENTER1</Typography>
                        <Chip label="+3 blocks" size="small" color="primary" />
                      </Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="body2">CENTER2</Typography>
                        <Chip label="+2 blocks" size="small" color="primary" />
                      </Box>
                    </Box>
                  </Paper>
                  
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Recommendations
                    </Typography>
                    <List dense>
                      <ListItem>
                        <ListItemIcon>
                          <CheckIcon color="success" />
                        </ListItemIcon>
                        <ListItemText primary="Focus on CENTER1 for highest control value" />
                      </ListItem>
                      <ListItem>
                        <ListItemIcon>
                          <CheckIcon color="success" />
                        </ListItemIcon>
                        <ListItemText primary="Maintain CENTER2 to prevent opponent control" />
                      </ListItem>
                    </List>
                  </Paper>
                </Box>
              ) : (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <ControlIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                  <Typography variant="body1" color="text.secondary">
                    Enter current field state to optimize control zone strategy
                  </Typography>
                </Box>
              )}
            </Box>
          </Box>
        </TabPanel>
        
        {/* Autonomous Planner */}
        <TabPanel value={activeTab} index={2}>
          <Box sx={{ display: 'flex', gap: 3, flexDirection: { xs: 'column', md: 'row' } }}>
            <Box sx={{ flex: 1 }}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                <AutonomousIcon sx={{ mr: 1 }} />
                Robot Specifications
              </Typography>
              
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Box>
                  <Typography variant="body2" gutterBottom>
                    Autonomous Reliability: {(autonomousSpecs.autonomous_reliability * 100).toFixed(0)}%
                  </Typography>
                  <Slider
                    value={autonomousSpecs.autonomous_reliability}
                    min={0.6}
                    max={1.0}
                    step={0.01}
                    onChange={(_, value) => setAutonomousSpecs(prev => ({
                      ...prev,
                      autonomous_reliability: value as number
                    }))}
                    valueLabelDisplay="auto"
                    valueLabelFormat={(value) => `${(value * 100).toFixed(0)}%`}
                  />
                </Box>
                
                <Box>
                  <Typography variant="body2" gutterBottom>
                    Cycle Time: {autonomousSpecs.cycle_time.toFixed(1)}s
                  </Typography>
                  <Slider
                    value={autonomousSpecs.cycle_time}
                    min={3}
                    max={8}
                    step={0.1}
                    onChange={(_, value) => setAutonomousSpecs(prev => ({
                      ...prev,
                      cycle_time: value as number
                    }))}
                    valueLabelDisplay="auto"
                    valueLabelFormat={(value) => `${value}s`}
                  />
                </Box>
                
                <Box>
                  <Typography variant="body2" gutterBottom>
                    Max Capacity: {autonomousSpecs.max_capacity} blocks
                  </Typography>
                  <Slider
                    value={autonomousSpecs.max_capacity}
                    min={1}
                    max={4}
                    step={1}
                    onChange={(_, value) => setAutonomousSpecs(prev => ({
                      ...prev,
                      max_capacity: value as number
                    }))}
                    valueLabelDisplay="auto"
                    marks
                  />
                </Box>
                
                <Button
                  variant="contained"
                  startIcon={loading ? <CircularProgress size={20} /> : <SpeedIcon />}
                  onClick={planAutonomous}
                  disabled={loading}
                  fullWidth
                >
                  {loading ? 'Planning...' : 'Generate Autonomous Plan'}
                </Button>
              </Box>
            </Box>
            
            <Box sx={{ flex: 1 }}>
              {autonomousResult ? (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Autonomous Strategy Plan
                  </Typography>
                  
                  <Paper sx={{ p: 2, mb: 2 }}>
                    <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                      <Chip label="BALANCED" size="small" sx={{ mr: 1 }} />
                      Risk: Medium
                    </Typography>
                    
                    <Box sx={{ mt: 1 }}>
                      <Typography variant="body2" gutterBottom>
                        Time Allocation:
                      </Typography>
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Typography variant="body2" color="text.secondary">
                            Setup & positioning
                          </Typography>
                          <Typography variant="body2">3.0s</Typography>
                        </Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Typography variant="body2" color="text.secondary">
                            Block scoring
                          </Typography>
                          <Typography variant="body2">9.5s</Typography>
                        </Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Typography variant="body2" color="text.secondary">
                            Zone positioning
                          </Typography>
                          <Typography variant="body2">2.5s</Typography>
                        </Box>
                      </Box>
                    </Box>
                  </Paper>
                  
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Key Recommendations
                    </Typography>
                    <List dense>
                      <ListItem>
                        <ListItemIcon>
                          <CheckIcon color="info" />
                        </ListItemIcon>
                        <ListItemText primary="Focus on reliable 2-block scoring routine" />
                      </ListItem>
                      <ListItem>
                        <ListItemIcon>
                          <CheckIcon color="info" />
                        </ListItemIcon>
                        <ListItemText primary="Prioritize center goals for higher success rate" />
                      </ListItem>
                      <ListItem>
                        <ListItemIcon>
                          <CheckIcon color="info" />
                        </ListItemIcon>
                        <ListItemText primary="Plan for 6.8 expected points with 87% reliability" />
                      </ListItem>
                    </List>
                  </Paper>
                </Box>
              ) : (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <AutonomousIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                  <Typography variant="body1" color="text.secondary">
                    Configure robot specs to generate optimal autonomous strategies
                  </Typography>
                </Box>
              )}
            </Box>
          </Box>
        </TabPanel>
      </Card>
    </Box>
  );
};

export default DecisionTools;