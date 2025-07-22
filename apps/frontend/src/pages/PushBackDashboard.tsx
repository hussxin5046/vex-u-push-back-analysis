import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  Chip,
  Alert,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Build as BuildIcon,
  Analytics as AnalyticsIcon,
  Tune as TuneIcon,
  SportsEsports as GameIcon,
  TrendingUp as TrendingUpIcon,
  Timeline as TimelineIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

const PushBackDashboard: React.FC = () => {
  const navigate = useNavigate();

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 700 }}>
        Push Back Strategy Dashboard
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Welcome to your Push Back strategy development platform. Build, analyze, and optimize strategies for the VEX U Push Back season.
      </Typography>

      {/* Quick Stats */}
      <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 2, mb: 4 }}>
        <Card sx={{ cursor: 'pointer' }} onClick={() => navigate('/strategy-builder')}>
          <CardContent sx={{ textAlign: 'center' }}>
            <BuildIcon sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              Active Strategies
            </Typography>
            <Typography variant="h4" color="primary.main" sx={{ fontWeight: 700 }}>
              3
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Ready for analysis
            </Typography>
          </CardContent>
        </Card>

        <Card sx={{ cursor: 'pointer' }} onClick={() => navigate('/analysis')}>
          <CardContent sx={{ textAlign: 'center' }}>
            <AnalyticsIcon sx={{ fontSize: 40, color: 'success.main', mb: 1 }} />
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              Analyses Complete
            </Typography>
            <Typography variant="h4" color="success.main" sx={{ fontWeight: 700 }}>
              12
            </Typography>
            <Typography variant="body2" color="text.secondary">
              This session
            </Typography>
          </CardContent>
        </Card>

        <Card>
          <CardContent sx={{ textAlign: 'center' }}>
            <TrendingUpIcon sx={{ fontSize: 40, color: 'warning.main', mb: 1 }} />
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              Win Rate
            </Typography>
            <Typography variant="h4" color="warning.main" sx={{ fontWeight: 700 }}>
              73%
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Monte Carlo avg
            </Typography>
          </CardContent>
        </Card>

        <Card>
          <CardContent sx={{ textAlign: 'center' }}>
            <GameIcon sx={{ fontSize: 40, color: 'secondary.main', mb: 1 }} />
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              Field Elements
            </Typography>
            <Typography variant="h4" color="secondary.main" sx={{ fontWeight: 700 }}>
              88
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Blocks available
            </Typography>
          </CardContent>
        </Card>
      </Box>

      {/* Push Back Game Overview */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, display: 'flex', alignItems: 'center' }}>
            <GameIcon sx={{ mr: 1 }} />
            Push Back Game Overview
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
            <Box sx={{ flex: '1 1 50%', minWidth: 300 }}>
              <Typography variant="body1" paragraph>
                <strong>Field Layout:</strong> 12' x 12' field with 4 Goals (2 Center, 2 Long), 2 Control Zones, and 88 Blocks
              </Typography>
              <Typography variant="body1" paragraph>
                <strong>Scoring:</strong> 3 points per block, 6-10 points for zone control, 8/30 points for parking, 7 points for autonomous win
              </Typography>
              <Typography variant="body1">
                <strong>Strategy Focus:</strong> Block flow optimization, goal priority decisions, parking timing, and offense/defense balance
              </Typography>
            </Box>
            <Box sx={{ flex: '1 1 40%', minWidth: 250 }}>
              <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 1 }}>
                <Box sx={{ textAlign: 'center', p: 1, backgroundColor: 'primary.light', borderRadius: 1, color: 'white' }}>
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>4</Typography>
                  <Typography variant="caption">Goals</Typography>
                </Box>
                <Box sx={{ textAlign: 'center', p: 1, backgroundColor: 'secondary.light', borderRadius: 1, color: 'white' }}>
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>2</Typography>
                  <Typography variant="caption">Control Zones</Typography>
                </Box>
                <Box sx={{ textAlign: 'center', p: 1, backgroundColor: 'success.light', borderRadius: 1, color: 'white' }}>
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>88</Typography>
                  <Typography variant="caption">Blocks</Typography>
                </Box>
                <Box sx={{ textAlign: 'center', p: 1, backgroundColor: 'warning.light', borderRadius: 1, color: 'white' }}>
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>105</Typography>
                  <Typography variant="caption">Seconds</Typography>
                </Box>
              </Box>
            </Box>
          </Box>
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
        <Card sx={{ flex: '1 1 45%', minWidth: 300 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
              Strategy Development
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Create and optimize your Push Back strategies
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Button 
                variant="contained" 
                startIcon={<BuildIcon />}
                onClick={() => navigate('/strategy-builder')}
                fullWidth
              >
                Strategy Builder
              </Button>
              <Button 
                variant="outlined" 
                startIcon={<AnalyticsIcon />}
                onClick={() => navigate('/analysis')}
                fullWidth
              >
                Quick Analysis
              </Button>
              <Button 
                variant="outlined" 
                startIcon={<TuneIcon />}
                onClick={() => navigate('/decision-tools')}
                fullWidth
              >
                Decision Tools
              </Button>
            </Box>
          </CardContent>
        </Card>

        <Card sx={{ flex: '1 1 45%', minWidth: 300 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
              Recent Activity
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Latest strategy analyses and optimizations
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {[
                { name: "Block Flow Maximizer", time: "2 hours ago", status: "completed" },
                { name: "Control Zone Strategy", time: "4 hours ago", status: "completed" },
                { name: "Autonomous Optimizer", time: "1 day ago", status: "completed" }
              ].map((item, index) => (
                <Box 
                  key={index}
                  sx={{ 
                    p: 2, 
                    backgroundColor: 'background.default', 
                    borderRadius: 1,
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center'
                  }}
                >
                  <Box>
                    <Typography variant="body1" sx={{ fontWeight: 500 }}>
                      {item.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {item.time}
                    </Typography>
                  </Box>
                  <Chip 
                    label={item.status} 
                    color="success" 
                    size="small" 
                    variant="outlined"
                  />
                </Box>
              ))}
            </Box>
          </CardContent>
        </Card>
      </Box>

      {/* Strategic Insights */}
      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, display: 'flex', alignItems: 'center' }}>
            <TimelineIcon sx={{ mr: 1 }} />
            Push Back Strategic Insights
          </Typography>
          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 2 }}>
            <Alert severity="info">
              <strong>Block Flow:</strong> Optimize collection patterns for maximum efficiency
            </Alert>
            <Alert severity="warning">
              <strong>Parking:</strong> Time parking decisions based on score differential
            </Alert>
            <Alert severity="success">
              <strong>Control Zones:</strong> Maintain consistent zone presence for bonus points
            </Alert>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default PushBackDashboard;