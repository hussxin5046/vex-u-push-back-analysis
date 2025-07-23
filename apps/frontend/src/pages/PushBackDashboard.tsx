import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  Chip,
  Alert,
  CircularProgress,
  Skeleton,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Build as BuildIcon,
  Analytics as AnalyticsIcon,
  Tune as TuneIcon,
  SportsEsports as GameIcon,
  TrendingUp as TrendingUpIcon,
  Timeline as TimelineIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { pushBackApiService } from '../services/pushBackApi';

interface DashboardStats {
  activeStrategies: number;
  completedAnalyses: number;
  averageWinRate: number;
  systemStatus: string;
}

interface RecentActivity {
  id: string;
  name: string;
  time: string;
  status: 'completed' | 'running' | 'failed';
  winRate?: number;
}

const PushBackDashboard: React.FC = () => {
  const navigate = useNavigate();
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [recentActivity, setRecentActivity] = useState<RecentActivity[]>([]);
  const [insights, setInsights] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch system status and health check
      const systemStatus = await pushBackApiService.getPushBackSystemStatus();
      
      // Fetch recent strategies
      const strategies = await pushBackApiService.getStrategies();
      
      // Run a quick sample analysis to get baseline stats
      const sampleRobotSpecs = {
        cycle_time: 5.0,
        pickup_reliability: 0.95,
        scoring_reliability: 0.98,
        autonomous_reliability: 0.88,
        max_capacity: 2,
        parking_capability: true
      };
      
      const quickAnalysis = await pushBackApiService.runMonteCarloSimulation(
        {
          id: 'dashboard-sample',
          name: 'Dashboard Sample',
          description: 'Quick analysis for dashboard',
          robot_specs: [sampleRobotSpecs],
          strategy_type: 'balanced',
          goal_priorities: { center: 0.6, long: 0.4 },
          autonomous_strategy: 'balanced',
          parking_strategy: 'late',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        },
        500 // Quick simulation for dashboard
      );
      
      // Get strategy archetypes for insights
      const archetypes = await pushBackApiService.getStrategyArchetypes();
      
      // Update dashboard stats
      setStats({
        activeStrategies: strategies.length,
        completedAnalyses: strategies.filter(s => s.updated_at).length,
        averageWinRate: quickAnalysis.win_rate,
        systemStatus: systemStatus.backend_status
      });
      
      // Generate recent activity from strategies
      const activity: RecentActivity[] = strategies.slice(0, 3).map((strategy, index) => ({
        id: strategy.id,
        name: strategy.name,
        time: getRelativeTime(strategy.updated_at),
        status: 'completed' as const,
        winRate: 0.65 + (index * 0.1) // Sample win rates
      }));
      
      setRecentActivity(activity);
      
      // Generate strategic insights
      const insightsList = [
        `Monte Carlo simulation shows ${(quickAnalysis.win_rate * 100).toFixed(1)}% average win rate`,
        `${Object.keys(archetypes).length} strategy archetypes available for optimization`,
        `System performance: ${systemStatus.backend_status.toUpperCase()}`
      ];
      
      setInsights(insightsList);
      
    } catch (err) {
      console.error('Dashboard data fetch error:', err);
      setError(err instanceof Error ? err.message : 'Failed to load dashboard data');
      
      // Set default data on error
      setStats({
        activeStrategies: 0,
        completedAnalyses: 0,
        averageWinRate: 0,
        systemStatus: 'unknown'
      });
    } finally {
      setLoading(false);
    }
  };
  
  const getRelativeTime = (dateString: string): string => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    
    if (diffHours < 1) return 'Less than 1 hour ago';
    if (diffHours < 24) return `${diffHours} hour${diffHours === 1 ? '' : 's'} ago`;
    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays} day${diffDays === 1 ? '' : 's'} ago`;
  };
  
  useEffect(() => {
    fetchDashboardData();
  }, []);

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 700 }}>
        Push Back Strategy Dashboard
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Welcome to your Push Back strategy development platform. Build, analyze, and optimize strategies for the VEX U Push Back season.
      </Typography>

      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} action={
          <Button color="inherit" size="small" onClick={fetchDashboardData}>
            <RefreshIcon sx={{ mr: 1 }} /> Retry
          </Button>
        }>
          {error}
        </Alert>
      )}

      {/* Quick Stats */}
      <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 2, mb: 4 }}>
        <Card sx={{ cursor: 'pointer' }} onClick={() => navigate('/strategy-builder')}>
          <CardContent sx={{ textAlign: 'center' }}>
            <BuildIcon sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              Active Strategies
            </Typography>
            {loading ? (
              <Skeleton variant="text" width={60} height={60} sx={{ mx: 'auto' }} />
            ) : (
              <Typography variant="h4" color="primary.main" sx={{ fontWeight: 700 }}>
                {stats?.activeStrategies || 0}
              </Typography>
            )}
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
            {loading ? (
              <Skeleton variant="text" width={60} height={60} sx={{ mx: 'auto' }} />
            ) : (
              <Typography variant="h4" color="success.main" sx={{ fontWeight: 700 }}>
                {stats?.completedAnalyses || 0}
              </Typography>
            )}
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
            {loading ? (
              <Skeleton variant="text" width={80} height={60} sx={{ mx: 'auto' }} />
            ) : (
              <Typography variant="h4" color="warning.main" sx={{ fontWeight: 700 }}>
                {((stats?.averageWinRate || 0) * 100).toFixed(0)}%
              </Typography>
            )}
            <Typography variant="body2" color="text.secondary">
              Monte Carlo avg
            </Typography>
          </CardContent>
        </Card>

        <Card>
          <CardContent sx={{ textAlign: 'center' }}>
            <GameIcon sx={{ fontSize: 40, color: 'secondary.main', mb: 1 }} />
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              System Status
            </Typography>
            {loading ? (
              <CircularProgress size={40} sx={{ my: 1 }} />
            ) : (
              <Typography variant="h4" color={stats?.systemStatus === 'healthy' ? 'success.main' : 'error.main'} sx={{ fontWeight: 700 }}>
                {stats?.systemStatus ? stats.systemStatus.charAt(0).toUpperCase() + stats.systemStatus.slice(1) : 'Unknown'}
              </Typography>
            )}
            <Typography variant="body2" color="text.secondary">
              Backend status
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
              {loading ? (
                [...Array(3)].map((_, index) => (
                  <Box key={index} sx={{ p: 2, backgroundColor: 'background.default', borderRadius: 1 }}>
                    <Skeleton variant="text" width="60%" height={24} />
                    <Skeleton variant="text" width="40%" height={20} />
                  </Box>
                ))
              ) : recentActivity.length > 0 ? (
                recentActivity.map((item) => (
                  <Box 
                    key={item.id}
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
                        {item.time} {item.winRate && `• ${(item.winRate * 100).toFixed(1)}% win rate`}
                      </Typography>
                    </Box>
                    <Chip 
                      label={item.status} 
                      color={item.status === 'completed' ? 'success' : item.status === 'running' ? 'warning' : 'error'} 
                      size="small" 
                      variant="outlined"
                    />
                  </Box>
                ))
              ) : (
                <Box sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="body2" color="text.secondary">
                    No recent activity. Create your first strategy!
                  </Typography>
                  <Button
                    variant="outlined"
                    size="small"
                    sx={{ mt: 1 }}
                    onClick={() => navigate('/strategy-builder')}
                  >
                    Get Started
                  </Button>
                </Box>
              )}
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
          {loading ? (
            <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 2 }}>
              {[...Array(3)].map((_, index) => (
                <Skeleton key={index} variant="rectangular" height={60} sx={{ borderRadius: 1 }} />
              ))}
            </Box>
          ) : (
            <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 2 }}>
              {insights.map((insight, index) => (
                <Alert 
                  key={index}
                  severity={index === 0 ? 'info' : index === 1 ? 'success' : 'warning'}
                >
                  {insight}
                </Alert>
              ))}
              {insights.length === 0 && (
                <Alert severity="info">
                  <strong>Getting Started:</strong> Create your first strategy to see personalized insights
                </Alert>
              )}
            </Box>
          )}
        </CardContent>
      </Card>
      
      {/* Refresh Button */}
      <Box sx={{ mt: 3, textAlign: 'center' }}>
        <Button
          variant="outlined"
          startIcon={loading ? <CircularProgress size={20} /> : <RefreshIcon />}
          onClick={fetchDashboardData}
          disabled={loading}
        >
          {loading ? 'Loading...' : 'Refresh Dashboard'}
        </Button>
      </Box>
    </Box>
  );
};

export default PushBackDashboard;