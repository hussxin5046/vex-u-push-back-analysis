import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  LinearProgress,
  Alert,
  Paper,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Psychology as AnalysisIcon,
  SmartToy as RobotIcon,
  Assessment as ReportsIcon,
  Timeline as TimelineIcon,
  TrendingUp as TrendingUpIcon,
} from '@mui/icons-material';
import { 
  useSystemStatus, 
  useMLModelStatus, 
  useAnalysisHistory,
  useHealthCheck 
} from '../hooks/useApi';

interface StatCardProps {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  color: 'primary' | 'secondary' | 'success' | 'warning' | 'error';
  subtitle?: string;
}

const StatCard: React.FC<StatCardProps> = ({ title, value, icon, color, subtitle }) => (
  <Card sx={{ height: '100%' }}>
    <CardContent>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <Box 
          sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            width: 48,
            height: 48,
            borderRadius: 2,
            backgroundColor: (theme) => theme.palette[color].main + '20',
            color: (theme) => theme.palette[color].main,
            mr: 2,
          }}
        >
          {icon}
        </Box>
        <Box sx={{ flex: 1 }}>
          <Typography variant="h6" component="div" sx={{ fontWeight: 600 }}>
            {title}
          </Typography>
          {subtitle && (
            <Typography variant="body2" color="text.secondary">
              {subtitle}
            </Typography>
          )}
        </Box>
      </Box>
      <Typography variant="h4" component="div" sx={{ fontWeight: 700 }}>
        {value}
      </Typography>
    </CardContent>
  </Card>
);

const Dashboard: React.FC = () => {
  const { data: systemStatus, isLoading: systemLoading } = useSystemStatus();
  const { data: mlStatus, isLoading: mlLoading } = useMLModelStatus();
  const { data: analysisHistory = [], isLoading: analysisLoading } = useAnalysisHistory();
  const { data: isBackendConnected = false } = useHealthCheck();

  if (systemLoading || mlLoading) {
    return (
      <Box sx={{ width: '100%', mt: 2 }}>
        <LinearProgress />
      </Box>
    );
  }

  const recentAnalyses = Array.isArray(analysisHistory) ? analysisHistory.slice(0, 5) : [];
  const mlModelsAvailable = mlStatus ? Object.values(mlStatus).filter(Boolean).length : 0;
  const totalMLModels = mlStatus ? Object.keys(mlStatus).length : 0;

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 700 }}>
        Dashboard
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Welcome to the VEX U Scoring Analysis Platform. Monitor your system status and access key insights.
      </Typography>

      {!isBackendConnected && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          Backend connection is offline. Some features may not be available.
        </Alert>
      )}

      {/* System Status Cards */}
      <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 3, mb: 3 }}>
        <StatCard
          title="System Status"
          value={isBackendConnected ? "Online" : "Offline"}
          icon={<DashboardIcon />}
          color={isBackendConnected ? "success" : "error"}
          subtitle="Backend Connection"
        />
        <StatCard
          title="ML Models"
          value={`${mlModelsAvailable}/${totalMLModels}`}
          icon={<RobotIcon />}
          color={mlModelsAvailable === totalMLModels ? "success" : "warning"}
          subtitle="Models Available"
        />
        <StatCard
          title="Analyses"
          value={Array.isArray(analysisHistory) ? analysisHistory.length : 0}
          icon={<AnalysisIcon />}
          color="primary"
          subtitle="Total Completed"
        />
        <StatCard
          title="Recent Activity"
          value={recentAnalyses.length}
          icon={<TimelineIcon />}
          color="secondary"
          subtitle="Last 24 hours"
        />
      </Box>

      {/* Content Grid */}
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 3 }}>
        {/* ML Model Status */}
        <Card sx={{ height: 'fit-content' }}>
          <CardContent>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
              ML Model Status
            </Typography>
            {mlStatus ? (
              <Box sx={{ mt: 2 }}>
                {Object.entries(mlStatus).map(([model, available]) => (
                  <Box key={model} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                    <Typography variant="body1" sx={{ textTransform: 'capitalize' }}>
                      {model.replace('_', ' ')}
                    </Typography>
                    <Chip
                      label={available ? 'Available' : 'Unavailable'}
                      color={available ? 'success' : 'error'}
                      size="small"
                      variant="outlined"
                    />
                  </Box>
                ))}
              </Box>
            ) : (
              <Typography color="text.secondary">
                ML model status unavailable
              </Typography>
            )}
          </CardContent>
        </Card>

        {/* Recent Analyses */}
        <Card sx={{ height: 'fit-content' }}>
          <CardContent>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
              Recent Analyses
            </Typography>
            {analysisLoading ? (
              <LinearProgress />
            ) : recentAnalyses.length > 0 ? (
              <Box sx={{ mt: 2 }}>
                {recentAnalyses.map((analysis: any) => (
                  <Paper
                    key={analysis.id}
                    elevation={0}
                    sx={{
                      p: 2,
                      mb: 1,
                      backgroundColor: (theme) => theme.palette.background.default,
                      borderRadius: 1,
                    }}
                  >
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Box>
                        <Typography variant="body1" sx={{ fontWeight: 500 }}>
                          {analysis.title}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {analysis.type.replace('_', ' ').toUpperCase()}
                        </Typography>
                      </Box>
                      <Chip
                        label={new Date(analysis.createdAt).toLocaleDateString()}
                        size="small"
                        variant="outlined"
                      />
                    </Box>
                  </Paper>
                ))}
              </Box>
            ) : (
              <Typography color="text.secondary">
                No recent analyses available
              </Typography>
            )}
          </CardContent>
        </Card>
      </Box>

      {/* Quick Actions */}
      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
            Quick Actions
          </Typography>
          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 2, mt: 2 }}>
            {[
              { icon: AnalysisIcon, label: 'Run Analysis', color: 'primary.main' },
              { icon: RobotIcon, label: 'ML Prediction', color: 'secondary.main' },
              { icon: ReportsIcon, label: 'Generate Report', color: 'success.main' },
              { icon: TrendingUpIcon, label: 'View Trends', color: 'warning.main' },
            ].map(({ icon: Icon, label, color }) => (
              <Box
                key={label}
                sx={{
                  p: 2,
                  textAlign: 'center',
                  border: '1px dashed',
                  borderColor: color,
                  borderRadius: 2,
                  cursor: 'pointer',
                  '&:hover': {
                    backgroundColor: color,
                    color: 'white',
                  },
                }}
              >
                <Icon sx={{ fontSize: 40, mb: 1 }} />
                <Typography variant="body1" sx={{ fontWeight: 500 }}>
                  {label}
                </Typography>
              </Box>
            ))}
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default Dashboard;