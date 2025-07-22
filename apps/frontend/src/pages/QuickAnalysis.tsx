import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
} from '@mui/material';
import {
  Analytics as AnalyticsIcon,
} from '@mui/icons-material';

const QuickAnalysis: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 700 }}>
        Quick Push Back Analysis
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Get instant strategic insights for your Push Back robot configuration.
      </Typography>

      <Card>
        <CardContent sx={{ textAlign: 'center', py: 6 }}>
          <AnalyticsIcon sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
          <Typography variant="h5" gutterBottom sx={{ fontWeight: 600 }}>
            Analysis Engine Coming Soon
          </Typography>
          <Typography variant="body1" color="text.secondary" paragraph>
            This will provide real-time analysis including:
          </Typography>
          <Box component="ul" sx={{ textAlign: 'left', display: 'inline-block' }}>
            <li>Block Flow Optimization</li>
            <li>Autonomous Strategy Analysis</li>
            <li>Parking Decision Timing</li>
            <li>Monte Carlo Simulations</li>
          </Box>
          <Box sx={{ mt: 3 }}>
            <Button variant="contained" size="large">
              Run Analysis
            </Button>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default QuickAnalysis;