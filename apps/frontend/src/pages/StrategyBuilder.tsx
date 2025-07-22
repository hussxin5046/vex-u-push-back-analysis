import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
} from '@mui/material';
import {
  Build as BuildIcon,
} from '@mui/icons-material';

const StrategyBuilder: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 700 }}>
        Push Back Strategy Builder
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Create and customize your Push Back strategy using our guided builder.
      </Typography>

      <Card>
        <CardContent sx={{ textAlign: 'center', py: 6 }}>
          <BuildIcon sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
          <Typography variant="h5" gutterBottom sx={{ fontWeight: 600 }}>
            Strategy Builder Coming Soon
          </Typography>
          <Typography variant="body1" color="text.secondary" paragraph>
            This will be a comprehensive strategy builder featuring:
          </Typography>
          <Box component="ul" sx={{ textAlign: 'left', display: 'inline-block' }}>
            <li>Robot Configuration Setup</li>
            <li>Strategy Archetype Selection</li>
            <li>Priority Settings Configuration</li>
            <li>Review & Save Functionality</li>
          </Box>
          <Box sx={{ mt: 3 }}>
            <Button variant="contained" size="large">
              Start Building
            </Button>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default StrategyBuilder;