import React from 'react';
import {
  Card,
  CardContent,
  CardActions,
  Typography,
  Box,
  Chip,
  Button,
  IconButton,
  LinearProgress,
  Collapse,
  Alert,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Download as DownloadIcon,
  Share as ShareIcon,
  Visibility as ViewIcon,
  Assessment as ChartIcon,
  TrendingUp as TrendingUpIcon,
  Psychology as AnalysisIcon,
} from '@mui/icons-material';
import { AnalysisResult } from '../types';

interface AnalysisCardProps {
  analysis: AnalysisResult;
  onView?: (analysis: AnalysisResult) => void;
  onDownload?: (analysis: AnalysisResult) => void;
  onShare?: (analysis: AnalysisResult) => void;
  showDetails?: boolean;
  loading?: boolean;
  variant?: 'default' | 'compact' | 'detailed';
}

const AnalysisCard: React.FC<AnalysisCardProps> = ({
  analysis,
  onView,
  onDownload,
  onShare,
  showDetails = true,
  loading = false,
  variant = 'default',
}) => {
  const [expanded, setExpanded] = React.useState(false);

  const getAnalysisIcon = (type: string) => {
    switch (type) {
      case 'scoring':
        return <TrendingUpIcon />;
      case 'strategy':
        return <AnalysisIcon />;
      case 'statistical':
        return <ChartIcon />;
      case 'ml_prediction':
        return <AnalysisIcon />;
      default:
        return <ChartIcon />;
    }
  };

  const getAnalysisColor = (type: string) => {
    switch (type) {
      case 'scoring':
        return 'success';
      case 'strategy':
        return 'primary';
      case 'statistical':
        return 'secondary';
      case 'ml_prediction':
        return 'warning';
      default:
        return 'default';
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return {
      date: date.toLocaleDateString(),
      time: date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };
  };

  const { date, time } = formatDate(analysis.createdAt);

  if (variant === 'compact') {
    return (
      <Card sx={{ mb: 1 }}>
        <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', flex: 1 }}>
              <Box sx={{ mr: 2, color: `${getAnalysisColor(analysis.type)}.main` }}>
                {getAnalysisIcon(analysis.type)}
              </Box>
              <Box sx={{ flex: 1 }}>
                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                  {analysis.title}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {analysis.type.replace('_', ' ').toUpperCase()} • {date}
                </Typography>
              </Box>
            </Box>
            <Box sx={{ display: 'flex', gap: 0.5 }}>
              {onView && (
                <IconButton size="small" onClick={() => onView(analysis)}>
                  <ViewIcon fontSize="small" />
                </IconButton>
              )}
              {onDownload && (
                <IconButton size="small" onClick={() => onDownload(analysis)}>
                  <DownloadIcon fontSize="small" />
                </IconButton>
              )}
            </Box>
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card sx={{ position: 'relative', overflow: 'visible' }}>
      {loading && (
        <LinearProgress
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            zIndex: 1,
          }}
        />
      )}

      <CardContent>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', flex: 1 }}>
            <Box
              sx={{
                mr: 2,
                p: 1,
                borderRadius: 1,
                backgroundColor: `${getAnalysisColor(analysis.type)}.main`,
                color: 'white',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              {getAnalysisIcon(analysis.type)}
            </Box>
            <Box sx={{ flex: 1 }}>
              <Typography variant="h6" component="div" sx={{ fontWeight: 600 }}>
                {analysis.title}
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                <Chip
                  label={analysis.type.replace('_', ' ')}
                  size="small"
                  color={getAnalysisColor(analysis.type) as any}
                  variant="outlined"
                />
                <Typography variant="caption" color="text.secondary">
                  {date} at {time}
                </Typography>
              </Box>
            </Box>
          </Box>
        </Box>

        {/* Summary */}
        <Typography variant="body2" color="text.secondary" paragraph>
          {analysis.summary}
        </Typography>

        {/* Metrics */}
        {variant === 'detailed' && analysis.data && (
          <Box sx={{ mb: 2 }}>
            <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 2 }}>
              {analysis.charts && analysis.charts.length > 0 && (
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h6" color="primary.main">
                    {analysis.charts.length}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Charts
                  </Typography>
                </Box>
              )}
              {analysis.recommendations && analysis.recommendations.length > 0 && (
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h6" color="secondary.main">
                    {analysis.recommendations.length}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Insights
                  </Typography>
                </Box>
              )}
              {analysis.data.accuracy && (
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h6" color="success.main">
                    {(analysis.data.accuracy * 100).toFixed(0)}%
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Accuracy
                  </Typography>
                </Box>
              )}
            </Box>
          </Box>
        )}

        {/* Expandable Details */}
        {showDetails && (analysis.recommendations || analysis.charts) && (
          <Collapse in={expanded} timeout="auto" unmountOnExit>
            <Box sx={{ mt: 2, pt: 2, borderTop: 1, borderColor: 'divider' }}>
              {/* Charts */}
              {analysis.charts && analysis.charts.length > 0 && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Available Charts ({analysis.charts.length})
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {analysis.charts.map((chart) => (
                      <Chip
                        key={chart.id}
                        label={chart.title}
                        size="small"
                        variant="outlined"
                        icon={<ChartIcon />}
                      />
                    ))}
                  </Box>
                </Box>
              )}

              {/* Recommendations */}
              {analysis.recommendations && analysis.recommendations.length > 0 && (
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Key Insights
                  </Typography>
                  {analysis.recommendations.slice(0, 3).map((recommendation, index) => (
                    <Alert key={index} severity="info" sx={{ mb: 1, fontSize: '0.875rem' }}>
                      {recommendation}
                    </Alert>
                  ))}
                  {analysis.recommendations.length > 3 && (
                    <Typography variant="caption" color="text.secondary">
                      +{analysis.recommendations.length - 3} more insights available
                    </Typography>
                  )}
                </Box>
              )}
            </Box>
          </Collapse>
        )}
      </CardContent>

      <CardActions sx={{ justifyContent: 'space-between', px: 2, pb: 2 }}>
        <Box>
          {showDetails && (analysis.recommendations || analysis.charts) && (
            <Button
              size="small"
              onClick={() => setExpanded(!expanded)}
              endIcon={expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            >
              {expanded ? 'Less' : 'More'} Details
            </Button>
          )}
        </Box>

        <Box sx={{ display: 'flex', gap: 1 }}>
          {onShare && (
            <IconButton
              size="small"
              onClick={() => onShare(analysis)}
              color="default"
            >
              <ShareIcon />
            </IconButton>
          )}
          {onDownload && (
            <Button
              size="small"
              startIcon={<DownloadIcon />}
              onClick={() => onDownload(analysis)}
              variant="outlined"
            >
              Export
            </Button>
          )}
          {onView && (
            <Button
              size="small"
              startIcon={<ViewIcon />}
              onClick={() => onView(analysis)}
              variant="contained"
            >
              View
            </Button>
          )}
        </Box>
      </CardActions>
    </Card>
  );
};

export default AnalysisCard;