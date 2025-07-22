import React, { useState, useMemo } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Chip,
  ToggleButton,
  ToggleButtonGroup,
  Button,
  Paper,
  Divider,
  Alert,
} from '@mui/material';
import {
  BarChart as BarChartIcon,
  ShowChart as LineChartIcon,
  PieChart as PieChartIcon,
  ScatterPlot as ScatterIcon,
  Download as DownloadIcon,
  Share as ShareIcon,
  FilterList as FilterIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { useNotification } from '../contexts/NotificationContext';
import { ChartData } from '../types';

interface ChartFilter {
  dataSource: string;
  dateRange: string;
  metric: string;
  groupBy: string;
  filterBy: string;
}

interface MockDataPoint {
  x: string | number;
  y: number;
  category?: string;
  label?: string;
}

const Visualizations: React.FC = () => {
  const [chartType, setChartType] = useState<'bar' | 'line' | 'pie' | 'scatter'>('bar');
  const [filters, setFilters] = useState<ChartFilter>({
    dataSource: 'recent_analyses',
    dateRange: 'last_30_days',
    metric: 'win_probability',
    groupBy: 'strategy_type',
    filterBy: 'all',
  });

  const { showNotification } = useNotification();

  const dataSources = [
    { value: 'recent_analyses', label: 'Recent Analyses' },
    { value: 'historical_matches', label: 'Historical Matches' },
    { value: 'strategy_simulations', label: 'Strategy Simulations' },
    { value: 'ml_predictions', label: 'ML Predictions' },
    { value: 'performance_metrics', label: 'Performance Metrics' },
  ];

  const dateRanges = [
    { value: 'last_7_days', label: 'Last 7 Days' },
    { value: 'last_30_days', label: 'Last 30 Days' },
    { value: 'last_90_days', label: 'Last 90 Days' },
    { value: 'this_season', label: 'This Season' },
    { value: 'all_time', label: 'All Time' },
  ];

  const metrics = [
    { value: 'win_probability', label: 'Win Probability' },
    { value: 'average_score', label: 'Average Score' },
    { value: 'score_consistency', label: 'Score Consistency' },
    { value: 'autonomous_points', label: 'Autonomous Points' },
    { value: 'driver_points', label: 'Driver Points' },
    { value: 'endgame_points', label: 'Endgame Points' },
    { value: 'match_duration', label: 'Match Duration' },
    { value: 'efficiency_rating', label: 'Efficiency Rating' },
  ];

  const groupByOptions = [
    { value: 'strategy_type', label: 'Strategy Type' },
    { value: 'skill_level', label: 'Skill Level' },
    { value: 'robot_role', label: 'Robot Role' },
    { value: 'alliance_color', label: 'Alliance Color' },
    { value: 'match_type', label: 'Match Type' },
    { value: 'competition', label: 'Competition' },
  ];

  const chartTypeIcons = {
    bar: <BarChartIcon />,
    line: <LineChartIcon />,
    pie: <PieChartIcon />,
    scatter: <ScatterIcon />,
  };

  // Generate mock data based on current filters
  const chartData: MockDataPoint[] = useMemo(() => {
    const categories = ['Offensive Rush', 'Defensive Control', 'Balanced', 'Autonomous Focus', 'Endgame Specialist'];
    const colors = ['#1976d2', '#388e3c', '#f57c00', '#d32f2f', '#7b1fa2'];
    
    return categories.map((category, index) => ({
      x: category,
      y: Math.random() * 100 + 20,
      category: category,
      label: `${category}: ${(Math.random() * 100 + 20).toFixed(1)}%`,
    }));
  }, [filters]);

  const handleFilterChange = (field: keyof ChartFilter, value: string) => {
    setFilters(prev => ({
      ...prev,
      [field]: value,
    }));
  };

  const exportChart = () => {
    showNotification({
      type: 'success',
      title: 'Chart Exported',
      message: 'Chart has been exported successfully.',
    });
  };

  const shareChart = () => {
    showNotification({
      type: 'info',
      title: 'Share Chart',
      message: 'Chart sharing link copied to clipboard.',
    });
  };

  const refreshData = () => {
    showNotification({
      type: 'success',
      title: 'Data Refreshed',
      message: 'Chart data has been refreshed with the latest information.',
    });
  };

  const renderChart = () => {
    const chartTitle = `${metrics.find(m => m.value === filters.metric)?.label} by ${groupByOptions.find(g => g.value === filters.groupBy)?.label}`;
    
    switch (chartType) {
      case 'bar':
        return (
          <Box sx={{ height: 400, display: 'flex', alignItems: 'end', justifyContent: 'space-around', p: 2 }}>
            {chartData.map((point, index) => (
              <Box key={index} sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flex: 1, mx: 1 }}>
                <Box
                  sx={{
                    width: '100%',
                    maxWidth: 60,
                    height: `${(point.y / 120) * 300}px`,
                    backgroundColor: `hsl(${index * 60}, 60%, 50%)`,
                    borderRadius: 1,
                    mb: 1,
                    position: 'relative',
                  }}
                >
                  <Typography
                    variant="caption"
                    sx={{
                      position: 'absolute',
                      top: -20,
                      left: '50%',
                      transform: 'translateX(-50%)',
                      fontWeight: 600,
                    }}
                  >
                    {point.y.toFixed(1)}
                  </Typography>
                </Box>
                <Typography variant="caption" sx={{ textAlign: 'center', fontSize: '0.7rem' }}>
                  {typeof point.x === 'string' ? point.x.split(' ').join('\n') : point.x}
                </Typography>
              </Box>
            ))}
          </Box>
        );

      case 'line':
        return (
          <Box sx={{ height: 400, p: 2, position: 'relative' }}>
            <svg width="100%" height="100%" viewBox="0 0 600 300">
              <defs>
                <linearGradient id="lineGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor="#1976d2" stopOpacity="0.3" />
                  <stop offset="100%" stopColor="#1976d2" stopOpacity="0.1" />
                </linearGradient>
              </defs>
              
              {/* Grid lines */}
              {[0, 1, 2, 3, 4].map(i => (
                <line
                  key={i}
                  x1="0"
                  y1={i * 60}
                  x2="600"
                  y2={i * 60}
                  stroke="#e0e0e0"
                  strokeWidth="1"
                />
              ))}
              
              {/* Data line */}
              <polyline
                fill="none"
                stroke="#1976d2"
                strokeWidth="3"
                points={chartData.map((point, index) => 
                  `${(index / (chartData.length - 1)) * 600},${300 - (point.y / 120) * 280}`
                ).join(' ')}
              />
              
              {/* Area fill */}
              <polygon
                fill="url(#lineGradient)"
                points={`0,300 ${chartData.map((point, index) => 
                  `${(index / (chartData.length - 1)) * 600},${300 - (point.y / 120) * 280}`
                ).join(' ')} 600,300`}
              />
              
              {/* Data points */}
              {chartData.map((point, index) => (
                <circle
                  key={index}
                  cx={(index / (chartData.length - 1)) * 600}
                  cy={300 - (point.y / 120) * 280}
                  r="4"
                  fill="#1976d2"
                />
              ))}
            </svg>
          </Box>
        );

      case 'pie':
        const total = chartData.reduce((sum, point) => sum + point.y, 0);
        let currentAngle = 0;
        
        return (
          <Box sx={{ height: 400, display: 'flex', alignItems: 'center', justifyContent: 'center', p: 2 }}>
            <Box sx={{ position: 'relative' }}>
              <svg width="300" height="300" viewBox="0 0 200 200">
                {chartData.map((point, index) => {
                  const percentage = point.y / total;
                  const angle = percentage * 360;
                  const startAngle = currentAngle;
                  const endAngle = currentAngle + angle;
                  
                  const x1 = 100 + 80 * Math.cos((startAngle - 90) * Math.PI / 180);
                  const y1 = 100 + 80 * Math.sin((startAngle - 90) * Math.PI / 180);
                  const x2 = 100 + 80 * Math.cos((endAngle - 90) * Math.PI / 180);
                  const y2 = 100 + 80 * Math.sin((endAngle - 90) * Math.PI / 180);
                  
                  const largeArcFlag = angle > 180 ? 1 : 0;
                  
                  const pathData = [
                    'M', 100, 100,
                    'L', x1, y1,
                    'A', 80, 80, 0, largeArcFlag, 1, x2, y2,
                    'Z'
                  ].join(' ');
                  
                  currentAngle += angle;
                  
                  return (
                    <path
                      key={index}
                      d={pathData}
                      fill={`hsl(${index * 60}, 60%, 50%)`}
                      stroke="white"
                      strokeWidth="2"
                    />
                  );
                })}
              </svg>
              
              <Box sx={{ position: 'absolute', right: -150, top: 50 }}>
                {chartData.map((point, index) => (
                  <Box key={index} sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <Box
                      sx={{
                        width: 12,
                        height: 12,
                        backgroundColor: `hsl(${index * 60}, 60%, 50%)`,
                        borderRadius: 1,
                        mr: 1,
                      }}
                    />
                    <Typography variant="caption">
                      {point.x} ({((point.y / total) * 100).toFixed(1)}%)
                    </Typography>
                  </Box>
                ))}
              </Box>
            </Box>
          </Box>
        );

      case 'scatter':
        return (
          <Box sx={{ height: 400, p: 2, position: 'relative' }}>
            <svg width="100%" height="100%" viewBox="0 0 600 300">
              {/* Grid */}
              {[0, 1, 2, 3, 4, 5].map(i => (
                <g key={i}>
                  <line x1={i * 100} y1="0" x2={i * 100} y2="300" stroke="#e0e0e0" strokeWidth="1" />
                  <line x1="0" y1={i * 50} x2="600" y2={i * 50} stroke="#e0e0e0" strokeWidth="1" />
                </g>
              ))}
              
              {/* Scatter points */}
              {chartData.map((point, index) => (
                <circle
                  key={index}
                  cx={Math.random() * 500 + 50}
                  cy={300 - (point.y / 120) * 280}
                  r="6"
                  fill={`hsl(${index * 60}, 60%, 50%)`}
                  opacity="0.7"
                />
              ))}
            </svg>
          </Box>
        );

      default:
        return null;
    }
  };

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 700 }}>
        Visualizations
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Create interactive charts and visualizations from your VEX U analysis data.
      </Typography>

      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '300px 1fr' }, gap: 3 }}>
        {/* Controls Panel */}
        <Box>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                <FilterIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                Chart Configuration
              </Typography>

              {/* Chart Type Selection */}
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Chart Type
                </Typography>
                <ToggleButtonGroup
                  value={chartType}
                  exclusive
                  onChange={(_, value) => value && setChartType(value)}
                  size="small"
                  fullWidth
                >
                  {Object.entries(chartTypeIcons).map(([type, icon]) => (
                    <ToggleButton key={type} value={type} sx={{ flex: 1 }}>
                      {icon}
                    </ToggleButton>
                  ))}
                </ToggleButtonGroup>
              </Box>

              <Divider sx={{ mb: 2 }} />

              {/* Data Source */}
              <FormControl fullWidth margin="normal" size="small">
                <InputLabel>Data Source</InputLabel>
                <Select
                  value={filters.dataSource}
                  onChange={(e) => handleFilterChange('dataSource', e.target.value)}
                >
                  {dataSources.map((source) => (
                    <MenuItem key={source.value} value={source.value}>
                      {source.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {/* Date Range */}
              <FormControl fullWidth margin="normal" size="small">
                <InputLabel>Date Range</InputLabel>
                <Select
                  value={filters.dateRange}
                  onChange={(e) => handleFilterChange('dateRange', e.target.value)}
                >
                  {dateRanges.map((range) => (
                    <MenuItem key={range.value} value={range.value}>
                      {range.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {/* Metric */}
              <FormControl fullWidth margin="normal" size="small">
                <InputLabel>Metric</InputLabel>
                <Select
                  value={filters.metric}
                  onChange={(e) => handleFilterChange('metric', e.target.value)}
                >
                  {metrics.map((metric) => (
                    <MenuItem key={metric.value} value={metric.value}>
                      {metric.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {/* Group By */}
              <FormControl fullWidth margin="normal" size="small">
                <InputLabel>Group By</InputLabel>
                <Select
                  value={filters.groupBy}
                  onChange={(e) => handleFilterChange('groupBy', e.target.value)}
                >
                  {groupByOptions.map((option) => (
                    <MenuItem key={option.value} value={option.value}>
                      {option.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {/* Custom Filter */}
              <TextField
                label="Custom Filter"
                placeholder="e.g., score > 100"
                value={filters.filterBy === 'all' ? '' : filters.filterBy}
                onChange={(e) => handleFilterChange('filterBy', e.target.value || 'all')}
                fullWidth
                margin="normal"
                size="small"
              />

              {/* Action Buttons */}
              <Box sx={{ mt: 3, display: 'flex', flexDirection: 'column', gap: 1 }}>
                <Button
                  startIcon={<RefreshIcon />}
                  onClick={refreshData}
                  variant="outlined"
                  size="small"
                  fullWidth
                >
                  Refresh Data
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Box>

        {/* Chart Display */}
        <Box>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  {metrics.find(m => m.value === filters.metric)?.label} by {groupByOptions.find(g => g.value === filters.groupBy)?.label}
                </Typography>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Button
                    startIcon={<DownloadIcon />}
                    onClick={exportChart}
                    variant="outlined"
                    size="small"
                  >
                    Export
                  </Button>
                  <Button
                    startIcon={<ShareIcon />}
                    onClick={shareChart}
                    variant="outlined"
                    size="small"
                  >
                    Share
                  </Button>
                </Box>
              </Box>

              {/* Chart Info */}
              <Box sx={{ mb: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                <Chip
                  label={dataSources.find(s => s.value === filters.dataSource)?.label}
                  size="small"
                  color="primary"
                  variant="outlined"
                />
                <Chip
                  label={dateRanges.find(r => r.value === filters.dateRange)?.label}
                  size="small"
                  color="secondary"
                  variant="outlined"
                />
                <Chip
                  label={`${chartData.length} data points`}
                  size="small"
                  variant="outlined"
                />
              </Box>

              {/* Chart Container */}
              <Paper variant="outlined" sx={{ mb: 2 }}>
                {renderChart()}
              </Paper>

              {/* Chart Insights */}
              <Alert severity="info">
                <Typography variant="body2">
                  <strong>Insights:</strong> Based on the current data, the highest performing category shows a {Math.max(...chartData.map(d => d.y)).toFixed(1)}% metric value, 
                  while the average across all categories is {(chartData.reduce((sum, d) => sum + d.y, 0) / chartData.length).toFixed(1)}%.
                </Typography>
              </Alert>
            </CardContent>
          </Card>
        </Box>
      </Box>
    </Box>
  );
};

export default Visualizations;