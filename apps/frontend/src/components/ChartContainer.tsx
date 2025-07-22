import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  IconButton,
  Menu,
  MenuItem,
  Skeleton,
  Alert,
  Typography,
  Tooltip,
  Button,
} from '@mui/material';
import {
  MoreVert as MoreVertIcon,
  Download as DownloadIcon,
  Fullscreen as FullscreenIcon,
  Share as ShareIcon,
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
} from '@mui/icons-material';
import { ChartData } from '../types';

interface ChartContainerProps {
  chart: ChartData;
  loading?: boolean;
  error?: string;
  height?: number | string;
  onExport?: (chart: ChartData, format: 'png' | 'svg' | 'pdf') => void;
  onShare?: (chart: ChartData) => void;
  onRefresh?: (chart: ChartData) => void;
  onFullscreen?: (chart: ChartData) => void;
  onSettings?: (chart: ChartData) => void;
  children?: React.ReactNode;
  showToolbar?: boolean;
  interactive?: boolean;
}

const ChartContainer: React.FC<ChartContainerProps> = ({
  chart,
  loading = false,
  error,
  height = 400,
  onExport,
  onShare,
  onRefresh,
  onFullscreen,
  onSettings,
  children,
  showToolbar = true,
  interactive = true,
}) => {
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [zoom, setZoom] = useState(1);

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev + 0.1, 2));
  };

  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev - 0.1, 0.5));
  };

  const handleZoomReset = () => {
    setZoom(1);
  };

  const exportFormats = [
    { format: 'png' as const, label: 'PNG Image' },
    { format: 'svg' as const, label: 'SVG Vector' },
    { format: 'pdf' as const, label: 'PDF Document' },
  ];

  return (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header with toolbar */}
      <CardHeader
        title={
          <Typography variant="h6" component="div" sx={{ fontWeight: 600 }}>
            {chart.title}
          </Typography>
        }
        subheader={
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography variant="caption" color="text.secondary">
              {chart.type.toUpperCase()} Chart
            </Typography>
            {chart.data && Array.isArray(chart.data) && (
              <Typography variant="caption" color="text.secondary">
                • {chart.data.length} data points
              </Typography>
            )}
          </Box>
        }
        action={
          showToolbar && (
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              {interactive && (
                <>
                  <Tooltip title="Zoom In">
                    <IconButton size="small" onClick={handleZoomIn} disabled={zoom >= 2}>
                      <ZoomInIcon />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Zoom Out">
                    <IconButton size="small" onClick={handleZoomOut} disabled={zoom <= 0.5}>
                      <ZoomOutIcon />
                    </IconButton>
                  </Tooltip>
                  {zoom !== 1 && (
                    <Button size="small" onClick={handleZoomReset} sx={{ minWidth: 'auto', px: 1 }}>
                      {Math.round(zoom * 100)}%
                    </Button>
                  )}
                </>
              )}
              
              {onRefresh && (
                <Tooltip title="Refresh Data">
                  <IconButton size="small" onClick={() => onRefresh(chart)} disabled={loading}>
                    <RefreshIcon />
                  </IconButton>
                </Tooltip>
              )}
              
              {onFullscreen && (
                <Tooltip title="Fullscreen">
                  <IconButton size="small" onClick={() => onFullscreen(chart)}>
                    <FullscreenIcon />
                  </IconButton>
                </Tooltip>
              )}

              <Tooltip title="More options">
                <IconButton size="small" onClick={handleMenuOpen}>
                  <MoreVertIcon />
                </IconButton>
              </Tooltip>

              <Menu
                anchorEl={anchorEl}
                open={Boolean(anchorEl)}
                onClose={handleMenuClose}
                transformOrigin={{ horizontal: 'right', vertical: 'top' }}
                anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
              >
                {onExport && exportFormats.map((format) => (
                  <MenuItem
                    key={format.format}
                    onClick={() => {
                      onExport(chart, format.format);
                      handleMenuClose();
                    }}
                  >
                    <DownloadIcon sx={{ mr: 1 }} />
                    Export as {format.label}
                  </MenuItem>
                ))}
                
                {onShare && (
                  <MenuItem
                    onClick={() => {
                      onShare(chart);
                      handleMenuClose();
                    }}
                  >
                    <ShareIcon sx={{ mr: 1 }} />
                    Share Chart
                  </MenuItem>
                )}
                
                {onSettings && (
                  <MenuItem
                    onClick={() => {
                      onSettings(chart);
                      handleMenuClose();
                    }}
                  >
                    <SettingsIcon sx={{ mr: 1 }} />
                    Chart Settings
                  </MenuItem>
                )}
              </Menu>
            </Box>
          )
        }
        sx={{ pb: 1 }}
      />

      {/* Chart content */}
      <CardContent sx={{ flex: 1, pt: 0, position: 'relative' }}>
        {loading ? (
          <Box sx={{ height }}>
            <Skeleton variant="rectangular" width="100%" height="100%" />
            <Box sx={{ mt: 1, display: 'flex', justifyContent: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                Loading chart data...
              </Typography>
            </Box>
          </Box>
        ) : error ? (
          <Alert 
            severity="error" 
            sx={{ height: 'auto', minHeight: 100, display: 'flex', alignItems: 'center' }}
            action={
              onRefresh && (
                <Button color="inherit" size="small" onClick={() => onRefresh(chart)}>
                  Retry
                </Button>
              )
            }
          >
            <Box>
              <Typography variant="subtitle2">Failed to load chart</Typography>
              <Typography variant="body2">{error}</Typography>
            </Box>
          </Alert>
        ) : (
          <Box
            sx={{
              height,
              position: 'relative',
              overflow: 'hidden',
              transform: `scale(${zoom})`,
              transformOrigin: 'top left',
              transition: 'transform 0.2s ease-in-out',
            }}
          >
            {children ? (
              children
            ) : (
              <Box
                sx={{
                  width: '100%',
                  height: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  border: '2px dashed',
                  borderColor: 'divider',
                  borderRadius: 1,
                  backgroundColor: 'action.hover',
                }}
              >
                <Typography variant="body2" color="text.secondary" textAlign="center">
                  Chart visualization goes here
                  <br />
                  <Typography variant="caption">
                    Type: {chart.type} | Data points: {Array.isArray(chart.data) ? chart.data.length : 0}
                  </Typography>
                </Typography>
              </Box>
            )}
          </Box>
        )}

        {/* Chart metadata footer */}
        {!loading && !error && chart.options?.description && (
          <Box sx={{ mt: 2, pt: 1, borderTop: 1, borderColor: 'divider' }}>
            <Typography variant="caption" color="text.secondary">
              {chart.options.description}
            </Typography>
          </Box>
        )}
        
        {/* X-axis and Y-axis labels */}
        {!loading && !error && (chart.xAxis || chart.yAxis) && (
          <Box sx={{ mt: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            {chart.yAxis && (
              <Typography 
                variant="caption" 
                color="text.secondary"
                sx={{ 
                  transform: 'rotate(-90deg)',
                  position: 'absolute',
                  left: -20,
                  top: '50%',
                  transformOrigin: 'center',
                }}
              >
                {chart.yAxis}
              </Typography>
            )}
            {chart.xAxis && (
              <Typography 
                variant="caption" 
                color="text.secondary"
                sx={{ 
                  position: 'absolute',
                  bottom: 10,
                  left: '50%',
                  transform: 'translateX(-50%)',
                }}
              >
                {chart.xAxis}
              </Typography>
            )}
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default ChartContainer;