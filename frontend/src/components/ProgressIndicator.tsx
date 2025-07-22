import React from 'react';
import {
  Box,
  LinearProgress,
  CircularProgress,
  Typography,
  Card,
  CardContent,
  Chip,
  Alert,
  Stepper,
  Step,
  StepLabel,
  Button,
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
  Pause as PauseIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon,
} from '@mui/icons-material';

interface ProgressStep {
  id: string;
  label: string;
  description?: string;
  status: 'pending' | 'active' | 'completed' | 'error';
  duration?: number;
}

interface ProgressIndicatorProps {
  // Basic progress props
  value?: number;
  max?: number;
  label?: string;
  description?: string;
  
  // Visual variant
  variant?: 'linear' | 'circular' | 'stepper' | 'card';
  size?: 'small' | 'medium' | 'large';
  color?: 'primary' | 'secondary' | 'success' | 'warning' | 'error';
  
  // Status and state
  status?: 'running' | 'paused' | 'completed' | 'error' | 'cancelled';
  showPercentage?: boolean;
  showETA?: boolean;
  animated?: boolean;
  
  // Stepper variant props
  steps?: ProgressStep[];
  activeStep?: number;
  
  // Additional info
  currentTask?: string;
  eta?: string;
  speed?: string;
  throughput?: string;
  
  // Actions
  onPause?: () => void;
  onResume?: () => void;
  onCancel?: () => void;
  onRetry?: () => void;
  
  // Styling
  height?: number;
  width?: string | number;
  className?: string;
}

const ProgressIndicator: React.FC<ProgressIndicatorProps> = ({
  value = 0,
  max = 100,
  label,
  description,
  variant = 'linear',
  size = 'medium',
  color = 'primary',
  status = 'running',
  showPercentage = true,
  showETA = false,
  animated = true,
  steps,
  activeStep = 0,
  currentTask,
  eta,
  speed,
  throughput,
  onPause,
  onResume,
  onCancel,
  onRetry,
  height,
  width,
  className,
}) => {
  const percentage = Math.round((value / max) * 100);
  const isIndeterminate = value === undefined || value < 0;

  const getStatusIcon = () => {
    switch (status) {
      case 'completed':
        return <CheckCircleIcon color="success" />;
      case 'error':
        return <CancelIcon color="error" />;
      case 'paused':
        return <PauseIcon color="warning" />;
      case 'cancelled':
        return <CancelIcon color="disabled" />;
      default:
        return null;
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'error':
        return 'error';
      case 'paused':
        return 'warning';
      case 'cancelled':
        return 'default';
      default:
        return color;
    }
  };

  const getSizeProps = () => {
    switch (size) {
      case 'small':
        return { height: height || 4, fontSize: '0.75rem' };
      case 'large':
        return { height: height || 12, fontSize: '1.1rem' };
      default:
        return { height: height || 8, fontSize: '0.875rem' };
    }
  };

  if (variant === 'circular') {
    const circularSize = size === 'small' ? 24 : size === 'large' ? 64 : 40;
    
    return (
      <Box
        className={className}
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 1,
          width: width || 'auto',
        }}
      >
        <Box sx={{ position: 'relative', display: 'inline-flex' }}>
          <CircularProgress
            variant={isIndeterminate ? 'indeterminate' : 'determinate'}
            value={percentage}
            size={circularSize}
            color={getStatusColor() as any}
            sx={{
              ...(animated && status === 'running' && {
                animation: 'pulse 2s ease-in-out infinite',
              }),
            }}
          />
          {showPercentage && !isIndeterminate && (
            <Box
              sx={{
                top: 0,
                left: 0,
                bottom: 0,
                right: 0,
                position: 'absolute',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Typography
                variant="caption"
                component="div"
                color="text.secondary"
                sx={{ fontSize: getSizeProps().fontSize }}
              >
                {percentage}%
              </Typography>
            </Box>
          )}
          {getStatusIcon() && (
            <Box
              sx={{
                position: 'absolute',
                top: -4,
                right: -4,
              }}
            >
              {getStatusIcon()}
            </Box>
          )}
        </Box>
        
        {label && (
          <Typography variant="body2" textAlign="center">
            {label}
          </Typography>
        )}
        
        {currentTask && (
          <Typography variant="caption" color="text.secondary" textAlign="center">
            {currentTask}
          </Typography>
        )}
      </Box>
    );
  }

  if (variant === 'stepper' && steps) {
    return (
      <Card className={className} sx={{ width: width || '100%' }}>
        <CardContent>
          {label && (
            <Typography variant="h6" gutterBottom>
              {label}
            </Typography>
          )}
          
          <Stepper activeStep={activeStep} orientation="vertical">
            {steps.map((step) => (
              <Step key={step.id}>
                <StepLabel
                  error={step.status === 'error'}
                  icon={
                    step.status === 'completed' ? (
                      <CheckCircleIcon color="success" />
                    ) : step.status === 'error' ? (
                      <CancelIcon color="error" />
                    ) : undefined
                  }
                >
                  <Typography variant="body1">{step.label}</Typography>
                  {step.description && (
                    <Typography variant="caption" color="text.secondary">
                      {step.description}
                    </Typography>
                  )}
                  {step.duration && step.status === 'completed' && (
                    <Chip
                      label={`${step.duration}ms`}
                      size="small"
                      variant="outlined"
                      sx={{ ml: 1 }}
                    />
                  )}
                </StepLabel>
              </Step>
            ))}
          </Stepper>
          
          {(onPause || onResume || onCancel) && (
            <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
              {status === 'running' && onPause && (
                <Button size="small" startIcon={<PauseIcon />} onClick={onPause}>
                  Pause
                </Button>
              )}
              {status === 'paused' && onResume && (
                <Button size="small" startIcon={<PlayIcon />} onClick={onResume}>
                  Resume
                </Button>
              )}
              {onCancel && status !== 'completed' && (
                <Button size="small" startIcon={<StopIcon />} onClick={onCancel} color="error">
                  Cancel
                </Button>
              )}
              {status === 'error' && onRetry && (
                <Button size="small" onClick={onRetry} color="primary">
                  Retry
                </Button>
              )}
            </Box>
          )}
        </CardContent>
      </Card>
    );
  }

  if (variant === 'card') {
    return (
      <Card className={className} sx={{ width: width || '100%' }}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              {label && (
                <Typography variant="h6" component="div">
                  {label}
                </Typography>
              )}
              {getStatusIcon()}
            </Box>
            {showPercentage && !isIndeterminate && (
              <Typography variant="h6" color={`${getStatusColor()}.main`}>
                {percentage}%
              </Typography>
            )}
          </Box>
          
          <LinearProgress
            variant={isIndeterminate ? 'indeterminate' : 'determinate'}
            value={percentage}
            color={getStatusColor() as any}
            sx={{
              height: getSizeProps().height,
              borderRadius: 1,
              mb: 2,
              ...(animated && status === 'running' && {
                '& .MuiLinearProgress-bar': {
                  animation: 'progress-pulse 2s ease-in-out infinite',
                },
              }),
            }}
          />
          
          {currentTask && (
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Current: {currentTask}
            </Typography>
          )}
          
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 1 }}>
            <Box sx={{ display: 'flex', gap: 2 }}>
              {eta && showETA && (
                <Typography variant="caption" color="text.secondary">
                  ETA: {eta}
                </Typography>
              )}
              {speed && (
                <Typography variant="caption" color="text.secondary">
                  Speed: {speed}
                </Typography>
              )}
              {throughput && (
                <Typography variant="caption" color="text.secondary">
                  Throughput: {throughput}
                </Typography>
              )}
            </Box>
            
            {(onPause || onResume || onCancel) && (
              <Box sx={{ display: 'flex', gap: 1 }}>
                {status === 'running' && onPause && (
                  <Button size="small" startIcon={<PauseIcon />} onClick={onPause}>
                    Pause
                  </Button>
                )}
                {status === 'paused' && onResume && (
                  <Button size="small" startIcon={<PlayIcon />} onClick={onResume}>
                    Resume
                  </Button>
                )}
                {onCancel && status !== 'completed' && (
                  <Button size="small" startIcon={<StopIcon />} onClick={onCancel} color="error">
                    Cancel
                  </Button>
                )}
              </Box>
            )}
          </Box>
          
          {status === 'error' && description && (
            <Alert severity="error" sx={{ mt: 2 }} action={
              onRetry && (
                <Button color="inherit" size="small" onClick={onRetry}>
                  Retry
                </Button>
              )
            }>
              {description}
            </Alert>
          )}
        </CardContent>
      </Card>
    );
  }

  // Default linear variant
  return (
    <Box className={className} sx={{ width: width || '100%' }}>
      {(label || (showPercentage && !isIndeterminate)) && (
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {label && (
              <Typography variant="body2" color="text.secondary">
                {label}
              </Typography>
            )}
            {getStatusIcon()}
          </Box>
          {showPercentage && !isIndeterminate && (
            <Typography variant="body2" color="text.secondary">
              {percentage}%
            </Typography>
          )}
        </Box>
      )}
      
      <LinearProgress
        variant={isIndeterminate ? 'indeterminate' : 'determinate'}
        value={percentage}
        color={getStatusColor() as any}
        sx={{
          height: getSizeProps().height,
          borderRadius: 1,
          ...(animated && status === 'running' && {
            '& .MuiLinearProgress-bar': {
              animation: 'progress-shimmer 1.5s ease-in-out infinite',
            },
          }),
        }}
      />
      
      {(currentTask || description) && (
        <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
          {currentTask || description}
        </Typography>
      )}
      
      {(eta || speed) && (
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
          {eta && showETA && (
            <Typography variant="caption" color="text.secondary">
              ETA: {eta}
            </Typography>
          )}
          {speed && (
            <Typography variant="caption" color="text.secondary">
              {speed}
            </Typography>
          )}
        </Box>
      )}
    </Box>
  );
};

export default ProgressIndicator;