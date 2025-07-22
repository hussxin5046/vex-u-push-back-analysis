import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  IconButton,
  Box,
  Divider,
  Chip,
  useMediaQuery,
  useTheme as useMuiTheme,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  Build as BuildIcon,
  Analytics as AnalyticsIcon,
  Tune as TuneIcon,
  Brightness4 as DarkModeIcon,
  Brightness7 as LightModeIcon,
  SportsEsports as GameIcon,
} from '@mui/icons-material';
import { useTheme } from '../contexts/ThemeContext';
import { useHealthCheck } from '../hooks/useApi';
import { NavigationItem } from '../types';

const drawerWidth = 280;

// Push Back focused navigation items
const navigationItems: NavigationItem[] = [
  {
    id: 'dashboard',
    label: 'Push Back Dashboard',
    path: '/',
    icon: 'dashboard',
  },
  {
    id: 'strategy-builder',
    label: 'Strategy Builder',
    path: '/strategy-builder',
    icon: 'build',
  },
  {
    id: 'analysis',
    label: 'Quick Analysis',
    path: '/analysis',
    icon: 'analytics',
  },
  {
    id: 'decision-tools',
    label: 'Decision Tools',
    path: '/decision-tools',
    icon: 'tune',
  },
];

const getIcon = (iconName: string) => {
  switch (iconName) {
    case 'dashboard':
      return <DashboardIcon />;
    case 'build':
      return <BuildIcon />;
    case 'analytics':
      return <AnalyticsIcon />;
    case 'tune':
      return <TuneIcon />;
    case 'game':
      return <GameIcon />;
    default:
      return <DashboardIcon />;
  }
};

interface NavigationProps {
  window?: () => Window;
}

const Navigation: React.FC<NavigationProps> = ({ window }) => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const { mode, toggleTheme } = useTheme();
  const muiTheme = useMuiTheme();
  const isMobile = useMediaQuery(muiTheme.breakpoints.down('md'));
  const { data: isBackendConnected = false } = useHealthCheck();

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleNavigation = (path: string) => {
    navigate(path);
    if (isMobile) {
      setMobileOpen(false);
    }
  };

  const renderNavigationItem = (item: NavigationItem, level = 0) => {
    const isActive = location.pathname === item.path;
    const hasChildren = item.children && item.children.length > 0;

    return (
      <React.Fragment key={item.id}>
        <ListItem disablePadding>
          <ListItemButton
            onClick={() => !hasChildren && handleNavigation(item.path)}
            selected={isActive}
            sx={{
              pl: 2 + level * 2,
              borderRadius: 1,
              mx: 1,
              mb: 0.5,
              '&.Mui-selected': {
                backgroundColor: muiTheme.palette.primary.main + '20',
                '&:hover': {
                  backgroundColor: muiTheme.palette.primary.main + '30',
                },
              },
            }}
          >
            <ListItemIcon sx={{ minWidth: 40 }}>
              {getIcon(item.icon || 'dashboard')}
            </ListItemIcon>
            <ListItemText 
              primary={item.label} 
              primaryTypographyProps={{
                fontSize: level > 0 ? '0.875rem' : '1rem',
                fontWeight: isActive ? 600 : 400,
              }}
            />
          </ListItemButton>
        </ListItem>
        {hasChildren && (
          <>
            {item.children!.map((child) => renderNavigationItem(child, level + 1))}
          </>
        )}
      </React.Fragment>
    );
  };

  const drawer = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ p: 2 }}>
        <Typography variant="h6" noWrap component="div" sx={{ fontWeight: 700 }}>
          VEX U Push Back
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
          Strategy Builder
        </Typography>
        <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
          <Chip
            label={isBackendConnected ? 'Connected' : 'Disconnected'}
            color={isBackendConnected ? 'success' : 'error'}
            size="small"
            variant="outlined"
          />
        </Box>
      </Box>

      <Divider />

      <Box sx={{ flex: 1, overflow: 'auto' }}>
        <List sx={{ py: 1 }}>
          {navigationItems.map((item) => renderNavigationItem(item))}
        </List>
      </Box>

      <Divider />
      
      {/* Quick Stats */}
      <Box sx={{ p: 2 }}>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Push Back Field
        </Typography>
        <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1, mb: 2 }}>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h6" sx={{ fontWeight: 600 }}>88</Typography>
            <Typography variant="caption" color="text.secondary">Blocks</Typography>
          </Box>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h6" sx={{ fontWeight: 600 }}>4</Typography>
            <Typography variant="caption" color="text.secondary">Goals</Typography>
          </Box>
        </Box>
        
        <ListItem disablePadding>
          <ListItemButton
            onClick={toggleTheme}
            sx={{
              borderRadius: 1,
              justifyContent: 'center',
            }}
          >
            <ListItemIcon sx={{ minWidth: 40, justifyContent: 'center' }}>
              {mode === 'dark' ? <LightModeIcon /> : <DarkModeIcon />}
            </ListItemIcon>
            <ListItemText 
              primary={mode === 'dark' ? 'Light Mode' : 'Dark Mode'}
              primaryTypographyProps={{ fontSize: '0.875rem' }}
            />
          </ListItemButton>
        </ListItem>
      </Box>
    </Box>
  );

  const container = window !== undefined ? () => window().document.body : undefined;

  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar
        position="fixed"
        sx={{
          width: { md: `calc(100% - ${drawerWidth}px)` },
          ml: { md: `${drawerWidth}px` },
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { md: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            VEX U Push Back Strategy Platform
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Chip
              label={isBackendConnected ? 'Backend Online' : 'Backend Offline'}
              color={isBackendConnected ? 'success' : 'error'}
              size="small"
              variant="filled"
              sx={{ display: { xs: 'none', sm: 'inline-flex' } }}
            />
            <IconButton color="inherit" onClick={toggleTheme}>
              {mode === 'dark' ? <LightModeIcon /> : <DarkModeIcon />}
            </IconButton>
          </Box>
        </Toolbar>
      </AppBar>

      <Box
        component="nav"
        sx={{ width: { md: drawerWidth }, flexShrink: { md: 0 } }}
      >
        <Drawer
          container={container}
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true,
          }}
          sx={{
            display: { xs: 'block', md: 'none' },
            '& .MuiDrawer-paper': { 
              boxSizing: 'border-box', 
              width: drawerWidth,
            },
          }}
        >
          {drawer}
        </Drawer>
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', md: 'block' },
            '& .MuiDrawer-paper': { 
              boxSizing: 'border-box', 
              width: drawerWidth,
            },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>
    </Box>
  );
};

export default Navigation;