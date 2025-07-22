import React, { createContext, useContext, useState, ReactNode } from 'react';
import { Snackbar, Alert, AlertColor } from '@mui/material';
import { NotificationConfig } from '../types';

interface NotificationContextType {
  showNotification: (config: Omit<NotificationConfig, 'id'>) => void;
  hideNotification: (id?: string) => void;
  clearAllNotifications: () => void;
}

const NotificationContext = createContext<NotificationContextType | undefined>(undefined);

export const useNotification = () => {
  const context = useContext(NotificationContext);
  if (context === undefined) {
    throw new Error('useNotification must be used within a NotificationProvider');
  }
  return context;
};

interface NotificationProviderProps {
  children: ReactNode;
}

export const NotificationProvider: React.FC<NotificationProviderProps> = ({ children }) => {
  const [notifications, setNotifications] = useState<NotificationConfig[]>([]);

  const showNotification = (config: Omit<NotificationConfig, 'id'>) => {
    const id = Math.random().toString(36).substr(2, 9);
    const notification: NotificationConfig = {
      ...config,
      id,
      duration: config.duration || 5000,
    };

    setNotifications((prev) => [...prev, notification]);

    // Auto-hide after duration
    if (notification.duration && notification.duration > 0) {
      setTimeout(() => {
        hideNotification(id);
      }, notification.duration);
    }
  };

  const hideNotification = (id?: string) => {
    if (id) {
      setNotifications((prev) => prev.filter((notification) => notification.id !== id));
    } else {
      // Hide the oldest notification if no ID provided
      setNotifications((prev) => prev.slice(1));
    }
  };

  const clearAllNotifications = () => {
    setNotifications([]);
  };

  const value = {
    showNotification,
    hideNotification,
    clearAllNotifications,
  };

  return (
    <NotificationContext.Provider value={value}>
      {children}
      {notifications.map((notification) => (
        <Snackbar
          key={notification.id}
          open={true}
          autoHideDuration={notification.duration}
          onClose={() => hideNotification(notification.id)}
          anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
          sx={{
            mt: notifications.indexOf(notification) * 7, // Stack notifications
          }}
        >
          <Alert
            onClose={() => hideNotification(notification.id)}
            severity={notification.type as AlertColor}
            variant="filled"
            sx={{ width: '100%' }}
          >
            <strong>{notification.title}</strong>
            {notification.message && (
              <div style={{ marginTop: 4, fontSize: '0.875rem' }}>
                {notification.message}
              </div>
            )}
          </Alert>
        </Snackbar>
      ))}
    </NotificationContext.Provider>
  );
};