import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';

// Contexts
import { CustomThemeProvider } from './contexts/ThemeContext';
import { NotificationProvider } from './contexts/NotificationContext';

// Components and Pages
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Analysis from './pages/Analysis';

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      retry: 1,
      refetchOnWindowFocus: false,
    },
    mutations: {
      retry: 1,
    },
  },
});

// Placeholder components for other routes
const SimulationPage = () => (
  <div>
    <h1>Simulation</h1>
    <p>Simulation tools and scenario generation coming soon!</p>
  </div>
);

const MLPage = () => (
  <div>
    <h1>ML Models</h1>
    <p>Machine learning model management coming soon!</p>
  </div>
);

const VisualizationPage = () => (
  <div>
    <h1>Visualization</h1>
    <p>Interactive charts and dashboards coming soon!</p>
  </div>
);

const ReportsPage = () => (
  <div>
    <h1>Reports</h1>
    <p>Report generation and management coming soon!</p>
  </div>
);

const DataPage = () => (
  <div>
    <h1>Data Management</h1>
    <p>Data upload and management tools coming soon!</p>
  </div>
);

const SettingsPage = () => (
  <div>
    <h1>Settings</h1>
    <p>Application settings and preferences coming soon!</p>
  </div>
);

const NotFoundPage = () => (
  <div style={{ textAlign: 'center', marginTop: '2rem' }}>
    <h1>404 - Page Not Found</h1>
    <p>The page you're looking for doesn't exist.</p>
  </div>
);

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <CustomThemeProvider>
        <NotificationProvider>
          <Router>
            <Layout>
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/analysis" element={<Analysis />} />
                <Route path="/analysis/scoring" element={<Analysis />} />
                <Route path="/analysis/statistical" element={<Analysis />} />
                <Route path="/analysis/strategy" element={<Analysis />} />
                <Route path="/simulation" element={<SimulationPage />} />
                <Route path="/ml" element={<MLPage />} />
                <Route path="/ml/train" element={<MLPage />} />
                <Route path="/ml/predict" element={<MLPage />} />
                <Route path="/ml/optimize" element={<MLPage />} />
                <Route path="/visualization" element={<VisualizationPage />} />
                <Route path="/reports" element={<ReportsPage />} />
                <Route path="/data" element={<DataPage />} />
                <Route path="/settings" element={<SettingsPage />} />
                <Route path="*" element={<NotFoundPage />} />
              </Routes>
            </Layout>
          </Router>
          <ReactQueryDevtools initialIsOpen={false} />
        </NotificationProvider>
      </CustomThemeProvider>
    </QueryClientProvider>
  );
}

export default App;