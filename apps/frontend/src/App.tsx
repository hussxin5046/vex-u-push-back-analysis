import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';

// Contexts
import { CustomThemeProvider } from './contexts/ThemeContext';
import { NotificationProvider } from './contexts/NotificationContext';

// Push Back Specific Components
import Layout from './components/Layout';
import PushBackDashboard from './pages/PushBackDashboard';
import StrategyBuilder from './pages/StrategyBuilder';
import QuickAnalysis from './pages/QuickAnalysis';
import DecisionTools from './pages/DecisionTools';

// Create a client optimized for Push Back analysis
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 2 * 60 * 1000, // 2 minutes for strategy data
      retry: 1,
      refetchOnWindowFocus: false,
    },
    mutations: {
      retry: 1,
    },
  },
});

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
                {/* Push Back focused routes */}
                <Route path="/" element={<PushBackDashboard />} />
                <Route path="/strategy-builder" element={<StrategyBuilder />} />
                <Route path="/analysis" element={<QuickAnalysis />} />
                <Route path="/decision-tools" element={<DecisionTools />} />
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