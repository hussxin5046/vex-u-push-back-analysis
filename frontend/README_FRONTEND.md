# VEX U Scoring Analysis Platform - Frontend

A modern React TypeScript frontend for the VEX U Scoring Analysis Platform, providing an intuitive interface for strategic analysis, machine learning predictions, and comprehensive reporting.

## 🚀 Features

- **Dashboard**: Real-time system status and quick access to key features
- **Analysis Center**: Run comprehensive scoring, strategy, and statistical analyses
- **ML Integration**: Machine learning model management and predictions
- **Interactive Visualizations**: Charts and graphs powered by Recharts
- **Report Generation**: Export detailed reports in multiple formats
- **Dark/Light Theme**: Toggle between themes with system preference detection
- **Responsive Design**: Optimized for desktop and mobile devices
- **Real-time Updates**: Live data refresh with React Query

## 🛠 Technology Stack

- **React 18** with TypeScript
- **Material-UI (MUI)** for UI components and theming
- **React Router** for navigation
- **React Query** for data fetching and caching
- **Recharts** for data visualization
- **Axios** for HTTP requests
- **Context API** for state management

## 📦 Installation

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create environment configuration:
   ```bash
   cp .env.example .env
   ```

4. Update the `.env` file with your backend URL:
   ```env
   REACT_APP_API_BASE_URL=http://localhost:8000
   ```

## 🔧 Development

Start the development server:
```bash
npm start
```

The application will open at `http://localhost:3000` and will automatically reload when you make changes.

## 🏗 Build

Create a production build:
```bash
npm run build
```

The build artifacts will be stored in the `build/` directory.

## 🧪 Testing

Run the test suite:
```bash
npm test
```

## 📁 Project Structure

```
frontend/
├── public/                 # Static files
├── src/
│   ├── components/        # Reusable UI components
│   │   ├── Layout.tsx     # Main layout wrapper
│   │   └── Navigation.tsx # Navigation sidebar
│   ├── contexts/          # React contexts
│   │   ├── ThemeContext.tsx      # Theme management
│   │   └── NotificationContext.tsx # Notifications
│   ├── hooks/             # Custom React hooks
│   │   └── useApi.ts      # API data fetching hooks
│   ├── pages/             # Page components
│   │   ├── Dashboard.tsx  # Main dashboard
│   │   └── Analysis.tsx   # Analysis center
│   ├── services/          # External services
│   │   └── api.ts         # API client
│   ├── types/             # TypeScript type definitions
│   │   └── index.ts       # Shared types
│   ├── utils/             # Utility functions
│   └── App.tsx            # Main app component
├── package.json
└── README.md
```

## 🎨 UI Components

### Theme System
- Supports light and dark themes
- Automatic system preference detection
- Persistent theme selection
- Custom color palette for VEX U branding

### Navigation
- Responsive sidebar navigation
- Hierarchical menu structure
- Active route highlighting
- Mobile-friendly drawer

### Notifications
- Toast notifications for user feedback
- Multiple severity levels (success, error, warning, info)
- Auto-dismiss functionality
- Action buttons support

## 🔌 API Integration

The frontend communicates with the Python backend through a RESTful API:

- **System Status**: Monitor backend connectivity and ML model availability
- **Analysis Operations**: Run scoring, strategy, and statistical analyses
- **ML Models**: Train models, make predictions, and optimize strategies
- **Data Management**: Upload competition data and manage datasets
- **Reporting**: Generate and export comprehensive reports

### API Service Structure

```typescript
// Example API call
const { data, isLoading, error } = useSystemStatus();

// Mutation for data modifications
const runAnalysis = useRunAnalysis();
await runAnalysis.mutateAsync({ type: 'scoring', params: {...} });
```

## 📊 Data Visualization

Charts and visualizations are built using Recharts:

- Line charts for performance trends
- Bar charts for comparative analysis
- Pie charts for distribution data
- Scatter plots for correlation analysis
- Interactive tooltips and legends

## 🔧 Configuration

Environment variables:

- `REACT_APP_API_BASE_URL`: Backend API base URL
- `REACT_APP_API_TIMEOUT`: Request timeout in milliseconds
- `REACT_APP_ENABLE_DEVTOOLS`: Enable React Query DevTools
- `REACT_APP_DEFAULT_THEME`: Default theme ('light' or 'dark')

## 🚀 Deployment

### Development Deployment
```bash
npm start
```

### Production Deployment
```bash
npm run build
npm install -g serve
serve -s build
```

### Docker Deployment
```dockerfile
FROM node:18-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## 🤝 Integration with Python Backend

The frontend is designed to work seamlessly with the Python backend:

1. **Start the Python backend** (from the project root):
   ```bash
   cd vex_u_scoring_analysis
   python main.py serve --port 8000
   ```

2. **Start the frontend** (from the frontend directory):
   ```bash
   npm start
   ```

3. **Access the application** at `http://localhost:3000`

## 🔍 Troubleshooting

### Common Issues

1. **Backend Connection Failed**
   - Ensure the Python backend is running
   - Check the API URL in `.env`
   - Verify firewall settings

2. **Build Errors**
   - Clear node_modules: `rm -rf node_modules && npm install`
   - Clear npm cache: `npm cache clean --force`

3. **Type Errors**
   - Ensure TypeScript definitions are up to date
   - Check import paths and type exports

## 📝 Contributing

1. Follow the existing code style and conventions
2. Use TypeScript for all new components
3. Add proper error handling and loading states
4. Include responsive design considerations
5. Test components across different screen sizes

## 📄 License

This project is part of the VEX U Scoring Analysis Platform and follows the same licensing terms as the parent project.