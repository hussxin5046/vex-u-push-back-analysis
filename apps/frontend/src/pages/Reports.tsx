import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  FormGroup,
  FormControlLabel,
  Checkbox,
  LinearProgress,
  Tabs,
  Tab,
} from '@mui/material';
import {
  Add as AddIcon,
  Download as DownloadIcon,
  Visibility as ViewIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Assessment as ReportIcon,
  PictureAsPdf as PdfIcon,
  TableChart as ExcelIcon,
  InsertDriveFile as WordIcon,
  Share as ShareIcon,
} from '@mui/icons-material';
import { useNotification } from '../contexts/NotificationContext';
import { ReportData, ReportSection } from '../types';

interface ReportTemplate {
  id: string;
  name: string;
  description: string;
  sections: string[];
  estimatedTime: number;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index, ...other }) => (
  <div
    role="tabpanel"
    hidden={value !== index}
    id={`report-tabpanel-${index}`}
    aria-labelledby={`report-tab-${index}`}
    {...other}
  >
    {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
  </div>
);

const Reports: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [reports, setReports] = useState<ReportData[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationProgress, setGenerationProgress] = useState(0);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [previewDialogOpen, setPreviewDialogOpen] = useState(false);
  const [selectedReport, setSelectedReport] = useState<ReportData | null>(null);
  
  const [newReport, setNewReport] = useState({
    title: '',
    type: 'comprehensive' as 'strategy' | 'statistical' | 'ml_insights' | 'comprehensive',
    template: '',
    includeCharts: true,
    includeRecommendations: true,
    dataRange: 'last_30_days',
    outputFormat: 'pdf',
  });

  const { showNotification } = useNotification();

  const reportTemplates: ReportTemplate[] = [
    {
      id: 'comprehensive',
      name: 'Comprehensive Analysis',
      description: 'Complete analysis including all metrics, charts, and insights',
      sections: ['Executive Summary', 'Performance Metrics', 'Strategy Analysis', 'ML Insights', 'Recommendations'],
      estimatedTime: 5,
    },
    {
      id: 'strategy_focus',
      name: 'Strategy-Focused Report',
      description: 'Detailed analysis of strategic approaches and effectiveness',
      sections: ['Strategy Overview', 'Performance Comparison', 'Optimization Opportunities'],
      estimatedTime: 3,
    },
    {
      id: 'statistical_summary',
      name: 'Statistical Summary',
      description: 'Statistical breakdown and trend analysis',
      sections: ['Data Summary', 'Statistical Analysis', 'Trend Identification'],
      estimatedTime: 2,
    },
    {
      id: 'ml_insights',
      name: 'ML Insights Report',
      description: 'Machine learning predictions and model insights',
      sections: ['Model Performance', 'Predictions', 'Feature Importance', 'Recommendations'],
      estimatedTime: 4,
    },
  ];

  const dataRanges = [
    { value: 'last_7_days', label: 'Last 7 Days' },
    { value: 'last_30_days', label: 'Last 30 Days' },
    { value: 'last_90_days', label: 'Last 90 Days' },
    { value: 'this_season', label: 'This Season' },
    { value: 'all_time', label: 'All Time' },
  ];

  const outputFormats = [
    { value: 'pdf', label: 'PDF Document', icon: <PdfIcon /> },
    { value: 'excel', label: 'Excel Workbook', icon: <ExcelIcon /> },
    { value: 'word', label: 'Word Document', icon: <WordIcon /> },
    { value: 'html', label: 'HTML Report', icon: <ReportIcon /> },
  ];

  const handleCreateReport = async () => {
    if (!newReport.title || !newReport.template) {
      showNotification({
        type: 'warning',
        title: 'Missing Information',
        message: 'Please provide a title and select a template.',
      });
      return;
    }

    setIsGenerating(true);
    setGenerationProgress(0);
    setCreateDialogOpen(false);

    try {
      // Simulate progressive report generation
      const steps = ['Collecting Data', 'Analyzing Metrics', 'Generating Charts', 'Creating Insights', 'Formatting Report'];
      
      for (let i = 0; i <= 100; i += 20) {
        setGenerationProgress(i);
        await new Promise(resolve => setTimeout(resolve, 800));
      }

      const template = reportTemplates.find(t => t.id === newReport.template);
      const mockReport: ReportData = {
        id: `report_${Date.now()}`,
        title: newReport.title,
        type: newReport.type,
        summary: `${template?.name} generated from ${dataRanges.find(d => d.value === newReport.dataRange)?.label} data`,
        sections: template?.sections.map((sectionTitle, index) => ({
          id: `section_${index}`,
          title: sectionTitle,
          content: `This section contains detailed analysis of ${sectionTitle.toLowerCase()}. The data shows significant insights that can help improve performance and strategic decision-making.`,
          importance: index === 0 ? 'high' : index === 1 ? 'medium' : 'low',
        })) || [],
        charts: [
          { id: 'chart1', type: 'bar', title: 'Performance Overview', data: [] },
          { id: 'chart2', type: 'line', title: 'Trend Analysis', data: [] },
        ],
        recommendations: [
          'Focus on autonomous period optimization',
          'Improve alliance coordination strategies',
          'Enhance driver control efficiency',
          'Consider strategy adjustments based on opponent analysis',
        ],
        metadata: {
          generated_at: new Date().toISOString(),
          data_range: newReport.dataRange,
          confidence_level: 0.85,
        },
      };

      setReports(prev => [mockReport, ...prev]);
      setNewReport({
        title: '',
        type: 'comprehensive',
        template: '',
        includeCharts: true,
        includeRecommendations: true,
        dataRange: 'last_30_days',
        outputFormat: 'pdf',
      });

      showNotification({
        type: 'success',
        title: 'Report Generated',
        message: `${template?.name} has been generated successfully.`,
      });
    } catch (error) {
      showNotification({
        type: 'error',
        title: 'Generation Failed',
        message: 'Failed to generate report. Please try again.',
      });
    } finally {
      setIsGenerating(false);
      setGenerationProgress(0);
    }
  };

  const downloadReport = (report: ReportData, format: string) => {
    const filename = `${report.title.replace(/\s+/g, '_')}.${format}`;
    showNotification({
      type: 'success',
      title: 'Download Started',
      message: `Downloading ${filename}...`,
    });
  };

  const deleteReport = (reportId: string) => {
    setReports(prev => prev.filter(r => r.id !== reportId));
    showNotification({
      type: 'info',
      title: 'Report Deleted',
      message: 'Report has been deleted successfully.',
    });
  };

  const previewReport = (report: ReportData) => {
    setSelectedReport(report);
    setPreviewDialogOpen(true);
  };

  const shareReport = (report: ReportData) => {
    showNotification({
      type: 'info',
      title: 'Share Link Created',
      message: 'Report sharing link has been copied to clipboard.',
    });
  };

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 700 }}>
        Reports
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Generate, manage, and share comprehensive VEX U analysis reports.
      </Typography>

      {/* Generation Progress */}
      {isGenerating && (
        <Alert severity="info" sx={{ mb: 3 }}>
          <Typography variant="body2" gutterBottom>
            Generating report... ({generationProgress}%)
          </Typography>
          <LinearProgress variant="determinate" value={generationProgress} />
        </Alert>
      )}

      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)}>
          <Tab label="Report Library" />
          <Tab label="Templates" />
        </Tabs>
      </Box>

      <TabPanel value={activeTab} index={0}>
        {/* Report Library */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Report Library ({reports.length})
          </Typography>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setCreateDialogOpen(true)}
            disabled={isGenerating}
          >
            Generate Report
          </Button>
        </Box>

        {reports.length > 0 ? (
          <TableContainer component={Paper} variant="outlined">
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Report</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell>Generated</TableCell>
                  <TableCell>Data Range</TableCell>
                  <TableCell>Confidence</TableCell>
                  <TableCell align="center">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {reports.map((report) => (
                  <TableRow key={report.id} hover>
                    <TableCell>
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        {report.title}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {report.sections.length} sections, {report.charts.length} charts
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={report.type.replace('_', ' ')}
                        size="small"
                        color="primary"
                        variant="outlined"
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {new Date(report.metadata.generated_at).toLocaleDateString()}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {new Date(report.metadata.generated_at).toLocaleTimeString()}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={dataRanges.find(d => d.value === report.metadata.data_range)?.label || report.metadata.data_range}
                        size="small"
                        variant="outlined"
                      />
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={`${(report.metadata.confidence_level * 100).toFixed(0)}%`}
                        size="small"
                        color={report.metadata.confidence_level > 0.8 ? 'success' : 'warning'}
                      />
                    </TableCell>
                    <TableCell align="center">
                      <IconButton size="small" onClick={() => previewReport(report)}>
                        <ViewIcon />
                      </IconButton>
                      <IconButton 
                        size="small" 
                        onClick={() => downloadReport(report, 'pdf')}
                        color="primary"
                      >
                        <DownloadIcon />
                      </IconButton>
                      <IconButton size="small" onClick={() => shareReport(report)}>
                        <ShareIcon />
                      </IconButton>
                      <IconButton 
                        size="small" 
                        onClick={() => deleteReport(report.id)}
                        color="error"
                      >
                        <DeleteIcon />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        ) : (
          <Card>
            <CardContent sx={{ textAlign: 'center', py: 8 }}>
              <ReportIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h6" color="text.secondary" gutterBottom>
                No Reports Generated Yet
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Generate your first report to get comprehensive insights from your VEX U analysis data.
              </Typography>
              <Button
                variant="outlined"
                startIcon={<AddIcon />}
                onClick={() => setCreateDialogOpen(true)}
              >
                Generate Your First Report
              </Button>
            </CardContent>
          </Card>
        )}
      </TabPanel>

      <TabPanel value={activeTab} index={1}>
        {/* Templates */}
        <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
          Available Templates
        </Typography>
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 3 }}>
          {reportTemplates.map((template) => (
            <Box key={template.id}>
              <Card sx={{ height: '100%' }}>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', mb: 2 }}>
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      {template.name}
                    </Typography>
                    <Chip
                      label={`~${template.estimatedTime} min`}
                      size="small"
                      color="secondary"
                    />
                  </Box>
                  
                  <Typography variant="body2" color="text.secondary" paragraph>
                    {template.description}
                  </Typography>
                  
                  <Typography variant="subtitle2" gutterBottom>
                    Included Sections:
                  </Typography>
                  <Box sx={{ mb: 2 }}>
                    {template.sections.map((section) => (
                      <Chip
                        key={section}
                        label={section}
                        size="small"
                        variant="outlined"
                        sx={{ mr: 1, mb: 1 }}
                      />
                    ))}
                  </Box>
                  
                  <Button
                    variant="outlined"
                    startIcon={<AddIcon />}
                    onClick={() => {
                      setNewReport(prev => ({ ...prev, template: template.id, type: template.id as any }));
                      setCreateDialogOpen(true);
                    }}
                    fullWidth
                  >
                    Use This Template
                  </Button>
                </CardContent>
              </Card>
            </Box>
          ))}
        </Box>
      </TabPanel>

      {/* Create Report Dialog */}
      <Dialog open={createDialogOpen} onClose={() => setCreateDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Generate New Report</DialogTitle>
        <DialogContent>
          <TextField
            label="Report Title"
            value={newReport.title}
            onChange={(e) => setNewReport(prev => ({ ...prev, title: e.target.value }))}
            fullWidth
            margin="normal"
          />
          
          <FormControl fullWidth margin="normal">
            <InputLabel>Template</InputLabel>
            <Select
              value={newReport.template}
              onChange={(e) => setNewReport(prev => ({ ...prev, template: e.target.value }))}
            >
              {reportTemplates.map((template) => (
                <MenuItem key={template.id} value={template.id}>
                  {template.name} (~{template.estimatedTime} min)
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          
          <FormControl fullWidth margin="normal">
            <InputLabel>Data Range</InputLabel>
            <Select
              value={newReport.dataRange}
              onChange={(e) => setNewReport(prev => ({ ...prev, dataRange: e.target.value }))}
            >
              {dataRanges.map((range) => (
                <MenuItem key={range.value} value={range.value}>
                  {range.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          
          <FormControl fullWidth margin="normal">
            <InputLabel>Output Format</InputLabel>
            <Select
              value={newReport.outputFormat}
              onChange={(e) => setNewReport(prev => ({ ...prev, outputFormat: e.target.value }))}
            >
              {outputFormats.map((format) => (
                <MenuItem key={format.value} value={format.value}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    {format.icon}
                    <Typography sx={{ ml: 1 }}>{format.label}</Typography>
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          
          <FormGroup sx={{ mt: 2 }}>
            <FormControlLabel
              control={
                <Checkbox
                  checked={newReport.includeCharts}
                  onChange={(e) => setNewReport(prev => ({ ...prev, includeCharts: e.target.checked }))}
                />
              }
              label="Include Charts and Visualizations"
            />
            <FormControlLabel
              control={
                <Checkbox
                  checked={newReport.includeRecommendations}
                  onChange={(e) => setNewReport(prev => ({ ...prev, includeRecommendations: e.target.checked }))}
                />
              }
              label="Include Recommendations"
            />
          </FormGroup>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleCreateReport} variant="contained">
            Generate Report
          </Button>
        </DialogActions>
      </Dialog>

      {/* Preview Dialog */}
      <Dialog open={previewDialogOpen} onClose={() => setPreviewDialogOpen(false)} maxWidth="lg" fullWidth>
        <DialogTitle>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="h6">{selectedReport?.title}</Typography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              {outputFormats.map((format) => (
                <IconButton
                  key={format.value}
                  size="small"
                  onClick={() => selectedReport && downloadReport(selectedReport, format.value)}
                >
                  {format.icon}
                </IconButton>
              ))}
            </Box>
          </Box>
        </DialogTitle>
        <DialogContent>
          {selectedReport && (
            <Box>
              <Alert severity="info" sx={{ mb: 2 }}>
                {selectedReport.summary}
              </Alert>
              
              {selectedReport.sections.map((section) => (
                <Card key={section.id} sx={{ mb: 2 }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      {section.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {section.content}
                    </Typography>
                  </CardContent>
                </Card>
              ))}
              
              {selectedReport.recommendations.length > 0 && (
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Recommendations
                    </Typography>
                    {selectedReport.recommendations.map((rec, index) => (
                      <Alert key={index} severity="info" sx={{ mb: 1 }}>
                        {rec}
                      </Alert>
                    ))}
                  </CardContent>
                </Card>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPreviewDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Reports;