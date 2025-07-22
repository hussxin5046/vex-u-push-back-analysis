import React, { useState } from 'react';
import {
  Button,
  ButtonGroup,
  ClickAwayListener,
  Grow,
  Paper,
  Popper,
  MenuItem,
  MenuList,
  ListItemIcon,
  ListItemText,
  CircularProgress,
  Typography,
  Box,
  Divider,
} from '@mui/material';
import {
  Download as DownloadIcon,
  ArrowDropDown as ArrowDropDownIcon,
  PictureAsPdf as PdfIcon,
  TableChart as ExcelIcon,
  InsertDriveFile as WordIcon,
  Code as JsonIcon,
  DataObject as CsvIcon,
  Image as ImageIcon,
  Share as ShareIcon,
} from '@mui/icons-material';

export interface ExportFormat {
  id: string;
  label: string;
  icon: React.ReactNode;
  description?: string;
  extension: string;
  mimeType: string;
}

export interface ExportOptions {
  includeHeaders?: boolean;
  includeFilters?: boolean;
  includeCharts?: boolean;
  dateRange?: string;
  customFields?: string[];
}

interface ExportButtonProps {
  data?: any;
  filename?: string;
  formats?: ExportFormat[];
  onExport: (format: ExportFormat, options?: ExportOptions) => Promise<void> | void;
  onShare?: () => void;
  loading?: boolean;
  disabled?: boolean;
  variant?: 'contained' | 'outlined' | 'text';
  size?: 'small' | 'medium' | 'large';
  showDropdown?: boolean;
  showShare?: boolean;
  defaultFormat?: string;
  customFormats?: ExportFormat[];
}

const defaultFormats: ExportFormat[] = [
  {
    id: 'pdf',
    label: 'PDF Document',
    icon: <PdfIcon />,
    description: 'Portable Document Format',
    extension: 'pdf',
    mimeType: 'application/pdf',
  },
  {
    id: 'excel',
    label: 'Excel Workbook',
    icon: <ExcelIcon />,
    description: 'Microsoft Excel format',
    extension: 'xlsx',
    mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  },
  {
    id: 'csv',
    label: 'CSV File',
    icon: <CsvIcon />,
    description: 'Comma-separated values',
    extension: 'csv',
    mimeType: 'text/csv',
  },
  {
    id: 'json',
    label: 'JSON Data',
    icon: <JsonIcon />,
    description: 'JavaScript Object Notation',
    extension: 'json',
    mimeType: 'application/json',
  },
  {
    id: 'png',
    label: 'PNG Image',
    icon: <ImageIcon />,
    description: 'Portable Network Graphics',
    extension: 'png',
    mimeType: 'image/png',
  },
];

const ExportButton: React.FC<ExportButtonProps> = ({
  data,
  filename = 'export',
  formats = defaultFormats,
  onExport,
  onShare,
  loading = false,
  disabled = false,
  variant = 'contained',
  size = 'medium',
  showDropdown = true,
  showShare = false,
  defaultFormat = 'pdf',
  customFormats = [],
}) => {
  const [open, setOpen] = useState(false);
  const [exportLoading, setExportLoading] = useState<string | null>(null);
  const anchorRef = React.useRef<HTMLDivElement>(null);

  const allFormats = [...formats, ...customFormats];
  const defaultFormatObj = allFormats.find(f => f.id === defaultFormat) || allFormats[0];

  const handleToggle = () => {
    setOpen((prevOpen) => !prevOpen);
  };

  const handleClose = (event: Event) => {
    if (anchorRef.current && anchorRef.current.contains(event.target as HTMLElement)) {
      return;
    }
    setOpen(false);
  };

  const handleExport = async (format: ExportFormat, options?: ExportOptions) => {
    setExportLoading(format.id);
    try {
      await onExport(format, options);
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setExportLoading(null);
      setOpen(false);
    }
  };

  const generateFilename = (format: ExportFormat) => {
    const timestamp = new Date().toISOString().slice(0, 19).replace(/[:-]/g, '');
    return `${filename}_${timestamp}.${format.extension}`;
  };

  const downloadData = (content: string, format: ExportFormat) => {
    const blob = new Blob([content], { type: format.mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = generateFilename(format);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleDefaultExport = () => {
    if (defaultFormatObj) {
      handleExport(defaultFormatObj);
    }
  };

  if (!showDropdown) {
    return (
      <Button
        variant={variant}
        size={size}
        onClick={handleDefaultExport}
        disabled={disabled || loading}
        startIcon={
          loading ? <CircularProgress size={16} /> : defaultFormatObj?.icon || <DownloadIcon />
        }
      >
        {loading ? 'Exporting...' : `Export ${defaultFormatObj?.label || 'Data'}`}
      </Button>
    );
  }

  return (
    <>
      <ButtonGroup variant={variant} ref={anchorRef} disabled={disabled || loading}>
        <Button
          onClick={handleDefaultExport}
          startIcon={
            exportLoading === defaultFormatObj?.id ? (
              <CircularProgress size={16} />
            ) : (
              defaultFormatObj?.icon || <DownloadIcon />
            )
          }
          disabled={disabled || loading || Boolean(exportLoading)}
        >
          {exportLoading === defaultFormatObj?.id
            ? 'Exporting...'
            : `Export ${defaultFormatObj?.label || 'Data'}`}
        </Button>
        <Button
          size={size}
          onClick={handleToggle}
          disabled={disabled || loading || Boolean(exportLoading)}
        >
          <ArrowDropDownIcon />
        </Button>
      </ButtonGroup>

      <Popper
        open={open}
        anchorEl={anchorRef.current}
        role={undefined}
        transition
        disablePortal
        placement="bottom-end"
        style={{ zIndex: 1300 }}
      >
        {({ TransitionProps, placement }) => (
          <Grow
            {...TransitionProps}
            style={{
              transformOrigin: placement === 'bottom' ? 'center top' : 'center bottom',
            }}
          >
            <Paper elevation={3}>
              <ClickAwayListener onClickAway={handleClose}>
                <MenuList autoFocusItem>
                  <Box sx={{ px: 2, py: 1 }}>
                    <Typography variant="subtitle2" color="text.secondary">
                      Export Options
                    </Typography>
                  </Box>
                  <Divider />
                  
                  {allFormats.map((format) => (
                    <MenuItem
                      key={format.id}
                      onClick={() => handleExport(format)}
                      disabled={exportLoading === format.id}
                    >
                      <ListItemIcon>
                        {exportLoading === format.id ? (
                          <CircularProgress size={20} />
                        ) : (
                          format.icon
                        )}
                      </ListItemIcon>
                      <ListItemText>
                        <Typography variant="body2">{format.label}</Typography>
                        {format.description && (
                          <Typography variant="caption" color="text.secondary">
                            {format.description}
                          </Typography>
                        )}
                      </ListItemText>
                    </MenuItem>
                  ))}

                  {showShare && onShare && (
                    <>
                      <Divider />
                      <MenuItem onClick={onShare}>
                        <ListItemIcon>
                          <ShareIcon />
                        </ListItemIcon>
                        <ListItemText>
                          <Typography variant="body2">Share Link</Typography>
                          <Typography variant="caption" color="text.secondary">
                            Generate shareable link
                          </Typography>
                        </ListItemText>
                      </MenuItem>
                    </>
                  )}
                </MenuList>
              </ClickAwayListener>
            </Paper>
          </Grow>
        )}
      </Popper>
    </>
  );
};

// Utility functions for common export formats
export const exportToCSV = (data: any[], filename: string) => {
  if (!Array.isArray(data) || data.length === 0) return;

  const headers = Object.keys(data[0]);
  const csvContent = [
    headers.join(','),
    ...data.map(row => 
      headers.map(header => {
        const value = row[header];
        const stringValue = value?.toString() || '';
        return stringValue.includes(',') ? `"${stringValue}"` : stringValue;
      }).join(',')
    ),
  ].join('\n');

  const blob = new Blob([csvContent], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${filename}.csv`;
  link.click();
  URL.revokeObjectURL(url);
};

export const exportToJSON = (data: any, filename: string) => {
  const jsonContent = JSON.stringify(data, null, 2);
  const blob = new Blob([jsonContent], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${filename}.json`;
  link.click();
  URL.revokeObjectURL(url);
};

export const exportToText = (content: string, filename: string) => {
  const blob = new Blob([content], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${filename}.txt`;
  link.click();
  URL.revokeObjectURL(url);
};

export default ExportButton;