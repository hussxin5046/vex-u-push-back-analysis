import React, { useState, useMemo } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  TableSortLabel,
  Paper,
  Checkbox,
  IconButton,
  TextField,
  InputAdornment,
  Box,
  Typography,
  Chip,
  Button,
  Menu,
  MenuItem,
  Tooltip,
  FormControl,
  InputLabel,
  Select,
  Alert,
  Skeleton,
} from '@mui/material';
import {
  Search as SearchIcon,
  FilterList as FilterIcon,
  Download as DownloadIcon,
  Visibility as ViewIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  MoreVert as MoreVertIcon,
  ViewColumn as ColumnsIcon,
} from '@mui/icons-material';

export interface TableColumn {
  id: string;
  label: string;
  minWidth?: number;
  align?: 'left' | 'right' | 'center';
  format?: (value: any) => string | React.ReactNode;
  sortable?: boolean;
  filterable?: boolean;
  hidden?: boolean;
  sticky?: boolean;
}

export interface TableRow {
  id: string;
  [key: string]: any;
}

interface TableAction {
  id: string;
  label: string;
  icon?: React.ReactNode;
  onClick: (row: TableRow) => void;
  color?: 'primary' | 'secondary' | 'error' | 'warning' | 'success';
  disabled?: (row: TableRow) => boolean;
}

interface DataTableProps {
  columns: TableColumn[];
  rows: TableRow[];
  loading?: boolean;
  error?: string;
  title?: string;
  selectable?: boolean;
  searchable?: boolean;
  filterable?: boolean;
  exportable?: boolean;
  actions?: TableAction[];
  onRowClick?: (row: TableRow) => void;
  onRowSelect?: (selectedRows: string[]) => void;
  onExport?: (data: TableRow[]) => void;
  initialPageSize?: number;
  stickyHeader?: boolean;
  dense?: boolean;
  emptyMessage?: string;
}

type Order = 'asc' | 'desc';

const DataTable: React.FC<DataTableProps> = ({
  columns,
  rows,
  loading = false,
  error,
  title,
  selectable = false,
  searchable = true,
  filterable = false,
  exportable = false,
  actions = [],
  onRowClick,
  onRowSelect,
  onExport,
  initialPageSize = 10,
  stickyHeader = false,
  dense = false,
  emptyMessage = 'No data available',
}) => {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(initialPageSize);
  const [orderBy, setOrderBy] = useState<string>('');
  const [order, setOrder] = useState<Order>('asc');
  const [selected, setSelected] = useState<string[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [filters, setFilters] = useState<Record<string, string>>({});
  const [visibleColumns, setVisibleColumns] = useState<string[]>(columns.map(col => col.id));
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [columnsMenuAnchor, setColumnsMenuAnchor] = useState<null | HTMLElement>(null);

  const visibleColumnDefs = useMemo(
    () => columns.filter(col => visibleColumns.includes(col.id) && !col.hidden),
    [columns, visibleColumns]
  );

  const filteredRows = useMemo(() => {
    let filtered = rows;

    // Apply search filter
    if (searchTerm) {
      filtered = filtered.filter(row =>
        visibleColumnDefs.some(column => {
          const value = row[column.id];
          return value?.toString().toLowerCase().includes(searchTerm.toLowerCase());
        })
      );
    }

    // Apply column filters
    filtered = filtered.filter(row => {
      return Object.entries(filters).every(([columnId, filterValue]) => {
        if (!filterValue) return true;
        const value = row[columnId];
        return value?.toString().toLowerCase().includes(filterValue.toLowerCase());
      });
    });

    return filtered;
  }, [rows, searchTerm, filters, visibleColumnDefs]);

  const sortedRows = useMemo(() => {
    if (!orderBy) return filteredRows;

    return [...filteredRows].sort((a, b) => {
      const aValue = a[orderBy];
      const bValue = b[orderBy];

      if (aValue === bValue) return 0;
      
      let comparison = 0;
      if (typeof aValue === 'number' && typeof bValue === 'number') {
        comparison = aValue - bValue;
      } else {
        comparison = String(aValue).localeCompare(String(bValue));
      }

      return order === 'desc' ? -comparison : comparison;
    });
  }, [filteredRows, orderBy, order]);

  const paginatedRows = useMemo(() => {
    const start = page * rowsPerPage;
    return sortedRows.slice(start, start + rowsPerPage);
  }, [sortedRows, page, rowsPerPage]);

  const handleRequestSort = (property: string) => {
    const isAsc = orderBy === property && order === 'asc';
    setOrder(isAsc ? 'desc' : 'asc');
    setOrderBy(property);
  };

  const handleSelectAllClick = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.checked) {
      const newSelected = sortedRows.map(row => row.id);
      setSelected(newSelected);
      onRowSelect?.(newSelected);
    } else {
      setSelected([]);
      onRowSelect?.([]);
    }
  };

  const handleRowSelect = (id: string) => {
    const selectedIndex = selected.indexOf(id);
    let newSelected: string[] = [];

    if (selectedIndex === -1) {
      newSelected = newSelected.concat(selected, id);
    } else if (selectedIndex === 0) {
      newSelected = newSelected.concat(selected.slice(1));
    } else if (selectedIndex === selected.length - 1) {
      newSelected = newSelected.concat(selected.slice(0, -1));
    } else if (selectedIndex > 0) {
      newSelected = newSelected.concat(
        selected.slice(0, selectedIndex),
        selected.slice(selectedIndex + 1)
      );
    }

    setSelected(newSelected);
    onRowSelect?.(newSelected);
  };

  const handleChangePage = (_: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const handleFilterChange = (columnId: string, value: string) => {
    setFilters(prev => ({
      ...prev,
      [columnId]: value,
    }));
    setPage(0);
  };

  const handleExport = () => {
    const dataToExport = selected.length > 0 
      ? sortedRows.filter(row => selected.includes(row.id))
      : sortedRows;
    onExport?.(dataToExport);
  };

  const toggleColumnVisibility = (columnId: string) => {
    setVisibleColumns(prev =>
      prev.includes(columnId)
        ? prev.filter(id => id !== columnId)
        : [...prev, columnId]
    );
  };

  const isSelected = (id: string) => selected.indexOf(id) !== -1;

  if (error) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        <Typography variant="h6">Error loading data</Typography>
        <Typography>{error}</Typography>
      </Alert>
    );
  }

  return (
    <Paper sx={{ width: '100%', overflow: 'hidden' }}>
      {/* Table Header with controls */}
      {(title || searchable || filterable || exportable) && (
        <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            {title && (
              <Typography variant="h6" component="div">
                {title}
                {selected.length > 0 && (
                  <Chip
                    label={`${selected.length} selected`}
                    size="small"
                    color="primary"
                    sx={{ ml: 2 }}
                  />
                )}
              </Typography>
            )}
            
            <Box sx={{ display: 'flex', gap: 1 }}>
              {exportable && (
                <Button
                  size="small"
                  startIcon={<DownloadIcon />}
                  onClick={handleExport}
                  disabled={sortedRows.length === 0}
                >
                  Export
                </Button>
              )}
              
              <IconButton
                size="small"
                onClick={(e) => setColumnsMenuAnchor(e.currentTarget)}
              >
                <ColumnsIcon />
              </IconButton>
            </Box>
          </Box>

          {/* Search and Filters */}
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'center' }}>
            {searchable && (
              <TextField
                placeholder="Search..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                size="small"
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <SearchIcon />
                    </InputAdornment>
                  ),
                }}
                sx={{ minWidth: 200 }}
              />
            )}

            {filterable && visibleColumnDefs.filter(col => col.filterable).map(column => (
              <FormControl size="small" key={column.id} sx={{ minWidth: 120 }}>
                <InputLabel>{column.label}</InputLabel>
                <Select
                  value={filters[column.id] || ''}
                  onChange={(e) => handleFilterChange(column.id, e.target.value)}
                  label={column.label}
                >
                  <MenuItem value="">All</MenuItem>
                  {Array.from(new Set(rows.map(row => row[column.id]))).map(value => (
                    <MenuItem key={String(value)} value={String(value)}>
                      {String(value)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            ))}
          </Box>
        </Box>
      )}

      {/* Table */}
      <TableContainer sx={{ maxHeight: 600 }}>
        <Table stickyHeader={stickyHeader} size={dense ? 'small' : 'medium'}>
          <TableHead>
            <TableRow>
              {selectable && (
                <TableCell padding="checkbox">
                  <Checkbox
                    color="primary"
                    indeterminate={selected.length > 0 && selected.length < sortedRows.length}
                    checked={sortedRows.length > 0 && selected.length === sortedRows.length}
                    onChange={handleSelectAllClick}
                    disabled={loading}
                  />
                </TableCell>
              )}
              
              {visibleColumnDefs.map((column) => (
                <TableCell
                  key={column.id}
                  align={column.align}
                  style={{ minWidth: column.minWidth, position: column.sticky ? 'sticky' : 'static' }}
                >
                  {column.sortable !== false ? (
                    <TableSortLabel
                      active={orderBy === column.id}
                      direction={orderBy === column.id ? order : 'asc'}
                      onClick={() => handleRequestSort(column.id)}
                      disabled={loading}
                    >
                      {column.label}
                    </TableSortLabel>
                  ) : (
                    column.label
                  )}
                </TableCell>
              ))}
              
              {actions.length > 0 && (
                <TableCell align="center">Actions</TableCell>
              )}
            </TableRow>
          </TableHead>
          
          <TableBody>
            {loading ? (
              Array.from({ length: rowsPerPage }).map((_, index) => (
                <TableRow key={index}>
                  {selectable && <TableCell><Skeleton /></TableCell>}
                  {visibleColumnDefs.map((column) => (
                    <TableCell key={column.id}>
                      <Skeleton />
                    </TableCell>
                  ))}
                  {actions.length > 0 && <TableCell><Skeleton /></TableCell>}
                </TableRow>
              ))
            ) : paginatedRows.length === 0 ? (
              <TableRow>
                <TableCell 
                  colSpan={visibleColumnDefs.length + (selectable ? 1 : 0) + (actions.length > 0 ? 1 : 0)}
                  align="center"
                  sx={{ py: 8 }}
                >
                  <Typography variant="body1" color="text.secondary">
                    {emptyMessage}
                  </Typography>
                </TableCell>
              </TableRow>
            ) : (
              paginatedRows.map((row) => {
                const isItemSelected = isSelected(row.id);
                return (
                  <TableRow
                    hover
                    key={row.id}
                    selected={isItemSelected}
                    onClick={onRowClick ? () => onRowClick(row) : undefined}
                    sx={{ cursor: onRowClick ? 'pointer' : 'default' }}
                  >
                    {selectable && (
                      <TableCell padding="checkbox">
                        <Checkbox
                          color="primary"
                          checked={isItemSelected}
                          onChange={() => handleRowSelect(row.id)}
                          onClick={(e) => e.stopPropagation()}
                        />
                      </TableCell>
                    )}
                    
                    {visibleColumnDefs.map((column) => {
                      const value = row[column.id];
                      return (
                        <TableCell key={column.id} align={column.align}>
                          {column.format ? column.format(value) : value}
                        </TableCell>
                      );
                    })}
                    
                    {actions.length > 0 && (
                      <TableCell align="center">
                        <Box sx={{ display: 'flex', justifyContent: 'center', gap: 0.5 }}>
                          {actions.slice(0, 3).map((action) => (
                            <Tooltip key={action.id} title={action.label}>
                              <IconButton
                                size="small"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  action.onClick(row);
                                }}
                                color={action.color}
                                disabled={action.disabled?.(row)}
                              >
                                {action.icon}
                              </IconButton>
                            </Tooltip>
                          ))}
                          {actions.length > 3 && (
                            <IconButton
                              size="small"
                              onClick={(e) => {
                                e.stopPropagation();
                                setAnchorEl(e.currentTarget);
                              }}
                            >
                              <MoreVertIcon />
                            </IconButton>
                          )}
                        </Box>
                      </TableCell>
                    )}
                  </TableRow>
                );
              })
            )}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Pagination */}
      <TablePagination
        rowsPerPageOptions={[5, 10, 25, 50]}
        component="div"
        count={sortedRows.length}
        rowsPerPage={rowsPerPage}
        page={page}
        onPageChange={handleChangePage}
        onRowsPerPageChange={handleChangeRowsPerPage}
      />

      {/* Action Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={() => setAnchorEl(null)}
      >
        {actions.slice(3).map((action) => (
          <MenuItem
            key={action.id}
            onClick={() => {
              // action.onClick would need the row context here
              setAnchorEl(null);
            }}
          >
            {action.icon && <Box sx={{ mr: 1, display: 'flex' }}>{action.icon}</Box>}
            {action.label}
          </MenuItem>
        ))}
      </Menu>

      {/* Column Visibility Menu */}
      <Menu
        anchorEl={columnsMenuAnchor}
        open={Boolean(columnsMenuAnchor)}
        onClose={() => setColumnsMenuAnchor(null)}
      >
        {columns.map((column) => (
          <MenuItem key={column.id}>
            <Checkbox
              checked={visibleColumns.includes(column.id)}
              onChange={() => toggleColumnVisibility(column.id)}
            />
            {column.label}
          </MenuItem>
        ))}
      </Menu>
    </Paper>
  );
};

export default DataTable;