# PushBackMLModel Project Layout

This document describes the current structure of the PushBackMLModel project.

## Root Directory Structure

```
PushBackMLModel/
├── vex_u_scoring_analysis/          # Main project directory
└── vex_u_scoring_analysis_backup/   # Backup directory
```

## Main Project Structure (vex_u_scoring_analysis/)

### Core Files
- `main.py` - Main entry point for the application
- `requirements.txt` - Python dependencies
- `setup.py` - Package setup configuration
- `README.md` - Project documentation

### Source Code (`src/`)
```
src/
├── __init__.py
├── analysis/
│   ├── __init__.py
│   ├── scoring_analyzer.py     # Scoring analysis functionality
│   ├── statistical_analyzer.py # Statistical analysis tools
│   └── strategy_analyzer.py    # Strategic analysis components
├── core/
│   ├── __init__.py
│   ├── scenario_generator.py   # Scenario generation logic
│   └── simulator.py           # Simulation engine
├── reporting/
│   ├── __init__.py
│   └── generator.py           # Report generation
└── visualization/
    ├── __init__.py
    ├── interactive.py         # Interactive visualizations
    └── static.py             # Static visualizations
```

### Testing (`tests/`)
```
tests/
├── __init__.py
├── comprehensive_tests.py      # Comprehensive test suite
├── scenario_analysis_tests.py  # Scenario analysis tests
├── system_integration.py      # Integration tests
├── test_enhanced_generator.py # Enhanced generator tests
└── test_simulator.py         # Simulator tests
```

### Documentation (`docs/`)
```
docs/
├── ANALYSIS_SUMMARY.md        # Analysis summary documentation
└── FILES_ANALYSIS.md         # File analysis documentation
```

### Demonstrations (`demos/`)
```
demos/
├── quick_demo.py             # Quick demonstration script
├── showcase_demo.py          # Showcase demonstration
└── statistical_demo.py      # Statistical analysis demo
```

### Scripts (`scripts/`)
```
scripts/
└── validate.py              # Validation script
```

### Reports (`reports/`)
```
reports/
└── VEX_U_Strategic_Report_VEX_U_Team_20250722_020349.html  # Generated report
```

## Project Type
This appears to be a **VEX U robotics competition scoring analysis system** with the following capabilities:
- Scoring analysis and simulation
- Statistical analysis tools
- Strategy optimization
- Interactive and static visualizations
- Comprehensive testing suite
- Report generation

## Architecture
The project follows a modular Python package structure with:
- Clear separation of concerns (analysis, core, reporting, visualization)
- Comprehensive test coverage
- Documentation and demonstrations
- Script-based utilities
- Generated reports output

## Backup Structure
The `vex_u_scoring_analysis_backup/` directory contains an identical copy of the main project structure, serving as a backup of the codebase.