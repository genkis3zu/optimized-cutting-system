# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Steel Cutting Optimization System - A Streamlit-based application for optimizing steel panel cutting operations with guillotine cut constraints. The system minimizes material waste while respecting real-world manufacturing constraints.

## Development Commands

### Setup and Installation
```bash
# Create virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Run Streamlit app
streamlit run app.py

# Run with custom port
streamlit run app.py --server.port 8080

# Development mode with auto-reload
streamlit run app.py --server.runOnSave true
```

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_optimizer.py -v

# Run with coverage
python -m pytest tests/ --cov=core --cov=cutting --cov-report=html
```

### Code Quality
```bash
# Run linting
python -m pylint core/ cutting/ ui/

# Format code
python -m black core/ cutting/ ui/ tests/

# Type checking
python -m mypy core/ cutting/ ui/
```

## Architecture Overview

### Core Algorithm Components

The system implements multiple 2D bin packing algorithms with guillotine cut constraints:

1. **Guillotine Cut Constraint**: All cuts must go from edge to edge (no L-shaped cuts)
2. **Material Blocks**: Panels grouped by material type for batch processing
3. **Cutting Order**: Top-to-bottom processing within each material block

### Critical Business Constraints

- **Panel Size Limits**:
  - Minimum: 50mm × 50mm
  - Maximum Width: 1500mm
  - Maximum Height: 3100mm
- **Standard Sheet Size**: 1500mm × 3100mm
- **Kerf (Cut Width)**: 3-5mm must be considered in calculations

### Module Responsibilities

- **core/**: Contains optimization algorithms and data models
  - `optimizer.py`: Algorithm selection and orchestration
  - `guillotine.py`: Guillotine-specific packing logic
  - `models.py`: Panel, SteelSheet, PlacementResult dataclasses
  - `text_parser.py`: Parses various text formats to Panel objects

- **cutting/**: Work instruction generation
  - `instruction.py`: Generates step-by-step cutting instructions
  - `sequence.py`: Optimizes cutting order for efficiency
  - `validator.py`: Validates size constraints and feasibility

- **ui/**: Streamlit interface components
  - `visualizer.py`: 2D visualization of cutting plans
  - `work_instruction_ui.py`: Displays cutting instructions
  - `components.py`: Reusable UI components

### Algorithm Implementation Priority

1. **First Fit Decreasing (FFD)**: Fast, baseline implementation
2. **Best Fit Decreasing (BFD)**: Better efficiency, moderate speed
3. **Bottom-Left-Fill (BLF)**: Optimized for guillotine constraints

### Data Flow

1. **Input**: Text data or manual panel entry → Parser → Panel objects
2. **Processing**: Panels → Optimizer (with material grouping) → Placement
3. **Output**: PlacementResult → Visualization + Work Instructions + Reports

### State Management

Use Streamlit's session state for:
- Current panel list
- Optimization results
- User preferences
- Cutting plan history

### Key Implementation Considerations

- **Material Grouping**: Always process panels by material type first
- **Rotation Logic**: Check `allow_rotation` flag before rotating panels
- **Efficiency Calculation**: `(used_area / total_sheet_area) * 100`
- **Work Instructions**: Must include safety notes and quality checkpoints

## Performance Targets

- Small batches (≤20 panels): < 1 second
- Medium batches (≤50 panels): < 5 seconds
- Large batches (≤100 panels): < 30 seconds

## Text Data Parser Formats

The system should support multiple input formats:
- CSV: `id,width,height,quantity,material,thickness`
- Tab-delimited: Copy-paste from Excel
- JSON: Structured panel data
- Custom text: Flexible parsing with configurable delimiters