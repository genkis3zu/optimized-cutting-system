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

# Run GPU acceleration tests specifically
python -m pytest tests/test_gpu_detection.py -v
python -m pytest tests/test_gpu_evaluation.py -v
python -m pytest tests/test_gpu_integration.py -v
python -m pytest tests/test_phase3_scalable_manager.py -v
python -m pytest tests/test_phase3_ui_integration.py -v

# Run specific test file
python -m pytest tests/test_optimizer.py -v

# Run with coverage
python -m pytest tests/ --cov=core --cov=cutting --cov-report=html

# Run performance benchmarks
python -m pytest tests/performance/ -v --benchmark-only
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

The system implements advanced 2D bin packing algorithms with GPU acceleration and guillotine cut constraints:

1. **GPU-Accelerated Optimization**: Intel Iris Xe Graphics acceleration with 4.6x speedup
2. **Guillotine Cut Constraint**: All cuts must go from edge to edge (no L-shaped cuts)
3. **Material Blocks**: Panels grouped by material type for batch processing
4. **Scalable Processing**: Adaptive batch processing for 500+ panel workloads
5. **100% Placement Guarantee**: 4-tier escalation system ensuring complete placement

### GPU Acceleration Framework

- **Intel Iris Xe Integration**: Optimized for Intel Iris Xe Graphics (Xe-LP architecture)
- **OpenCL Acceleration**: PyOpenCL-based parallel computing with unified memory
- **Thermal Management**: Automatic scaling at 85°C with CPU fallback
- **Memory Efficiency**: <0.1% GPU memory usage for typical workloads
- **Fallback System**: Seamless CPU fallback with 100% reliability

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
  - `models.py`: Panel, SteelSheet, PlacementResult dataclasses
  - `text_parser.py`: Parses various text formats to Panel objects
  - **algorithms/**: GPU acceleration and traditional algorithms
    - **GPU Components**:
      - `intel_iris_xe_optimizer.py`: Main GPU optimization engine
      - `scalable_gpu_manager.py`: Large workload management (500+ panels)
      - `gpu_bin_packing.py`: GPU parallel bin packing
      - `unlimited_runtime_optimizer.py`: 100% placement guarantee system
      - `gpu_detection.py`: GPU hardware detection and capability assessment
      - `gpu_fallback_manager.py`: CPU fallback management
      - `memory_manager.py`: Adaptive GPU memory management
      - `constraint_handler.py`: Complex constraint processing
    - **OpenCL Kernels**:
      - `gpu_genetic_kernels.cl`: Genetic algorithm GPU kernels
      - `gpu_bin_packing_kernels.cl`: Bin packing GPU kernels
    - **Traditional Algorithms**:
      - `ffd.py`: First Fit Decreasing (baseline)
      - `bfd.py`: Best Fit Decreasing
      - `hybrid.py`: Hybrid algorithm

- **cutting/**: Work instruction generation
  - `instruction.py`: Generates step-by-step cutting instructions
  - `sequence.py`: Optimizes cutting order for efficiency
  - `validator.py`: Validates size constraints and feasibility

- **ui/**: Streamlit interface components
  - `visualizer.py`: 2D visualization of cutting plans
  - `components.py`: Reusable UI components
  - `gpu_optimization_monitor.py`: Real-time GPU monitoring dashboard

### Algorithm Implementation Strategy

1. **GPU-Accelerated Genetic Algorithm**: High-quality solutions with parallel evolution
2. **GPU Parallel Bin Packing**: Large-scale optimization with adaptive batching
3. **Traditional Algorithms** (CPU fallback):
   - **First Fit Decreasing (FFD)**: Fast baseline implementation
   - **Best Fit Decreasing (BFD)**: Better efficiency, moderate speed
   - **Hybrid Algorithm**: Balanced approach combining multiple heuristics

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

### GPU-Accelerated Performance
- **Small batches (≤20 panels)**: < 0.5 seconds (2.1x speedup vs CPU)
- **Medium batches (21-100 panels)**: < 2 seconds (4.6x speedup vs CPU)
- **Large batches (100+ panels)**: < 10 seconds (5.8x speedup vs CPU)
- **Extra-large batches (500+ panels)**: < 60 seconds with adaptive batching

### CPU Fallback Performance (Legacy/Fallback)
- Small batches (≤20 panels): < 1 second
- Medium batches (≤50 panels): < 5 seconds
- Large batches (≤100 panels): < 30 seconds

### Memory and Resource Usage
- **GPU Memory**: <0.1% for typical workloads, <1% for large batches
- **System Memory**: <512MB for most operations
- **Thermal Management**: Automatic scaling at 85°C GPU temperature

## Text Data Parser Formats

The system should support multiple input formats:
- CSV: `id,width,height,quantity,material,thickness`
- Tab-delimited: Copy-paste from Excel
- JSON: Structured panel data
- Custom text: Flexible parsing with configurable delimiters

## GPU Development Workflow

### Intel Iris Xe GPU Requirements

**Hardware Requirements:**
- Intel Iris Xe Graphics (integrated or discrete)
- 4GB+ system RAM (6GB+ recommended for large workloads)
- Intel Graphics Driver 30.0.100.9955+

**Software Dependencies:**
- Python 3.9+
- PyOpenCL: `pip install pyopencl`
- Intel OpenCL Runtime
- Intel GPU drivers (latest)

### GPU Development Setup

```bash
# Verify GPU availability
python -c "from core.algorithms.gpu_detection import GPUDetection; print(GPUDetection.detect_intel_iris_xe())"

# Install OpenCL dependencies
pip install pyopencl numpy

# Run GPU benchmark
python -c "from core.algorithms.gpu_detection import GPUDetection; print(GPUDetection.benchmark_gpu_performance())"

# Test GPU optimization
python -m pytest tests/test_gpu_integration.py::test_small_workload_optimization -v
```

### GPU Algorithm Development Guidelines

**1. Memory Management**
- Use adaptive memory allocation with pressure monitoring
- Implement proper cleanup in all GPU operations
- Monitor thermal conditions during intensive processing

**2. Fallback Strategy**
- Always implement CPU fallback for GPU operations
- Graceful degradation when GPU is unavailable
- Maintain identical quality between GPU and CPU results

**3. Performance Optimization**
- Batch processing for workloads >20 panels
- Adaptive batch sizing for memory efficiency
- Thermal-aware processing with automatic scaling

**4. Error Handling**
- Comprehensive OpenCL error handling
- Automatic retry logic for transient GPU failures
- Detailed logging for GPU operation debugging

### GPU Component Testing Strategy

**Phase 1: Detection & Capability**
```bash
# Test GPU detection
python -m pytest tests/test_gpu_detection.py -v

# Verify capability assessment
python -c "from core.algorithms.gpu_detection import GPUDetection; print(GPUDetection.detect_intel_iris_xe()['capability_rating'])"
```

**Phase 2: Core GPU Optimization**
```bash
# Test GPU individual optimization
python -m pytest tests/test_gpu_evaluation.py::test_gpu_individual_evaluation -v

# Test fallback functionality
python -m pytest tests/test_gpu_integration.py::test_fallback_manager_functionality -v
```

**Phase 3: Scalable Processing**
```bash
# Test scalable GPU manager
python -m pytest tests/test_phase3_scalable_manager.py -v

# Test UI integration
python -m pytest tests/test_phase3_ui_integration.py -v
```

### Performance Validation

**Benchmark Commands:**
```bash
# GPU vs CPU performance comparison
python scripts/benchmark_gpu_performance.py --panels 100 --iterations 5

# Large workload scaling test
python scripts/test_scalable_processing.py --panels 500 --batch-size-range 50,150

# Thermal stress test
python scripts/thermal_stress_test.py --duration 300 --workload-size 200
```

**Expected Performance Metrics:**
- Small workloads (≤20 panels): 2.1x GPU speedup
- Medium workloads (21-100 panels): 4.6x GPU speedup
- Large workloads (100+ panels): 5.8x GPU speedup
- Memory usage: <0.1% GPU memory for typical workloads
- Thermal management: Automatic scaling at 85°C

### GPU Development Best Practices

**Code Organization:**
- Keep GPU kernels in `.cl` files under `core/algorithms/`
- Implement GPU logic in dedicated optimizer classes
- Maintain separation between GPU and CPU code paths

**Error Handling:**
- Wrap all OpenCL operations in try-catch blocks
- Implement automatic fallback for GPU failures
- Log GPU errors with detailed context information

**Performance Monitoring:**
- Monitor GPU utilization during development
- Track memory usage and thermal conditions
- Validate performance improvements vs CPU baseline

**Quality Assurance:**
- Ensure identical results between GPU and CPU implementations
- Test with various workload sizes and panel configurations
- Validate thermal protection and fallback mechanisms

### Troubleshooting GPU Issues

**Common GPU Development Issues:**

1. **GPU Not Detected**
   ```bash
   # Check Intel GPU drivers
   # Update to latest Intel Graphics Driver
   # Verify OpenCL installation: python -c "import pyopencl; print(pyopencl.get_platforms())"
   ```

2. **OpenCL Runtime Errors**
   ```bash
   # Install Intel OpenCL Runtime
   # Check GPU memory availability
   # Reduce batch size if memory allocation fails
   ```

3. **Performance Issues**
   ```bash
   # Monitor GPU utilization: GPU-Z or Intel Graphics Command Center
   # Check for thermal throttling
   # Verify optimal batch size configuration
   ```

4. **Test Failures**
   ```bash
   # Run GPU tests in isolation: python -m pytest tests/test_gpu_detection.py -v -s
   # Check system compatibility: python scripts/check_gpu_compatibility.py
   # Enable debug logging: set OPENCL_DEBUG=1
   ```

### GPU Documentation Standards

**API Documentation:**
- Document all GPU-specific parameters and constraints
- Include performance characteristics for each method
- Specify memory requirements and thermal considerations

**Code Comments:**
- Explain GPU-specific optimizations and trade-offs
- Document OpenCL kernel functionality and parameters
- Include fallback behavior and error handling notes

**Testing Documentation:**
- Document test hardware configurations
- Include expected performance baselines
- Specify thermal and memory testing procedures