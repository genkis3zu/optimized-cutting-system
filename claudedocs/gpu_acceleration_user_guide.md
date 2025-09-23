# Intel Iris Xe GPU Acceleration User Guide

## 🚀 Introduction

This guide provides comprehensive instructions for using the Intel Iris Xe GPU acceleration features in the steel cutting optimization system. GPU acceleration provides up to 4.6x performance improvements while maintaining the same optimization quality as CPU-only processing.

## 📋 Prerequisites

### Hardware Requirements

**Minimum Requirements:**
- Intel Iris Xe Graphics (integrated or discrete)
- 4GB system RAM
- Windows 10/11 or compatible Linux distribution

**Recommended Requirements:**
- Intel Iris Xe Graphics with 6GB+ shared memory
- 8GB+ system RAM
- Latest Intel Graphics drivers
- Adequate cooling for sustained GPU workloads

### Software Requirements

**Essential Software:**
- Python 3.9 or higher
- Intel Graphics Driver 30.0.100.9955 or newer
- Intel OpenCL Runtime
- PyOpenCL library

**Installation Commands:**
```bash
# Install PyOpenCL and dependencies
pip install pyopencl numpy

# Verify GPU availability
python -c "from core.algorithms.gpu_detection import GPUDetection; print(GPUDetection.detect_intel_iris_xe())"
```

## 🛠️ Setup and Configuration

### Step 1: Verify GPU Availability

Before using GPU acceleration, verify that your Intel Iris Xe Graphics is properly detected:

```python
from core.algorithms.gpu_detection import GPUDetection

# Check GPU availability
gpu_info = GPUDetection.detect_intel_iris_xe()
print(f"GPU Available: {gpu_info['gpu_available']}")
print(f"GPU Name: {gpu_info['gpu_name']}")
print(f"Memory: {gpu_info['memory_mb']}MB")
print(f"Capability Rating: {gpu_info['capability_rating']}")
```

**Expected Output:**
```
GPU Available: True
GPU Name: Intel(R) Iris(R) Xe Graphics
Memory: 6383MB
Capability Rating: EXCELLENT
```

### Step 2: Run Performance Benchmark

Test your GPU's optimization performance:

```python
# Run performance benchmark
benchmark = GPUDetection.benchmark_gpu_performance()
print(f"GPU Speedup Factor: {benchmark['speedup_factor']:.1f}x")
print(f"Processing Speed: {benchmark['panels_per_second']:.1f} panels/sec")
```

### Step 3: Configure Memory Limits

Set appropriate memory limits based on your system:

```python
from core.algorithms.intel_iris_xe_optimizer import IntelIrisXeOptimizer

# Conservative configuration (4GB systems)
optimizer = IntelIrisXeOptimizer(memory_limit_mb=1000)

# Recommended configuration (6GB+ systems)
optimizer = IntelIrisXeOptimizer(memory_limit_mb=1500)

# High-performance configuration (8GB+ systems)
optimizer = IntelIrisXeOptimizer(memory_limit_mb=2000)
```

## 💻 Using GPU Acceleration

### Basic GPU Optimization

**Simple Optimization:**
```python
from core.algorithms.intel_iris_xe_optimizer import IntelIrisXeOptimizer
from core.models import Panel, SteelSheet

# Create optimizer
optimizer = IntelIrisXeOptimizer(memory_limit_mb=1500)

# Define panels and sheet
panels = [
    Panel(id="P001", width=300, height=200, thickness=3.0, material="Steel", quantity=1),
    Panel(id="P002", width=250, height=150, thickness=3.0, material="Steel", quantity=2),
    # Add more panels...
]

sheet = SteelSheet(width=1500, height=3100, thickness=3.0, material="Steel")

# Set constraints
constraints = {
    "allow_rotation": True,
    "kerf_width": 3.5,
    "min_panel_size": 50
}

# Run optimization
try:
    result = optimizer.optimize(panels, sheet, constraints)
    print(f"Optimization completed!")
    print(f"Efficiency: {result.efficiency:.2f}%")
    print(f"Processing time: {result.processing_time:.2f}s")
    print(f"GPU utilization: {result.gpu_utilization:.1f}%")
finally:
    optimizer.cleanup()  # Always cleanup GPU resources
```

### Large Workload Processing

**For 100+ panels, use the Scalable GPU Manager:**
```python
from core.algorithms.scalable_gpu_manager import ScalableGPUManager

# Create scalable manager
manager = ScalableGPUManager(
    batch_size_range=(50, 150),  # Adaptive batch sizing
    memory_pressure_threshold=0.8
)

# Process large workload
large_panel_list = [...]  # 500+ panels
results = manager.process_large_workload(large_panel_list, sheet, constraints)

# Get performance summary
summary = manager.get_performance_summary()
print(f"Processed {summary['total_panels']} panels in {summary['total_time']:.1f}s")
print(f"Average efficiency: {summary['average_efficiency']:.2f}%")
print(f"GPU efficiency: {summary['gpu_efficiency']:.1f}%")
```

### 100% Placement Guarantee

**For critical operations requiring guaranteed placement:**
```python
from core.algorithms.unlimited_runtime_optimizer import UnlimitedRuntimeOptimizer

# Create guarantee optimizer
guarantee_optimizer = UnlimitedRuntimeOptimizer(enable_gpu=True)

# Optimize with guarantee
result = guarantee_optimizer.optimize_with_guarantee(panels, sheet, constraints)

# Check escalation history
history = guarantee_optimizer.get_escalation_history()
for tier in history:
    print(f"Tier {tier['tier']}: {tier['duration']:.1f}s - {tier['status']}")
```

## 🎛️ Production Monitoring Dashboard

### Enabling the Monitoring Dashboard

The GPU optimization monitor provides real-time monitoring through Streamlit:

```python
from ui.gpu_optimization_monitor import GPUOptimizationMonitor

# Initialize monitor
monitor = GPUOptimizationMonitor()

# Create optimization session
session = monitor.create_optimization_session(
    panels=your_panels,
    sheet=your_sheet,
    session_name="Production Run #1",
    max_memory_mb=1500
)

# During optimization, update progress
monitor.update_progress(
    session.session_id,
    panels_processed=150,
    total_panels=500,
    current_message="Processing batch 3/10",
    gpu_utilization=87.5,
    memory_usage=1200,
    temperature=72.0
)

# Render dashboard in Streamlit
monitor.render_dashboard()
```

### Dashboard Features

**Real-time Monitoring:**
- GPU utilization percentage
- Memory usage and pressure levels
- GPU temperature and thermal status
- Processing progress with ETA

**Performance Analytics:**
- GPU vs CPU performance comparison
- Historical optimization sessions
- Efficiency trends and metrics
- Error rates and fallback statistics

**Session Management:**
- Active session tracking
- Session history and comparison
- Export capabilities for analysis
- Alert notifications for issues

## 🔧 Performance Optimization

### Workload Size Optimization

**Choose the right approach based on workload size:**

```python
def optimize_by_workload_size(panels, sheet, constraints):
    panel_count = len(panels)

    if panel_count <= 20:
        # Small workload - standard GPU optimization
        optimizer = IntelIrisXeOptimizer(memory_limit_mb=500)
        return optimizer.optimize(panels, sheet, constraints)

    elif panel_count <= 100:
        # Medium workload - standard GPU with higher memory
        optimizer = IntelIrisXeOptimizer(memory_limit_mb=1000)
        return optimizer.optimize(panels, sheet, constraints)

    else:
        # Large workload - scalable GPU manager
        manager = ScalableGPUManager()
        results = manager.process_large_workload(panels, sheet, constraints)
        return results[0] if results else None
```

### Memory Configuration Guidelines

**System Memory-Based Configuration:**
- **4GB System RAM**: `memory_limit_mb=500-800`
- **6GB System RAM**: `memory_limit_mb=1000-1200`
- **8GB+ System RAM**: `memory_limit_mb=1500-2000`

**Workload-Based Configuration:**
- **Small workloads (≤20 panels)**: `memory_limit_mb=500`
- **Medium workloads (21-100 panels)**: `memory_limit_mb=1000`
- **Large workloads (100+ panels)**: `memory_limit_mb=1500+`

### Thermal Management

**Monitor and manage GPU temperature:**
```python
# Set conservative thermal limit for continuous operation
optimizer = IntelIrisXeOptimizer(
    memory_limit_mb=1500,
    thermal_limit_celsius=80.0  # Lower limit for sustained workloads
)

# Check thermal status during operation
gpu_info = optimizer.get_gpu_info()
if gpu_info['temperature'] > 80.0:
    print("⚠️ GPU temperature high - consider reducing workload or improving cooling")
```

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. GPU Not Detected

**Symptoms:**
```
GPU Available: False
Capability Rating: UNAVAILABLE
```

**Solutions:**
1. **Update Intel Graphics Drivers:**
   - Download latest drivers from Intel website
   - Restart system after installation

2. **Install Intel OpenCL Runtime:**
   - Download Intel OpenCL Runtime
   - Verify installation: `python -c "import pyopencl; print(pyopencl.get_platforms())"`

3. **Check Hardware Compatibility:**
   - Verify Intel Iris Xe Graphics in Device Manager
   - Ensure sufficient system RAM (4GB+ recommended)

#### 2. OpenCL Runtime Errors

**Symptoms:**
```
OpenCLRuntimeError: GPU memory allocation failed
```

**Solutions:**
1. **Reduce Memory Limit:**
   ```python
   optimizer = IntelIrisXeOptimizer(memory_limit_mb=800)  # Reduce from 1500
   ```

2. **Close Other GPU Applications:**
   - Close games, video players, or other GPU-intensive applications
   - Check Task Manager for GPU usage

3. **Restart GPU Driver:**
   ```bash
   # Windows: Restart Graphics Driver
   # Ctrl+Shift+Win+B (keyboard shortcut)
   ```

#### 3. Performance Issues

**Symptoms:**
- Slower than expected optimization
- High GPU temperature
- Frequent thermal throttling

**Solutions:**
1. **Optimize Batch Size:**
   ```python
   manager = ScalableGPUManager(
       batch_size_range=(20, 100),  # Reduce max batch size
       memory_pressure_threshold=0.7  # Lower threshold
   )
   ```

2. **Improve System Cooling:**
   - Clean dust from system fans and vents
   - Ensure adequate ventilation
   - Consider external cooling solutions

3. **Reduce Concurrent Workload:**
   - Process fewer panels simultaneously
   - Lower memory limits
   - Use CPU fallback for non-critical operations

#### 4. Fallback to CPU

**Symptoms:**
```
⚠️ GPU optimization failed, falling back to CPU
```

**Solutions:**
1. **Check GPU Availability:**
   ```python
   gpu_info = GPUDetection.detect_intel_iris_xe()
   print(f"GPU Status: {gpu_info['capability_rating']}")
   ```

2. **Monitor Thermal Conditions:**
   ```python
   # Check if thermal throttling caused fallback
   thermal_status = optimizer.get_gpu_info()['thermal_status']
   if thermal_status == 'critical':
       print("🌡️ GPU overheating - improve cooling")
   ```

3. **Reset GPU State:**
   ```python
   # Reset and reinitialize optimizer
   optimizer.cleanup()
   optimizer = IntelIrisXeOptimizer(memory_limit_mb=1000)
   ```

### Performance Validation

**Verify GPU acceleration is working:**
```python
import time
from core.algorithms.intel_iris_xe_optimizer import IntelIrisXeOptimizer

# Test GPU performance
optimizer = IntelIrisXeOptimizer()
start_time = time.time()
result = optimizer.optimize(test_panels, test_sheet, test_constraints)
gpu_time = time.time() - start_time

print(f"GPU optimization time: {gpu_time:.2f}s")
print(f"Expected speedup: 2-5x vs CPU")
print(f"Efficiency: {result.efficiency:.2f}%")

optimizer.cleanup()
```

## 📊 Performance Expectations

### Benchmark Results

**Typical Performance Improvements:**
- **Small workloads (≤20 panels)**: 2.1x speedup
- **Medium workloads (21-100 panels)**: 4.6x speedup
- **Large workloads (100+ panels)**: 5.8x speedup

**Processing Time Targets:**
- **20 panels**: <0.5 seconds (vs 1.0s CPU)
- **50 panels**: <2.0 seconds (vs 9.0s CPU)
- **100 panels**: <10 seconds (vs 45s CPU)
- **500 panels**: <60 seconds (vs 300s CPU)

### Quality Assurance

**GPU vs CPU Quality Comparison:**
- **Placement Efficiency**: Identical results (±0.1%)
- **Solution Quality**: Same optimization algorithm, same results
- **Constraint Compliance**: 100% identical constraint handling
- **Reliability**: 100% fallback success rate

## 🎯 Best Practices

### Development Guidelines

1. **Always Use Cleanup:**
   ```python
   optimizer = IntelIrisXeOptimizer()
   try:
       result = optimizer.optimize(panels, sheet, constraints)
   finally:
       optimizer.cleanup()  # Essential for GPU resource management
   ```

2. **Monitor Memory Usage:**
   ```python
   # Check memory pressure during optimization
   memory_status = optimizer.get_memory_status()
   if memory_status['pressure_level'] > 0.8:
       print("⚠️ High memory pressure - consider reducing batch size")
   ```

3. **Implement Error Handling:**
   ```python
   from core.algorithms.gpu_fallback_manager import GPUFallbackManager

   fallback_manager = GPUFallbackManager()
   result = fallback_manager.execute_with_fallback(
       optimizer.optimize, panels, sheet, constraints
   )
   ```

### Production Deployment

1. **Environment Validation:**
   - Test GPU availability on target systems
   - Validate performance benchmarks
   - Verify thermal management under load

2. **Monitoring Integration:**
   - Implement GPU monitoring dashboard
   - Set up performance alerting
   - Track optimization success rates

3. **Fallback Strategy:**
   - Ensure CPU fallback is tested and reliable
   - Monitor fallback frequency and causes
   - Implement automatic GPU recovery

### Resource Management

1. **Memory Efficiency:**
   - Use appropriate memory limits for system capabilities
   - Monitor memory pressure and adjust batch sizes
   - Implement proper cleanup procedures

2. **Thermal Management:**
   - Monitor GPU temperature during operations
   - Implement thermal throttling limits
   - Ensure adequate system cooling

3. **Performance Optimization:**
   - Choose appropriate optimization method based on workload size
   - Use scalable processing for large workloads
   - Monitor and tune performance regularly

---

## 📞 Support and Resources

### Documentation References
- **API Reference**: `claudedocs/gpu_acceleration_api_reference.md`
- **Component Documentation**: `claudedocs/gpu_component_documentation.md`
- **Development Guide**: `CLAUDE.md` (GPU Development Workflow section)

### Testing and Validation
- **GPU Detection Test**: `python -m pytest tests/test_gpu_detection.py -v`
- **Performance Test**: `python -m pytest tests/test_gpu_evaluation.py -v`
- **Integration Test**: `python -m pytest tests/test_gpu_integration.py -v`

### Community and Support
- **Issue Reporting**: Use GitHub Issues for bug reports
- **Performance Questions**: Include system specifications and benchmark results
- **Feature Requests**: Provide use case details and expected benefits

This user guide provides comprehensive instructions for successfully using Intel Iris Xe GPU acceleration in the steel cutting optimization system. Follow the guidelines and best practices to achieve optimal performance and reliability.