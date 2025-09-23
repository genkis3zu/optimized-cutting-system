"""
GPU Detection and Capability Verification System

Provides robust detection of Intel Iris Xe Graphics and validates
OpenCL capabilities for steel cutting optimization acceleration.

Features:
- Intel GPU detection and validation
- OpenCL capability assessment
- Performance benchmarking
- Thermal monitoring support
- Fallback strategy determination
"""

import logging
import platform
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
    cl = None

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

logger = logging.getLogger(__name__)

class GPUCapabilityLevel(Enum):
    """GPU capability levels for optimization strategy selection"""
    NONE = "none"           # No GPU or OpenCL
    BASIC = "basic"         # Basic GPU, limited benefits
    GOOD = "good"           # Good GPU performance
    EXCELLENT = "excellent" # Optimal GPU performance

@dataclass
class GPUDeviceInfo:
    """Information about detected GPU device"""
    name: str
    vendor: str
    driver_version: str
    opencl_version: str
    compute_units: int
    max_work_group_size: int
    global_memory_mb: int
    local_memory_kb: int
    is_intel_integrated: bool
    capability_level: GPUCapabilityLevel

@dataclass
class GPUBenchmarkResult:
    """Results from GPU performance benchmarking"""
    memory_bandwidth_gbps: float
    compute_performance_gflops: float
    kernel_compile_time_ms: float
    context_creation_time_ms: float
    data_transfer_speed_gbps: float
    thermal_baseline_temp: float

class GPUDetectionSystem:
    """
    Comprehensive GPU detection and capability assessment system.

    Provides robust detection of Intel Iris Xe Graphics and evaluates
    suitability for genetic algorithm acceleration.
    """

    def __init__(self):
        self.opencl_available = OPENCL_AVAILABLE
        self.detected_devices: List[GPUDeviceInfo] = []
        self.intel_device: Optional[GPUDeviceInfo] = None
        self.benchmark_results: Optional[GPUBenchmarkResult] = None

    def detect_intel_iris_xe(self) -> bool:
        """
        Detect and validate Intel Iris Xe Graphics for optimization use.

        Returns:
            True if suitable Intel GPU found, False otherwise
        """
        if not self.opencl_available:
            logger.info("PyOpenCL not available - GPU acceleration disabled")
            return False

        try:
            logger.info("Scanning for Intel Iris Xe Graphics...")

            # Get all OpenCL platforms
            platforms = cl.get_platforms()

            for platform in platforms:
                logger.debug(f"Checking platform: {platform.name}")

                if "Intel" not in platform.name:
                    continue

                # Get GPU devices from Intel platform
                try:
                    devices = platform.get_devices(cl.device_type.GPU)
                except cl.RuntimeError:
                    # No GPU devices on this platform
                    continue

                for device in devices:
                    device_info = self._analyze_device(device, platform)
                    self.detected_devices.append(device_info)

                    # Check if this is suitable Intel Iris Xe
                    if self._is_suitable_intel_gpu(device_info):
                        self.intel_device = device_info
                        logger.info(f"✅ Intel Iris Xe Graphics detected: {device_info.name}")
                        logger.info(f"   Compute Units: {device_info.compute_units}")
                        logger.info(f"   Memory: {device_info.global_memory_mb} MB")
                        logger.info(f"   Capability: {device_info.capability_level.value}")
                        return True

            logger.warning("❌ No suitable Intel Iris Xe Graphics found")
            return False

        except Exception as e:
            logger.error(f"GPU detection failed: {e}")
            return False

    def _analyze_device(self, device: 'cl.Device', platform: 'cl.Platform') -> GPUDeviceInfo:
        """Analyze OpenCL device capabilities"""
        try:
            # Get device information
            name = device.name.strip()
            vendor = device.vendor.strip()
            driver_version = device.driver_version.strip()
            opencl_version = device.opencl_c_version.strip()

            compute_units = device.max_compute_units
            max_work_group_size = device.max_work_group_size
            global_memory_mb = device.global_mem_size // (1024 * 1024)
            local_memory_kb = device.local_mem_size // 1024

            # Determine if this is Intel integrated graphics
            is_intel_integrated = (
                "Intel" in vendor and
                ("Iris" in name or "Xe" in name or "UHD" in name)
            )

            # Assess capability level
            capability_level = self._assess_capability_level(
                name, compute_units, global_memory_mb, max_work_group_size
            )

            return GPUDeviceInfo(
                name=name,
                vendor=vendor,
                driver_version=driver_version,
                opencl_version=opencl_version,
                compute_units=compute_units,
                max_work_group_size=max_work_group_size,
                global_memory_mb=global_memory_mb,
                local_memory_kb=local_memory_kb,
                is_intel_integrated=is_intel_integrated,
                capability_level=capability_level
            )

        except Exception as e:
            logger.warning(f"Failed to analyze device {device.name}: {e}")
            # Return minimal device info
            return GPUDeviceInfo(
                name=getattr(device, 'name', 'Unknown'),
                vendor=getattr(device, 'vendor', 'Unknown'),
                driver_version="Unknown",
                opencl_version="Unknown",
                compute_units=0,
                max_work_group_size=1,
                global_memory_mb=0,
                local_memory_kb=0,
                is_intel_integrated=False,
                capability_level=GPUCapabilityLevel.NONE
            )

    def _assess_capability_level(
        self,
        name: str,
        compute_units: int,
        memory_mb: int,
        max_work_group_size: int
    ) -> GPUCapabilityLevel:
        """Assess GPU capability level for optimization tasks"""

        # Check for Intel Iris Xe (excellent for genetic algorithms)
        if "Iris" in name and "Xe" in name:
            if compute_units >= 80 and memory_mb >= 2048:
                return GPUCapabilityLevel.EXCELLENT
            elif compute_units >= 48 and memory_mb >= 1024:
                return GPUCapabilityLevel.GOOD
            else:
                return GPUCapabilityLevel.BASIC

        # Check for other Intel integrated graphics
        elif "Intel" in name:
            if compute_units >= 32 and memory_mb >= 1024:
                return GPUCapabilityLevel.GOOD
            elif compute_units >= 16 and memory_mb >= 512:
                return GPUCapabilityLevel.BASIC
            else:
                return GPUCapabilityLevel.NONE

        # Other vendors (NVIDIA, AMD)
        elif compute_units >= 100 and memory_mb >= 4096:
            return GPUCapabilityLevel.EXCELLENT
        elif compute_units >= 50 and memory_mb >= 2048:
            return GPUCapabilityLevel.GOOD
        elif compute_units >= 20 and memory_mb >= 1024:
            return GPUCapabilityLevel.BASIC
        else:
            return GPUCapabilityLevel.NONE

    def _is_suitable_intel_gpu(self, device_info: GPUDeviceInfo) -> bool:
        """Check if device is suitable for genetic algorithm acceleration"""
        return (
            device_info.is_intel_integrated and
            device_info.capability_level in [GPUCapabilityLevel.GOOD, GPUCapabilityLevel.EXCELLENT] and
            device_info.compute_units >= 32 and
            device_info.global_memory_mb >= 1024 and
            device_info.max_work_group_size >= 32
        )

    def benchmark_gpu_performance(self) -> Optional[GPUBenchmarkResult]:
        """
        Benchmark GPU performance for genetic algorithm workloads.

        Returns:
            Benchmark results or None if GPU not available
        """
        if not self.intel_device:
            logger.warning("No Intel GPU available for benchmarking")
            return None

        try:
            logger.info("🔄 Running GPU performance benchmark...")

            # Create OpenCL context for benchmarking
            platforms = cl.get_platforms()
            intel_device = None

            for platform in platforms:
                if "Intel" in platform.name:
                    devices = platform.get_devices(cl.device_type.GPU)
                    for device in devices:
                        if device.name.strip() == self.intel_device.name:
                            intel_device = device
                            break
                    if intel_device:
                        break

            if not intel_device:
                logger.error("Could not find Intel device for benchmarking")
                return None

            # Context creation benchmark
            context_start = time.time()
            context = cl.Context([intel_device])
            queue = cl.CommandQueue(context)
            context_time_ms = (time.time() - context_start) * 1000

            # Simple kernel compilation benchmark
            compile_start = time.time()
            simple_kernel = """
            __kernel void simple_test(__global float* data) {
                int id = get_global_id(0);
                data[id] = data[id] * 2.0f;
            }
            """
            program = cl.Program(context, simple_kernel).build()
            compile_time_ms = (time.time() - compile_start) * 1000

            # Memory bandwidth test
            test_size = 1024 * 1024  # 1MB test
            import numpy as np
            host_array = np.random.random(test_size).astype(np.float32)

            # Memory transfer benchmark
            transfer_start = time.time()
            gpu_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=test_size * 4)
            cl.enqueue_copy(queue, gpu_buffer, host_array).wait()
            result_array = np.empty_like(host_array)
            cl.enqueue_copy(queue, result_array, gpu_buffer).wait()
            transfer_time = time.time() - transfer_start

            # Calculate transfer speed (GB/s)
            data_transferred_gb = (test_size * 4 * 2) / (1024 * 1024 * 1024)  # Round trip
            transfer_speed_gbps = data_transferred_gb / transfer_time if transfer_time > 0 else 0

            # Compute performance test (simple GFLOPS estimate)
            compute_start = time.time()
            kernel = program.simple_test
            kernel.set_args(gpu_buffer)
            cl.enqueue_nd_range_kernel(queue, kernel, (test_size,), None).wait()
            compute_time = time.time() - compute_start

            # Rough GFLOPS calculation (one multiply per element)
            gflops = (test_size / compute_time) / 1e9 if compute_time > 0 else 0

            # Thermal baseline
            thermal_temp = self._get_thermal_baseline()

            # Memory bandwidth estimate (rough)
            memory_bandwidth_gbps = min(51.2, transfer_speed_gbps * 10)  # Intel Iris Xe theoretical max

            self.benchmark_results = GPUBenchmarkResult(
                memory_bandwidth_gbps=memory_bandwidth_gbps,
                compute_performance_gflops=gflops,
                kernel_compile_time_ms=compile_time_ms,
                context_creation_time_ms=context_time_ms,
                data_transfer_speed_gbps=transfer_speed_gbps,
                thermal_baseline_temp=thermal_temp
            )

            logger.info("✅ GPU benchmark completed:")
            logger.info(f"   Context creation: {context_time_ms:.1f}ms")
            logger.info(f"   Kernel compile: {compile_time_ms:.1f}ms")
            logger.info(f"   Transfer speed: {transfer_speed_gbps:.2f} GB/s")
            logger.info(f"   Compute performance: {gflops:.2f} GFLOPS")
            logger.info(f"   Thermal baseline: {thermal_temp:.1f}°C")

            return self.benchmark_results

        except Exception as e:
            logger.error(f"GPU benchmark failed: {e}")
            return None

    def _get_thermal_baseline(self) -> float:
        """Get baseline thermal temperature"""
        if not PSUTIL_AVAILABLE:
            return 45.0  # Default assumption

        try:
            temps = psutil.sensors_temperatures()

            # Try to get CPU temperature (shared with integrated GPU)
            if 'coretemp' in temps:
                return max(sensor.current for sensor in temps['coretemp'])
            elif 'cpu_thermal' in temps:
                return temps['cpu_thermal'][0].current
            else:
                return 45.0  # Default if no sensors found

        except Exception:
            return 45.0  # Default on error

    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get optimization recommendations based on detected hardware"""
        if not self.intel_device:
            return {
                'use_gpu': False,
                'reason': 'No suitable Intel GPU detected',
                'recommended_strategy': 'CPU-only optimization',
                'max_population_size': 100,
                'max_generations': 100
            }

        capability = self.intel_device.capability_level

        if capability == GPUCapabilityLevel.EXCELLENT:
            return {
                'use_gpu': True,
                'reason': 'Excellent GPU capability detected',
                'recommended_strategy': 'Full GPU acceleration',
                'max_population_size': 200,
                'max_generations': 200,
                'thermal_monitoring': True,
                'adaptive_workload': True
            }
        elif capability == GPUCapabilityLevel.GOOD:
            return {
                'use_gpu': True,
                'reason': 'Good GPU capability detected',
                'recommended_strategy': 'Selective GPU acceleration',
                'max_population_size': 100,
                'max_generations': 100,
                'thermal_monitoring': True,
                'adaptive_workload': True
            }
        else:
            return {
                'use_gpu': False,
                'reason': 'GPU capability too limited',
                'recommended_strategy': 'CPU-only optimization',
                'max_population_size': 50,
                'max_generations': 50
            }

    def get_detection_summary(self) -> Dict[str, Any]:
        """Get comprehensive detection and capability summary"""
        return {
            'system_info': {
                'platform': platform.system(),
                'python_version': platform.python_version(),
                'opencl_available': self.opencl_available,
                'psutil_available': PSUTIL_AVAILABLE
            },
            'detected_devices': [
                {
                    'name': device.name,
                    'vendor': device.vendor,
                    'compute_units': device.compute_units,
                    'memory_mb': device.global_memory_mb,
                    'capability': device.capability_level.value,
                    'is_intel': device.is_intel_integrated
                }
                for device in self.detected_devices
            ],
            'selected_device': {
                'name': self.intel_device.name,
                'capability': self.intel_device.capability_level.value,
                'compute_units': self.intel_device.compute_units,
                'memory_mb': self.intel_device.global_memory_mb
            } if self.intel_device else None,
            'benchmark_results': {
                'context_creation_ms': self.benchmark_results.context_creation_time_ms,
                'kernel_compile_ms': self.benchmark_results.kernel_compile_time_ms,
                'transfer_speed_gbps': self.benchmark_results.data_transfer_speed_gbps,
                'compute_gflops': self.benchmark_results.compute_performance_gflops,
                'thermal_baseline': self.benchmark_results.thermal_baseline_temp
            } if self.benchmark_results else None,
            'recommendations': self.get_optimization_recommendations()
        }


def detect_intel_iris_xe() -> Tuple[bool, Optional[GPUDetectionSystem]]:
    """
    Convenience function for Intel Iris Xe detection.

    Returns:
        Tuple of (success, detection_system)
    """
    detector = GPUDetectionSystem()
    success = detector.detect_intel_iris_xe()
    return success, detector


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("🔍 Intel Iris Xe Graphics Detection System")
    print("=" * 50)

    # Run detection
    success, detector = detect_intel_iris_xe()

    if success:
        print("✅ Intel Iris Xe Graphics detected successfully!")

        # Run benchmark
        benchmark = detector.benchmark_gpu_performance()

        # Get recommendations
        recommendations = detector.get_optimization_recommendations()
        print(f"\n📋 Recommendations:")
        for key, value in recommendations.items():
            print(f"   {key}: {value}")

    else:
        print("❌ Intel Iris Xe Graphics not found")

    # Print summary
    summary = detector.get_detection_summary()
    print(f"\n📊 Detection Summary:")
    print(f"   OpenCL Available: {summary['system_info']['opencl_available']}")
    print(f"   Detected Devices: {len(summary['detected_devices'])}")

    if summary['selected_device']:
        device = summary['selected_device']
        print(f"   Selected: {device['name']} ({device['capability']})")