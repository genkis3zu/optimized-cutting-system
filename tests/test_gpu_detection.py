"""
Test GPU Detection and OpenCL Functionality

Comprehensive testing suite for Intel Iris Xe GPU detection,
OpenCL context creation, and performance validation.
"""

import pytest
import logging
from unittest.mock import patch, MagicMock
from core.algorithms.gpu_detection import (
    GPUDetectionSystem,
    GPUCapabilityLevel,
    GPUDeviceInfo,
    detect_intel_iris_xe,
    OPENCL_AVAILABLE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestGPUDetection:
    """Test suite for GPU detection functionality"""

    def test_opencl_availability(self):
        """Test that OpenCL is available"""
        if OPENCL_AVAILABLE:
            logger.info("✅ OpenCL is available for testing")
        else:
            pytest.skip("OpenCL not available - skipping GPU tests")

    @pytest.mark.skipif(not OPENCL_AVAILABLE, reason="OpenCL not available")
    def test_intel_iris_xe_detection(self):
        """Test Intel Iris Xe detection"""
        success, detector = detect_intel_iris_xe()

        logger.info(f"GPU Detection Result: {success}")

        # Log detected devices
        for i, device in enumerate(detector.detected_devices):
            logger.info(f"Device {i}: {device.name} ({device.vendor})")
            logger.info(f"  Capability: {device.capability_level.value}")
            logger.info(f"  Compute Units: {device.compute_units}")
            logger.info(f"  Memory: {device.global_memory_mb} MB")

        # If detection succeeded, validate the device
        if success:
            assert detector.intel_device is not None
            assert detector.intel_device.is_intel_integrated
            assert detector.intel_device.capability_level in [
                GPUCapabilityLevel.GOOD,
                GPUCapabilityLevel.EXCELLENT
            ]
            logger.info(f"✅ Selected Intel GPU: {detector.intel_device.name}")
        else:
            logger.info("ℹ️ No suitable Intel GPU found (this may be expected)")

    @pytest.mark.skipif(not OPENCL_AVAILABLE, reason="OpenCL not available")
    def test_gpu_benchmarking(self):
        """Test GPU performance benchmarking"""
        success, detector = detect_intel_iris_xe()

        if not success:
            pytest.skip("No Intel GPU available for benchmark testing")

        # Run benchmark
        benchmark_result = detector.benchmark_gpu_performance()

        if benchmark_result:
            logger.info("✅ GPU Benchmark Results:")
            logger.info(f"  Context Creation: {benchmark_result.context_creation_time_ms:.1f}ms")
            logger.info(f"  Kernel Compile: {benchmark_result.kernel_compile_time_ms:.1f}ms")
            logger.info(f"  Transfer Speed: {benchmark_result.data_transfer_speed_gbps:.2f} GB/s")
            logger.info(f"  Compute Performance: {benchmark_result.compute_performance_gflops:.2f} GFLOPS")
            logger.info(f"  Thermal Baseline: {benchmark_result.thermal_baseline_temp:.1f}°C")

            # Validate benchmark results
            assert benchmark_result.context_creation_time_ms > 0
            assert benchmark_result.kernel_compile_time_ms > 0
            assert benchmark_result.data_transfer_speed_gbps >= 0
            assert benchmark_result.compute_performance_gflops >= 0
            assert 20 <= benchmark_result.thermal_baseline_temp <= 100  # Reasonable temp range
        else:
            pytest.fail("Benchmark failed to run")

    @pytest.mark.skipif(not OPENCL_AVAILABLE, reason="OpenCL not available")
    def test_optimization_recommendations(self):
        """Test optimization recommendations"""
        success, detector = detect_intel_iris_xe()

        recommendations = detector.get_optimization_recommendations()

        logger.info("📋 Optimization Recommendations:")
        for key, value in recommendations.items():
            logger.info(f"  {key}: {value}")

        # Validate recommendation structure
        assert 'use_gpu' in recommendations
        assert 'reason' in recommendations
        assert 'recommended_strategy' in recommendations

        if success:
            # If GPU detected, should recommend GPU usage
            assert recommendations['use_gpu'] is True
            assert 'thermal_monitoring' in recommendations
            assert 'adaptive_workload' in recommendations
        else:
            # If no GPU, should recommend CPU-only
            assert recommendations['use_gpu'] is False
            assert recommendations['recommended_strategy'] == 'CPU-only optimization'

    def test_capability_assessment(self):
        """Test GPU capability level assessment"""
        detector = GPUDetectionSystem()

        # Test excellent Intel Iris Xe
        level = detector._assess_capability_level("Intel(R) Iris(R) Xe Graphics", 96, 4096, 256)
        assert level == GPUCapabilityLevel.EXCELLENT

        # Test good Intel Iris Xe
        level = detector._assess_capability_level("Intel(R) Iris(R) Xe Graphics", 80, 2048, 256)
        assert level == GPUCapabilityLevel.EXCELLENT

        # Test basic Intel Iris Xe
        level = detector._assess_capability_level("Intel(R) Iris(R) Xe Graphics", 48, 1024, 256)
        assert level == GPUCapabilityLevel.GOOD

        # Test insufficient capability
        level = detector._assess_capability_level("Intel(R) UHD Graphics", 16, 512, 256)
        assert level == GPUCapabilityLevel.BASIC

    def test_detection_summary(self):
        """Test detection summary generation"""
        success, detector = detect_intel_iris_xe()

        summary = detector.get_detection_summary()

        logger.info("📊 Detection Summary:")
        logger.info(f"  Platform: {summary['system_info']['platform']}")
        logger.info(f"  OpenCL Available: {summary['system_info']['opencl_available']}")
        logger.info(f"  Detected Devices: {len(summary['detected_devices'])}")

        # Validate summary structure
        assert 'system_info' in summary
        assert 'detected_devices' in summary
        assert 'recommendations' in summary

        # System info should be populated
        assert summary['system_info']['opencl_available'] == OPENCL_AVAILABLE

        if success:
            assert summary['selected_device'] is not None
            logger.info(f"  Selected Device: {summary['selected_device']['name']}")

        if summary['benchmark_results']:
            logger.info("  Benchmark Results Available: ✅")
        else:
            logger.info("  Benchmark Results Available: ❌")


class TestGPUDetectionWithMocks:
    """Test GPU detection with mocked OpenCL for edge cases"""

    def test_no_opencl_available(self):
        """Test behavior when OpenCL is not available"""
        with patch('core.algorithms.gpu_detection.OPENCL_AVAILABLE', False):
            detector = GPUDetectionSystem()
            success = detector.detect_intel_iris_xe()

            assert success is False
            assert len(detector.detected_devices) == 0
            assert detector.intel_device is None

    @patch('core.algorithms.gpu_detection.OPENCL_AVAILABLE', True)
    @patch('core.algorithms.gpu_detection.cl')
    def test_no_intel_platform(self, mock_cl):
        """Test behavior when no Intel platform is found"""
        # Mock platform without Intel
        mock_platform = MagicMock()
        mock_platform.name = "NVIDIA CUDA Platform"
        mock_cl.get_platforms.return_value = [mock_platform]

        detector = GPUDetectionSystem()
        success = detector.detect_intel_iris_xe()

        assert success is False
        assert detector.intel_device is None

    @patch('core.algorithms.gpu_detection.OPENCL_AVAILABLE', True)
    @patch('core.algorithms.gpu_detection.cl')
    def test_intel_platform_no_gpu(self, mock_cl):
        """Test behavior when Intel platform has no GPU devices"""
        # Mock Intel platform without GPU devices
        mock_platform = MagicMock()
        mock_platform.name = "Intel(R) OpenCL Platform"
        mock_platform.get_devices.side_effect = Exception("No GPU devices")
        mock_cl.get_platforms.return_value = [mock_platform]

        detector = GPUDetectionSystem()
        success = detector.detect_intel_iris_xe()

        assert success is False
        assert detector.intel_device is None

    @patch('core.algorithms.gpu_detection.OPENCL_AVAILABLE', True)
    @patch('core.algorithms.gpu_detection.cl')
    def test_unsuitable_intel_gpu(self, mock_cl):
        """Test behavior with unsuitable Intel GPU"""
        # Mock Intel platform with unsuitable GPU
        mock_device = MagicMock()
        mock_device.name = "Intel(R) UHD Graphics 620"
        mock_device.vendor = "Intel"
        mock_device.max_compute_units = 16  # Too few for good performance
        mock_device.global_mem_size = 512 * 1024 * 1024  # 512MB - too little
        mock_device.max_work_group_size = 256
        mock_device.local_mem_size = 64 * 1024
        mock_device.driver_version = "1.0"
        mock_device.opencl_c_version = "OpenCL C 1.2"

        mock_platform = MagicMock()
        mock_platform.name = "Intel(R) OpenCL Platform"
        mock_platform.get_devices.return_value = [mock_device]
        mock_cl.get_platforms.return_value = [mock_platform]
        mock_cl.device_type.GPU = "GPU"

        detector = GPUDetectionSystem()
        success = detector.detect_intel_iris_xe()

        assert success is False  # Unsuitable for genetic algorithms
        assert len(detector.detected_devices) == 1
        assert detector.detected_devices[0].capability_level == GPUCapabilityLevel.BASIC


if __name__ == "__main__":
    # Run basic detection test
    logger.info("🔍 Running GPU Detection Tests")
    logger.info("=" * 50)

    # Test 1: Basic detection
    success, detector = detect_intel_iris_xe()
    logger.info(f"Detection Result: {'✅ Success' if success else '❌ Failed'}")

    if success:
        # Test 2: Benchmark
        logger.info("\n🔄 Running Performance Benchmark...")
        benchmark = detector.benchmark_gpu_performance()

        if benchmark:
            logger.info("✅ Benchmark completed successfully")
        else:
            logger.info("❌ Benchmark failed")

        # Test 3: Recommendations
        logger.info("\n📋 Getting Optimization Recommendations...")
        recommendations = detector.get_optimization_recommendations()

        for key, value in recommendations.items():
            logger.info(f"  {key}: {value}")

    logger.info("\n📊 Detection tests completed")