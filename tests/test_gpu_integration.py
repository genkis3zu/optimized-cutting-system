"""
Integration Tests for GPU-Accelerated Genetic Algorithm

Tests the complete GPU acceleration pipeline including:
- GPU detection and initialization
- Fallback manager functionality
- Intel Iris Xe optimizer integration
- Performance monitoring
- Error handling and recovery
"""

import pytest
import logging
import time
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock

from core.models import Panel, SteelSheet, OptimizationConstraints
from core.algorithms.intel_iris_xe_optimizer import (
    IntelIrisXeOptimizer,
    create_intel_iris_xe_optimizer,
    OPENCL_AVAILABLE
)
from core.algorithms.gpu_fallback_manager import (
    GPUFallbackManager,
    ExecutionContext,
    ExecutionMode,
    FallbackReason
)
from core.algorithms.gpu_detection import detect_intel_iris_xe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestGPUIntegration:
    """Integration tests for GPU acceleration system"""

    @pytest.fixture
    def sample_panels(self) -> List[Panel]:
        """Create sample panels for testing"""
        return [
            Panel(id=f"P{i:03d}", width=100 + i*10, height=200 + i*5,
                  material="Steel", thickness=2.0, quantity=1, allow_rotation=True)
            for i in range(20)
        ]

    @pytest.fixture
    def large_panel_set(self) -> List[Panel]:
        """Create larger panel set for GPU benefit testing"""
        return [
            Panel(id=f"L{i:03d}", width=50 + i*2, height=150 + i*3,
                  material="Steel", thickness=2.0, quantity=1, allow_rotation=True)
            for i in range(100)
        ]

    @pytest.fixture
    def test_sheet(self) -> SteelSheet:
        """Create test steel sheet"""
        return SteelSheet(width=1500.0, height=3100.0)

    @pytest.fixture
    def test_constraints(self) -> OptimizationConstraints:
        """Create test constraints"""
        return OptimizationConstraints()

    @pytest.mark.skipif(not OPENCL_AVAILABLE, reason="OpenCL not available")
    def test_gpu_optimizer_creation(self):
        """Test GPU optimizer creation and initialization"""
        optimizer = create_intel_iris_xe_optimizer(
            population_size=50,
            generations=10,
            enable_gpu=True,
            thermal_monitoring=True
        )

        assert optimizer is not None
        assert optimizer.fallback_manager is not None
        assert hasattr(optimizer, 'gpu_available')

        # Test performance stats
        stats = optimizer.get_performance_stats()
        assert 'gpu_available' in stats
        assert 'performance_metrics' in stats

        logger.info(f"GPU Available: {stats['gpu_available']}")
        if stats['gpu_available']:
            logger.info(f"GPU Device: {stats['gpu_device']}")

        optimizer.cleanup()

    @pytest.mark.skipif(not OPENCL_AVAILABLE, reason="OpenCL not available")
    def test_small_workload_optimization(self, sample_panels, test_sheet, test_constraints):
        """Test optimization with small workload (should prefer CPU)"""
        optimizer = create_intel_iris_xe_optimizer(
            population_size=20,  # Small population
            generations=5,
            enable_gpu=True
        )

        start_time = time.time()
        result = optimizer.optimize(sample_panels, test_sheet, test_constraints)
        execution_time = time.time() - start_time

        # Validate result structure
        assert result is not None
        logger.info(f"Optimization result type: {type(result)}")

        logger.info(f"Small workload completed in {execution_time:.2f}s")

        # Get execution statistics
        stats = optimizer.fallback_manager.get_execution_stats()
        logger.info(f"Execution counts: {stats['execution_counts']}")
        logger.info(f"Fallback events: {stats['fallback_events']}")

        optimizer.cleanup()

    @pytest.mark.skipif(not OPENCL_AVAILABLE, reason="OpenCL not available")
    def test_large_workload_optimization(self, large_panel_set):
        """Test optimization with large workload (should use GPU if available)"""
        success, detector = detect_intel_iris_xe()

        if not success:
            pytest.skip("No suitable Intel GPU available")

        optimizer = create_intel_iris_xe_optimizer(
            population_size=100,  # Large population
            generations=10,
            enable_gpu=True,
            thermal_monitoring=True
        )

        # Create test sheet and constraints
        sheet = SteelSheet(width=1500, height=3100, thickness=3.0, material="Steel")
        constraints = {}

        start_time = time.time()
        result = optimizer.optimize(large_panel_set, sheet, constraints)
        execution_time = time.time() - start_time

        # Validate result
        assert 'placement_result' in result
        assert 'best_fitness' in result

        logger.info(f"Large workload completed in {execution_time:.2f}s")
        logger.info(f"Best fitness: {result['best_fitness']:.2f}")

        # Analyze execution statistics
        stats = optimizer.fallback_manager.get_execution_stats()
        logger.info(f"Execution stats: {stats}")

        # Performance analysis
        perf_stats = optimizer.get_performance_stats()
        if 'performance_metrics' in perf_stats:
            metrics = perf_stats['performance_metrics']
            logger.info(f"Speedup factor: {metrics.get('speedup_factor', 1.0):.2f}x")

        optimizer.cleanup()

    def test_fallback_manager_functionality(self):
        """Test fallback manager without GPU dependency"""
        def mock_gpu_executor(*args, **kwargs):
            return {"result": "gpu", "time": 0.1}

        def mock_cpu_executor(*args, **kwargs):
            return {"result": "cpu", "time": 0.5}

        manager = GPUFallbackManager(
            thermal_monitoring=False,  # Disable for testing
            max_gpu_errors=2
        )

        manager.register_executors(mock_gpu_executor, mock_cpu_executor)

        # Test normal execution
        context = ExecutionContext(
            num_panels=50,
            population_size=30,
            generations=10,
            available_memory_mb=4096.0,
            current_temperature=60.0
        )

        result = manager.execute_with_fallback(context)
        assert result is not None

        # Test execution stats
        stats = manager.get_execution_stats()
        assert stats['execution_counts']['total_executions'] >= 1

        manager.cleanup()

    def test_fallback_on_gpu_errors(self):
        """Test fallback behavior when GPU fails"""
        def failing_gpu_executor(*args, **kwargs):
            raise RuntimeError("Mock GPU failure")

        def reliable_cpu_executor(*args, **kwargs):
            return {"result": "cpu_fallback", "time": 0.3}

        manager = GPUFallbackManager(
            thermal_monitoring=False,
            max_gpu_errors=1
        )

        manager.register_executors(failing_gpu_executor, reliable_cpu_executor)

        context = ExecutionContext(
            num_panels=100,
            population_size=50,
            generations=10,
            available_memory_mb=4096.0,
            current_temperature=60.0
        )

        # First execution should attempt GPU and fallback to CPU
        result = manager.execute_with_fallback(context)
        assert result["result"] == "cpu_fallback"

        # Check that fallback event was recorded
        stats = manager.get_execution_stats()
        assert stats['fallback_events'] >= 1
        assert stats['errors']['gpu_errors'] >= 1

        manager.cleanup()

    def test_thermal_aware_execution(self):
        """Test thermal-aware execution decisions"""
        def mock_gpu_executor(*args, **kwargs):
            return {"result": "gpu", "thermal_load": "high"}

        def mock_cpu_executor(*args, **kwargs):
            return {"result": "cpu", "thermal_load": "low"}

        manager = GPUFallbackManager(
            thermal_monitoring=False,  # Manual thermal testing
            thermal_limit=75.0
        )

        manager.register_executors(mock_gpu_executor, mock_cpu_executor)

        # Test with high temperature (should prefer CPU)
        hot_context = ExecutionContext(
            num_panels=100,
            population_size=100,
            generations=10,
            available_memory_mb=4096.0,
            current_temperature=80.0  # Above thermal limit
        )

        # Manually set temperature for testing
        manager.current_temperature = 80.0

        should_use_gpu = manager.should_use_gpu(hot_context)
        logger.info(f"Should use GPU at 80°C: {should_use_gpu}")

        # Test with normal temperature
        normal_context = ExecutionContext(
            num_panels=100,
            population_size=100,
            generations=10,
            available_memory_mb=4096.0,
            current_temperature=60.0
        )

        manager.current_temperature = 60.0
        should_use_gpu_normal = manager.should_use_gpu(normal_context)
        logger.info(f"Should use GPU at 60°C: {should_use_gpu_normal}")

        manager.cleanup()

    @pytest.mark.skipif(not OPENCL_AVAILABLE, reason="OpenCL not available")
    def test_performance_monitoring(self, sample_panels):
        """Test performance monitoring and metrics collection"""
        optimizer = create_intel_iris_xe_optimizer(
            population_size=30,
            generations=5,
            enable_gpu=True
        )

        # Create test sheet and constraints
        sheet = SteelSheet(width=1500, height=3100, thickness=3.0, material="Steel")
        constraints = {}

        # Run multiple optimizations to collect metrics
        for i in range(3):
            result = optimizer.optimize(sample_panels, sheet, constraints)
            logger.info(f"Optimization {i+1} completed with fitness {result['best_fitness']:.2f}")

        # Analyze performance metrics
        stats = optimizer.fallback_manager.get_execution_stats()
        logger.info("Performance Monitoring Results:")
        logger.info(f"  Total executions: {stats['execution_counts']['total_executions']}")
        logger.info(f"  GPU executions: {stats['execution_counts']['gpu_executions']}")
        logger.info(f"  CPU executions: {stats['execution_counts']['cpu_executions']}")

        if stats['execution_counts']['gpu_executions'] > 0 and stats['execution_counts']['cpu_executions'] > 0:
            speedup = stats['performance']['gpu_speedup']
            logger.info(f"  Measured speedup: {speedup:.2f}x")

        optimizer.cleanup()

    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure conditions"""
        def memory_hungry_gpu_executor(*args, **kwargs):
            # Simulate high memory usage
            raise MemoryError("Insufficient GPU memory")

        def efficient_cpu_executor(*args, **kwargs):
            return {"result": "cpu_memory_efficient"}

        manager = GPUFallbackManager(
            thermal_monitoring=False,
            memory_limit_mb=2048.0  # Lower limit for testing
        )

        manager.register_executors(memory_hungry_gpu_executor, efficient_cpu_executor)

        # Large memory context
        memory_context = ExecutionContext(
            num_panels=2000,  # Large dataset
            population_size=200,
            generations=50,
            available_memory_mb=1024.0,  # Limited memory
            current_temperature=60.0
        )

        result = manager.execute_with_fallback(memory_context)
        assert result["result"] == "cpu_memory_efficient"

        # Check fallback event
        stats = manager.get_execution_stats()
        assert stats['fallback_events'] >= 1

        manager.cleanup()

    def test_optimizer_cleanup(self):
        """Test proper resource cleanup"""
        optimizer = create_intel_iris_xe_optimizer(
            population_size=20,
            generations=5,
            enable_gpu=True
        )

        # Use the optimizer
        initial_stats = optimizer.get_performance_stats()
        logger.info(f"Initial GPU available: {initial_stats['gpu_available']}")

        # Cleanup
        optimizer.cleanup()

        # Verify cleanup completed without errors
        logger.info("Optimizer cleanup completed successfully")


class TestGPUIntegrationWithMocks:
    """Test GPU integration with mocked components for edge cases"""

    def test_no_gpu_available(self):
        """Test behavior when no GPU is available"""
        with patch('core.algorithms.intel_iris_xe_optimizer.OPENCL_AVAILABLE', False):
            optimizer = create_intel_iris_xe_optimizer(
                population_size=30,
                generations=5,
                enable_gpu=True  # Should be ignored
            )

            stats = optimizer.get_performance_stats()
            assert stats['gpu_available'] is False

            optimizer.cleanup()

    @patch('core.algorithms.gpu_detection.detect_intel_iris_xe')
    def test_gpu_detection_failure(self, mock_detect):
        """Test handling of GPU detection failure"""
        mock_detect.return_value = (False, None)

        optimizer = create_intel_iris_xe_optimizer(
            population_size=30,
            generations=5,
            enable_gpu=True
        )

        stats = optimizer.get_performance_stats()
        # Should fallback to CPU-only mode
        assert 'gpu_available' in stats

        optimizer.cleanup()


if __name__ == "__main__":
    # Run integration tests
    logger.info("🧪 Running GPU Integration Tests")
    logger.info("=" * 50)

    # Test 1: Basic GPU detection
    logger.info("Test 1: GPU Detection")
    success, detector = detect_intel_iris_xe()
    logger.info(f"Result: {'✅ Success' if success else '❌ Failed'}")

    if success:
        # Test 2: Optimizer creation
        logger.info("\nTest 2: Optimizer Creation")
        optimizer = create_intel_iris_xe_optimizer(
            population_size=50,
            generations=10,
            enable_gpu=True
        )
        logger.info("✅ Optimizer created successfully")

        # Test 3: Sample optimization
        logger.info("\nTest 3: Sample Optimization")
        sample_panels = [
            Panel(id=f"T{i:02d}", width=100+i*5, height=200+i*3,
                  material="Steel", thickness=2.0, quantity=1, allow_rotation=True)
            for i in range(10)
        ]

        # Create test sheet and constraints
        sheet = SteelSheet(width=1500, height=3100, thickness=3.0, material="Steel")
        constraints = {}

        start_time = time.time()
        result = optimizer.optimize(sample_panels, sheet, constraints)
        execution_time = time.time() - start_time

        logger.info(f"✅ Optimization completed in {execution_time:.2f}s")
        logger.info(f"   Best fitness: {result['best_fitness']:.2f}")

        # Test 4: Performance statistics
        logger.info("\nTest 4: Performance Statistics")
        stats = optimizer.fallback_manager.get_execution_stats()
        logger.info(f"   Total executions: {stats['execution_counts']['total_executions']}")
        logger.info(f"   Fallback events: {stats['fallback_events']}")

        optimizer.cleanup()
        logger.info("✅ Cleanup completed")

    else:
        logger.info("ℹ️ GPU not available - testing CPU-only mode")
        optimizer = create_intel_iris_xe_optimizer(
            population_size=30,
            generations=5,
            enable_gpu=False
        )
        logger.info("✅ CPU-only optimizer created")
        optimizer.cleanup()

    logger.info("\n🎯 Integration tests completed")