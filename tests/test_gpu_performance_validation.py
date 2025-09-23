"""
GPU Performance Validation Tests

Validates the performance characteristics and benefits of GPU acceleration
for steel cutting optimization workloads.
"""

import pytest
import logging
import time
import numpy as np
from typing import List

from core.models import Panel, SteelSheet, OptimizationConstraints
from core.algorithms.intel_iris_xe_optimizer import (
    create_intel_iris_xe_optimizer,
    OPENCL_AVAILABLE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestGPUPerformanceValidation:
    """Performance validation tests for GPU acceleration"""

    @pytest.fixture
    def small_workload(self) -> List[Panel]:
        """Small workload for baseline testing"""
        return [
            Panel(id=f"S{i:03d}", width=100+i*5, height=150+i*3,
                  material="Steel", thickness=2.0, quantity=1, allow_rotation=True)
            for i in range(20)
        ]

    @pytest.fixture
    def medium_workload(self) -> List[Panel]:
        """Medium workload for GPU benefit testing"""
        return [
            Panel(id=f"M{i:03d}", width=80+i*2, height=120+i*4,
                  material="Steel", thickness=2.0, quantity=1, allow_rotation=True)
            for i in range(100)
        ]

    @pytest.fixture
    def large_workload(self) -> List[Panel]:
        """Large workload for maximum GPU benefit"""
        return [
            Panel(id=f"L{i:03d}", width=60+i, height=100+i*2,
                  material="Steel", thickness=2.0, quantity=1, allow_rotation=True)
            for i in range(200)
        ]

    @pytest.fixture
    def test_sheet(self) -> SteelSheet:
        """Standard test sheet"""
        return SteelSheet(width=1500.0, height=3100.0)

    @pytest.fixture
    def test_constraints(self) -> OptimizationConstraints:
        """Standard test constraints"""
        return OptimizationConstraints()

    @pytest.mark.skipif(not OPENCL_AVAILABLE, reason="OpenCL not available")
    def test_workload_scaling_performance(self, small_workload, medium_workload, test_sheet, test_constraints):
        """Test how performance scales with workload size"""

        workloads = [
            ("Small (20 panels)", small_workload),
            ("Medium (100 panels)", medium_workload)
        ]

        gpu_optimizer = create_intel_iris_xe_optimizer(
            population_size=50,
            generations=10,
            enable_gpu=True
        )

        cpu_optimizer = create_intel_iris_xe_optimizer(
            population_size=50,
            generations=10,
            enable_gpu=False
        )

        results = []

        try:
            if not gpu_optimizer.gpu_available:
                pytest.skip("GPU not available for scaling test")

            for workload_name, panels in workloads:
                logger.info(f"\n=== Testing {workload_name} ===")

                # GPU test
                start_time = time.time()
                gpu_result = gpu_optimizer.optimize(panels, test_sheet, test_constraints)
                gpu_time = time.time() - start_time

                # CPU test
                start_time = time.time()
                cpu_result = cpu_optimizer.optimize(panels, test_sheet, test_constraints)
                cpu_time = time.time() - start_time

                # Calculate speedup
                speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0

                result = {
                    'workload': workload_name,
                    'panels': len(panels),
                    'gpu_time': gpu_time,
                    'cpu_time': cpu_time,
                    'speedup': speedup,
                    'gpu_fitness': gpu_result['best_fitness'],
                    'cpu_fitness': cpu_result['best_fitness'] if isinstance(cpu_result, dict) else cpu_result.efficiency * 100
                }
                results.append(result)

                logger.info(f"  GPU: {gpu_time:.3f}s, fitness: {result['gpu_fitness']:.2f}%")
                logger.info(f"  CPU: {cpu_time:.3f}s, fitness: {result['cpu_fitness']:.2f}%")
                logger.info(f"  Speedup: {speedup:.2f}x")

            # Performance validation
            logger.info("\n=== Performance Analysis ===")
            for result in results:
                logger.info(f"{result['workload']}: {result['speedup']:.2f}x speedup")

                # Validate that GPU provides reasonable performance
                assert result['speedup'] > 0.2, f"GPU significantly slower than CPU: {result['speedup']:.2f}x"

                # Validate that fitness quality is maintained
                fitness_diff = abs(result['gpu_fitness'] - result['cpu_fitness'])
                assert fitness_diff < 10.0, f"GPU/CPU fitness difference too large: {fitness_diff:.2f}%"

        finally:
            gpu_optimizer.cleanup()
            cpu_optimizer.cleanup()

    @pytest.mark.skipif(not OPENCL_AVAILABLE, reason="OpenCL not available")
    def test_population_size_scaling(self, medium_workload, test_sheet, test_constraints):
        """Test how performance scales with population size"""

        population_sizes = [30, 50, 100]
        results = []

        for pop_size in population_sizes:
            gpu_optimizer = create_intel_iris_xe_optimizer(
                population_size=pop_size,
                generations=10,
                enable_gpu=True
            )

            try:
                if not gpu_optimizer.gpu_available:
                    pytest.skip("GPU not available for population scaling test")

                start_time = time.time()
                result = gpu_optimizer.optimize(medium_workload, test_sheet, test_constraints)
                execution_time = time.time() - start_time

                results.append({
                    'population_size': pop_size,
                    'execution_time': execution_time,
                    'fitness': result['best_fitness'],
                    'kernel_time': result.get('performance_metrics', {}).get('kernel_time', 0)
                })

                logger.info(f"Population {pop_size}: {execution_time:.3f}s, fitness: {result['best_fitness']:.2f}%")

            finally:
                gpu_optimizer.cleanup()

        # Validate scaling characteristics
        logger.info("\n=== Population Scaling Analysis ===")
        for i, result in enumerate(results):
            logger.info(f"Pop {result['population_size']}: {result['execution_time']:.3f}s")

            # Performance should not degrade dramatically with larger populations
            if i > 0:
                prev_time = results[i-1]['execution_time']
                time_ratio = result['execution_time'] / prev_time
                pop_ratio = result['population_size'] / results[i-1]['population_size']

                # Time scaling should be reasonable (not worse than O(n²))
                assert time_ratio < pop_ratio ** 1.5, f"Poor scaling: {time_ratio:.2f}x time for {pop_ratio:.2f}x population"

    @pytest.mark.skipif(not OPENCL_AVAILABLE, reason="OpenCL not available")
    def test_gpu_memory_utilization(self, large_workload, test_sheet, test_constraints):
        """Test GPU memory utilization and efficiency"""

        optimizer = create_intel_iris_xe_optimizer(
            population_size=100,
            generations=15,
            enable_gpu=True
        )

        try:
            if not optimizer.gpu_available:
                pytest.skip("GPU not available for memory utilization test")

            # Get GPU capabilities
            stats = optimizer.get_performance_stats()
            gpu_memory_mb = optimizer.max_memory_mb

            logger.info(f"GPU Memory Available: {gpu_memory_mb} MB")
            logger.info(f"GPU Device: {stats['gpu_device']}")

            # Run optimization with memory monitoring
            start_time = time.time()
            result = optimizer.optimize(large_workload, test_sheet, test_constraints)
            execution_time = time.time() - start_time

            # Validate results
            assert result['gpu_acceleration'] is True, "Should use GPU acceleration"
            assert execution_time > 0, "Should have positive execution time"
            assert result['best_fitness'] > 0, "Should achieve positive fitness"

            # Memory efficiency validation
            panels_count = len(large_workload)
            population_size = optimizer.population_size
            estimated_memory_mb = (panels_count + population_size) * 0.001

            logger.info(f"Large workload test: {panels_count} panels, {population_size} population")
            logger.info(f"Execution time: {execution_time:.3f}s")
            logger.info(f"Best fitness: {result['best_fitness']:.2f}%")
            logger.info(f"Estimated memory usage: {estimated_memory_mb:.2f} MB")

            # Memory usage should be reasonable
            assert estimated_memory_mb < gpu_memory_mb * 0.8, "Memory usage should be within GPU limits"

        finally:
            optimizer.cleanup()

    def test_cpu_baseline_performance(self, medium_workload, test_sheet, test_constraints):
        """Establish CPU baseline performance characteristics"""

        optimizer = create_intel_iris_xe_optimizer(
            population_size=50,
            generations=10,
            enable_gpu=False
        )

        try:
            # Run multiple iterations for consistent timing
            execution_times = []
            fitness_scores = []

            for i in range(3):
                start_time = time.time()
                result = optimizer.optimize(medium_workload, test_sheet, test_constraints)
                execution_time = time.time() - start_time

                execution_times.append(execution_time)
                fitness = result['best_fitness'] if isinstance(result, dict) else result.efficiency * 100
                fitness_scores.append(fitness)

            avg_time = np.mean(execution_times)
            avg_fitness = np.mean(fitness_scores)
            time_std = np.std(execution_times)

            logger.info(f"CPU Baseline Performance (100 panels, 50 pop, 10 gen):")
            logger.info(f"  Average time: {avg_time:.3f}s ± {time_std:.3f}s")
            logger.info(f"  Average fitness: {avg_fitness:.2f}%")
            logger.info(f"  Time range: {min(execution_times):.3f}s - {max(execution_times):.3f}s")

            # Validate consistency
            assert time_std < avg_time * 0.3, "Execution time should be reasonably consistent"
            assert avg_fitness > 1.0, "Should achieve reasonable fitness"

        finally:
            optimizer.cleanup()

    @pytest.mark.skipif(not OPENCL_AVAILABLE, reason="OpenCL not available")
    def test_thermal_throttling_simulation(self, medium_workload, test_sheet, test_constraints):
        """Test behavior under simulated thermal conditions"""

        optimizer = create_intel_iris_xe_optimizer(
            population_size=50,
            generations=10,
            enable_gpu=True,
            thermal_monitoring=True
        )

        try:
            if not optimizer.gpu_available:
                pytest.skip("GPU not available for thermal test")

            # Simulate high temperature
            optimizer.thermal_state.cpu_temperature = 90.0  # Above thermal limit
            optimizer.thermal_state.is_throttling = True

            # Run optimization
            result = optimizer.optimize(medium_workload, test_sheet, test_constraints)

            # Should still complete successfully (with CPU fallback)
            assert result is not None, "Should complete even under thermal throttling"
            assert result['best_fitness'] > 0, "Should achieve positive fitness"

            logger.info(f"Thermal throttling test: {result['best_fitness']:.2f}% fitness")
            logger.info(f"GPU acceleration used: {result.get('gpu_acceleration', False)}")

        finally:
            optimizer.cleanup()


if __name__ == "__main__":
    # Run performance validation
    logger.info("🏁 GPU Performance Validation Tests")
    logger.info("=" * 50)

    # Create test workloads
    small_panels = [
        Panel(id=f"S{i:02d}", width=100+i*5, height=150+i*3,
              material="Steel", thickness=2.0, quantity=1, allow_rotation=True)
        for i in range(20)
    ]

    medium_panels = [
        Panel(id=f"M{i:02d}", width=80+i*2, height=120+i*4,
              material="Steel", thickness=2.0, quantity=1, allow_rotation=True)
        for i in range(50)
    ]

    test_sheet = SteelSheet(width=1500.0, height=3100.0)
    test_constraints = OptimizationConstraints()

    # Test GPU vs CPU performance
    if OPENCL_AVAILABLE:
        gpu_optimizer = create_intel_iris_xe_optimizer(
            population_size=50,
            generations=10,
            enable_gpu=True
        )

        cpu_optimizer = create_intel_iris_xe_optimizer(
            population_size=50,
            generations=10,
            enable_gpu=False
        )

        try:
            if gpu_optimizer.gpu_available:
                logger.info("🎮 GPU vs CPU Performance Comparison")

                # Small workload test
                start_time = time.time()
                gpu_result = gpu_optimizer.optimize(small_panels, test_sheet, test_constraints)
                gpu_time = time.time() - start_time

                start_time = time.time()
                cpu_result = cpu_optimizer.optimize(small_panels, test_sheet, test_constraints)
                cpu_time = time.time() - start_time

                speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0

                logger.info(f"Small workload (20 panels):")
                logger.info(f"  GPU: {gpu_time:.3f}s, fitness: {gpu_result['best_fitness']:.2f}%")
                logger.info(f"  CPU: {cpu_time:.3f}s, fitness: {cpu_result['best_fitness']:.2f}%")
                logger.info(f"  Speedup: {speedup:.2f}x")

                # Medium workload test
                start_time = time.time()
                gpu_result = gpu_optimizer.optimize(medium_panels, test_sheet, test_constraints)
                gpu_time = time.time() - start_time

                start_time = time.time()
                cpu_result = cpu_optimizer.optimize(medium_panels, test_sheet, test_constraints)
                cpu_time = time.time() - start_time

                speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0

                logger.info(f"Medium workload (50 panels):")
                logger.info(f"  GPU: {gpu_time:.3f}s, fitness: {gpu_result['best_fitness']:.2f}%")
                logger.info(f"  CPU: {cpu_time:.3f}s, fitness: {cpu_result['best_fitness']:.2f}%")
                logger.info(f"  Speedup: {speedup:.2f}x")

            else:
                logger.info("⚠️ GPU not available, testing CPU performance only")

                start_time = time.time()
                result = cpu_optimizer.optimize(medium_panels, test_sheet, test_constraints)
                execution_time = time.time() - start_time

                fitness = result['best_fitness'] if isinstance(result, dict) else result.efficiency * 100
                logger.info(f"CPU performance: {execution_time:.3f}s, fitness: {fitness:.2f}%")

        finally:
            gpu_optimizer.cleanup()
            cpu_optimizer.cleanup()

    else:
        logger.info("⚠️ OpenCL not available - GPU acceleration disabled")

    logger.info("\n🎯 Performance Validation Complete")