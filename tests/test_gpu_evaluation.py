"""
GPU Individual Evaluation System Tests

Validates the GPU-accelerated genetic algorithm individual evaluation
including parallel fitness computation and genetic operations.
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

class TestGPUEvaluation:
    """Tests for GPU-accelerated individual evaluation system"""

    @pytest.fixture
    def test_panels(self) -> List[Panel]:
        """Create test panels for evaluation"""
        return [
            Panel(id="P001", width=200, height=150, material="Steel", thickness=2.0, quantity=1, allow_rotation=True),
            Panel(id="P002", width=180, height=120, material="Steel", thickness=2.0, quantity=1, allow_rotation=True),
            Panel(id="P003", width=250, height=100, material="Steel", thickness=2.0, quantity=1, allow_rotation=True),
            Panel(id="P004", width=300, height=200, material="Steel", thickness=2.0, quantity=1, allow_rotation=True),
            Panel(id="P005", width=150, height=180, material="Steel", thickness=2.0, quantity=1, allow_rotation=True),
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
    def test_gpu_population_initialization(self, test_panels):
        """Test genetic algorithm population initialization"""
        optimizer = create_intel_iris_xe_optimizer(
            population_size=50,
            generations=10,
            enable_gpu=True
        )

        try:
            # Test population initialization
            population = optimizer._initialize_gpu_population(len(test_panels))

            # Validate population structure
            assert population.shape == (50, len(test_panels))
            assert population.dtype == np.int32

            # Check that each individual is a permutation
            for individual in population:
                sorted_individual = np.sort(individual)
                expected = np.arange(len(test_panels))
                assert np.array_equal(sorted_individual, expected), "Individual should be a valid permutation"

            logger.info(f"✅ Population initialization: {population.shape}")

        finally:
            optimizer.cleanup()

    @pytest.mark.skipif(not OPENCL_AVAILABLE, reason="OpenCL not available")
    def test_gpu_fitness_evaluation(self, test_panels, test_sheet):
        """Test GPU-accelerated fitness evaluation"""
        optimizer = create_intel_iris_xe_optimizer(
            population_size=30,
            generations=5,
            enable_gpu=True
        )

        try:
            if not optimizer.gpu_available:
                pytest.skip("GPU not available for testing")

            # Initialize population
            population = optimizer._initialize_gpu_population(len(test_panels))

            # Test GPU evaluation
            start_time = time.time()
            gpu_fitness = optimizer._gpu_evaluate_population(test_panels, population, test_sheet)
            gpu_time = time.time() - start_time

            # Test CPU evaluation for comparison
            start_time = time.time()
            cpu_fitness = optimizer._cpu_evaluate_population(test_panels, population, test_sheet)
            cpu_time = time.time() - start_time

            # Validate results
            assert len(gpu_fitness) == len(population), "GPU fitness should match population size"
            assert len(cpu_fitness) == len(population), "CPU fitness should match population size"
            assert np.all(gpu_fitness >= 0), "Fitness scores should be non-negative"
            assert np.all(gpu_fitness <= 100), "Fitness scores should not exceed 100%"

            # Compare GPU vs CPU results (should be similar for simple evaluation)
            max_difference = np.max(np.abs(gpu_fitness - cpu_fitness))
            assert max_difference < 5.0, f"GPU/CPU fitness difference too large: {max_difference}"

            logger.info(f"✅ GPU evaluation: {gpu_time:.4f}s, CPU: {cpu_time:.4f}s")
            logger.info(f"   GPU fitness range: {np.min(gpu_fitness):.2f} - {np.max(gpu_fitness):.2f}")
            logger.info(f"   Max GPU/CPU difference: {max_difference:.2f}")

            # Performance metrics
            if gpu_time > 0:
                speedup_estimate = cpu_time / gpu_time
                logger.info(f"   Estimated speedup: {speedup_estimate:.2f}x")

        finally:
            optimizer.cleanup()

    @pytest.mark.skipif(not OPENCL_AVAILABLE, reason="OpenCL not available")
    def test_genetic_operations(self, test_panels):
        """Test genetic algorithm operations (selection, mutation)"""
        optimizer = create_intel_iris_xe_optimizer(
            population_size=20,
            generations=3,
            mutation_rate=0.2,
            enable_gpu=True
        )

        try:
            # Initialize population and fitness
            population = optimizer._initialize_gpu_population(len(test_panels))
            fitness_scores = np.random.uniform(0, 100, len(population))

            # Test genetic operations
            new_population = optimizer._genetic_operations(population, fitness_scores)

            # Validate new population
            assert new_population.shape == population.shape, "Population shape should be preserved"

            # Check that new population contains valid permutations
            for individual in new_population:
                sorted_individual = np.sort(individual)
                expected = np.arange(len(test_panels))
                assert np.array_equal(sorted_individual, expected), "Individual should remain a valid permutation"

            # Check that population has evolved (should be different)
            differences = np.sum(population != new_population)
            logger.info(f"✅ Genetic operations: {differences} gene differences")
            assert differences > 0, "Population should have evolved"

        finally:
            optimizer.cleanup()

    @pytest.mark.skipif(not OPENCL_AVAILABLE, reason="OpenCL not available")
    def test_placement_generation(self, test_panels, test_sheet):
        """Test placement result generation from genetic encoding"""
        optimizer = create_intel_iris_xe_optimizer(
            population_size=10,
            generations=2,
            enable_gpu=True
        )

        try:
            # Create a test genetic encoding
            genes = np.arange(len(test_panels))
            np.random.shuffle(genes)

            # Generate placement
            placement = optimizer._generate_placement_from_genes(test_panels, genes, test_sheet)

            # Validate placement result
            assert placement is not None, "Placement should be generated"
            assert hasattr(placement, 'panels'), "Placement should have panels"
            assert hasattr(placement, 'efficiency'), "Placement should have efficiency"
            assert 0 <= placement.efficiency <= 1, "Efficiency should be 0-1"

            # Check placed panels
            placed_panel_ids = {p.panel.id for p in placement.panels}
            test_panel_ids = {p.id for p in test_panels}

            logger.info(f"✅ Placement generation: {len(placement.panels)} panels placed")
            logger.info(f"   Efficiency: {placement.efficiency*100:.2f}%")
            logger.info(f"   Panel coverage: {len(placed_panel_ids)}/{len(test_panel_ids)}")

        finally:
            optimizer.cleanup()

    @pytest.mark.skipif(not OPENCL_AVAILABLE, reason="OpenCL not available")
    def test_full_gpu_optimization(self, test_panels, test_sheet, test_constraints):
        """Test complete GPU optimization workflow"""
        optimizer = create_intel_iris_xe_optimizer(
            population_size=30,
            generations=10,
            enable_gpu=True
        )

        try:
            if not optimizer.gpu_available:
                pytest.skip("GPU not available for testing")

            # Run optimization
            start_time = time.time()
            result = optimizer.optimize(test_panels, test_sheet, test_constraints)
            execution_time = time.time() - start_time

            # Validate result structure
            assert 'placement_result' in result, "Result should contain placement_result"
            assert 'best_fitness' in result, "Result should contain best_fitness"
            assert 'execution_time' in result, "Result should contain execution_time"
            assert 'gpu_acceleration' in result, "Result should indicate GPU acceleration"

            # Validate result values
            assert result['best_fitness'] >= 0, "Best fitness should be non-negative"
            assert result['execution_time'] > 0, "Execution time should be positive"
            assert result['gpu_acceleration'] is True, "Should indicate GPU acceleration was used"

            # Validate placement result structure
            placement = result['placement_result']
            assert hasattr(placement, 'panels'), "Placement should have panels"
            assert hasattr(placement, 'efficiency'), "Placement should have efficiency"

            # Performance metrics validation
            if 'performance_metrics' in result:
                metrics = result['performance_metrics']
                assert 'kernel_time' in metrics, "Should have kernel execution time"
                assert 'speedup_estimate' in metrics, "Should have speedup estimate"

                logger.info(f"✅ Full GPU optimization completed in {execution_time:.3f}s")
                logger.info(f"   Best fitness: {result['best_fitness']:.2f}%")
                logger.info(f"   Kernel time: {metrics['kernel_time']:.4f}s")
                logger.info(f"   Speedup estimate: {metrics['speedup_estimate']:.2f}x")

            # Get performance statistics
            stats = optimizer.get_performance_stats()
            logger.info(f"   GPU available: {stats['gpu_available']}")
            if stats['gpu_available']:
                logger.info(f"   GPU device: {stats['gpu_device']}")

        finally:
            optimizer.cleanup()

    def test_cpu_fallback_operation(self, test_panels, test_sheet, test_constraints):
        """Test CPU fallback when GPU is not available"""
        # Force CPU-only mode
        optimizer = create_intel_iris_xe_optimizer(
            population_size=20,
            generations=5,
            enable_gpu=False
        )

        try:
            # Run optimization with CPU fallback
            result = optimizer.optimize(test_panels, test_sheet, test_constraints)

            # Should still work and return valid results
            assert result is not None, "CPU fallback should return valid results"
            logger.info(f"Result type: {type(result)}")

            if isinstance(result, dict):
                assert 'best_fitness' in result, "CPU fallback should calculate fitness"
                # Check that GPU acceleration flag is not set
                assert result.get('gpu_acceleration', False) is False, "Should not indicate GPU acceleration"
                logger.info(f"   Best fitness: {result['best_fitness']:.2f}%")
            else:
                # Handle case where result is a PlacementResult object
                assert hasattr(result, 'efficiency'), "CPU fallback should have efficiency"
                logger.info(f"   Efficiency: {result.efficiency*100:.2f}%")

            logger.info("✅ CPU fallback operation successful")

        finally:
            optimizer.cleanup()

    @pytest.mark.skipif(not OPENCL_AVAILABLE, reason="OpenCL not available")
    def test_performance_comparison(self, test_panels, test_sheet, test_constraints):
        """Compare GPU vs CPU performance"""
        # Test with larger workload for better GPU benefit
        large_panels = test_panels * 10  # 50 panels total

        # GPU optimizer
        gpu_optimizer = create_intel_iris_xe_optimizer(
            population_size=100,
            generations=20,
            enable_gpu=True
        )

        # CPU optimizer
        cpu_optimizer = create_intel_iris_xe_optimizer(
            population_size=100,
            generations=20,
            enable_gpu=False
        )

        try:
            if gpu_optimizer.gpu_available:
                # GPU test
                start_time = time.time()
                gpu_result = gpu_optimizer.optimize(large_panels, test_sheet, test_constraints)
                gpu_time = time.time() - start_time

                # CPU test
                start_time = time.time()
                cpu_result = cpu_optimizer.optimize(large_panels, test_sheet, test_constraints)
                cpu_time = time.time() - start_time

                # Performance comparison
                speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0

                logger.info(f"🏁 Performance Comparison (50 panels, 100 pop, 20 gen):")
                logger.info(f"   GPU time: {gpu_time:.3f}s, fitness: {gpu_result['best_fitness']:.2f}%")
                logger.info(f"   CPU time: {cpu_time:.3f}s, fitness: {cpu_result['best_fitness']:.2f}%")
                logger.info(f"   Speedup: {speedup:.2f}x")

                # GPU should be at least as good as CPU for large workloads
                assert speedup >= 0.5, f"GPU should not be significantly slower than CPU: {speedup:.2f}x"

            else:
                pytest.skip("GPU not available for performance comparison")

        finally:
            gpu_optimizer.cleanup()
            cpu_optimizer.cleanup()


if __name__ == "__main__":
    # Run evaluation tests
    logger.info("🧪 Running GPU Individual Evaluation Tests")
    logger.info("=" * 60)

    # Test basic functionality
    test_optimizer = create_intel_iris_xe_optimizer(
        population_size=20,
        generations=5,
        enable_gpu=True
    )

    try:
        test_panels = [
            Panel(id=f"T{i:02d}", width=100+i*10, height=150+i*5,
                  material="Steel", thickness=2.0, quantity=1, allow_rotation=True)
            for i in range(8)
        ]
        test_sheet = SteelSheet(width=1500.0, height=3100.0)

        if test_optimizer.gpu_available:
            logger.info("✅ GPU available for testing")

            # Test population initialization
            population = test_optimizer._initialize_gpu_population(len(test_panels))
            logger.info(f"✅ Population initialized: {population.shape}")

            # Test GPU evaluation
            fitness = test_optimizer._gpu_evaluate_population(test_panels, population, test_sheet)
            logger.info(f"✅ GPU evaluation: {len(fitness)} fitness scores")
            logger.info(f"   Fitness range: {np.min(fitness):.2f} - {np.max(fitness):.2f}")

            # Test genetic operations
            new_population = test_optimizer._genetic_operations(population, fitness)
            differences = np.sum(population != new_population)
            logger.info(f"✅ Genetic operations: {differences} gene changes")

        else:
            logger.info("⚠️ GPU not available, testing CPU fallback")

    finally:
        test_optimizer.cleanup()

    logger.info("\n🎯 GPU Individual Evaluation System Tests Complete")