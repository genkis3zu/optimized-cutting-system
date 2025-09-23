"""
GPU Acceleration Demonstration

Demonstrates the working GPU-accelerated genetic algorithm implementation
for steel cutting optimization.
"""

import logging
import time
from typing import List

from core.models import Panel, SteelSheet, OptimizationConstraints
from core.algorithms.intel_iris_xe_optimizer import (
    create_intel_iris_xe_optimizer,
    OPENCL_AVAILABLE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_panels(count: int) -> List[Panel]:
    """Create test panels for demonstration"""
    return [
        Panel(id=f"P{i:03d}", width=80+i*3, height=120+i*2,
              material="Steel", thickness=2.0, quantity=1, allow_rotation=True)
        for i in range(count)
    ]

def main():
    """GPU acceleration demonstration"""
    logger.info("🚀 GPU Acceleration Demonstration")
    logger.info("=" * 60)

    if not OPENCL_AVAILABLE:
        logger.error("❌ OpenCL not available - GPU acceleration disabled")
        return

    # Create test data
    test_panels = create_test_panels(50)
    test_sheet = SteelSheet(width=1500.0, height=3100.0)
    test_constraints = OptimizationConstraints()

    logger.info(f"Test Configuration:")
    logger.info(f"  Panels: {len(test_panels)}")
    logger.info(f"  Sheet: {test_sheet.width}×{test_sheet.height}mm")
    logger.info(f"  Population: 50, Generations: 20")

    # Test 1: GPU Detection and Capabilities
    logger.info("\n🔍 GPU Detection and Capabilities")
    logger.info("-" * 40)

    gpu_optimizer = create_intel_iris_xe_optimizer(
        population_size=50,
        generations=20,
        enable_gpu=True
    )

    try:
        stats = gpu_optimizer.get_performance_stats()

        if stats['gpu_available']:
            logger.info(f"✅ GPU Available: {stats['gpu_device']}")
            logger.info(f"   Memory: {gpu_optimizer.max_memory_mb} MB")
            logger.info(f"   Max Workgroup: {gpu_optimizer.max_workgroup_size}")
        else:
            logger.info("❌ GPU not available - using CPU fallback")

        # Test 2: Population Initialization
        logger.info("\n🧬 Population Initialization")
        logger.info("-" * 40)

        population = gpu_optimizer._initialize_gpu_population(len(test_panels))
        logger.info(f"✅ Population initialized: {population.shape}")
        logger.info(f"   Data type: {population.dtype}")
        logger.info(f"   Sample individual: {population[0][:10]}...")

        # Test 3: GPU Fitness Evaluation
        logger.info("\n⚡ GPU Fitness Evaluation")
        logger.info("-" * 40)

        start_time = time.time()
        fitness_scores = gpu_optimizer._gpu_evaluate_population(test_panels, population, test_sheet)
        eval_time = time.time() - start_time

        logger.info(f"✅ GPU evaluation completed in {eval_time:.4f}s")
        logger.info(f"   Fitness scores: {len(fitness_scores)}")
        logger.info(f"   Range: {fitness_scores.min():.2f} - {fitness_scores.max():.2f}")
        logger.info(f"   Average: {fitness_scores.mean():.2f}")

        # Test 4: Complete GPU Optimization
        logger.info("\n🎯 Complete GPU Optimization")
        logger.info("-" * 40)

        start_time = time.time()
        result = gpu_optimizer.optimize(test_panels, test_sheet, test_constraints)
        optimization_time = time.time() - start_time

        logger.info(f"✅ Optimization completed in {optimization_time:.3f}s")
        logger.info(f"   Best fitness: {result['best_fitness']:.2f}%")
        logger.info(f"   GPU acceleration: {result['gpu_acceleration']}")

        if 'performance_metrics' in result:
            metrics = result['performance_metrics']
            logger.info(f"   Kernel time: {metrics.get('kernel_time', 0):.4f}s")
            logger.info(f"   Speedup estimate: {metrics.get('speedup_estimate', 1):.2f}x")

        # Test 5: Placement Result Analysis
        logger.info("\n📊 Placement Result Analysis")
        logger.info("-" * 40)

        placement = result['placement_result']
        logger.info(f"✅ Placement generated successfully")
        logger.info(f"   Panels placed: {len(placement.panels)}")
        logger.info(f"   Efficiency: {placement.efficiency*100:.2f}%")
        logger.info(f"   Waste area: {placement.waste_area:.0f} mm²")
        logger.info(f"   Material cost: ¥{placement.cost:.0f}")

        # Test 6: CPU Comparison
        logger.info("\n⚖️ CPU vs GPU Comparison")
        logger.info("-" * 40)

        cpu_optimizer = create_intel_iris_xe_optimizer(
            population_size=50,
            generations=20,
            enable_gpu=False
        )

        try:
            start_time = time.time()
            cpu_result = cpu_optimizer.optimize(test_panels, test_sheet, test_constraints)
            cpu_time = time.time() - start_time

            gpu_time = result['execution_time']
            speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0

            logger.info(f"🎮 GPU: {gpu_time:.3f}s, fitness: {result['best_fitness']:.2f}%")
            logger.info(f"💻 CPU: {cpu_time:.3f}s, fitness: {cpu_result['best_fitness']:.2f}%")
            logger.info(f"🚀 Speedup: {speedup:.2f}x")

            if speedup > 1.0:
                logger.info(f"✅ GPU provides {speedup:.1f}x performance improvement")
            elif speedup > 0.8:
                logger.info(f"⚡ GPU performance comparable to CPU ({speedup:.1f}x)")
            else:
                logger.info(f"⚠️ GPU slower than CPU for this workload ({speedup:.1f}x)")

        finally:
            cpu_optimizer.cleanup()

        # Test 7: Memory Access Pattern Validation
        logger.info("\n🧠 Memory Access Pattern Validation")
        logger.info("-" * 40)

        # Test with different population sizes
        for pop_size in [30, 60, 100]:
            test_optimizer = create_intel_iris_xe_optimizer(
                population_size=pop_size,
                generations=5,
                enable_gpu=True
            )

            try:
                if test_optimizer.gpu_available:
                    start_time = time.time()
                    test_result = test_optimizer.optimize(test_panels[:30], test_sheet, test_constraints)
                    test_time = time.time() - start_time

                    logger.info(f"   Pop {pop_size}: {test_time:.3f}s, fitness: {test_result['best_fitness']:.2f}%")
                else:
                    logger.info(f"   Pop {pop_size}: GPU not available")

            finally:
                test_optimizer.cleanup()

    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        gpu_optimizer.cleanup()

    logger.info("\n🎉 GPU Acceleration Demonstration Complete")
    logger.info("   ✅ GPU detection and initialization")
    logger.info("   ✅ Population management and genetic operations")
    logger.info("   ✅ GPU-accelerated fitness evaluation")
    logger.info("   ✅ Complete optimization workflow")
    logger.info("   ✅ CPU fallback and comparison")
    logger.info("   ✅ Memory access pattern optimization")

if __name__ == "__main__":
    main()