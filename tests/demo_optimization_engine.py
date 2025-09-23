#!/usr/bin/env python3
"""
Demonstration of the complete Steel Cutting Optimization Engine
Steel cutting optimization engine demonstration

This script demonstrates the enhanced optimization engine with multiple algorithms,
material separation, timeout handling, and performance monitoring.
"""

import time
import logging
from typing import List

from core.models import Panel, OptimizationConstraints
from core.optimizer import create_optimization_engine

def setup_logging():
    """Configure logging for demonstration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def create_demo_panels() -> List[Panel]:
    """Create sample panels for demonstration"""
    return [
        # Large structural panels (SS400 steel)
        Panel("Frame_Large_1", 1200, 800, 2, "SS400", 6.0, priority=1),
        Panel("Frame_Large_2", 1000, 600, 1, "SS400", 6.0, priority=1),

        # Medium panels (SS400 steel)
        Panel("Frame_Medium_1", 600, 400, 4, "SS400", 6.0, priority=2),
        Panel("Frame_Medium_2", 500, 350, 3, "SS400", 6.0, priority=2),

        # Small reinforcement panels (SS400 steel)
        Panel("Reinforcement_1", 300, 200, 6, "SS400", 6.0, priority=3),
        Panel("Reinforcement_2", 250, 150, 8, "SS400", 6.0, priority=3),

        # Stainless steel covers (SUS304)
        Panel("Cover_1", 500, 300, 2, "SUS304", 3.0, priority=2, allow_rotation=True),
        Panel("Cover_2", 400, 250, 3, "SUS304", 3.0, priority=2, allow_rotation=True),

        # Aluminum panels (AL6061)
        Panel("Aluminum_1", 350, 200, 4, "AL6061", 2.0, priority=3, allow_rotation=True),
        Panel("Aluminum_2", 300, 180, 2, "AL6061", 2.0, priority=3, allow_rotation=True),
    ]

def demo_single_algorithm():
    """Demonstrate single algorithm optimization"""
    print("\n" + "="*70)
    print("SINGLE ALGORITHM OPTIMIZATION DEMO")
    print("="*70)

    engine = create_optimization_engine()
    panels = create_demo_panels()[:5]  # Use first 5 panels for quick demo

    # Test FFD algorithm
    print(f"\nTesting FFD algorithm with {len(panels)} panel types...")
    constraints = OptimizationConstraints(
        material_separation=False,
        time_budget=5.0,
        target_efficiency=0.70
    )

    start_time = time.time()
    results = engine.optimize(panels, constraints, algorithm_hint="FFD")
    processing_time = time.time() - start_time

    if results:
        result = results[0]
        print(f"[OK] FFD Results:")
        print(f"   - Panels placed: {len(result.panels)}")
        print(f"   - Efficiency: {result.efficiency:.1%}")
        print(f"   - Processing time: {processing_time:.3f}s")
        print(f"   - Waste area: {result.waste_area:.0f} mm^2")
    else:
        print("[FAIL] FFD optimization failed")

def demo_material_separation():
    """Demonstrate material separation optimization"""
    print("\n" + "="*70)
    print("MATERIAL SEPARATION OPTIMIZATION DEMO")
    print("="*70)

    engine = create_optimization_engine()
    panels = create_demo_panels()

    # Count panels by material
    material_count = {}
    total_panels = 0
    for panel in panels:
        total_panels += panel.quantity
        material_count[panel.material] = material_count.get(panel.material, 0) + panel.quantity

    print(f"\nOptimizing {len(panels)} panel types ({total_panels} total panels)")
    print("Material distribution:")
    for material, count in material_count.items():
        print(f"   - {material}: {count} panels")

    constraints = OptimizationConstraints(
        material_separation=True,
        time_budget=15.0,
        target_efficiency=0.75
    )

    start_time = time.time()
    results = engine.optimize(panels, constraints)
    processing_time = time.time() - start_time

    print(f"\n[OK] Material Separation Results:")
    print(f"   - Number of sheets: {len(results)}")
    print(f"   - Total processing time: {processing_time:.3f}s")

    total_panels_placed = 0
    total_efficiency_weighted = 0
    total_area = 0

    for i, result in enumerate(results, 1):
        total_panels_placed += len(result.panels)
        total_efficiency_weighted += result.efficiency * result.used_area
        total_area += result.used_area

        print(f"\n   Sheet {i} ({result.material_block}):")
        print(f"     - Panels placed: {len(result.panels)}")
        print(f"     - Efficiency: {result.efficiency:.1%}")
        print(f"     - Used area: {result.used_area:.0f} mm^2")

    overall_efficiency = total_efficiency_weighted / total_area if total_area > 0 else 0
    print(f"\n   [STATS] Overall Statistics:")
    print(f"     - Total panels placed: {total_panels_placed}/{total_panels}")
    print(f"     - Overall efficiency: {overall_efficiency:.1%}")
    print(f"     - Placement rate: {total_panels_placed/total_panels:.1%}")

def demo_algorithm_comparison():
    """Demonstrate comparison between different algorithms"""
    print("\n" + "="*70)
    print("ALGORITHM COMPARISON DEMO")
    print("="*70)

    engine = create_optimization_engine()
    panels = create_demo_panels()[:6]  # Use subset for faster comparison

    algorithms = ["FFD"]
    if "BFD" in engine.algorithms:
        algorithms.append("BFD")
    if "HYBRID" in engine.algorithms:
        algorithms.append("HYBRID")

    print(f"\nComparing algorithms: {', '.join(algorithms)}")
    print(f"Using {len(panels)} panel types for comparison")

    results_comparison = {}

    for algorithm in algorithms:
        print(f"\n[TEST] Testing {algorithm}...")

        constraints = OptimizationConstraints(
            material_separation=False,
            time_budget=10.0,
            target_efficiency=0.75
        )

        start_time = time.time()
        try:
            results = engine.optimize(panels, constraints, algorithm_hint=algorithm)
            processing_time = time.time() - start_time

            if results and results[0].panels:
                result = results[0]
                results_comparison[algorithm] = {
                    'efficiency': result.efficiency,
                    'panels_placed': len(result.panels),
                    'processing_time': processing_time,
                    'success': True
                }
                print(f"   [OK] Success: {result.efficiency:.1%} efficiency, {processing_time:.3f}s")
            else:
                results_comparison[algorithm] = {
                    'efficiency': 0,
                    'panels_placed': 0,
                    'processing_time': processing_time,
                    'success': False
                }
                print(f"   [FAIL] Failed to place panels")

        except Exception as e:
            processing_time = time.time() - start_time
            results_comparison[algorithm] = {
                'efficiency': 0,
                'panels_placed': 0,
                'processing_time': processing_time,
                'success': False
            }
            print(f"   [FAIL] Error: {e}")

    # Display comparison results
    print(f"\n[STATS] Algorithm Comparison Results:")
    print(f"{'Algorithm':<12} {'Efficiency':<12} {'Panels':<8} {'Time (s)':<10} {'Status'}")
    print("-" * 50)

    for algorithm, data in results_comparison.items():
        status = "[OK] OK" if data['success'] else "[FAIL] FAIL"
        print(f"{algorithm:<12} {data['efficiency']:>10.1%} {data['panels_placed']:>6} "
              f"{data['processing_time']:>8.3f} {status}")

def demo_performance_monitoring():
    """Demonstrate performance monitoring capabilities"""
    print("\n" + "="*70)
    print("PERFORMANCE MONITORING DEMO")
    print("="*70)

    engine = create_optimization_engine()
    panels = create_demo_panels()

    print(f"Running performance analysis with {len(panels)} panel types...")

    # Get initial performance baseline
    initial_stats = engine.performance_monitor.get_average_performance()
    print(f"Initial runs recorded: {initial_stats.get('total_runs', 0)}")

    # Run multiple optimizations to collect performance data
    for i in range(3):
        print(f"\nRun {i+1}/3...")

        constraints = OptimizationConstraints(
            material_separation=True,
            time_budget=8.0,
            target_efficiency=0.70
        )

        results = engine.optimize(panels, constraints)

        if results:
            total_panels = sum(len(r.panels) for r in results)
            avg_efficiency = sum(r.efficiency for r in results) / len(results)
            print(f"   - Placed {total_panels} panels with {avg_efficiency:.1%} avg efficiency")

    # Display performance statistics
    final_stats = engine.performance_monitor.get_average_performance()
    timeout_stats = engine.timeout_manager.get_timeout_statistics()

    print(f"\n[PERFORMANCE] Performance Summary:")
    print(f"   - Total optimization runs: {final_stats.get('total_runs', 0)}")
    if final_stats.get('total_runs', 0) > 0:
        print(f"   - Average duration: {final_stats.get('avg_duration', 0):.3f}s")
        print(f"   - Average peak memory: {final_stats.get('avg_peak_memory', 0):.1f}MB")

    print(f"\n[TIMEOUT] Timeout Statistics:")
    print(f"   - Total timeouts: {timeout_stats.get('total_timeouts', 0)}")
    if timeout_stats.get('total_timeouts', 0) > 0:
        print(f"   - Average timeout ratio: {timeout_stats.get('avg_timeout_ratio', 0):.2f}")

def main():
    """Main demonstration function"""
    setup_logging()

    print("[TOOL] STEEL CUTTING OPTIMIZATION ENGINE DEMONSTRATION")
    print("   Steel cutting optimization engine demonstration")
    print()
    print("This demonstration shows the complete optimization engine with:")
    print("* Multiple algorithms (FFD, BFD, Hybrid)")
    print("* Material separation and grouping")
    print("* Timeout handling and error recovery")
    print("* Performance monitoring and analysis")
    print("* Production-ready error handling")

    try:
        # Run demonstrations
        demo_single_algorithm()
        demo_material_separation()
        demo_algorithm_comparison()
        demo_performance_monitoring()

        print("\n" + "="*70)
        print("[OK] DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nThe optimization engine is ready for production use!")
        print("Key features implemented:")
        print("* [OK] Complete OptimizationEngine.optimize() method")
        print("* [OK] Material grouping and processing")
        print("* [OK] Robust error handling and timeout management")
        print("* [OK] Enhanced algorithm selection logic")
        print("* [OK] Performance monitoring and resource tracking")
        print("* [OK] Support for multiple algorithms (FFD, BFD, Hybrid)")
        print("* [OK] Production-ready error recovery")

    except Exception as e:
        print(f"\n[FAIL] Demonstration failed: {e}")
        raise

if __name__ == "__main__":
    main()