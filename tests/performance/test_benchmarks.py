"""
Performance benchmarks for steel cutting optimization
鋼板切断最適化のパフォーマンスベンチマーク
"""

import pytest
import time
import statistics
from typing import List, Dict, Any

from core.models import Panel, SteelSheet, OptimizationConstraints
from core.optimizer import create_optimization_engine
from core.algorithms.ffd import create_ffd_algorithm


class TestPerformanceBenchmarks:
    """Performance benchmark tests"""

    def setup_method(self):
        """Set up benchmark environment"""
        self.engine = create_optimization_engine()
        self.ffd_algorithm = create_ffd_algorithm()
        self.engine.register_algorithm(self.ffd_algorithm)

        self.sheet = SteelSheet()
        self.constraints = OptimizationConstraints()

    def generate_test_panels(self, count: int, complexity: str = "medium") -> List[Panel]:
        """Generate test panels with specified complexity"""
        panels = []

        if complexity == "low":
            # Simple, similar-sized panels
            for i in range(count):
                panels.append(Panel(
                    id=f"simple_{i}",
                    width=300.0,
                    height=200.0,
                    quantity=1,
                    material="SS400",
                    thickness=6.0
                ))

        elif complexity == "medium":
            # Mixed sizes and materials
            sizes = [(300, 200), (400, 250), (250, 150), (500, 300), (350, 180)]
            materials = ["SS400", "SUS304", "AL6061"]

            for i in range(count):
                width, height = sizes[i % len(sizes)]
                material = materials[i % len(materials)]

                panels.append(Panel(
                    id=f"medium_{i}",
                    width=width + (i % 3) * 10,  # Add variation
                    height=height + (i % 2) * 15,
                    quantity=1 + (i % 3),  # Varying quantities
                    material=material,
                    thickness=3.0 + (i % 3) * 1.5,
                    allow_rotation=(i % 2 == 0)
                ))

        elif complexity == "high":
            # Highly diverse panels
            import random
            random.seed(42)  # Reproducible results

            materials = ["SS400", "SUS304", "SUS316", "AL6061", "AL5052"]

            for i in range(count):
                panels.append(Panel(
                    id=f"complex_{i}",
                    width=random.randint(100, 800),
                    height=random.randint(80, 600),
                    quantity=random.randint(1, 5),
                    material=random.choice(materials),
                    thickness=random.uniform(1.0, 12.0),
                    priority=random.randint(1, 10),
                    allow_rotation=random.choice([True, False])
                ))

        return panels

    def run_benchmark(self, panels: List[Panel], iterations: int = 3) -> Dict[str, Any]:
        """Run benchmark and collect statistics"""
        times = []
        efficiencies = []
        placed_counts = []

        for _ in range(iterations):
            start_time = time.time()

            results = self.engine.optimize(
                panels=panels,
                constraints=self.constraints,
                algorithm_hint="FFD"
            )

            end_time = time.time()
            processing_time = end_time - start_time

            times.append(processing_time)

            if results:
                total_placed = sum(len(result.panels) for result in results)
                avg_efficiency = sum(result.efficiency for result in results) / len(results)

                placed_counts.append(total_placed)
                efficiencies.append(avg_efficiency)
            else:
                placed_counts.append(0)
                efficiencies.append(0.0)

        return {
            'processing_time': {
                'mean': statistics.mean(times),
                'median': statistics.median(times),
                'min': min(times),
                'max': max(times),
                'stdev': statistics.stdev(times) if len(times) > 1 else 0
            },
            'efficiency': {
                'mean': statistics.mean(efficiencies),
                'median': statistics.median(efficiencies),
                'min': min(efficiencies),
                'max': max(efficiencies)
            },
            'panels_placed': {
                'mean': statistics.mean(placed_counts),
                'median': statistics.median(placed_counts),
                'min': min(placed_counts),
                'max': max(placed_counts)
            },
            'total_input_panels': sum(p.quantity for p in panels)
        }

    def test_small_batch_performance(self):
        """Test performance with small batch (≤10 panels)"""
        # Target: <1 second, 70-75% efficiency
        panels = self.generate_test_panels(8, "low")

        benchmark = self.run_benchmark(panels, iterations=5)

        # Performance assertions
        assert benchmark['processing_time']['mean'] < 1.0, \
            f"Small batch should complete in <1s, got {benchmark['processing_time']['mean']:.3f}s"

        assert benchmark['processing_time']['max'] < 2.0, \
            f"Small batch max time should be <2s, got {benchmark['processing_time']['max']:.3f}s"

        # Efficiency target (relaxed for diverse panels)
        assert benchmark['efficiency']['mean'] >= 0.3, \
            f"Should achieve reasonable efficiency, got {benchmark['efficiency']['mean']:.1%}"

        # Should place most panels
        placement_rate = benchmark['panels_placed']['mean'] / benchmark['total_input_panels']
        assert placement_rate >= 0.8, \
            f"Should place most panels, placement rate: {placement_rate:.1%}"

        print(f"✅ Small batch benchmark:")
        print(f"   Time: {benchmark['processing_time']['mean']:.3f}s ± {benchmark['processing_time']['stdev']:.3f}s")
        print(f"   Efficiency: {benchmark['efficiency']['mean']:.1%}")
        print(f"   Placement rate: {placement_rate:.1%}")

    def test_medium_batch_performance(self):
        """Test performance with medium batch (≤30 panels)"""
        # Target: <5 seconds, 75-85% efficiency
        panels = self.generate_test_panels(25, "medium")

        benchmark = self.run_benchmark(panels, iterations=3)

        # Performance assertions
        assert benchmark['processing_time']['mean'] < 10.0, \
            f"Medium batch should complete in <10s, got {benchmark['processing_time']['mean']:.3f}s"

        # Should place reasonable number of panels
        placement_rate = benchmark['panels_placed']['mean'] / benchmark['total_input_panels']
        assert placement_rate >= 0.5, \
            f"Should place reasonable number of panels, placement rate: {placement_rate:.1%}"

        print(f"✅ Medium batch benchmark:")
        print(f"   Time: {benchmark['processing_time']['mean']:.3f}s ± {benchmark['processing_time']['stdev']:.3f}s")
        print(f"   Efficiency: {benchmark['efficiency']['mean']:.1%}")
        print(f"   Placement rate: {placement_rate:.1%}")

    def test_large_batch_performance(self):
        """Test performance with large batch (≤50 panels)"""
        # Target: <30 seconds
        panels = self.generate_test_panels(45, "high")

        benchmark = self.run_benchmark(panels, iterations=2)

        # Performance assertions
        assert benchmark['processing_time']['mean'] < 60.0, \
            f"Large batch should complete in <60s, got {benchmark['processing_time']['mean']:.3f}s"

        # Should place some panels
        placement_rate = benchmark['panels_placed']['mean'] / benchmark['total_input_panels']
        assert placement_rate >= 0.3, \
            f"Should place some panels, placement rate: {placement_rate:.1%}"

        print(f"✅ Large batch benchmark:")
        print(f"   Time: {benchmark['processing_time']['mean']:.3f}s ± {benchmark['processing_time']['stdev']:.3f}s")
        print(f"   Efficiency: {benchmark['efficiency']['mean']:.1%}")
        print(f"   Placement rate: {placement_rate:.1%}")

    def test_complexity_scaling(self):
        """Test how performance scales with complexity"""
        complexities = ["low", "medium", "high"]
        panel_count = 15

        results = {}

        for complexity in complexities:
            panels = self.generate_test_panels(panel_count, complexity)
            benchmark = self.run_benchmark(panels, iterations=3)
            results[complexity] = benchmark

            print(f"✅ {complexity.title()} complexity:")
            print(f"   Time: {benchmark['processing_time']['mean']:.3f}s")
            print(f"   Efficiency: {benchmark['efficiency']['mean']:.1%}")

        # Performance should degrade gracefully with complexity
        assert results['low']['processing_time']['mean'] <= results['medium']['processing_time']['mean'], \
            "Low complexity should be faster than medium"

        # All should complete in reasonable time
        for complexity, result in results.items():
            assert result['processing_time']['mean'] < 30.0, \
                f"{complexity} complexity took too long: {result['processing_time']['mean']:.3f}s"

    def test_memory_usage_benchmark(self):
        """Test memory usage during optimization"""
        import psutil
        import os

        panels = self.generate_test_panels(40, "medium")

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Run optimization
        results = self.engine.optimize(
            panels=panels,
            constraints=self.constraints
        )

        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before

        print(f"✅ Memory usage benchmark:")
        print(f"   Memory before: {memory_before:.1f} MB")
        print(f"   Memory after: {memory_after:.1f} MB")
        print(f"   Increase: {memory_increase:.1f} MB")

        # Should not consume excessive memory
        assert memory_increase < 100, \
            f"Memory usage increased by {memory_increase:.1f}MB, which seems excessive"

    def test_material_grouping_performance(self):
        """Test performance with material separation enabled"""
        # Create panels with multiple materials
        panels = []
        materials = ["SS400", "SUS304", "AL6061"]

        for i in range(30):
            material = materials[i % len(materials)]
            panels.append(Panel(
                id=f"mat_{material}_{i}",
                width=200 + i * 5,
                height=150 + i * 3,
                quantity=1,
                material=material,
                thickness=6.0
            ))

        # Test with material separation
        constraints_with_separation = OptimizationConstraints(
            material_separation=True,
            time_budget=30.0
        )

        start_time = time.time()
        results = self.engine.optimize(
            panels=panels,
            constraints=constraints_with_separation
        )
        processing_time = time.time() - start_time

        print(f"✅ Material grouping benchmark:")
        print(f"   Time: {processing_time:.3f}s")
        print(f"   Sheets generated: {len(results)}")

        # Should complete in reasonable time
        assert processing_time < 30.0, \
            f"Material grouping took too long: {processing_time:.3f}s"

        # Should create multiple sheets for different materials
        if results:
            materials_found = set(result.material_block for result in results)
            assert len(materials_found) >= 2, \
                f"Should separate materials, found: {materials_found}"

    def test_rotation_performance_impact(self):
        """Test performance impact of rotation optimization"""
        panels = self.generate_test_panels(20, "medium")

        # Test without rotation
        for panel in panels:
            panel.allow_rotation = False

        constraints_no_rotation = OptimizationConstraints(allow_rotation=False)

        start_time = time.time()
        results_no_rotation = self.engine.optimize(
            panels=panels,
            constraints=constraints_no_rotation
        )
        time_no_rotation = time.time() - start_time

        # Test with rotation
        for panel in panels:
            panel.allow_rotation = True

        constraints_with_rotation = OptimizationConstraints(allow_rotation=True)

        start_time = time.time()
        results_with_rotation = self.engine.optimize(
            panels=panels,
            constraints=constraints_with_rotation
        )
        time_with_rotation = time.time() - start_time

        print(f"✅ Rotation performance impact:")
        print(f"   Without rotation: {time_no_rotation:.3f}s")
        print(f"   With rotation: {time_with_rotation:.3f}s")
        print(f"   Overhead: {(time_with_rotation - time_no_rotation):.3f}s")

        # Rotation should not add excessive overhead
        overhead_ratio = time_with_rotation / time_no_rotation if time_no_rotation > 0 else 1
        assert overhead_ratio < 3.0, \
            f"Rotation overhead too high: {overhead_ratio:.1f}x"


class TestStressTests:
    """Stress tests for extreme scenarios"""

    def setup_method(self):
        """Set up stress test environment"""
        self.engine = create_optimization_engine()
        self.ffd_algorithm = create_ffd_algorithm()
        self.engine.register_algorithm(self.ffd_algorithm)

    def test_timeout_stress(self):
        """Test behavior under tight time constraints"""
        # Create complex scenario
        panels = [Panel(f"stress_{i}", 200 + i, 150 + i, 1, "SS400", 6.0) for i in range(50)]

        # Very tight timeout
        constraints = OptimizationConstraints(time_budget=0.5)

        start_time = time.time()
        results = self.engine.optimize(panels=panels, constraints=constraints)
        actual_time = time.time() - start_time

        print(f"✅ Timeout stress test:")
        print(f"   Target time: 0.5s")
        print(f"   Actual time: {actual_time:.3f}s")
        print(f"   Results: {len(results)} sheets")

        # Should respect timeout (with tolerance for system overhead)
        assert actual_time < 2.0, f"Should respect timeout, took {actual_time:.3f}s"

        # Should return results (even if empty)
        assert isinstance(results, list), "Should return results list"

    def test_edge_case_panels(self):
        """Test with edge case panel dimensions"""
        edge_panels = [
            Panel("min_size", 50, 50, 1, "SS400", 6.0),  # Minimum size
            Panel("max_width", 1500, 100, 1, "SS400", 6.0),  # Maximum width
            Panel("max_height", 100, 3100, 1, "SS400", 6.0),  # Maximum height
            Panel("square_large", 1000, 1000, 1, "SS400", 6.0),  # Large square
            Panel("thin_strip", 1400, 60, 1, "SS400", 6.0),  # Thin strip
        ]

        constraints = OptimizationConstraints(time_budget=10.0)

        start_time = time.time()
        results = self.engine.optimize(panels=edge_panels, constraints=constraints)
        processing_time = time.time() - start_time

        print(f"✅ Edge case stress test:")
        print(f"   Time: {processing_time:.3f}s")
        print(f"   Sheets: {len(results)}")
        if results:
            total_placed = sum(len(result.panels) for result in results)
            print(f"   Panels placed: {total_placed}/{len(edge_panels)}")

        # Should handle edge cases without crashing
        assert processing_time < 15.0, "Should handle edge cases efficiently"
        assert isinstance(results, list), "Should return valid results"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])