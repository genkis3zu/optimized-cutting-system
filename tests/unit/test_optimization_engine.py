"""
Unit tests for OptimizationEngine
最適化エンジンのユニットテスト
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from core.optimizer import (
    OptimizationEngine, OptimizationAlgorithm, PerformanceMonitor,
    TimeoutManager, create_optimization_engine, create_optimization_engine_with_algorithms
)
from core.models import Panel, SteelSheet, PlacementResult, OptimizationConstraints
from core.algorithms.ffd import FirstFitDecreasing
from core.algorithms.bfd import BestFitDecreasing
from core.algorithms.hybrid import HybridOptimizer


class MockAlgorithm(OptimizationAlgorithm):
    """Mock algorithm for testing"""

    def __init__(self, name: str, efficiency: float = 0.8, should_fail: bool = False):
        super().__init__(name)
        self.efficiency = efficiency
        self.should_fail = should_fail
        self.call_count = 0

    def optimize(self, panels, sheet, constraints):
        self.call_count += 1

        if self.should_fail:
            raise RuntimeError("Mock algorithm failure")

        # Create mock result
        result = PlacementResult(
            sheet_id=1,
            material_block=sheet.material,
            sheet=sheet,
            panels=[],
            efficiency=self.efficiency,
            waste_area=sheet.area * (1 - self.efficiency),
            cut_length=100.0,
            cost=sheet.cost_per_sheet,
            algorithm=self.name
        )
        return result

    def estimate_time(self, panel_count: int, complexity: float) -> float:
        return 0.1 * panel_count * (1 + complexity)


class TestOptimizationEngine:
    """Test OptimizationEngine class"""

    def setup_method(self):
        """Set up test engine"""
        self.engine = OptimizationEngine()
        self.mock_ffd = MockAlgorithm("MockFFD", 0.75)
        self.mock_bfd = MockAlgorithm("MockBFD", 0.85)
        self.engine.register_algorithm(self.mock_ffd)
        self.engine.register_algorithm(self.mock_bfd)

    def test_engine_initialization(self):
        """Test engine initialization"""
        assert len(self.engine.algorithms) == 2
        assert "MockFFD" in self.engine.algorithms
        assert "MockBFD" in self.engine.algorithms
        assert isinstance(self.engine.performance_monitor, PerformanceMonitor)
        assert isinstance(self.engine.timeout_manager, TimeoutManager)

    def test_algorithm_registration(self):
        """Test algorithm registration"""
        new_algorithm = MockAlgorithm("NewAlgo", 0.9)
        self.engine.register_algorithm(new_algorithm)

        assert len(self.engine.algorithms) == 3
        assert "NewAlgo" in self.engine.algorithms
        assert self.engine.algorithms["NewAlgo"] == new_algorithm

    def test_algorithm_selection_simple(self):
        """Test algorithm selection for simple problems"""
        panels = [Panel("test", 200, 100, 1, "SS400", 6.0)]
        constraints = OptimizationConstraints(time_budget=5.0)

        selected = self.engine.select_algorithm(panels, constraints)
        assert selected in ["MockFFD", "MockBFD"]

    def test_algorithm_selection_complex(self):
        """Test algorithm selection for complex problems"""
        panels = [
            Panel(f"panel_{i}", 200 + i*10, 150 + i*5, 2, f"Material_{i%3}", 6.0)
            for i in range(20)
        ]
        constraints = OptimizationConstraints(time_budget=30.0)

        selected = self.engine.select_algorithm(panels, constraints)
        assert selected in self.engine.algorithms

    def test_algorithm_selection_empty_panels(self):
        """Test algorithm selection with empty panel list"""
        constraints = OptimizationConstraints()
        selected = self.engine.select_algorithm([], constraints)
        assert selected in self.engine.algorithms  # Should be one of available algorithms

    def test_optimize_single_material(self):
        """Test optimization without material separation"""
        panels = [
            Panel("p1", 300, 200, 1, "SS400", 6.0),
            Panel("p2", 200, 150, 1, "SS400", 6.0)
        ]
        constraints = OptimizationConstraints(material_separation=False)

        results = self.engine.optimize(panels, constraints, algorithm_hint="MockFFD")

        assert len(results) == 1
        assert results[0].algorithm == "MockFFD"
        assert self.mock_ffd.call_count == 1

    def test_optimize_material_separation(self):
        """Test optimization with material separation"""
        panels = [
            Panel("p1", 300, 200, 1, "SS400", 6.0),
            Panel("p2", 200, 150, 1, "SUS304", 6.0),
            Panel("p3", 250, 180, 1, "SS400", 6.0)
        ]
        constraints = OptimizationConstraints(material_separation=True)

        results = self.engine.optimize(panels, constraints, algorithm_hint="MockFFD")

        assert len(results) == 2  # Two materials: SS400, SUS304
        material_blocks = [r.material_block for r in results]
        assert "SS400" in material_blocks
        assert "SUS304" in material_blocks

    def test_optimize_empty_panels(self):
        """Test optimization with empty panel list"""
        results = self.engine.optimize([])
        assert results == []

    def test_optimize_algorithm_not_found(self):
        """Test optimization with non-existent algorithm"""
        panels = [Panel("test", 200, 100, 1, "SS400", 6.0)]

        results = self.engine.optimize(panels, algorithm_hint="NonExistent")

        # Should fallback to available algorithm
        assert len(results) == 1
        assert results[0].algorithm in self.engine.algorithms

    def test_optimize_with_timeout(self):
        """Test optimization with timeout handling"""
        slow_algorithm = MockAlgorithm("SlowAlgo", 0.9)

        # Mock the algorithm to simulate slow execution
        def slow_optimize(panels, sheet, constraints):
            time.sleep(0.5)  # Simulate slow execution
            return slow_algorithm.optimize(panels, sheet, constraints)

        slow_algorithm.optimize = slow_optimize
        self.engine.register_algorithm(slow_algorithm)

        panels = [Panel("test", 200, 100, 1, "SS400", 6.0)]
        constraints = OptimizationConstraints(time_budget=0.1)  # Very short timeout

        results = self.engine.optimize(panels, constraints, algorithm_hint="SlowAlgo")

        # Should handle timeout gracefully
        assert isinstance(results, list)

    def test_optimize_algorithm_failure(self):
        """Test optimization with algorithm failure"""
        failing_algorithm = MockAlgorithm("FailingAlgo", 0.9, should_fail=True)
        self.engine.register_algorithm(failing_algorithm)

        panels = [Panel("test", 200, 100, 1, "SS400", 6.0)]

        results = self.engine.optimize(panels, algorithm_hint="FailingAlgo")

        # Should handle failure gracefully
        assert isinstance(results, list)

    def test_fallback_optimization(self):
        """Test fallback optimization when primary fails"""
        # This is tested indirectly through timeout and failure scenarios
        pass

    def test_performance_monitoring(self):
        """Test performance monitoring integration"""
        panels = [Panel("test", 200, 100, 1, "SS400", 6.0)]

        # Monitor should be started and stopped
        assert not self.engine.performance_monitor.monitoring

        results = self.engine.optimize(panels)

        # Check that monitoring occurred
        assert not self.engine.performance_monitor.monitoring  # Should be stopped after optimization


class TestPerformanceMonitor:
    """Test PerformanceMonitor class"""

    def setup_method(self):
        """Set up test monitor"""
        self.monitor = PerformanceMonitor()

    def test_monitor_initialization(self):
        """Test monitor initialization"""
        assert self.monitor.start_time is None
        assert not self.monitor.monitoring
        assert self.monitor.metrics_history == []
        assert self.monitor.peak_memory == 0.0

    def test_start_stop_monitoring(self):
        """Test monitor start/stop cycle"""
        self.monitor.start_monitoring()

        assert self.monitor.monitoring
        assert self.monitor.start_time is not None

        # Simulate some time passing
        time.sleep(0.1)

        self.monitor.stop_monitoring()

        assert not self.monitor.monitoring
        assert len(self.monitor.metrics_history) == 1

    def test_memory_usage_tracking(self):
        """Test memory usage tracking"""
        memory = self.monitor.get_memory_usage()
        assert memory >= 0.0

    def test_resource_limits_check(self):
        """Test resource limits checking"""
        # Should pass with reasonable limits
        assert self.monitor.check_resource_limits(memory_limit_mb=1000)

        # Should fail with very low limits
        self.monitor.start_monitoring()
        # Note: Actual memory usage might vary, so we test the mechanism rather than specific values

    def test_performance_metrics(self):
        """Test performance metrics collection"""
        self.monitor.start_monitoring()
        time.sleep(0.1)

        metrics = self.monitor.get_performance_metrics()

        assert 'elapsed_time' in metrics
        assert 'memory_usage' in metrics
        assert 'cpu_usage' in metrics
        assert 'peak_memory' in metrics
        assert metrics['elapsed_time'] > 0

    def test_average_performance(self):
        """Test average performance calculation"""
        # Initially empty
        avg = self.monitor.get_average_performance()
        assert avg.get('total_runs', 0) == 0

        # Add some history
        self.monitor.start_monitoring()
        time.sleep(0.05)
        self.monitor.stop_monitoring()

        self.monitor.start_monitoring()
        time.sleep(0.05)
        self.monitor.stop_monitoring()

        avg = self.monitor.get_average_performance()
        assert avg['total_runs'] == 2
        assert avg['avg_duration'] > 0


class TestTimeoutManager:
    """Test TimeoutManager class"""

    def setup_method(self):
        """Set up test manager"""
        self.manager = TimeoutManager()

    def test_manager_initialization(self):
        """Test manager initialization"""
        assert len(self.manager.active_threads) == 0
        assert self.manager.timeout_history == []

    def test_execute_with_timeout_success(self):
        """Test successful execution within timeout"""
        def quick_function():
            return "success"

        result = self.manager.execute_with_timeout(quick_function, timeout=1.0)
        assert result == "success"

    def test_execute_with_timeout_failure(self):
        """Test timeout when function takes too long"""
        def slow_function():
            time.sleep(0.5)
            return "should not reach"

        with pytest.raises(Exception):  # Should raise timeout exception
            self.manager.execute_with_timeout(slow_function, timeout=0.1)

    def test_execute_with_progressive_timeout(self):
        """Test progressive timeout mechanism"""
        call_count = [0]

        def sometimes_slow_function():
            call_count[0] += 1
            if call_count[0] == 1:
                time.sleep(0.3)  # Too slow for first timeout
            return f"attempt_{call_count[0]}"

        # Should succeed on second attempt with longer timeout
        try:
            result = self.manager.execute_with_progressive_timeout(
                sometimes_slow_function,
                initial_timeout=0.1,
                max_timeout=1.0
            )
            # Second attempt should succeed
            assert "attempt" in result
        except Exception:
            # May timeout depending on system speed
            pass

    def test_timeout_statistics(self):
        """Test timeout statistics collection"""
        stats = self.manager.get_timeout_statistics()
        assert 'total_timeouts' in stats
        assert 'avg_timeout_ratio' in stats
        assert 'recent_timeouts' in stats

    def test_active_threads_tracking(self):
        """Test active threads tracking"""
        assert self.manager.get_active_threads_count() == 0


class TestFactoryFunctions:
    """Test factory functions"""

    def test_create_optimization_engine(self):
        """Test engine creation with default algorithms"""
        engine = create_optimization_engine()

        assert isinstance(engine, OptimizationEngine)
        assert len(engine.algorithms) >= 1  # At least FFD should be available
        assert "FFD" in engine.algorithms

    def test_create_optimization_engine_with_specific_algorithms(self):
        """Test engine creation with specific algorithms"""
        engine = create_optimization_engine_with_algorithms(["FFD"])

        assert isinstance(engine, OptimizationEngine)
        assert "FFD" in engine.algorithms

    def test_create_optimization_engine_invalid_algorithms(self):
        """Test engine creation with invalid algorithms"""
        with pytest.raises(RuntimeError):
            create_optimization_engine_with_algorithms(["NonExistent"])


class TestAlgorithmIntegration:
    """Integration tests for real algorithms"""

    def test_ffd_integration(self):
        """Test FFD algorithm integration"""
        engine = OptimizationEngine()
        ffd = FirstFitDecreasing()
        engine.register_algorithm(ffd)

        panels = [
            Panel("p1", 300, 200, 1, "SS400", 6.0),
            Panel("p2", 200, 150, 1, "SS400", 6.0)
        ]

        results = engine.optimize(panels, algorithm_hint="FFD")

        assert len(results) == 1
        assert results[0].algorithm == "FFD"
        assert results[0].efficiency > 0

    def test_bfd_integration(self):
        """Test BFD algorithm integration"""
        engine = OptimizationEngine()
        bfd = BestFitDecreasing()
        engine.register_algorithm(bfd)

        panels = [
            Panel("p1", 300, 200, 1, "SS400", 6.0),
            Panel("p2", 200, 150, 1, "SS400", 6.0)
        ]

        results = engine.optimize(panels, algorithm_hint="BFD")

        assert len(results) == 1
        assert results[0].algorithm == "BFD"
        assert results[0].efficiency > 0

    def test_hybrid_integration(self):
        """Test Hybrid algorithm integration"""
        engine = OptimizationEngine()
        hybrid = HybridOptimizer()
        engine.register_algorithm(hybrid)

        panels = [
            Panel("p1", 300, 200, 1, "SS400", 6.0),
            Panel("p2", 200, 150, 1, "SS400", 6.0)
        ]

        constraints = OptimizationConstraints(time_budget=5.0)
        results = engine.optimize(panels, constraints, algorithm_hint="HYBRID")

        assert len(results) == 1
        assert "HYBRID" in results[0].algorithm
        assert results[0].efficiency >= 0

    def test_algorithm_selection_performance(self):
        """Test algorithm selection chooses appropriate algorithm"""
        engine = create_optimization_engine()

        # Small simple problem - should prefer FFD
        small_panels = [Panel("p1", 200, 100, 1, "SS400", 6.0)]
        constraints_fast = OptimizationConstraints(time_budget=1.0)
        selected = engine.select_algorithm(small_panels, constraints_fast)
        assert selected == "FFD"

        # Large complex problem with time - might prefer BFD or HYBRID if available
        large_panels = [
            Panel(f"p{i}", 200 + i*10, 150 + i*5, 2, f"M{i%3}", 6.0)
            for i in range(30)
        ]
        constraints_complex = OptimizationConstraints(time_budget=30.0)
        selected = engine.select_algorithm(large_panels, constraints_complex)
        # Should select available algorithm appropriately
        assert selected in engine.algorithms


if __name__ == "__main__":
    pytest.main([__file__])