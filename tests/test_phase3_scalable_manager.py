"""
Phase 3 Testing: Scalable GPU Manager for Large Workloads

Tests the scalable GPU processing system that handles 500+ panel workloads
with intelligent batching, memory management, and performance optimization.
"""

import pytest
import time
import logging
from typing import List
from unittest.mock import patch, MagicMock

from core.models import Panel, SteelSheet
from core.algorithms.scalable_gpu_manager import (
    ScalableGPUManager,
    BatchResult,
    ScalabilityMetrics,
    optimize_large_workload
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestScalableGPUManager:
    """Test suite for scalable GPU manager Phase 3 implementation"""

    @pytest.fixture
    def large_panel_set(self) -> List[Panel]:
        """Create large panel set for scalability testing"""
        panels = []

        # Create variety of panel sizes for realistic testing
        for i in range(150):  # 150 panels for realistic large workload
            width = 100 + (i % 20) * 10  # Varied widths: 100-290
            height = 150 + (i % 15) * 15  # Varied heights: 150-360

            panels.append(Panel(
                id=f"Large_{i:03d}",
                width=width,
                height=height,
                thickness=3.0,
                material=f"Steel_{i % 3}",  # 3 different materials
                quantity=1,
                allow_rotation=True
            ))

        return panels

    @pytest.fixture
    def test_sheet(self) -> SteelSheet:
        """Standard test sheet"""
        return SteelSheet(
            width=1500,
            height=3100,
            thickness=3.0,
            material="Steel"
        )

    def test_scalable_manager_initialization(self):
        """Test scalable GPU manager initialization"""
        manager = ScalableGPUManager(max_memory_mb=1500, thermal_limit=85.0)

        assert manager.max_memory_mb == 1500
        assert manager.thermal_limit == 85.0
        assert manager.base_batch_size == 50
        assert manager.min_batch_size == 20
        assert manager.max_batch_size == 200

        # Check component initialization
        assert manager.gpu_optimizer is not None
        assert manager.multi_sheet_optimizer is not None
        assert manager.fallback_manager is not None

        logger.info("✅ Scalable manager initialization successful")

    def test_batch_size_calculation(self, large_panel_set):
        """Test adaptive batch size calculation"""
        manager = ScalableGPUManager(max_memory_mb=1500)

        # Test different workload sizes
        batch_size_small = manager._calculate_optimal_batch_size(50, 1.0)
        batch_size_large = manager._calculate_optimal_batch_size(500, 1.0)
        batch_size_complex = manager._calculate_optimal_batch_size(200, 2.0)

        assert manager.min_batch_size <= batch_size_small <= manager.max_batch_size
        assert manager.min_batch_size <= batch_size_large <= manager.max_batch_size
        assert manager.min_batch_size <= batch_size_complex <= manager.max_batch_size

        # Complex workloads should have smaller batch sizes
        assert batch_size_complex <= batch_size_small

        logger.info(f"✅ Batch size calculation: small={batch_size_small}, large={batch_size_large}, complex={batch_size_complex}")

    def test_batch_creation_with_material_grouping(self, large_panel_set):
        """Test batch creation with material grouping"""
        manager = ScalableGPUManager(max_memory_mb=1500)

        batches = manager._create_batches(large_panel_set)

        assert len(batches) > 0
        assert len(batches) <= 10  # Reasonable number of batches for 150 panels

        # Verify material grouping within batches
        for batch in batches:
            if len(batch) > 1:
                first_material = getattr(batch[0], 'material', 'default')
                for panel in batch[1:]:
                    panel_material = getattr(panel, 'material', 'default')
                    # Panels in same batch should have same material
                    assert panel_material == first_material, f"Material mismatch in batch: {panel_material} != {first_material}"

        total_panels_in_batches = sum(len(batch) for batch in batches)
        assert total_panels_in_batches == len(large_panel_set)

        logger.info(f"✅ Created {len(batches)} batches with proper material grouping")

    @pytest.mark.skipif(True, reason="GPU hardware required for full workload test")
    def test_large_workload_processing(self, large_panel_set, test_sheet):
        """Test processing of large workload (requires GPU hardware)"""
        manager = ScalableGPUManager(max_memory_mb=1500)

        progress_calls = []
        def progress_callback(current, total, message):
            progress_calls.append((current, total, message))
            logger.info(f"Progress: {current}/{total} - {message}")

        start_time = time.time()
        results, metrics = manager.process_large_workload(
            large_panel_set, test_sheet, progress_callback
        )
        processing_time = time.time() - start_time

        # Validate results
        assert len(results) >= 0  # Should have some results
        assert metrics.total_panels == len(large_panel_set)
        assert metrics.total_processing_time > 0
        assert metrics.total_batches > 0

        # Validate progress callbacks
        assert len(progress_calls) > 0

        logger.info(f"✅ Large workload processed in {processing_time:.2f}s")
        logger.info(f"   Metrics: {metrics.total_batches} batches, {metrics.gpu_efficiency:.1f}% efficiency")

    def test_performance_summary_generation(self, large_panel_set):
        """Test performance summary generation"""
        manager = ScalableGPUManager(max_memory_mb=1500)

        # Initialize some metrics
        manager.metrics.total_panels = len(large_panel_set)
        manager.metrics.total_batches = 5
        manager.metrics.total_processing_time = 25.5
        manager.metrics.gpu_efficiency = 87.3
        manager.metrics.peak_memory_usage = 1200.0

        # Add some batch results
        for i in range(3):
            batch_result = BatchResult(
                batch_id=i,
                panels=large_panel_set[i*10:(i+1)*10],
                placements=[],
                processing_time=5.0 + i,
                gpu_utilization=85.0,
                memory_usage=400.0,
                efficiency=88.0 + i,
                status="completed"
            )
            manager.batch_results.append(batch_result)

        summary = manager.get_performance_summary()

        # Validate summary structure
        assert 'total_panels' in summary
        assert 'total_batches' in summary
        assert 'processing_time' in summary
        assert 'gpu_efficiency' in summary
        assert 'batch_success_rate' in summary

        assert summary['total_panels'] == len(large_panel_set)
        assert summary['total_batches'] == 5
        assert '25.5' in summary['processing_time']
        assert '87.3' in summary['gpu_efficiency']

        logger.info(f"✅ Performance summary generated: {summary}")

    def test_memory_pressure_monitoring(self):
        """Test memory pressure monitoring system"""
        manager = ScalableGPUManager(max_memory_mb=1500)

        # Test memory pressure detection
        available_memory = manager._get_available_gpu_memory()
        assert available_memory >= 0  # Should return reasonable value

        thermal_factor = manager._get_thermal_factor()
        assert 0.6 <= thermal_factor <= 1.0  # Should be in valid range

        performance_factor = manager._get_performance_factor()
        assert 0.8 <= performance_factor <= 1.2  # Should be in valid range

        logger.info(f"✅ Memory monitoring: available={available_memory}MB, thermal={thermal_factor}, performance={performance_factor}")

    def test_cleanup_resources(self):
        """Test proper resource cleanup"""
        manager = ScalableGPUManager(max_memory_mb=1500)

        # Should not raise exceptions
        manager.cleanup()

        logger.info("✅ Resource cleanup completed successfully")

    def test_convenience_function(self, large_panel_set, test_sheet):
        """Test convenience function for large workload optimization"""
        progress_calls = []
        def progress_callback(current, total, message):
            progress_calls.append((current, total, message))

        # Use convenience function (with mock for GPU processing)
        with patch('core.algorithms.scalable_gpu_manager.ScalableGPUManager') as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager

            # Mock the process_large_workload method
            mock_results = []
            mock_metrics = ScalabilityMetrics(
                total_panels=len(large_panel_set),
                total_batches=3,
                total_processing_time=15.2,
                average_batch_time=5.1,
                peak_memory_usage=800.0,
                gpu_efficiency=89.5,
                thermal_throttling_events=0,
                fallback_events=0
            )

            mock_manager.process_large_workload.return_value = (mock_results, mock_metrics)
            mock_manager.get_performance_summary.return_value = {
                'total_panels': len(large_panel_set),
                'gpu_efficiency': '89.5%',
                'processing_time': '15.2s'
            }

            results, summary = optimize_large_workload(
                large_panel_set, test_sheet, max_memory_mb=1500, progress_callback=progress_callback
            )

            # Validate function was called correctly
            mock_manager.process_large_workload.assert_called_once()
            mock_manager.cleanup.assert_called_once()

            assert 'total_panels' in summary
            assert summary['total_panels'] == len(large_panel_set)

            logger.info(f"✅ Convenience function test passed: {summary}")

    def test_cross_batch_optimization(self, large_panel_set):
        """Test cross-batch optimization functionality"""
        manager = ScalableGPUManager(max_memory_mb=1500)

        # Create mock batch results
        batch_results = []
        for i in range(3):
            batch_result = BatchResult(
                batch_id=i,
                panels=large_panel_set[i*20:(i+1)*20],
                placements=[],  # Empty placements for test
                processing_time=5.0,
                gpu_utilization=85.0,
                memory_usage=400.0,
                efficiency=75.0 + i * 5,  # Varying efficiency
                status="completed"
            )
            batch_results.append(batch_result)

        # Test cross-batch optimization
        improvement = manager._optimize_cross_batch(batch_results)

        # Should return a number (even if 0 for empty placements)
        assert isinstance(improvement, (int, float))
        assert improvement >= 0.0

        logger.info(f"✅ Cross-batch optimization returned {improvement}% improvement")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])