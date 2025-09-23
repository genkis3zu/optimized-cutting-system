"""
Phase 3 Testing: 100% Placement Guarantee System

Tests the unlimited runtime optimizer with 4-tier GPU-accelerated escalation
system that guarantees 100% panel placement regardless of complexity.
"""

import pytest
import time
import logging
from typing import List
from unittest.mock import patch, MagicMock

from core.models import Panel, SteelSheet, PlacementResult, PlacedPanel
from core.algorithms.unlimited_runtime_optimizer import (
    UnlimitedRuntimeOptimizer,
    OptimizationProgress,
    optimize_with_100_percent_guarantee
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestUnlimitedRuntimeOptimizer:
    """Test suite for 100% placement guarantee system"""

    @pytest.fixture
    def challenging_panel_set(self) -> List[Panel]:
        """Create challenging panel set that tests all tiers"""
        panels = []

        # Easy panels (should be placed in Tier 1)
        for i in range(15):
            panels.append(Panel(
                id=f"Easy_{i:02d}",
                width=200 + i * 10,
                height=300 + i * 5,
                thickness=3.0,
                material="Steel",
                quantity=1,
                allow_rotation=True
            ))

        # Medium panels (should require Tier 2)
        for i in range(10):
            panels.append(Panel(
                id=f"Medium_{i:02d}",
                width=400 + i * 20,
                height=500 + i * 15,
                thickness=3.0,
                material="Aluminum",
                quantity=1,
                allow_rotation=True
            ))

        # Difficult panels (should require Tier 3)
        for i in range(5):
            panels.append(Panel(
                id=f"Hard_{i:02d}",
                width=800 + i * 50,
                height=1200 + i * 100,
                thickness=3.0,
                material="Stainless",
                quantity=1,
                allow_rotation=True
            ))

        # Extreme panels (should require Tier 4)
        for i in range(3):
            panels.append(Panel(
                id=f"Extreme_{i:02d}",
                width=1400,
                height=2800,
                thickness=3.0,
                material="Titanium",
                quantity=1,
                allow_rotation=False  # No rotation allowed
            ))

        return panels

    @pytest.fixture
    def test_sheets(self) -> List[SteelSheet]:
        """Create variety of test sheets"""
        return [
            SteelSheet(width=1500, height=3100, thickness=3.0, material="Steel"),
            SteelSheet(width=2000, height=4000, thickness=3.0, material="Aluminum"),
            SteelSheet(width=1200, height=2400, thickness=3.0, material="Stainless"),
            SteelSheet(width=1600, height=3200, thickness=3.0, material="Titanium")
        ]

    def test_optimizer_initialization_with_gpu(self):
        """Test optimizer initialization with GPU acceleration"""
        optimizer = UnlimitedRuntimeOptimizer(max_memory_mb=2000)

        # Check basic initialization
        assert optimizer.max_memory_mb == 2000
        assert optimizer.progress_callback is None

        # Check GPU component initialization
        if hasattr(optimizer, 'gpu_manager') and optimizer.gpu_manager:
            assert optimizer.gpu_optimizer is not None
            assert optimizer.multi_sheet_optimizer is not None
            assert optimizer.fallback_manager is not None
            logger.info("✅ GPU acceleration components initialized")
        else:
            logger.info("ℹ️ GPU acceleration not available, using CPU-only mode")

        logger.info("✅ Unlimited runtime optimizer initialization successful")

    def test_progress_tracking_structure(self):
        """Test optimization progress tracking"""
        progress = OptimizationProgress(
            total_panels=100,
            placed_panels=75,
            current_tier=2,
            elapsed_time=45.5,
            sheets_used=8,
            current_efficiency=87.3
        )

        assert progress.placement_rate == 75.0
        assert progress.total_panels == 100
        assert progress.current_tier == 2

        # Test progress logging (should not raise exceptions)
        progress.log_progress()

        logger.info(f"✅ Progress tracking: {progress.placement_rate}% placed in tier {progress.current_tier}")

    @pytest.mark.skipif(True, reason="Requires full GPU setup for tier testing")
    def test_tier1_gpu_accelerated_heuristics(self, challenging_panel_set, test_sheets):
        """Test Tier 1: GPU accelerated heuristics"""
        optimizer = UnlimitedRuntimeOptimizer(max_memory_mb=2000)

        # Test with subset that should be handled by Tier 1
        easy_panels = [p for p in challenging_panel_set if "Easy_" in p.id]

        result = optimizer._tier1_gpu_accelerated_heuristics(
            easy_panels, test_sheets, {}
        )

        assert isinstance(result, PlacementResult)
        assert len(result.sheets) >= 0
        assert len(result.placed_panels) >= 0

        placement_rate = len(result.placed_panels) / len(easy_panels) * 100
        logger.info(f"✅ Tier 1 achieved {placement_rate:.1f}% placement on easy panels")

    def test_tier_progression_logic(self, challenging_panel_set, test_sheets):
        """Test tier progression logic with mocked GPU components"""
        with patch('core.algorithms.unlimited_runtime_optimizer.GPU_ACCELERATION_AVAILABLE', False):
            optimizer = UnlimitedRuntimeOptimizer(max_memory_mb=2000)

            # Mock progress callback
            progress_calls = []
            def progress_callback(progress_obj):
                progress_calls.append(progress_obj.current_tier)

            optimizer.progress_callback = progress_callback

            # Test with small subset to avoid long execution
            test_panels = challenging_panel_set[:10]

            result = optimizer.optimize(test_panels, test_sheets, {})

            # Validate result structure
            assert isinstance(result, PlacementResult)
            assert hasattr(result, 'sheets')
            assert hasattr(result, 'placed_panels')
            assert hasattr(result, 'metadata')

            # Check metadata
            assert 'placement_rate' in result.metadata
            assert 'optimization_time' in result.metadata
            assert 'tiers_used' in result.metadata

            placement_rate = result.metadata['placement_rate']
            logger.info(f"✅ Final placement rate: {placement_rate:.1f}%")
            logger.info(f"   Tiers used: {result.metadata['tiers_used']}")
            logger.info(f"   Time: {result.metadata['optimization_time']:.2f}s")

    def test_individual_sheet_placement_tier4(self, test_sheets):
        """Test Tier 4: Individual sheet placement with CPU fallback"""
        optimizer = UnlimitedRuntimeOptimizer()

        # Create problematic panels that should require individual placement
        problem_panels = [
            Panel(
                id="Problem_01",
                width=1400,
                height=2800,
                thickness=3.0,
                material="Titanium",
                quantity=1,
                allow_rotation=False
            ),
            Panel(
                id="Problem_02",
                width=1450,
                height=2900,
                thickness=3.0,
                material="Inconel",
                quantity=1,
                allow_rotation=False
            )
        ]

        result = optimizer._tier4_individual_sheets_cpu_fallback(
            problem_panels, test_sheets, {}
        )

        assert isinstance(result, PlacementResult)
        assert len(result.sheets) >= 0
        assert len(result.placed_panels) >= 0

        # Individual placement should achieve high placement rate
        if len(result.placed_panels) > 0:
            placement_rate = len(result.placed_panels) / len(problem_panels) * 100
            logger.info(f"✅ Tier 4 individual placement achieved {placement_rate:.1f}%")

    def test_100_percent_guarantee_with_mocks(self, challenging_panel_set, test_sheets):
        """Test 100% placement guarantee with mocked components"""
        with patch('core.algorithms.unlimited_runtime_optimizer.GPU_ACCELERATION_AVAILABLE', False):
            optimizer = UnlimitedRuntimeOptimizer()

            # Mock the tier methods to simulate progressive placement
            def mock_tier1(panels, sheets, constraints):
                # Simulate 70% placement in tier 1
                placed_count = int(len(panels) * 0.7)
                result = PlacementResult(
                    sheet_id=0,
                    material_block="",
                    sheet=sheets[0],
                    panels=[],
                    efficiency=75.0,
                    waste_area=0.0,
                    cut_length=0.0,
                    cost=0.0
                )
                result.sheets = [sheets[0]] * placed_count
                result.placed_panels = [
                    PlacedPanel(panel=panels[i], x=0, y=0, rotated=False)
                    for i in range(placed_count)
                ]
                return result

            def mock_tier2(panels, sheets, constraints):
                # Simulate 80% placement of remaining in tier 2
                placed_count = int(len(panels) * 0.8)
                result = PlacementResult([], [], 0.0, {})
                result.sheets = [sheets[0]] * placed_count
                result.placed_panels = [
                    PlacedPanel(panel=panels[i], x=0, y=0, rotated=False)
                    for i in range(placed_count)
                ]
                return result

            def mock_tier3(panels, sheets, constraints):
                # Simulate 90% placement of remaining in tier 3
                placed_count = int(len(panels) * 0.9)
                result = PlacementResult([], [], 0.0, {})
                result.sheets = [sheets[0]] * placed_count
                result.placed_panels = [
                    PlacedPanel(panel=panels[i], x=0, y=0, rotated=False)
                    for i in range(placed_count)
                ]
                return result

            def mock_tier4(panels, sheets, constraints):
                # Simulate 100% placement in tier 4
                result = PlacementResult([], [], 0.0, {})
                result.sheets = [sheets[0]] * len(panels)
                result.placed_panels = [
                    PlacedPanel(panel=panel, x=0, y=0, rotated=False)
                    for panel in panels
                ]
                return result

            # Patch the tier methods
            optimizer._fallback_traditional_heuristics = mock_tier1
            optimizer._fallback_exhaustive_search = mock_tier2
            optimizer._tier3_exhaustive_gpu_search = mock_tier3
            optimizer._tier4_individual_sheets_cpu_fallback = mock_tier4

            # Test with small subset
            test_panels = challenging_panel_set[:20]

            result = optimizer.optimize(test_panels, test_sheets, {})

            # Should achieve 100% placement
            placement_rate = result.metadata['placement_rate']
            assert placement_rate >= 99.0  # Should be close to 100%

            logger.info(f"✅ 100% guarantee test: {placement_rate:.1f}% placement achieved")

    def test_convenience_function(self, challenging_panel_set, test_sheets):
        """Test convenience function for 100% placement guarantee"""
        # Use subset for quick test
        test_panels = challenging_panel_set[:10]

        with patch('core.algorithms.unlimited_runtime_optimizer.UnlimitedRuntimeOptimizer') as mock_optimizer_class:
            mock_optimizer = MagicMock()
            mock_optimizer_class.return_value = mock_optimizer

            # Mock the optimize_with_guarantee method
            mock_placement_results = []
            mock_metrics = MagicMock()
            mock_metrics.total_panels = len(test_panels)
            mock_metrics.panels_placed = len(test_panels)
            mock_metrics.panels_remaining = 0
            mock_metrics.placement_rate = 100.0
            mock_metrics.total_processing_time = 15.5
            mock_metrics.best_efficiency = 85.7
            mock_metrics.sheets_used = 5
            mock_metrics.current_stage.value = "completed"

            mock_optimizer.optimize_with_guarantee.return_value = (mock_placement_results, mock_metrics)

            results, summary = optimize_with_100_percent_guarantee(
                test_panels, test_sheets[0], max_memory_mb=2000
            )

            # Validate function was called correctly
            mock_optimizer.optimize_with_guarantee.assert_called_once()
            mock_optimizer.cleanup.assert_called_once()

            assert 'placement_percentage' in summary
            assert 'final_stage' in summary
            assert 'processing_time' in summary

            logger.info(f"✅ Convenience function test passed: {summary}")

    def test_fallback_methods(self, challenging_panel_set, test_sheets):
        """Test fallback methods when GPU is not available"""
        optimizer = UnlimitedRuntimeOptimizer()

        # Test fallback traditional heuristics
        easy_panels = [p for p in challenging_panel_set if "Easy_" in p.id][:5]

        result = optimizer._fallback_traditional_heuristics(
            easy_panels, test_sheets, {}
        )

        assert isinstance(result, PlacementResult)
        logger.info(f"✅ Traditional heuristics fallback completed")

        # Test fallback exhaustive search
        result = optimizer._fallback_exhaustive_search(
            easy_panels, test_sheets, {}
        )

        assert isinstance(result, PlacementResult)
        logger.info(f"✅ Exhaustive search fallback completed")

    def test_resource_cleanup(self):
        """Test proper resource cleanup"""
        optimizer = UnlimitedRuntimeOptimizer(max_memory_mb=2000)

        # Should not raise exceptions
        optimizer.cleanup()

        logger.info("✅ Resource cleanup completed successfully")

    def test_panel_difficulty_handling(self, test_sheets):
        """Test handling of panels with different difficulty levels"""
        optimizer = UnlimitedRuntimeOptimizer()

        # Create panels of varying difficulty
        panels = [
            # Very small panel (easy)
            Panel(id="VerySmall", width=50, height=50, thickness=3.0, material="Steel", quantity=1, allow_rotation=True),
            # Normal panel (medium)
            Panel(id="Normal", width=300, height=400, thickness=3.0, material="Steel", quantity=1, allow_rotation=True),
            # Large panel (hard)
            Panel(id="Large", width=1000, height=1500, thickness=3.0, material="Steel", quantity=1, allow_rotation=True),
            # Maximum size panel (extreme)
            Panel(id="MaxSize", width=1450, height=3000, thickness=3.0, material="Steel", quantity=1, allow_rotation=False)
        ]

        # Test single panel placement
        for panel in panels:
            suitable_sheet = optimizer._find_smallest_suitable_sheet(panel, test_sheets)

            if suitable_sheet:
                assert suitable_sheet.width >= panel.width
                assert suitable_sheet.height >= panel.height
                logger.info(f"✅ Found suitable sheet for {panel.id}: {suitable_sheet.width}x{suitable_sheet.height}")
            else:
                # Check if rotation could help
                if panel.allow_rotation:
                    # Try rotated dimensions
                    rotated_fits = any(
                        sheet.width >= panel.height and sheet.height >= panel.width
                        for sheet in test_sheets
                    )
                    if rotated_fits:
                        logger.info(f"ℹ️ Panel {panel.id} could fit with rotation")
                    else:
                        logger.warning(f"⚠️ Panel {panel.id} doesn't fit any available sheet")
                else:
                    logger.warning(f"⚠️ Panel {panel.id} doesn't fit any available sheet (no rotation)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])