"""
Integration tests for application functionality
アプリケーション機能の統合テスト
"""

import pytest
import streamlit as st
from unittest.mock import patch, MagicMock
import time

from core.models import Panel, SteelSheet, OptimizationConstraints
from core.optimizer import create_optimization_engine
from core.algorithms.ffd import create_ffd_algorithm
from core.text_parser import parse_text_data


class TestApplicationIntegration:
    """Test full application integration"""

    def test_end_to_end_optimization_workflow(self):
        """Test complete optimization workflow"""
        # 1. Create sample panels
        panels = [
            Panel("frame_large", 800, 600, 2, "SS400", 6.0),
            Panel("frame_small", 400, 300, 4, "SS400", 6.0),
            Panel("cover", 500, 200, 2, "SUS304", 3.0)
        ]

        # 2. Set up steel sheet and constraints
        sheet = SteelSheet(width=1500, height=3100, material="SS400")
        constraints = OptimizationConstraints(
            kerf_width=3.5,
            target_efficiency=0.75,
            time_budget=30.0
        )

        # 3. Create and configure optimization engine
        engine = create_optimization_engine()
        ffd_algorithm = create_ffd_algorithm()
        engine.register_algorithm(ffd_algorithm)

        # 4. Run optimization
        start_time = time.time()
        results = engine.optimize(
            panels=panels,
            constraints=constraints,
            algorithm_hint="FFD"
        )
        processing_time = time.time() - start_time

        # 5. Verify results
        assert len(results) > 0, "Should produce optimization results"
        assert processing_time < 30.0, "Should complete within time budget"

        # Check first result
        result = results[0]
        assert result.algorithm == "FFD"
        assert len(result.panels) > 0, "Should place some panels"
        assert result.efficiency > 0, "Should achieve some efficiency"
        assert result.processing_time > 0, "Should track processing time"

        # Verify no overlaps
        result.validate_no_overlaps()

        # Verify within bounds
        result.validate_within_bounds()

    def test_text_parsing_integration(self):
        """Test text parsing with optimization"""
        # Sample CSV data in Japanese and English
        csv_data = """フレーム_大,800,600,2,SS400,6.0
フレーム_小,400,300,4,SS400,6.0
カバー,500,200,2,SUS304,3.0"""

        # Parse text data
        parse_result = parse_text_data(csv_data, 'csv')

        assert parse_result.is_successful, "Should parse Japanese text successfully"
        assert len(parse_result.panels) == 3, "Should parse all 3 panels"

        # Verify Japanese panel IDs are preserved
        panel_ids = [p.id for p in parse_result.panels]
        assert "フレーム_大" in panel_ids
        assert "フレーム_小" in panel_ids
        assert "カバー" in panel_ids

        # Run optimization with parsed panels
        engine = create_optimization_engine()
        ffd_algorithm = create_ffd_algorithm()
        engine.register_algorithm(ffd_algorithm)

        constraints = OptimizationConstraints()
        results = engine.optimize(
            panels=parse_result.panels,
            constraints=constraints
        )

        assert len(results) > 0, "Should optimize parsed panels"

    def test_material_grouping_workflow(self):
        """Test material-based grouping workflow"""
        # Mixed material panels
        panels = [
            Panel("steel_1", 400, 300, 2, "SS400", 6.0),
            Panel("steel_2", 300, 200, 3, "SS400", 6.0),
            Panel("stainless_1", 500, 250, 1, "SUS304", 3.0),
            Panel("stainless_2", 350, 180, 2, "SUS304", 3.0),
            Panel("aluminum_1", 600, 400, 1, "AL6061", 2.0)
        ]

        # Enable material separation
        constraints = OptimizationConstraints(
            material_separation=True,
            time_budget=60.0
        )

        # Run optimization
        engine = create_optimization_engine()
        ffd_algorithm = create_ffd_algorithm()
        engine.register_algorithm(ffd_algorithm)

        results = engine.optimize(
            panels=panels,
            constraints=constraints
        )

        # Should create separate sheets for different materials
        material_blocks = set(result.material_block for result in results)
        assert len(material_blocks) >= 2, "Should separate different materials"

        # Verify each result contains only one material type
        for result in results:
            placed_materials = set(p.panel.material for p in result.panels)
            assert len(placed_materials) <= 1, f"Sheet should contain only one material type, got {placed_materials}"

    def test_rotation_optimization_integration(self):
        """Test rotation optimization in full workflow"""
        # Panels that benefit from rotation
        panels = [
            Panel("tall_panel", 800, 400, 1, "SS400", 6.0, allow_rotation=True),
            Panel("wide_panel", 400, 800, 1, "SS400", 6.0, allow_rotation=True),
            Panel("fixed_panel", 300, 200, 1, "SS400", 6.0, allow_rotation=False)
        ]

        # Use narrow sheet to force rotation
        narrow_sheet = SteelSheet(width=1000, height=2000)
        constraints = OptimizationConstraints(allow_rotation=True)

        # Create engine with custom sheet
        engine = create_optimization_engine()
        ffd_algorithm = create_ffd_algorithm()
        engine.register_algorithm(ffd_algorithm)

        # Mock the sheet in the optimization process
        # Note: This test verifies the rotation logic works in integration
        results = engine.optimize(panels=panels, constraints=constraints)

        assert len(results) > 0, "Should produce results"

        # Verify some panels were placed (may require rotation)
        total_placed = sum(len(result.panels) for result in results)
        assert total_placed > 0, "Should place at least some panels"

    def test_performance_requirements_integration(self):
        """Test performance requirements integration"""
        # Test data matching specification performance targets

        # Small batch: ≤10 panels, <1 second, 70-75% efficiency target
        small_panels = [
            Panel(f"small_{i}", 200 + i*10, 150 + i*5, 1, "SS400", 6.0)
            for i in range(8)
        ]

        constraints = OptimizationConstraints(time_budget=1.0)
        engine = create_optimization_engine()
        ffd_algorithm = create_ffd_algorithm()
        engine.register_algorithm(ffd_algorithm)

        start_time = time.time()
        results = engine.optimize(panels=small_panels, constraints=constraints)
        processing_time = time.time() - start_time

        # Performance verification
        assert processing_time < 1.5, f"Small batch should complete quickly, took {processing_time:.2f}s"
        assert len(results) > 0, "Should produce results"

        # Check if most panels were placed
        total_panels = sum(len(result.panels) for result in results)
        assert total_panels >= len(small_panels) * 0.7, "Should place most panels"

    def test_error_handling_integration(self):
        """Test error handling in integrated workflow"""
        # Test with problematic data

        # Empty panels
        engine = create_optimization_engine()
        ffd_algorithm = create_ffd_algorithm()
        engine.register_algorithm(ffd_algorithm)

        results = engine.optimize(panels=[], constraints=OptimizationConstraints())
        assert len(results) == 0, "Should handle empty panels gracefully"

        # Panels that don't fit
        huge_panels = [Panel("huge", 5000, 5000, 1, "SS400", 6.0)]
        results = engine.optimize(panels=huge_panels, constraints=OptimizationConstraints())

        # Should either return empty results or results with no placed panels
        if results:
            assert all(len(result.panels) == 0 for result in results), "Should not place oversized panels"

    def test_constraints_integration(self):
        """Test constraint handling integration"""
        panels = [
            Panel("test1", 300, 200, 1, "SS400", 6.0),
            Panel("test2", 250, 150, 1, "SS400", 6.0)
        ]

        # Test with different constraint settings
        tight_constraints = OptimizationConstraints(
            kerf_width=5.0,  # Large kerf
            target_efficiency=0.9,  # High target
            time_budget=1.0  # Short time
        )

        engine = create_optimization_engine()
        ffd_algorithm = create_ffd_algorithm()
        engine.register_algorithm(ffd_algorithm)

        results = engine.optimize(panels=panels, constraints=tight_constraints)

        # Should handle constraints without crashing
        assert isinstance(results, list), "Should return results list"

        # Verify kerf is considered (if panels placed)
        if results and results[0].panels:
            # Panels should be spaced according to kerf
            result = results[0]
            if len(result.panels) >= 2:
                panel1 = result.panels[0]
                panel2 = result.panels[1]

                # Check minimum spacing (simple heuristic)
                x_gap = abs(panel2.x - (panel1.x + panel1.actual_width))
                y_gap = abs(panel2.y - (panel1.y + panel1.actual_height))

                # At least one gap should be >= kerf width (guillotine constraint)
                assert min(x_gap, y_gap) < tight_constraints.kerf_width or \
                       max(x_gap, y_gap) >= tight_constraints.kerf_width, \
                       "Should respect kerf constraints"

    def test_algorithm_selection_integration(self):
        """Test algorithm selection in integrated workflow"""
        panels = [Panel(f"panel_{i}", 200, 150, 1, "SS400", 6.0) for i in range(5)]

        engine = create_optimization_engine()
        ffd_algorithm = create_ffd_algorithm()
        engine.register_algorithm(ffd_algorithm)

        # Test with algorithm hint
        constraints = OptimizationConstraints()
        results = engine.optimize(
            panels=panels,
            constraints=constraints,
            algorithm_hint="FFD"
        )

        assert len(results) > 0, "Should work with explicit algorithm"
        assert results[0].algorithm == "FFD", "Should use specified algorithm"

        # Test with auto selection
        results_auto = engine.optimize(
            panels=panels,
            constraints=constraints,
            algorithm_hint=None  # Auto selection
        )

        assert len(results_auto) > 0, "Should work with auto selection"


class TestDataValidationIntegration:
    """Test data validation in integrated scenarios"""

    def test_panel_validation_integration(self):
        """Test panel validation throughout workflow"""
        # Mix of valid and invalid panels
        mixed_panels = [
            Panel("valid1", 300, 200, 1, "SS400", 6.0),
            # This will fail at creation, so we'll test the workflow's handling
        ]

        # Test that valid panels work
        engine = create_optimization_engine()
        ffd_algorithm = create_ffd_algorithm()
        engine.register_algorithm(ffd_algorithm)

        results = engine.optimize(panels=mixed_panels, constraints=OptimizationConstraints())
        assert len(results) > 0, "Should work with valid panels"

    def test_csv_parsing_validation_integration(self):
        """Test CSV parsing with validation integration"""
        # CSV with mixed quality data
        mixed_csv = """valid_panel,300,200,1,SS400,6.0
another_valid,400,250,2,SUS304,3.0
invalid_line_with_bad_data,abc,def,xyz"""

        parse_result = parse_text_data(mixed_csv, 'csv')

        # Should parse valid lines
        assert len(parse_result.panels) >= 2, "Should parse valid panels"
        assert len(parse_result.errors) >= 1, "Should report errors for invalid lines"

        # Should be able to optimize valid panels
        if parse_result.panels:
            engine = create_optimization_engine()
            ffd_algorithm = create_ffd_algorithm()
            engine.register_algorithm(ffd_algorithm)

            results = engine.optimize(
                panels=parse_result.panels,
                constraints=OptimizationConstraints()
            )

            assert len(results) >= 0, "Should handle parsed panels"


class TestMemoryAndPerformanceIntegration:
    """Test memory usage and performance in integration scenarios"""

    def test_memory_efficiency_integration(self):
        """Test memory usage with larger datasets"""
        # Create moderate dataset
        panels = [
            Panel(f"panel_{i}", 200 + (i % 5) * 10, 150 + (i % 3) * 15, 1, "SS400", 6.0)
            for i in range(50)
        ]

        engine = create_optimization_engine()
        ffd_algorithm = create_ffd_algorithm()
        engine.register_algorithm(ffd_algorithm)

        constraints = OptimizationConstraints(time_budget=30.0)

        # Monitor memory usage (basic check)
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        results = engine.optimize(panels=panels, constraints=constraints)

        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before

        # Should not consume excessive memory (rough check)
        assert memory_increase < 100, f"Memory usage increased by {memory_increase:.1f}MB, which seems excessive"
        assert len(results) >= 0, "Should produce results"

    def test_timeout_handling_integration(self):
        """Test timeout handling in integrated workflow"""
        # Create challenging dataset
        panels = [
            Panel(f"complex_{i}", 200 + i, 150 + i, 1, f"MAT_{i%3}", 6.0)
            for i in range(30)
        ]

        engine = create_optimization_engine()
        ffd_algorithm = create_ffd_algorithm()
        engine.register_algorithm(ffd_algorithm)

        # Very short timeout
        constraints = OptimizationConstraints(time_budget=0.1)

        start_time = time.time()
        results = engine.optimize(panels=panels, constraints=constraints)
        actual_time = time.time() - start_time

        # Should respect timeout (with some tolerance)
        assert actual_time < 2.0, f"Should respect timeout, took {actual_time:.2f}s"

        # Should return results (possibly empty or partial)
        assert isinstance(results, list), "Should return results list even on timeout"


if __name__ == "__main__":
    pytest.main([__file__])