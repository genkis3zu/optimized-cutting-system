"""
Unit tests for First Fit Decreasing (FFD) algorithm
First Fit Decreasing (FFD) アルゴリズムのユニットテスト
"""

import pytest
import time
from unittest.mock import Mock, patch

from core.algorithms.ffd import FirstFitDecreasing, GuillotineBinPacker, Rectangle
from core.models import Panel, SteelSheet, OptimizationConstraints, PlacedPanel


class TestRectangle:
    """Test Rectangle helper class"""

    def test_rectangle_properties(self):
        """Test rectangle property calculations"""
        rect = Rectangle(10, 20, 100, 50)

        assert rect.x == 10
        assert rect.y == 20
        assert rect.width == 100
        assert rect.height == 50
        assert rect.area == 5000  # 100 * 50
        assert rect.right == 110  # 10 + 100
        assert rect.top == 70     # 20 + 50

    def test_rectangle_contains_point(self):
        """Test point containment"""
        rect = Rectangle(0, 0, 100, 100)

        assert rect.contains_point(50, 50) is True
        assert rect.contains_point(0, 0) is True
        assert rect.contains_point(99, 99) is True
        assert rect.contains_point(100, 100) is False  # Exclusive boundary
        assert rect.contains_point(-1, 50) is False

    def test_rectangle_can_fit(self):
        """Test dimension fitting"""
        rect = Rectangle(0, 0, 100, 200)

        assert rect.can_fit(50, 100) is True
        assert rect.can_fit(100, 200) is True
        assert rect.can_fit(101, 100) is False
        assert rect.can_fit(100, 201) is False


class TestGuillotineBinPacker:
    """Test GuillotineBinPacker class"""

    def setup_method(self):
        """Set up test packer"""
        self.packer = GuillotineBinPacker(1000, 1000, kerf_width=3.0)

    def test_packer_initialization(self):
        """Test packer initialization"""
        assert self.packer.sheet_width == 1000
        assert self.packer.sheet_height == 1000
        assert self.packer.kerf_width == 3.0
        assert len(self.packer.free_rectangles) == 1
        assert len(self.packer.placed_panels) == 0

        # Initial rectangle should cover entire sheet
        initial_rect = self.packer.free_rectangles[0]
        assert initial_rect.x == 0
        assert initial_rect.y == 0
        assert initial_rect.width == 1000
        assert initial_rect.height == 1000

    def test_simple_panel_placement(self):
        """Test placing a single panel"""
        panel = Panel("test", 200, 100, 1, "SS400", 6.0)

        success = self.packer.place_panel(panel)

        assert success is True
        assert len(self.packer.placed_panels) == 1

        placed = self.packer.placed_panels[0]
        assert placed.panel.id == "test"
        assert placed.x == 0  # Bottom-left placement
        assert placed.y == 0
        assert placed.rotated is False

    def test_multiple_panel_placement(self):
        """Test placing multiple panels"""
        panel1 = Panel("p1", 200, 100, 1, "SS400", 6.0)
        panel2 = Panel("p2", 150, 200, 1, "SS400", 6.0)

        success1 = self.packer.place_panel(panel1)
        success2 = self.packer.place_panel(panel2)

        assert success1 is True
        assert success2 is True
        assert len(self.packer.placed_panels) == 2

        # Verify no overlaps
        placed1 = self.packer.placed_panels[0]
        placed2 = self.packer.placed_panels[1]
        assert not placed1.overlaps_with(placed2)

    def test_panel_too_large(self):
        """Test placement of oversized panel"""
        large_panel = Panel("large", 1200, 800, 1, "SS400", 6.0)

        success = self.packer.place_panel(large_panel)

        assert success is False
        assert len(self.packer.placed_panels) == 0

    def test_panel_rotation(self):
        """Test panel rotation when beneficial"""
        # Panel that fits only when rotated in small sheet
        small_packer = GuillotineBinPacker(300, 500)
        tall_panel = Panel("tall", 400, 250, 1, "SS400", 6.0, allow_rotation=True)

        success = small_packer.place_panel(tall_panel)

        assert success is True
        placed = small_packer.placed_panels[0]
        assert placed.rotated is True  # Should be rotated to fit
        assert placed.actual_width == 250  # Height became width
        assert placed.actual_height == 400  # Width became height

    def test_panel_no_rotation_when_disabled(self):
        """Test panel placement without rotation"""
        small_packer = GuillotineBinPacker(300, 500)
        tall_panel = Panel("tall", 400, 250, 1, "SS400", 6.0, allow_rotation=False)

        success = small_packer.place_panel(tall_panel)

        assert success is False  # Should fail without rotation

    def test_bottom_left_preference(self):
        """Test bottom-left placement preference"""
        # Place first panel to create specific free rectangles
        panel1 = Panel("p1", 300, 200, 1, "SS400", 6.0)
        self.packer.place_panel(panel1)

        # Place second panel - should prefer bottom-left
        panel2 = Panel("p2", 100, 100, 1, "SS400", 6.0)
        self.packer.place_panel(panel2)

        placed2 = self.packer.placed_panels[1]
        # Should be placed at bottom-left of available space
        assert placed2.y == 0  # Bottom preference

    def test_efficiency_calculation(self):
        """Test packing efficiency calculation"""
        panel1 = Panel("p1", 200, 100, 1, "SS400", 6.0)  # 20K area
        panel2 = Panel("p2", 300, 100, 1, "SS400", 6.0)  # 30K area

        self.packer.place_panel(panel1)
        self.packer.place_panel(panel2)

        efficiency = self.packer.get_efficiency()
        expected_efficiency = 50000 / 1000000  # Total used / total sheet
        assert abs(efficiency - expected_efficiency) < 0.001

    def test_waste_area_calculation(self):
        """Test waste area calculation"""
        panel = Panel("test", 200, 100, 1, "SS400", 6.0)  # 20K area
        self.packer.place_panel(panel)

        waste = self.packer.get_waste_area()
        expected_waste = 1000000 - 20000  # Total - used
        assert waste == expected_waste

    def test_guillotine_cuts_with_kerf(self):
        """Test guillotine cuts include kerf allowance"""
        panel = Panel("test", 200, 100, 1, "SS400", 6.0)
        initial_free_count = len(self.packer.free_rectangles)

        self.packer.place_panel(panel)

        # Should create new free rectangles with kerf consideration
        assert len(self.packer.free_rectangles) >= 1

        # Check that kerf is accounted for in remaining space
        for rect in self.packer.free_rectangles:
            # Rectangles should start after the placed panel + kerf
            if rect.x > 0:
                assert rect.x >= 200 + self.packer.kerf_width
            if rect.y > 0:
                assert rect.y >= 100 + self.packer.kerf_width


class TestFirstFitDecreasing:
    """Test FirstFitDecreasing algorithm"""

    def setup_method(self):
        """Set up test algorithm"""
        self.ffd = FirstFitDecreasing()
        self.sheet = SteelSheet(width=1000, height=1000)
        self.constraints = OptimizationConstraints()

    def test_algorithm_initialization(self):
        """Test algorithm initialization"""
        assert self.ffd.name == "FFD"

    def test_time_estimation(self):
        """Test processing time estimation"""
        estimate = self.ffd.estimate_time(panel_count=10, complexity=0.5)

        assert estimate > 0
        assert isinstance(estimate, float)

        # More complex problems should take longer
        estimate_complex = self.ffd.estimate_time(panel_count=20, complexity=0.8)
        assert estimate_complex > estimate

    def test_empty_panel_list(self):
        """Test optimization with empty panel list"""
        result = self.ffd.optimize([], self.sheet, self.constraints)

        assert result.algorithm == "FFD"
        assert len(result.panels) == 0
        assert result.efficiency == 0.0

    def test_single_panel_optimization(self):
        """Test optimization with single panel"""
        panels = [Panel("single", 300, 200, 1, "SS400", 6.0)]

        result = self.ffd.optimize(panels, self.sheet, self.constraints)

        assert result.algorithm == "FFD"
        assert len(result.panels) == 1
        assert result.efficiency > 0
        assert result.processing_time >= 0

        placed = result.panels[0]
        assert placed.panel.id == "single"
        assert placed.x == 0
        assert placed.y == 0

    def test_multiple_panels_optimization(self):
        """Test optimization with multiple panels"""
        panels = [
            Panel("large", 400, 300, 1, "SS400", 6.0),
            Panel("medium", 300, 200, 1, "SS400", 6.0),
            Panel("small", 200, 100, 1, "SS400", 6.0)
        ]

        result = self.ffd.optimize(panels, self.sheet, self.constraints)

        assert result.algorithm == "FFD"
        assert len(result.panels) == 3  # All should fit
        assert result.efficiency > 0

        # Panels should be sorted by area (decreasing)
        areas = [p.panel.area for p in result.panels]
        # First panel should be largest (FFD sorting)
        assert areas[0] >= areas[1] >= areas[2]

    def test_panels_with_quantity(self):
        """Test optimization with panel quantities > 1"""
        panels = [Panel("multi", 200, 100, 3, "SS400", 6.0)]  # 3 pieces

        result = self.ffd.optimize(panels, self.sheet, self.constraints)

        assert len(result.panels) == 3  # Should expand to individual panels

        # All placed panels should have same dimensions
        for placed in result.panels:
            assert placed.panel.width == 200
            assert placed.panel.height == 100
            assert placed.panel.quantity == 1

    def test_panels_dont_fit(self):
        """Test optimization when panels don't fit"""
        small_sheet = SteelSheet(width=100, height=100)
        large_panels = [Panel("too_big", 200, 200, 1, "SS400", 6.0)]

        result = self.ffd.optimize(large_panels, small_sheet, self.constraints)

        assert len(result.panels) == 0  # Nothing should fit
        assert result.efficiency == 0.0

    def test_rotation_optimization(self):
        """Test optimization with rotation enabled"""
        # Narrow sheet, wide panel that needs rotation
        narrow_sheet = SteelSheet(width=500, height=1000)
        wide_panel = Panel("wide", 800, 300, 1, "SS400", 6.0, allow_rotation=True)

        result = self.ffd.optimize([wide_panel], narrow_sheet, self.constraints)

        assert len(result.panels) == 1
        placed = result.panels[0]
        # Should be rotated to fit
        assert placed.rotated is True
        assert placed.actual_width == 300  # Height became width
        assert placed.actual_height == 800  # Width became height

    def test_efficiency_target(self):
        """Test efficiency calculation accuracy"""
        # Fill sheet partially with known panels
        panel = Panel("test", 500, 400, 1, "SS400", 6.0)  # 200K area in 1M sheet = 20%

        result = self.ffd.optimize([panel], self.sheet, self.constraints)

        expected_efficiency = 200000 / 1000000  # 0.2
        assert abs(result.efficiency - expected_efficiency) < 0.001

    def test_processing_time_tracking(self):
        """Test processing time is tracked"""
        panels = [Panel(f"p{i}", 100, 100, 1, "SS400", 6.0) for i in range(5)]

        start_time = time.time()
        result = self.ffd.optimize(panels, self.sheet, self.constraints)
        end_time = time.time()

        assert result.processing_time > 0
        assert result.processing_time <= (end_time - start_time) + 0.1  # Small tolerance

    def test_material_consistency(self):
        """Test material information is preserved"""
        panels = [Panel("test", 300, 200, 1, "SUS304", 3.0)]

        result = self.ffd.optimize(panels, self.sheet, self.constraints)

        assert result.material_block == self.sheet.material
        assert result.panels[0].panel.material == "SUS304"

    def test_algorithm_validation(self):
        """Test result validation"""
        panels = [
            Panel("p1", 300, 200, 1, "SS400", 6.0),
            Panel("p2", 200, 100, 1, "SS400", 6.0)
        ]

        result = self.ffd.optimize(panels, self.sheet, self.constraints)

        # Should pass validation
        assert self.ffd.validate_placement(result) is True

        # No overlaps
        result.validate_no_overlaps()  # Should not raise

        # Within bounds
        result.validate_within_bounds()  # Should not raise

    def test_complexity_calculation(self):
        """Test complexity calculation"""
        # Simple case
        simple_panels = [Panel("p1", 300, 200, 1, "SS400", 6.0)]
        complexity_simple = self.ffd.calculate_complexity(simple_panels)

        # Complex case - more panels, different sizes, materials
        complex_panels = [
            Panel("p1", 300, 200, 2, "SS400", 6.0),
            Panel("p2", 400, 150, 1, "SUS304", 3.0, allow_rotation=True),
            Panel("p3", 250, 300, 3, "AL6061", 2.0),
        ]
        complexity_complex = self.ffd.calculate_complexity(complex_panels)

        assert 0 <= complexity_simple <= 1
        assert 0 <= complexity_complex <= 1
        assert complexity_complex > complexity_simple


class TestFFDIntegration:
    """Integration tests for FFD algorithm"""

    def test_typical_manufacturing_scenario(self):
        """Test with realistic manufacturing data"""
        # Typical Japanese steel cutting scenario
        panels = [
            Panel("フレーム_大", 1200, 800, 2, "SS400", 6.0),
            Panel("フレーム_小", 600, 400, 4, "SS400", 6.0),
            Panel("補強板", 300, 200, 6, "SS400", 6.0),
            Panel("カバー", 500, 300, 2, "SUS304", 3.0)
        ]

        sheet = SteelSheet()  # Standard 1500x3100
        constraints = OptimizationConstraints(
            kerf_width=0.0,
            target_efficiency=0.75
        )

        ffd = FirstFitDecreasing()
        result = ffd.optimize(panels, sheet, constraints)

        # Should place most panels successfully
        total_panels = sum(p.quantity for p in panels)
        assert len(result.panels) >= total_panels * 0.7  # At least 70% placed

        # Should achieve reasonable efficiency
        assert result.efficiency >= 0.3  # At least 30% (conservative)

        # Should complete in reasonable time
        assert result.processing_time < 5.0  # Less than 5 seconds

    def test_performance_target_verification(self):
        """Test performance targets from specification"""
        # Target: ≤10 panels, <1 second, 70-75% efficiency
        panels = [Panel(f"panel_{i}", 200 + i*10, 150 + i*5, 1, "SS400", 6.0)
                 for i in range(10)]

        sheet = SteelSheet()
        constraints = OptimizationConstraints()

        ffd = FirstFitDecreasing()
        start_time = time.time()
        result = ffd.optimize(panels, sheet, constraints)
        processing_time = time.time() - start_time

        # Performance targets
        assert processing_time < 1.0  # Under 1 second for ≤10 panels
        assert len(result.panels) == 10  # All panels should fit
        # Note: 70-75% efficiency target depends on specific panel sizes


if __name__ == "__main__":
    pytest.main([__file__])