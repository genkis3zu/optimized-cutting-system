"""
Unit tests for core data models
コアデータモデルのユニットテスト
"""

import pytest
from unittest.mock import patch
from datetime import datetime

from core.models import (
    Panel, SteelSheet, PlacedPanel, PlacementResult,
    OptimizationConstraints, PanelAPI
)


class TestPanel:
    """Test Panel data model"""

    def test_valid_panel_creation(self):
        """Test creating a valid panel"""
        panel = Panel(
            id="test_panel",
            width=300.0,
            height=200.0,
            quantity=2,
            material="SS400",
            thickness=6.0
        )

        assert panel.id == "test_panel"
        assert panel.width == 300.0
        assert panel.height == 200.0
        assert panel.quantity == 2
        assert panel.material == "SS400"
        assert panel.thickness == 6.0
        assert panel.priority == 1  # default value
        assert panel.allow_rotation is True  # default value
        assert panel.area == 60000.0  # 300 * 200

    def test_panel_size_validation(self):
        """Test panel size constraint validation"""
        # Test minimum size constraint
        with pytest.raises(ValueError, match="must be between 50-1500mm"):
            Panel(
                id="too_small",
                width=30.0,  # Below minimum
                height=200.0,
                quantity=1,
                material="SS400",
                thickness=6.0
            )

        # Test maximum width constraint
        with pytest.raises(ValueError, match="must be between 50-1500mm"):
            Panel(
                id="too_wide",
                width=2000.0,  # Above maximum
                height=200.0,
                quantity=1,
                material="SS400",
                thickness=6.0
            )

        # Test maximum height constraint
        with pytest.raises(ValueError, match="must be between 50-3100mm"):
            Panel(
                id="too_tall",
                width=300.0,
                height=4000.0,  # Above maximum
                quantity=1,
                material="SS400",
                thickness=6.0
            )

    def test_panel_quantity_validation(self):
        """Test panel quantity validation"""
        with pytest.raises(ValueError, match="quantity must be positive"):
            Panel(
                id="zero_qty",
                width=300.0,
                height=200.0,
                quantity=0,  # Invalid
                material="SS400",
                thickness=6.0
            )

    def test_panel_thickness_validation(self):
        """Test panel thickness validation"""
        with pytest.raises(ValueError, match="thickness must be positive"):
            Panel(
                id="zero_thickness",
                width=300.0,
                height=200.0,
                quantity=1,
                material="SS400",
                thickness=0.0  # Invalid
            )

    def test_panel_rotation(self):
        """Test panel rotation functionality"""
        panel = Panel(
            id="rotatable",
            width=300.0,
            height=200.0,
            quantity=1,
            material="SS400",
            thickness=6.0,
            allow_rotation=True
        )

        rotated = panel.rotated
        assert rotated.width == 200.0  # swapped
        assert rotated.height == 300.0  # swapped
        assert rotated.id == "rotatable_rotated"

        # Test non-rotatable panel
        panel.allow_rotation = False
        non_rotated = panel.rotated
        assert non_rotated.width == 300.0  # unchanged
        assert non_rotated.height == 200.0  # unchanged

    def test_panel_fits_in_sheet(self):
        """Test sheet fitting logic"""
        panel = Panel(
            id="test",
            width=300.0,
            height=200.0,
            quantity=1,
            material="SS400",
            thickness=6.0,
            allow_rotation=True
        )

        # Should fit in standard sheet
        assert panel.fits_in_sheet(1500.0, 3100.0) is True

        # Should not fit in small sheet
        assert panel.fits_in_sheet(250.0, 150.0) is False

        # Should fit with rotation
        assert panel.fits_in_sheet(250.0, 350.0) is True

        # Test without rotation allowed
        panel.allow_rotation = False
        assert panel.fits_in_sheet(250.0, 350.0) is False


class TestSteelSheet:
    """Test SteelSheet data model"""

    def test_valid_sheet_creation(self):
        """Test creating a valid steel sheet"""
        sheet = SteelSheet(
            width=1500.0,
            height=3100.0,
            thickness=6.0,
            material="SS400",
            cost_per_sheet=15000.0
        )

        assert sheet.width == 1500.0
        assert sheet.height == 3100.0
        assert sheet.area == 4650000.0  # 1500 * 3100
        assert sheet.material == "SS400"
        assert sheet.cost_per_sheet == 15000.0

    def test_sheet_validation(self):
        """Test sheet dimension validation"""
        with pytest.raises(ValueError, match="dimensions must be positive"):
            SteelSheet(width=0.0, height=3100.0)

        with pytest.raises(ValueError, match="thickness must be positive"):
            SteelSheet(width=1500.0, height=3100.0, thickness=0.0)


class TestPlacedPanel:
    """Test PlacedPanel data model"""

    def test_placed_panel_creation(self):
        """Test creating a placed panel"""
        panel = Panel(
            id="test",
            width=300.0,
            height=200.0,
            quantity=1,
            material="SS400",
            thickness=6.0
        )

        placed = PlacedPanel(
            panel=panel,
            x=100.0,
            y=50.0,
            rotated=False
        )

        assert placed.x == 100.0
        assert placed.y == 50.0
        assert placed.rotated is False
        assert placed.actual_width == 300.0
        assert placed.actual_height == 200.0
        assert placed.bounds == (100.0, 50.0, 400.0, 250.0)

    def test_rotated_placed_panel(self):
        """Test rotated placed panel dimensions"""
        panel = Panel(
            id="test",
            width=300.0,
            height=200.0,
            quantity=1,
            material="SS400",
            thickness=6.0
        )

        placed = PlacedPanel(
            panel=panel,
            x=0.0,
            y=0.0,
            rotated=True
        )

        assert placed.actual_width == 200.0  # swapped
        assert placed.actual_height == 300.0  # swapped
        assert placed.bounds == (0.0, 0.0, 200.0, 300.0)

    def test_panel_overlap_detection(self):
        """Test overlap detection between panels"""
        panel1 = Panel("p1", 100.0, 100.0, 1, "SS400", 6.0)
        panel2 = Panel("p2", 100.0, 100.0, 1, "SS400", 6.0)

        placed1 = PlacedPanel(panel1, 0.0, 0.0, False)
        placed2 = PlacedPanel(panel2, 50.0, 50.0, False)  # Overlapping
        placed3 = PlacedPanel(panel2, 200.0, 200.0, False)  # Non-overlapping

        assert placed1.overlaps_with(placed2) is True
        assert placed1.overlaps_with(placed3) is False


class TestPlacementResult:
    """Test PlacementResult data model"""

    def test_placement_result_creation(self):
        """Test creating a placement result"""
        sheet = SteelSheet()
        panel = Panel("test", 300.0, 200.0, 1, "SS400", 6.0)
        placed = PlacedPanel(panel, 0.0, 0.0, False)

        result = PlacementResult(
            sheet_id=1,
            material_block="SS400",
            sheet=sheet,
            panels=[placed],
            efficiency=0.0,  # Will be calculated
            waste_area=0.0,
            cut_length=1000.0,
            cost=15000.0
        )

        assert result.sheet_id == 1
        assert result.material_block == "SS400"
        assert len(result.panels) == 1
        assert result.used_area == 60000.0  # 300 * 200
        assert result.total_area == 4650000.0  # 1500 * 3100

    def test_efficiency_calculation(self):
        """Test efficiency calculation"""
        sheet = SteelSheet(width=1000.0, height=1000.0)  # 1M mm²
        panel = Panel("test", 300.0, 200.0, 1, "SS400", 6.0)  # 60K mm²
        placed = PlacedPanel(panel, 0.0, 0.0, False)

        result = PlacementResult(
            sheet_id=1,
            material_block="SS400",
            sheet=sheet,
            panels=[placed],
            efficiency=0.0,
            waste_area=0.0,
            cut_length=0.0,
            cost=0.0
        )

        efficiency = result.calculate_efficiency()
        assert efficiency == 0.06  # 60K / 1M = 0.06
        assert result.efficiency == 0.06

    def test_overlap_validation(self):
        """Test overlap validation"""
        sheet = SteelSheet()
        panel1 = Panel("p1", 100.0, 100.0, 1, "SS400", 6.0)
        panel2 = Panel("p2", 100.0, 100.0, 1, "SS400", 6.0)

        placed1 = PlacedPanel(panel1, 0.0, 0.0, False)
        placed2 = PlacedPanel(panel2, 50.0, 50.0, False)  # Overlapping

        result = PlacementResult(
            sheet_id=1,
            material_block="SS400",
            sheet=sheet,
            panels=[placed1, placed2],
            efficiency=0.0,
            waste_area=0.0,
            cut_length=0.0,
            cost=0.0
        )

        with pytest.raises(ValueError, match="overlap"):
            result.validate_no_overlaps()

    def test_bounds_validation(self):
        """Test bounds validation"""
        sheet = SteelSheet(width=1000.0, height=1000.0)
        panel = Panel("test", 100.0, 100.0, 1, "SS400", 6.0)
        placed = PlacedPanel(panel, 950.0, 950.0, False)  # Exceeds bounds

        result = PlacementResult(
            sheet_id=1,
            material_block="SS400",
            sheet=sheet,
            panels=[placed],
            efficiency=0.0,
            waste_area=0.0,
            cut_length=0.0,
            cost=0.0
        )

        with pytest.raises(ValueError, match="exceeds sheet bounds"):
            result.validate_within_bounds()


class TestOptimizationConstraints:
    """Test OptimizationConstraints data model"""

    def test_valid_constraints(self):
        """Test creating valid constraints"""
        constraints = OptimizationConstraints(
            max_sheets=5,
            kerf_width=0.0,
            target_efficiency=0.8
        )

        assert constraints.max_sheets == 5
        assert constraints.kerf_width == 0.0
        assert constraints.target_efficiency == 0.8
        assert constraints.validate() is True

    def test_constraint_validation(self):
        """Test constraint validation"""
        # Test invalid max_sheets
        with pytest.raises(ValueError, match="Max sheets must be positive"):
            constraints = OptimizationConstraints(max_sheets=0)
            constraints.validate()

        # Test invalid kerf_width
        with pytest.raises(ValueError, match="Kerf width cannot be negative"):
            constraints = OptimizationConstraints(kerf_width=-1.0)
            constraints.validate()

        # Test invalid target_efficiency
        with pytest.raises(ValueError, match="Target efficiency must be between 0 and 1"):
            constraints = OptimizationConstraints(target_efficiency=1.5)
            constraints.validate()


class TestPanelAPI:
    """Test Pydantic API model"""

    def test_valid_panel_api(self):
        """Test creating valid PanelAPI instance"""
        panel_api = PanelAPI(
            id="api_test",
            width=300.0,
            height=200.0,
            quantity=2,
            material="SS400",
            thickness=6.0
        )

        assert panel_api.id == "api_test"
        assert panel_api.width == 300.0

        # Test conversion to Panel
        panel = panel_api.to_panel()
        assert isinstance(panel, Panel)
        assert panel.id == "api_test"
        assert panel.width == 300.0

    def test_panel_api_validation(self):
        """Test PanelAPI Pydantic validation"""
        # Test invalid width
        with pytest.raises(Exception):  # Pydantic validation error
            PanelAPI(
                id="test",
                width=30.0,  # Below minimum
                height=200.0,
                quantity=1,
                material="SS400",
                thickness=6.0
            )

        # Test invalid quantity
        with pytest.raises(Exception):  # Pydantic validation error
            PanelAPI(
                id="test",
                width=300.0,
                height=200.0,
                quantity=0,  # Below minimum
                material="SS400",
                thickness=6.0
            )


if __name__ == "__main__":
    pytest.main([__file__])