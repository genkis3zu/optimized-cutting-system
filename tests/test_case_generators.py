"""
Test Case Generators for 100% Panel Placement Guarantee Testing
100%配置保証テストケース生成器

Generates comprehensive test scenarios including:
- Edge cases for boundary conditions
- Stress tests for algorithmic robustness
- Regression tests for performance validation
- Real-world scenarios from production data
"""

import random
import math
from typing import List, Dict, Tuple, Generator, Any
from dataclasses import dataclass
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.models import Panel, OptimizationConstraints
from tests.placement_guarantee_framework import PlacementTestCase


class TestCaseGenerator:
    """
    Systematic test case generation for comprehensive coverage

    Generates test cases across multiple dimensions:
    - Panel size distributions (small, medium, large, mixed)
    - Quantity patterns (uniform, skewed, bulk)
    - Material constraints (single, multiple materials)
    - Geometric constraints (rotation, aspect ratios)
    - Edge cases (boundary sizes, degenerate cases)
    """

    def __init__(self, seed: int = 42):
        """Initialize generator with reproducible seed"""
        random.seed(seed)
        self.panel_id_counter = 1

        # Standard sheet sizes available in the system
        self.standard_sheets = [
            (1500, 3100),  # Standard large sheet
            (1200, 2400),  # Medium sheet
            (1000, 2000),  # Small sheet
            (800, 1600),   # Compact sheet
            (600, 1200)    # Minimal sheet
        ]

        # Panel size categories for systematic testing
        self.size_categories = {
            'tiny': (50, 100),      # Minimum size range
            'small': (100, 300),    # Small panels
            'medium': (300, 600),   # Medium panels
            'large': (600, 1000),   # Large panels
            'xlarge': (1000, 1500), # Extra large panels
            'boundary': [(49, 49), (50, 50), (1499, 1499), (1500, 1500),
                        (51, 3099), (1500, 3100)]  # Boundary test cases
        }

    def generate_simple_test_cases(self) -> List[PlacementTestCase]:
        """Generate basic test cases for fundamental validation"""
        test_cases = []

        # Test 1: Single small panel - should always achieve 100%
        test_cases.append(PlacementTestCase(
            name="single_small_panel",
            panels=[self._create_panel("single_1", 200, 150, 1, "STEEL", allow_rotation=True)],
            constraints=self._create_unlimited_constraints(),
            expected_placement_rate=100.0,
            max_execution_time=30.0,
            description="Single small panel - basic functionality test",
            category="basic"
        ))

        # Test 2: Multiple identical small panels - perfect grid fit
        identical_panels = [self._create_panel("grid_1", 150, 100, 20, "STEEL", allow_rotation=False)]
        test_cases.append(PlacementTestCase(
            name="identical_grid_panels",
            panels=identical_panels,
            constraints=self._create_unlimited_constraints(),
            expected_placement_rate=100.0,
            max_execution_time=60.0,
            description="Multiple identical panels for perfect grid placement",
            category="basic"
        ))

        # Test 3: Mixed sizes within single sheet capacity
        mixed_panels = [
            self._create_panel("mix_1", 400, 300, 2, "STEEL"),
            self._create_panel("mix_2", 200, 150, 4, "STEEL"),
            self._create_panel("mix_3", 100, 80, 8, "STEEL")
        ]
        test_cases.append(PlacementTestCase(
            name="mixed_sizes_single_sheet",
            panels=mixed_panels,
            constraints=self._create_unlimited_constraints(),
            expected_placement_rate=100.0,
            max_execution_time=90.0,
            description="Mixed panel sizes within single sheet capacity",
            category="basic"
        ))

        return test_cases

    def generate_edge_case_tests(self) -> List[PlacementTestCase]:
        """Generate edge cases for boundary condition testing"""
        test_cases = []

        # Edge Case 1: Minimum size panels
        min_panels = [self._create_panel("min_1", 50, 50, 10, "STEEL")]
        test_cases.append(PlacementTestCase(
            name="minimum_size_panels",
            panels=min_panels,
            constraints=self._create_unlimited_constraints(),
            expected_placement_rate=100.0,
            max_execution_time=60.0,
            description="Minimum size panels (50x50mm) - boundary test",
            category="edge_case"
        ))

        # Edge Case 2: Maximum size panels (just under sheet limit)
        max_panels = [self._create_panel("max_1", 1499, 3099, 1, "STEEL")]
        test_cases.append(PlacementTestCase(
            name="maximum_size_panels",
            panels=max_panels,
            constraints=self._create_unlimited_constraints(),
            expected_placement_rate=100.0,
            max_execution_time=60.0,
            description="Maximum size panels (1499x3099mm) - boundary test",
            category="edge_case"
        ))

        # Edge Case 3: Extreme aspect ratio panels
        extreme_aspect_panels = [
            self._create_panel("narrow_1", 50, 1000, 3, "STEEL"),
            self._create_panel("wide_1", 1000, 50, 3, "STEEL")
        ]
        test_cases.append(PlacementTestCase(
            name="extreme_aspect_ratios",
            panels=extreme_aspect_panels,
            constraints=self._create_unlimited_constraints(),
            expected_placement_rate=100.0,
            max_execution_time=120.0,
            description="Extreme aspect ratio panels - geometric edge case",
            category="edge_case"
        ))

        # Edge Case 4: Single panel larger than standard sheet
        oversized_panel = [self._create_panel("oversized_1", 1200, 3200, 1, "STEEL")]
        test_cases.append(PlacementTestCase(
            name="oversized_single_panel",
            panels=oversized_panel,
            constraints=self._create_unlimited_constraints(),
            expected_placement_rate=100.0,  # Should use larger sheet
            max_execution_time=60.0,
            description="Single panel larger than standard sheet - sheet selection test",
            category="edge_case"
        ))

        # Edge Case 5: Rotation-dependent placement
        rotation_panels = [
            self._create_panel("rot_1", 800, 1200, 2, "STEEL", allow_rotation=True),
            self._create_panel("no_rot_1", 800, 1200, 2, "STEEL", allow_rotation=False)
        ]
        test_cases.append(PlacementTestCase(
            name="rotation_dependency",
            panels=rotation_panels,
            constraints=self._create_unlimited_constraints(),
            expected_placement_rate=100.0,
            max_execution_time=120.0,
            description="Panels requiring rotation for optimal placement",
            category="edge_case"
        ))

        return test_cases

    def generate_stress_tests(self) -> List[PlacementTestCase]:
        """Generate stress tests for algorithmic robustness"""
        test_cases = []

        # Stress Test 1: High quantity bulk panels
        bulk_panels = [self._create_panel("bulk_1", 300, 200, 100, "STEEL")]
        test_cases.append(PlacementTestCase(
            name="high_quantity_bulk",
            panels=bulk_panels,
            constraints=self._create_unlimited_constraints(),
            expected_placement_rate=100.0,
            max_execution_time=300.0,  # Allow more time for bulk processing
            description="High quantity bulk panels - stress test",
            category="stress"
        ))

        # Stress Test 2: Many different panel types
        diverse_panels = []
        for i in range(50):
            w = random.randint(100, 800)
            h = random.randint(100, 800)
            qty = random.randint(1, 5)
            diverse_panels.append(self._create_panel(f"diverse_{i}", w, h, qty, "STEEL"))

        test_cases.append(PlacementTestCase(
            name="many_diverse_panels",
            panels=diverse_panels,
            constraints=self._create_unlimited_constraints(),
            expected_placement_rate=100.0,
            max_execution_time=300.0,
            description="Many diverse panel types - algorithmic stress test",
            category="stress"
        ))

        # Stress Test 3: Mixed material constraints
        multi_material_panels = []
        materials = ["STEEL", "ALUMINUM", "STAINLESS"]
        for i, material in enumerate(materials):
            for j in range(10):
                w = random.randint(200, 600)
                h = random.randint(200, 600)
                qty = random.randint(2, 8)
                multi_material_panels.append(self._create_panel(f"{material.lower()}_{j}", w, h, qty, material))

        test_cases.append(PlacementTestCase(
            name="multi_material_stress",
            panels=multi_material_panels,
            constraints=OptimizationConstraints(
                material_separation=True,  # Force material separation
                max_sheets=1000,
                time_budget=0.0,
                kerf_width=3.0,
                target_efficiency=0.01
            ),
            expected_placement_rate=100.0,
            max_execution_time=300.0,
            description="Multi-material panels with separation constraints",
            category="stress"
        ))

        # Stress Test 4: Pathological packing case
        pathological_panels = [
            self._create_panel("path_1", 751, 1551, 1, "STEEL"),  # Slightly over half sheet
            self._create_panel("path_2", 500, 500, 10, "STEEL"),   # Many medium squares
            self._create_panel("path_3", 100, 1500, 5, "STEEL")    # Thin strips
        ]
        test_cases.append(PlacementTestCase(
            name="pathological_packing",
            panels=pathological_panels,
            constraints=self._create_unlimited_constraints(),
            expected_placement_rate=100.0,
            max_execution_time=300.0,
            description="Pathological packing scenario - difficult geometrical arrangement",
            category="stress"
        ))

        return test_cases

    def generate_real_world_tests(self) -> List[PlacementTestCase]:
        """Generate tests based on real production data patterns"""
        test_cases = []

        # Real World 1: Typical Japanese manufacturing pattern (from data0923.txt analysis)
        japanese_panels = [
            self._create_panel("LUX_892x576", 892, 576, 16, "STEEL", pi_code="77131000"),
            self._create_panel("LUX_767x571", 767.5, 571, 2, "STEEL", pi_code="77131000"),
            self._create_panel("LUX_601x571", 601.5, 571, 2, "STEEL", pi_code="77131000"),
            self._create_panel("LUX_902x516", 902, 516, 8, "STEEL", pi_code="77131000"),
            self._create_panel("LUX_932x516", 932, 516, 20, "STEEL", pi_code="77131000"),
            self._create_panel("LUX_892x616", 892, 616, 20, "STEEL", pi_code="77131000"),
            self._create_panel("LUX_1162x616", 1162, 616, 2, "STEEL", pi_code="77131000")
        ]
        test_cases.append(PlacementTestCase(
            name="japanese_manufacturing_pattern",
            panels=japanese_panels,
            constraints=self._create_unlimited_constraints(),
            expected_placement_rate=100.0,
            max_execution_time=300.0,
            description="Real Japanese manufacturing data pattern with PI expansion",
            category="real_world"
        ))

        # Real World 2: Mixed batch production
        mixed_batch_panels = []
        # Large panels (low quantity)
        mixed_batch_panels.extend([
            self._create_panel("large_1", 1200, 800, 2, "STEEL"),
            self._create_panel("large_2", 1000, 900, 1, "STEEL")
        ])
        # Medium panels (medium quantity)
        mixed_batch_panels.extend([
            self._create_panel("med_1", 600, 400, 8, "STEEL"),
            self._create_panel("med_2", 500, 500, 6, "STEEL")
        ])
        # Small panels (high quantity)
        mixed_batch_panels.extend([
            self._create_panel("small_1", 200, 150, 25, "STEEL"),
            self._create_panel("small_2", 180, 120, 30, "STEEL")
        ])

        test_cases.append(PlacementTestCase(
            name="mixed_batch_production",
            panels=mixed_batch_panels,
            constraints=self._create_unlimited_constraints(),
            expected_placement_rate=100.0,
            max_execution_time=300.0,
            description="Mixed batch production pattern - varied sizes and quantities",
            category="real_world"
        ))

        # Real World 3: Material efficiency focus
        efficiency_panels = [
            self._create_panel("eff_1", 750, 1550, 4, "STEEL"),  # Perfect half-sheet fit
            self._create_panel("eff_2", 500, 1033, 9, "STEEL"),  # Third-sheet fit
            self._create_panel("eff_3", 375, 775, 16, "STEEL")   # Quarter-sheet fit
        ]
        test_cases.append(PlacementTestCase(
            name="material_efficiency_focus",
            panels=efficiency_panels,
            constraints=OptimizationConstraints(
                material_separation=False,
                max_sheets=1000,
                time_budget=0.0,
                kerf_width=3.0,
                target_efficiency=0.85  # High efficiency target
            ),
            expected_placement_rate=100.0,
            max_execution_time=180.0,
            description="High material efficiency requirement scenario",
            category="real_world"
        ))

        return test_cases

    def generate_regression_tests(self) -> List[PlacementTestCase]:
        """Generate regression tests for performance validation"""
        test_cases = []

        # Regression 1: Performance baseline - small batch
        small_batch = [self._create_panel(f"perf_small_{i}", 200+i*10, 150+i*5, 2, "STEEL")
                      for i in range(10)]
        test_cases.append(PlacementTestCase(
            name="performance_baseline_small",
            panels=small_batch,
            constraints=self._create_unlimited_constraints(),
            expected_placement_rate=100.0,
            max_execution_time=1.0,  # Strict time limit
            description="Performance baseline for small batches (≤20 panels)",
            category="regression"
        ))

        # Regression 2: Performance baseline - medium batch
        medium_batch = [self._create_panel(f"perf_med_{i}", 200+i*5, 150+i*3, 1, "STEEL")
                       for i in range(40)]
        test_cases.append(PlacementTestCase(
            name="performance_baseline_medium",
            panels=medium_batch,
            constraints=self._create_unlimited_constraints(),
            expected_placement_rate=100.0,
            max_execution_time=5.0,  # Strict time limit
            description="Performance baseline for medium batches (≤50 panels)",
            category="regression"
        ))

        # Regression 3: Memory usage test
        memory_test_panels = [self._create_panel(f"mem_{i}", 300, 200, 2, "STEEL")
                             for i in range(80)]
        test_cases.append(PlacementTestCase(
            name="memory_usage_regression",
            panels=memory_test_panels,
            constraints=self._create_unlimited_constraints(),
            expected_placement_rate=100.0,
            max_execution_time=30.0,
            description="Memory usage regression test for large batches",
            category="regression"
        ))

        return test_cases

    def generate_pi_expansion_tests(self) -> List[PlacementTestCase]:
        """Generate tests specifically for PI expansion functionality"""
        test_cases = []

        # PI Test 1: Standard PI expansion
        pi_panels = [
            self._create_panel("pi_std_1", 800, 600, 5, "STEEL", pi_code="77131000"),
            self._create_panel("pi_std_2", 600, 400, 8, "STEEL", pi_code="77131000")
        ]
        test_cases.append(PlacementTestCase(
            name="standard_pi_expansion",
            panels=pi_panels,
            constraints=self._create_unlimited_constraints(),
            expected_placement_rate=100.0,
            max_execution_time=120.0,
            description="Standard PI code expansion functionality test",
            category="pi_expansion"
        ))

        # PI Test 2: Mixed PI codes
        mixed_pi_panels = [
            self._create_panel("pi_mix_1", 800, 600, 3, "STEEL", pi_code="77131000"),
            self._create_panel("pi_mix_2", 700, 500, 4, "STEEL", pi_code="77141000"),
            self._create_panel("pi_mix_3", 600, 400, 5, "STEEL", pi_code="")  # No PI code
        ]
        test_cases.append(PlacementTestCase(
            name="mixed_pi_codes",
            panels=mixed_pi_panels,
            constraints=self._create_unlimited_constraints(),
            expected_placement_rate=100.0,
            max_execution_time=150.0,
            description="Mixed PI codes with different expansion rules",
            category="pi_expansion"
        ))

        return test_cases

    def generate_comprehensive_test_suite(self) -> List[PlacementTestCase]:
        """Generate comprehensive test suite covering all categories"""
        all_tests = []
        all_tests.extend(self.generate_simple_test_cases())
        all_tests.extend(self.generate_edge_case_tests())
        all_tests.extend(self.generate_stress_tests())
        all_tests.extend(self.generate_real_world_tests())
        all_tests.extend(self.generate_regression_tests())
        all_tests.extend(self.generate_pi_expansion_tests())
        return all_tests

    def _create_panel(self, id_prefix: str, width: float, height: float, quantity: int,
                     material: str, allow_rotation: bool = True, pi_code: str = "") -> Panel:
        """Create a panel with specified parameters"""
        panel = Panel(
            id=f"{id_prefix}_{self.panel_id_counter}",
            width=width,
            height=height,
            quantity=quantity,
            material=material,
            thickness=0.5,  # Standard thickness
            priority=1,
            allow_rotation=allow_rotation,
            pi_code=pi_code
        )
        self.panel_id_counter += 1
        return panel

    def _create_unlimited_constraints(self) -> OptimizationConstraints:
        """Create optimization constraints that allow unlimited sheets and time"""
        return OptimizationConstraints(
            material_separation=False,
            max_sheets=1000,  # Allow many sheets
            time_budget=0.0,  # No time limit
            kerf_width=0.0,   # No cutting kerf for simplicity
            target_efficiency=0.01  # Very low efficiency target - focus on placement
        )


class ProductionDataTestGenerator:
    """
    Generate test cases from real production data files

    Loads actual production data and creates test cases that validate
    100% placement guarantee using real-world panel distributions.
    """

    def __init__(self, data_dir: Path = None):
        """Initialize with path to production data directory"""
        self.data_dir = data_dir or (project_root / "sample_data")

    def generate_production_tests(self) -> List[PlacementTestCase]:
        """Generate tests from all available production data files"""
        test_cases = []

        # Load known production data files
        production_files = [
            "data0923.txt",
            "sizaidata.txt"
        ]

        for filename in production_files:
            file_path = self.data_dir / filename
            if file_path.exists():
                try:
                    test_case = self._create_production_test_case(file_path)
                    if test_case:
                        test_cases.append(test_case)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

        return test_cases

    def _create_production_test_case(self, file_path: Path) -> PlacementTestCase:
        """Create a test case from a production data file"""
        from core.text_parser import parse_cutting_data_file

        # Parse the production data
        try:
            parse_result = parse_cutting_data_file(str(file_path))
            panels = parse_result.panels

            if not panels:
                return None

            total_panels = sum(panel.quantity for panel in panels)

            return PlacementTestCase(
                name=f"production_data_{file_path.stem}",
                panels=panels,
                constraints=OptimizationConstraints(
                    material_separation=False,
                    max_sheets=1000,
                    time_budget=0.0,
                    kerf_width=3.0,  # Realistic kerf width
                    target_efficiency=0.01
                ),
                expected_placement_rate=100.0,
                max_execution_time=600.0,  # Allow up to 10 minutes for production data
                description=f"Real production data from {file_path.name} ({total_panels} total panels)",
                category="production_data"
            )

        except Exception as e:
            print(f"Failed to parse production data {file_path}: {e}")
            return None


def generate_all_test_cases() -> List[PlacementTestCase]:
    """Generate complete set of test cases for 100% placement guarantee validation"""
    generator = TestCaseGenerator()
    production_generator = ProductionDataTestGenerator()

    all_tests = []
    all_tests.extend(generator.generate_comprehensive_test_suite())
    all_tests.extend(production_generator.generate_production_tests())

    return all_tests