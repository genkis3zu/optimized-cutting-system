"""
Algorithm Correctness Validation System for Steel Cutting Optimization
アルゴリズム正確性検証システム

Provides comprehensive validation of algorithm correctness including:
- Geometric placement validation (overlaps, boundaries, rotations)
- Physical constraint compliance (material properties, cutting limitations)
- Mathematical verification (area calculations, efficiency metrics)
- Data integrity checks (panel counts, material consistency)
"""

import math
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.models import Panel, SteelSheet, PlacementResult, PlacedPanel, OptimizationConstraints


class ValidationLevel(Enum):
    """Validation strictness levels"""
    BASIC = "basic"           # Essential checks only
    STANDARD = "standard"     # Comprehensive validation
    STRICT = "strict"         # Maximum validation with tight tolerances
    PARANOID = "paranoid"     # Exhaustive validation for critical applications


@dataclass
class ValidationIssue:
    """Represents a validation issue found during algorithm verification"""
    level: str  # ERROR, WARNING, INFO
    category: str  # OVERLAP, BOUNDARY, ROTATION, MATERIAL, EFFICIENCY, DATA
    message: str
    panel_id: Optional[str] = None
    sheet_id: Optional[int] = None
    coordinates: Optional[Tuple[float, float, float, float]] = None  # x, y, width, height
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report for algorithm results"""
    total_checks: int
    passed_checks: int
    issues: List[ValidationIssue]
    validation_level: ValidationLevel
    execution_time: float
    summary: Dict[str, Any]

    @property
    def has_errors(self) -> bool:
        """Check if any ERROR level issues were found"""
        return any(issue.level == "ERROR" for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if any WARNING level issues were found"""
        return any(issue.level == "WARNING" for issue in self.issues)

    @property
    def error_count(self) -> int:
        """Count of ERROR level issues"""
        return sum(1 for issue in self.issues if issue.level == "ERROR")

    @property
    def warning_count(self) -> int:
        """Count of WARNING level issues"""
        return sum(1 for issue in self.issues if issue.level == "WARNING")


class AlgorithmCorrectnessValidator:
    """
    Comprehensive validator for steel cutting optimization algorithm correctness

    Validates multiple aspects of algorithm output:
    1. Geometric Correctness: No overlaps, within boundaries, valid rotations
    2. Physical Constraints: Material properties, cutting limitations, kerf width
    3. Mathematical Accuracy: Area calculations, efficiency metrics, panel counts
    4. Data Integrity: Consistent panel tracking, material assignments, sheet usage
    """

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD,
                 tolerance: float = 0.001):
        """
        Initialize validator with specified validation level and numerical tolerance

        Args:
            validation_level: Strictness of validation checks
            tolerance: Numerical tolerance for floating-point comparisons (mm)
        """
        self.validation_level = validation_level
        self.tolerance = tolerance
        self.logger = self._setup_logging()

        # Validation thresholds based on level
        self.thresholds = self._get_validation_thresholds()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for validation process"""
        logger = logging.getLogger('algorithm_validator')
        logger.setLevel(logging.DEBUG)
        return logger

    def _get_validation_thresholds(self) -> Dict[str, float]:
        """Get validation thresholds based on validation level"""
        thresholds = {
            ValidationLevel.BASIC: {
                'overlap_tolerance': 1.0,          # Allow 1mm overlap
                'boundary_tolerance': 1.0,         # Allow 1mm boundary violation
                'efficiency_tolerance': 0.05,      # 5% efficiency tolerance
                'area_tolerance': 10.0             # 10mm² area calculation tolerance
            },
            ValidationLevel.STANDARD: {
                'overlap_tolerance': 0.1,          # Allow 0.1mm overlap
                'boundary_tolerance': 0.1,         # Allow 0.1mm boundary violation
                'efficiency_tolerance': 0.01,      # 1% efficiency tolerance
                'area_tolerance': 1.0              # 1mm² area calculation tolerance
            },
            ValidationLevel.STRICT: {
                'overlap_tolerance': 0.01,         # Allow 0.01mm overlap
                'boundary_tolerance': 0.01,        # Allow 0.01mm boundary violation
                'efficiency_tolerance': 0.001,     # 0.1% efficiency tolerance
                'area_tolerance': 0.1              # 0.1mm² area calculation tolerance
            },
            ValidationLevel.PARANOID: {
                'overlap_tolerance': 0.0,          # No overlap allowed
                'boundary_tolerance': 0.0,         # No boundary violation allowed
                'efficiency_tolerance': 0.0001,    # 0.01% efficiency tolerance
                'area_tolerance': 0.01             # 0.01mm² area calculation tolerance
            }
        }
        return thresholds[self.validation_level]

    def validate_placement_results(self, results: List[PlacementResult],
                                 input_panels: List[Panel],
                                 constraints: OptimizationConstraints) -> ValidationReport:
        """
        Comprehensive validation of placement results

        Args:
            results: List of placement results from optimization algorithm
            input_panels: Original input panels before optimization
            constraints: Optimization constraints used

        Returns:
            ValidationReport with detailed findings
        """
        import time
        start_time = time.time()

        issues = []
        total_checks = 0

        self.logger.info(f"Starting algorithm correctness validation at {self.validation_level.value} level")

        # 1. Geometric Validation
        geometric_issues, geometric_checks = self._validate_geometric_correctness(results)
        issues.extend(geometric_issues)
        total_checks += geometric_checks

        # 2. Physical Constraint Validation
        physical_issues, physical_checks = self._validate_physical_constraints(results, constraints)
        issues.extend(physical_issues)
        total_checks += physical_checks

        # 3. Mathematical Validation
        math_issues, math_checks = self._validate_mathematical_accuracy(results, input_panels)
        issues.extend(math_issues)
        total_checks += math_checks

        # 4. Data Integrity Validation
        data_issues, data_checks = self._validate_data_integrity(results, input_panels)
        issues.extend(data_issues)
        total_checks += data_checks

        # 5. Performance Validation (for non-basic levels)
        if self.validation_level != ValidationLevel.BASIC:
            perf_issues, perf_checks = self._validate_performance_metrics(results, input_panels, constraints)
            issues.extend(perf_issues)
            total_checks += perf_checks

        execution_time = time.time() - start_time
        passed_checks = total_checks - len([i for i in issues if i.level == "ERROR"])

        # Generate summary
        summary = self._generate_validation_summary(results, input_panels, issues)

        self.logger.info(f"Validation completed: {passed_checks}/{total_checks} checks passed, "
                        f"{len(issues)} issues found in {execution_time:.3f}s")

        return ValidationReport(
            total_checks=total_checks,
            passed_checks=passed_checks,
            issues=issues,
            validation_level=self.validation_level,
            execution_time=execution_time,
            summary=summary
        )

    def _validate_geometric_correctness(self, results: List[PlacementResult]) -> Tuple[List[ValidationIssue], int]:
        """Validate geometric placement correctness"""
        issues = []
        checks = 0

        for sheet_idx, result in enumerate(results):
            sheet = result.sheet
            placed_panels = result.panels

            # Check each panel for boundary compliance
            for panel in placed_panels:
                checks += 1
                boundary_issue = self._check_panel_boundaries(panel, sheet, sheet_idx + 1)
                if boundary_issue:
                    issues.append(boundary_issue)

                # Check rotation validity
                checks += 1
                rotation_issue = self._check_rotation_validity(panel, sheet_idx + 1)
                if rotation_issue:
                    issues.append(rotation_issue)

            # Check all panel pairs for overlaps
            for i, panel1 in enumerate(placed_panels):
                for j, panel2 in enumerate(placed_panels[i+1:], i+1):
                    checks += 1
                    overlap_issue = self._check_panel_overlap(panel1, panel2, sheet_idx + 1)
                    if overlap_issue:
                        issues.append(overlap_issue)

        return issues, checks

    def _validate_physical_constraints(self, results: List[PlacementResult],
                                     constraints: OptimizationConstraints) -> Tuple[List[ValidationIssue], int]:
        """Validate physical constraint compliance"""
        issues = []
        checks = 0

        for sheet_idx, result in enumerate(results):
            # Check material consistency
            if constraints.material_separation:
                checks += 1
                material_issue = self._check_material_consistency(result, sheet_idx + 1)
                if material_issue:
                    issues.append(material_issue)

            # Check kerf width consideration (if specified)
            if constraints.kerf_width > 0:
                for panel in result.panels:
                    checks += 1
                    kerf_issue = self._check_kerf_compliance(panel, constraints.kerf_width, sheet_idx + 1)
                    if kerf_issue:
                        issues.append(kerf_issue)

            # Check sheet size constraints
            checks += 1
            sheet_issue = self._check_sheet_size_validity(result.sheet, sheet_idx + 1)
            if sheet_issue:
                issues.append(sheet_issue)

        return issues, checks

    def _validate_mathematical_accuracy(self, results: List[PlacementResult],
                                      input_panels: List[Panel]) -> Tuple[List[ValidationIssue], int]:
        """Validate mathematical calculations"""
        issues = []
        checks = 0

        for sheet_idx, result in enumerate(results):
            # Validate efficiency calculation
            checks += 1
            efficiency_issue = self._check_efficiency_calculation(result, sheet_idx + 1)
            if efficiency_issue:
                issues.append(efficiency_issue)

            # Validate waste area calculation
            checks += 1
            waste_issue = self._check_waste_area_calculation(result, sheet_idx + 1)
            if waste_issue:
                issues.append(waste_issue)

            # Validate individual panel area calculations
            for panel in result.panels:
                checks += 1
                area_issue = self._check_panel_area_calculation(panel, sheet_idx + 1)
                if area_issue:
                    issues.append(area_issue)

        return issues, checks

    def _validate_data_integrity(self, results: List[PlacementResult],
                               input_panels: List[Panel]) -> Tuple[List[ValidationIssue], int]:
        """Validate data integrity and consistency"""
        issues = []
        checks = 0

        # Check total panel count consistency
        checks += 1
        count_issue = self._check_panel_count_consistency(results, input_panels)
        if count_issue:
            issues.append(count_issue)

        # Check for duplicate panel placements
        checks += 1
        duplicate_issue = self._check_duplicate_placements(results)
        if duplicate_issue:
            issues.append(duplicate_issue)

        # Check panel ID integrity
        checks += 1
        id_issue = self._check_panel_id_integrity(results, input_panels)
        if id_issue:
            issues.append(id_issue)

        return issues, checks

    def _validate_performance_metrics(self, results: List[PlacementResult],
                                    input_panels: List[Panel],
                                    constraints: OptimizationConstraints) -> Tuple[List[ValidationIssue], int]:
        """Validate performance-related metrics"""
        issues = []
        checks = 0

        # Check if target efficiency is met (if specified)
        if constraints.target_efficiency > 0:
            for sheet_idx, result in enumerate(results):
                checks += 1
                if result.efficiency < constraints.target_efficiency - self.thresholds['efficiency_tolerance']:
                    issues.append(ValidationIssue(
                        level="WARNING",
                        category="EFFICIENCY",
                        message=f"Sheet efficiency below target: {result.efficiency:.3f} < {constraints.target_efficiency:.3f}",
                        sheet_id=sheet_idx + 1,
                        expected_value=constraints.target_efficiency,
                        actual_value=result.efficiency
                    ))

        # Check sheet utilization efficiency
        checks += 1
        utilization_issue = self._check_sheet_utilization(results, input_panels)
        if utilization_issue:
            issues.append(utilization_issue)

        return issues, checks

    def _check_panel_boundaries(self, panel: PlacedPanel, sheet: SteelSheet, sheet_id: int) -> Optional[ValidationIssue]:
        """Check if panel is within sheet boundaries"""
        tolerance = self.thresholds['boundary_tolerance']

        # Check left/top boundaries
        if panel.x < -tolerance or panel.y < -tolerance:
            return ValidationIssue(
                level="ERROR",
                category="BOUNDARY",
                message=f"Panel extends beyond sheet boundaries (negative position)",
                panel_id=panel.panel.id,
                sheet_id=sheet_id,
                coordinates=(panel.x, panel.y, panel.actual_width, panel.actual_height),
                expected_value="≥ 0",
                actual_value=f"({panel.x:.3f}, {panel.y:.3f})"
            )

        # Check right/bottom boundaries
        right_edge = panel.x + panel.actual_width
        bottom_edge = panel.y + panel.actual_height

        if right_edge > sheet.width + tolerance:
            return ValidationIssue(
                level="ERROR",
                category="BOUNDARY",
                message=f"Panel extends beyond sheet width: {right_edge:.3f} > {sheet.width:.3f}",
                panel_id=panel.panel.id,
                sheet_id=sheet_id,
                coordinates=(panel.x, panel.y, panel.actual_width, panel.actual_height),
                expected_value=f"≤ {sheet.width}",
                actual_value=f"{right_edge:.3f}"
            )

        if bottom_edge > sheet.height + tolerance:
            return ValidationIssue(
                level="ERROR",
                category="BOUNDARY",
                message=f"Panel extends beyond sheet height: {bottom_edge:.3f} > {sheet.height:.3f}",
                panel_id=panel.panel.id,
                sheet_id=sheet_id,
                coordinates=(panel.x, panel.y, panel.actual_width, panel.actual_height),
                expected_value=f"≤ {sheet.height}",
                actual_value=f"{bottom_edge:.3f}"
            )

        return None

    def _check_panel_overlap(self, panel1: PlacedPanel, panel2: PlacedPanel, sheet_id: int) -> Optional[ValidationIssue]:
        """Check if two panels overlap"""
        tolerance = self.thresholds['overlap_tolerance']

        # Calculate overlap area
        x_overlap = max(0, min(panel1.x + panel1.actual_width, panel2.x + panel2.actual_width) -
                          max(panel1.x, panel2.x))
        y_overlap = max(0, min(panel1.y + panel1.actual_height, panel2.y + panel2.actual_height) -
                          max(panel1.y, panel2.y))

        overlap_area = x_overlap * y_overlap

        if overlap_area > tolerance:
            return ValidationIssue(
                level="ERROR",
                category="OVERLAP",
                message=f"Panel overlap detected: {overlap_area:.3f}mm² overlap between {panel1.panel.id} and {panel2.panel.id}",
                panel_id=f"{panel1.panel.id} & {panel2.panel.id}",
                sheet_id=sheet_id,
                expected_value=f"≤ {tolerance}",
                actual_value=f"{overlap_area:.3f}"
            )

        return None

    def _check_rotation_validity(self, panel: PlacedPanel, sheet_id: int) -> Optional[ValidationIssue]:
        """Check if panel rotation is valid"""
        if panel.rotated and not panel.panel.allow_rotation:
            return ValidationIssue(
                level="ERROR",
                category="ROTATION",
                message=f"Panel rotated but rotation not allowed",
                panel_id=panel.panel.id,
                sheet_id=sheet_id,
                expected_value="rotation=False",
                actual_value="rotation=True"
            )
        return None

    def _check_efficiency_calculation(self, result: PlacementResult, sheet_id: int) -> Optional[ValidationIssue]:
        """Check if efficiency calculation is correct"""
        if not result.panels:
            return None

        # Calculate actual efficiency
        total_panel_area = sum(panel.actual_width * panel.actual_height for panel in result.panels)
        calculated_efficiency = total_panel_area / result.sheet.area if result.sheet.area > 0 else 0

        efficiency_diff = abs(calculated_efficiency - result.efficiency)
        tolerance = self.thresholds['efficiency_tolerance']

        if efficiency_diff > tolerance:
            return ValidationIssue(
                level="ERROR",
                category="EFFICIENCY",
                message=f"Efficiency calculation error: calculated={calculated_efficiency:.6f}, reported={result.efficiency:.6f}",
                sheet_id=sheet_id,
                expected_value=f"{calculated_efficiency:.6f}",
                actual_value=f"{result.efficiency:.6f}"
            )

        return None

    def _check_waste_area_calculation(self, result: PlacementResult, sheet_id: int) -> Optional[ValidationIssue]:
        """Check if waste area calculation is correct"""
        if not result.panels:
            expected_waste = result.sheet.area
        else:
            total_panel_area = sum(panel.actual_width * panel.actual_height for panel in result.panels)
            expected_waste = result.sheet.area - total_panel_area

        waste_diff = abs(expected_waste - result.waste_area)
        tolerance = self.thresholds['area_tolerance']

        if waste_diff > tolerance:
            return ValidationIssue(
                level="ERROR",
                category="EFFICIENCY",
                message=f"Waste area calculation error: calculated={expected_waste:.3f}, reported={result.waste_area:.3f}",
                sheet_id=sheet_id,
                expected_value=f"{expected_waste:.3f}",
                actual_value=f"{result.waste_area:.3f}"
            )

        return None

    def _check_panel_area_calculation(self, panel: PlacedPanel, sheet_id: int) -> Optional[ValidationIssue]:
        """Check if individual panel area calculation is correct"""
        expected_area = panel.actual_width * panel.actual_height
        calculated_area = panel.panel.cutting_area if hasattr(panel.panel, 'cutting_area') else panel.panel.area

        area_diff = abs(expected_area - calculated_area)
        tolerance = self.thresholds['area_tolerance']

        if area_diff > tolerance:
            return ValidationIssue(
                level="WARNING",
                category="AREA",
                message=f"Panel area calculation discrepancy: expected={expected_area:.3f}, calculated={calculated_area:.3f}",
                panel_id=panel.panel.id,
                sheet_id=sheet_id,
                expected_value=f"{expected_area:.3f}",
                actual_value=f"{calculated_area:.3f}"
            )

        return None

    def _check_material_consistency(self, result: PlacementResult, sheet_id: int) -> Optional[ValidationIssue]:
        """Check if all panels on sheet have consistent material"""
        if not result.panels:
            return None

        expected_material = result.material_block
        for panel in result.panels:
            if panel.panel.material != expected_material:
                return ValidationIssue(
                    level="ERROR",
                    category="MATERIAL",
                    message=f"Material inconsistency: panel material {panel.panel.material} != sheet material {expected_material}",
                    panel_id=panel.panel.id,
                    sheet_id=sheet_id,
                    expected_value=expected_material,
                    actual_value=panel.panel.material
                )

        return None

    def _check_kerf_compliance(self, panel: PlacedPanel, kerf_width: float, sheet_id: int) -> Optional[ValidationIssue]:
        """Check if kerf width is properly considered"""
        # This is a placeholder for kerf width validation
        # Implementation depends on how kerf is handled in the algorithm
        return None

    def _check_sheet_size_validity(self, sheet: SteelSheet, sheet_id: int) -> Optional[ValidationIssue]:
        """Check if sheet size is valid"""
        # Standard size limits from business requirements
        max_width = 1500.0
        max_height = 3100.0

        if sheet.width > max_width or sheet.height > max_height:
            return ValidationIssue(
                level="WARNING",
                category="MATERIAL",
                message=f"Sheet size exceeds standard limits: {sheet.width}x{sheet.height} > {max_width}x{max_height}",
                sheet_id=sheet_id,
                expected_value=f"≤ {max_width}x{max_height}",
                actual_value=f"{sheet.width}x{sheet.height}"
            )

        return None

    def _check_panel_count_consistency(self, results: List[PlacementResult],
                                     input_panels: List[Panel]) -> Optional[ValidationIssue]:
        """Check if total placed panels match input panel quantities"""
        total_input = sum(panel.quantity for panel in input_panels)
        total_placed = sum(len(result.panels) for result in results)

        if total_placed > total_input:
            return ValidationIssue(
                level="ERROR",
                category="DATA",
                message=f"More panels placed than input: {total_placed} > {total_input}",
                expected_value=f"≤ {total_input}",
                actual_value=f"{total_placed}"
            )

        return None

    def _check_duplicate_placements(self, results: List[PlacementResult]) -> Optional[ValidationIssue]:
        """Check for duplicate panel placements"""
        placed_panel_ids = []
        for result in results:
            for panel in result.panels:
                panel_id = panel.panel.id
                if panel_id in placed_panel_ids:
                    return ValidationIssue(
                        level="ERROR",
                        category="DATA",
                        message=f"Duplicate panel placement detected: {panel_id}",
                        panel_id=panel_id
                    )
                placed_panel_ids.append(panel_id)

        return None

    def _check_panel_id_integrity(self, results: List[PlacementResult],
                                input_panels: List[Panel]) -> Optional[ValidationIssue]:
        """Check if all placed panels have valid IDs from input"""
        input_panel_ids = {panel.id for panel in input_panels}

        for result in results:
            for placed_panel in result.panels:
                panel_id = placed_panel.panel.id
                # Handle individual panel IDs (e.g., "panel_1", "panel_2" from "panel")
                base_id = panel_id.rsplit('_', 1)[0] if '_' in panel_id else panel_id

                if base_id not in input_panel_ids and panel_id not in input_panel_ids:
                    return ValidationIssue(
                        level="ERROR",
                        category="DATA",
                        message=f"Placed panel ID not found in input: {panel_id}",
                        panel_id=panel_id
                    )

        return None

    def _check_sheet_utilization(self, results: List[PlacementResult],
                               input_panels: List[Panel]) -> Optional[ValidationIssue]:
        """Check overall sheet utilization efficiency"""
        if not results:
            return None

        total_sheet_area = sum(result.sheet.area for result in results)
        total_panel_area = sum(
            sum(panel.actual_width * panel.actual_height for panel in result.panels)
            for result in results
        )

        overall_efficiency = total_panel_area / total_sheet_area if total_sheet_area > 0 else 0

        # Warning if efficiency is very low (suggests poor algorithm performance)
        if overall_efficiency < 0.3:  # Less than 30% efficiency
            return ValidationIssue(
                level="WARNING",
                category="EFFICIENCY",
                message=f"Very low overall material efficiency: {overall_efficiency:.1%}",
                expected_value="> 30%",
                actual_value=f"{overall_efficiency:.1%}"
            )

        return None

    def _generate_validation_summary(self, results: List[PlacementResult],
                                   input_panels: List[Panel],
                                   issues: List[ValidationIssue]) -> Dict[str, Any]:
        """Generate comprehensive validation summary"""
        total_input_panels = sum(panel.quantity for panel in input_panels)
        total_placed_panels = sum(len(result.panels) for result in results)
        placement_rate = (total_placed_panels / total_input_panels * 100) if total_input_panels > 0 else 0

        error_categories = {}
        warning_categories = {}

        for issue in issues:
            category = issue.category
            if issue.level == "ERROR":
                error_categories[category] = error_categories.get(category, 0) + 1
            elif issue.level == "WARNING":
                warning_categories[category] = warning_categories.get(category, 0) + 1

        return {
            'placement_summary': {
                'input_panels': total_input_panels,
                'placed_panels': total_placed_panels,
                'placement_rate_percent': placement_rate,
                'sheets_used': len(results)
            },
            'error_categories': error_categories,
            'warning_categories': warning_categories,
            'validation_level': self.validation_level.value,
            'tolerance_mm': self.tolerance
        }