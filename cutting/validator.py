"""
Enhanced Validator for Steel Cutting Operations
鋼板切断作業用強化バリデーター

Validates dimensions, constraints, and manufacturing feasibility
寸法、制約、製造可能性をバリデーション
"""

import logging
import math
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

from core.models import Panel, SteelSheet, PlacementResult, PlacedPanel, OptimizationConstraints
from cutting.instruction import WorkInstruction, CuttingInstruction, SafetyLevel


class ValidationLevel(Enum):
    """Validation strictness level"""
    BASIC = "basic"           # Basic dimension checks
    STANDARD = "standard"     # Standard manufacturing checks
    STRICT = "strict"         # Strict quality requirements
    PRODUCTION = "production" # Production-ready validation


class ValidationResult(Enum):
    """Validation result status"""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Individual validation issue"""
    level: ValidationResult
    category: str
    message: str
    suggestion: Optional[str] = None
    affected_items: List[str] = None
    japanese_message: Optional[str] = None

    def __post_init__(self):
        if self.affected_items is None:
            self.affected_items = []


@dataclass
class ValidationReport:
    """Complete validation report"""
    overall_result: ValidationResult
    issues: List[ValidationIssue]
    passed_checks: int
    total_checks: int
    execution_time: float
    validated_at: str

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate percentage"""
        return (self.passed_checks / self.total_checks * 100) if self.total_checks > 0 else 0.0

    @property
    def has_critical_issues(self) -> bool:
        """Check if there are critical issues"""
        return any(issue.level == ValidationResult.CRITICAL for issue in self.issues)

    @property
    def has_failures(self) -> bool:
        """Check if there are any failures"""
        return any(issue.level in [ValidationResult.FAIL, ValidationResult.CRITICAL] for issue in self.issues)


@dataclass
class MachineConstraints:
    """Machine-specific constraints"""
    max_cutting_length: float = 3200.0    # mm
    min_cutting_length: float = 10.0      # mm
    max_material_thickness: float = 50.0  # mm
    min_material_thickness: float = 1.0   # mm
    kerf_width_range: Tuple[float, float] = (2.0, 6.0)  # mm
    max_cutting_speed: float = 3000.0     # mm/min
    coolant_required_materials: Set[str] = None
    supported_materials: Set[str] = None

    def __post_init__(self):
        if self.coolant_required_materials is None:
            self.coolant_required_materials = {'SUS304', 'SUS316', 'SUS430'}
        if self.supported_materials is None:
            self.supported_materials = {'SS400', 'SUS304', 'SUS316', 'AL6061', 'AL5052', 'SUS430'}


@dataclass
class QualityStandards:
    """Quality standards for cutting operations"""
    max_dimensional_tolerance: float = 0.5    # mm
    max_edge_roughness: float = 3.2          # μm Ra
    max_heat_affected_zone: float = 1.0      # mm
    parallelism_tolerance: float = 0.2       # mm
    perpendicularity_tolerance: float = 0.3  # mm
    min_corner_radius: float = 0.5           # mm


class EnhancedValidator:
    """
    Enhanced validator for steel cutting operations
    鋼板切断作業用強化バリデーター
    """

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.logger = logging.getLogger(__name__)

        # Japanese standard specifications
        self.jp_standards = {
            'min_panel_size': (50.0, 50.0),      # JIS規格最小サイズ
            'max_panel_size': (1500.0, 3100.0),  # 標準シートサイズ
            'standard_thicknesses': [1.6, 2.3, 3.2, 4.5, 6.0, 9.0, 12.0, 16.0, 19.0, 25.0],
            'material_compatibility': {
                'SS400': ['structural', 'general'],
                'SUS304': ['food_grade', 'chemical'],
                'SUS316': ['marine', 'chemical'],
                'AL6061': ['lightweight', 'aerospace'],
                'AL5052': ['marine', 'general']
            }
        }

        # Default machine constraints
        self.machine_constraints = MachineConstraints()
        self.quality_standards = QualityStandards()

    def validate_panels(
        self,
        panels: List[Panel],
        constraints: Optional[OptimizationConstraints] = None
    ) -> ValidationReport:
        """
        Validate panel specifications
        パネル仕様をバリデーション
        """
        start_time = time.time()
        issues = []
        checks_passed = 0
        total_checks = 0

        self.logger.info(f"Validating {len(panels)} panels with {self.validation_level.value} level")

        for panel in panels:
            panel_issues, panel_checks = self._validate_single_panel(panel)
            issues.extend(panel_issues)
            checks_passed += panel_checks['passed']
            total_checks += panel_checks['total']

        # Cross-panel validation
        cross_issues, cross_checks = self._validate_panel_compatibility(panels)
        issues.extend(cross_issues)
        checks_passed += cross_checks['passed']
        total_checks += cross_checks['total']

        # Determine overall result
        overall_result = self._determine_overall_result(issues)

        execution_time = time.time() - start_time

        return ValidationReport(
            overall_result=overall_result,
            issues=issues,
            passed_checks=checks_passed,
            total_checks=total_checks,
            execution_time=execution_time,
            validated_at=datetime.now().isoformat()
        )

    def validate_placement(
        self,
        placement_result: PlacementResult,
        sheet: SteelSheet,
        constraints: OptimizationConstraints
    ) -> ValidationReport:
        """
        Validate placement result
        配置結果をバリデーション
        """
        start_time = time.time()
        issues = []
        checks_passed = 0
        total_checks = 0

        self.logger.info(f"Validating placement with {len(placement_result.panels)} placed panels")

        # Validate sheet constraints
        sheet_issues, sheet_checks = self._validate_sheet_constraints(placement_result, sheet)
        issues.extend(sheet_issues)
        checks_passed += sheet_checks['passed']
        total_checks += sheet_checks['total']

        # Validate guillotine constraints
        guillotine_issues, guillotine_checks = self._validate_guillotine_constraints(placement_result, constraints)
        issues.extend(guillotine_issues)
        checks_passed += guillotine_checks['passed']
        total_checks += guillotine_checks['total']

        # Validate panel placement
        placement_issues, placement_checks = self._validate_panel_placement(placement_result.panels, sheet)
        issues.extend(placement_issues)
        checks_passed += placement_checks['passed']
        total_checks += placement_checks['total']

        # Efficiency validation
        efficiency_issues, efficiency_checks = self._validate_efficiency(placement_result, constraints)
        issues.extend(efficiency_issues)
        checks_passed += efficiency_checks['passed']
        total_checks += efficiency_checks['total']

        overall_result = self._determine_overall_result(issues)
        execution_time = time.time() - start_time

        return ValidationReport(
            overall_result=overall_result,
            issues=issues,
            passed_checks=checks_passed,
            total_checks=total_checks,
            execution_time=execution_time,
            validated_at=datetime.now().isoformat()
        )

    def validate_work_instruction(
        self,
        work_instruction: WorkInstruction
    ) -> ValidationReport:
        """
        Validate work instruction feasibility
        作業指示の実行可能性をバリデーション
        """
        start_time = time.time()
        issues = []
        checks_passed = 0
        total_checks = 0

        self.logger.info(f"Validating work instruction with {work_instruction.total_steps} steps")

        # Validate cutting sequence
        sequence_issues, sequence_checks = self._validate_cutting_sequence(work_instruction.cutting_sequence)
        issues.extend(sequence_issues)
        checks_passed += sequence_checks['passed']
        total_checks += sequence_checks['total']

        # Validate machine compatibility
        machine_issues, machine_checks = self._validate_machine_compatibility(work_instruction)
        issues.extend(machine_issues)
        checks_passed += machine_checks['passed']
        total_checks += machine_checks['total']

        # Validate safety requirements
        safety_issues, safety_checks = self._validate_safety_requirements(work_instruction)
        issues.extend(safety_issues)
        checks_passed += safety_checks['passed']
        total_checks += safety_checks['total']

        # Validate time estimates
        time_issues, time_checks = self._validate_time_estimates(work_instruction)
        issues.extend(time_issues)
        checks_passed += time_checks['passed']
        total_checks += time_checks['total']

        overall_result = self._determine_overall_result(issues)
        execution_time = time.time() - start_time

        return ValidationReport(
            overall_result=overall_result,
            issues=issues,
            passed_checks=checks_passed,
            total_checks=total_checks,
            execution_time=execution_time,
            validated_at=datetime.now().isoformat()
        )

    def _validate_single_panel(self, panel: Panel) -> Tuple[List[ValidationIssue], Dict[str, int]]:
        """Validate individual panel specifications"""
        issues = []
        passed = 0
        total = 0

        # Dimension validation
        total += 1
        min_w, min_h = self.jp_standards['min_panel_size']
        max_w, max_h = self.jp_standards['max_panel_size']

        if panel.width < min_w or panel.height < min_h:
            issues.append(ValidationIssue(
                level=ValidationResult.FAIL,
                category="dimensions",
                message=f"Panel {panel.id} below minimum size: {panel.width}×{panel.height}mm",
                japanese_message=f"パネル {panel.id} は最小サイズ未満: {panel.width}×{panel.height}mm",
                suggestion=f"Increase size to at least {min_w}×{min_h}mm",
                affected_items=[panel.id]
            ))
        elif panel.width > max_w or panel.height > max_h:
            issues.append(ValidationIssue(
                level=ValidationResult.FAIL,
                category="dimensions",
                message=f"Panel {panel.id} exceeds maximum size: {panel.width}×{panel.height}mm",
                japanese_message=f"パネル {panel.id} は最大サイズ超過: {panel.width}×{panel.height}mm",
                suggestion=f"Reduce size to within {max_w}×{max_h}mm or split panel",
                affected_items=[panel.id]
            ))
        else:
            passed += 1

        # Material validation
        total += 1
        if panel.material not in self.machine_constraints.supported_materials:
            issues.append(ValidationIssue(
                level=ValidationResult.CRITICAL,
                category="material",
                message=f"Unsupported material: {panel.material}",
                japanese_message=f"未対応材質: {panel.material}",
                suggestion=f"Use supported materials: {', '.join(self.machine_constraints.supported_materials)}",
                affected_items=[panel.id]
            ))
        else:
            passed += 1

        # Thickness validation
        total += 1
        if not (self.machine_constraints.min_material_thickness <=
                panel.thickness <= self.machine_constraints.max_material_thickness):
            issues.append(ValidationIssue(
                level=ValidationResult.FAIL,
                category="thickness",
                message=f"Panel {panel.id} thickness {panel.thickness}mm outside machine limits",
                japanese_message=f"パネル {panel.id} の板厚 {panel.thickness}mm は機械制限外",
                suggestion=f"Use thickness between {self.machine_constraints.min_material_thickness}-{self.machine_constraints.max_material_thickness}mm",
                affected_items=[panel.id]
            ))
        else:
            passed += 1

        # Quantity validation
        total += 1
        if panel.quantity <= 0:
            issues.append(ValidationIssue(
                level=ValidationResult.FAIL,
                category="quantity",
                message=f"Invalid quantity for panel {panel.id}: {panel.quantity}",
                japanese_message=f"パネル {panel.id} の数量が無効: {panel.quantity}",
                suggestion="Set quantity to positive integer",
                affected_items=[panel.id]
            ))
        elif panel.quantity > 100:
            issues.append(ValidationIssue(
                level=ValidationResult.WARNING,
                category="quantity",
                message=f"Large quantity for panel {panel.id}: {panel.quantity}",
                japanese_message=f"パネル {panel.id} の数量が多い: {panel.quantity}",
                suggestion="Consider batch processing for large quantities",
                affected_items=[panel.id]
            ))
            passed += 1
        else:
            passed += 1

        # Standard thickness check (warning only)
        if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PRODUCTION]:
            total += 1
            if panel.thickness not in self.jp_standards['standard_thicknesses']:
                issues.append(ValidationIssue(
                    level=ValidationResult.WARNING,
                    category="standards",
                    message=f"Non-standard thickness {panel.thickness}mm for panel {panel.id}",
                    japanese_message=f"パネル {panel.id} の板厚 {panel.thickness}mm は非標準",
                    suggestion=f"Consider standard thicknesses: {self.jp_standards['standard_thicknesses']}",
                    affected_items=[panel.id]
                ))
            else:
                passed += 1

        return issues, {'passed': passed, 'total': total}

    def _validate_panel_compatibility(self, panels: List[Panel]) -> Tuple[List[ValidationIssue], Dict[str, int]]:
        """Validate compatibility between panels"""
        issues = []
        passed = 0
        total = 0

        # Check material mixing
        total += 1
        materials = set(panel.material for panel in panels)
        if len(materials) > 3:
            issues.append(ValidationIssue(
                level=ValidationResult.WARNING,
                category="materials",
                message=f"Many different materials ({len(materials)}) may complicate setup",
                japanese_message=f"多種材質 ({len(materials)}) でセットアップが複雑化",
                suggestion="Consider grouping by material for batch processing"
            ))
        else:
            passed += 1

        # Check thickness variation
        total += 1
        thicknesses = set(panel.thickness for panel in panels)
        if len(thicknesses) > 5:
            issues.append(ValidationIssue(
                level=ValidationResult.WARNING,
                category="thickness",
                message=f"Many different thicknesses ({len(thicknesses)}) may require frequent setup changes",
                japanese_message=f"多種板厚 ({len(thicknesses)}) で段取り変更が頻繁",
                suggestion="Group panels by thickness when possible"
            ))
        else:
            passed += 1

        # Check for duplicate panels
        total += 1
        panel_signatures = {}
        duplicates = []

        for panel in panels:
            signature = (panel.width, panel.height, panel.material, panel.thickness)
            if signature in panel_signatures:
                duplicates.append((panel.id, panel_signatures[signature]))
            else:
                panel_signatures[signature] = panel.id

        if duplicates:
            issues.append(ValidationIssue(
                level=ValidationResult.WARNING,
                category="efficiency",
                message=f"Found {len(duplicates)} duplicate panel specifications",
                japanese_message=f"{len(duplicates)} 個の重複パネル仕様を発見",
                suggestion="Consider consolidating duplicate panels with increased quantities",
                affected_items=[dup[0] for dup in duplicates]
            ))
        else:
            passed += 1

        return issues, {'passed': passed, 'total': total}

    def _validate_sheet_constraints(
        self,
        placement_result: PlacementResult,
        sheet: SteelSheet
    ) -> Tuple[List[ValidationIssue], Dict[str, int]]:
        """Validate sheet constraint compliance"""
        issues = []
        passed = 0
        total = 0

        # Check sheet size limits
        total += 1
        max_w, max_h = self.jp_standards['max_panel_size']
        if sheet.width > max_w or sheet.height > max_h:
            issues.append(ValidationIssue(
                level=ValidationResult.WARNING,
                category="sheet_size",
                message=f"Non-standard sheet size: {sheet.width}×{sheet.height}mm",
                japanese_message=f"非標準シートサイズ: {sheet.width}×{sheet.height}mm",
                suggestion=f"Standard size is {max_w}×{max_h}mm"
            ))
        else:
            passed += 1

        # Check boundary compliance
        total += 1
        boundary_violations = []
        for placed_panel in placement_result.panels:
            x1, y1, x2, y2 = placed_panel.bounds
            if x1 < 0 or y1 < 0 or x2 > sheet.width or y2 > sheet.height:
                boundary_violations.append(placed_panel.panel.id)

        if boundary_violations:
            issues.append(ValidationIssue(
                level=ValidationResult.CRITICAL,
                category="boundaries",
                message="Panels exceed sheet boundaries",
                japanese_message="パネルがシート境界を超過",
                suggestion="Adjust panel placement or use larger sheet",
                affected_items=boundary_violations
            ))
        else:
            passed += 1

        return issues, {'passed': passed, 'total': total}

    def _validate_guillotine_constraints(
        self,
        placement_result: PlacementResult,
        constraints: OptimizationConstraints
    ) -> Tuple[List[ValidationIssue], Dict[str, int]]:
        """Validate guillotine cutting constraints"""
        issues = []
        passed = 0
        total = 0

        # Check for overlapping panels
        total += 1
        overlaps = self._find_overlapping_panels(placement_result.panels)
        if overlaps:
            issues.append(ValidationIssue(
                level=ValidationResult.CRITICAL,
                category="overlaps",
                message=f"Found {len(overlaps)} overlapping panels",
                japanese_message=f"{len(overlaps)} 個の重複パネルを発見",
                suggestion="Resolve panel overlaps before cutting",
                affected_items=[f"{p1}-{p2}" for p1, p2 in overlaps]
            ))
        else:
            passed += 1

        # Check kerf allowance
        total += 1
        insufficient_kerf = self._check_kerf_allowance(placement_result.panels, constraints.kerf_width)
        if insufficient_kerf:
            issues.append(ValidationIssue(
                level=ValidationResult.FAIL,
                category="kerf",
                message="Insufficient kerf allowance between panels",
                japanese_message="パネル間のケルフ余裕不足",
                suggestion=f"Ensure at least {constraints.kerf_width}mm between panels",
                affected_items=insufficient_kerf
            ))
        else:
            passed += 1

        return issues, {'passed': passed, 'total': total}

    def _validate_panel_placement(
        self,
        placed_panels: List[PlacedPanel],
        sheet: SteelSheet
    ) -> Tuple[List[ValidationIssue], Dict[str, int]]:
        """Validate individual panel placements"""
        issues = []
        passed = 0
        total = 0

        for placed_panel in placed_panels:
            # Check rotation validity
            total += 1
            if placed_panel.rotated and not placed_panel.panel.allow_rotation:
                issues.append(ValidationIssue(
                    level=ValidationResult.FAIL,
                    category="rotation",
                    message=f"Panel {placed_panel.panel.id} rotated despite rotation prohibition",
                    japanese_message=f"回転禁止のパネル {placed_panel.panel.id} が回転",
                    suggestion="Disable rotation or allow rotation for this panel",
                    affected_items=[placed_panel.panel.id]
                ))
            else:
                passed += 1

            # Check minimum distance from edges (production level)
            if self.validation_level == ValidationLevel.PRODUCTION:
                total += 1
                edge_margin = 10.0  # mm
                x1, y1, x2, y2 = placed_panel.bounds

                if (x1 < edge_margin or y1 < edge_margin or
                    x2 > sheet.width - edge_margin or y2 > sheet.height - edge_margin):
                    issues.append(ValidationIssue(
                        level=ValidationResult.WARNING,
                        category="edge_distance",
                        message=f"Panel {placed_panel.panel.id} too close to sheet edge",
                        japanese_message=f"パネル {placed_panel.panel.id} がシート端に近すぎ",
                        suggestion=f"Maintain at least {edge_margin}mm from edges",
                        affected_items=[placed_panel.panel.id]
                    ))
                else:
                    passed += 1

        return issues, {'passed': passed, 'total': total}

    def _validate_efficiency(
        self,
        placement_result: PlacementResult,
        constraints: OptimizationConstraints
    ) -> Tuple[List[ValidationIssue], Dict[str, int]]:
        """Validate efficiency requirements"""
        issues = []
        passed = 0
        total = 0

        # Check target efficiency
        total += 1
        if placement_result.efficiency < constraints.target_efficiency:
            severity = ValidationResult.WARNING if placement_result.efficiency > 0.5 else ValidationResult.FAIL
            issues.append(ValidationIssue(
                level=severity,
                category="efficiency",
                message=f"Efficiency {placement_result.efficiency:.1%} below target {constraints.target_efficiency:.1%}",
                japanese_message=f"効率 {placement_result.efficiency:.1%} が目標 {constraints.target_efficiency:.1%} 未満",
                suggestion="Consider different algorithm or panel arrangement"
            ))
        else:
            passed += 1

        # Check waste area (strict/production levels)
        if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PRODUCTION]:
            total += 1
            waste_threshold = 0.3  # 30% waste is concerning
            if placement_result.efficiency < (1.0 - waste_threshold):
                issues.append(ValidationIssue(
                    level=ValidationResult.WARNING,
                    category="waste",
                    message=f"High waste area: {placement_result.waste_area:.0f}mm²",
                    japanese_message=f"廃材面積大: {placement_result.waste_area:.0f}mm²",
                    suggestion="Review panel arrangement for better material utilization"
                ))
            else:
                passed += 1

        return issues, {'passed': passed, 'total': total}

    def _validate_cutting_sequence(
        self,
        cutting_sequence: List[CuttingInstruction]
    ) -> Tuple[List[ValidationIssue], Dict[str, int]]:
        """Validate cutting sequence feasibility"""
        issues = []
        passed = 0
        total = 0

        # Check for valid guillotine cuts
        total += 1
        non_guillotine = []
        for instruction in cutting_sequence:
            if not instruction.is_edge_to_edge:
                non_guillotine.append(str(instruction.step_number))

        if non_guillotine:
            issues.append(ValidationIssue(
                level=ValidationResult.CRITICAL,
                category="guillotine",
                message=f"Non-guillotine cuts detected in steps: {', '.join(non_guillotine)}",
                japanese_message=f"非ギロチン切断を検出 ステップ: {', '.join(non_guillotine)}",
                suggestion="Ensure all cuts go from edge to edge",
                affected_items=non_guillotine
            ))
        else:
            passed += 1

        # Check cut length limits
        total += 1
        excessive_cuts = []
        for instruction in cutting_sequence:
            if instruction.dimension > self.machine_constraints.max_cutting_length:
                excessive_cuts.append(str(instruction.step_number))

        if excessive_cuts:
            issues.append(ValidationIssue(
                level=ValidationResult.FAIL,
                category="cut_length",
                message=f"Cuts exceed machine limit in steps: {', '.join(excessive_cuts)}",
                japanese_message=f"機械制限超過 ステップ: {', '.join(excessive_cuts)}",
                suggestion=f"Split cuts to under {self.machine_constraints.max_cutting_length}mm",
                affected_items=excessive_cuts
            ))
        else:
            passed += 1

        return issues, {'passed': passed, 'total': total}

    def _validate_machine_compatibility(
        self,
        work_instruction: WorkInstruction
    ) -> Tuple[List[ValidationIssue], Dict[str, int]]:
        """Validate machine compatibility"""
        issues = []
        passed = 0
        total = 0

        # Check material compatibility
        total += 1
        if work_instruction.material_type not in self.machine_constraints.supported_materials:
            issues.append(ValidationIssue(
                level=ValidationResult.CRITICAL,
                category="machine_material",
                message=f"Material {work_instruction.material_type} not supported by machine",
                japanese_message=f"材質 {work_instruction.material_type} は機械未対応",
                suggestion=f"Use supported materials: {', '.join(self.machine_constraints.supported_materials)}"
            ))
        else:
            passed += 1

        # Check kerf width
        total += 1
        kerf_min, kerf_max = self.machine_constraints.kerf_width_range
        if not (kerf_min <= work_instruction.kerf_width <= kerf_max):
            issues.append(ValidationIssue(
                level=ValidationResult.WARNING,
                category="kerf_width",
                message=f"Kerf width {work_instruction.kerf_width}mm outside typical range",
                japanese_message=f"ケルフ幅 {work_instruction.kerf_width}mm が標準範囲外",
                suggestion=f"Consider kerf width between {kerf_min}-{kerf_max}mm"
            ))
        else:
            passed += 1

        return issues, {'passed': passed, 'total': total}

    def _validate_safety_requirements(
        self,
        work_instruction: WorkInstruction
    ) -> Tuple[List[ValidationIssue], Dict[str, int]]:
        """Validate safety requirements"""
        issues = []
        passed = 0
        total = 0

        # Check for high-risk operations
        total += 1
        high_risk_steps = [
            inst.step_number for inst in work_instruction.cutting_sequence
            if inst.safety_level in [SafetyLevel.HIGH, SafetyLevel.CRITICAL]
        ]

        if len(high_risk_steps) > len(work_instruction.cutting_sequence) * 0.3:
            issues.append(ValidationIssue(
                level=ValidationResult.WARNING,
                category="safety_risk",
                message=f"High proportion of high-risk operations: {len(high_risk_steps)}/{len(work_instruction.cutting_sequence)}",
                japanese_message=f"高リスク作業の割合が高い: {len(high_risk_steps)}/{len(work_instruction.cutting_sequence)}",
                suggestion="Review sequence to minimize risk concentration"
            ))
        else:
            passed += 1

        # Check coolant requirements
        total += 1
        if (work_instruction.material_type in self.machine_constraints.coolant_required_materials and
            not work_instruction.machine_settings.get('coolant_required', False)):
            issues.append(ValidationIssue(
                level=ValidationResult.FAIL,
                category="coolant",
                message=f"Coolant required for material {work_instruction.material_type}",
                japanese_message=f"材質 {work_instruction.material_type} にはクーラント必要",
                suggestion="Enable coolant in machine settings"
            ))
        else:
            passed += 1

        return issues, {'passed': passed, 'total': total}

    def _validate_time_estimates(
        self,
        work_instruction: WorkInstruction
    ) -> Tuple[List[ValidationIssue], Dict[str, int]]:
        """Validate time estimates"""
        issues = []
        passed = 0
        total = 0

        # Check for unrealistic time estimates
        total += 1
        avg_time_per_step = work_instruction.estimated_total_time / work_instruction.total_steps
        if avg_time_per_step < 0.5:  # Less than 30 seconds per step
            issues.append(ValidationIssue(
                level=ValidationResult.WARNING,
                category="time_estimate",
                message=f"Very fast estimated time: {avg_time_per_step:.1f}min per step",
                japanese_message=f"非常に短い見積時間: 1ステップ {avg_time_per_step:.1f}分",
                suggestion="Verify time estimates include setup and handling"
            ))
        elif avg_time_per_step > 10:  # More than 10 minutes per step
            issues.append(ValidationIssue(
                level=ValidationResult.WARNING,
                category="time_estimate",
                message=f"Very slow estimated time: {avg_time_per_step:.1f}min per step",
                japanese_message=f"非常に長い見積時間: 1ステップ {avg_time_per_step:.1f}分",
                suggestion="Review cutting parameters for efficiency"
            ))
        else:
            passed += 1

        return issues, {'passed': passed, 'total': total}

    def _find_overlapping_panels(self, placed_panels: List[PlacedPanel]) -> List[Tuple[str, str]]:
        """Find overlapping panels"""
        overlaps = []

        for i, panel1 in enumerate(placed_panels):
            for j, panel2 in enumerate(placed_panels[i+1:], i+1):
                if self._panels_overlap(panel1, panel2):
                    overlaps.append((panel1.panel.id, panel2.panel.id))

        return overlaps

    def _panels_overlap(self, panel1: PlacedPanel, panel2: PlacedPanel) -> bool:
        """Check if two panels overlap"""
        x1_1, y1_1, x2_1, y2_1 = panel1.bounds
        x1_2, y1_2, x2_2, y2_2 = panel2.bounds

        return not (x2_1 <= x1_2 or x2_2 <= x1_1 or y2_1 <= y1_2 or y2_2 <= y1_1)

    def _check_kerf_allowance(self, placed_panels: List[PlacedPanel], kerf_width: float) -> List[str]:
        """Check for insufficient kerf allowance between panels"""
        insufficient = []

        for i, panel1 in enumerate(placed_panels):
            for j, panel2 in enumerate(placed_panels[i+1:], i+1):
                distance = self._calculate_panel_distance(panel1, panel2)
                if 0 < distance < kerf_width:
                    insufficient.append(f"{panel1.panel.id}-{panel2.panel.id}")

        return insufficient

    def _calculate_panel_distance(self, panel1: PlacedPanel, panel2: PlacedPanel) -> float:
        """Calculate minimum distance between two panels"""
        x1_1, y1_1, x2_1, y2_1 = panel1.bounds
        x1_2, y1_2, x2_2, y2_2 = panel2.bounds

        # Calculate minimum distance between rectangles
        dx = max(0, max(x1_1 - x2_2, x1_2 - x2_1))
        dy = max(0, max(y1_1 - y2_2, y1_2 - y2_1))

        return math.sqrt(dx**2 + dy**2)

    def _determine_overall_result(self, issues: List[ValidationIssue]) -> ValidationResult:
        """Determine overall validation result"""
        if any(issue.level == ValidationResult.CRITICAL for issue in issues):
            return ValidationResult.CRITICAL
        elif any(issue.level == ValidationResult.FAIL for issue in issues):
            return ValidationResult.FAIL
        elif any(issue.level == ValidationResult.WARNING for issue in issues):
            return ValidationResult.WARNING
        else:
            return ValidationResult.PASS


def create_enhanced_validator(validation_level: ValidationLevel = ValidationLevel.STANDARD) -> EnhancedValidator:
    """Create enhanced validator instance"""
    return EnhancedValidator(validation_level)
