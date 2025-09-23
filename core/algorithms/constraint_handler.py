"""
Complex Constraint Handling for GPU-Accelerated Bin Packing

Advanced constraint management system for steel cutting optimization including:
- Rotation constraints and optimization
- Material compatibility and grouping
- Guillotine cut constraints
- Kerf width considerations
- Thermal cutting limitations
"""

import logging
import numpy as np
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from core.models import Panel, SteelSheet, PlacedPanel

logger = logging.getLogger(__name__)

class RotationStrategy(Enum):
    """Panel rotation strategies"""
    NO_ROTATION = "no_rotation"
    ALLOW_90_DEGREES = "allow_90_degrees"
    OPTIMIZE_ORIENTATION = "optimize_orientation"
    MINIMIZE_WASTE = "minimize_waste"

class MaterialGrouping(Enum):
    """Material grouping strategies"""
    STRICT_SEPARATION = "strict_separation"
    COMPATIBLE_MIXING = "compatible_mixing"
    THICKNESS_GROUPING = "thickness_grouping"
    PRIORITY_BASED = "priority_based"

@dataclass
class ConstraintViolation:
    """Represents a constraint violation with details"""
    constraint_type: str
    severity: str  # "error", "warning", "info"
    description: str
    affected_panels: List[str]
    suggested_fix: Optional[str] = None

@dataclass
class PlacementConstraints:
    """Comprehensive placement constraints"""
    kerf_width: float = 3.0  # mm
    min_edge_distance: float = 10.0  # mm from sheet edges
    max_cutting_length: float = 0.0  # 0 = unlimited
    allow_rotation: bool = True
    rotation_strategy: RotationStrategy = RotationStrategy.OPTIMIZE_ORIENTATION
    material_grouping: MaterialGrouping = MaterialGrouping.STRICT_SEPARATION
    max_sheets_per_material: int = 10
    thermal_cutting_constraints: bool = True
    guillotine_cuts_only: bool = True

class ComplexConstraintHandler:
    """
    Advanced constraint handling for GPU-accelerated bin packing with support for
    complex rotation, material, and cutting constraints.
    """

    def __init__(self, constraints: PlacementConstraints):
        self.constraints = constraints
        self.material_compatibility_matrix = self._initialize_material_compatibility()
        self.violation_history: List[ConstraintViolation] = []

    def _initialize_material_compatibility(self) -> Dict[Tuple[str, str], bool]:
        """Initialize material compatibility matrix"""
        # Define material compatibility rules
        compatible_materials = {
            ("Steel", "Steel"): True,
            ("Aluminum", "Aluminum"): True,
            ("Stainless", "Stainless"): True,
            ("Steel", "Aluminum"): False,  # Different cutting parameters
            ("Steel", "Stainless"): False,  # Different cutting parameters
            ("Aluminum", "Stainless"): False,  # Different cutting parameters
        }

        return compatible_materials

    def validate_placement_constraints(self, panels: List[Panel],
                                     sheet: SteelSheet) -> List[ConstraintViolation]:
        """
        Validate all placement constraints for a given panel set and sheet.

        Args:
            panels: List of panels to validate
            sheet: Target steel sheet

        Returns:
            List of constraint violations found
        """
        violations = []

        # Material constraint validation
        violations.extend(self._validate_material_constraints(panels))

        # Size constraint validation
        violations.extend(self._validate_size_constraints(panels, sheet))

        # Rotation constraint validation
        violations.extend(self._validate_rotation_constraints(panels))

        # Thermal cutting constraint validation
        if self.constraints.thermal_cutting_constraints:
            violations.extend(self._validate_thermal_constraints(panels, sheet))

        self.violation_history.extend(violations)
        return violations

    def _validate_material_constraints(self, panels: List[Panel]) -> List[ConstraintViolation]:
        """Validate material compatibility constraints"""
        violations = []

        if self.constraints.material_grouping == MaterialGrouping.STRICT_SEPARATION:
            # Check for material mixing
            materials = set(panel.material for panel in panels)
            if len(materials) > 1:
                violations.append(ConstraintViolation(
                    constraint_type="material_separation",
                    severity="error",
                    description=f"Multiple materials detected: {materials}. Strict separation required.",
                    affected_panels=[p.id for p in panels],
                    suggested_fix="Group panels by material type into separate sheets"
                ))

        elif self.constraints.material_grouping == MaterialGrouping.COMPATIBLE_MIXING:
            # Check material compatibility
            materials = list(set(panel.material for panel in panels))
            for i, mat1 in enumerate(materials):
                for mat2 in materials[i+1:]:
                    if not self.material_compatibility_matrix.get((mat1, mat2), False):
                        affected_panels = [p.id for p in panels if p.material in [mat1, mat2]]
                        violations.append(ConstraintViolation(
                            constraint_type="material_compatibility",
                            severity="error",
                            description=f"Incompatible materials: {mat1} and {mat2}",
                            affected_panels=affected_panels,
                            suggested_fix="Separate incompatible materials to different sheets"
                        ))

        elif self.constraints.material_grouping == MaterialGrouping.THICKNESS_GROUPING:
            # Check thickness grouping
            thicknesses = set(panel.thickness for panel in panels)
            if len(thicknesses) > 2:  # Allow max 2 thicknesses per sheet
                violations.append(ConstraintViolation(
                    constraint_type="thickness_grouping",
                    severity="warning",
                    description=f"Multiple thicknesses detected: {thicknesses}. Consider grouping by thickness.",
                    affected_panels=[p.id for p in panels],
                    suggested_fix="Group panels with similar thickness for optimal cutting"
                ))

        return violations

    def _validate_size_constraints(self, panels: List[Panel], sheet: SteelSheet) -> List[ConstraintViolation]:
        """Validate panel size constraints"""
        violations = []

        for panel in panels:
            # Check if panel fits in sheet (considering kerf and edge distance)
            effective_sheet_width = sheet.width - 2 * self.constraints.min_edge_distance
            effective_sheet_height = sheet.height - 2 * self.constraints.min_edge_distance

            fits_normal = (panel.width <= effective_sheet_width and
                          panel.height <= effective_sheet_height)
            fits_rotated = (panel.height <= effective_sheet_width and
                           panel.width <= effective_sheet_height)

            if not fits_normal and not fits_rotated:
                violations.append(ConstraintViolation(
                    constraint_type="size_constraint",
                    severity="error",
                    description=f"Panel {panel.id} ({panel.width}x{panel.height}) too large for sheet",
                    affected_panels=[panel.id],
                    suggested_fix="Use larger sheet or split panel"
                ))
            elif not fits_normal and fits_rotated and not panel.allow_rotation:
                violations.append(ConstraintViolation(
                    constraint_type="rotation_required",
                    severity="warning",
                    description=f"Panel {panel.id} requires rotation but rotation disabled",
                    affected_panels=[panel.id],
                    suggested_fix="Enable rotation for this panel"
                ))

        return violations

    def _validate_rotation_constraints(self, panels: List[Panel]) -> List[ConstraintViolation]:
        """Validate rotation constraints"""
        violations = []

        for panel in panels:
            if not panel.allow_rotation and self.constraints.rotation_strategy != RotationStrategy.NO_ROTATION:
                violations.append(ConstraintViolation(
                    constraint_type="rotation_mismatch",
                    severity="info",
                    description=f"Panel {panel.id} rotation disabled but global rotation enabled",
                    affected_panels=[panel.id],
                    suggested_fix="Check rotation settings consistency"
                ))

        return violations

    def _validate_thermal_constraints(self, panels: List[Panel], sheet: SteelSheet) -> List[ConstraintViolation]:
        """Validate thermal cutting constraints"""
        violations = []

        # Check for excessive cutting length that might cause thermal issues
        total_perimeter = sum(2 * (panel.width + panel.height) for panel in panels)

        if self.constraints.max_cutting_length > 0 and total_perimeter > self.constraints.max_cutting_length:
            violations.append(ConstraintViolation(
                constraint_type="thermal_cutting",
                severity="warning",
                description=f"Total cutting length {total_perimeter:.0f}mm exceeds thermal limit {self.constraints.max_cutting_length:.0f}mm",
                affected_panels=[p.id for p in panels],
                suggested_fix="Reduce panel count or use multiple sheets"
            ))

        return violations

    def optimize_panel_rotations(self, panels: List[Panel],
                                sheet: SteelSheet) -> List[Panel]:
        """
        Optimize panel rotations based on the configured rotation strategy.

        Args:
            panels: List of panels to optimize
            sheet: Target steel sheet

        Returns:
            List of panels with optimized rotation settings
        """
        optimized_panels = []

        for panel in panels:
            optimized_panel = self._optimize_individual_rotation(panel, sheet)
            optimized_panels.append(optimized_panel)

        logger.info(f"Rotation optimization completed for {len(panels)} panels")
        return optimized_panels

    def _optimize_individual_rotation(self, panel: Panel, sheet: SteelSheet) -> Panel:
        """Optimize rotation for an individual panel"""
        if not panel.allow_rotation:
            return panel

        original_panel = panel

        if self.constraints.rotation_strategy == RotationStrategy.NO_ROTATION:
            return panel

        elif self.constraints.rotation_strategy == RotationStrategy.ALLOW_90_DEGREES:
            # Simply allow rotation - decision made during placement
            return panel

        elif self.constraints.rotation_strategy == RotationStrategy.OPTIMIZE_ORIENTATION:
            # Choose orientation that better matches sheet aspect ratio
            sheet_aspect = sheet.width / sheet.height
            panel_aspect_normal = panel.width / panel.height
            panel_aspect_rotated = panel.height / panel.width

            # Choose orientation closer to sheet aspect ratio
            normal_diff = abs(panel_aspect_normal - sheet_aspect)
            rotated_diff = abs(panel_aspect_rotated - sheet_aspect)

            if rotated_diff < normal_diff:
                # Create rotated panel
                rotated_panel = Panel(
                    id=panel.id,
                    width=panel.height,  # Swap dimensions
                    height=panel.width,
                    material=panel.material,
                    thickness=panel.thickness,
                    quantity=panel.quantity,
                    allow_rotation=panel.allow_rotation,
                    priority=panel.priority
                )
                return rotated_panel

        elif self.constraints.rotation_strategy == RotationStrategy.MINIMIZE_WASTE:
            # Choose orientation that minimizes potential waste
            # This is a simplified heuristic - actual optimization happens during placement
            if panel.width > panel.height and sheet.width < sheet.height:
                # Tall sheet, wide panel - consider rotation
                if panel.height <= sheet.width and panel.width <= sheet.height:
                    rotated_panel = Panel(
                        id=panel.id,
                        width=panel.height,
                        height=panel.width,
                        material=panel.material,
                        thickness=panel.thickness,
                        quantity=panel.quantity,
                        allow_rotation=panel.allow_rotation,
                        priority=panel.priority
                    )
                    return rotated_panel

        return original_panel

    def group_panels_by_material(self, panels: List[Panel]) -> Dict[str, List[Panel]]:
        """
        Group panels by material according to the configured grouping strategy.

        Args:
            panels: List of panels to group

        Returns:
            Dictionary mapping material types to panel lists
        """
        if self.constraints.material_grouping == MaterialGrouping.STRICT_SEPARATION:
            return self._group_by_strict_material(panels)

        elif self.constraints.material_grouping == MaterialGrouping.COMPATIBLE_MIXING:
            return self._group_by_compatible_materials(panels)

        elif self.constraints.material_grouping == MaterialGrouping.THICKNESS_GROUPING:
            return self._group_by_thickness(panels)

        elif self.constraints.material_grouping == MaterialGrouping.PRIORITY_BASED:
            return self._group_by_priority(panels)

        else:
            # Default: group by material
            return self._group_by_strict_material(panels)

    def _group_by_strict_material(self, panels: List[Panel]) -> Dict[str, List[Panel]]:
        """Group panels by exact material match"""
        groups = {}
        for panel in panels:
            if panel.material not in groups:
                groups[panel.material] = []
            groups[panel.material].append(panel)
        return groups

    def _group_by_compatible_materials(self, panels: List[Panel]) -> Dict[str, List[Panel]]:
        """Group panels by compatible materials"""
        groups = {}
        used_materials = set()

        for panel in panels:
            # Find existing compatible group
            compatible_group = None
            for group_key in groups:
                group_materials = set(p.material for p in groups[group_key])
                if all(self.material_compatibility_matrix.get((panel.material, mat), False)
                      for mat in group_materials):
                    compatible_group = group_key
                    break

            if compatible_group:
                groups[compatible_group].append(panel)
            else:
                # Create new group
                group_key = f"group_{len(groups) + 1}_{panel.material}"
                groups[group_key] = [panel]

        return groups

    def _group_by_thickness(self, panels: List[Panel]) -> Dict[str, List[Panel]]:
        """Group panels by thickness"""
        groups = {}
        for panel in panels:
            thickness_key = f"{panel.material}_{panel.thickness}mm"
            if thickness_key not in groups:
                groups[thickness_key] = []
            groups[thickness_key].append(panel)
        return groups

    def _group_by_priority(self, panels: List[Panel]) -> Dict[str, List[Panel]]:
        """Group panels by priority level"""
        groups = {}
        for panel in panels:
            priority_key = f"{panel.material}_priority_{panel.priority}"
            if priority_key not in groups:
                groups[priority_key] = []
            groups[priority_key].append(panel)
        return groups

    def validate_guillotine_cuts(self, placed_panels: List[PlacedPanel],
                                sheet: SteelSheet) -> List[ConstraintViolation]:
        """
        Validate that all placements allow valid guillotine cuts.

        Args:
            placed_panels: List of placed panels to validate
            sheet: Sheet containing the placements

        Returns:
            List of guillotine constraint violations
        """
        violations = []

        if not self.constraints.guillotine_cuts_only:
            return violations

        # Simplified guillotine validation
        # A proper implementation would use sophisticated algorithms
        for i, panel1 in enumerate(placed_panels):
            for panel2 in placed_panels[i+1:]:
                if self._panels_prevent_guillotine_cut(panel1, panel2, sheet):
                    violations.append(ConstraintViolation(
                        constraint_type="guillotine_constraint",
                        severity="error",
                        description=f"Panels {panel1.panel.id} and {panel2.panel.id} prevent guillotine cuts",
                        affected_panels=[panel1.panel.id, panel2.panel.id],
                        suggested_fix="Reorganize panel placement to allow edge-to-edge cuts"
                    ))

        return violations

    def _panels_prevent_guillotine_cut(self, panel1: PlacedPanel, panel2: PlacedPanel,
                                      sheet: SteelSheet) -> bool:
        """Check if two panels prevent guillotine cuts (simplified implementation)"""
        # This is a simplified check - a full implementation would be more complex

        # Check if panels are in a configuration that prevents straight cuts
        p1_x1, p1_y1 = panel1.x, panel1.y
        p1_x2, p1_y2 = panel1.x + panel1.panel.width, panel1.y + panel1.panel.height

        p2_x1, p2_y1 = panel2.x, panel2.y
        p2_x2, p2_y2 = panel2.x + panel2.panel.width, panel2.y + panel2.panel.height

        # Check if panels create an L-shape or other non-guillotine configuration
        # This is a very simplified heuristic
        overlaps_x = not (p1_x2 <= p2_x1 or p2_x2 <= p1_x1)
        overlaps_y = not (p1_y2 <= p2_y1 or p2_y2 <= p1_y1)

        if overlaps_x and overlaps_y:
            # Panels overlap - definitely not valid
            return True

        # Additional guillotine validation would go here
        return False

    def apply_kerf_adjustments(self, placed_panels: List[PlacedPanel]) -> List[PlacedPanel]:
        """
        Apply kerf width adjustments to placed panels.

        Args:
            placed_panels: List of placed panels

        Returns:
            List of panels with kerf adjustments applied
        """
        if self.constraints.kerf_width <= 0:
            return placed_panels

        adjusted_panels = []

        for panel in placed_panels:
            # Adjust panel dimensions to account for kerf
            adjusted_width = panel.panel.width - self.constraints.kerf_width
            adjusted_height = panel.panel.height - self.constraints.kerf_width

            # Ensure minimum dimensions
            if adjusted_width > 0 and adjusted_height > 0:
                # Create adjusted panel
                adjusted_panel_data = Panel(
                    id=panel.panel.id,
                    width=adjusted_width,
                    height=adjusted_height,
                    material=panel.panel.material,
                    thickness=panel.panel.thickness,
                    quantity=panel.panel.quantity,
                    allow_rotation=panel.panel.allow_rotation,
                    priority=panel.panel.priority
                )

                adjusted_placed_panel = PlacedPanel(
                    panel=adjusted_panel_data,
                    x=panel.x + self.constraints.kerf_width / 2,  # Center the cut
                    y=panel.y + self.constraints.kerf_width / 2,
                    rotated=panel.rotated
                )

                adjusted_panels.append(adjusted_placed_panel)
            else:
                # Panel too small after kerf adjustment
                logger.warning(f"Panel {panel.panel.id} too small after kerf adjustment")
                adjusted_panels.append(panel)  # Keep original

        return adjusted_panels

    def get_constraint_summary(self) -> Dict[str, Any]:
        """Get summary of current constraints and violations"""
        return {
            'constraints': {
                'kerf_width': self.constraints.kerf_width,
                'min_edge_distance': self.constraints.min_edge_distance,
                'rotation_strategy': self.constraints.rotation_strategy.value,
                'material_grouping': self.constraints.material_grouping.value,
                'guillotine_cuts_only': self.constraints.guillotine_cuts_only,
                'thermal_cutting_constraints': self.constraints.thermal_cutting_constraints
            },
            'violation_history': [
                {
                    'type': v.constraint_type,
                    'severity': v.severity,
                    'description': v.description,
                    'affected_count': len(v.affected_panels)
                }
                for v in self.violation_history
            ],
            'total_violations': len(self.violation_history),
            'error_count': len([v for v in self.violation_history if v.severity == "error"]),
            'warning_count': len([v for v in self.violation_history if v.severity == "warning"])
        }


def create_constraint_handler(kerf_width: float = 3.0,
                            allow_rotation: bool = True,
                            material_grouping: str = "strict_separation",
                            **kwargs) -> ComplexConstraintHandler:
    """Factory function to create constraint handler with common settings"""

    rotation_strategy = RotationStrategy.OPTIMIZE_ORIENTATION if allow_rotation else RotationStrategy.NO_ROTATION

    material_grouping_enum = {
        "strict_separation": MaterialGrouping.STRICT_SEPARATION,
        "compatible_mixing": MaterialGrouping.COMPATIBLE_MIXING,
        "thickness_grouping": MaterialGrouping.THICKNESS_GROUPING,
        "priority_based": MaterialGrouping.PRIORITY_BASED
    }.get(material_grouping, MaterialGrouping.STRICT_SEPARATION)

    constraints = PlacementConstraints(
        kerf_width=kerf_width,
        allow_rotation=allow_rotation,
        rotation_strategy=rotation_strategy,
        material_grouping=material_grouping_enum,
        **kwargs
    )

    return ComplexConstraintHandler(constraints)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create test panels
    test_panels = [
        Panel(id="P001", width=200, height=150, material="Steel", thickness=2.0),
        Panel(id="P002", width=300, height=100, material="Steel", thickness=2.0),
        Panel(id="P003", width=150, height=200, material="Aluminum", thickness=3.0),
    ]

    test_sheet = SteelSheet(width=1500.0, height=3100.0)

    # Create constraint handler
    handler = create_constraint_handler(
        kerf_width=3.0,
        allow_rotation=True,
        material_grouping="strict_separation"
    )

    # Validate constraints
    violations = handler.validate_placement_constraints(test_panels, test_sheet)
    print(f"Found {len(violations)} constraint violations")

    for violation in violations:
        print(f"  {violation.severity.upper()}: {violation.description}")

    # Group panels by material
    material_groups = handler.group_panels_by_material(test_panels)
    print(f"Material groups: {list(material_groups.keys())}")

    # Get constraint summary
    summary = handler.get_constraint_summary()
    print(f"Constraint summary: {summary}")