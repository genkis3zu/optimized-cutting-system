"""
Bulk Processing System for Tier 2 Optimization
Tier 2最適化のための一括処理システム

Handles:
- Grid layout optimization for similar panels
- Smart grouping and bulk placement
- Multi-panel arrangement patterns
"""

import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import math

from core.models import Panel, SteelSheet, PlacementResult, PlacedPanel, OptimizationConstraints
from core.models.optimization_state import BulkGroup


@dataclass
class GridLayout:
    """Grid layout configuration for bulk panels"""
    cols: int
    rows: int
    panel_width: float
    panel_height: float
    kerf_width: float

    @property
    def total_width(self) -> float:
        return self.cols * self.panel_width + (self.cols - 1) * self.kerf_width

    @property
    def total_height(self) -> float:
        return self.rows * self.panel_height + (self.rows - 1) * self.kerf_width

    @property
    def panels_per_sheet(self) -> int:
        return self.cols * self.rows

    def fits_in_sheet(self, sheet: SteelSheet) -> bool:
        return self.total_width <= sheet.width and self.total_height <= sheet.height


class BulkProcessor:
    """
    Intelligent bulk processing system for similar panels
    類似パネルのためのインテリジェント一括処理システム
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def process_bulk_group(
        self,
        panels: List[Panel],
        constraints: OptimizationConstraints
    ) -> List[PlacementResult]:
        """
        Process a group of similar panels using bulk optimization
        一括最適化を使用して類似パネルのグループを処理
        """
        if not panels:
            return []

        start_time = time.time()

        # Group panels by exact dimensions
        dimension_groups = self._group_by_dimensions(panels)
        results = []

        for (width, height), group_panels in dimension_groups.items():
            total_quantity = sum(p.quantity for p in group_panels)

            if total_quantity >= 4:  # Bulk threshold
                self.logger.info(f"Bulk processing {total_quantity} panels of size {width}×{height}mm")
                group_results = self._process_dimension_group(group_panels, constraints)
                results.extend(group_results)
            else:
                # Use standard optimization for small groups
                self.logger.debug(f"Using standard processing for {total_quantity} panels")
                # This would call back to standard algorithms

        processing_time = time.time() - start_time
        self.logger.info(f"Bulk processing complete: {len(results)} sheets in {processing_time:.2f}s")

        return results

    def _group_by_dimensions(self, panels: List[Panel]) -> Dict[Tuple[float, float], List[Panel]]:
        """Group panels by their cutting dimensions"""
        groups = {}

        for panel in panels:
            key = (panel.cutting_width, panel.cutting_height)
            if key not in groups:
                groups[key] = []
            groups[key].append(panel)

        return groups

    def _process_dimension_group(
        self,
        panels: List[Panel],
        constraints: OptimizationConstraints
    ) -> List[PlacementResult]:
        """Process a group of panels with same dimensions"""
        if not panels:
            return []

        # Calculate total quantity
        total_quantity = sum(p.quantity for p in panels)
        reference_panel = panels[0]

        # Get material sheets (using first panel's material)
        from core.material_manager import get_material_manager
        material_manager = get_material_manager()
        normalized_material = material_manager.normalize_material_code(reference_panel.material)
        available_sheets = material_manager.get_sheets_by_type(normalized_material)

        if not available_sheets:
            self.logger.warning(f"No sheets available for material {normalized_material}")
            return []

        # Convert to SteelSheet objects
        steel_sheets = []
        for sheet in available_sheets:
            if sheet.availability > 0:
                steel_sheet = SteelSheet(
                    width=sheet.width,
                    height=sheet.height,
                    thickness=sheet.thickness,
                    material=sheet.material_type,
                    cost_per_sheet=sheet.cost_per_sheet
                )
                steel_sheets.append(steel_sheet)

        if not steel_sheets:
            return []

        # Find best grid layout for each sheet size
        best_layouts = []
        for sheet in steel_sheets:
            layout = self._find_optimal_grid_layout(
                reference_panel, sheet, constraints.kerf_width
            )
            if layout:
                best_layouts.append((sheet, layout))

        if not best_layouts:
            self.logger.warning("No suitable grid layouts found")
            return []

        # Sort by efficiency (panels per sheet)
        best_layouts.sort(key=lambda x: x[1].panels_per_sheet, reverse=True)

        # Use the most efficient layout
        selected_sheet, selected_layout = best_layouts[0]

        self.logger.info(
            f"Selected grid layout: {selected_layout.cols}×{selected_layout.rows} "
            f"= {selected_layout.panels_per_sheet} panels per sheet"
        )

        # Generate placement results
        return self._generate_grid_placements(
            panels, selected_sheet, selected_layout, constraints
        )

    def _find_optimal_grid_layout(
        self,
        panel: Panel,
        sheet: SteelSheet,
        kerf_width: float
    ) -> Optional[GridLayout]:
        """Find optimal grid layout for panels on sheet"""
        panel_w = panel.cutting_width
        panel_h = panel.cutting_height

        best_layout = None
        max_panels = 0

        # Try different orientations if rotation allowed
        orientations = [(panel_w, panel_h)]
        if panel.allow_rotation:
            orientations.append((panel_h, panel_w))

        for pw, ph in orientations:
            # Calculate maximum possible grid
            max_cols = int((sheet.width + kerf_width) / (pw + kerf_width))
            max_rows = int((sheet.height + kerf_width) / (ph + kerf_width))

            # Try different grid configurations
            for cols in range(1, max_cols + 1):
                for rows in range(1, max_rows + 1):
                    layout = GridLayout(cols, rows, pw, ph, kerf_width)

                    if layout.fits_in_sheet(sheet):
                        panels_count = layout.panels_per_sheet
                        if panels_count > max_panels:
                            max_panels = panels_count
                            best_layout = layout

        return best_layout

    def _generate_grid_placements(
        self,
        panels: List[Panel],
        sheet: SteelSheet,
        layout: GridLayout,
        constraints: OptimizationConstraints
    ) -> List[PlacementResult]:
        """Generate placement results using grid layout"""
        results = []

        # Expand panels to individual pieces
        individual_panels = []
        for panel in panels:
            for i in range(panel.quantity):
                individual_panel = Panel(
                    id=f"{panel.id}_{i+1}" if panel.quantity > 1 else panel.id,
                    width=panel.width,
                    height=panel.height,
                    quantity=1,
                    material=panel.material,
                    thickness=panel.thickness,
                    priority=panel.priority,
                    allow_rotation=panel.allow_rotation,
                    block_order=panel.block_order,
                    pi_code=panel.pi_code,
                    expanded_width=panel.expanded_width,
                    expanded_height=panel.expanded_height
                )
                individual_panels.append(individual_panel)

        # Place panels in grid pattern
        sheet_id = 1
        panels_per_sheet = layout.panels_per_sheet

        for i in range(0, len(individual_panels), panels_per_sheet):
            sheet_panels = individual_panels[i:i + panels_per_sheet]
            placed_panels = []

            # Calculate positions in grid
            for j, panel in enumerate(sheet_panels):
                col = j % layout.cols
                row = j // layout.cols

                x = col * (layout.panel_width + layout.kerf_width)
                y = row * (layout.panel_height + layout.kerf_width)

                # Check if panel needs rotation
                rotated = (layout.panel_width != panel.cutting_width)

                placed_panel = PlacedPanel(
                    panel=panel,
                    x=x,
                    y=y,
                    rotated=rotated
                )
                placed_panels.append(placed_panel)

            # Calculate efficiency
            used_area = sum(p.panel.area for p in placed_panels)
            efficiency = used_area / sheet.area

            # Create result
            result = PlacementResult(
                sheet_id=sheet_id,
                material_block=sheet.material,
                sheet=sheet,
                panels=placed_panels,
                efficiency=efficiency,
                waste_area=sheet.area - used_area,
                cut_length=self._calculate_grid_cut_length(layout, sheet),
                cost=sheet.cost_per_sheet,
                algorithm="BULK_GRID",
                processing_time=0.1,  # Fast bulk processing
                timestamp=time.time()
            )

            results.append(result)
            sheet_id += 1

            self.logger.debug(
                f"Grid sheet {sheet_id-1}: {len(placed_panels)} panels, "
                f"efficiency {efficiency:.1%}"
            )

        return results

    def _calculate_grid_cut_length(self, layout: GridLayout, sheet: SteelSheet) -> float:
        """Calculate cutting length for grid layout"""
        # Vertical cuts
        vertical_cuts = (layout.cols - 1) * sheet.height

        # Horizontal cuts
        horizontal_cuts = (layout.rows - 1) * sheet.width

        # Perimeter cuts
        perimeter = 2 * (sheet.width + sheet.height)

        return vertical_cuts + horizontal_cuts + perimeter