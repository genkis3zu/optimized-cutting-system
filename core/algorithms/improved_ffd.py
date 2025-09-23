"""
Improved First Fit Decreasing Algorithm for 100% Placement
100%配置を目指す改善されたFirst Fit Decreasingアルゴリズム
"""

import time
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from datetime import datetime

from core.models import (
    Panel, SteelSheet, PlacementResult, PlacedPanel,
    OptimizationConstraints
)
from core.optimizer import OptimizationAlgorithm


@dataclass
class Rectangle:
    """Improved Rectangle with better space management"""
    x: float
    y: float
    width: float
    height: float
    used: bool = False

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y + self.height

    def can_fit(self, width: float, height: float) -> bool:
        """Check if given dimensions can fit"""
        return not self.used and width <= self.width and height <= self.height

    def fits_better(self, other: 'Rectangle', width: float, height: float) -> bool:
        """Check if this rectangle fits better than other"""
        if not self.can_fit(width, height):
            return False
        if not other.can_fit(width, height):
            return True

        # Prefer smaller leftover area
        self_leftover = self.area - (width * height)
        other_leftover = other.area - (width * height)

        return self_leftover < other_leftover


class ImprovedGuillotineBinPacker:
    """Improved Guillotine Bin Packer with better space utilization"""

    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
        self.free_rectangles: List[Rectangle] = [Rectangle(0, 0, width, height)]
        self.placed_panels: List[PlacedPanel] = []
        self.used_area = 0.0

    def reset(self):
        """Reset the packer for new sheet"""
        self.free_rectangles = [Rectangle(0, 0, self.width, self.height)]
        self.placed_panels = []
        self.used_area = 0.0

    def place_panel(self, panel: Panel) -> bool:
        """Place panel using improved best-fit algorithm"""
        # Try both orientations
        orientations = [
            (panel.cutting_width, panel.cutting_height, False),
        ]

        if panel.allow_rotation:
            orientations.append((panel.cutting_height, panel.cutting_width, True))

        best_rect = None
        best_orientation = None

        # Find best fitting rectangle for any orientation
        for width, height, is_rotated in orientations:
            for rect in self.free_rectangles:
                if rect.can_fit(width, height):
                    if best_rect is None or rect.fits_better(best_rect, width, height):
                        best_rect = rect
                        best_orientation = (width, height, is_rotated)

        if best_rect is None:
            return False

        # Place the panel
        width, height, is_rotated = best_orientation

        placed_panel = PlacedPanel(
            panel=panel,
            x=best_rect.x,
            y=best_rect.y,
            rotated=is_rotated
        )

        self.placed_panels.append(placed_panel)
        self.used_area += width * height

        # Split the rectangle
        self._split_rectangle(best_rect, width, height)

        return True

    def _split_rectangle(self, rect: Rectangle, used_width: float, used_height: float):
        """Split rectangle after placing a panel"""
        # Remove the used rectangle
        self.free_rectangles.remove(rect)

        # Create new rectangles from remaining space
        remaining_width = rect.width - used_width
        remaining_height = rect.height - used_height

        # Right rectangle (horizontal split)
        if remaining_width > 0:
            right_rect = Rectangle(
                x=rect.x + used_width,
                y=rect.y,
                width=remaining_width,
                height=rect.height
            )
            self.free_rectangles.append(right_rect)

        # Top rectangle (vertical split)
        if remaining_height > 0:
            top_rect = Rectangle(
                x=rect.x,
                y=rect.y + used_height,
                width=used_width,
                height=remaining_height
            )
            self.free_rectangles.append(top_rect)

        # Clean up overlapping rectangles
        self._remove_overlapping_rectangles()

    def _remove_overlapping_rectangles(self):
        """Remove rectangles that are completely contained within others"""
        i = 0
        while i < len(self.free_rectangles):
            j = i + 1
            while j < len(self.free_rectangles):
                rect1 = self.free_rectangles[i]
                rect2 = self.free_rectangles[j]

                # Check if rect1 contains rect2
                if (rect1.x <= rect2.x and rect1.y <= rect2.y and
                    rect1.right >= rect2.right and rect1.top >= rect2.top):
                    self.free_rectangles.pop(j)
                    continue

                # Check if rect2 contains rect1
                if (rect2.x <= rect1.x and rect2.y <= rect1.y and
                    rect2.right >= rect1.right and rect2.top >= rect1.top):
                    self.free_rectangles.pop(i)
                    i -= 1
                    break

                j += 1
            i += 1

    @property
    def efficiency(self) -> float:
        """Calculate current efficiency"""
        total_area = self.width * self.height
        return self.used_area / total_area if total_area > 0 else 0.0


class ImprovedFirstFitDecreasing(OptimizationAlgorithm):
    """Improved FFD with better space utilization for 100% placement"""

    def __init__(self):
        super().__init__("Improved_FFD")

    def estimate_time(self, panel_count: int, complexity: float) -> float:
        """Estimate processing time"""
        return min(panel_count * 0.01, 10.0)

    def optimize(self, panels: List[Panel], sheet: SteelSheet, constraints: OptimizationConstraints) -> PlacementResult:
        """Optimize panel placement with improved algorithm"""
        if not panels:
            return PlacementResult(
                sheet_id=1,
                material_block=sheet.material,
                sheet=sheet,
                panels=[],
                efficiency=0.0,
                waste_area=sheet.area,
                cut_length=0.0,
                cost=sheet.cost_per_sheet,
                algorithm="Improved_FFD",
                processing_time=0.0
            )

        start_time = time.time()

        # Expand panels by quantity
        expanded_panels = []
        for panel in panels:
            for i in range(panel.quantity):
                # Create panel copy with cutting dimensions
                panel_copy = Panel(
                    id=f"{panel.id}_{i+1}",
                    width=panel.width,
                    height=panel.height,
                    quantity=1,
                    material=panel.material,
                    thickness=panel.thickness,
                    priority=panel.priority,
                    allow_rotation=panel.allow_rotation
                )
                # Apply existing cutting dimensions if available
                if hasattr(panel, '_cutting_width'):
                    panel_copy._cutting_width = panel._cutting_width
                    panel_copy._cutting_height = panel._cutting_height
                else:
                    # Fall back to original dimensions
                    panel_copy._cutting_width = panel.width
                    panel_copy._cutting_height = panel.height
                expanded_panels.append(panel_copy)

        # Sort by area (largest first) with tie-breaking
        expanded_panels.sort(key=lambda p: (
            -p.cutting_area,  # Largest area first
            -max(p.cutting_width, p.cutting_height),  # Longest dimension
            -min(p.cutting_width, p.cutting_height)   # Shorter dimension
        ))

        # Use improved packer
        packer = ImprovedGuillotineBinPacker(sheet.width, sheet.height)
        placed_panels = []

        # First pass: try to place all panels
        for panel in expanded_panels:
            if packer.place_panel(panel):
                placed_panels.extend(packer.placed_panels[-1:])  # Add last placed panel

        # If not all panels placed, optimize placement order
        if len(placed_panels) < len(expanded_panels):
            self.logger.info(f"First pass: {len(placed_panels)}/{len(expanded_panels)} panels placed. Optimizing...")
            placed_panels = self._optimize_placement(expanded_panels, sheet)

        processing_time = time.time() - start_time

        # Calculate metrics
        used_area = sum(p.actual_width * p.actual_height for p in placed_panels)
        efficiency = used_area / sheet.area if sheet.area > 0 else 0.0

        self.logger.info(f"Improved FFD: {len(placed_panels)}/{len(expanded_panels)} panels placed ({efficiency:.1%})")

        return PlacementResult(
            sheet_id=1,
            material_block=sheet.material,
            sheet=sheet,
            panels=placed_panels,
            efficiency=efficiency,
            waste_area=sheet.area - used_area,
            cut_length=self._calculate_cut_length(placed_panels),
            cost=sheet.cost_per_sheet,
            algorithm="Improved_FFD",
            processing_time=processing_time
        )

    def _optimize_placement(self, panels: List[Panel], sheet: SteelSheet) -> List[PlacedPanel]:
        """Optimize placement order to achieve higher efficiency"""
        best_placement = []
        best_count = 0

        # Try different sorting strategies
        strategies = [
            # Strategy 1: Area descending
            lambda p: (-p.cutting_area, -max(p.cutting_width, p.cutting_height)),
            # Strategy 2: Longest side first
            lambda p: (-max(p.cutting_width, p.cutting_height), -p.cutting_area),
            # Strategy 3: Aspect ratio consideration
            lambda p: (-p.cutting_area, -abs(p.cutting_width - p.cutting_height)),
            # Strategy 4: Width priority
            lambda p: (-p.cutting_width, -p.cutting_height),
            # Strategy 5: Height priority
            lambda p: (-p.cutting_height, -p.cutting_width),
        ]

        for i, strategy in enumerate(strategies):
            sorted_panels = sorted(panels, key=strategy)

            packer = ImprovedGuillotineBinPacker(sheet.width, sheet.height)
            current_placement = []

            for panel in sorted_panels:
                if packer.place_panel(panel):
                    current_placement.extend(packer.placed_panels[-1:])

            if len(current_placement) > best_count:
                best_count = len(current_placement)
                best_placement = current_placement
                self.logger.info(f"Strategy {i+1}: {best_count}/{len(panels)} panels placed")

                # If we achieved 100% placement, use this result
                if best_count == len(panels):
                    self.logger.info(f"🎉 100% placement achieved with strategy {i+1}!")
                    break

        return best_placement

    def _calculate_cut_length(self, placed_panels: List[PlacedPanel]) -> float:
        """Calculate total cut length"""
        if not placed_panels:
            return 0.0

        # Simplified cut length calculation
        total_length = 0.0
        for panel in placed_panels:
            perimeter = 2 * (panel.actual_width + panel.actual_height)
            total_length += perimeter

        return total_length