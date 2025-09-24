"""
First Fit Decreasing (FFD) Algorithm with Guillotine Constraints
ギロチン制約付きFirst Fit Decreasing (FFD) アルゴリズム

Target: 70-75% efficiency in <1 second for ≤10 panels
目標: 10パネル以下で1秒未満、効率70-75%
"""

import time
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

from core.models import (
    Panel, SteelSheet, PlacementResult, PlacedPanel, 
    OptimizationConstraints
)
from core.optimizer import OptimizationAlgorithm


@dataclass
class Rectangle:
    """Rectangle representation for guillotine algorithm"""
    x: float
    y: float
    width: float
    height: float
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def right(self) -> float:
        return self.x + self.width
    
    @property
    def top(self) -> float:
        return self.y + self.height
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside rectangle"""
        return self.x <= x < self.right and self.y <= y < self.top
    
    def can_fit(self, width: float, height: float) -> bool:
        """Check if given dimensions can fit in this rectangle"""
        return width <= self.width and height <= self.height


class GuillotineBinPacker:
    """
    Guillotine Binary Tree Bin Packing Algorithm
    ギロチン二分木ビンパッキングアルゴリズム
    """
    
    def __init__(self, sheet_width: float, sheet_height: float, kerf_width: float = 0.0):
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.kerf_width = kerf_width
        
        # Available rectangles for placement
        self.free_rectangles: List[Rectangle] = [
            Rectangle(0, 0, sheet_width, sheet_height)
        ]
        
        # Placed panels
        self.placed_panels: List[PlacedPanel] = []
        
        self.logger = logging.getLogger(__name__)
    
    def find_best_position(self, panel: Panel) -> Optional[Tuple[Rectangle, bool, float, float]]:
        """
        Find best position for panel using Bottom-Left-Fill strategy
        Bottom-Left-Fill戦略を使用してパネルの最適位置を検索

        Returns: (rectangle, rotated, x, y) or None
        """
        best_rect = None
        best_rotated = False
        best_x = float('inf')
        best_y = float('inf')
        best_area_fit = float('inf')

        # Try both orientations if rotation is allowed, using cutting dimensions
        orientations = [(panel.cutting_width, panel.cutting_height, False)]
        if panel.allow_rotation:
            orientations.append((panel.cutting_height, panel.cutting_width, True))

        for width, height, rotated in orientations:
            for rect in self.free_rectangles:
                if rect.can_fit(width, height):
                    # Check for overlap with existing placed panels
                    if self._would_overlap(rect.x, rect.y, width, height):
                        continue  # Skip this position if it would overlap

                    # Bottom-left scoring: prioritize bottom, then left
                    y_score = rect.y
                    x_score = rect.x
                    area_fit = rect.area - (width * height)

                    # Better position criteria:
                    # 1. Lower Y coordinate (bottom preference)
                    # 2. Lower X coordinate (left preference)
                    # 3. Better area fit (less waste)
                    is_better = (
                        y_score < best_y or
                        (y_score == best_y and x_score < best_x) or
                        (y_score == best_y and x_score == best_x and area_fit < best_area_fit)
                    )

                    if is_better:
                        best_rect = rect
                        best_rotated = rotated
                        best_x = rect.x
                        best_y = rect.y
                        best_area_fit = area_fit

        if best_rect:
            width = panel.height if best_rotated else panel.width
            height = panel.width if best_rotated else panel.height
            return best_rect, best_rotated, best_x, best_y

        return None

    def _would_overlap(self, x: float, y: float, width: float, height: float) -> bool:
        """
        Check if placing a panel at given position would overlap with existing panels
        指定位置にパネルを配置した場合の既存パネルとの重複チェック
        """
        new_x1, new_y1 = x, y
        new_x2, new_y2 = x + width, y + height

        for placed_panel in self.placed_panels:
            # Get bounds of existing panel
            existing_x1 = placed_panel.x
            existing_y1 = placed_panel.y
            existing_x2 = placed_panel.x + placed_panel.actual_width
            existing_y2 = placed_panel.y + placed_panel.actual_height

            # Check for overlap (not just touching - actual overlap)
            if not (new_x2 <= existing_x1 or new_x1 >= existing_x2 or
                    new_y2 <= existing_y1 or new_y1 >= existing_y2):
                return True

        return False
    
    def place_panel(self, panel: Panel) -> bool:
        """
        Place panel in best available position
        利用可能な最適位置にパネルを配置
        """
        position = self.find_best_position(panel)
        if not position:
            return False
        
        rect, rotated, x, y = position
        
        # Calculate actual dimensions considering rotation (using cutting dimensions)
        actual_width = panel.cutting_height if rotated else panel.cutting_width
        actual_height = panel.cutting_width if rotated else panel.cutting_height
        
        # Create placed panel
        placed_panel = PlacedPanel(
            panel=panel,
            x=x,
            y=y,
            rotated=rotated
        )
        
        self.placed_panels.append(placed_panel)
        
        # Split the used rectangle with guillotine cuts
        self._split_rectangle_guillotine(rect, x, y, actual_width, actual_height)
        
        self.logger.debug(
            f"Placed panel {panel.id} at ({x:.1f}, {y:.1f}) "
            f"size {actual_width:.1f}×{actual_height:.1f} "
            f"{'rotated' if rotated else 'normal'}"
        )
        
        return True
    
    def _split_rectangle_guillotine(
        self, 
        rect: Rectangle, 
        used_x: float, 
        used_y: float, 
        used_width: float, 
        used_height: float
    ):
        """
        Split rectangle using guillotine cuts
        ギロチンカットを使用して矩形を分割
        """
        # Remove the used rectangle
        self.free_rectangles.remove(rect)
        
        # Add kerf (cutting allowance) to dimensions
        kerf = self.kerf_width
        
        # Create new rectangles from guillotine cuts
        new_rectangles = []
        
        # Right remainder (vertical cut)
        if used_x + used_width + kerf < rect.right:
            new_rectangles.append(Rectangle(
                x=used_x + used_width + kerf,
                y=rect.y,
                width=rect.right - (used_x + used_width + kerf),
                height=rect.height
            ))
        
        # Top remainder (horizontal cut)
        if used_y + used_height + kerf < rect.top:
            new_rectangles.append(Rectangle(
                x=rect.x,
                y=used_y + used_height + kerf,
                width=rect.width,
                height=rect.top - (used_y + used_height + kerf)
            ))
        
        # Add valid rectangles (remove too small pieces)
        min_size = 50.0  # Minimum usable size
        for new_rect in new_rectangles:
            if new_rect.width >= min_size and new_rect.height >= min_size:
                self.free_rectangles.append(new_rect)
        
        # Remove overlapping rectangles and merge adjacent ones
        self._cleanup_rectangles()
    
    def _cleanup_rectangles(self):
        """Remove overlapping rectangles and merge adjacent ones"""
        # Remove rectangles that are completely inside others
        rectangles_to_remove = []
        
        for i, rect1 in enumerate(self.free_rectangles):
            for j, rect2 in enumerate(self.free_rectangles):
                if i != j and self._is_inside(rect1, rect2):
                    rectangles_to_remove.append(rect1)
                    break
        
        for rect in rectangles_to_remove:
            if rect in self.free_rectangles:
                self.free_rectangles.remove(rect)
    
    def _is_inside(self, rect1: Rectangle, rect2: Rectangle) -> bool:
        """Check if rect1 is completely inside rect2"""
        return (
            rect1.x >= rect2.x and
            rect1.y >= rect2.y and
            rect1.right <= rect2.right and
            rect1.top <= rect2.top
        )
    
    def get_efficiency(self) -> float:
        """Calculate packing efficiency"""
        if not self.placed_panels:
            return 0.0
        
        used_area = sum(panel.panel.area for panel in self.placed_panels)
        total_area = self.sheet_width * self.sheet_height
        
        return used_area / total_area if total_area > 0 else 0.0
    
    def get_waste_area(self) -> float:
        """Calculate waste area"""
        used_area = sum(panel.panel.area for panel in self.placed_panels)
        total_area = self.sheet_width * self.sheet_height
        return total_area - used_area


class FirstFitDecreasing(OptimizationAlgorithm):
    """
    First Fit Decreasing algorithm implementation
    First Fit Decreasing アルゴリズム実装
    """
    
    def __init__(self):
        super().__init__("FFD")
    
    def estimate_time(self, panel_count: int, complexity: float) -> float:
        """
        Estimate processing time for FFD
        FFDの処理時間見積もり
        """
        # FFD is O(n log n) for sorting + O(n²) for placement
        base_time = 0.01  # Base processing time
        return base_time * panel_count * (1 + complexity)
    
    def optimize(
        self,
        panels: List[Panel],
        sheet: SteelSheet,
        constraints: OptimizationConstraints
    ) -> PlacementResult:
        """
        Execute First Fit Decreasing optimization
        First Fit Decreasing最適化の実行
        """
        start_time = time.time()
        
        self.logger.info(f"Starting FFD optimization for {len(panels)} panels")
        
        # Expand panels based on quantity and sort by area (decreasing)
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
                    block_order=panel.block_order
                )
                individual_panels.append(individual_panel)
        
        # Sort by area (decreasing) - core of FFD algorithm
        individual_panels.sort(key=lambda p: p.area, reverse=True)
        
        self.logger.debug(f"Sorted {len(individual_panels)} individual panels by area")
        
        # Initialize bin packer
        packer = GuillotineBinPacker(
            sheet.width, 
            sheet.height, 
            constraints.kerf_width
        )
        
        # Place panels using First Fit strategy
        placed_count = 0
        for panel in individual_panels:
            if packer.place_panel(panel):
                placed_count += 1
            else:
                self.logger.debug(f"Could not place panel {panel.id}")
        
        # Calculate results
        efficiency = packer.get_efficiency()
        waste_area = packer.get_waste_area()
        processing_time = time.time() - start_time
        
        # Create result
        result = PlacementResult(
            sheet_id=1,
            material_block=sheet.material,
            sheet=sheet,
            panels=packer.placed_panels,
            efficiency=efficiency,
            waste_area=waste_area,
            cut_length=self._calculate_cut_length(packer.placed_panels, sheet),
            cost=sheet.cost_per_sheet,
            algorithm="FFD",
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
        self.logger.info(
            f"FFD completed: {placed_count}/{len(individual_panels)} panels placed, "
            f"efficiency: {efficiency:.1%}, time: {processing_time:.3f}s"
        )
        
        return result
    
    def _calculate_cut_length(self, placed_panels: List[PlacedPanel], sheet: SteelSheet) -> float:
        """
        Calculate total cutting length for guillotine cuts
        ギロチンカットの総切断長を計算
        """
        if not placed_panels:
            return 0.0
        
        # For simplicity, estimate cutting length based on panel perimeters
        # In practice, this would require detailed cut sequence analysis
        total_length = 0.0
        
        for placed_panel in placed_panels:
            # Approximate: 2 cuts per panel (simplified)
            panel_cuts = (
                placed_panel.actual_width + 
                placed_panel.actual_height
            ) * 1.5  # Factor for guillotine constraint overhead
            
            total_length += panel_cuts
        
        return total_length


# Factory function
def create_ffd_algorithm() -> FirstFitDecreasing:
    """Create FFD algorithm instance"""
    return FirstFitDecreasing()