"""
Best Fit Decreasing (BFD) Algorithm with Advanced Guillotine Constraints
ギロチン制約付きBest Fit Decreasing (BFD) アルゴリズム

Target: 80-85% efficiency in <5 seconds for ≤30 panels
目標: 30パネル以下で5秒未満、効率80-85%
"""

import time
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from datetime import datetime
import math

from core.models import (
    Panel, SteelSheet, PlacementResult, PlacedPanel,
    OptimizationConstraints
)
from core.optimizer import OptimizationAlgorithm


@dataclass
class Rectangle:
    """Enhanced rectangle representation for BFD algorithm"""
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

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio (width/height)"""
        return self.width / self.height if self.height > 0 else float('inf')

    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside rectangle"""
        return self.x <= x < self.right and self.y <= y < self.top

    def can_fit(self, width: float, height: float) -> bool:
        """Check if given dimensions can fit in this rectangle"""
        return width <= self.width and height <= self.height

    def distance_to_origin(self) -> float:
        """Calculate distance from origin (for bottom-left preference)"""
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def waste_after_placement(self, panel_width: float, panel_height: float) -> float:
        """Calculate waste area after placing panel"""
        if not self.can_fit(panel_width, panel_height):
            return float('inf')
        return self.area - (panel_width * panel_height)


@dataclass
class PlacementScore:
    """Advanced scoring system for placement quality"""
    area_fit: float  # Lower is better - waste area
    bottom_left_score: float  # Lower is better - prefer bottom-left
    edge_contact: float  # Higher is better - contact with existing panels
    fragmentation: float  # Lower is better - avoid creating small pieces
    aspect_ratio_match: float  # Lower is better - match aspect ratios

    @property
    def total_score(self) -> float:
        """Calculate weighted total score (lower is better)"""
        return (
            self.area_fit * 0.35 +
            self.bottom_left_score * 0.25 +
            -self.edge_contact * 0.15 +  # Negative because higher contact is better
            self.fragmentation * 0.15 +
            self.aspect_ratio_match * 0.10
        )


class AdvancedGuillotineBinPacker:
    """
    Advanced Guillotine Binary Tree Bin Packing Algorithm for BFD
    BFD用の高度なギロチン二分木ビンパッキングアルゴリズム
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

        # Performance tracking
        self.placement_attempts = 0
        self.scoring_calculations = 0

        self.logger = logging.getLogger(__name__)

    def find_best_position(self, panel: Panel) -> Optional[Tuple[Rectangle, bool, float, float, PlacementScore]]:
        """
        Find best position for panel using advanced Best Fit scoring
        高度なBest Fitスコアリングを使用してパネルの最適位置を検索

        Returns: (rectangle, rotated, x, y, score) or None
        """
        best_rect = None
        best_rotated = False
        best_x = float('inf')
        best_y = float('inf')
        best_score = None

        # Try both orientations if rotation is allowed, using cutting dimensions
        orientations = [(panel.cutting_width, panel.cutting_height, False)]
        if panel.allow_rotation:
            orientations.append((panel.cutting_height, panel.cutting_width, True))

        for width, height, rotated in orientations:
            for rect in self.free_rectangles:
                if rect.can_fit(width, height):
                    self.placement_attempts += 1

                    # Calculate comprehensive placement score
                    score = self._calculate_placement_score(rect, width, height, panel)
                    self.scoring_calculations += 1

                    # Best position criteria based on total score
                    if best_score is None or score.total_score < best_score.total_score:
                        best_rect = rect
                        best_rotated = rotated
                        best_x = rect.x
                        best_y = rect.y
                        best_score = score

        if best_rect and best_score:
            self.logger.debug(
                f"Best position for {panel.id}: score={best_score.total_score:.3f}, "
                f"area_fit={best_score.area_fit:.1f}, "
                f"bottom_left={best_score.bottom_left_score:.3f}"
            )
            return best_rect, best_rotated, best_x, best_y, best_score

        return None

    def _calculate_placement_score(
        self,
        rect: Rectangle,
        panel_width: float,
        panel_height: float,
        panel: Panel
    ) -> PlacementScore:
        """
        Calculate comprehensive placement score for Best Fit optimization
        Best Fit最適化のための包括的配置スコアを計算
        """
        # 1. Area fit score (waste area)
        waste_area = rect.waste_after_placement(panel_width, panel_height)
        area_fit = waste_area / rect.area if rect.area > 0 else float('inf')

        # 2. Bottom-left preference score
        bottom_left_score = rect.distance_to_origin() / (self.sheet_width + self.sheet_height)

        # 3. Edge contact score (prefer positions that touch existing panels)
        edge_contact = self._calculate_edge_contact(rect.x, rect.y, panel_width, panel_height)

        # 4. Fragmentation score (avoid creating unusable small pieces)
        fragmentation = self._calculate_fragmentation_penalty(rect, panel_width, panel_height)

        # 5. Aspect ratio matching score
        panel_aspect = panel_width / panel_height
        rect_aspect = rect.aspect_ratio
        aspect_ratio_match = abs(panel_aspect - rect_aspect) / max(panel_aspect, rect_aspect)

        return PlacementScore(
            area_fit=area_fit,
            bottom_left_score=bottom_left_score,
            edge_contact=edge_contact,
            fragmentation=fragmentation,
            aspect_ratio_match=aspect_ratio_match
        )

    def _calculate_edge_contact(
        self,
        x: float,
        y: float,
        width: float,
        height: float
    ) -> float:
        """
        Calculate edge contact score (higher is better)
        エッジ接触スコアを計算（高い方が良い）
        """
        contact_score = 0.0

        # Check contact with sheet edges
        if x == 0:  # Left edge
            contact_score += height
        if y == 0:  # Bottom edge
            contact_score += width
        if x + width == self.sheet_width:  # Right edge
            contact_score += height
        if y + height == self.sheet_height:  # Top edge
            contact_score += width

        # Check contact with placed panels
        for placed_panel in self.placed_panels:
            px1, py1, px2, py2 = placed_panel.bounds

            # Check if panels share an edge
            if self._panels_share_edge(x, y, x + width, y + height, px1, py1, px2, py2):
                # Add contact length to score
                overlap_length = self._calculate_edge_overlap(
                    x, y, x + width, y + height, px1, py1, px2, py2
                )
                contact_score += overlap_length

        # Normalize by panel perimeter
        panel_perimeter = 2 * (width + height)
        return contact_score / panel_perimeter if panel_perimeter > 0 else 0

    def _panels_share_edge(
        self,
        x1: float, y1: float, x2: float, y2: float,
        px1: float, py1: float, px2: float, py2: float
    ) -> bool:
        """Check if two rectangles share an edge"""
        # Vertical edge sharing
        if (x1 == px2 or x2 == px1) and not (y2 <= py1 or y1 >= py2):
            return True
        # Horizontal edge sharing
        if (y1 == py2 or y2 == py1) and not (x2 <= px1 or x1 >= px2):
            return True
        return False

    def _calculate_edge_overlap(
        self,
        x1: float, y1: float, x2: float, y2: float,
        px1: float, py1: float, px2: float, py2: float
    ) -> float:
        """Calculate length of shared edge between two rectangles"""
        overlap = 0.0

        # Vertical edge overlap
        if x1 == px2 or x2 == px1:
            overlap = max(0, min(y2, py2) - max(y1, py1))
        # Horizontal edge overlap
        elif y1 == py2 or y2 == py1:
            overlap = max(0, min(x2, px2) - max(x1, px1))

        return overlap

    def _calculate_fragmentation_penalty(
        self,
        rect: Rectangle,
        panel_width: float,
        panel_height: float
    ) -> float:
        """
        Calculate penalty for creating small, unusable fragments
        使用不可能な小片を作るペナルティを計算
        """
        if not rect.can_fit(panel_width, panel_height):
            return float('inf')

        # Calculate resulting fragments after placement
        right_width = rect.width - panel_width
        top_height = rect.height - panel_height

        penalty = 0.0
        min_usable_size = 50.0  # Minimum usable dimension

        # Penalty for creating unusably narrow right fragment
        if 0 < right_width < min_usable_size:
            penalty += (min_usable_size - right_width) * rect.height

        # Penalty for creating unusably short top fragment
        if 0 < top_height < min_usable_size:
            penalty += (min_usable_size - top_height) * rect.width

        # Normalize by sheet area
        return penalty / (self.sheet_width * self.sheet_height)

    def place_panel(self, panel: Panel) -> bool:
        """
        Place panel in best available position using BFD strategy
        BFD戦略を使用して利用可能な最適位置にパネルを配置
        """
        position = self.find_best_position(panel)
        if not position:
            return False

        rect, rotated, x, y, score = position

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
            f"{'rotated' if rotated else 'normal'} "
            f"score={score.total_score:.3f}"
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
        Split rectangle using optimized guillotine cuts for BFD
        BFD用の最適化されたギロチンカットを使用して矩形を分割
        """
        # Remove the used rectangle
        self.free_rectangles.remove(rect)

        # Add kerf (cutting allowance) to dimensions
        kerf = self.kerf_width

        # Create new rectangles from guillotine cuts
        new_rectangles = []

        # Guillotine strategy: Create two non-overlapping rectangles
        # Choice 1: Vertical cut first (split vertically, then horizontally)

        # Right rectangle (full height)
        if used_x + used_width + kerf < rect.right:
            right_rect = Rectangle(
                x=used_x + used_width + kerf,
                y=rect.y,
                width=rect.right - (used_x + used_width + kerf),
                height=rect.height
            )
            new_rectangles.append(right_rect)

        # Top rectangle (only over the used area)
        if used_y + used_height + kerf < rect.top:
            top_rect = Rectangle(
                x=rect.x,
                y=used_y + used_height + kerf,
                width=used_x + used_width,  # Only up to the used width
                height=rect.top - (used_y + used_height + kerf)
            )
            new_rectangles.append(top_rect)

        # Filter rectangles by minimum size requirements
        min_size = 50.0
        valid_rectangles = [
            rect for rect in new_rectangles
            if rect.width >= min_size and rect.height >= min_size
        ]

        # Add valid rectangles to free list
        self.free_rectangles.extend(valid_rectangles)

        # Optimize rectangle list (remove overlaps and merge)
        self._optimize_rectangles()

    def _optimize_rectangles(self):
        """
        Optimize rectangle list by removing overlaps and merging adjacent rectangles
        重複を除去し隣接矩形をマージして矩形リストを最適化
        """
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

        # Attempt to merge adjacent rectangles (basic implementation)
        self._merge_adjacent_rectangles()

    def _is_inside(self, rect1: Rectangle, rect2: Rectangle) -> bool:
        """Check if rect1 is completely inside rect2"""
        return (
            rect1.x >= rect2.x and
            rect1.y >= rect2.y and
            rect1.right <= rect2.right and
            rect1.top <= rect2.top
        )

    def _merge_adjacent_rectangles(self):
        """
        Merge adjacent rectangles to reduce fragmentation
        隣接矩形をマージして断片化を削減
        """
        merged = True
        while merged:
            merged = False
            for i, rect1 in enumerate(self.free_rectangles):
                for j, rect2 in enumerate(self.free_rectangles[i+1:], i+1):
                    if self._can_merge_rectangles(rect1, rect2):
                        # Merge rectangles
                        merged_rect = self._merge_two_rectangles(rect1, rect2)
                        if merged_rect:
                            self.free_rectangles.remove(rect1)
                            self.free_rectangles.remove(rect2)
                            self.free_rectangles.append(merged_rect)
                            merged = True
                            break
                if merged:
                    break

    def _can_merge_rectangles(self, rect1: Rectangle, rect2: Rectangle) -> bool:
        """Check if two rectangles can be merged"""
        # Horizontal merge (same height, adjacent horizontally)
        if (rect1.height == rect2.height and rect1.y == rect2.y and
            (rect1.right == rect2.x or rect2.right == rect1.x)):
            return True

        # Vertical merge (same width, adjacent vertically)
        if (rect1.width == rect2.width and rect1.x == rect2.x and
            (rect1.top == rect2.y or rect2.top == rect1.y)):
            return True

        return False

    def _merge_two_rectangles(self, rect1: Rectangle, rect2: Rectangle) -> Optional[Rectangle]:
        """Merge two adjacent rectangles"""
        # Horizontal merge
        if (rect1.height == rect2.height and rect1.y == rect2.y):
            min_x = min(rect1.x, rect2.x)
            total_width = rect1.width + rect2.width
            return Rectangle(min_x, rect1.y, total_width, rect1.height)

        # Vertical merge
        if (rect1.width == rect2.width and rect1.x == rect2.x):
            min_y = min(rect1.y, rect2.y)
            total_height = rect1.height + rect2.height
            return Rectangle(rect1.x, min_y, rect1.width, total_height)

        return None

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

    def get_performance_metrics(self) -> Dict[str, int]:
        """Get performance metrics for algorithm tuning"""
        return {
            'placement_attempts': self.placement_attempts,
            'scoring_calculations': self.scoring_calculations,
            'free_rectangles': len(self.free_rectangles),
            'placed_panels': len(self.placed_panels)
        }


class BestFitDecreasing(OptimizationAlgorithm):
    """
    Best Fit Decreasing algorithm implementation with advanced scoring
    高度なスコアリング機能付きBest Fit Decreasingアルゴリズム実装
    """

    def __init__(self):
        super().__init__("BFD")

    def estimate_time(self, panel_count: int, complexity: float) -> float:
        """
        Estimate processing time for BFD
        BFDの処理時間見積もり
        """
        # BFD is O(n log n) for sorting + O(n² * k) for placement where k is scoring complexity
        base_time = 0.02  # Higher base time than FFD due to advanced scoring
        scoring_factor = 1.5  # Additional time for comprehensive scoring

        return base_time * panel_count * (1 + complexity) * scoring_factor

    def optimize(
        self,
        panels: List[Panel],
        sheet: SteelSheet,
        constraints: OptimizationConstraints
    ) -> PlacementResult:
        """
        Execute Best Fit Decreasing optimization with advanced features
        高度な機能付きBest Fit Decreasing最適化の実行
        """
        start_time = time.time()

        self.logger.info(f"Starting BFD optimization for {len(panels)} panels")

        # Expand panels based on quantity and create smart sorting
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

        # Advanced sorting for BFD: Multi-criteria optimization
        individual_panels.sort(key=self._get_sorting_key, reverse=True)

        self.logger.debug(f"Sorted {len(individual_panels)} individual panels with advanced criteria")

        # Initialize advanced bin packer
        packer = AdvancedGuillotineBinPacker(
            sheet.width,
            sheet.height,
            constraints.kerf_width
        )

        # Place panels using Best Fit strategy with progress tracking
        placed_count = 0
        skipped_panels = []

        for i, panel in enumerate(individual_panels):
            if packer.place_panel(panel):
                placed_count += 1

                # Log progress for large batches
                if len(individual_panels) > 20 and (i + 1) % 10 == 0:
                    interim_efficiency = packer.get_efficiency()
                    self.logger.debug(
                        f"Progress: {i + 1}/{len(individual_panels)} panels, "
                        f"placed: {placed_count}, efficiency: {interim_efficiency:.1%}"
                    )
            else:
                skipped_panels.append(panel)
                self.logger.debug(f"Could not place panel {panel.id}")

        # Log performance metrics
        metrics = packer.get_performance_metrics()
        self.logger.debug(f"Performance metrics: {metrics}")

        # Calculate final results
        efficiency = packer.get_efficiency()
        waste_area = packer.get_waste_area()
        processing_time = time.time() - start_time

        # Enhanced cut length calculation
        cut_length = self._calculate_optimized_cut_length(packer.placed_panels, sheet)

        # Create comprehensive result
        result = PlacementResult(
            sheet_id=1,
            material_block=sheet.material,
            sheet=sheet,
            panels=packer.placed_panels,
            efficiency=efficiency,
            waste_area=waste_area,
            cut_length=cut_length,
            cost=sheet.cost_per_sheet,
            algorithm="BFD",
            processing_time=processing_time,
            timestamp=datetime.now()
        )

        self.logger.info(
            f"BFD completed: {placed_count}/{len(individual_panels)} panels placed, "
            f"efficiency: {efficiency:.1%}, time: {processing_time:.3f}s, "
            f"skipped: {len(skipped_panels)}"
        )

        # Performance target validation
        target_efficiency = constraints.target_efficiency
        if efficiency >= target_efficiency:
            self.logger.info(f"✓ Target efficiency {target_efficiency:.1%} achieved: {efficiency:.1%}")
        else:
            self.logger.warning(f"✗ Target efficiency {target_efficiency:.1%} not met: {efficiency:.1%}")

        if processing_time <= constraints.time_budget:
            self.logger.info(f"✓ Time budget {constraints.time_budget:.1f}s met: {processing_time:.3f}s")
        else:
            self.logger.warning(f"✗ Time budget {constraints.time_budget:.1f}s exceeded: {processing_time:.3f}s")

        return result

    def _get_sorting_key(self, panel: Panel) -> Tuple[float, float, float, int]:
        """
        Advanced sorting key for Best Fit Decreasing
        Best Fit Decreasing用の高度なソートキー
        """
        # Primary: Area (larger first)
        area = panel.area

        # Secondary: Aspect ratio preference (closer to square is better for packing)
        aspect_ratio = max(panel.width, panel.height) / min(panel.width, panel.height)
        aspect_penalty = aspect_ratio  # Lower penalty is better

        # Tertiary: Perimeter (larger perimeter panels first, they're harder to place)
        perimeter = 2 * (panel.width + panel.height)

        # Quaternary: Priority (higher priority first)
        priority = panel.priority

        return (area, -aspect_penalty, perimeter, priority)

    def _calculate_optimized_cut_length(self, placed_panels: List[PlacedPanel], sheet: SteelSheet) -> float:
        """
        Calculate optimized cutting length for guillotine cuts
        ギロチンカットの最適化された切断長を計算
        """
        if not placed_panels:
            return 0.0

        # Enhanced cutting length calculation considering guillotine constraints
        total_length = 0.0

        # Group panels by cutting lines
        vertical_cuts = set()
        horizontal_cuts = set()

        for placed_panel in placed_panels:
            x1, y1, x2, y2 = placed_panel.bounds

            # Add cutting lines (with kerf consideration)
            vertical_cuts.add(x1)
            vertical_cuts.add(x2)
            horizontal_cuts.add(y1)
            horizontal_cuts.add(y2)

        # Calculate total cutting length
        # Vertical cuts span the full height
        for x in vertical_cuts:
            if 0 < x < sheet.width:  # Exclude sheet edges
                total_length += sheet.height

        # Horizontal cuts span the full width
        for y in horizontal_cuts:
            if 0 < y < sheet.height:  # Exclude sheet edges
                total_length += sheet.width

        # Add efficiency factor for guillotine constraint overhead
        guillotine_factor = 1.2  # 20% overhead for guillotine cutting

        return total_length * guillotine_factor


# Factory function
def create_bfd_algorithm() -> BestFitDecreasing:
    """Create BFD algorithm instance"""
    return BestFitDecreasing()