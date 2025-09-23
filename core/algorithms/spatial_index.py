"""
Spatial Index System for Efficient Panel Placement

Implements R-tree-like spatial indexing for O(log n) overlap detection
and efficient position searching in 2D bin packing.
"""

import math
from typing import List, Tuple, Optional, Set, Dict
from dataclasses import dataclass, field
from collections import defaultdict

from core.models import PlacedPanel


@dataclass
class Rectangle:
    """Represents a rectangle in 2D space"""
    x: float
    y: float
    width: float
    height: float

    @property
    def x2(self) -> float:
        """Right edge coordinate"""
        return self.x + self.width

    @property
    def y2(self) -> float:
        """Top edge coordinate"""
        return self.y + self.height

    @property
    def area(self) -> float:
        """Rectangle area"""
        return self.width * self.height

    def intersects(self, other: 'Rectangle', tolerance: float = 0.0) -> bool:
        """
        Check if this rectangle intersects with another.

        Args:
            other: Another rectangle
            tolerance: Overlap tolerance (negative for gap requirement)

        Returns:
            True if rectangles intersect
        """
        return not (
            self.x2 + tolerance <= other.x or
            other.x2 + tolerance <= self.x or
            self.y2 + tolerance <= other.y or
            other.y2 + tolerance <= self.y
        )

    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside rectangle"""
        return self.x <= x <= self.x2 and self.y <= y <= self.y2

    def contains_rectangle(self, other: 'Rectangle') -> bool:
        """Check if this rectangle completely contains another"""
        return (
            self.x <= other.x and
            self.y <= other.y and
            self.x2 >= other.x2 and
            self.y2 >= other.y2
        )


@dataclass
class GridCell:
    """Grid cell for spatial partitioning"""
    x: int
    y: int
    rectangles: Set[int] = field(default_factory=set)  # IDs of rectangles in this cell


class SpatialIndex:
    """
    Spatial index for efficient rectangle overlap detection.

    Uses grid-based partitioning for fast spatial queries.
    Provides O(log n) average case overlap detection.
    """

    def __init__(self, width: float, height: float, cell_size: Optional[float] = None):
        """
        Initialize spatial index.

        Args:
            width: Total space width
            height: Total space height
            cell_size: Grid cell size (auto-calculated if None)
        """
        self.width = width
        self.height = height

        # Auto-calculate optimal cell size based on space dimensions
        if cell_size is None:
            # Use approximately 10x10 grid for reasonable performance
            self.cell_size = max(width, height) / 10
        else:
            self.cell_size = cell_size

        # Calculate grid dimensions
        self.grid_cols = max(1, int(math.ceil(width / self.cell_size)))
        self.grid_rows = max(1, int(math.ceil(height / self.cell_size)))

        # Initialize grid
        self.grid: Dict[Tuple[int, int], GridCell] = {}
        for i in range(self.grid_cols):
            for j in range(self.grid_rows):
                self.grid[(i, j)] = GridCell(i, j)

        # Store rectangles
        self.rectangles: Dict[int, Rectangle] = {}
        self.next_id = 0

        # Performance metrics
        self.queries = 0
        self.comparisons = 0

    def add_rectangle(self, x: float, y: float, width: float, height: float) -> int:
        """
        Add a rectangle to the spatial index.

        Args:
            x, y: Bottom-left corner coordinates
            width, height: Rectangle dimensions

        Returns:
            Unique ID for the added rectangle
        """
        rect = Rectangle(x, y, width, height)
        rect_id = self.next_id
        self.next_id += 1

        self.rectangles[rect_id] = rect

        # Add to relevant grid cells
        cells = self._get_cells_for_rectangle(rect)
        for cell_coords in cells:
            if cell_coords in self.grid:
                self.grid[cell_coords].rectangles.add(rect_id)

        return rect_id

    def remove_rectangle(self, rect_id: int):
        """
        Remove a rectangle from the spatial index.

        Args:
            rect_id: ID of rectangle to remove
        """
        if rect_id not in self.rectangles:
            return

        rect = self.rectangles[rect_id]

        # Remove from grid cells
        cells = self._get_cells_for_rectangle(rect)
        for cell_coords in cells:
            if cell_coords in self.grid:
                self.grid[cell_coords].rectangles.discard(rect_id)

        # Remove from rectangles dict
        del self.rectangles[rect_id]

    def check_collision(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        tolerance: float = 0.0
    ) -> bool:
        """
        Check if a rectangle would collide with existing rectangles.

        Args:
            x, y: Bottom-left corner coordinates
            width, height: Rectangle dimensions
            tolerance: Overlap tolerance

        Returns:
            True if collision would occur
        """
        self.queries += 1
        test_rect = Rectangle(x, y, width, height)

        # Get potentially colliding rectangles from grid
        cells = self._get_cells_for_rectangle(test_rect)
        checked_ids = set()

        for cell_coords in cells:
            if cell_coords not in self.grid:
                continue

            cell = self.grid[cell_coords]
            for rect_id in cell.rectangles:
                if rect_id in checked_ids:
                    continue

                checked_ids.add(rect_id)
                self.comparisons += 1

                if self.rectangles[rect_id].intersects(test_rect, tolerance):
                    return True

        return False

    def find_collisions(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        tolerance: float = 0.0
    ) -> List[int]:
        """
        Find all rectangles that would collide with a test rectangle.

        Args:
            x, y: Bottom-left corner coordinates
            width, height: Rectangle dimensions
            tolerance: Overlap tolerance

        Returns:
            List of rectangle IDs that would collide
        """
        self.queries += 1
        test_rect = Rectangle(x, y, width, height)
        collisions = []

        # Get potentially colliding rectangles from grid
        cells = self._get_cells_for_rectangle(test_rect)
        checked_ids = set()

        for cell_coords in cells:
            if cell_coords not in self.grid:
                continue

            cell = self.grid[cell_coords]
            for rect_id in cell.rectangles:
                if rect_id in checked_ids:
                    continue

                checked_ids.add(rect_id)
                self.comparisons += 1

                if self.rectangles[rect_id].intersects(test_rect, tolerance):
                    collisions.append(rect_id)

        return collisions

    def find_free_positions(
        self,
        panel_width: float,
        panel_height: float,
        step_size: Optional[float] = None
    ) -> List[Tuple[float, float]]:
        """
        Find potential free positions for a panel.

        Args:
            panel_width: Panel width
            panel_height: Panel height
            step_size: Search step size (auto if None)

        Returns:
            List of (x, y) positions where panel could fit
        """
        if step_size is None:
            step_size = min(panel_width, panel_height) / 4

        positions = []

        # Start with corners and edges of existing rectangles
        candidate_positions = [(0, 0)]  # Origin

        for rect in self.rectangles.values():
            # Right edge
            candidate_positions.append((rect.x2, rect.y))
            # Top edge
            candidate_positions.append((rect.x, rect.y2))
            # Top-right corner
            candidate_positions.append((rect.x2, rect.y2))

        # Test each candidate position
        for x, y in candidate_positions:
            # Check bounds
            if x + panel_width > self.width or y + panel_height > self.height:
                continue

            # Check collisions
            if not self.check_collision(x, y, panel_width, panel_height):
                positions.append((x, y))

        # Sort by bottom-left position preference
        positions.sort(key=lambda p: (p[1], p[0]))

        return positions

    def get_bounding_box(self) -> Optional[Rectangle]:
        """
        Get the bounding box of all rectangles.

        Returns:
            Rectangle representing the bounding box, or None if empty
        """
        if not self.rectangles:
            return None

        min_x = min(r.x for r in self.rectangles.values())
        min_y = min(r.y for r in self.rectangles.values())
        max_x = max(r.x2 for r in self.rectangles.values())
        max_y = max(r.y2 for r in self.rectangles.values())

        return Rectangle(min_x, min_y, max_x - min_x, max_y - min_y)

    def get_fill_rate(self) -> float:
        """
        Calculate the fill rate of the space.

        Returns:
            Percentage of space filled (0-100)
        """
        if not self.rectangles:
            return 0.0

        total_area = sum(r.area for r in self.rectangles.values())
        space_area = self.width * self.height

        if space_area == 0:
            return 0.0

        return (total_area / space_area) * 100

    def clear(self):
        """Clear all rectangles from the index"""
        for cell in self.grid.values():
            cell.rectangles.clear()
        self.rectangles.clear()
        self.next_id = 0
        self.queries = 0
        self.comparisons = 0

    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        avg_comparisons = self.comparisons / self.queries if self.queries > 0 else 0

        return {
            'total_queries': self.queries,
            'total_comparisons': self.comparisons,
            'avg_comparisons_per_query': avg_comparisons,
            'rectangles_count': len(self.rectangles),
            'fill_rate': self.get_fill_rate(),
            'grid_cells': self.grid_cols * self.grid_rows
        }

    def _get_cells_for_rectangle(self, rect: Rectangle) -> List[Tuple[int, int]]:
        """
        Get grid cells that a rectangle overlaps.

        Args:
            rect: Rectangle to check

        Returns:
            List of (col, row) grid cell coordinates
        """
        # Calculate cell range
        min_col = max(0, int(rect.x / self.cell_size))
        max_col = min(self.grid_cols - 1, int(rect.x2 / self.cell_size))
        min_row = max(0, int(rect.y / self.cell_size))
        max_row = min(self.grid_rows - 1, int(rect.y2 / self.cell_size))

        cells = []
        for col in range(min_col, max_col + 1):
            for row in range(min_row, max_row + 1):
                cells.append((col, row))

        return cells


class CompactSpatialIndex:
    """
    Memory-efficient spatial index using compressed data structures.

    Optimized for memory usage while maintaining O(log n) performance.
    """

    def __init__(self, width: float, height: float):
        """
        Initialize compact spatial index.

        Args:
            width: Total space width
            height: Total space height
        """
        self.width = width
        self.height = height

        # Use single list for rectangles (more memory efficient)
        self.rectangles: List[Tuple[float, float, float, float]] = []

        # Simplified grid using bitset-like structure
        grid_resolution = 32
        self.grid_cols = min(grid_resolution, int(width / 50))
        self.grid_rows = min(grid_resolution, int(height / 50))
        self.cell_width = width / self.grid_cols
        self.cell_height = height / self.grid_rows

        # Grid stores indices instead of sets (more compact)
        self.grid: List[List[int]] = [[] for _ in range(self.grid_cols * self.grid_rows)]

    def add_rectangle(self, x: float, y: float, width: float, height: float) -> int:
        """Add rectangle with minimal memory overhead"""
        rect_id = len(self.rectangles)
        self.rectangles.append((x, y, width, height))

        # Add to grid cells
        min_col = int(x / self.cell_width)
        max_col = min(self.grid_cols - 1, int((x + width) / self.cell_width))
        min_row = int(y / self.cell_height)
        max_row = min(self.grid_rows - 1, int((y + height) / self.cell_height))

        for col in range(min_col, max_col + 1):
            for row in range(min_row, max_row + 1):
                cell_idx = row * self.grid_cols + col
                if cell_idx < len(self.grid):
                    self.grid[cell_idx].append(rect_id)

        return rect_id

    def check_collision(self, x: float, y: float, width: float, height: float) -> bool:
        """Memory-efficient collision detection"""
        # Get grid cells
        min_col = max(0, int(x / self.cell_width))
        max_col = min(self.grid_cols - 1, int((x + width) / self.cell_width))
        min_row = max(0, int(y / self.cell_height))
        max_row = min(self.grid_rows - 1, int((y + height) / self.cell_height))

        checked = set()

        for col in range(min_col, max_col + 1):
            for row in range(min_row, max_row + 1):
                cell_idx = row * self.grid_cols + col
                if cell_idx >= len(self.grid):
                    continue

                for rect_id in self.grid[cell_idx]:
                    if rect_id in checked:
                        continue
                    checked.add(rect_id)

                    rx, ry, rw, rh = self.rectangles[rect_id]
                    # Check overlap
                    if not (x + width <= rx or rx + rw <= x or
                            y + height <= ry or ry + rh <= y):
                        return True

        return False

    def clear(self):
        """Clear with minimal memory operations"""
        self.rectangles.clear()
        for cell in self.grid:
            cell.clear()


def create_spatial_index_from_placements(
    placements: List[PlacedPanel],
    sheet_width: float,
    sheet_height: float,
    compact: bool = False
) -> SpatialIndex:
    """
    Create a spatial index from existing panel placements.

    Args:
        placements: List of placed panels
        sheet_width: Sheet width
        sheet_height: Sheet height
        compact: Use compact index for memory efficiency

    Returns:
        Populated spatial index
    """
    if compact:
        index = CompactSpatialIndex(sheet_width, sheet_height)
    else:
        index = SpatialIndex(sheet_width, sheet_height)

    for placed in placements:
        if placed.rotated:
            width = placed.panel.cutting_height
            height = placed.panel.cutting_width
        else:
            width = placed.panel.cutting_width
            height = placed.panel.cutting_height

        index.add_rectangle(placed.x, placed.y, width, height)

    return index