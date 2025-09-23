"""
Enhanced optimization state models for 100% placement guarantee
100%配置保証のための拡張最適化状態モデル
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime
from enum import Enum

from core.models import Panel, SteelSheet, PlacementResult


class PlacementStatus(Enum):
    """Panel placement status tracking"""
    PENDING = "pending"
    PLACED = "placed"
    PROCESSING = "processing"
    FAILED = "failed"
    GUARANTEED = "guaranteed"  # Will be placed in fallback


@dataclass
class PanelPlacementState:
    """Individual panel placement state tracking"""
    panel: Panel
    status: PlacementStatus = PlacementStatus.PENDING
    attempts: int = 0
    last_attempt_time: Optional[datetime] = None
    assigned_sheet_id: Optional[int] = None
    placement_tier: Optional[int] = None  # 1=FFD/BFD, 2=Bulk, 3=Guarantee
    failure_reasons: List[str] = field(default_factory=list)


@dataclass
class BulkGroup:
    """Bulk processing group for similar panels"""
    group_id: str
    panels: List[Panel]
    pattern: Tuple[float, float]  # (width, height) pattern
    total_quantity: int
    grid_layout: Optional[Tuple[int, int]] = None  # (cols, rows)
    efficiency_estimate: float = 0.0


@dataclass
class SheetAllocation:
    """Multi-sheet allocation planning"""
    sheet_id: int
    sheet: SteelSheet
    allocated_panels: List[Panel]
    capacity_used: float  # 0.0 to 1.0
    efficiency_estimate: float
    allocation_strategy: str  # "primary", "bulk", "fallback"
    constraints_met: bool = True


@dataclass
class GlobalOptimizationState:
    """Global optimization state for 100% placement guarantee"""
    total_panels: int
    panels_placed: int
    panels_pending: int
    current_tier: int  # 1, 2, or 3

    # Panel tracking
    panel_states: Dict[str, PanelPlacementState] = field(default_factory=dict)
    unplaced_panels: List[Panel] = field(default_factory=list)

    # Bulk processing
    bulk_groups: List[BulkGroup] = field(default_factory=list)

    # Multi-sheet planning
    sheet_allocations: List[SheetAllocation] = field(default_factory=list)
    sheets_used: int = 0
    max_sheets_allowed: int = 1000

    # Guarantee tracking
    guarantee_active: bool = False
    guarantee_start_time: Optional[datetime] = None

    def __post_init__(self):
        """Initialize tracking state"""
        self.placement_rate = 0.0
        self.efficiency_estimate = 0.0
        self.start_time = datetime.now()

    @property
    def placement_rate(self) -> float:
        """Calculate current placement rate"""
        if self.total_panels == 0:
            return 0.0
        return self.panels_placed / self.total_panels

    @property
    def is_complete(self) -> bool:
        """Check if all panels are placed"""
        return self.panels_placed == self.total_panels

    @property
    def needs_guarantee(self) -> bool:
        """Check if guarantee system is needed"""
        return self.panels_pending > 0 and self.current_tier >= 3

    def update_panel_status(self, panel_id: str, status: PlacementStatus, **kwargs):
        """Update panel placement status"""
        if panel_id in self.panel_states:
            state = self.panel_states[panel_id]
            state.status = status
            state.last_attempt_time = datetime.now()

            # Update additional fields
            for key, value in kwargs.items():
                if hasattr(state, key):
                    setattr(state, key, value)

            # Update global counters
            self._update_counters()

    def _update_counters(self):
        """Update global placement counters"""
        self.panels_placed = sum(
            1 for state in self.panel_states.values()
            if state.status == PlacementStatus.PLACED
        )
        self.panels_pending = sum(
            1 for state in self.panel_states.values()
            if state.status in [PlacementStatus.PENDING, PlacementStatus.PROCESSING]
        )

    def add_sheet_allocation(self, sheet: SteelSheet, strategy: str = "primary") -> int:
        """Add new sheet allocation"""
        sheet_id = len(self.sheet_allocations) + 1
        allocation = SheetAllocation(
            sheet_id=sheet_id,
            sheet=sheet,
            allocated_panels=[],
            capacity_used=0.0,
            efficiency_estimate=0.0,
            allocation_strategy=strategy
        )
        self.sheet_allocations.append(allocation)
        self.sheets_used += 1
        return sheet_id

    def get_available_capacity(self) -> float:
        """Calculate total available capacity across all sheets"""
        total_capacity = 0.0
        for allocation in self.sheet_allocations:
            remaining_capacity = 1.0 - allocation.capacity_used
            total_capacity += remaining_capacity
        return total_capacity

    def estimate_additional_sheets_needed(self) -> int:
        """Estimate how many additional sheets are needed"""
        if self.panels_pending == 0:
            return 0

        # Calculate remaining panel area
        remaining_area = sum(
            state.panel.area for state in self.panel_states.values()
            if state.status == PlacementStatus.PENDING
        )

        # Estimate with conservative efficiency (60% for guarantee system)
        if self.sheet_allocations:
            sheet_area = self.sheet_allocations[0].sheet.area
            sheets_needed = remaining_area / (sheet_area * 0.6)
            return max(1, int(sheets_needed) + 1)

        return 1

    def create_bulk_groups(self, panels: List[Panel]) -> List[BulkGroup]:
        """Create bulk processing groups for similar panels"""
        # Group panels by dimensions
        dimension_groups: Dict[Tuple[float, float], List[Panel]] = {}

        for panel in panels:
            # Use cutting dimensions for grouping
            key = (panel.cutting_width, panel.cutting_height)
            if key not in dimension_groups:
                dimension_groups[key] = []
            dimension_groups[key].append(panel)

        # Create bulk groups for panels with quantity >= 4
        bulk_groups = []
        for pattern, group_panels in dimension_groups.items():
            total_quantity = sum(p.quantity for p in group_panels)

            if total_quantity >= 4:  # Minimum bulk threshold
                group_id = f"bulk_{len(bulk_groups)+1}"
                bulk_group = BulkGroup(
                    group_id=group_id,
                    panels=group_panels,
                    pattern=pattern,
                    total_quantity=total_quantity
                )
                bulk_groups.append(bulk_group)

        self.bulk_groups = bulk_groups
        return bulk_groups

    def get_summary(self) -> Dict[str, any]:
        """Get optimization state summary"""
        return {
            'total_panels': self.total_panels,
            'panels_placed': self.panels_placed,
            'panels_pending': self.panels_pending,
            'placement_rate': f"{self.placement_rate:.1%}",
            'sheets_used': self.sheets_used,
            'current_tier': self.current_tier,
            'bulk_groups': len(self.bulk_groups),
            'guarantee_active': self.guarantee_active,
            'elapsed_time': (datetime.now() - self.start_time).total_seconds()
        }