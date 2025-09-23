"""
Global Optimization Engine for 100% Placement Guarantee
100%配置保証のためのグローバル最適化エンジン

Architecture:
- Tier 1: Enhanced FFD/BFD with bulk awareness
- Tier 2: Intelligent bulk processing with grid layouts
- Tier 3: Guaranteed placement fallback system
"""

import time
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from core.models import Panel, SteelSheet, PlacementResult, OptimizationConstraints
from core.models.optimization_state import (
    GlobalOptimizationState, PanelPlacementState, PlacementStatus,
    BulkGroup, SheetAllocation
)
from core.optimizer import OptimizationEngine, OptimizationAlgorithm


class GlobalOptimizationEngine:
    """
    Global optimization engine coordinating 3-tier algorithm system
    3層アルゴリズムシステムを調整するグローバル最適化エンジン
    """

    def __init__(self, base_engine: OptimizationEngine):
        self.base_engine = base_engine
        self.logger = logging.getLogger(__name__)

        # Tier thresholds and parameters
        self.tier1_size_threshold = 20  # Use enhanced FFD/BFD for ≤20 panels
        self.tier2_bulk_threshold = 4   # Bulk process groups with ≥4 similar panels
        self.tier3_guarantee_active = True  # Always guarantee placement

        # Performance targets per tier
        self.tier_targets = {
            1: {'efficiency': 0.75, 'time_budget': 5.0},
            2: {'efficiency': 0.80, 'time_budget': 15.0},
            3: {'efficiency': 0.60, 'time_budget': 0.0}  # No time limit for guarantee
        }

    def optimize_with_guarantee(
        self,
        panels: List[Panel],
        constraints: Optional[OptimizationConstraints] = None
    ) -> Tuple[List[PlacementResult], GlobalOptimizationState]:
        """
        Main optimization with 100% placement guarantee
        100%配置保証付きメイン最適化
        """
        start_time = time.time()

        if constraints is None:
            constraints = OptimizationConstraints()

        # Initialize global state
        state = GlobalOptimizationState(
            total_panels=sum(p.quantity for p in panels),
            panels_placed=0,
            panels_pending=sum(p.quantity for p in panels),
            current_tier=1
        )

        # Initialize panel states
        for panel in panels:
            for i in range(panel.quantity):
                panel_id = f"{panel.id}_{i+1}" if panel.quantity > 1 else panel.id
                state.panel_states[panel_id] = PanelPlacementState(panel=panel)

        self.logger.info(
            f"Starting global optimization: {len(panels)} panel types, "
            f"{state.total_panels} total panels"
        )

        all_results = []

        # **TIER 1: Enhanced FFD/BFD with Bulk Awareness**
        try:
            tier1_results, state = self._execute_tier1(panels, constraints, state)
            all_results.extend(tier1_results)

            if state.is_complete:
                self.logger.info("✓ Tier 1 achieved 100% placement")
                return all_results, state

        except Exception as e:
            self.logger.error(f"Tier 1 failed: {e}")

        # **TIER 2: Intelligent Bulk Processing**
        if state.panels_pending > 0:
            state.current_tier = 2
            try:
                tier2_results, state = self._execute_tier2(state, constraints)
                all_results.extend(tier2_results)

                if state.is_complete:
                    self.logger.info("✓ Tier 2 achieved 100% placement")
                    return all_results, state

            except Exception as e:
                self.logger.error(f"Tier 2 failed: {e}")

        # **TIER 3: Guaranteed Placement Fallback**
        if state.panels_pending > 0:
            state.current_tier = 3
            state.guarantee_active = True
            state.guarantee_start_time = datetime.now()

            self.logger.warning(
                f"Activating Tier 3 guarantee system for {state.panels_pending} remaining panels"
            )

            try:
                tier3_results, state = self._execute_tier3(state, constraints)
                all_results.extend(tier3_results)

                if not state.is_complete:
                    # This should never happen - log as critical error
                    self.logger.critical(
                        f"GUARANTEE FAILED: {state.panels_pending} panels still unplaced!"
                    )

            except Exception as e:
                self.logger.critical(f"Tier 3 guarantee system failed: {e}")

        # Final validation
        total_time = time.time() - start_time
        final_placement_rate = state.placement_rate

        self.logger.info(
            f"Global optimization complete: {final_placement_rate:.1%} placement rate, "
            f"{state.sheets_used} sheets used, {total_time:.2f}s"
        )

        if final_placement_rate < 1.0:
            self.logger.error(
                f"FAILED TO GUARANTEE 100% PLACEMENT: {final_placement_rate:.1%} achieved"
            )

        return all_results, state

    def _execute_tier1(
        self,
        panels: List[Panel],
        constraints: OptimizationConstraints,
        state: GlobalOptimizationState
    ) -> Tuple[List[PlacementResult], GlobalOptimizationState]:
        """
        Tier 1: Enhanced FFD/BFD with bulk awareness
        Tier 1: バルク認識付き拡張FFD/BFD
        """
        self.logger.info("Executing Tier 1: Enhanced FFD/BFD")

        # Create bulk groups for awareness
        bulk_groups = state.create_bulk_groups(panels)

        # Use existing optimization engine with enhanced constraints
        tier1_constraints = OptimizationConstraints(
            max_sheets=min(50, constraints.max_sheets),  # Limit sheets for tier 1
            kerf_width=constraints.kerf_width,
            min_waste_piece=constraints.min_waste_piece,
            allow_rotation=constraints.allow_rotation,
            material_separation=constraints.material_separation,
            time_budget=self.tier_targets[1]['time_budget'],
            target_efficiency=self.tier_targets[1]['efficiency']
        )

        # Execute optimization
        results = self.base_engine.optimize(panels, tier1_constraints)

        # Update state with placement results
        placed_panels = set()
        for result in results:
            for placed_panel in result.panels:
                placed_panels.add(placed_panel.panel.id)
                state.update_panel_status(
                    placed_panel.panel.id,
                    PlacementStatus.PLACED,
                    assigned_sheet_id=result.sheet_id,
                    placement_tier=1
                )

        # Track unplaced panels
        state.unplaced_panels = []
        for panel in panels:
            remaining_quantity = panel.quantity
            for i in range(panel.quantity):
                panel_id = f"{panel.id}_{i+1}" if panel.quantity > 1 else panel.id
                if panel_id in placed_panels:
                    remaining_quantity -= 1

            if remaining_quantity > 0:
                unplaced_panel = Panel(
                    id=panel.id,
                    width=panel.width,
                    height=panel.height,
                    quantity=remaining_quantity,
                    material=panel.material,
                    thickness=panel.thickness,
                    priority=panel.priority,
                    allow_rotation=panel.allow_rotation,
                    block_order=panel.block_order,
                    pi_code=panel.pi_code,
                    expanded_width=panel.expanded_width,
                    expanded_height=panel.expanded_height
                )
                state.unplaced_panels.append(unplaced_panel)

        # Update sheet allocations
        for result in results:
            state.add_sheet_allocation(result.sheet, "tier1_primary")

        self.logger.info(
            f"Tier 1 complete: {len(results)} sheets, "
            f"{state.placement_rate:.1%} placement rate"
        )

        return results, state

    def _execute_tier2(
        self,
        state: GlobalOptimizationState,
        constraints: OptimizationConstraints
    ) -> Tuple[List[PlacementResult], GlobalOptimizationState]:
        """
        Tier 2: Intelligent bulk processing with grid layouts
        Tier 2: グリッドレイアウト付きインテリジェント一括処理
        """
        self.logger.info(f"Executing Tier 2: Bulk processing for {len(state.unplaced_panels)} remaining panels")

        results = []

        # Import bulk processor
        try:
            from core.algorithms.bulk_processor import BulkProcessor
            bulk_processor = BulkProcessor()
        except ImportError:
            self.logger.warning("Bulk processor not available, using enhanced FFD fallback")
            return self._tier2_fallback(state, constraints)

        # Process bulk groups
        processed_panels = set()

        for bulk_group in state.bulk_groups:
            # Find unplaced panels for this group
            group_unplaced = [
                panel for panel in state.unplaced_panels
                if (panel.cutting_width, panel.cutting_height) == bulk_group.pattern
                and panel.id not in processed_panels
            ]

            if not group_unplaced:
                continue

            self.logger.info(f"Processing bulk group {bulk_group.group_id}: {bulk_group.pattern}")

            # Use bulk processor for this group
            tier2_constraints = OptimizationConstraints(
                max_sheets=10,  # Allow multiple sheets for bulk
                kerf_width=constraints.kerf_width,
                min_waste_piece=constraints.min_waste_piece,
                allow_rotation=constraints.allow_rotation,
                material_separation=False,  # Bulk process same material
                time_budget=self.tier_targets[2]['time_budget'],
                target_efficiency=self.tier_targets[2]['efficiency']
            )

            bulk_results = bulk_processor.process_bulk_group(
                group_unplaced, tier2_constraints
            )

            results.extend(bulk_results)

            # Update state
            for result in bulk_results:
                for placed_panel in result.panels:
                    processed_panels.add(placed_panel.panel.id)
                    state.update_panel_status(
                        placed_panel.panel.id,
                        PlacementStatus.PLACED,
                        assigned_sheet_id=result.sheet_id,
                        placement_tier=2
                    )

                state.add_sheet_allocation(result.sheet, "tier2_bulk")

        # Process remaining non-bulk panels with enhanced FFD
        remaining_panels = [
            panel for panel in state.unplaced_panels
            if panel.id not in processed_panels
        ]

        if remaining_panels:
            self.logger.info(f"Processing {len(remaining_panels)} non-bulk panels with enhanced FFD")

            # Use best available algorithm for remaining panels
            remaining_results = self.base_engine.optimize(
                remaining_panels,
                OptimizationConstraints(
                    max_sheets=20,
                    time_budget=5.0,
                    target_efficiency=0.70
                )
            )

            results.extend(remaining_results)

            # Update state for remaining panels
            for result in remaining_results:
                for placed_panel in result.panels:
                    state.update_panel_status(
                        placed_panel.panel.id,
                        PlacementStatus.PLACED,
                        assigned_sheet_id=result.sheet_id,
                        placement_tier=2
                    )

                state.add_sheet_allocation(result.sheet, "tier2_remaining")

        self.logger.info(
            f"Tier 2 complete: {len(results)} additional sheets, "
            f"{state.placement_rate:.1%} total placement rate"
        )

        return results, state

    def _execute_tier3(
        self,
        state: GlobalOptimizationState,
        constraints: OptimizationConstraints
    ) -> Tuple[List[PlacementResult], GlobalOptimizationState]:
        """
        Tier 3: Guaranteed placement fallback system
        Tier 3: 保証配置フォールバックシステム
        """
        self.logger.info(f"Executing Tier 3: Guarantee system for {state.panels_pending} panels")

        results = []

        # Import guaranteed placement system
        try:
            from core.algorithms.guaranteed_placement import GuaranteedPlacementSystem
            guarantee_system = GuaranteedPlacementSystem()
        except ImportError:
            self.logger.warning("Guaranteed placement system not available, using simple fallback")
            return self._tier3_simple_fallback(state, constraints)

        # Get all remaining unplaced panels
        unplaced_panels = []
        for panel_id, panel_state in state.panel_states.items():
            if panel_state.status != PlacementStatus.PLACED:
                unplaced_panels.append(panel_state.panel)

        if not unplaced_panels:
            return results, state

        # Execute guaranteed placement
        guarantee_constraints = OptimizationConstraints(
            max_sheets=1000,  # No limit on sheets for guarantee
            kerf_width=constraints.kerf_width,
            min_waste_piece=25.0,  # Smaller minimum for guarantee
            allow_rotation=True,   # Force rotation allowed
            material_separation=False,
            time_budget=0.0,  # No time limit
            target_efficiency=0.50  # Lower efficiency acceptable for guarantee
        )

        guarantee_results = guarantee_system.guarantee_placement(
            unplaced_panels, guarantee_constraints
        )

        results.extend(guarantee_results)

        # Update state - mark all as placed
        for result in guarantee_results:
            for placed_panel in result.panels:
                state.update_panel_status(
                    placed_panel.panel.id,
                    PlacementStatus.GUARANTEED,
                    assigned_sheet_id=result.sheet_id,
                    placement_tier=3
                )

            state.add_sheet_allocation(result.sheet, "tier3_guarantee")

        self.logger.info(
            f"Tier 3 complete: {len(guarantee_results)} guarantee sheets, "
            f"{state.placement_rate:.1%} final placement rate"
        )

        return results, state

    def _tier2_fallback(
        self,
        state: GlobalOptimizationState,
        constraints: OptimizationConstraints
    ) -> Tuple[List[PlacementResult], GlobalOptimizationState]:
        """Fallback for Tier 2 when bulk processor is not available"""
        self.logger.info("Using Tier 2 fallback: Enhanced FFD for remaining panels")

        # Use enhanced FFD with relaxed constraints
        fallback_results = self.base_engine.optimize(
            state.unplaced_panels,
            OptimizationConstraints(
                max_sheets=30,
                time_budget=10.0,
                target_efficiency=0.65
            )
        )

        # Update state
        for result in fallback_results:
            for placed_panel in result.panels:
                state.update_panel_status(
                    placed_panel.panel.id,
                    PlacementStatus.PLACED,
                    assigned_sheet_id=result.sheet_id,
                    placement_tier=2
                )

            state.add_sheet_allocation(result.sheet, "tier2_fallback")

        return fallback_results, state

    def _tier3_simple_fallback(
        self,
        state: GlobalOptimizationState,
        constraints: OptimizationConstraints
    ) -> Tuple[List[PlacementResult], GlobalOptimizationState]:
        """Simple fallback for Tier 3 guarantee system"""
        self.logger.warning("Using simple Tier 3 fallback: One panel per sheet")

        results = []

        # Place each remaining panel on its own sheet if necessary
        for panel_id, panel_state in state.panel_states.items():
            if panel_state.status != PlacementStatus.PLACED:
                panel = panel_state.panel

                # Create individual optimization
                individual_result = self.base_engine.optimize(
                    [Panel(
                        id=panel.id,
                        width=panel.width,
                        height=panel.height,
                        quantity=1,
                        material=panel.material,
                        thickness=panel.thickness,
                        priority=panel.priority,
                        allow_rotation=True,  # Force allow rotation
                        block_order=panel.block_order
                    )],
                    OptimizationConstraints(
                        max_sheets=1,
                        time_budget=0.0,
                        target_efficiency=0.1  # Any efficiency acceptable
                    )
                )

                if individual_result:
                    results.extend(individual_result)

                    for result in individual_result:
                        state.update_panel_status(
                            panel_id,
                            PlacementStatus.GUARANTEED,
                            assigned_sheet_id=result.sheet_id,
                            placement_tier=3
                        )

                        state.add_sheet_allocation(result.sheet, "tier3_individual")

        return results, state


# Factory function
def create_global_optimization_engine(base_engine: OptimizationEngine) -> GlobalOptimizationEngine:
    """Create global optimization engine with base engine"""
    return GlobalOptimizationEngine(base_engine)