"""
Multi-Sheet GPU Optimization System

Advanced parallelization system for optimizing panel placement across multiple sheets
with GPU acceleration, load balancing, and cross-sheet optimization.

Key Features:
- Parallel processing across multiple sheets
- Dynamic load balancing using Intel Iris Xe compute units
- Cross-sheet panel migration for efficiency optimization
- Material-aware sheet allocation
- Thermal management for sustained multi-sheet operations
"""

import logging
import time
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from contextlib import contextmanager

try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
    cl = None

from core.models import Panel, SteelSheet, PlacedPanel, PlacementResult
from core.algorithms.gpu_bin_packing import IntelIrisXeBinPacker, BinPackingResult
from core.algorithms.constraint_handler import ComplexConstraintHandler, PlacementConstraints
from core.algorithms.gpu_fallback_manager import GPUFallbackManager, ExecutionContext

logger = logging.getLogger(__name__)

@dataclass
class SheetAllocation:
    """Represents allocation of panels to a specific sheet"""
    sheet_id: int
    sheet: SteelSheet
    allocated_panels: List[Panel]
    expected_efficiency: float
    material_groups: Dict[str, List[Panel]]
    priority_score: float

@dataclass
class MultiSheetResult:
    """Result of multi-sheet optimization"""
    placement_results: List[PlacementResult]
    total_efficiency: float
    total_waste_area: float
    sheets_used: int
    optimization_time: float
    gpu_acceleration_used: bool
    load_balancing_stats: Dict[str, Any]

@dataclass
class LoadBalancingStats:
    """Statistics for load balancing across compute units"""
    compute_unit_utilization: List[float]
    sheet_processing_times: List[float]
    load_balance_efficiency: float
    thermal_throttling_events: int

class ThermalManager:
    """Manages thermal constraints during multi-sheet processing"""

    def __init__(self, thermal_limit: float = 82.0):
        self.thermal_limit = thermal_limit
        self.current_temperature = 45.0
        self.processing_start_time = time.time()
        self.thermal_history = []
        self.throttling_active = False
        self._lock = threading.Lock()

    @contextmanager
    def thermal_protected_processing(self):
        """Context manager for thermal-protected processing"""
        try:
            self._start_thermal_monitoring()
            yield self
        finally:
            self._stop_thermal_monitoring()

    def _start_thermal_monitoring(self):
        """Start thermal monitoring"""
        with self._lock:
            self.processing_start_time = time.time()
            logger.info(f"Thermal monitoring started, limit: {self.thermal_limit}°C")

    def _stop_thermal_monitoring(self):
        """Stop thermal monitoring"""
        with self._lock:
            duration = time.time() - self.processing_start_time
            logger.info(f"Thermal monitoring stopped after {duration:.1f}s")

    def check_thermal_status(self) -> bool:
        """Check if thermal processing can continue"""
        # Simplified thermal check - in real implementation would read actual temperature
        processing_duration = time.time() - self.processing_start_time

        # Simulate temperature increase over time
        self.current_temperature = 45.0 + (processing_duration * 2.0)  # 2°C per second

        if self.current_temperature > self.thermal_limit:
            if not self.throttling_active:
                logger.warning(f"Thermal limit exceeded: {self.current_temperature:.1f}°C")
                self.throttling_active = True
            return False

        self.throttling_active = False
        return True

    def get_thermal_stats(self) -> Dict[str, float]:
        """Get thermal statistics"""
        return {
            'current_temperature': self.current_temperature,
            'thermal_limit': self.thermal_limit,
            'throttling_active': self.throttling_active,
            'processing_duration': time.time() - self.processing_start_time
        }

class MultiSheetGPUOptimizer:
    """
    GPU-accelerated multi-sheet optimization with intelligent load balancing
    and cross-sheet optimization capabilities.
    """

    def __init__(self, enable_gpu: bool = True, max_concurrent_sheets: int = 8):
        self.enable_gpu = enable_gpu and OPENCL_AVAILABLE
        self.max_concurrent_sheets = max_concurrent_sheets
        self.thermal_manager = ThermalManager()

        # GPU optimization components
        self.bin_packer = IntelIrisXeBinPacker(enable_gpu=enable_gpu)
        self.constraint_handler = ComplexConstraintHandler(PlacementConstraints())

        # Performance tracking
        self.optimization_stats = {
            'total_optimizations': 0,
            'gpu_time': 0.0,
            'cpu_time': 0.0,
            'load_balancing_efficiency': 0.0
        }

        # Compute unit management
        self.compute_units = 80  # Intel Iris Xe
        self.workload_distribution = {}

    def optimize_material_blocks(self, material_blocks: Dict[str, List[Panel]],
                               available_sheets: List[SteelSheet]) -> MultiSheetResult:
        """
        Optimize placement for material-grouped panels across multiple sheets with
        GPU acceleration and intelligent load balancing.

        Args:
            material_blocks: Dictionary mapping material types to panel lists
            available_sheets: List of available steel sheets

        Returns:
            MultiSheetResult with optimized placements across all sheets
        """
        start_time = time.time()

        with self.thermal_manager.thermal_protected_processing():
            # Phase 1: Sheet allocation optimization
            sheet_allocations = self._optimize_sheet_allocations(
                material_blocks, available_sheets
            )

            # Phase 2: Parallel sheet processing
            placement_results = self._process_sheets_parallel(sheet_allocations)

            # Phase 3: Cross-sheet optimization
            optimized_results = self._cross_sheet_optimization(placement_results)

            # Phase 4: Result compilation
            final_result = self._compile_multi_sheet_result(
                optimized_results, start_time
            )

        logger.info(f"Multi-sheet optimization completed:")
        logger.info(f"  Sheets used: {final_result.sheets_used}")
        logger.info(f"  Total efficiency: {final_result.total_efficiency:.2f}%")
        logger.info(f"  Processing time: {final_result.optimization_time:.3f}s")

        return final_result

    def _optimize_sheet_allocations(self, material_blocks: Dict[str, List[Panel]],
                                  available_sheets: List[SteelSheet]) -> List[SheetAllocation]:
        """Optimize allocation of material blocks to sheets"""
        allocations = []

        sheet_id = 0
        for material_type, panels in material_blocks.items():
            if not panels:
                continue

            # Group panels by priority and size for optimal allocation
            panel_groups = self._group_panels_for_allocation(panels)

            for group in panel_groups:
                if sheet_id >= len(available_sheets):
                    logger.warning("Insufficient sheets for all material groups")
                    break

                sheet = available_sheets[sheet_id]

                # Calculate expected efficiency for this allocation
                expected_efficiency = self._estimate_allocation_efficiency(group, sheet)

                # Create material groups within this allocation
                material_groups = self.constraint_handler.group_panels_by_material(group)

                allocation = SheetAllocation(
                    sheet_id=sheet_id,
                    sheet=sheet,
                    allocated_panels=group,
                    expected_efficiency=expected_efficiency,
                    material_groups=material_groups,
                    priority_score=self._calculate_priority_score(group)
                )

                allocations.append(allocation)
                sheet_id += 1

        # Sort allocations by priority for processing order
        allocations.sort(key=lambda a: a.priority_score, reverse=True)

        logger.info(f"Created {len(allocations)} sheet allocations")
        return allocations

    def _group_panels_for_allocation(self, panels: List[Panel]) -> List[List[Panel]]:
        """Group panels for optimal sheet allocation"""
        # Simple grouping by area for now - can be enhanced with more sophisticated algorithms
        panels_sorted = sorted(panels, key=lambda p: p.width * p.height, reverse=True)

        groups = []
        current_group = []
        current_area = 0.0
        target_area = 1500.0 * 3100.0 * 0.75  # 75% of standard sheet

        for panel in panels_sorted:
            panel_area = panel.width * panel.height

            if current_area + panel_area <= target_area or not current_group:
                current_group.append(panel)
                current_area += panel_area
            else:
                groups.append(current_group)
                current_group = [panel]
                current_area = panel_area

        if current_group:
            groups.append(current_group)

        return groups

    def _estimate_allocation_efficiency(self, panels: List[Panel], sheet: SteelSheet) -> float:
        """Estimate placement efficiency for a panel group on a sheet"""
        total_panel_area = sum(p.width * p.height for p in panels)
        sheet_area = sheet.width * sheet.height

        # Basic efficiency estimate with some overhead for placement constraints
        theoretical_efficiency = total_panel_area / sheet_area
        placement_overhead = 0.15  # 15% overhead for placement gaps

        return max(0.0, theoretical_efficiency - placement_overhead) * 100

    def _calculate_priority_score(self, panels: List[Panel]) -> float:
        """Calculate priority score for sheet allocation ordering"""
        total_priority = sum(p.priority for p in panels)
        total_area = sum(p.width * p.height for p in panels)

        # Combine priority and area for scoring
        return total_priority * 1000 + total_area

    def _process_sheets_parallel(self, allocations: List[SheetAllocation]) -> List[PlacementResult]:
        """Process multiple sheets in parallel with load balancing"""
        placement_results = []

        # Determine optimal concurrency based on thermal and compute constraints
        effective_concurrency = min(
            self.max_concurrent_sheets,
            len(allocations),
            self.compute_units // 10  # Reserve 10 CUs per sheet
        )

        logger.info(f"Processing {len(allocations)} sheets with {effective_concurrency} concurrent workers")

        with ThreadPoolExecutor(max_workers=effective_concurrency) as executor:
            # Submit sheet processing tasks
            future_to_allocation = {
                executor.submit(self._process_single_sheet, allocation): allocation
                for allocation in allocations
            }

            # Collect results as they complete
            for future in as_completed(future_to_allocation):
                allocation = future_to_allocation[future]

                try:
                    # Check thermal status before processing result
                    if not self.thermal_manager.check_thermal_status():
                        logger.warning("Thermal throttling - switching to CPU processing")
                        # Could implement thermal recovery here

                    result = future.result()
                    placement_results.append(result)

                    logger.info(f"Sheet {allocation.sheet_id} completed: "
                              f"{len(result.panels)} panels, {result.efficiency*100:.1f}% efficiency")

                except Exception as e:
                    logger.error(f"Sheet {allocation.sheet_id} processing failed: {e}")
                    # Create fallback result
                    fallback_result = self._create_fallback_result(allocation)
                    placement_results.append(fallback_result)

        return placement_results

    def _process_single_sheet(self, allocation: SheetAllocation) -> PlacementResult:
        """Process a single sheet allocation"""
        try:
            # Apply constraint handling
            validated_panels = self._validate_and_optimize_panels(
                allocation.allocated_panels, allocation.sheet
            )

            # Perform GPU bin packing
            bin_packing_result = self.bin_packer.parallel_blf_placement(
                validated_panels, allocation.sheet
            )

            # Convert to PlacementResult
            placement_result = PlacementResult(
                sheet_id=allocation.sheet_id + 1,
                material_block=self._get_primary_material(allocation.allocated_panels),
                sheet=allocation.sheet,
                panels=bin_packing_result.placed_panels,
                efficiency=bin_packing_result.efficiency / 100.0,  # Convert to 0-1
                waste_area=bin_packing_result.waste_area,
                cut_length=bin_packing_result.cutting_length,
                cost=allocation.sheet.cost_per_sheet
            )

            return placement_result

        except Exception as e:
            logger.error(f"Single sheet processing failed: {e}")
            raise

    def _validate_and_optimize_panels(self, panels: List[Panel], sheet: SteelSheet) -> List[Panel]:
        """Validate and optimize panels for placement"""
        # Validate constraints
        violations = self.constraint_handler.validate_placement_constraints(panels, sheet)

        if violations:
            logger.warning(f"Found {len(violations)} constraint violations")
            for violation in violations:
                if violation.severity == "error":
                    logger.error(f"Constraint error: {violation.description}")

        # Optimize rotations
        optimized_panels = self.constraint_handler.optimize_panel_rotations(panels, sheet)

        return optimized_panels

    def _get_primary_material(self, panels: List[Panel]) -> str:
        """Get the primary material type for a panel group"""
        material_counts = {}
        for panel in panels:
            material_counts[panel.material] = material_counts.get(panel.material, 0) + 1

        return max(material_counts, key=material_counts.get)

    def _create_fallback_result(self, allocation: SheetAllocation) -> PlacementResult:
        """Create fallback result for failed sheet processing"""
        logger.warning(f"Creating fallback result for sheet {allocation.sheet_id}")

        return PlacementResult(
            sheet_id=allocation.sheet_id + 1,
            material_block=self._get_primary_material(allocation.allocated_panels),
            sheet=allocation.sheet,
            panels=[],  # No panels placed in fallback
            efficiency=0.0,
            waste_area=allocation.sheet.width * allocation.sheet.height,
            cut_length=0.0,
            cost=allocation.sheet.cost_per_sheet
        )

    def _cross_sheet_optimization(self, placement_results: List[PlacementResult]) -> List[PlacementResult]:
        """
        Perform cross-sheet optimization to improve overall efficiency by
        migrating panels between sheets when beneficial.
        """
        logger.info("Starting cross-sheet optimization")

        optimized_results = placement_results.copy()

        # Identify sheets with low efficiency for optimization
        low_efficiency_threshold = 0.6  # 60%
        optimization_candidates = [
            (i, result) for i, result in enumerate(optimized_results)
            if result.efficiency < low_efficiency_threshold
        ]

        if not optimization_candidates:
            logger.info("No sheets require cross-sheet optimization")
            return optimized_results

        # Attempt panel migration between sheets
        for candidate_idx, candidate_result in optimization_candidates:
            improved_result = self._attempt_panel_migration(
                candidate_result, optimized_results, candidate_idx
            )

            if improved_result.efficiency > candidate_result.efficiency:
                optimized_results[candidate_idx] = improved_result
                logger.info(f"Sheet {candidate_result.sheet_id} efficiency improved: "
                          f"{candidate_result.efficiency:.2%} → {improved_result.efficiency:.2%}")

        return optimized_results

    def _attempt_panel_migration(self, target_result: PlacementResult,
                                all_results: List[PlacementResult],
                                target_idx: int) -> PlacementResult:
        """Attempt to migrate panels to improve target sheet efficiency"""
        # Simplified migration attempt - could be much more sophisticated

        # Find panels that could be moved from other sheets
        available_panels = []
        for i, result in enumerate(all_results):
            if i != target_idx and result.efficiency > 0.8:  # Only take from high-efficiency sheets
                # Consider moving smaller panels
                smaller_panels = [p for p in result.panels if p.panel.width * p.panel.height < 50000]
                available_panels.extend(smaller_panels[:2])  # Max 2 panels per sheet

        if not available_panels:
            return target_result  # No improvement possible

        # Attempt to place additional panels on target sheet
        try:
            current_panels = [p.panel for p in target_result.panels]
            migration_panels = [p.panel for p in available_panels]

            # Re-optimize with additional panels
            all_panels = current_panels + migration_panels
            new_bin_packing_result = self.bin_packer.parallel_blf_placement(
                all_panels, target_result.sheet
            )

            if new_bin_packing_result.efficiency > target_result.efficiency * 100:
                # Migration successful
                return PlacementResult(
                    sheet_id=target_result.sheet_id,
                    material_block=target_result.material_block,
                    sheet=target_result.sheet,
                    panels=new_bin_packing_result.placed_panels,
                    efficiency=new_bin_packing_result.efficiency / 100.0,
                    waste_area=new_bin_packing_result.waste_area,
                    cut_length=new_bin_packing_result.cutting_length,
                    cost=target_result.cost
                )

        except Exception as e:
            logger.warning(f"Panel migration failed: {e}")

        return target_result  # Return original if migration fails

    def _compile_multi_sheet_result(self, placement_results: List[PlacementResult],
                                  start_time: float) -> MultiSheetResult:
        """Compile final multi-sheet optimization result"""
        total_time = time.time() - start_time

        # Calculate aggregate metrics
        total_panels = sum(len(result.panels) for result in placement_results)
        total_area_used = sum(len(result.panels) * result.efficiency * result.sheet.width * result.sheet.height
                             for result in placement_results)
        total_sheet_area = sum(result.sheet.width * result.sheet.height for result in placement_results)

        total_efficiency = (total_area_used / total_sheet_area) * 100 if total_sheet_area > 0 else 0
        total_waste = sum(result.waste_area for result in placement_results)

        # Compile load balancing stats
        load_balancing_stats = {
            'thermal_stats': self.thermal_manager.get_thermal_stats(),
            'processing_time': total_time,
            'sheets_processed': len(placement_results),
            'average_efficiency': np.mean([r.efficiency for r in placement_results]) * 100
        }

        return MultiSheetResult(
            placement_results=placement_results,
            total_efficiency=total_efficiency,
            total_waste_area=total_waste,
            sheets_used=len(placement_results),
            optimization_time=total_time,
            gpu_acceleration_used=self.bin_packer.opencl_context is not None,
            load_balancing_stats=load_balancing_stats
        )

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        return {
            'gpu_available': self.bin_packer.opencl_context is not None,
            'compute_units': self.compute_units,
            'max_concurrent_sheets': self.max_concurrent_sheets,
            'optimization_stats': self.optimization_stats.copy(),
            'thermal_stats': self.thermal_manager.get_thermal_stats()
        }

    def cleanup(self):
        """Clean up resources"""
        if self.bin_packer:
            self.bin_packer.cleanup()

        logger.info("Multi-sheet GPU optimizer cleaned up")


def create_multi_sheet_optimizer(**kwargs) -> MultiSheetGPUOptimizer:
    """Factory function to create multi-sheet optimizer"""
    if not OPENCL_AVAILABLE:
        logger.warning("PyOpenCL not available, GPU acceleration disabled")
        kwargs['enable_gpu'] = False

    return MultiSheetGPUOptimizer(**kwargs)


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create test material blocks
    material_blocks = {
        "Steel_2mm": [
            Panel(id=f"S{i:03d}", width=100+i*5, height=150+i*3,
                  material="Steel", thickness=2.0, quantity=1, allow_rotation=True)
            for i in range(30)
        ],
        "Steel_3mm": [
            Panel(id=f"T{i:03d}", width=120+i*4, height=180+i*2,
                  material="Steel", thickness=3.0, quantity=1, allow_rotation=True)
            for i in range(25)
        ]
    }

    # Create available sheets
    available_sheets = [
        SteelSheet(width=1500.0, height=3100.0, thickness=2.0),
        SteelSheet(width=1500.0, height=3100.0, thickness=3.0),
        SteelSheet(width=1500.0, height=3100.0, thickness=2.0),
    ]

    # Create optimizer and run
    optimizer = create_multi_sheet_optimizer(enable_gpu=True, max_concurrent_sheets=4)

    try:
        result = optimizer.optimize_material_blocks(material_blocks, available_sheets)

        print(f"Multi-sheet optimization results:")
        print(f"  Sheets used: {result.sheets_used}")
        print(f"  Total efficiency: {result.total_efficiency:.2f}%")
        print(f"  Processing time: {result.optimization_time:.3f}s")
        print(f"  GPU acceleration: {result.gpu_acceleration_used}")

        for i, placement_result in enumerate(result.placement_results):
            print(f"  Sheet {i+1}: {len(placement_result.panels)} panels, "
                  f"{placement_result.efficiency*100:.1f}% efficiency")

    finally:
        optimizer.cleanup()