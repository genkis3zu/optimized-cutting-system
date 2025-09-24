"""
Optimization engine for steel cutting with algorithm selection
鋼板切断最適化エンジン（アルゴリズム選択機能付き）
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
import threading
from datetime import datetime

from core.models import (
    Panel, SteelSheet, PlacementResult, PlacedPanel,
    OptimizationConstraints
)
from core.material_manager import get_material_manager
from core.pi_manager import get_pi_manager


@dataclass
class OptimizationMetrics:
    """Performance metrics for optimization"""
    algorithm: str
    processing_time: float
    efficiency: float
    panels_placed: int
    sheets_used: int
    memory_usage: float = 0.0
    iterations: int = 0


class OptimizationAlgorithm(ABC):
    """
    Abstract base class for optimization algorithms
    最適化アルゴリズムの抽象基底クラス
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def optimize(
        self,
        panels: List[Panel],
        sheet: SteelSheet,
        constraints: OptimizationConstraints
    ) -> PlacementResult:
        """
        Main optimization method
        メイン最適化メソッド
        """
        # Implementation must be provided by concrete classes
        raise NotImplementedError(f"Algorithm {self.name} must implement optimize() method")
    
    @abstractmethod
    def estimate_time(self, panel_count: int, complexity: float) -> float:
        """
        Estimate processing time in seconds
        処理時間の見積もり（秒）
        """
        # Implementation must be provided by concrete classes
        raise NotImplementedError(f"Algorithm {self.name} must implement estimate_time() method")
    
    def calculate_complexity(self, panels: List[Panel]) -> float:
        """
        Calculate problem complexity (0-1)
        問題の複雑度を計算（0-1）
        """
        if not panels:
            return 0.0
        
        # Size diversity factor
        unique_sizes = len(set((p.width, p.height) for p in panels))
        size_diversity = min(1.0, unique_sizes / len(panels))
        
        # Quantity factor
        total_quantity = sum(p.quantity for p in panels)
        quantity_factor = min(1.0, total_quantity / 100)
        
        # Rotation complexity
        rotation_factor = sum(1 for p in panels if p.allow_rotation) / len(panels)
        
        # Material diversity
        unique_materials = len(set(p.material for p in panels))
        material_factor = min(1.0, unique_materials / 10)
        
        # Combined complexity
        complexity = (
            size_diversity * 0.4 +
            quantity_factor * 0.3 +
            rotation_factor * 0.2 +
            material_factor * 0.1
        )
        
        return min(1.0, complexity)
    
    def group_by_material(self, panels: List[Panel]) -> Dict[str, List[Panel]]:
        """
        Group panels by material type
        材質別にパネルをグループ化
        """
        material_groups = defaultdict(list)
        for panel in panels:
            # Expand panels based on quantity
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
                material_groups[panel.material].append(individual_panel)
        
        return dict(material_groups)
    
    def validate_placement(self, placement: PlacementResult) -> bool:
        """
        Validate placement result
        配置結果の検証
        """
        try:
            # Re-enable overlap validation after fixing FFD algorithm
            placement.validate_no_overlaps()
            placement.validate_within_bounds()
            return True
        except ValueError as e:
            self.logger.error(f"Placement validation failed: {e}")
            return False


class OptimizationEngine:
    """
    Main optimization engine with algorithm selection
    アルゴリズム選択機能付きメイン最適化エンジン
    """
    
    def __init__(self):
        self.algorithms = {}
        self.logger = logging.getLogger(__name__)
        self.performance_monitor = PerformanceMonitor()
        self.timeout_manager = TimeoutManager()
        
    def register_algorithm(self, algorithm: OptimizationAlgorithm):
        """Register an optimization algorithm"""
        self.algorithms[algorithm.name] = algorithm
        self.logger.info(f"Registered algorithm: {algorithm.name}")
    
    def select_algorithm(
        self,
        panels: List[Panel],
        constraints: OptimizationConstraints
    ) -> str:
        """
        Select optimal algorithm based on problem characteristics
        問題特性に基づく最適アルゴリズムの選択
        """
        if not panels:
            # Default for empty input - use first available algorithm
            return next(iter(self.algorithms.keys())) if self.algorithms else "FFD"

        # Calculate problem characteristics
        complexity = self._calculate_problem_complexity(panels)
        time_budget = constraints.time_budget
        panel_count = sum(p.quantity for p in panels)
        total_area = sum(p.area * p.quantity for p in panels)

        # Material diversity factor
        unique_materials = len(set(p.material for p in panels))
        material_factor = min(1.0, unique_materials / 5.0)

        # Size variance factor (affects packing difficulty)
        areas = [p.area for p in panels]
        avg_area = sum(areas) / len(areas)
        variance = sum((a - avg_area) ** 2 for a in areas) / len(areas)
        size_variance = min(1.0, variance / (avg_area ** 2))

        # Adjusted complexity considering additional factors
        adjusted_complexity = min(1.0, complexity + material_factor * 0.1 + size_variance * 0.1)

        self.logger.debug(
            f"Algorithm selection factors: complexity={complexity:.3f}, "
            f"adjusted={adjusted_complexity:.3f}, panels={panel_count}, "
            f"time_budget={time_budget:.1f}s, materials={unique_materials}"
        )

        # Enhanced algorithm selection logic based on Japanese manufacturing specs
        available_algorithms = list(self.algorithms.keys())

        # Priority 1: Small, simple problems - use FFD for speed
        if panel_count <= 10 and adjusted_complexity < 0.3 and time_budget >= 1.0:
            if "FFD" in available_algorithms:
                self.logger.info("Selected FFD: small simple problem, target 70-75% efficiency")
                return "FFD"
            elif "MockFFD" in available_algorithms:  # For testing
                self.logger.info("Selected MockFFD: small simple problem (test mode)")
                return "MockFFD"

        # Priority 2: Medium problems with good time budget - use BFD if available
        if panel_count <= 50 and adjusted_complexity < 0.7 and time_budget >= 5.0:
            if "BFD" in available_algorithms:
                self.logger.info("Selected BFD: medium problem, target 80-85% efficiency")
                return "BFD"
            elif "FFD" in available_algorithms:
                self.logger.info("Selected FFD: BFD not available, fallback for medium problem")
                return "FFD"

        # Priority 3: Large/complex problems with sufficient time - use HYBRID
        if time_budget >= 30.0 and total_area > 100000:  # Large cutting area
            if "HYBRID" in available_algorithms:
                self.logger.info("Selected HYBRID: large complex problem, target 85%+ efficiency")
                return "HYBRID"
            elif "BFD" in available_algorithms:
                self.logger.info("Selected BFD: HYBRID not available, fallback for large problem")
                return "BFD"

        # Priority 4: Time-constrained problems
        if time_budget < 1.0:
            if "FFD_FAST" in available_algorithms:
                self.logger.info("Selected FFD_FAST: very tight time constraint")
                return "FFD_FAST"
            elif "FFD" in available_algorithms:
                self.logger.info("Selected FFD: tight time constraint, fast execution needed")
                return "FFD"

        # Priority 5: High material diversity - needs special handling
        if unique_materials > 3 and constraints.material_separation:
            if "BFD" in available_algorithms:
                self.logger.info("Selected BFD: high material diversity, better grouping")
                return "BFD"
            elif "FFD" in available_algorithms:
                self.logger.info("Selected FFD: fallback for material diversity")
                return "FFD"

        # Default fallback: use best available algorithm
        if "FFD" in available_algorithms:
            self.logger.info("Selected FFD: default fallback algorithm")
            return "FFD"
        elif "MockFFD" in available_algorithms:  # For testing
            self.logger.info("Selected MockFFD: default fallback (test mode)")
            return "MockFFD"
        elif available_algorithms:
            fallback = available_algorithms[0]
            self.logger.warning(f"Selected {fallback}: only available algorithm")
            return fallback
        else:
            self.logger.error("No algorithms available!")
            raise RuntimeError("No optimization algorithms registered")
    
    def _calculate_problem_complexity(self, panels: List[Panel]) -> float:
        """Calculate normalized problem complexity"""
        if not panels:
            return 0.0
        
        # Use first algorithm to calculate complexity
        if self.algorithms:
            first_algo = next(iter(self.algorithms.values()))
            return first_algo.calculate_complexity(panels)
        
        # Fallback calculation
        unique_sizes = len(set((p.width, p.height) for p in panels))
        total_panels = sum(p.quantity for p in panels)
        diversity = unique_sizes / len(panels) if panels else 0
        
        return min(1.0, (total_panels * diversity) / 1000)

    def _calculate_expanded_dimensions(self, panels: List[Panel], pi_manager) -> Dict[str, float]:
        """
        Calculate expanded dimensions for all panels using PI codes
        PIコードを使用してすべてのパネルの展開寸法を計算

        Args:
            panels: List of panels to process
            pi_manager: PIManager instance

        Returns:
            Dictionary with expansion summary statistics
        """
        expanded_count = 0
        total_w_expansion = 0.0
        total_h_expansion = 0.0

        for panel in panels:
            panel.calculate_expanded_dimensions(pi_manager)

            if panel.has_expansion():
                expanded_count += 1
                expansion_info = panel.get_expansion_info()
                total_w_expansion += expansion_info['w_expansion']
                total_h_expansion += expansion_info['h_expansion']

        avg_w_expansion = total_w_expansion / expanded_count if expanded_count > 0 else 0.0
        avg_h_expansion = total_h_expansion / expanded_count if expanded_count > 0 else 0.0

        return {
            'expanded_panels': expanded_count,
            'total_panels': len(panels),
            'avg_w_expansion': avg_w_expansion,
            'avg_h_expansion': avg_h_expansion,
            'expansion_percentage': (expanded_count / len(panels)) * 100 if panels else 0.0
        }

    def select_best_materials(self, panels: List[Panel]) -> Dict[str, List[SteelSheet]]:
        """
        Select best materials from inventory for each material type
        在庫から各材質タイプに最適な母材を選択
        """
        material_manager = get_material_manager()
        material_groups = {}

        # Group panels by material type
        for panel in panels:
            if panel.material not in material_groups:
                material_groups[panel.material] = []
            material_groups[panel.material].append(panel)

        selected_materials = {}

        for material_type, material_panels in material_groups.items():
            # Normalize material code using material manager
            normalized_material = material_manager.normalize_material_code(material_type)

            # Get available sheets for this material type
            available_sheets = material_manager.get_sheets_by_type(normalized_material)

            if not available_sheets:
                self.logger.warning(f"No sheets available for material type: {material_type} (normalized: {normalized_material})")
                continue

            # Convert MaterialSheet to SteelSheet for compatibility
            steel_sheets = []
            for sheet in available_sheets:
                if sheet.availability > 0:  # Only include sheets with availability
                    steel_sheet = SteelSheet(
                        width=sheet.width,
                        height=sheet.height,
                        thickness=sheet.thickness,
                        material=sheet.material_type,
                        cost_per_sheet=sheet.cost_per_sheet
                    )
                    steel_sheets.append(steel_sheet)

            if steel_sheets:
                # Sort by efficiency: smaller sheets first, then by cost
                steel_sheets.sort(key=lambda s: (s.area, s.cost_per_sheet))
                selected_materials[normalized_material] = steel_sheets
                self.logger.info(f"Selected {len(steel_sheets)} sheet options for material {normalized_material}")
            else:
                self.logger.warning(f"No sheets with availability for material type: {normalized_material}")

        return selected_materials

    def optimize(
        self,
        panels: List[Panel],
        constraints: Optional[OptimizationConstraints] = None,
        algorithm_hint: Optional[str] = None
    ) -> List[PlacementResult]:
        """
        Main optimization method
        メイン最適化メソッド
        """
        if not panels:
            self.logger.warning("No panels provided for optimization")
            return []
        
        # Set default constraints
        if constraints is None:
            constraints = OptimizationConstraints()
        
        constraints.validate()
        
        # Select algorithm
        algorithm_name = algorithm_hint or self.select_algorithm(panels, constraints)
        
        if algorithm_name not in self.algorithms:
            self.logger.warning(f"Algorithm {algorithm_name} not found, using available fallback")
            # Find first available algorithm
            if self.algorithms:
                algorithm_name = next(iter(self.algorithms.keys()))
                self.logger.info(f"Using fallback algorithm: {algorithm_name}")
            else:
                raise RuntimeError("No algorithms registered in optimization engine")

        algorithm = self.algorithms[algorithm_name]
        
        self.logger.info(
            f"Starting optimization with {algorithm_name} "
            f"for {len(panels)} panel types, "
            f"total quantity: {sum(p.quantity for p in panels)}"
        )
        
        # Calculate expanded dimensions using PI codes
        pi_manager = get_pi_manager()
        expansion_summary = self._calculate_expanded_dimensions(panels, pi_manager)
        if expansion_summary['expanded_panels'] > 0:
            self.logger.info(
                f"Calculated expanded dimensions for {expansion_summary['expanded_panels']} panels "
                f"(Average expansion: W+{expansion_summary['avg_w_expansion']:.1f}mm, "
                f"H+{expansion_summary['avg_h_expansion']:.1f}mm)"
            )

        # Start performance monitoring
        start_time = time.time()
        self.performance_monitor.start_monitoring()

        try:
            # Select best materials from inventory
            available_materials = self.select_best_materials(panels)

            if not available_materials:
                raise RuntimeError("No suitable materials found in inventory for the given panels")

            # Group panels by material if required
            if constraints.material_separation:
                material_groups = algorithm.group_by_material(panels)
                results = []

                for material, material_panels in material_groups.items():
                    # Normalize material code
                    material_manager = get_material_manager()
                    normalized_material = material_manager.normalize_material_code(material)

                    self.logger.info(f"Processing material block: {material} -> {normalized_material} ({len(material_panels)} panels)")

                    # Get available sheets for this material
                    if normalized_material not in available_materials:
                        self.logger.warning(f"No inventory found for material {normalized_material}, skipping {len(material_panels)} panels")
                        continue

                    available_sheets = available_materials[normalized_material]

                    # Try each available sheet until we find a good fit
                    best_result = None
                    best_efficiency = 0

                    # Sort material panels by priority and block order for consistency
                    material_panels.sort(key=lambda p: (p.priority, p.block_order))

                    for sheet in available_sheets:
                        self.logger.debug(f"Trying sheet {sheet.width}×{sheet.height}mm for material {normalized_material}")

                        # Optimize material block with individual timeout
                        material_constraints = OptimizationConstraints(
                            max_sheets=1,  # Try one sheet at a time
                            kerf_width=constraints.kerf_width,
                            min_waste_piece=constraints.min_waste_piece,
                            allow_rotation=constraints.allow_rotation,
                            material_separation=False,  # Already separated
                            time_budget=constraints.time_budget / len(material_groups) / len(available_sheets),  # Distribute time budget
                            target_efficiency=constraints.target_efficiency
                        )

                        result = self._optimize_with_timeout(
                            algorithm, material_panels, sheet, material_constraints
                        )

                        if result and result.efficiency > best_efficiency:
                            best_result = result
                            best_efficiency = result.efficiency
                            self.logger.info(f"Better solution found with {sheet.width}×{sheet.height}mm: {result.efficiency:.1%} efficiency")

                    # Use the best result found for this material
                    if best_result:
                        best_result.material_block = normalized_material
                        best_result.sheet_id = len(results) + 1  # Sequential sheet IDs
                        results.append(best_result)

                        self.logger.info(
                            f"Material {normalized_material}: {len(best_result.panels)} panels placed, "
                            f"efficiency {best_result.efficiency:.1%} on {best_result.sheet.width}×{best_result.sheet.height}mm sheet"
                        )
                    else:
                        self.logger.warning(f"Failed to optimize material block: {normalized_material}")

                # Log overall results
                total_panels = sum(len(r.panels) for r in results)
                total_used_area = sum(r.used_area for r in results)
                total_efficiency = sum(r.efficiency * r.used_area for r in results) / total_used_area if total_used_area > 0 else 0
                self.logger.info(
                    f"Material separation complete: {len(results)} sheets, "
                    f"{total_panels} total panels, overall efficiency {total_efficiency:.1%}"
                )

                return results
            else:
                # Multi-sheet optimization without material separation
                # Use the best available material/sheet combination for all panels
                if available_materials:
                    # Choose the material with the most available sheet options
                    best_material = max(available_materials.keys(),
                                      key=lambda k: len(available_materials[k]))
                    available_sheets = available_materials[best_material]

                    # Choose the largest available sheet for maximum capacity
                    best_sheet = max(available_sheets, key=lambda s: s.area)

                    self.logger.info(
                        f"Multi-sheet optimization: Using {best_material} sheets ({best_sheet.width}×{best_sheet.height}mm) "
                        f"for {len(panels)} panel types, total quantity: {sum(p.quantity for p in panels)}"
                    )

                    results = []
                    remaining_panels = panels.copy()
                    sheet_count = 0
                    max_sheets = constraints.max_sheets

                    # 100% placement guarantee: Continue until all panels are placed
                    # 100%配置保証: 全パネルが配置されるまで継続
                    while remaining_panels and sheet_count < 1000:  # Safety limit to prevent infinite loop
                        sheet_count += 1

                        # Optimize current sheet
                        sheet_constraints = OptimizationConstraints(
                            max_sheets=1,
                            kerf_width=constraints.kerf_width,
                            min_waste_piece=constraints.min_waste_piece,
                            allow_rotation=constraints.allow_rotation,
                            material_separation=False,
                            time_budget=constraints.time_budget / max_sheets,  # Distribute time budget
                            target_efficiency=constraints.target_efficiency
                        )

                        result = self._optimize_with_timeout(
                            algorithm, remaining_panels, best_sheet, sheet_constraints
                        )

                        if result and len(result.panels) > 0:
                            result.sheet_id = sheet_count
                            result.material_block = best_material
                            results.append(result)

                            # Remove placed panels from remaining list
                            # Fix: Handle individual panel IDs correctly (e.g., "562210_1", "562210_2")
                            new_remaining = []
                            for panel in remaining_panels:
                                # Count how many individual panels with base ID were placed
                                base_id = panel.id
                                placed_count = sum(1 for p in result.panels
                                                 if p.panel.id == base_id or p.panel.id.startswith(f"{base_id}_"))

                                if placed_count < panel.quantity:
                                    # Create panel with reduced quantity
                                    remaining_panel = Panel(
                                        id=panel.id,
                                        width=panel.width,
                                        height=panel.height,
                                        quantity=panel.quantity - placed_count,
                                        material=panel.material,
                                        thickness=panel.thickness,
                                        priority=panel.priority,
                                        allow_rotation=panel.allow_rotation,
                                        block_order=panel.block_order,
                                        pi_code=panel.pi_code,
                                        expanded_width=panel.expanded_width,
                                        expanded_height=panel.expanded_height
                                    )
                                    new_remaining.append(remaining_panel)

                                    self.logger.debug(
                                        f"Panel {base_id}: {placed_count}/{panel.quantity} placed, "
                                        f"{remaining_panel.quantity} remaining"
                                    )
                                else:
                                    self.logger.debug(f"Panel {base_id}: All {panel.quantity} panels placed")
                            remaining_panels = new_remaining

                            self.logger.info(
                                f"Sheet {sheet_count}: {len(result.panels)} panels placed, "
                                f"efficiency {result.efficiency:.1%}, {len(remaining_panels)} panel types remaining"
                            )
                        else:
                            # Force placement of at least one panel to ensure progress
                            # 進捗を保証するため最低1つのパネルを強制配置
                            if remaining_panels:
                                self.logger.warning(f"Sheet {sheet_count} failed, attempting force placement")

                                # Try to place the smallest panel
                                smallest_panel = min(remaining_panels, key=lambda p: p.area)
                                force_result = self._force_single_panel_placement(smallest_panel, best_sheet, algorithm, sheet_constraints)

                                if force_result:
                                    force_result.sheet_id = sheet_count
                                    force_result.material_block = best_material
                                    results.append(force_result)

                                    # Update remaining panels
                                    for panel in remaining_panels:
                                        if panel.id == smallest_panel.id:
                                            if panel.quantity > 1:
                                                panel.quantity -= 1
                                                break
                                            else:
                                                remaining_panels.remove(panel)
                                                break

                                    self.logger.info(f"Force placement successful: 1 panel on sheet {sheet_count}")
                                else:
                                    self.logger.error("Force placement failed, stopping optimization")
                                    break
                            else:
                                break

                    # Log final results
                    total_placed = sum(len(r.panels) for r in results)
                    total_input = sum(p.quantity for p in panels)
                    placement_rate = (total_placed / total_input) * 100 if total_input > 0 else 0

                    self.logger.info(
                        f"Multi-sheet optimization complete: {len(results)} sheets, "
                        f"{total_placed}/{total_input} panels placed ({placement_rate:.1f}%)"
                    )

                    return results
                else:
                    self.logger.error("No materials available for optimization")
                    return []
        
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            return []
        
        finally:
            processing_time = time.time() - start_time
            self.performance_monitor.stop_monitoring()
            
            self.logger.info(f"Optimization completed in {processing_time:.2f} seconds")

    def _force_single_panel_placement(
        self,
        panel: Panel,
        sheet: SteelSheet,
        algorithm: OptimizationAlgorithm,
        constraints: OptimizationConstraints
    ) -> Optional[PlacementResult]:
        """
        Force placement of a single panel on a new sheet
        単一パネルの新シートへの強制配置
        """
        try:
            single_panel = Panel(
                id=f"{panel.id}_force",
                width=panel.width,
                height=panel.height,
                quantity=1,
                material=panel.material,
                thickness=panel.thickness,
                allow_rotation=panel.allow_rotation
            )

            result = algorithm.optimize([single_panel], sheet, constraints)
            if result and len(result.panels) > 0:
                return result

        except Exception as e:
            self.logger.error(f"Force placement error: {e}")

        return None
    
    def _optimize_with_timeout(
        self,
        algorithm: OptimizationAlgorithm,
        panels: List[Panel],
        sheet: SteelSheet,
        constraints: OptimizationConstraints
    ) -> Optional[PlacementResult]:
        """Execute optimization with timeout handling and error recovery"""
        start_time = time.time()

        def run_optimization():
            try:
                return algorithm.optimize(panels, sheet, constraints)
            except Exception as e:
                self.logger.error(f"Algorithm {algorithm.name} internal error: {e}")
                # Return partial result if available
                return self._create_empty_result(sheet, algorithm.name, start_time)

        try:
            # Monitor resource usage during optimization
            if not self.performance_monitor.check_resource_limits():
                self.logger.warning("Resource limits exceeded, using conservative approach")
                constraints.time_budget = min(constraints.time_budget, 5.0)

            # Use timeout manager for time-limited execution
            result = self.timeout_manager.execute_with_timeout(
                run_optimization,
                timeout=constraints.time_budget
            )

            if result:
                # Validate result integrity
                if algorithm.validate_placement(result):
                    # Calculate final metrics and timing
                    processing_time = time.time() - start_time
                    result.algorithm = algorithm.name
                    result.processing_time = processing_time
                    result.calculate_efficiency()

                    # Check if efficiency meets target
                    if result.efficiency >= constraints.target_efficiency:
                        self.logger.info(
                            f"Algorithm {algorithm.name} achieved target efficiency: "
                            f"{result.efficiency:.1%} (target: {constraints.target_efficiency:.1%})"
                        )
                    else:
                        self.logger.info(
                            f"Algorithm {algorithm.name} efficiency {result.efficiency:.1%} "
                            f"below target {constraints.target_efficiency:.1%} but acceptable"
                        )

                    # Validate performance targets
                    estimated_time = algorithm.estimate_time(len(panels), algorithm.calculate_complexity(panels))
                    if processing_time > estimated_time * 2:
                        self.logger.warning(
                            f"Algorithm {algorithm.name} took {processing_time:.3f}s, "
                            f"estimated {estimated_time:.3f}s (performance warning)"
                        )

                    return result
                else:
                    self.logger.error(f"Algorithm {algorithm.name} produced invalid placement")
                    return self._create_empty_result(sheet, algorithm.name, start_time)
            else:
                self.logger.warning(f"Algorithm {algorithm.name} returned no result")
                return self._create_empty_result(sheet, algorithm.name, start_time)

        except FutureTimeoutError:
            self.logger.warning(
                f"Algorithm {algorithm.name} timed out after {constraints.time_budget:.1f}s, "
                f"trying fallback approach"
            )
            # Attempt quick fallback optimization
            return self._fallback_optimization(algorithm, panels, sheet, constraints, start_time)

        except Exception as e:
            self.logger.error(f"Algorithm {algorithm.name} failed with exception: {e}")
            return self._create_empty_result(sheet, algorithm.name, start_time)

    def _create_empty_result(self, sheet: SteelSheet, algorithm_name: str, start_time: float) -> PlacementResult:
        """Create empty result for failed optimizations"""
        return PlacementResult(
            sheet_id=1,
            material_block=sheet.material,
            sheet=sheet,
            panels=[],
            efficiency=0.0,
            waste_area=sheet.area,
            cut_length=0.0,
            cost=sheet.cost_per_sheet,
            algorithm=f"{algorithm_name}_FAILED",
            processing_time=time.time() - start_time,
            timestamp=datetime.now()
        )

    def _fallback_optimization(
        self,
        algorithm: OptimizationAlgorithm,
        panels: List[Panel],
        sheet: SteelSheet,
        constraints: OptimizationConstraints,
        start_time: float
    ) -> PlacementResult:
        """Attempt quick fallback optimization with reduced constraints"""
        try:
            # Try with only the largest panels and reduced time
            fallback_panels = sorted(panels, key=lambda p: p.area, reverse=True)[:min(10, len(panels))]
            fallback_constraints = OptimizationConstraints(
                max_sheets=1,
                kerf_width=constraints.kerf_width,
                min_waste_piece=constraints.min_waste_piece,
                allow_rotation=constraints.allow_rotation,
                material_separation=False,
                time_budget=2.0,  # Quick fallback
                target_efficiency=0.3  # Lower target for fallback
            )

            self.logger.info(f"Attempting fallback optimization with {len(fallback_panels)} largest panels")

            result = self.timeout_manager.execute_with_timeout(
                lambda: algorithm.optimize(fallback_panels, sheet, fallback_constraints),
                timeout=2.0
            )

            if result and algorithm.validate_placement(result):
                result.algorithm = f"{algorithm.name}_FALLBACK"
                result.processing_time = time.time() - start_time
                result.calculate_efficiency()
                self.logger.info(f"Fallback optimization placed {len(result.panels)} panels")
                return result

        except Exception as e:
            self.logger.error(f"Fallback optimization failed: {e}")

        # Return empty result if fallback also fails
        return self._create_empty_result(sheet, algorithm.name, start_time)


class PerformanceMonitor:
    """Monitor optimization performance and resource usage"""

    def __init__(self):
        self.start_time = None
        self.monitoring = False
        self.metrics_history = []
        self.peak_memory = 0.0
        self.logger = logging.getLogger(f"{__name__}.PerformanceMonitor")

    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.monitoring = True
        self.peak_memory = 0.0
        self.logger.debug("Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring and log summary"""
        if self.monitoring and self.start_time:
            duration = time.time() - self.start_time
            current_memory = self.get_memory_usage()

            self.logger.info(
                f"Performance summary: duration={duration:.3f}s, "
                f"peak_memory={self.peak_memory:.1f}MB, "
                f"final_memory={current_memory:.1f}MB"
            )

            # Store metrics for analysis
            self.metrics_history.append({
                'duration': duration,
                'peak_memory': self.peak_memory,
                'final_memory': current_memory,
                'timestamp': time.time()
            })

        self.monitoring = False

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)

            # Track peak memory usage
            if self.monitoring and memory_mb > self.peak_memory:
                self.peak_memory = memory_mb

            return memory_mb
        except ImportError:
            return 0.0
        except Exception as e:
            self.logger.warning(f"Failed to get memory usage: {e}")
            return 0.0

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            return 0.0
        except Exception:
            return 0.0

    def check_resource_limits(
        self,
        memory_limit_mb: float = 512,
        cpu_limit_percent: float = 90
    ) -> bool:
        """Check if resource usage is within limits"""
        if not self.monitoring:
            return True

        memory_usage = self.get_memory_usage()
        cpu_usage = self.get_cpu_usage()

        memory_ok = memory_usage < memory_limit_mb
        cpu_ok = cpu_usage < cpu_limit_percent

        if not memory_ok:
            self.logger.warning(
                f"Memory usage {memory_usage:.1f}MB exceeds limit {memory_limit_mb}MB"
            )

        if not cpu_ok:
            self.logger.warning(
                f"CPU usage {cpu_usage:.1f}% exceeds limit {cpu_limit_percent}%"
            )

        return memory_ok and cpu_ok

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time if self.start_time else 0

        return {
            'elapsed_time': elapsed_time,
            'memory_usage': self.get_memory_usage(),
            'cpu_usage': self.get_cpu_usage(),
            'peak_memory': self.peak_memory
        }

    def get_average_performance(self) -> Dict[str, float]:
        """Get average performance metrics from history"""
        if not self.metrics_history:
            return {'avg_duration': 0, 'avg_peak_memory': 0, 'avg_final_memory': 0}

        return {
            'avg_duration': sum(m['duration'] for m in self.metrics_history) / len(self.metrics_history),
            'avg_peak_memory': sum(m['peak_memory'] for m in self.metrics_history) / len(self.metrics_history),
            'avg_final_memory': sum(m['final_memory'] for m in self.metrics_history) / len(self.metrics_history),
            'total_runs': len(self.metrics_history)
        }


class TimeoutManager:
    """Manage algorithm timeouts and recovery with enhanced monitoring"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.TimeoutManager")
        self.active_threads = set()
        self.timeout_history = []

    def execute_with_timeout(self, func, timeout: float, check_interval: float = 0.1):
        """Execute function with timeout and progress monitoring"""
        result = [None]
        exception = [None]
        progress = {'last_check': time.time()}

        def target():
            try:
                result[0] = func()
            except Exception as e:
                exception[0] = e
            finally:
                progress['completed'] = True

        # Create and start thread
        thread = threading.Thread(target=target, daemon=True)
        thread_id = id(thread)
        self.active_threads.add(thread_id)

        start_time = time.time()
        thread.start()

        try:
            # Monitor execution with periodic checks
            elapsed = 0
            while thread.is_alive() and elapsed < timeout:
                thread.join(timeout=check_interval)
                elapsed = time.time() - start_time

                # Log progress periodically for long operations
                if elapsed > 5.0 and (elapsed % 5.0) < check_interval:
                    self.logger.debug(f"Function running for {elapsed:.1f}s (timeout: {timeout:.1f}s)")

            if thread.is_alive():
                # Thread is still running, timeout occurred
                execution_time = time.time() - start_time
                self.logger.warning(
                    f"Function timed out after {execution_time:.3f}s (limit: {timeout:.1f}s)"
                )

                # Record timeout for analysis
                self.timeout_history.append({
                    'timeout_limit': timeout,
                    'actual_time': execution_time,
                    'timestamp': time.time()
                })

                # Note: Cannot actually kill thread in Python safely
                # Thread will continue running in background but results are ignored
                raise FutureTimeoutError(f"Operation timed out after {timeout:.1f} seconds")

        finally:
            # Clean up thread tracking
            self.active_threads.discard(thread_id)

        # Check for exceptions
        if exception[0]:
            raise exception[0]

        return result[0]

    def execute_with_progressive_timeout(
        self,
        func,
        initial_timeout: float,
        max_timeout: float,
        timeout_multiplier: float = 2.0
    ):
        """Execute with progressively increasing timeout on failure"""
        current_timeout = initial_timeout
        attempt = 1
        max_attempts = 3

        while attempt <= max_attempts and current_timeout <= max_timeout:
            try:
                self.logger.debug(
                    f"Attempt {attempt}: timeout={current_timeout:.1f}s"
                )
                return self.execute_with_timeout(func, current_timeout)

            except FutureTimeoutError:
                if attempt == max_attempts or current_timeout >= max_timeout:
                    self.logger.error(
                        f"All {max_attempts} attempts failed, max timeout {max_timeout:.1f}s reached"
                    )
                    raise

                # Increase timeout for next attempt
                current_timeout = min(current_timeout * timeout_multiplier, max_timeout)
                attempt += 1
                self.logger.info(
                    f"Timeout occurred, retrying with {current_timeout:.1f}s timeout"
                )

        raise FutureTimeoutError(f"Failed after {max_attempts} attempts")

    def get_timeout_statistics(self) -> Dict[str, float]:
        """Get timeout statistics for performance analysis"""
        if not self.timeout_history:
            return {
                'total_timeouts': 0,
                'avg_timeout_ratio': 0.0,
                'recent_timeouts': 0
            }

        total_operations = len(self.timeout_history)
        avg_timeout_ratio = sum(
            min(1.0, h['actual_time'] / h['timeout_limit']) for h in self.timeout_history
        ) / total_operations

        return {
            'total_timeouts': total_operations,
            'avg_timeout_ratio': avg_timeout_ratio,
            'recent_timeouts': len([h for h in self.timeout_history if time.time() - h['timestamp'] < 3600])
        }

    def get_active_threads_count(self) -> int:
        """Get number of currently active threads"""
        return len(self.active_threads)

    def clear_timeout_history(self):
        """Clear timeout history (useful for testing or reset)"""
        self.timeout_history.clear()
        self.logger.debug("Timeout history cleared")


# Factory function for creating optimization engine
def create_optimization_engine() -> OptimizationEngine:
    """
    Create and configure optimization engine with default algorithms
    デフォルトアルゴリズム付き最適化エンジンの作成
    """
    engine = OptimizationEngine()

    # Register available algorithms
    try:
        from core.algorithms.ffd import create_ffd_algorithm
        ffd_algorithm = create_ffd_algorithm()
        engine.register_algorithm(ffd_algorithm)

        # Register additional algorithms when available
        # This allows graceful degradation if some algorithms aren't implemented yet

        try:
            from core.algorithms.bfd import create_bfd_algorithm
            bfd_algorithm = create_bfd_algorithm()
            engine.register_algorithm(bfd_algorithm)
        except ImportError:
            engine.logger.info("BFD algorithm not available (not implemented yet)")

        try:
            from core.algorithms.hybrid import create_hybrid_algorithm
            hybrid_algorithm = create_hybrid_algorithm()
            engine.register_algorithm(hybrid_algorithm)
        except ImportError:
            engine.logger.info("Hybrid algorithm not available (not implemented yet)")

        try:
            from core.algorithms.genetic import create_genetic_algorithm
            genetic_algorithm = create_genetic_algorithm()
            engine.register_algorithm(genetic_algorithm)
        except ImportError:
            engine.logger.info("Genetic algorithm not available (not implemented yet)")

    except ImportError as e:
        engine.logger.error(f"Failed to import algorithms: {e}")
        raise RuntimeError("No optimization algorithms could be loaded")

    engine.logger.info(f"Optimization engine created with {len(engine.algorithms)} algorithms")
    return engine


def create_optimization_engine_with_algorithms(algorithm_names: List[str]) -> OptimizationEngine:
    """
    Create optimization engine with specific algorithms only
    特定アルゴリズムのみで最適化エンジンを作成
    """
    engine = OptimizationEngine()

    algorithm_factories = {
        'FFD': lambda: __import__('core.algorithms.ffd', fromlist=['create_ffd_algorithm']).create_ffd_algorithm(),
        'BFD': lambda: __import__('core.algorithms.bfd', fromlist=['create_bfd_algorithm']).create_bfd_algorithm(),
        'HYBRID': lambda: __import__('core.algorithms.hybrid', fromlist=['create_hybrid_algorithm']).create_hybrid_algorithm(),
    }

    registered_count = 0
    for name in algorithm_names:
        if name in algorithm_factories:
            try:
                algorithm = algorithm_factories[name]()
                engine.register_algorithm(algorithm)
                registered_count += 1
            except ImportError:
                engine.logger.warning(f"Algorithm {name} not available")
            except Exception as e:
                engine.logger.error(f"Failed to create algorithm {name}: {e}")
        else:
            engine.logger.warning(f"Unknown algorithm: {name}")

    if registered_count == 0:
        raise RuntimeError(f"No algorithms could be registered from {algorithm_names}")

    engine.logger.info(f"Created engine with {registered_count} specific algorithms")
    return engine