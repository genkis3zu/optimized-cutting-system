#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Material batch optimizer with simple math and algorithm selection
単純計算とアルゴリズム選択による材料バッチ最適化
"""

import logging
import time
import math
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from models.core_models import Panel, SteelSheet, PlacedPanel, PlacementResult, OptimizationType
from models.batch_models import MaterialBatch, ProcessingResult, BatchStatus
from models.optimization_models import OptimizationStrategy, PlacementMetrics


class SimpleMathCalculator:
    """
    Simple mathematical calculator for straightforward panel layouts
    単純なパネル配置用の数学計算機
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_single_size_placement(
        self,
        panel: Panel,
        sheet: SteelSheet,
        kerf_width: float = 3.0
    ) -> Optional[PlacementResult]:
        """
        Calculate placement for single size panels using mathematical approach
        単一サイズパネルの配置を数学的アプローチで計算

        Example: 968x712 panel on 1500x3100 sheet
        例: 1500x3100シートに968x712パネル
        """
        cutting_w = panel.cutting_width
        cutting_h = panel.cutting_height
        sheet_w = sheet.width
        sheet_h = sheet.height

        self.logger.info(f"Calculating placement for {cutting_w}x{cutting_h} on {sheet_w}x{sheet_h}")

        # Case 1: Panel width matches sheet width exactly
        if abs(cutting_w - sheet_w) <= 1.0:
            return self._calculate_vertical_stack(panel, sheet, kerf_width)

        # Case 2: Multiple panels fit in width (including single column)
        panels_per_width = int((sheet_w + kerf_width) // (cutting_w + kerf_width))
        if panels_per_width >= 1:
            return self._calculate_grid_placement(panel, sheet, kerf_width, panels_per_width)

        # Case 3: Check if rotation helps
        if panel.allow_rotation:
            if abs(cutting_h - sheet_w) <= 1.0:
                rotated_panel = Panel(
                    id=panel.id + "_rot",
                    width=panel.height,
                    height=panel.width,
                    quantity=panel.quantity,
                    material=panel.material,
                    thickness=panel.thickness,
                    allow_rotation=panel.allow_rotation,
                    expanded_width=panel.expanded_height,
                    expanded_height=panel.expanded_width
                )
                return self._calculate_vertical_stack(rotated_panel, sheet, kerf_width, rotated=True)

        return None

    def _calculate_vertical_stack(
        self,
        panel: Panel,
        sheet: SteelSheet,
        kerf_width: float,
        rotated: bool = False
    ) -> PlacementResult:
        """
        Calculate vertical stacking (panel width = sheet width)
        垂直積み重ね計算（パネル幅 = シート幅）
        """
        cutting_h = panel.cutting_height
        sheet_h = sheet.height

        # Calculate how many panels fit vertically
        panels_per_height = int((sheet_h + kerf_width) // (cutting_h + kerf_width))
        max_panels = min(panels_per_height, panel.quantity)

        # Create placed panels
        placed_panels = []
        for i in range(max_panels):
            y_position = i * (cutting_h + kerf_width)
            placed_panel = PlacedPanel(
                panel=Panel(
                    id=f"{panel.id}_{i+1}",
                    width=panel.width,
                    height=panel.height,
                    quantity=1,
                    material=panel.material,
                    thickness=panel.thickness,
                    expanded_width=panel.expanded_width,
                    expanded_height=panel.expanded_height
                ),
                x=0.0,
                y=y_position,
                rotated=rotated
            )
            placed_panels.append(placed_panel)

        # Calculate efficiency
        used_area = max_panels * panel.cutting_area
        efficiency = used_area / sheet.area

        self.logger.info(f"Vertical stack: {max_panels} panels, efficiency: {efficiency:.1%}")

        return PlacementResult(
            sheet_id=1,
            material_block=panel.material,
            sheet=sheet,
            panels=placed_panels,
            efficiency=efficiency,
            algorithm="SimpleMath_VerticalStack",
            processing_time=0.001  # Nearly instantaneous
        )

    def _calculate_grid_placement(
        self,
        panel: Panel,
        sheet: SteelSheet,
        kerf_width: float,
        panels_per_width: int
    ) -> PlacementResult:
        """
        Calculate grid placement for multiple panels
        複数パネルのグリッド配置計算
        """
        cutting_w = panel.cutting_width
        cutting_h = panel.cutting_height
        sheet_h = sheet.height

        # Calculate how many rows fit
        panels_per_height = int((sheet_h + kerf_width) // (cutting_h + kerf_width))
        max_panels_per_sheet = panels_per_width * panels_per_height
        max_panels = min(max_panels_per_sheet, panel.quantity)

        # Create placed panels in grid pattern
        placed_panels = []
        panel_count = 0

        for row in range(panels_per_height):
            if panel_count >= max_panels:
                break
            for col in range(panels_per_width):
                if panel_count >= max_panels:
                    break

                x_position = col * (cutting_w + kerf_width)
                y_position = row * (cutting_h + kerf_width)

                placed_panel = PlacedPanel(
                    panel=Panel(
                        id=f"{panel.id}_{panel_count+1}",
                        width=panel.width,
                        height=panel.height,
                        quantity=1,
                        material=panel.material,
                        thickness=panel.thickness,
                        expanded_width=panel.expanded_width,
                        expanded_height=panel.expanded_height
                    ),
                    x=x_position,
                    y=y_position,
                    rotated=False
                )
                placed_panels.append(placed_panel)
                panel_count += 1

        # Calculate efficiency
        used_area = max_panels * panel.cutting_area
        efficiency = used_area / sheet.area

        self.logger.info(f"Grid placement: {max_panels} panels ({panels_per_width}x{panels_per_height}), efficiency: {efficiency:.1%}")

        return PlacementResult(
            sheet_id=1,
            material_block=panel.material,
            sheet=sheet,
            panels=placed_panels,
            efficiency=efficiency,
            algorithm="SimpleMath_Grid",
            processing_time=0.002  # Nearly instantaneous
        )


class ComplexAlgorithmOptimizer:
    """
    Complex algorithm optimizer for cases where simple math is insufficient
    単純計算では不十分な場合の複雑アルゴリズム最適化
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def optimize_complex_batch(
        self,
        batch: MaterialBatch,
        kerf_width: float = 3.0
    ) -> ProcessingResult:
        """
        Optimize batch using more complex algorithms
        より複雑なアルゴリズムを使用してバッチを最適化
        """
        # For now, implement a simple first-fit decreasing approach
        # This can be expanded with more sophisticated algorithms later

        individual_panels = batch.get_individual_panels()
        individual_panels.sort(key=lambda p: p.cutting_area, reverse=True)

        placement_results = []
        unplaced_panels = individual_panels.copy()
        sheet_id = 1

        while unplaced_panels:
            result = self._pack_single_sheet(unplaced_panels, batch.sheet_template, kerf_width, sheet_id)
            if not result.panels:  # No panels could be placed
                break

            placement_results.append(result)

            # Remove placed panels from unplaced list
            placed_panel_ids = {panel.panel.id for panel in result.panels}
            unplaced_panels = [panel for panel in unplaced_panels if panel.id not in placed_panel_ids]
            sheet_id += 1

        # Calculate overall efficiency
        if placement_results:
            total_placed_area = sum(
                sum(panel.actual_width * panel.actual_height for panel in result.panels)
                for result in placement_results
            )
            total_sheet_area = sum(result.sheet.area for result in placement_results)
            efficiency = total_placed_area / total_sheet_area
        else:
            efficiency = 0.0

        return ProcessingResult(
            batch=batch,
            placement_results=placement_results,
            efficiency=efficiency,
            unplaced_panels=unplaced_panels,
            algorithm_used="ComplexAlgorithm_FirstFit",
            processing_time=0.1  # Placeholder
        )

    def _pack_single_sheet(
        self,
        panels: List[Panel],
        sheet: SteelSheet,
        kerf_width: float,
        sheet_id: int
    ) -> PlacementResult:
        """
        Pack panels into a single sheet using simple first-fit
        単一シートにパネルを単純first-fitで配置
        """
        placed_panels = []
        current_x = 0.0
        current_y = 0.0
        row_height = 0.0

        for panel in panels:
            # Check if panel fits in current position
            cutting_w = panel.cutting_width
            cutting_h = panel.cutting_height

            # Try normal orientation
            if self._fits_at_position(current_x, current_y, cutting_w, cutting_h, sheet, kerf_width):
                placed_panel = PlacedPanel(
                    panel=panel,
                    x=current_x,
                    y=current_y,
                    rotated=False
                )
                placed_panels.append(placed_panel)
                current_x += cutting_w + kerf_width
                row_height = max(row_height, cutting_h)
                continue

            # Try rotated orientation if allowed
            if panel.allow_rotation and self._fits_at_position(current_x, current_y, cutting_h, cutting_w, sheet, kerf_width):
                placed_panel = PlacedPanel(
                    panel=panel,
                    x=current_x,
                    y=current_y,
                    rotated=True
                )
                placed_panels.append(placed_panel)
                current_x += cutting_h + kerf_width
                row_height = max(row_height, cutting_w)
                continue

            # Move to next row
            current_x = 0.0
            current_y += row_height + kerf_width
            row_height = 0.0

            # Try again in new row
            if self._fits_at_position(current_x, current_y, cutting_w, cutting_h, sheet, kerf_width):
                placed_panel = PlacedPanel(
                    panel=panel,
                    x=current_x,
                    y=current_y,
                    rotated=False
                )
                placed_panels.append(placed_panel)
                current_x += cutting_w + kerf_width
                row_height = cutting_h

        # Calculate efficiency
        if placed_panels:
            used_area = sum(panel.actual_width * panel.actual_height for panel in placed_panels)
            efficiency = used_area / sheet.area
        else:
            efficiency = 0.0

        return PlacementResult(
            sheet_id=sheet_id,
            material_block=sheet.material,
            sheet=sheet,
            panels=placed_panels,
            efficiency=efficiency,
            algorithm="ComplexAlgorithm_FirstFit"
        )

    def _fits_at_position(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        sheet: SteelSheet,
        kerf_width: float
    ) -> bool:
        """Check if panel fits at given position"""
        return (x + width <= sheet.width and
                y + height <= sheet.height)


class MaterialBatchOptimizer:
    """
    Main optimizer for material batches
    材料バッチのメイン最適化器
    """

    def __init__(self):
        self.simple_calculator = SimpleMathCalculator()
        self.complex_optimizer = ComplexAlgorithmOptimizer()
        self.logger = logging.getLogger(__name__)

    def optimize_batch(
        self,
        batch: MaterialBatch,
        kerf_width: float = 3.0
    ) -> ProcessingResult:
        """
        Optimize a material batch using appropriate strategy
        適切な戦略を使用して材料バッチを最適化
        """
        start_time = time.time()

        # Determine optimization type
        optimization_type = batch.determine_optimization_type()
        self.logger.info(f"Optimizing batch {batch.material_type} with {optimization_type.value}")

        if optimization_type == OptimizationType.SIMPLE_MATH:
            result = self._optimize_with_simple_math(batch, kerf_width)
        else:
            result = self.complex_optimizer.optimize_complex_batch(batch, kerf_width)

        result.processing_time = time.time() - start_time

        # Update batch status
        if result.efficiency >= 0.7 and not result.unplaced_panels:
            batch.status = BatchStatus.COMPLETED
        elif result.efficiency < 0.7 or result.unplaced_panels:
            batch.status = BatchStatus.LOW_EFFICIENCY
        else:
            batch.status = BatchStatus.FAILED

        return result

    def _optimize_with_simple_math(
        self,
        batch: MaterialBatch,
        kerf_width: float
    ) -> ProcessingResult:
        """
        Optimize batch using simple mathematical calculations
        単純数学計算を使用してバッチを最適化
        """
        if len(batch.panels) != 1:
            raise ValueError("Simple math optimization requires single panel type")

        panel = batch.panels[0]
        placement_results = []
        remaining_quantity = panel.quantity
        sheet_id = 1

        while remaining_quantity > 0:
            # Create panel with remaining quantity for calculation
            current_panel = Panel(
                id=panel.id,
                width=panel.width,
                height=panel.height,
                quantity=remaining_quantity,
                material=panel.material,
                thickness=panel.thickness,
                allow_rotation=panel.allow_rotation,
                expanded_width=panel.expanded_width,
                expanded_height=panel.expanded_height
            )

            result = self.simple_calculator.calculate_single_size_placement(
                current_panel, batch.sheet_template, kerf_width
            )

            if not result:
                break  # Cannot place any more panels

            result.sheet_id = sheet_id
            placement_results.append(result)

            placed_count = len(result.panels)
            remaining_quantity -= placed_count
            sheet_id += 1

        # Calculate overall metrics
        if placement_results:
            total_placed_area = sum(
                sum(panel.actual_width * panel.actual_height for panel in result.panels)
                for result in placement_results
            )
            total_sheet_area = sum(result.sheet.area for result in placement_results)
            efficiency = total_placed_area / total_sheet_area
            total_placed = sum(len(result.panels) for result in placement_results)
        else:
            efficiency = 0.0
            total_placed = 0

        # Create unplaced panels if any remain
        unplaced_panels = []
        if remaining_quantity > 0:
            unplaced_panels.append(Panel(
                id=f"{panel.id}_unplaced",
                width=panel.width,
                height=panel.height,
                quantity=remaining_quantity,
                material=panel.material,
                thickness=panel.thickness,
                expanded_width=panel.expanded_width,
                expanded_height=panel.expanded_height
            ))

        return ProcessingResult(
            batch=batch,
            placement_results=placement_results,
            efficiency=efficiency,
            unplaced_panels=unplaced_panels,
            algorithm_used="SimpleMath",
            processing_time=0.01  # Very fast
        )