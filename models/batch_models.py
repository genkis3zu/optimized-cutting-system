#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch processing models for material-based optimization
材料別最適化のバッチ処理モデル
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

from .core_models import Panel, SteelSheet, PlacementResult, OptimizationType


class BatchStatus(Enum):
    """
    Status of batch processing
    バッチ処理の状態
    """
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    LOW_EFFICIENCY = "low_efficiency"  # Marked for residual processing


@dataclass
class MaterialBatch:
    """
    A batch of panels of the same material type for individual optimization
    個別最適化用の同一材料パネルのバッチ
    """
    material_type: str
    thickness: float
    panels: List[Panel] = field(default_factory=list)
    sheet_template: SteelSheet = field(default_factory=lambda: SteelSheet())
    optimization_type: OptimizationType = OptimizationType.SIMPLE_MATH
    status: BatchStatus = BatchStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate batch consistency"""
        if self.panels:
            # Ensure all panels have the same material and thickness
            for panel in self.panels:
                if panel.material != self.material_type:
                    raise ValueError(f"Panel material {panel.material} doesn't match batch material {self.material_type}")
                if panel.thickness != self.thickness:
                    raise ValueError(f"Panel thickness {panel.thickness} doesn't match batch thickness {self.thickness}")

    @property
    def total_panels(self) -> int:
        """Total number of individual panels in batch"""
        return sum(panel.quantity for panel in self.panels)

    @property
    def total_area(self) -> float:
        """Total cutting area of all panels in batch"""
        return sum(panel.cutting_area * panel.quantity for panel in self.panels)

    @property
    def unique_sizes(self) -> int:
        """Number of unique panel sizes in batch"""
        return len(set((panel.cutting_width, panel.cutting_height) for panel in self.panels))

    def get_individual_panels(self) -> List[Panel]:
        """
        Expand quantity-based panels into individual panels
        数量ベースのパネルを個別パネルに展開
        """
        individual_panels = []
        for panel in self.panels:
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
        return individual_panels

    def determine_optimization_type(self) -> OptimizationType:
        """
        Determine the most appropriate optimization type for this batch
        このバッチに最適な最適化タイプを判定
        """
        if self.unique_sizes == 1:
            # Single size - check if simple math applies
            panel = self.panels[0]
            cutting_w = panel.cutting_width
            cutting_h = panel.cutting_height
            sheet_w = self.sheet_template.width
            sheet_h = self.sheet_template.height

            # Case 1: Panel width matches sheet width exactly
            if abs(cutting_w - sheet_w) < 1.0:
                return OptimizationType.SIMPLE_MATH

            # Case 2: Two panels fit exactly in width
            if abs(cutting_w * 2 - sheet_w) < 1.0:
                return OptimizationType.SIMPLE_MATH

            # Case 3: Three panels fit exactly in width
            if abs(cutting_w * 3 - sheet_w) < 1.0:
                return OptimizationType.SIMPLE_MATH

            # Case 4: Simple rectangular grid (any integer grid that fits)
            panels_per_width = int(sheet_w // cutting_w)
            panels_per_height = int(sheet_h // cutting_h)
            if (panels_per_width >= 1 and panels_per_height >= 1 and
                panels_per_width * cutting_w <= sheet_w and
                panels_per_height * cutting_h <= sheet_h):
                return OptimizationType.SIMPLE_MATH

        # Complex cases require algorithm
        return OptimizationType.COMPLEX_ALGORITHM


@dataclass
class ProcessingResult:
    """
    Result of processing a single material batch
    単一材料バッチの処理結果
    """
    batch: MaterialBatch
    placement_results: List[PlacementResult] = field(default_factory=list)
    efficiency: float = 0.0
    total_sheets_used: int = 0
    unplaced_panels: List[Panel] = field(default_factory=list)
    processing_time: float = 0.0
    algorithm_used: str = ""
    status: BatchStatus = BatchStatus.COMPLETED

    def __post_init__(self):
        """Calculate aggregate metrics"""
        if self.placement_results:
            total_placed_area = sum(
                sum(panel.actual_width * panel.actual_height for panel in result.panels)
                for result in self.placement_results
            )
            total_sheet_area = sum(result.sheet.area for result in self.placement_results)
            self.efficiency = total_placed_area / total_sheet_area if total_sheet_area > 0 else 0.0
            self.total_sheets_used = len(self.placement_results)

    @property
    def total_cost(self) -> float:
        """Total cost of sheets used"""
        return sum(result.cost for result in self.placement_results)

    @property
    def total_waste_area(self) -> float:
        """Total waste area across all sheets"""
        return sum(result.waste_area for result in self.placement_results)

    @property
    def is_low_efficiency(self) -> bool:
        """Check if this result should be considered for residual processing"""
        return self.efficiency < 0.7 or len(self.unplaced_panels) > 0


@dataclass
class BatchOptimizationResult:
    """
    Complete result of batch optimization process
    バッチ最適化プロセスの完全結果
    """
    processing_results: List[ProcessingResult] = field(default_factory=list)
    high_efficiency_results: List[ProcessingResult] = field(default_factory=list)
    low_efficiency_results: List[ProcessingResult] = field(default_factory=list)
    overall_efficiency: float = 0.0
    total_processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def classify_results(self, efficiency_threshold: float = 0.7):
        """
        Classify results into high and low efficiency categories
        効率の高低で結果を分類
        """
        self.high_efficiency_results.clear()
        self.low_efficiency_results.clear()

        for result in self.processing_results:
            if result.efficiency >= efficiency_threshold and not result.unplaced_panels:
                self.high_efficiency_results.append(result)
            else:
                self.low_efficiency_results.append(result)

        # Calculate overall efficiency
        if self.processing_results:
            total_placed_area = sum(
                sum(
                    sum(panel.actual_width * panel.actual_height for panel in placement.panels)
                    for placement in result.placement_results
                )
                for result in self.processing_results
            )
            total_sheet_area = sum(
                sum(placement.sheet.area for placement in result.placement_results)
                for result in self.processing_results
            )
            self.overall_efficiency = total_placed_area / total_sheet_area if total_sheet_area > 0 else 0.0

    def get_residual_materials(self) -> List['ResidualMaterial']:
        """
        Extract residual materials from low efficiency results
        低効率結果から余り材を抽出
        """
        from .residual_models import ResidualMaterial  # Avoid circular import

        residual_materials = []
        for result in self.low_efficiency_results:
            if result.unplaced_panels or result.efficiency < 0.7:
                residual_materials.append(ResidualMaterial.from_processing_result(result))

        return residual_materials