#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Residual material models for handling low-efficiency batches
低効率バッチ処理用の余り材モデル
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from enum import Enum

from .core_models import Panel, SteelSheet, PlacementResult
from .batch_models import ProcessingResult


class ResidualType(Enum):
    """
    Type of residual material
    余り材のタイプ
    """
    UNPLACED_PANELS = "unplaced_panels"  # Panels that couldn't be placed
    LOW_EFFICIENCY_BATCH = "low_efficiency_batch"  # Batch with low placement efficiency
    PARTIAL_SHEET = "partial_sheet"  # Sheet with remaining space


class CombinationStrategy(Enum):
    """
    Strategy for combining residual materials
    余り材の組み合わせ戦略
    """
    SUPPLEMENT_SHORTAGE = "supplement_shortage"  # Add materials to complete placement
    UTILIZE_REMAINDER = "utilize_remainder"  # Use remaining space for other panels
    REPACK_ENTIRELY = "repack_entirely"  # Completely repack all panels


@dataclass
class ResidualMaterial:
    """
    Represents residual material from low-efficiency batch processing
    低効率バッチ処理からの余り材を表す
    """
    material_type: str
    thickness: float
    unplaced_panels: List[Panel] = field(default_factory=list)
    partial_placements: List[PlacementResult] = field(default_factory=list)
    residual_type: ResidualType = ResidualType.UNPLACED_PANELS
    original_efficiency: float = 0.0
    available_sheet_space: Dict[str, float] = field(default_factory=dict)  # Remaining space info
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_processing_result(cls, processing_result: ProcessingResult) -> 'ResidualMaterial':
        """
        Create ResidualMaterial from ProcessingResult
        ProcessingResultから余り材を作成
        """
        # Calculate available space from partial placements
        available_space = {}
        for i, placement in enumerate(processing_result.placement_results):
            used_area = sum(panel.actual_width * panel.actual_height for panel in placement.panels)
            remaining_area = placement.sheet.area - used_area
            available_space[f"sheet_{i}"] = remaining_area

        # Determine residual type
        if processing_result.unplaced_panels:
            residual_type = ResidualType.UNPLACED_PANELS
        elif processing_result.efficiency < 0.7:
            residual_type = ResidualType.LOW_EFFICIENCY_BATCH
        else:
            residual_type = ResidualType.PARTIAL_SHEET

        return cls(
            material_type=processing_result.batch.material_type,
            thickness=processing_result.batch.thickness,
            unplaced_panels=processing_result.unplaced_panels.copy(),
            partial_placements=processing_result.placement_results.copy(),
            residual_type=residual_type,
            original_efficiency=processing_result.efficiency,
            available_sheet_space=available_space
        )

    @property
    def total_unplaced_area(self) -> float:
        """Total area of unplaced panels"""
        return sum(panel.cutting_area * panel.quantity for panel in self.unplaced_panels)

    @property
    def total_available_space(self) -> float:
        """Total available space in partial sheets"""
        return sum(self.available_sheet_space.values())

    @property
    def can_be_optimized(self) -> bool:
        """Check if this residual material is worth optimizing"""
        return (len(self.unplaced_panels) > 0 or
                self.original_efficiency < 0.6 or
                self.total_available_space > 1000000)  # 1m² threshold

    def get_optimization_priority(self) -> float:
        """
        Calculate optimization priority (0-1, higher is more urgent)
        最適化優先度を計算（0-1、高いほど緊急）
        """
        unplaced_factor = len(self.unplaced_panels) / 10.0  # Normalize by typical batch size
        efficiency_factor = 1.0 - self.original_efficiency  # Lower efficiency = higher priority
        area_factor = min(1.0, self.total_unplaced_area / 5000000.0)  # 5m² normalization

        priority = (unplaced_factor * 0.4 +
                   efficiency_factor * 0.4 +
                   area_factor * 0.2)

        return min(1.0, priority)


@dataclass
class CombinationCandidate:
    """
    A candidate combination of residual materials
    余り材の組み合わせ候補
    """
    residual_materials: List[ResidualMaterial] = field(default_factory=list)
    strategy: CombinationStrategy = CombinationStrategy.UTILIZE_REMAINDER
    estimated_efficiency: float = 0.0
    estimated_sheets_needed: int = 0
    combination_score: float = 0.0  # Overall desirability score

    def __post_init__(self):
        """Calculate combination metrics"""
        self.calculate_combination_score()

    def calculate_combination_score(self):
        """
        Calculate desirability score for this combination
        この組み合わせの望ましさスコアを計算
        """
        if not self.residual_materials:
            self.combination_score = 0.0
            return

        # Factors for scoring
        material_consistency = 1.0  # All same material (enforced)
        total_area = sum(rm.total_unplaced_area for rm in self.residual_materials)
        area_efficiency = min(1.0, total_area / 4650000.0)  # Standard sheet area
        priority_avg = sum(rm.get_optimization_priority() for rm in self.residual_materials) / len(self.residual_materials)

        self.combination_score = (
            material_consistency * 0.3 +
            area_efficiency * 0.4 +
            priority_avg * 0.3
        )

    @property
    def total_panels(self) -> int:
        """Total number of unplaced panels in combination"""
        return sum(len(rm.unplaced_panels) for rm in self.residual_materials)

    @property
    def material_type(self) -> str:
        """Material type of this combination (should be consistent)"""
        return self.residual_materials[0].material_type if self.residual_materials else ""

    def get_all_unplaced_panels(self) -> List[Panel]:
        """Get all unplaced panels from all residual materials"""
        all_panels = []
        for rm in self.residual_materials:
            all_panels.extend(rm.unplaced_panels)
        return all_panels


@dataclass
class CombinationResult:
    """
    Result of optimizing a combination of residual materials
    余り材組み合わせの最適化結果
    """
    combination_candidate: CombinationCandidate
    placement_results: List[PlacementResult] = field(default_factory=list)
    final_efficiency: float = 0.0
    improvement_ratio: float = 0.0  # Improvement over original efficiency
    still_unplaced: List[Panel] = field(default_factory=list)
    processing_time: float = 0.0
    success: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Calculate result metrics"""
        if self.placement_results:
            total_placed_area = sum(
                sum(panel.actual_width * panel.actual_height for panel in result.panels)
                for result in self.placement_results
            )
            total_sheet_area = sum(result.sheet.area for result in self.placement_results)
            self.final_efficiency = total_placed_area / total_sheet_area if total_sheet_area > 0 else 0.0

            # Calculate improvement
            original_efficiency = sum(
                rm.original_efficiency for rm in self.combination_candidate.residual_materials
            ) / len(self.combination_candidate.residual_materials)

            self.improvement_ratio = (self.final_efficiency - original_efficiency) / original_efficiency if original_efficiency > 0 else 0.0
            self.success = self.final_efficiency > 0.7 and len(self.still_unplaced) == 0

    @property
    def total_cost(self) -> float:
        """Total cost of sheets used in optimization"""
        return sum(result.cost for result in self.placement_results)

    @property
    def total_sheets_used(self) -> int:
        """Total number of sheets used"""
        return len(self.placement_results)

    def is_improvement(self, threshold: float = 0.05) -> bool:
        """Check if this represents a meaningful improvement"""
        return self.improvement_ratio > threshold and self.success