#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization strategy and metrics models
最適化戦略とメトリクスのモデル
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime
from enum import Enum

from .core_models import Panel, SteelSheet, OptimizationType


class OptimizationStrategy(Enum):
    """
    Optimization strategy selection
    最適化戦略の選択
    """
    SINGLE_SIZE_MATH = "single_size_math"  # Mathematical calculation for single size
    MULTI_SIZE_SIMPLE = "multi_size_simple"  # Simple algorithm for multiple sizes
    COMPLEX_PACKING = "complex_packing"  # Complex bin packing algorithm


@dataclass
class PlacementMetrics:
    """
    Metrics for evaluating placement quality
    配置品質評価のメトリクス
    """
    efficiency: float = 0.0  # Material utilization ratio (0-1)
    waste_area: float = 0.0  # Wasted area in mm²
    cut_length: float = 0.0  # Total cutting length in mm
    sheet_count: int = 0  # Number of sheets used
    processing_time: float = 0.0  # Processing time in seconds
    algorithm_complexity: float = 0.0  # Algorithm complexity score (0-1)

    @property
    def waste_percentage(self) -> float:
        """Waste as percentage of total area"""
        total_area = self.waste_area / (1 - self.efficiency) if self.efficiency < 1.0 else 0.0
        return (self.waste_area / total_area * 100) if total_area > 0 else 0.0

    @property
    def cost_efficiency_score(self) -> float:
        """Combined score considering efficiency and processing cost"""
        time_penalty = min(1.0, self.processing_time / 60.0)  # Normalize by 1 minute
        return self.efficiency * (1 - time_penalty * 0.1)  # 10% time penalty weight

    def is_acceptable(self, min_efficiency: float = 0.6, max_time: float = 30.0) -> bool:
        """Check if metrics meet acceptability criteria"""
        return self.efficiency >= min_efficiency and self.processing_time <= max_time


@dataclass
class OptimizationTask:
    """
    A task for optimization processing
    最適化処理のタスク
    """
    task_id: str
    panels: List[Panel] = field(default_factory=list)
    sheet_template: SteelSheet = field(default_factory=lambda: SteelSheet())
    strategy: OptimizationStrategy = OptimizationStrategy.SINGLE_SIZE_MATH
    priority: float = 1.0  # Task priority (0-1)
    constraints: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    # Callback functions for progress reporting
    progress_callback: Optional[Callable[[float], None]] = None
    completion_callback: Optional[Callable[['OptimizationResult'], None]] = None

    def estimate_complexity(self) -> float:
        """
        Estimate task complexity (0-1)
        タスクの複雑度を推定（0-1）
        """
        if not self.panels:
            return 0.0

        # Factors affecting complexity
        panel_count = len(self.panels)
        unique_sizes = len(set((p.cutting_width, p.cutting_height) for p in self.panels))
        total_quantity = sum(p.quantity for p in self.panels)
        rotation_allowed = sum(1 for p in self.panels if p.allow_rotation)

        # Normalized factors
        count_factor = min(1.0, panel_count / 50.0)  # Normalize by 50 panels
        size_diversity = unique_sizes / panel_count if panel_count > 0 else 0.0
        quantity_factor = min(1.0, total_quantity / 100.0)  # Normalize by 100 total pieces
        rotation_factor = rotation_allowed / panel_count if panel_count > 0 else 0.0

        complexity = (
            count_factor * 0.3 +
            size_diversity * 0.3 +
            quantity_factor * 0.2 +
            rotation_factor * 0.2
        )

        return min(1.0, complexity)

    def estimate_processing_time(self) -> float:
        """
        Estimate processing time in seconds
        処理時間を推定（秒）
        """
        complexity = self.estimate_complexity()

        if self.strategy == OptimizationStrategy.SINGLE_SIZE_MATH:
            return 0.1 + complexity * 0.5  # Very fast for math calculations
        elif self.strategy == OptimizationStrategy.MULTI_SIZE_SIMPLE:
            return 0.5 + complexity * 2.0  # Moderate time for simple algorithm
        else:  # COMPLEX_PACKING
            return 2.0 + complexity * 10.0  # Longer time for complex algorithm

    def should_use_simple_math(self) -> bool:
        """
        Determine if simple mathematical calculation is sufficient
        単純数学計算で十分かを判定
        """
        if len(self.panels) != 1:
            return False  # Multiple panel types require algorithm

        panel = self.panels[0]
        cutting_w = panel.cutting_width
        cutting_h = panel.cutting_height
        sheet_w = self.sheet_template.width
        sheet_h = self.sheet_template.height

        # Case 1: Panel width matches sheet width exactly (±1mm tolerance)
        if abs(cutting_w - sheet_w) <= 1.0:
            return True

        # Case 2: Two panels fit exactly in width
        if abs(cutting_w * 2 - sheet_w) <= 1.0:
            return True

        # Case 3: Three panels fit exactly in width
        if abs(cutting_w * 3 - sheet_w) <= 1.0:
            return True

        # Case 4: Simple rectangular grid
        panels_per_width = int(sheet_w // cutting_w)
        panels_per_height = int(sheet_h // cutting_h)
        if panels_per_width >= 1 and panels_per_height >= 1:
            # Check if it's a clean fit
            used_width = panels_per_width * cutting_w
            used_height = panels_per_height * cutting_h
            if used_width <= sheet_w and used_height <= sheet_h:
                return True

        return False


@dataclass
class OptimizationResult:
    """
    Result of optimization task execution
    最適化タスク実行の結果
    """
    task: OptimizationTask
    metrics: PlacementMetrics = field(default_factory=PlacementMetrics)
    placement_results: List = field(default_factory=list)  # List[PlacementResult]
    unplaced_panels: List[Panel] = field(default_factory=list)
    success: bool = False
    error_message: str = ""
    algorithm_used: str = ""
    completed_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate and finalize result"""
        if self.placement_results and not self.unplaced_panels and self.metrics.efficiency > 0.6:
            self.success = True

    @property
    def total_panels_placed(self) -> int:
        """Total number of panels successfully placed"""
        return sum(len(result.panels) for result in self.placement_results)

    @property
    def placement_rate(self) -> float:
        """Rate of panels successfully placed (0-1)"""
        total_panels = sum(panel.quantity for panel in self.task.panels)
        if total_panels == 0:
            return 0.0
        return self.total_panels_placed / total_panels

    def get_summary(self) -> Dict[str, Any]:
        """Get summary information for reporting"""
        return {
            'task_id': self.task.task_id,
            'success': self.success,
            'efficiency': self.metrics.efficiency,
            'sheets_used': self.metrics.sheet_count,
            'processing_time': self.metrics.processing_time,
            'placement_rate': self.placement_rate,
            'algorithm_used': self.algorithm_used,
            'waste_percentage': self.metrics.waste_percentage,
            'error': self.error_message if not self.success else None
        }